# src/smart_detector/core.py
from __future__ import annotations
import time
import cv2
import numpy as np
import logging
from ultralytics import YOLO
import zmq
import pickle  # For sending numpy arrays via ZMQ if needed, or use msgpack
from collections import deque
from threading import current_thread, Event
from datetime import datetime

from .utils import get_config, get_class_color, generate_event_id
from . import config as app_config  # For CV2 constants

logger = logging.getLogger(__name__)
_MODEL_INSTANCE = None


def _load_model(model_path: str, device: str):
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        t0 = time.perf_counter()
        logger.info(f"Loading model from {model_path} for device {device}")
        try:
            _MODEL_INSTANCE = YOLO(model_path)  # Ultralytics can load .pt, .onnx
            _MODEL_INSTANCE.to(device)
            # Perform a dummy inference to fully initialize the model
            # Adjust dummy input size if needed, or use a sample image
            dummy_input = np.zeros((640, 480, 3), dtype=np.uint8)
            _MODEL_INSTANCE.predict(dummy_input, verbose=False)
            logger.info(f"Model loaded and initialized in {time.perf_counter() - t0:.2f} s")
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}", exc_info=True)
            _MODEL_INSTANCE = None  # Ensure it's None if loading failed
            raise
    return _MODEL_INSTANCE


def _create_rtsp_capture():
    cfg = get_config()
    rtsp_cfg = cfg['rtsp']
    source = rtsp_cfg['url']

    cap = None
    gst_pipeline = None

    if isinstance(source, str) and source.lower().startswith("rtsp"):
        # Try GStreamer pipeline first for low latency
        # This pipeline is generic; specific camera/encoder might need adjustments
        gst_pipeline = (
            f"rtspsrc location={source} latency=100 drop-on-latency=true ! "
            f"queue ! rtph264depay ! h264parse ! avdec_h264 ! "  # Assuming H.264
            # Or for H.265: f"rtph265depay ! h265parse ! avdec_h265 ! "
            f"videoconvert ! appsink sync=false max-buffers=2 drop=true"
        )
        logger.info(f"Attempting to open RTSP stream via GStreamer: {source}")
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            logger.warning(f"GStreamer RTSP failed. Falling back to OpenCV default for {source}.")
            cap = cv2.VideoCapture(source)  # OpenCV's FFMPEG backend
    else:  # For local files or webcam
        logger.info(f"Opening video source: {source}")
        cap = cv2.VideoCapture(source)

    if not cap or not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return None

    # Try to set capture properties (camera might not support all)
    if rtsp_cfg.get('capture_resolution_width') and rtsp_cfg.get('capture_resolution_height'):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, rtsp_cfg['capture_resolution_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rtsp_cfg['capture_resolution_height'])
    if rtsp_cfg.get('capture_fps'):
        cap.set(cv2.CAP_PROP_FPS, rtsp_cfg['capture_fps'])

    # Set a small buffer size for OpenCV; main resilience buffer is managed in capture_frames
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Successfully opened video source. Resolution: {actual_w}x{actual_h}, FPS: {actual_fps:.2f}")
    if actual_w == 0 or actual_h == 0:
        logger.error("Video source opened but frame dimensions are zero. Check camera/stream.")
        cap.release()
        return None

    return cap


def capture_frames_thread_func(stop_event: Event, zmq_context: zmq.Context):
    """
    Reads frames from RTSP/camera, manages a resilience buffer, and publishes them via ZeroMQ.
    """
    cfg = get_config()
    rtsp_cfg = cfg['rtsp']
    zmq_cfg = cfg['zeromq']
    current_thread().name = "CaptureThread"

    publisher = zmq_context.socket(zmq.PUB)
    publisher.bind(zmq_cfg['frame_pub_url'])
    logger.info(f"Frame publisher bound to {zmq_cfg['frame_pub_url']}")

    cap = None
    resilience_buffer = deque(maxlen=rtsp_cfg.get('resilience_buffer_size', 64))
    frame_counter = 0

    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            logger.info("Attempting to connect to video source...")
            cap = _create_rtsp_capture()
            if cap is None:
                logger.error(f"Failed to connect. Retrying in {rtsp_cfg['reconnect_delay_seconds']}s...")
                time.sleep(rtsp_cfg['reconnect_delay_seconds'])
                continue
            logger.info("Video source connected.")
            resilience_buffer.clear()  # Clear buffer on new connection

        ret, frame = cap.read()

        if not ret:
            logger.warning("Failed to grab frame from source. Attempting to reconnect.")
            if cap:
                cap.release()
            cap = None  # Trigger reconnection
            time.sleep(rtsp_cfg.get('reconnect_delay_seconds', 5))
            continue

        timestamp = time.time()
        frame_id = f"frame_{timestamp:.6f}_{frame_counter}"
        frame_counter += 1

        # Add to resilience buffer
        resilience_buffer.append({'id': frame_id, 'timestamp': timestamp, 'frame': frame, 'seq': frame_counter})

        # Publish the latest frame from the buffer (or the current one)
        # This ensures we publish even if the buffer isn't full yet
        # The main idea of resilience buffer is for internal recovery or specific access
        # For pub/sub, just send the current one.
        # However, the requirement "Кадры складываются в кольцевой буфер на 64 шт.: если сеть на секунду «заглохнет»,
        # в памяти ещё останутся непросмотренные кадры, и детектор не заметит паузы."
        # This implies the buffer is to ensure continuous feed to *consumers* during short glitches.
        # So, if read fails, we might try to send from buffer if not empty.
        # For now, simpler: if read works, publish. If read fails, reconnect logic handles it.
        # The description also says "Если связь обрывается дольше чем на секунду, блок сам переподключается"
        # The buffer mainly ensures frames are available if `cap.read()` is slow but not terminally failed.

        # For ZMQ, it's better to send data that can be pickled or is otherwise serializable.
        # Sending raw numpy arrays is fine with pickle.
        try:
            # Topic: "frames" (optional, if subscriber wants to filter)
            # Message: [frame_id, timestamp, frame_data]
            frame_data_msg = pickle.dumps(
                {'id': frame_id, 'timestamp': timestamp, 'frame': frame, 'seq': frame_counter})
            publisher.send_multipart([b"frames_raw", frame_data_msg])
            logger.debug(f"Published frame {frame_id}, size {len(frame_data_msg)}")
        except zmq.ZMQError as e:
            logger.error(f"ZMQ error publishing frame: {e}")
        except Exception as e:
            logger.error(f"Error pickling/sending frame: {e}", exc_info=True)

        # Control capture rate if camera FPS is higher than desired processing FPS, though VideoCapture usually handles this.
        # time.sleep(1.0 / rtsp_cfg.get('capture_fps', 25)) # This might be too simplistic.

    logger.info("Capture thread stopping.")
    if cap:
        cap.release()
    publisher.close()


def _perform_inference(model, frame, infer_cfg):
    results = model.predict(
        source=frame,
        conf=infer_cfg['confidence_threshold'],
        iou=infer_cfg.get('iou_threshold', 0.45),
        classes=infer_cfg.get('active_class_ids'),  # Filter by specific class IDs
        device=infer_cfg['device'],
        verbose=False
    )
    return results[0]  # Results for the first image


def process_frames_thread_func(stop_event: Event, zmq_context: zmq.Context):
    """
    Subscribes to raw frames, performs inference, and publishes detection events.
    """
    cfg = get_config()
    infer_cfg = cfg['inference']
    zmq_cfg = cfg['zeromq']
    current_thread().name = "ProcessingThread"

    model = None
    try:
        model = _load_model(infer_cfg['model_path'], infer_cfg['device'])
    except Exception:
        logger.error("Failed to load model in ProcessingThread. Thread will exit.")
        return  # Exit thread if model fails to load

    frame_subscriber = zmq_context.socket(zmq.SUB)
    frame_subscriber.connect(zmq_cfg['frame_pub_url'])
    frame_subscriber.subscribe(b"frames_raw")  # Subscribe to raw frames topic
    logger.info(f"Frame subscriber connected to {zmq_cfg['frame_pub_url']}")
    annotated_publisher = zmq_context.socket(zmq.PUB)
    annotated_publisher.bind(zmq_cfg['annotated_pub_url'])

    event_publisher = zmq_context.socket(zmq.PUB)
    event_publisher.bind(zmq_cfg['event_pub_url'])
    logger.info(f"Event publisher bound to {zmq_cfg['event_pub_url']}")

    poller = zmq.Poller()
    poller.register(frame_subscriber, zmq.POLLIN)

    process_nth_frame = infer_cfg.get('process_every_nth_frame', 1)
    frame_count = 0

    active_class_names_map = {k: v for k, v in infer_cfg['classes_map'].items() if
                              k in infer_cfg.get('active_class_ids', [])}

    while not stop_event.is_set():
        socks = dict(poller.poll(timeout=1000))  # Timeout to allow checking stop_event
        if frame_subscriber in socks and socks[frame_subscriber] == zmq.POLLIN:
            try:
                topic, frame_data_msg = frame_subscriber.recv_multipart()
                frame_payload = pickle.loads(frame_data_msg)

                received_frame_id = frame_payload['id']
                timestamp = frame_payload['timestamp']
                frame = frame_payload['frame']
                original_seq = frame_payload['seq']

                logger.debug(f"Processing received frame {received_frame_id} (seq {original_seq})")
            except pickle.UnpicklingError as e:
                logger.error(f"Error unpickling frame data: {e}")
                continue
            except Exception as e:
                logger.error(f"Error receiving/unpacking frame: {e}", exc_info=True)
                continue

            frame_count += 1
            if frame_count % process_nth_frame != 0:
                # Publish a "heartbeat" or passthrough event if GUI needs all frames with some annotation
                # For now, skip inference for this frame as per requirement.
                # However, GUI needs frame for live view. This means GUI should subscribe to raw frames too.
                # This processing module only publishes *detection events*.
                logger.debug(
                    f"Skipping inference for frame {received_frame_id} (seq {original_seq}) due to process_every_nth_frame={process_nth_frame}")
                continue

            t_infer_start = time.perf_counter()
            try:
                if model is None:  # Should have been caught at init, but as a safeguard
                    logger.error("Model is not loaded. Cannot perform inference.")
                    time.sleep(1)  # Avoid busy loop if model load failed
                    continue
                results = _perform_inference(model, frame, infer_cfg)
            except Exception as e:
                logger.error(f"Exception during inference: {e}", exc_info=True)
                continue

            infer_time_ms = (time.perf_counter() - t_infer_start) * 1000
            logger.debug(
                f"Inference for frame {received_frame_id} took {infer_time_ms:.2f} ms. Found {len(results.boxes)} raw boxes.")

            detections = []
            if results.boxes:
                for box in results.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    if class_id in active_class_names_map:
                        class_name = active_class_names_map[class_id]
                        detection_event = {
                            'event_id': generate_event_id(),
                            'frame_id': received_frame_id,  # Link to the source frame
                            'timestamp': timestamp,  # Timestamp of the frame capture
                            'detection_time': time.time(),  # Timestamp of this detection event
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': round(conf, 3),
                            'bbox_xyxy': [round(c) for c in xyxy],  # [x1, y1, x2, y2]
                            'original_frame_seq': original_seq
                        }
                        detections.append(detection_event)
                        logger.debug(f"Detected: {class_name} (conf: {conf:.2f}) at {detection_event['bbox_xyxy']}")

            if detections:
                # Publish as a single message containing a list of detections for this frame
                try:
                    event_msg_payload = {'frame_id': received_frame_id, 'timestamp': timestamp,
                                         'detections': detections}
                    event_data_msg = pickle.dumps(event_msg_payload)
                    event_publisher.send_multipart([b"detection_events", event_data_msg])
                    vis_frame = draw_detections_on_frame(frame, detections, infer_cfg['classes_map'])
                    annotated_msg = pickle.dumps({
                        'id': received_frame_id,
                        'timestamp': timestamp,
                        'frame': vis_frame
                    })
                    annotated_publisher.send_multipart([b"frames_annotated", annotated_msg])
                    logger.info(
                        f"Published {len(detections)} detection events for frame {received_frame_id} (seq {original_seq})")
                except zmq.ZMQError as e:
                    logger.error(f"ZMQ error publishing detection event: {e}")
                except Exception as e:
                    logger.error(f"Error pickling/sending detection event: {e}", exc_info=True)
            elif frame_count % (process_nth_frame * 5) == 0:  # Log occasionally if no detections
                logger.debug(f"No active detections for frame {received_frame_id} (seq {original_seq})")

    logger.info("Processing thread stopping.")
    annotated_publisher.close()
    frame_subscriber.close()
    event_publisher.close()


def draw_detections_on_frame(frame, detections_list, classes_map):
    """ Draws detections from a list of detection events onto a frame. """
    if not detections_list:
        return frame

    vis_frame = frame.copy()
    for det_event in detections_list:
        x1, y1, x2, y2 = det_event['bbox_xyxy']
        conf = det_event['confidence']
        class_name = det_event['class_name']

        color = get_class_color(class_name)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {conf:.2f}"
        text_size, _ = cv2.getTextSize(label, app_config.CV2_FONT, app_config.CV2_FONT_SCALE_SMALL,
                                       app_config.CV2_FONT_THICKNESS)
        cv2.rectangle(vis_frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
        cv2.putText(vis_frame, label, (x1, y1 - 5), app_config.CV2_FONT, app_config.CV2_FONT_SCALE_SMALL,
                    app_config.CV2_COLOR_BLACK, app_config.CV2_FONT_THICKNESS, app_config.CV2_LINE_TYPE)
    return vis_frame