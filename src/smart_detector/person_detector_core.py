# src/smart_detector/person_detector_core.py
from datetime import datetime

import cv2
import time
import logging
from pathlib import Path
from collections import deque
from ultralytics import YOLO
import numpy as np
from threading import Thread, Event, current_thread as get_current_thread  # For thread name
import queue  # Python's standard queue

from .utils import get_config, get_detection_color
from .video_utils import create_video_capture, save_video_segment, draw_detections
from . import config_constants as consts

logger = logging.getLogger(__name__)
_model_instance = None


def _load_model_once(model_path_str: str, device: str):
    global _model_instance
    if _model_instance is None:
        logger.info(f"Loading model from {model_path_str} for device '{device}'")
        try:
            _model_instance = YOLO(model_path_str)
            dummy_input = np.zeros((480, 640, 3), dtype=np.uint8)  # Common small size
            # Use half=True for GPU for potential speedup if model supports FP16
            _model_instance.predict(dummy_input, device=device, verbose=False, half=(device != "cpu"))
            logger.info(f"Model loaded and initialized on '{device}'.")
        except Exception as e:
            logger.error(f"Error loading model {model_path_str}: {e}", exc_info=True)
            _model_instance = None
            raise
    return _model_instance


def _perform_inference(frame: np.ndarray, infer_cfg: dict, frame_dims: tuple[int, int]):
    # frame_dims is (frame_width, frame_height)
    model = _model_instance
    if model is None:
        logger.error("Model not loaded. Cannot perform inference.")
        return []

    results = model.predict(
        source=frame.copy(),  # Process a copy
        conf=infer_cfg['confidence_threshold'],
        classes=[infer_cfg['person_class_id']],
        device=infer_cfg['device'],
        verbose=False,
        half=(infer_cfg['device'] != "cpu"),
    )

    detections = []
    if results and results[0].boxes:
        frame_width, frame_height = frame_dims
        # Define bbox area limits for filtering (as a ratio of frame area)
        # These are heuristics, adjust based on your scene and typical person size
        max_allowed_bbox_area_ratio = infer_cfg.get('max_person_bbox_area_ratio', 0.90)  # Tune this!
        min_allowed_bbox_area_ratio = infer_cfg.get('min_person_bbox_area_ratio', 0.005)  # Tune this!

        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = results[0].names.get(class_id, f"class_{class_id}") if results[
                0].names else f"class_{class_id}"

            if class_id == infer_cfg['person_class_id']:
                # Apply BBox size filter for "whole window" false positives
                x1, y1, x2, y2 = map(int, xyxy)
                bbox_width = x2 - x1
                bbox_height = y2 - y1

                if bbox_width <= 0 or bbox_height <= 0:  # Invalid bbox
                    logger.debug(f"Invalid bbox dimensions for person: w={bbox_width}, h={bbox_height}. Skipping.")
                    continue

                bbox_area = bbox_width * bbox_height
                frame_area = frame_width * frame_height

                if frame_area == 0:  # Should not happen if frame_dims are valid
                    bbox_area_ratio = 0
                else:
                    bbox_area_ratio = bbox_area / frame_area

                if not (min_allowed_bbox_area_ratio <= bbox_area_ratio <= max_allowed_bbox_area_ratio):
                    logger.info(  # Log as info for tuning
                        f"Filtering out 'person' detection due to area ratio: {bbox_area_ratio:.3f} "
                        f"(conf: {conf:.2f}, bbox: [{x1},{y1},{x2},{y2}]). "
                        f"Allowed range: [{min_allowed_bbox_area_ratio}-{max_allowed_bbox_area_ratio}]."
                    )
                    continue

                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox_xyxy': xyxy
                })
    return detections


def capture_thread_func(stop_event: Event, frame_queue: queue.Queue, cfg: dict):
    current_thread_name = "CaptureThread"
    get_current_thread().name = current_thread_name

    rtsp_cfg = cfg['rtsp']
    cap = None
    frame_counter = 0
    desired_fps = rtsp_cfg.get('capture_fps', 15)
    # For latency, we want to push frames as fast as they come, up to desired_fps.
    # The processing loop will handle dropping if it can't keep up.
    # So, minimal or no artificial delay here unless camera is much faster than desired_fps.
    min_frame_interval = 1.0 / desired_fps if desired_fps > 0 else 0

    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            logger.info(f"[{current_thread_name}] Attempting to connect to video source: {rtsp_cfg['url']}")
            if cap: cap.release()
            cap = create_video_capture(
                rtsp_cfg['url'],
                rtsp_cfg['capture_resolution_width'],
                rtsp_cfg['capture_resolution_height'],
                desired_fps
            )
            if cap is None:
                logger.error(
                    f"[{current_thread_name}] Failed to connect. Retrying in {rtsp_cfg['reconnect_delay_seconds']}s...")
                stop_event.wait(rtsp_cfg['reconnect_delay_seconds'])
                continue
            logger.info(f"[{current_thread_name}] Video source connected.")
            frame_counter = 0

        loop_start_time = time.perf_counter()
        ret, frame = cap.read()  # Blocking read

        if not ret or frame is None:
            logger.warning(f"[{current_thread_name}] Failed to grab/retrieve frame. Reconnecting...")
            if cap: cap.release()
            cap = None
            stop_event.wait(rtsp_cfg.get('reconnect_delay_seconds', 5))
            continue

        timestamp = time.time()
        try:
            frame_id = f"f_{timestamp:.3f}_{frame_counter}"
            # Put (timestamp, frame_id, frame.copy()) onto the queue.
            # Using put with timeout allows graceful shutdown if queue is full and main thread is stopping.
            frame_queue.put({'timestamp': timestamp, 'id': frame_id, 'frame': frame.copy()}, timeout=0.5)
            logger.debug(f"[{current_thread_name}] Queued frame {frame_id}")
            frame_counter += 1
        except queue.Full:
            logger.warning(
                f"[{current_thread_name}] Frame queue is full. Frame {frame_id} dropped. Processing might be too slow.")
            # To prioritize low latency display, if queue is full, we could try to empty older items
            # This ensures the processing loop gets fresher frames if it falls behind.
            # However, this can lead to lost frames for recording if processing can't keep up with pre-roll needs.
            # For now, just drop the current one.
            pass

        # Optional: If camera is significantly faster than desired_fps, enforce min_frame_interval
        elapsed_time = time.perf_counter() - loop_start_time
        if min_frame_interval > 0:
            sleep_duration = min_frame_interval - elapsed_time
            if sleep_duration > 0:
                stop_event.wait(sleep_duration)

    logger.info(f"[{current_thread_name}] Stopping.")
    if cap:
        cap.release()
    try:
        frame_queue.put_nowait(None)  # Sentinel, non-blocking
    except queue.Full:
        logger.warning(f"[{current_thread_name}] Could not put sentinel on full queue during shutdown.")


def run_person_detection_loop(stop_event: Event, cfg: dict):
    get_current_thread().name = "MainProcessingThread"
    infer_cfg = cfg['inference']
    record_cfg = cfg['person_event_recording']
    display_cfg = cfg['display']
    rtsp_cfg = cfg['rtsp']

    try:
        _load_model_once(str(infer_cfg['model_path']), infer_cfg['device'])
    except Exception as e:
        logger.critical(f"Failed to load model: {e}. Application cannot continue.")
        stop_event.set()
        return

    # Frame queue with a size limit to manage memory and prioritize recent frames for display/latency
    # Size based on a few seconds of frames at capture rate
    queue_capacity = int(
        rtsp_cfg['capture_fps'] * max(record_cfg['frame_buffer_seconds'], 5))  # At least 5s or buffer_seconds
    frame_queue = queue.Queue(maxsize=queue_capacity)

    capture_th = Thread(
        target=capture_thread_func,
        args=(stop_event, frame_queue, cfg),
        daemon=True
    )
    capture_th.start()

    frames_per_buffer_sec = rtsp_cfg['capture_fps']
    recent_frames_buffer_maxlen = int(record_cfg['frame_buffer_seconds'] * frames_per_buffer_sec)
    recent_frames_buffer = deque(
        maxlen=recent_frames_buffer_maxlen)  # Stores {'timestamp', 'id', 'frame', 'detections' (optional)}

    is_person_currently_detected_in_event = False
    last_person_detection_time_in_event = 0
    active_recording_frames = []
    is_recording_active = False
    current_clip_start_timestamp = 0  # Timestamp of the frame that started the clip (could be pre-roll)

    output_dir = Path(record_cfg['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)

    fps_calc_frames = 0
    fps_calc_last_time = time.perf_counter()
    displayed_fps = 0.0

    last_processed_frame_data = None  # Store the latest fully processed data for display

    process_frame_counter = 0  # Frames on which inference is run

    try:
        while not stop_event.is_set():
            current_frame_data = None
            try:
                # Prioritize latest frame for low latency display:
                # Try to get the newest item, discarding older ones in the queue if processing is slow.
                # This is aggressive for latency; might lose pre-roll frames if capture is much faster than processing.
                # A balance is needed. For now, simple get().
                # To get latest:
                # while True:
                #     try:
                #        current_frame_data = frame_queue.get_nowait() # Get latest
                #     except queue.Empty:
                #         break # No more new frames, process what we have (or wait if None)
                # if current_frame_data is None and last_processed_frame_data is None: # Nothing to do yet
                #     time.sleep(0.005) # Small sleep to yield
                #     continue

                # Simpler: block for a short time, process what you get.
                current_frame_data = frame_queue.get(timeout=0.1)  # Short timeout
                if current_frame_data is None:  # Sentinel
                    logger.info("Received sentinel. Shutting down processing loop.")
                    break
                last_processed_frame_data = current_frame_data  # Update for display

            except queue.Empty:
                # No new frame in a short while. Check recording timeout.
                # Process display with last known frame if available.
                if is_recording_active and \
                        (time.time() - last_person_detection_time_in_event > record_cfg['post_event_timeout_seconds']):
                    logger.info(
                        f"Person event ended (timeout). Finalizing clip for event started at {datetime.fromtimestamp(current_clip_start_timestamp).isoformat()}.")
                    # Finalize recording (code similar to below)
                    # ... (Add save_video_segment call here) ...
                    if record_cfg['enabled'] and active_recording_frames:
                        first_valid_frame_for_dims = next((f for f in active_recording_frames if f is not None), None)
                        if first_valid_frame_for_dims is not None:
                            fh, fw = first_valid_frame_for_dims.shape[:2]
                            save_video_segment(active_recording_frames,
                                               output_dir / f"{record_cfg['filename_prefix']}_{datetime.fromtimestamp(current_clip_start_timestamp).strftime('%Y%m%d-%H%M%S%f')[:-3]}.mp4",
                                               rtsp_cfg['capture_fps'], (fw, fh))
                    active_recording_frames.clear()
                    is_recording_active = False
                    is_person_currently_detected_in_event = False

                # Update display even if no new frame, to keep it responsive for "q"
                if display_cfg['show_window'] and last_processed_frame_data:
                    # (Simplified display update, full logic below)
                    cv2.imshow(display_cfg['window_name'], last_processed_frame_data['frame'])  # Show last good frame
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): stop_event.set(); break
                continue

            timestamp = current_frame_data['timestamp']
            frame_id = current_frame_data['id']
            frame = current_frame_data['frame']

            # Ensure frame has valid dimensions before proceeding
            if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                logger.warning(f"Received invalid frame (None or zero dimensions) {frame_id}. Skipping.")
                continue

            frame_height, frame_width = frame.shape[:2]

            # Add to recent_frames_buffer. This buffer stores raw frames.
            # Detections will be associated later if inference is run.
            recent_frames_buffer.append({'timestamp': timestamp, 'id': frame_id, 'frame': frame, 'detections': []})

            detections_this_frame = []
            run_inference_on_this_frame = (process_frame_counter % infer_cfg['process_every_nth_frame'] == 0)

            if run_inference_on_this_frame:
                inference_start_time = time.perf_counter()
                detections_this_frame = _perform_inference(frame, infer_cfg, (frame_width, frame_height))
                logger.debug(
                    f"Inference on frame {frame_id} ({frame_width}x{frame_height}) took {(time.perf_counter() - inference_start_time) * 1000:.2f} ms. Found {len(detections_this_frame)} persons.")
                # Update the frame in buffer with its detections
                if recent_frames_buffer:  # Ensure buffer not empty (should not be)
                    # Find the corresponding frame in buffer and update. Assuming current frame is the last one.
                    # This can be tricky if buffer allows out-of-order items, but deque appends to right.
                    if recent_frames_buffer[-1]['id'] == frame_id:
                        recent_frames_buffer[-1]['detections'] = detections_this_frame
                    else:  # Should not happen with current logic
                        logger.warning(
                            f"Frame ID mismatch updating detections in buffer. Last buffer ID: {recent_frames_buffer[-1]['id']}, current: {frame_id}")

            process_frame_counter += 1

            # Use detections_this_frame (which might be empty if inference was skipped, or from inference)
            # For recording logic, we need to know if *any* recent frame had a person
            person_detected_this_inference_cycle = any(
                d['class_id'] == infer_cfg['person_class_id'] for d in detections_this_frame)

            if person_detected_this_inference_cycle:
                last_person_detection_time_in_event = time.time()  # Wall clock for timeout
                is_person_currently_detected_in_event = True  # A person is part of the ongoing "event"

                if not is_recording_active and record_cfg['enabled']:
                    is_recording_active = True
                    # Clip start time is the timestamp of the *earliest frame* we'll include (from pre-roll)
                    # Find pre-roll frames from recent_frames_buffer
                    frames_for_new_clip = []
                    pre_event_start_ts = timestamp - record_cfg['pre_event_seconds']

                    # Iterate a snapshot of the buffer to get pre-roll
                    # The buffer contains {'timestamp', 'id', 'frame', 'detections'}
                    # Detections might be empty for frames where inference was skipped.
                    # The important part for pre-roll are the frames themselves.
                    actual_clip_start_frame_ts = timestamp  # Default to current frame's timestamp

                    for buffered_item in list(recent_frames_buffer):
                        if buffered_item['timestamp'] >= pre_event_start_ts and buffered_item['timestamp'] <= timestamp:
                            frames_for_new_clip.append(buffered_item['frame'])
                            if buffered_item['timestamp'] < actual_clip_start_frame_ts:
                                actual_clip_start_frame_ts = buffered_item['timestamp']

                    if not frames_for_new_clip:  # Should not happen if current frame is in buffer
                        frames_for_new_clip.append(frame)  # Fallback
                        actual_clip_start_frame_ts = timestamp

                    active_recording_frames.extend(frames_for_new_clip)
                    current_clip_start_timestamp = actual_clip_start_frame_ts  # Timestamp of the true start of the clip content

                    logger.info(
                        f"Person detected! Starting new recording for event that includes frame {frame_id}. "
                        f"Clip starts around {datetime.fromtimestamp(current_clip_start_timestamp).isoformat()}. "
                        f"Added {len(frames_for_new_clip)} frames (incl. pre-roll)."
                    )

            # If recording is active, add the current raw frame (from queue)
            if is_recording_active:
                # Avoid adding frame if it was ALREADY added as part of initial pre-roll batch
                # This requires careful check, e.g. by frame ID or if active_recording_frames is non-empty
                # and the current frame is not the one that triggered the pre-roll.
                # Simpler: if active_recording_frames is empty, pre-roll logic handles the first batch.
                # If not empty, it means recording is ongoing, so add current frame.
                if not (
                        person_detected_this_inference_cycle and not active_recording_frames):  # If not the triggering frame of a new recording
                    if not active_recording_frames or not np.array_equal(active_recording_frames[-1],
                                                                         frame):  # Avoid duplicates if logic is tricky
                        active_recording_frames.append(frame)

            # End recording if timeout
            if is_recording_active and not person_detected_this_inference_cycle and is_person_currently_detected_in_event:
                # Person was part of this event, but not seen in *this* inference cycle.
                # Check timeout from last *actual* detection.
                if time.time() - last_person_detection_time_in_event > record_cfg['post_event_timeout_seconds']:
                    logger.info(
                        f"Person event ended (timeout after last detection). Finalizing clip from event started at {datetime.fromtimestamp(current_clip_start_timestamp).isoformat()}.")
                    if record_cfg['enabled'] and active_recording_frames:
                        first_valid_frame = next(
                            (f for f in active_recording_frames if f is not None and f.shape[0] > 0 and f.shape[1] > 0),
                            None)
                        if first_valid_frame is not None:
                            fh, fw = first_valid_frame.shape[:2]
                            save_video_segment(active_recording_frames,
                                               output_dir / f"{record_cfg['filename_prefix']}_{datetime.fromtimestamp(current_clip_start_timestamp).strftime('%Y%m%d-%H%M%S%f')[:-3]}.mp4",
                                               rtsp_cfg['capture_fps'], (fw, fh))
                        else:
                            logger.warning("Could not save clip as no valid frames were recorded.")
                    active_recording_frames.clear()
                    is_recording_active = False
                    is_person_currently_detected_in_event = False

            # Display: Always try to update display with the current_frame_data for responsiveness
            if display_cfg['show_window']:
                vis_frame = frame.copy()  # Use the raw frame from queue for display base

                # Overlay detections from *this specific frame* if inference was run.
                # Otherwise, detections_this_frame will be empty.
                if detections_this_frame:
                    vis_frame = draw_detections(vis_frame, detections_this_frame, infer_cfg['person_class_id'],
                                                get_detection_color)

                fps_calc_frames += 1
                loop_time = time.perf_counter()
                if loop_time - fps_calc_last_time >= display_cfg['fps_display_interval']:
                    if (loop_time - fps_calc_last_time) > 0:  # Avoid div by zero
                        displayed_fps = fps_calc_frames / (loop_time - fps_calc_last_time)
                    fps_calc_frames = 0
                    fps_calc_last_time = loop_time

                cv2.putText(vis_frame, f"FPS: {displayed_fps:.1f} (Proc)", consts.INFO_TEXT_POSITION,
                            consts.CV2_FONT, consts.CV2_FONT_SCALE_INFO, consts.CV2_COLOR_FPS,
                            consts.CV2_FONT_THICKNESS)
                if is_recording_active:
                    cv2.putText(vis_frame, "REC", consts.RECORDING_INDICATOR_POSITION,
                                consts.CV2_FONT, consts.CV2_FONT_SCALE_INFO, consts.CV2_COLOR_RECORDING_INDICATOR, 2)
                try:
                    cv2.imshow(display_cfg['window_name'], vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): stop_event.set(); break
                except cv2.error as e:
                    if "NULL window" in str(e) or "Invalid window handle" in str(e):
                        logger.warning("OpenCV window seems closed. Disabling further display attempts.")
                        display_cfg['show_window'] = False
                        if stop_event.is_set(): break
                    else:
                        logger.error(f"OpenCV display error: {e}", exc_info=True); stop_event.set(); break

            # Yield slightly if not displaying, to allow capture thread to fill queue if it's faster
            # if not display_cfg['show_window']:
            #    time.sleep(0.001)


    except Exception as e:
        logger.error(f"Unhandled error in main processing loop: {e}", exc_info=True)
        stop_event.set()
    finally:
        logger.info("Main processing loop finalizing...")
        if is_recording_active and record_cfg['enabled'] and active_recording_frames:
            logger.info("Application shutting down. Saving pending recording segment...")
            first_valid_frame = next(
                (f for f in active_recording_frames if f is not None and f.shape[0] > 0 and f.shape[1] > 0), None)
            if first_valid_frame is not None:
                fh, fw = first_valid_frame.shape[:2]
                save_video_segment(active_recording_frames,
                                   output_dir / f"{record_cfg['filename_prefix']}_shutdown_{datetime.fromtimestamp(current_clip_start_timestamp if current_clip_start_timestamp else time.time()).strftime('%Y%m%d-%H%M%S%f')[:-3]}.mp4",
                                   rtsp_cfg['capture_fps'], (fw, fh))
            else:
                logger.warning("Could not save final clip as no valid frames were in buffer.")

        if display_cfg.get('show_window', False):
            try:
                cv2.destroyAllWindows()
            except:
                pass

        if capture_th.is_alive():
            logger.info("Waiting for capture thread to finish...")
            # Sentinel already sent by capture_th itself, or by this loop if it exited normally.
            capture_th.join(timeout=3.0)  # Increased timeout
            if capture_th.is_alive():
                logger.warning("Capture thread did not finish in time.")
        logger.info("Main processing loop stopped.")