# src/smart_detector/person_detector_core.py
import threading
from datetime import datetime, timedelta

import cv2
import time
import logging
from pathlib import Path
from collections import deque
from ultralytics import YOLO
import numpy as np
from threading import Thread, Event, Lock
import queue  # Python's standard queue

from .utils import get_config, get_detection_color
from .video_utils import create_video_capture, save_video_segment, draw_detections
from . import config as consts

logger = logging.getLogger(__name__)
_model_instance = None  # Global for the loaded model


def _load_model_once(model_path_str: str, device: str):
    global _model_instance
    if _model_instance is None:
        logger.info(f"Loading model from {model_path_str} for device '{device}'")
        try:
            _model_instance = YOLO(model_path_str)
            # Perform a dummy inference to initialize (especially for GPU)
            dummy_input = np.zeros((480, 640, 3), dtype=np.uint8)
            _model_instance.predict(dummy_input, device=device, verbose=False, half=(device != "cpu"))
            logger.info(f"Model loaded and initialized on '{device}'.")
        except Exception as e:
            logger.error(f"Error loading model {model_path_str}: {e}", exc_info=True)
            _model_instance = None  # Ensure it's None if loading failed
            raise  # Re-raise to be caught by caller
    return _model_instance


def _perform_inference(frame: np.ndarray, infer_cfg: dict):
    model = _model_instance  # Assumes model is loaded
    if model is None:
        logger.error("Model not loaded. Cannot perform inference.")
        return []

    results = model.predict(
        source=frame.copy(),
        conf=infer_cfg['confidence_threshold'],
        classes=[infer_cfg['person_class_id']],  # Detect only persons
        device=infer_cfg['device'],
        verbose=False,
        # half=(infer_cfg['device'] != "cpu"), # Enable FP16 for GPU
        # imgsz= (height, width) # Optional: if resizing needed by model explicitly
    )

    detections = []
    if results and results[0].boxes:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            # Ultralytics results might have 'names' attribute for class names
            class_name = results[0].names[class_id] if results[0].names else f"class_{class_id}"

            if class_id == infer_cfg['person_class_id']:
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,  # Should be 'person'
                    'confidence': conf,
                    'bbox_xyxy': xyxy
                })
    return detections


def capture_thread_func(stop_event: Event, frame_queue: queue.Queue, cfg: dict):
    """Reads frames from source and puts them into a queue."""
    current_thread_name = "CaptureThread"
    threading.current_thread().name = current_thread_name  # Set thread name for logging

    rtsp_cfg = cfg['rtsp']
    cap = None
    frame_counter = 0

    desired_fps = rtsp_cfg.get('capture_fps', 15)
    frame_delay = 1.0 / desired_fps if desired_fps > 0 else 0.033  # Target delay

    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            logger.info(f"[{current_thread_name}] Attempting to connect to video source: {rtsp_cfg['url']}")
            if cap: cap.release()
            cap = create_video_capture(
                rtsp_cfg['url'],
                rtsp_cfg['capture_resolution_width'],
                rtsp_cfg['capture_resolution_height'],
                desired_fps  # Pass desired FPS to video_utils
            )
            if cap is None:
                logger.error(
                    f"[{current_thread_name}] Failed to connect. Retrying in {rtsp_cfg['reconnect_delay_seconds']}s...")
                stop_event.wait(rtsp_cfg['reconnect_delay_seconds'])
                continue
            logger.info(f"[{current_thread_name}] Video source connected.")
            frame_counter = 0  # Reset counter on new connection

        loop_start_time = time.perf_counter()
        ret, frame = cap.read()

        if not ret or frame is None:
            logger.warning(f"[{current_thread_name}] Failed to grab/retrieve frame. Reconnecting...")
            cap.release()
            cap = None
            stop_event.wait(rtsp_cfg.get('reconnect_delay_seconds', 5))
            continue

        timestamp = time.time()  # Wall-clock time for the event
        try:
            # Put (timestamp, frame_id, frame) onto the queue
            frame_id = f"f_{timestamp:.3f}_{frame_counter}"
            frame_queue.put((timestamp, frame_id, frame.copy()), timeout=1.0)  # copy frame
            logger.debug(f"[{current_thread_name}] Queued frame {frame_id}")
            frame_counter += 1
        except queue.Full:
            logger.warning(
                f"[{current_thread_name}] Frame queue is full. Dropping frame. Processing might be too slow.")
            # Optional: clear queue or drop older items if this happens often
            # while not frame_queue.empty(): try: frame_queue.get_nowait() except queue.Empty: break

        # Aim for desired FPS by calculating sleep time
        elapsed_time = time.perf_counter() - loop_start_time
        sleep_duration = frame_delay - elapsed_time
        if sleep_duration > 0:
            stop_event.wait(sleep_duration)  # Interruptible sleep

    logger.info(f"[{current_thread_name}] Stopping.")
    if cap:
        cap.release()
    # Sentinel to signal processing thread to stop if it's waiting on queue
    try:
        frame_queue.put(None, timeout=0.5)
    except queue.Full:
        pass  # If full, processing thread should pick up existing items then the None.


def run_person_detection_loop(stop_event: Event, cfg: dict):
    """Main loop for processing frames, detecting persons, and recording."""
    infer_cfg = cfg['inference']
    record_cfg = cfg['person_event_recording']
    display_cfg = cfg['display']
    rtsp_cfg = cfg['rtsp']  # For FPS info

    try:
        _load_model_once(str(infer_cfg['model_path']), infer_cfg['device'])
    except Exception as e:
        logger.critical(f"Failed to load model: {e}. Application cannot continue.")
        stop_event.set()  # Signal other threads (like maintenance) to stop
        return

    # Frame queue for communication between capture and processing
    # Max size helps prevent excessive memory use if processing falls behind
    # Max size: capture_fps * buffer_seconds from record_cfg + a margin
    queue_size = int(rtsp_cfg['capture_fps'] * (record_cfg['frame_buffer_seconds'] + 5))
    frame_queue = queue.Queue(maxsize=queue_size)

    # Start capture thread
    capture_th = Thread(
        target=capture_thread_func,
        args=(stop_event, frame_queue, cfg),
        name="CaptureThread",
        daemon=True  # Exits if main thread exits
    )
    capture_th.start()

    # --- State variables for person event recording ---
    # Buffer to hold recent frames for pre-event recording. Stores (timestamp, frame_id, frame)
    # Max length based on frame_buffer_seconds and capture_fps.
    frames_per_buffer_sec = rtsp_cfg['capture_fps']
    recent_frames_buffer_maxlen = int(record_cfg['frame_buffer_seconds'] * frames_per_buffer_sec)
    recent_frames_buffer = deque(maxlen=recent_frames_buffer_maxlen)
    logger.info(
        f"Recent frames buffer size: {recent_frames_buffer_maxlen} frames (~{record_cfg['frame_buffer_seconds']}s at {frames_per_buffer_sec} FPS).")

    is_person_currently_detected_in_event = False  # Is a person part of the current ongoing event
    last_person_detection_time_in_event = 0  # Timestamp of the last frame a person was seen in this event
    active_recording_frames = []  # Frames collected for the current video segment
    is_recording_active = False  # Actively saving frames to active_recording_frames
    current_clip_start_time = None  # Timestamp when current clip recording started

    output_dir = Path(record_cfg['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # For FPS display
    fps_calc_frames = 0
    fps_calc_last_time = time.perf_counter()
    displayed_fps = 0.0

    process_frame_counter = 0  # Counts frames processed by inference

    try:
        while not stop_event.is_set():
            try:
                queued_item = frame_queue.get(timeout=1.0)  # Wait for a frame
                if queued_item is None:  # Sentinel from capture thread
                    logger.info("Received sentinel from capture thread. Shutting down processing loop.")
                    break
                timestamp, frame_id, frame = queued_item
            except queue.Empty:
                # Timeout waiting for frame, check if recording needs to be finalized
                if is_recording_active and (
                        time.time() - last_person_detection_time_in_event > record_cfg['post_event_timeout_seconds']):
                    logger.info(
                        f"Person event ended due to timeout (no new frames and post_event_timeout exceeded). Finalizing clip.")
                    if save_video_segment(active_recording_frames,
                                          output_dir / f"{record_cfg['filename_prefix']}_{datetime.fromtimestamp(current_clip_start_time).strftime('%Y%m%d-%H%M%S%f')[:-3]}.mp4",
                                          rtsp_cfg['capture_fps'],
                                          (frame.shape[1], frame.shape[0]) if active_recording_frames else (
                                          0, 0)):  # Use last known frame size if possible
                        logger.info("Clip saved.")
                    active_recording_frames.clear()
                    is_recording_active = False
                    is_person_currently_detected_in_event = False
                continue  # Go back to try getting a frame

            # Add current frame to recent_frames_buffer (for pre-roll)
            recent_frames_buffer.append({'timestamp': timestamp, 'id': frame_id, 'frame': frame})

            detections = []
            # Process frame for inference based on process_every_nth_frame
            if process_frame_counter % infer_cfg['process_every_nth_frame'] == 0:
                inference_start_time = time.perf_counter()
                detections = _perform_inference(frame, infer_cfg)
                logger.debug(
                    f"Inference on frame {frame_id} took {(time.perf_counter() - inference_start_time) * 1000:.2f} ms. Found {len(detections)} persons.")
            process_frame_counter += 1

            person_detected_this_frame = any(d['class_id'] == infer_cfg['person_class_id'] for d in detections)

            if person_detected_this_frame:
                last_person_detection_time_in_event = time.time()  # Use wall clock for timeout management
                is_person_currently_detected_in_event = True

                if not is_recording_active and record_cfg['enabled']:
                    is_recording_active = True
                    current_clip_start_time = timestamp  # Use frame timestamp as clip start time
                    logger.info(f"Person detected! Starting new recording segment for event around {frame_id}.")
                    # Add pre-event frames from buffer
                    # Iterate a copy of the deque for safety if needed, though here it's single-threaded access
                    # Frames in recent_frames_buffer are {'timestamp': ts, 'id': fid, 'frame': frm}
                    cutoff_ts = timestamp - record_cfg['pre_event_seconds']
                    for buffered_item in list(recent_frames_buffer):  # Iterate snapshot
                        if buffered_item['timestamp'] >= cutoff_ts and buffered_item[
                            'timestamp'] <= timestamp:  # Include current frame
                            active_recording_frames.append(buffered_item['frame'])
                    if not any(f is frame for f in
                               active_recording_frames) and active_recording_frames:  # Ensure current frame is added if pre-roll grabbed it
                        pass  # Already added if condition met
                    elif not active_recording_frames:  # If pre_event_seconds is 0 or buffer too short
                        active_recording_frames.append(frame)

                    logger.info(f"Added {len(active_recording_frames)} frames (incl. pre-roll) to new clip.")

            # If recording is active, always add the *current raw frame* to ensure continuity
            # This is important if process_every_nth_frame > 1
            if is_recording_active:
                # Avoid duplicate if current frame was already added via pre-roll logic
                # The pre-roll logic should add the *current* frame that triggered the recording.
                # So, only add if it's a *subsequent* frame in an active recording.
                # Check if the current frame 'frame' is already the last one in active_recording_frames by object identity or deep comparison
                is_current_frame_already_added = False
                if active_recording_frames:
                    # Using np.array_equal for content check is expensive.
                    # A simpler check if frame_id was part of pre-roll might be better.
                    # For now, assume pre-roll logic handles the *triggering* frame.
                    # Add subsequent frames here.
                    # This logic might need refinement if process_every_nth_frame > 1.
                    # If skipping frames for inference, we still want to record all raw frames during an event.
                    # The `frame` here IS the current raw frame from the queue.
                    if not (person_detected_this_frame and len(active_recording_frames) > 0 and np.array_equal(
                            active_recording_frames[-1], frame)):
                        active_recording_frames.append(frame)

            # Check if an ongoing recording should end
            if is_recording_active and not person_detected_this_frame and is_person_currently_detected_in_event:
                # Person was part of the event, but not seen in *this* processed frame.
                # Start post_event_timeout check.
                if time.time() - last_person_detection_time_in_event > record_cfg['post_event_timeout_seconds']:
                    logger.info(
                        f"Person event ended (timeout after last detection). Finalizing clip from event started at {datetime.fromtimestamp(current_clip_start_time).isoformat()}.")
                    if record_cfg['enabled']:
                        # Ensure frame dimensions are from a valid frame in the list
                        f_width, f_height = 0, 0
                        if active_recording_frames and active_recording_frames[0] is not None:
                            f_height, f_width = active_recording_frames[0].shape[:2]
                        else:  # Fallback to config if no frames or first frame is bad
                            f_width = cfg['rtsp']['capture_resolution_width']
                            f_height = cfg['rtsp']['capture_resolution_height']

                        save_video_segment(active_recording_frames,
                                           output_dir / f"{record_cfg['filename_prefix']}_{datetime.fromtimestamp(current_clip_start_time).strftime('%Y%m%d-%H%M%S%f')[:-3]}.mp4",
                                           rtsp_cfg['capture_fps'],
                                           (f_width, f_height))
                    active_recording_frames.clear()
                    is_recording_active = False
                    is_person_currently_detected_in_event = False  # Reset for next event
                    current_clip_start_time = None

            # Display logic
            if display_cfg['show_window']:
                vis_frame = frame.copy()  # Start with the raw current frame
                if detections:  # If inference was run on this frame and found something
                    vis_frame = draw_detections(vis_frame, detections, infer_cfg['person_class_id'],
                                                get_detection_color)

                # FPS calculation
                fps_calc_frames += 1
                current_time = time.perf_counter()
                if current_time - fps_calc_last_time >= display_cfg['fps_display_interval']:
                    displayed_fps = fps_calc_frames / (current_time - fps_calc_last_time)
                    fps_calc_frames = 0
                    fps_calc_last_time = current_time

                cv2.putText(vis_frame, f"FPS: {displayed_fps:.1f}", consts.INFO_TEXT_POSITION,
                            consts.CV2_FONT, consts.CV2_FONT_SCALE_INFO, consts.CV2_COLOR_FPS,
                            consts.CV2_FONT_THICKNESS)
                if is_recording_active:
                    cv2.putText(vis_frame, "REC", consts.RECORDING_INDICATOR_POSITION,
                                consts.CV2_FONT, consts.CV2_FONT_SCALE_INFO, consts.CV2_COLOR_RECORDING_INDICATOR, 2)

                try:
                    cv2.imshow(display_cfg['window_name'], vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User pressed 'q' in display window. Signaling shutdown.")
                        stop_event.set()
                except cv2.error as e:  # Handle window closed externally
                    if "NULL window" in str(e) or "Invalid window handle" in str(e):
                        logger.warning("OpenCV window seems closed. Disabling display and continuing if not stopping.")
                        display_cfg['show_window'] = False  # Stop trying to show
                        # If q was not pressed, and we want to continue headless:
                        # if not stop_event.is_set(): pass
                        # else: # if already stopping, break
                        if stop_event.is_set(): break
                    else:
                        logger.error(f"OpenCV display error: {e}", exc_info=True)
                        stop_event.set()  # Stop on other OpenCV errors

    except Exception as e:
        logger.error(f"Error in main processing loop: {e}", exc_info=True)
        stop_event.set()
    finally:
        logger.info("Person detection loop finalizing...")
        if is_recording_active and record_cfg['enabled']:  # Save any pending clip on exit
            logger.info("Application shutting down. Saving pending recording segment...")
            f_width, f_height = 0, 0
            if active_recording_frames and active_recording_frames[0] is not None:
                f_height, f_width = active_recording_frames[0].shape[:2]
            else:  # Fallback
                f_width = cfg['rtsp']['capture_resolution_width']
                f_height = cfg['rtsp']['capture_resolution_height']
            save_video_segment(active_recording_frames,
                               output_dir / f"{record_cfg['filename_prefix']}_final_{datetime.fromtimestamp(current_clip_start_time if current_clip_start_time else time.time()).strftime('%Y%m%d-%H%M%S%f')[:-3]}.mp4",
                               rtsp_cfg['capture_fps'],
                               (f_width, f_height))

        if display_cfg.get('show_window', False):  # Use .get for safety
            try:
                cv2.destroyAllWindows()
            except:
                pass  # Ignore errors on destroy

        # Ensure capture thread is signaled and joined if it hasn't finished
        if capture_th.is_alive():
            logger.info("Waiting for capture thread to finish...")
            if not frame_queue.full():  # Avoid blocking if queue is full and capture thread might be stuck on put
                try:
                    frame_queue.put(None, timeout=0.1)  # Send sentinel if not already sent
                except:
                    pass
            capture_th.join(timeout=2.0)
            if capture_th.is_alive():
                logger.warning("Capture thread did not finish in time.")
        logger.info("Person detection loop stopped.")

# Add this function to src/smart_detector/person_detector_core.py

def archive_maintenance_thread_func(stop_event: Event, cfg: dict):
    """Periodically cleans up old clips and manages disk space."""
    current_thread_name = "ArchiveMaintThread"
    threading.current_thread().name = current_thread_name

    maint_cfg = cfg.get('archive_maintenance')
    if not maint_cfg or not maint_cfg.get('enabled', False):
        logger.info(f"[{current_thread_name}] Archive maintenance is disabled.")
        return

    clips_dir = Path(cfg['person_event_recording']['output_directory']) # Base path to clean
    max_age_days = maint_cfg.get('max_clip_age_days', 7)
    min_disk_gb = maint_cfg.get('min_disk_free_gb_threshold', 5)
    interval_sec = maint_cfg.get('cleanup_interval_seconds', 3600)

    logger.info(f"[{current_thread_name}] Started. Cleaning '{clips_dir}'. Max age: {max_age_days} days, Min disk: {min_disk_gb}GB. Interval: {interval_sec}s.")

    while not stop_event.wait(interval_sec): # Interruptible sleep for the interval
        logger.info(f"[{current_thread_name}] Running cleanup cycle...")
        now = datetime.now()
        files_deleted_age = 0
        files_deleted_space = 0

        # 1. Delete by age
        if max_age_days > 0:
            age_limit_ts = time.mktime((now - timedelta(days=max_age_days)).timetuple())
            for f_path in clips_dir.glob('*.mp4'): # Adjust glob if other extensions used
                try:
                    if f_path.is_file() and f_path.stat().st_mtime < age_limit_ts:
                        logger.info(f"[{current_thread_name}] Deleting old clip (age): {f_path.name}")
                        f_path.unlink()
                        files_deleted_age += 1
                except Exception as e:
                    logger.error(f"[{current_thread_name}] Error deleting {f_path} by age: {e}")
        if files_deleted_age > 0:
            logger.info(f"[{current_thread_name}] Deleted {files_deleted_age} clips due to age.")


        # 2. Delete by disk space (oldest first if threshold breached)
        if min_disk_gb > 0:
            # Import get_disk_free_space_gb from .utils
            from .utils import get_disk_free_space_gb
            current_free_gb = get_disk_free_space_gb(clips_dir)
            logger.info(f"[{current_thread_name}] Disk space: {current_free_gb:.2f}GB free. Threshold: {min_disk_gb}GB.")
            if current_free_gb < min_disk_gb:
                logger.warning(f"[{current_thread_name}] Low disk space. Attempting to free up by deleting oldest clips...")
                # Get all clips, sort by modification time (oldest first)
                all_clips = sorted(
                    [p for p in clips_dir.glob('*.mp4') if p.is_file()],
                    key=lambda p: p.stat().st_mtime
                )
                for old_clip in all_clips:
                    if get_disk_free_space_gb(clips_dir) >= min_disk_gb:
                        logger.info(f"[{current_thread_name}] Disk space threshold met after deleting {files_deleted_space} files.")
                        break
                    try:
                        logger.info(f"[{current_thread_name}] Deleting oldest clip (space): {old_clip.name}")
                        old_clip.unlink()
                        files_deleted_space += 1
                    except Exception as e:
                        logger.error(f"[{current_thread_name}] Error deleting {old_clip} for disk space: {e}")
                if files_deleted_space > 0:
                     logger.info(f"[{current_thread_name}] Deleted {files_deleted_space} clips to free disk space.")
                if get_disk_free_space_gb(clips_dir) < min_disk_gb:
                    logger.warning(f"[{current_thread_name}] Still below disk space threshold after attempting cleanup.")

        logger.info(f"[{current_thread_name}] Cleanup cycle finished.")
    logger.info(f"[{current_thread_name}] Stopping.")