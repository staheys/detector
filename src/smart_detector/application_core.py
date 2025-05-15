# src/smart_detector/application_core.py
import cv2
import time
import logging
from pathlib import Path
from collections import deque
from ultralytics import YOLO
import numpy as np
from threading import Thread, Event, current_thread as get_current_thread
import queue
from datetime import datetime, timedelta
import json  # For metadata

from .utils import get_config, get_detection_color
from .video_utils import create_video_capture, save_video_segment, draw_detections_on_frame_for_clip
from . import config_constants as consts

logger = logging.getLogger(__name__)
_model_instance = None  # Global for YOLO model


def _load_model_once(model_path_str: str, device: str):  # Без изменений
    global _model_instance
    if _model_instance is None:
        logger.info(f"Loading model from {model_path_str} for device '{device}'")
        try:
            _model_instance = YOLO(model_path_str)
            dummy_input = np.zeros((480, 640, 3), dtype=np.uint8)
            _model_instance.predict(dummy_input, device=device, verbose=False, half=(device != "cpu"))
            logger.info(f"Model loaded and initialized on '{device}'.")
        except Exception as e:
            logger.error(f"Error loading model {model_path_str}: {e}", exc_info=True)
            _model_instance = None;
            raise
    return _model_instance


def _perform_inference(frame: np.ndarray, infer_cfg: dict, frame_dims: tuple[int, int]):  # Без изменений
    model = _model_instance;
    if model is None: return []
    results = model.predict(source=frame.copy(), conf=infer_cfg['confidence_threshold'],
                            classes=[infer_cfg['person_class_id']], device=infer_cfg['device'],
                            verbose=False, half=(infer_cfg['device'] != "cpu"))
    detections = []
    if results and results[0].boxes:
        frame_width, frame_height = frame_dims
        max_r, min_r = infer_cfg.get('max_person_bbox_area_ratio', 0.90), infer_cfg.get('min_person_bbox_area_ratio',
                                                                                        0.005)
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist();
            conf = float(box.conf[0].cpu().numpy());
            cid = int(box.cls[0].cpu().numpy())
            cname = results[0].names.get(cid, f"class_{cid}") if results[0].names else f"class_{cid}"
            if cid == infer_cfg['person_class_id']:
                x1, y1, x2, y2 = map(int, xyxy);
                bw = x2 - x1;
                bh = y2 - y1
                if bw <= 0 or bh <= 0: continue
                b_area = bw * bh;
                f_area = frame_width * frame_height;
                ratio = b_area / f_area if f_area > 0 else 0
                if not (min_r <= ratio <= max_r): logger.info(f"Filter person by area ratio: {ratio:.3f}"); continue
                detections.append({'class_id': cid, 'class_name': cname, 'confidence': conf, 'bbox_xyxy': xyxy,
                                   'frame_dims': frame_dims})  # Add frame_dims for metadata
    return detections


def capture_thread_func(stop_event: Event, frame_queue: queue.Queue, stream_cfg: dict, stream_name: str):
    """Generic capture thread for a single RTSP stream."""
    ct_name = f"CaptureThread-{stream_name}";
    get_current_thread().name = ct_name
    cap = None;
    frame_counter = 0
    url = stream_cfg['url']
    width, height, fps = stream_cfg['capture_resolution_width'], stream_cfg['capture_resolution_height'], stream_cfg[
        'capture_fps']
    reconnect_delay = stream_cfg['reconnect_delay_seconds']
    min_frame_interval = 1.0 / fps if fps > 0 else 0

    logger.info(f"[{ct_name}] Starting for URL: {url}")

    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            logger.info(f"[{ct_name}] Connecting to: {url}")
            if cap: cap.release()
            cap = create_video_capture(url, width, height, fps)
            if cap is None: logger.error(f"[{ct_name}] Retry in {reconnect_delay}s..."); stop_event.wait(
                reconnect_delay); continue
            logger.info(f"[{ct_name}] Connected.");
            frame_counter = 0

        loop_start_time = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning(f"[{ct_name}] No frame from {url}. Reconnecting...");
            if cap: cap.release(); cap = None; stop_event.wait(reconnect_delay); continue

        timestamp = time.time()
        try:
            frame_id = f"{stream_name}_f_{timestamp:.3f}_{frame_counter}"
            # For detection stream, we might add empty 'detections_for_clip'
            # For display stream, this field is not strictly needed but harmless
            item_to_queue = {'timestamp': timestamp, 'id': frame_id, 'frame': frame.copy(),
                             'raw_frame_counter': frame_counter, 'stream_name': stream_name,
                             'detections_for_clip': [] if stream_name == "detection" else None}
            frame_queue.put(item_to_queue, timeout=0.5)
            logger.debug(f"[{ct_name}] Queued frame {frame_id} (raw_seq: {frame_counter})")
            frame_counter += 1
        except queue.Full:
            logger.warning(f"[{ct_name}] Frame queue full for {stream_name}. Frame {frame_id} dropped.")

        elapsed_time = time.perf_counter() - loop_start_time
        if min_frame_interval > 0:
            sleep_duration = min_frame_interval - elapsed_time
            if sleep_duration > 0: stop_event.wait(sleep_duration)

    logger.info(f"[{ct_name}] Stopping for {url}.");
    if cap: cap.release()
    try:
        frame_queue.put_nowait(None)  # Sentinel
    except queue.Full:
        logger.warning(f"[{ct_name}] Could not put sentinel on full queue for {stream_name}.")


def run_application_logic(stop_event: Event, cfg: dict):
    get_current_thread().name = "MainAppLogicThread"
    # Config sections
    cfg_s_display_archive = cfg.get('stream_display_archive', {})
    cfg_s_detection = cfg.get('stream_detection', {})
    infer_cfg = cfg['inference']
    event_record_cfg = cfg['person_event_recording']
    periodic_record_cfg = cfg.get('periodic_archiving', {})
    display_cfg = cfg['display']

    # Load model if detection stream is enabled
    if cfg_s_detection.get('enabled', False) and event_record_cfg.get('enabled', False):
        try:
            _load_model_once(str(infer_cfg['model_path']), infer_cfg['device'])
        except Exception as e:
            logger.critical(f"Model load fail: {e}."); stop_event.set(); return

    # Queues and Threads for each stream
    display_archive_queue = None
    detection_queue = None
    capture_threads = []

    if cfg_s_display_archive.get('enabled', False):
        queue_cap = int(cfg_s_display_archive['capture_fps'] * 10)  # ~10s buffer
        display_archive_queue = queue.Queue(maxsize=queue_cap)
        t_disp = Thread(target=capture_thread_func,
                        args=(stop_event, display_archive_queue, cfg_s_display_archive, "display_archive"),
                        daemon=True)
        capture_threads.append(t_disp)

    if cfg_s_detection.get('enabled', False):
        queue_cap = int(cfg_s_detection['capture_fps'] * (event_record_cfg.get('frame_buffer_seconds', 8) + 5))
        detection_queue = queue.Queue(maxsize=queue_cap)
        t_det = Thread(target=capture_thread_func,
                       args=(stop_event, detection_queue, cfg_s_detection, "detection"),
                       daemon=True)
        capture_threads.append(t_det)

    for t in capture_threads: t.start()

    # --- State for Display and Periodic Archive (from display_archive_queue) ---
    vis_frame_for_display = None  # Latest clean frame for user display
    periodic_enabled = periodic_record_cfg.get('enabled', False) and cfg_s_display_archive.get('enabled', False)
    periodic_segment_raw_frames = []
    periodic_segment_start_time_wc = time.time()
    periodic_segment_start_ts_frame = 0
    periodic_output_dir = Path(periodic_record_cfg.get('output_directory', 'periodic_archive'))
    periodic_segment_duration_sec = periodic_record_cfg.get('segment_duration_minutes', 30) * 60
    periodic_fps = cfg_s_display_archive.get('capture_fps', 15)

    # --- State for Person Event Recording (from detection_queue) ---
    event_rec_enabled = event_record_cfg.get('enabled', False) and cfg_s_detection.get('enabled', False)
    person_buffer_maxlen = int(cfg_s_detection['capture_fps'] * event_record_cfg.get('frame_buffer_seconds', 8))
    person_recent_frames_buffer = deque(
        maxlen=person_buffer_maxlen)  # Stores {'ts', 'id', 'frame', 'detections_for_clip'} from detection stream
    is_person_event_active = False;
    last_person_seen_time = 0
    person_clip_frames_to_write = [];
    person_clip_start_ts = 0
    person_event_output_dir = Path(event_record_cfg['output_directory'])
    draw_boxes_on_person_clips = event_record_cfg.get('draw_detections_on_clips', True)
    detection_stream_fps = cfg_s_detection.get('capture_fps', 10)

    # --- General ---
    fps_display_calc_frames = 0;
    fps_display_calc_last_time = time.perf_counter();
    displayed_fps_val = 0.0
    detection_stream_raw_frame_counter = 0  # For Nth frame logic on detection stream

    try:
        while not stop_event.is_set():
            processed_display_frame = False
            processed_detection_frame = False

            # --- Process Display/Archive Stream ---
            if display_archive_queue:
                try:
                    display_data = display_archive_queue.get_nowait()  # Non-blocking
                    if display_data is None:  # Sentinel for this queue
                        logger.info("Display/Archive capture thread signaled stop.");
                        display_archive_queue = None  # Stop trying
                    else:
                        vis_frame_for_display = display_data['frame']  # Update display frame
                        processed_display_frame = True

                        # Periodic Archival (uses raw frames from display_data['frame'])
                        if periodic_enabled:
                            if not periodic_segment_raw_frames:
                                periodic_segment_start_ts_frame = display_data['timestamp']
                                periodic_segment_start_time_wc = time.time()
                                logger.info(
                                    f"Starting new periodic archive from Ch101 @ {datetime.fromtimestamp(display_data['timestamp']).isoformat()}")
                            periodic_segment_raw_frames.append(display_data['frame'].copy())

                            if time.time() - periodic_segment_start_time_wc >= periodic_segment_duration_sec:
                                logger.info(
                                    f"Periodic archive (Ch101) segment due. Finalizing from {datetime.fromtimestamp(periodic_segment_start_ts_frame).isoformat()}.")
                                first_f = next((f for f in periodic_segment_raw_frames if f is not None), None)
                                if first_f: fh, fw = first_f.shape[:2]; save_video_segment(periodic_segment_raw_frames,
                                                                                           periodic_output_dir / f"{periodic_record_cfg['filename_prefix']}_{datetime.fromtimestamp(periodic_segment_start_ts_frame).strftime('%Y%m%d-%H%M%S%f')[:-3]}",
                                                                                           periodic_fps, (fw, fh))
                                periodic_segment_raw_frames.clear()
                except queue.Empty:
                    pass  # No new frame for display/archive stream

            # --- Process Detection Stream ---
            if detection_queue:
                try:
                    detect_data = detection_queue.get_nowait()  # Non-blocking
                    if detect_data is None:  # Sentinel
                        logger.info("Detection capture thread signaled stop.");
                        detection_queue = None  # Stop trying
                    else:
                        processed_detection_frame = True
                        timestamp = detect_data['timestamp'];
                        frame_id = detect_data['id']
                        raw_frame_det_stream = detect_data['frame']
                        raw_capture_seq = detect_data['raw_frame_counter']
                        # detections_for_clip is initially empty from capture thread
                        # Will be filled by inference if it runs

                        if raw_frame_det_stream is None or raw_frame_det_stream.shape[0] == 0:
                            logger.warning(f"Invalid frame {frame_id} from detection queue.");
                            continue

                        fh_det, fw_det = raw_frame_det_stream.shape[:2]
                        current_buffer_item = {'timestamp': timestamp, 'id': frame_id, 'frame': raw_frame_det_stream,
                                               'detections_for_clip': []}
                        if event_rec_enabled: person_recent_frames_buffer.append(current_buffer_item)

                        detections_this_cycle = []
                        run_inference = (raw_capture_seq % infer_cfg['process_every_nth_frame'] == 0)
                        if run_inference and event_rec_enabled:  # Only run inference if event recording is on
                            inf_start = time.perf_counter()
                            detections_this_cycle = _perform_inference(raw_frame_det_stream, infer_cfg,
                                                                       (fw_det, fh_det))
                            logger.debug(
                                f"DetStream Inf {frame_id} (seq:{raw_capture_seq}) {(time.perf_counter() - inf_start) * 1000:.1f}ms. Found {len(detections_this_cycle)} persons.")
                            if person_recent_frames_buffer and person_recent_frames_buffer[-1]['id'] == frame_id:
                                person_recent_frames_buffer[-1]['detections_for_clip'] = detections_this_cycle

                        person_in_inference = any(
                            d['class_id'] == infer_cfg['person_class_id'] for d in detections_this_cycle)

                        if event_rec_enabled:
                            if person_in_inference:
                                last_person_seen_time = time.time()
                                if not is_person_event_active:
                                    is_person_event_active = True;
                                    actual_clip_start_ts = timestamp
                                    temp_clip_frames = []
                                    pre_event_ts_target = timestamp - event_record_cfg['pre_event_seconds']
                                    for item_data in list(person_recent_frames_buffer):
                                        if item_data['timestamp'] >= pre_event_ts_target and item_data[
                                            'timestamp'] <= timestamp:
                                            frame_c = item_data['frame']
                                            dets_c = item_data.get('detections_for_clip', [])
                                            if draw_boxes_on_person_clips and dets_c:
                                                frame_c = draw_detections_on_frame_for_clip(frame_c, dets_c, infer_cfg[
                                                    'person_class_id'], get_detection_color)
                                            temp_clip_frames.append(frame_c)
                                            if item_data['timestamp'] < actual_clip_start_ts: actual_clip_start_ts = \
                                            item_data['timestamp']
                                    if not temp_clip_frames:  # Fallback
                                        f_add = raw_frame_det_stream;
                                        if draw_boxes_on_person_clips and detections_this_cycle: f_add = draw_detections_on_frame_for_clip(
                                            f_add, detections_this_cycle, infer_cfg['person_class_id'],
                                            get_detection_color)
                                        temp_clip_frames.append(f_add)
                                    person_clip_frames_to_write.extend(temp_clip_frames);
                                    person_clip_start_ts = actual_clip_start_ts
                                    logger.info(
                                        f"Person event started (Ch102) for {frame_id}. Clip starts ~{datetime.fromtimestamp(person_clip_start_ts).isoformat()}. Added {len(temp_clip_frames)} pre-roll.")

                            if is_person_event_active:  # Add current (possibly annotated) frame to active clip
                                f_add_active = raw_frame_det_stream
                                if draw_boxes_on_person_clips and detections_this_cycle:
                                    f_add_active = draw_detections_on_frame_for_clip(f_add_active,
                                                                                     detections_this_cycle,
                                                                                     infer_cfg['person_class_id'],
                                                                                     get_detection_color)
                                if not person_clip_frames_to_write or not np.array_equal(
                                        person_clip_frames_to_write[-1], f_add_active):
                                    person_clip_frames_to_write.append(f_add_active)

                            if is_person_event_active and not person_in_inference and \
                                    (time.time() - last_person_seen_time > event_record_cfg[
                                        'post_event_timeout_seconds']):
                                logger.info(
                                    f"Person event timeout (Ch102). Finalizing clip from {datetime.fromtimestamp(person_clip_start_ts).isoformat()}.")
                                if person_clip_frames_to_write:
                                    # Save AVI
                                    clip_filename_base = f"{event_record_cfg['filename_prefix']}_{datetime.fromtimestamp(person_clip_start_ts).strftime('%Y%m%d-%H%M%S%f')[:-3]}"
                                    clip_filepath_base = person_event_output_dir / clip_filename_base

                                    first_f = next((f for f in person_clip_frames_to_write if f is not None), None)
                                    if first_f:
                                        fh_ev, fw_ev = first_f.shape[:2]
                                        save_video_segment(person_clip_frames_to_write, clip_filepath_base,
                                                           detection_stream_fps, (fw_ev, fh_ev))
                                        # Save Metadata
                                        meta_data_to_save = {
                                            "event_start_timestamp": person_clip_start_ts,
                                            "event_start_iso": datetime.fromtimestamp(person_clip_start_ts).isoformat(),
                                            "video_filename": clip_filepath_base.with_suffix('.avi').name,
                                            "source_stream_config": "stream_detection",  # Indicate source
                                            "total_frames": len(person_clip_frames_to_write),
                                            # Could add first/last detection details if needed
                                        }
                                        meta_filepath = clip_filepath_base.with_suffix('.meta.json')
                                        try:
                                            with open(meta_filepath, 'w') as f_meta:
                                                json.dump(meta_data_to_save, f_meta, indent=2)
                                            logger.info(f"Saved metadata: {meta_filepath}")
                                        except Exception as e_meta:
                                            logger.error(f"Failed to save metadata {meta_filepath}: {e_meta}")
                                person_clip_frames_to_write.clear();
                                is_person_event_active = False
                except queue.Empty:
                    pass  # No new frame for detection stream

            # --- Combined Timeout Finalization (if queues become None due to thread stop) ---
            if display_archive_queue is None and periodic_enabled and periodic_segment_raw_frames:  # Display stream stopped
                logger.info("Display stream ended. Finalizing pending periodic archive.")
                # ... (save periodic segment logic) ...
                first_f = next((f for f in periodic_segment_raw_frames if f is not None), None);
                if first_f: fh, fw = first_f.shape[:2]; save_video_segment(periodic_segment_raw_frames,
                                                                           periodic_output_dir / f"{periodic_record_cfg['filename_prefix']}_force_end_{datetime.fromtimestamp(periodic_segment_start_ts_frame).strftime('%Y%m%d-%H%M%S%f')[:-3]}",
                                                                           periodic_fps, (fw, fh))
                periodic_segment_raw_frames.clear()

            if detection_queue is None and event_rec_enabled and is_person_event_active and person_clip_frames_to_write:  # Detection stream stopped
                logger.info("Detection stream ended. Finalizing pending person event clip.")
                # ... (save person event clip logic with metadata) ...
                clip_fn_base = f"{event_record_cfg['filename_prefix']}_force_end_{datetime.fromtimestamp(person_clip_start_ts).strftime('%Y%m%d-%H%M%S%f')[:-3]}"
                clip_fp_base = person_event_output_dir / clip_fn_base
                first_f = next((f for f in person_clip_frames_to_write if f is not None), None);
                if first_f: fh, fw = first_f.shape[:2]; save_video_segment(person_clip_frames_to_write, clip_fp_base,
                                                                           detection_stream_fps, (fw, fh))  # Save AVI
                # ... (save metadata)
                meta_fp = clip_fp_base.with_suffix('.meta.json')
                # ... (json.dump as above) ...
                person_clip_frames_to_write.clear()

            # --- Display Update (shows clean vis_frame_for_display from display_archive_queue) ---
            if display_cfg['show_window'] and vis_frame_for_display is not None:
                fps_display_calc_frames += 1;
                loop_time = time.perf_counter()
                if loop_time - fps_display_calc_last_time >= display_cfg['fps_display_interval']:
                    if (loop_time - fps_display_calc_last_time) > 0: displayed_fps_val = fps_display_calc_frames / (
                                loop_time - fps_display_calc_last_time)
                    fps_display_calc_frames = 0;
                    fps_display_calc_last_time = loop_time

                display_overlay_frame = vis_frame_for_display.copy()
                cv2.putText(display_overlay_frame, f"FPS: {displayed_fps_val:.1f} (Disp)", consts.INFO_TEXT_POSITION,
                            consts.CV2_FONT, consts.CV2_FONT_SCALE_INFO, consts.CV2_COLOR_FPS,
                            consts.CV2_FONT_THICKNESS)
                if is_person_event_active and event_rec_enabled:
                    cv2.putText(display_overlay_frame, "REC (Person Ch102)", consts.RECORDING_INDICATOR_POSITION,
                                consts.CV2_FONT, consts.CV2_FONT_SCALE_INFO, consts.CV2_COLOR_RECORDING_INDICATOR, 2)
                if periodic_enabled and periodic_segment_raw_frames:  # Check if frames are being collected for periodic
                    cv2.putText(display_overlay_frame, "REC (Archive Ch101)", consts.ARCHIVE_INDICATOR_POSITION,
                                consts.CV2_FONT, consts.CV2_FONT_SCALE_INFO, consts.CV2_COLOR_ARCHIVE_INDICATOR, 2)
                try:
                    cv2.imshow(display_cfg['window_name'], display_overlay_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): stop_event.set(); break
                except cv2.error as e:
                    if "NULL window" in str(e):
                        logger.warning("OpenCV window closed."); display_cfg['show_window'] = False
                    else:
                        logger.error(f"OpenCV display error: {e}"); stop_event.set(); break

            # If no frames processed from either queue, sleep briefly to prevent busy-looping 100% CPU
            if not processed_display_frame and not processed_detection_frame:
                time.sleep(0.005)  # 5ms sleep

            # Check if both capture threads have finished (queues became None)
            if display_archive_queue is None and detection_queue is None and not stop_event.is_set():
                logger.info("All capture threads seem to have stopped. Exiting main loop.")
                break


    except Exception as e:
        logger.critical(f"CRITICAL MAIN APP LOGIC ERROR: {e}", exc_info=True); stop_event.set()
    finally:
        logger.info("Main application logic finalizing...")
        # Finalize person event clip if active
        if event_rec_enabled and is_person_event_active and person_clip_frames_to_write:
            logger.info("Shutdown: Saving pending person event clip (Ch102)...")
            clip_fn_base = f"{event_record_cfg['filename_prefix']}_shutdown_{datetime.fromtimestamp(person_clip_start_ts if person_clip_start_ts else time.time()).strftime('%Y%m%d-%H%M%S%f')[:-3]}"
            clip_fp_base = person_event_output_dir / clip_fn_base
            first_f = next((f for f in person_clip_frames_to_write if f is not None), None)
            if first_f: fh, fw = first_f.shape[:2]; save_video_segment(person_clip_frames_to_write, clip_fp_base,
                                                                       detection_stream_fps, (fw, fh))
            meta_fp = clip_fp_base.with_suffix('.meta.json')
            # ... (save metadata for shutdown clip)
            try:
                with open(meta_fp, 'w') as f_meta:
                    json.dump({"event_start_timestamp": person_clip_start_ts, "reason": "shutdown",
                               "video_filename": clip_fp_base.with_suffix('.avi').name}, f_meta, indent=2)
            except Exception as e_m:
                logger.error(f"Err saving shutdown meta: {e_m}")

        # Finalize periodic archive clip if active
        if periodic_enabled and periodic_segment_raw_frames:
            logger.info("Shutdown: Saving pending periodic archive segment (Ch101)...")
            first_f = next((f for f in periodic_segment_raw_frames if f is not None), None)
            if first_f: fh, fw = first_f.shape[:2];save_video_segment(periodic_segment_raw_frames,
                                                                      periodic_output_dir / f"{periodic_record_cfg['filename_prefix']}_shutdown_{datetime.fromtimestamp(periodic_segment_start_ts_frame if periodic_segment_start_ts_frame else time.time()).strftime('%Y%m%d-%H%M%S%f')[:-3]}",
                                                                      periodic_fps, (fw, fh))

        if display_cfg.get('show_window', False):
            try:
                cv2.destroyAllWindows()
            except:
                pass

        logger.info("Waiting for capture threads to complete shutdown...")
        for t in capture_threads:
            if t.is_alive():
                # Queues should have received None sentinel from their respective threads
                t.join(timeout=3.0)
                if t.is_alive(): logger.warning(f"Capture thread {t.name} did not finish.")
        logger.info("Main application logic stopped.")