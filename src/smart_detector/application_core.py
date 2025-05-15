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
import json

from .utils import get_detection_color
from .video_utils import create_video_capture, save_video_segment, draw_detections_on_frame_for_clip
from . import config_constants as consts

logger = logging.getLogger(__name__)
_model_instance = None


# _load_model_once и _perform_inference остаются без изменений от предыдущей версии
def _load_model_once(model_path_str: str, device: str):
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


def _perform_inference(frame: np.ndarray, infer_cfg: dict, frame_dims: tuple[int, int]):
    model = _model_instance;
    if model is None: return []
    # Проверяем, что кадр не пустой перед передачей в модель
    if frame is None or frame.size == 0:
        logger.warning("Performing inference on an empty or None frame. Skipping.")
        return []
    results = model.predict(source=frame.copy(), conf=infer_cfg['confidence_threshold'],
                            classes=[infer_cfg['person_class_id']], device=infer_cfg['device'],
                            verbose=False, half=(infer_cfg['device'] != "cpu"))
    detections = []
    if results and results[0].boxes:
        frame_width, frame_height = frame_dims
        max_r = infer_cfg.get('max_person_bbox_area_ratio', 0.90)
        min_r = infer_cfg.get('min_person_bbox_area_ratio', 0.005)
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
                if not (min_r <= ratio <= max_r):
                    logger.debug(
                        f"Filtering 'person' due to area ratio: {ratio:.3f} (conf: {conf:.2f}). Allowed: [{min_r}-{max_r}].")
                    continue
                detections.append({'class_id': cid, 'class_name': cname, 'confidence': conf, 'bbox_xyxy': xyxy,
                                   'frame_dims': frame_dims})
    return detections


# capture_thread_func остается без изменений от предыдущей версии
def capture_thread_func(stop_event: Event, frame_queue: queue.Queue, stream_cfg: dict, stream_name: str):
    ct_name = f"CaptureThread-{stream_name}";
    get_current_thread().name = ct_name
    cap = None;
    frame_counter = 0
    url = stream_cfg['url']
    width, height, fps = stream_cfg['capture_resolution_width'], stream_cfg['capture_resolution_height'], stream_cfg[
        'capture_fps']
    reconnect_delay = stream_cfg['reconnect_delay_seconds']
    min_frame_interval = 1.0 / fps if fps > 0 else 0
    logger.info(f"[{ct_name}] Starting for URL: {url} @ {fps} FPS target")
    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            logger.info(f"[{ct_name}] Connecting to: {url}")
            if cap: cap.release(); cap = None
            cap = create_video_capture(url, width, height, fps)
            if cap is None: logger.error(f"[{ct_name}] Failed. Retry in {reconnect_delay}s..."); stop_event.wait(
                reconnect_delay); continue
            logger.info(f"[{ct_name}] Connected to {url}.");
            frame_counter = 0
        loop_start_time = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:  # Добавим проверку frame is None здесь
            logger.warning(
                f"[{ct_name}] No frame from {url} (ret={ret}, frame is None={frame is None}). Reconnecting...");
            if cap: cap.release(); cap = None; stop_event.wait(reconnect_delay); continue  # Используем reconnect_delay
        timestamp = time.time()
        try:
            frame_id = f"{stream_name}_f_{timestamp:.3f}_{frame_counter}"
            item_to_queue = {'timestamp': timestamp, 'id': frame_id, 'frame': frame.copy(),  # Копируем кадр
                             'raw_frame_counter': frame_counter, 'stream_name': stream_name,
                             'detections_for_clip': []}
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
        frame_queue.put_nowait(None)
    except queue.Full:
        logger.warning(f"[{ct_name}] Could not put sentinel on full queue for {stream_name} during shutdown.")


def display_thread_func(stop_event: Event, display_frame_queue: queue.Queue, display_cfg: dict, app_cfg: dict):
    get_current_thread().name = "DisplayThread"
    window_name = display_cfg.get('window_name', "Live Feed (Clean)")
    logger.info(f"[DisplayThread] Started. Window: '{window_name}'")
    fps_calc_frames = 0;
    fps_calc_last_time = time.perf_counter();
    displayed_fps_val = 0.0
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    except Exception as e:
        logger.error(f"[DisplayThread] Failed to create OpenCV window: {e}"); return

    while not stop_event.is_set():
        try:
            frame_data = display_frame_queue.get(timeout=0.05)
            if frame_data is None: logger.info("[DisplayThread] Received sentinel. Stopping."); break

            raw_frame_to_display = frame_data['frame']
            # ИСПРАВЛЕНИЕ: Проверка, что кадр не None и не пустой
            if raw_frame_to_display is None or raw_frame_to_display.size == 0:
                logger.debug("[DisplayThread] Received invalid frame for display (None or empty).")
                time.sleep(0.01);
                continue

            display_overlay_frame = raw_frame_to_display.copy()
            fps_calc_frames += 1;
            loop_time = time.perf_counter()
            if loop_time - fps_calc_last_time >= display_cfg['fps_display_interval']:
                if (loop_time - fps_calc_last_time) > 0: displayed_fps_val = fps_calc_frames / (
                            loop_time - fps_calc_last_time)
                fps_calc_frames = 0;
                fps_calc_last_time = loop_time
            cv2.putText(display_overlay_frame, f"FPS: {displayed_fps_val:.1f} (Disp)", consts.INFO_TEXT_POSITION,
                        consts.CV2_FONT, consts.CV2_FONT_SCALE_INFO, consts.CV2_COLOR_FPS, consts.CV2_FONT_THICKNESS)

            runtime_state = app_cfg.get('runtime_state', {})
            if runtime_state.get('is_person_event_active', False):
                cv2.putText(display_overlay_frame, "REC (Person Ch102)", consts.RECORDING_INDICATOR_POSITION,
                            consts.CV2_FONT, consts.CV2_FONT_SCALE_INFO, consts.CV2_COLOR_RECORDING_INDICATOR, 2)
            if runtime_state.get('is_periodic_archiving_active', False):
                cv2.putText(display_overlay_frame, "REC (Archive Ch101)", consts.ARCHIVE_INDICATOR_POSITION,
                            consts.CV2_FONT, consts.CV2_FONT_SCALE_INFO, consts.CV2_COLOR_ARCHIVE_INDICATOR, 2)

            cv2.imshow(window_name, display_overlay_frame)
            key = cv2.waitKey(1)
            if key != -1 and (key & 0xFF == ord('q') or key == 27):  # q или Esc
                logger.info("[DisplayThread] 'q' or ESC pressed. Signaling global shutdown.")
                stop_event.set();
                break
        except queue.Empty:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("[DisplayThread] Window closed by user. Signaling global shutdown.")
                stop_event.set();
                break
            key_check = cv2.waitKey(1)
            if key_check != -1 and (key_check & 0xFF == ord('q') or key_check == 27): stop_event.set(); break
            continue
        except cv2.error as e_cv:
            if "NULL window" in str(e_cv) or "Invalid window handle" in str(e_cv):
                logger.warning("[DisplayThread] OpenCV window closed externally."); break
            else:
                logger.error(f"[DisplayThread] OpenCV display error: {e_cv}", exc_info=True); break
        except Exception as e:
            logger.error(f"[DisplayThread] Unexpected error: {e}", exc_info=True); break
    logger.info("[DisplayThread] Stopping.");
    try:
        cv2.destroyWindow(window_name)
    except Exception:
        pass


# Вспомогательные функции для финализации клипов
def _finalize_person_clip(reason, event_record_cfg, person_event_output_dir, detection_stream_fps,
                          person_clip_frames_to_write_ref, person_clip_start_ts, is_person_event_active_state_ref):
    # is_person_event_active_state_ref - это список [is_active_bool]
    # person_clip_frames_to_write_ref - это сам список person_clip_frames_to_write
    if not person_clip_frames_to_write_ref:  # Проверяем, есть ли кадры
        logger.info(
            f"Person event ({reason}). No frames to write for clip from {datetime.fromtimestamp(person_clip_start_ts).isoformat()}.")
        is_person_event_active_state_ref[0] = False  # Сбрасываем флаг
        return

    logger.info(
        f"Person event ({reason}). Finalizing clip from {datetime.fromtimestamp(person_clip_start_ts).isoformat()}. Frames: {len(person_clip_frames_to_write_ref)}")
    clip_filename_base = f"{event_record_cfg['filename_prefix']}_{datetime.fromtimestamp(person_clip_start_ts).strftime('%Y%m%d-%H%M%S%f')[:-3]}"
    clip_filepath_base = person_event_output_dir / clip_filename_base

    # ИСПРАВЛЕНИЕ ValueError: Проверяем, что first_f не None перед использованием
    first_f = next((f for f in person_clip_frames_to_write_ref if f is not None and f.size > 0),
                   None)  # Добавил f.size > 0
    if first_f is not None:  # Явная проверка на None
        fh_ev, fw_ev = first_f.shape[:2]
        if save_video_segment(person_clip_frames_to_write_ref, clip_filepath_base, detection_stream_fps,
                              (fw_ev, fh_ev)):
            meta_data = {"event_start_timestamp": person_clip_start_ts,
                         "event_start_iso": datetime.fromtimestamp(person_clip_start_ts).isoformat(),
                         "video_filename": clip_filepath_base.with_suffix('.avi').name,
                         "source_stream_config": "stream_detection",
                         "total_frames": len(person_clip_frames_to_write_ref),
                         "reason_for_finalize": reason}
            meta_filepath = clip_filepath_base.with_suffix('.meta.json')
            try:
                with open(meta_filepath, 'w') as f_meta:
                    json.dump(meta_data, f_meta, indent=2)
                logger.info(f"Saved metadata: {meta_filepath}")
            except Exception as e_meta:
                logger.error(f"Failed to save metadata {meta_filepath}: {e_meta}")
    else:
        logger.warning(
            f"Could not finalize person clip for event at {datetime.fromtimestamp(person_clip_start_ts).isoformat()}: No valid frames found in buffer.")

    person_clip_frames_to_write_ref.clear()
    is_person_event_active_state_ref[0] = False  # Обновляем состояние через ссылку


def _finalize_periodic_clip(reason, periodic_record_cfg, periodic_output_dir, periodic_fps,
                            periodic_segment_raw_frames_ref, periodic_segment_start_ts_frame,
                            periodic_segment_start_time_wc_state_ref):
    # periodic_segment_start_time_wc_state_ref - это список [wc_time, frame_ts]
    # periodic_segment_raw_frames_ref - это сам список periodic_segment_raw_frames
    if not periodic_segment_raw_frames_ref:
        logger.info(
            f"Periodic archive ({reason}). No frames to write for segment from {datetime.fromtimestamp(periodic_segment_start_ts_frame).isoformat()}.")
        # Сбрасываем время начала следующего сегмента, даже если этот был пуст
        periodic_segment_start_time_wc_state_ref[0] = time.time()
        periodic_segment_start_time_wc_state_ref[1] = 0
        return

    logger.info(
        f"Periodic archive ({reason}). Finalizing segment from {datetime.fromtimestamp(periodic_segment_start_ts_frame).isoformat()}. Frames: {len(periodic_segment_raw_frames_ref)}")

    # ИСПРАВЛЕНИЕ ValueError:
    first_f = next((f for f in periodic_segment_raw_frames_ref if f is not None and f.size > 0),
                   None)  # Добавил f.size > 0
    if first_f is not None:  # Явная проверка
        fh, fw = first_f.shape[:2]
        save_video_segment(periodic_segment_raw_frames_ref,
                           periodic_output_dir / f"{periodic_record_cfg['filename_prefix']}_{datetime.fromtimestamp(periodic_segment_start_ts_frame).strftime('%Y%m%d-%H%M%S%f')[:-3]}",
                           periodic_fps, (fw, fh))
    else:
        logger.warning(
            f"Could not finalize periodic clip from {datetime.fromtimestamp(periodic_segment_start_ts_frame).isoformat()}: No valid frames.")

    periodic_segment_raw_frames_ref.clear()
    periodic_segment_start_time_wc_state_ref[0] = time.time()
    periodic_segment_start_time_wc_state_ref[1] = 0


def run_application_logic(stop_event: Event, cfg: dict):
    get_current_thread().name = "MainAppLogicThread"
    cfg_s_display_archive = cfg.get('stream_display_archive', {})
    cfg_s_detection = cfg.get('stream_detection', {})
    infer_cfg = cfg['inference']
    event_record_cfg = cfg['person_event_recording']
    periodic_record_cfg = cfg.get('periodic_archiving', {})
    display_cfg = cfg['display']

    cfg['runtime_state'] = {'is_person_event_active': False, 'is_periodic_archiving_active': False}

    if cfg_s_detection.get('enabled', False) and event_record_cfg.get('enabled', False):
        try:
            _load_model_once(str(infer_cfg['model_path']), infer_cfg['device'])
        except Exception as e:
            logger.critical(f"Model load fail: {e}."); stop_event.set(); return

    display_archive_queue = None;
    detection_queue = None;
    display_feed_queue = None
    capture_threads = [];
    other_threads = []

    if cfg_s_display_archive.get('enabled', False):
        fps_da = cfg_s_display_archive.get('capture_fps', 15)
        display_archive_queue = queue.Queue(maxsize=int(fps_da * 5))
        display_feed_queue = queue.Queue(maxsize=int(fps_da * 2))
        t_disp_cap = Thread(target=capture_thread_func,
                            args=(stop_event, display_archive_queue, cfg_s_display_archive, "display_archive"),
                            daemon=True, name="CaptureThread-DisplayArchive")
        capture_threads.append(t_disp_cap)
        if display_cfg.get('show_window', False):
            t_display_feed = Thread(target=display_thread_func, args=(stop_event, display_feed_queue, display_cfg, cfg),
                                    daemon=True, name="DisplayFeedThread")
            other_threads.append(t_display_feed)

    if cfg_s_detection.get('enabled', False):
        fps_dt = cfg_s_detection.get('capture_fps', 10)
        buffer_s = event_record_cfg.get('frame_buffer_seconds', 8)
        detection_queue = queue.Queue(maxsize=int(fps_dt * (buffer_s + 2)))  # Уменьшил немного буфер
        t_det_cap = Thread(target=capture_thread_func, args=(stop_event, detection_queue, cfg_s_detection, "detection"),
                           daemon=True, name="CaptureThread-Detection")
        capture_threads.append(t_det_cap)

    for t in capture_threads: t.start()
    for t in other_threads: t.start()

    periodic_enabled = periodic_record_cfg.get('enabled', False) and cfg_s_display_archive.get('enabled', False)
    periodic_segment_raw_frames = [];
    _periodic_segment_start_time_wc_state = [time.time(), 0]  # [wall_clock_start, first_frame_timestamp]
    periodic_output_dir = Path(periodic_record_cfg.get('output_directory', 'periodic_archive'))
    periodic_segment_duration_sec = periodic_record_cfg.get('segment_duration_minutes', 30) * 60
    periodic_fps = cfg_s_display_archive.get('capture_fps', 15)

    event_rec_enabled = event_record_cfg.get('enabled', False) and cfg_s_detection.get('enabled', False)
    person_buffer_maxlen = int(cfg_s_detection.get('capture_fps', 10) * event_record_cfg.get('frame_buffer_seconds', 8))
    person_recent_frames_buffer = deque(maxlen=person_buffer_maxlen)
    _is_person_event_active_state = [False]  # Используем список для передачи по ссылке в lambda/finalize
    last_person_seen_time = 0
    person_clip_frames_to_write = []  # Будет содержать кадры с нарисованными рамками
    person_clip_start_ts = 0
    person_event_output_dir = Path(event_record_cfg['output_directory'])
    draw_boxes_on_person_clips = event_record_cfg.get('draw_detections_on_clips', True)
    detection_stream_fps = cfg_s_detection.get('capture_fps', 10)

    _finalize_person_clip_func = lambda reason: _finalize_person_clip(reason, event_record_cfg, person_event_output_dir,
                                                                      detection_stream_fps, person_clip_frames_to_write,
                                                                      person_clip_start_ts,
                                                                      _is_person_event_active_state)
    _finalize_periodic_clip_func = lambda reason: _finalize_periodic_clip(reason, periodic_record_cfg,
                                                                          periodic_output_dir, periodic_fps,
                                                                          periodic_segment_raw_frames,
                                                                          _periodic_segment_start_time_wc_state[1],
                                                                          _periodic_segment_start_time_wc_state)

    try:
        while not stop_event.is_set():
            processed_disp_f_cycle = False;
            processed_det_f_cycle = False

            if display_archive_queue:
                try:
                    display_data = display_archive_queue.get(timeout=0.005)  # Очень короткий таймаут
                    if display_data is None:
                        display_archive_queue = None; logger.info("Display/Archive capture source stopped.")
                    else:
                        processed_disp_f_cycle = True
                        if display_feed_queue and display_cfg.get('show_window'):
                            try:
                                display_feed_queue.put_nowait(display_data)
                            except queue.Full:
                                logger.debug("[MainLogic] Display feed queue full.")  # Debug, т.к. может быть часто

                        if periodic_enabled:
                            if not periodic_segment_raw_frames:  # Start new segment
                                _periodic_segment_start_time_wc_state[1] = display_data['timestamp']  # first_frame_ts
                                _periodic_segment_start_time_wc_state[0] = time.time()  # wall_clock_start
                                logger.info(
                                    f"Starting new periodic archive (Ch101) from {datetime.fromtimestamp(display_data['timestamp']).isoformat()}.")
                                cfg['runtime_state']['is_periodic_archiving_active'] = True
                            periodic_segment_raw_frames.append(display_data['frame'].copy())
                            if time.time() - _periodic_segment_start_time_wc_state[0] >= periodic_segment_duration_sec:
                                _finalize_periodic_clip_func("duration_met")
                                cfg['runtime_state']['is_periodic_archiving_active'] = False
                except queue.Empty:
                    pass

            if detection_queue:
                try:
                    detect_data = detection_queue.get(timeout=0.005)  # Очень короткий таймаут
                    if detect_data is None:
                        detection_queue = None; logger.info("Detection capture source stopped.")
                    else:
                        processed_det_f_cycle = True
                        ts, fid, raw_f_det, raw_seq = detect_data['timestamp'], detect_data['id'], detect_data['frame'], \
                        detect_data['raw_frame_counter']
                        if raw_f_det is None or raw_f_det.size == 0: logger.warning(
                            f"Invalid frame {fid} from det_q."); continue  # Проверка .size
                        fh_det, fw_det = raw_f_det.shape[:2]
                        current_buffer_item = {'timestamp': ts, 'id': fid, 'frame': raw_f_det,
                                               'detections_for_clip': []}
                        if event_rec_enabled: person_recent_frames_buffer.append(current_buffer_item)

                        dets_this_cycle = []
                        if (raw_seq % infer_cfg['process_every_nth_frame'] == 0) and event_rec_enabled:
                            inf_s = time.perf_counter()
                            dets_this_cycle = _perform_inference(raw_f_det, infer_cfg, (fw_det, fh_det))
                            logger.debug(
                                f"DetStream Inf {fid}(seq:{raw_seq}) {(time.perf_counter() - inf_s) * 1000:.1f}ms. Found {len(dets_this_cycle)}p.")
                            if event_rec_enabled and person_recent_frames_buffer and person_recent_frames_buffer[-1][
                                'id'] == fid:
                                person_recent_frames_buffer[-1]['detections_for_clip'] = dets_this_cycle

                        person_in_inf = any(d['class_id'] == infer_cfg['person_class_id'] for d in dets_this_cycle)
                        if event_rec_enabled:
                            if person_in_inf:
                                last_person_seen_time = time.time()
                                if not _is_person_event_active_state[0]:  # Если событие неактивно
                                    _is_person_event_active_state[0] = True;
                                    cfg['runtime_state']['is_person_event_active'] = True
                                    actual_clip_start_ts = ts
                                    temp_clip_f = []
                                    pre_event_target = ts - event_record_cfg['pre_event_seconds']
                                    for item_d in list(person_recent_frames_buffer):
                                        if item_d['timestamp'] >= pre_event_target and item_d['timestamp'] <= ts:
                                            f_c = item_d['frame'];
                                            dets_c = item_d.get('detections_for_clip', [])
                                            if draw_boxes_on_person_clips and dets_c: f_c = draw_detections_on_frame_for_clip(
                                                f_c, dets_c, infer_cfg['person_class_id'], get_detection_color)
                                            temp_clip_f.append(f_c)
                                            if item_d['timestamp'] < actual_clip_start_ts: actual_clip_start_ts = \
                                            item_d['timestamp']
                                    if not temp_clip_f:
                                        f_add = raw_f_det.copy();  # Копируем, чтобы не изменить оригинал в буфере
                                        if draw_boxes_on_person_clips and dets_this_cycle: f_add = draw_detections_on_frame_for_clip(
                                            f_add, dets_this_cycle, infer_cfg['person_class_id'], get_detection_color)
                                        temp_clip_f.append(f_add)
                                    person_clip_frames_to_write.extend(temp_clip_f);
                                    person_clip_start_ts = actual_clip_start_ts
                                    logger.info(
                                        f"Person event (Ch102) for {fid}. Clip starts ~{datetime.fromtimestamp(person_clip_start_ts).isoformat()}. Added {len(temp_clip_f)} pre-roll.")

                            if _is_person_event_active_state[0]:
                                f_add_active = raw_f_det.copy()  # Копируем
                                if draw_boxes_on_person_clips and dets_this_cycle: f_add_active = draw_detections_on_frame_for_clip(
                                    f_add_active, dets_this_cycle, infer_cfg['person_class_id'], get_detection_color)
                                if not person_clip_frames_to_write or not np.array_equal(
                                    person_clip_frames_to_write[-1], f_add_active): person_clip_frames_to_write.append(
                                    f_add_active)

                            if _is_person_event_active_state[0] and not person_in_inf and (
                                    time.time() - last_person_seen_time > event_record_cfg[
                                'post_event_timeout_seconds']):
                                _finalize_person_clip_func("timeout")
                                cfg['runtime_state']['is_person_event_active'] = False  # Обновляем состояние
                except queue.Empty:
                    pass

            if not processed_det_f_cycle and event_rec_enabled and _is_person_event_active_state[0] and \
                    (time.time() - last_person_seen_time > event_record_cfg['post_event_timeout_seconds']):
                _finalize_person_clip_func("timeout_no_new_detection_frames")
                cfg['runtime_state']['is_person_event_active'] = False

            if not processed_disp_f_cycle and periodic_enabled and periodic_segment_raw_frames and \
                    (time.time() - _periodic_segment_start_time_wc_state[0] >= periodic_segment_duration_sec):
                _finalize_periodic_clip_func("duration_no_new_display_frames")
                cfg['runtime_state']['is_periodic_archiving_active'] = False

            if not processed_disp_f_cycle and not processed_det_f_cycle: time.sleep(0.001)  # Уменьшил паузу

            if (display_archive_queue is None or not cfg_s_display_archive.get('enabled', False)) and \
                    (detection_queue is None or not cfg_s_detection.get('enabled', False)) and \
                    not stop_event.is_set():
                logger.info("All relevant capture queues stopped. Exiting main logic loop.")
                break
    except Exception as e:
        logger.critical(f"CRITICAL MAIN APP LOGIC ERROR: {e}", exc_info=True); stop_event.set()
    finally:
        logger.info("Main application logic finalizing...")
        if event_rec_enabled and _is_person_event_active_state[0]: _finalize_person_clip_func("shutdown")
        if periodic_enabled and periodic_segment_raw_frames: _finalize_periodic_clip_func("shutdown")

        if display_feed_queue:
            try:
                display_feed_queue.put_nowait(None)
            except queue.Full:
                pass
        logger.info("Waiting for ALL threads to complete shutdown...")
        all_threads_to_join = capture_threads + other_threads
        for t in all_threads_to_join:
            if t.is_alive():
                logger.debug(f"Joining thread {t.name}...")
                t.join(timeout=2.0)  # Уменьшил таймаут для более быстрой остановки
                if t.is_alive(): logger.warning(f"Thread {t.name} did not finish in time.")
            else:
                logger.debug(f"Thread {t.name} already stopped.")
        logger.info("Main application logic stopped.")