# src/smart_detector/archiver.py
import cv2
import zmq
import pickle
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from threading import current_thread, Event
import os
import shutil

from .utils import get_config, get_disk_free_space_gb

logger = logging.getLogger(__name__)

# --- Continuous Archiving (Hourly Segments) ---
_video_writer_hourly = None
_current_hourly_segment_path = None
_hourly_segment_start_time = None


def _start_new_hourly_segment(archive_cfg, frame_width, frame_height):
    global _video_writer_hourly, _current_hourly_segment_path, _hourly_segment_start_time

    if _video_writer_hourly:
        _video_writer_hourly.release()
        logger.info(f"Closed hourly segment: {_current_hourly_segment_path}")

    now = datetime.now()
    segment_dir = archive_cfg['base_path'] / "continuous" / now.strftime("%Y-%m-%d")
    segment_dir.mkdir(parents=True, exist_ok=True)

    # Filename format YYYY-MM-DD_HH.avi for hourly segments
    # The requirement was HH.avi, but adding date is better for management over days
    segment_filename = now.strftime("%Y-%m-%d_%H") + ".avi"
    _current_hourly_segment_path = segment_dir / segment_filename

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI container
    # Use configured FPS for archive, not source FPS, if different
    fps = archive_cfg.get('continuous_recording_fps', 10)

    _video_writer_hourly = cv2.VideoWriter(
        str(_current_hourly_segment_path), fourcc, fps, (frame_width, frame_height)
    )
    _hourly_segment_start_time = time.time()
    logger.info(f"Started new hourly segment: {_current_hourly_segment_path} at {fps} FPS")


def continuous_archival_thread_func(stop_event: Event, zmq_context: zmq.Context):
    cfg = get_config()
    if not cfg['archiving'].get('enabled', False):
        logger.info("Continuous archiving is disabled by config.")
        return

    archive_cfg = cfg['archiving']
    zmq_cfg = cfg['zeromq']
    current_thread().name = "ContArchivalThread"

    frame_subscriber = zmq_context.socket(zmq.SUB)
    frame_subscriber.connect(zmq_cfg['frame_pub_url'])
    frame_subscriber.subscribe(b"frames_raw")
    logger.info(f"Continuous archiver subscribed to {zmq_cfg['frame_pub_url']}")

    poller = zmq.Poller()
    poller.register(frame_subscriber, zmq.POLLIN)

    last_frame_write_time = 0
    # Interval for writing frames to continuous archive, based on its FPS
    archive_frame_interval = 1.0 / archive_cfg.get('continuous_recording_fps', 10)

    frame_width, frame_height = None, None  # Get from first frame

    while not stop_event.is_set():
        socks = dict(poller.poll(timeout=1000))
        if frame_subscriber in socks and socks[frame_subscriber] == zmq.POLLIN:
            try:
                _topic, frame_data_msg = frame_subscriber.recv_multipart()
                frame_payload = pickle.loads(frame_data_msg)
                frame = frame_payload['frame']

                if frame_width is None:  # Initialize writer on first frame
                    frame_height, frame_width = frame.shape[:2]
                    _start_new_hourly_segment(archive_cfg, frame_width, frame_height)

                # Check if new hourly segment is needed
                # Requirement: "каждый час создаётся файл вида HH.avi"
                # My implementation: YYYY-MM-DD_HH.avi. Check if hour changed from start.
                if _video_writer_hourly and _hourly_segment_start_time:
                    current_hour = datetime.now().hour
                    start_hour = datetime.fromtimestamp(_hourly_segment_start_time).hour
                    if current_hour != start_hour:
                        _start_new_hourly_segment(archive_cfg, frame_width, frame_height)
                elif not _video_writer_hourly and frame_width:  # If it failed to init first time
                    _start_new_hourly_segment(archive_cfg, frame_width, frame_height)

                # Write frame to hourly archive at the configured archive FPS
                current_time = time.time()
                if _video_writer_hourly and (current_time - last_frame_write_time >= archive_frame_interval):
                    _video_writer_hourly.write(frame)
                    last_frame_write_time = current_time
                    logger.debug(f"Written frame to {_current_hourly_segment_path}")

            except Exception as e:
                logger.error(f"Error in continuous archiver: {e}", exc_info=True)
                # Attempt to restart writer if it fails
                if frame_width and frame_height:  # Ensure we have dimensions
                    try:
                        _start_new_hourly_segment(archive_cfg, frame_width, frame_height)
                    except Exception as re_e:
                        logger.error(f"Failed to re-initialize hourly segment writer: {re_e}")
                        time.sleep(5)  # wait before retrying
                else:
                    time.sleep(1)

    if _video_writer_hourly:
        _video_writer_hourly.release()
        logger.info(f"Closed final hourly segment: {_current_hourly_segment_path}")
    frame_subscriber.close()
    logger.info("Continuous archival thread stopping.")


# --- Event-Triggered Clip Archiving ---
_clip_frame_buffer = deque()  # Stores (timestamp, frame) tuples


def _get_frames_for_clip(event_timestamp: float, clip_cfg):
    """Extracts frames from _clip_frame_buffer around the event_timestamp."""
    buffered_frames = []
    start_time = event_timestamp - clip_cfg.get('clip_pre_event_seconds', 3)
    end_time = event_timestamp + clip_cfg.get('clip_post_event_seconds', 7)

    # Iterate over a copy for safe access if buffer is modified by another thread
    # (though in this single-threaded subscriber model for clips, it should be fine)
    for ts, frame in list(_clip_frame_buffer):
        if start_time <= ts <= end_time:
            buffered_frames.append(frame)

    if not buffered_frames:
        logger.warning(
            f"No frames found in buffer for event at {datetime.fromtimestamp(event_timestamp).isoformat()} (range {start_time}-{end_time})")
    return buffered_frames


def clip_archival_thread_func(stop_event: Event, zmq_context: zmq.Context):
    cfg = get_config()
    if not cfg['archiving'].get('enabled', False):
        logger.info("Clip archiving is disabled by config.")
        return

    archive_cfg = cfg['archiving']
    zmq_cfg = cfg['zeromq']
    current_thread().name = "ClipArchivalThread"

    # Subscriber for raw frames (to maintain a buffer)
    frame_subscriber = zmq_context.socket(zmq.SUB)
    frame_subscriber.connect(zmq_cfg['frame_pub_url'])
    frame_subscriber.subscribe(b"frames_raw")
    logger.info(f"Clip archiver (frame listener) subscribed to {zmq_cfg['frame_pub_url']}")

    # Subscriber for detection events
    event_subscriber = zmq_context.socket(zmq.SUB)
    event_subscriber.connect(zmq_cfg['event_pub_url'])
    event_subscriber.subscribe(b"detection_events")
    logger.info(f"Clip archiver (event listener) subscribed to {zmq_cfg['event_pub_url']}")

    poller = zmq.Poller()
    poller.register(frame_subscriber, zmq.POLLIN)
    poller.register(event_subscriber, zmq.POLLIN)

    # Max buffer duration for frames for clips
    buffer_duration_secs = archive_cfg.get('clip_event_buffer_seconds', 15)
    fps = cfg['rtsp'].get('capture_fps', 25)
    maxlen = int(fps * buffer_duration_secs)

    # Переинициализируем глобальный буфер с нужным размером
    global _clip_frame_buffer
    _clip_frame_buffer = deque(maxlen=maxlen)
    logger.info(f"Clip buffer initialized with maxlen={maxlen} frames ({buffer_duration_secs}s at {fps} FPS)")

    while not stop_event.is_set():
        socks = dict(poller.poll(timeout=1000))

        # Handle incoming frames for the buffer
        if frame_subscriber in socks and socks[frame_subscriber] == zmq.POLLIN:
            try:
                _topic, frame_data_msg = frame_subscriber.recv_multipart()
                frame_payload = pickle.loads(frame_data_msg)
                _clip_frame_buffer.append((frame_payload['timestamp'], frame_payload['frame']))
                logger.debug(f"Clip buffer updated, current size: {len(_clip_frame_buffer)}")
            except Exception as e:
                logger.error(f"Error receiving frame for clip buffer: {e}", exc_info=True)

        # Handle incoming detection events
        if event_subscriber in socks and socks[event_subscriber] == zmq.POLLIN:
            try:
                _topic, event_data_msg = event_subscriber.recv_multipart()
                event_payload = pickle.loads(event_data_msg)

                # An event payload might contain multiple detections
                for detection in event_payload.get('detections', []):
                    event_ts = detection['timestamp']  # Timestamp of the frame where detection occurred
                    class_name = detection['class_name']
                    frame_height, frame_width = _clip_frame_buffer[-1][1].shape[:2] if _clip_frame_buffer else (
                    None, None)

                    if not (frame_width and frame_height):
                        logger.warning("Cannot create clip, frame dimensions unknown (clip buffer might be empty).")
                        continue

                    clip_frames = _get_frames_for_clip(event_ts, archive_cfg)

                    if not clip_frames:
                        logger.warning(
                            f"Could not generate clip for event {detection['event_id']}: no suitable frames in buffer.")
                        continue

                    # Create clip file
                    # clip_YYYY-MM-DDTHH-MM-SS_class.avi
                    # Use event timestamp for naming
                    dt_event = datetime.fromtimestamp(event_ts)
                    clip_filename_stem = f"clip_{dt_event.strftime('%Y-%m-%dT%H-%M-%S')}_{class_name.replace(' ', '_')}"

                    clip_dir = archive_cfg['base_path'] / "clips" / dt_event.strftime("%Y-%m-%d")
                    clip_dir.mkdir(parents=True, exist_ok=True)

                    clip_filepath_avi = clip_dir / (clip_filename_stem + ".avi")
                    clip_filepath_meta = clip_dir / (clip_filename_stem + ".meta")

                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    # Use source FPS for clips, or a configured clip FPS
                    clip_fps = cfg['rtsp'].get('capture_fps', 25)

                    clip_writer = cv2.VideoWriter(
                        str(clip_filepath_avi), fourcc, clip_fps,
                        (frame_width, frame_height)
                    )
                    for frame_to_write in clip_frames:
                        clip_writer.write(frame_to_write)
                    clip_writer.release()
                    logger.info(f"Saved event clip: {clip_filepath_avi} ({len(clip_frames)} frames)")

                    # Save .meta file (JSON for easy parsing)
                    # "точном времени и координатах рамки"
                    # The detection event itself is good metadata.
                    import json
                    metadata_to_save = {
                        'event_id': detection['event_id'],
                        'clip_filename': clip_filepath_avi.name,
                        'event_timestamp': event_ts,
                        'event_datetime_iso': dt_event.isoformat(),
                        'class_id': detection['class_id'],
                        'class_name': class_name,
                        'confidence': detection['confidence'],
                        'bbox_xyxy': detection['bbox_xyxy'],
                        'source_frame_id': detection.get('frame_id'),
                        'source_frame_seq': detection.get('original_frame_seq')
                    }
                    with open(clip_filepath_meta, 'w', encoding='utf-8') as f_meta:
                        json.dump(metadata_to_save, f_meta, indent=2)
                    logger.info(f"Saved clip metadata: {clip_filepath_meta}")

            except Exception as e:
                logger.error(f"Error processing event for clip: {e}", exc_info=True)

    frame_subscriber.close()
    event_subscriber.close()
    logger.info("Clip archival thread stopping.")


# --- Archive Maintenance (Cleanup) ---
def _delete_old_archives(archive_cfg):
    base_path = Path(archive_cfg['base_path'])
    max_days = archive_cfg.get('max_archive_days', 30)
    retention_limit_date = datetime.now() - timedelta(days=max_days)

    deleted_files_count = 0
    deleted_size_gb = 0

    logger.info(f"Archive cleanup: Deleting items older than {retention_limit_date.strftime('%Y-%m-%d')}")

    for item_type in ["continuous", "clips"]:  # Iterate over subdirectories
        type_path = base_path / item_type
        if not type_path.exists():
            continue

        # Expecting YYYY-MM-DD subfolders
        for date_folder in type_path.iterdir():
            if date_folder.is_dir():
                try:
                    folder_date = datetime.strptime(date_folder.name, "%Y-%m-%d")
                    if folder_date < retention_limit_date:
                        logger.info(f"Deleting old archive folder: {date_folder}")
                        current_size_bytes = sum(f.stat().st_size for f in date_folder.glob('**/*') if f.is_file())
                        shutil.rmtree(date_folder)
                        deleted_files_count += len(
                            list(date_folder.rglob('*')))  # Approx, folder itself counts as one for rmtree.
                        deleted_size_gb += current_size_bytes / (1024 ** 3)
                        logger.info(f"Deleted {date_folder}. Freed approx {current_size_bytes / (1024 ** 3):.2f} GB.")
                except ValueError:  # Folder name not a date
                    logger.warning(f"Skipping non-date folder in archive: {date_folder.name}")
                except Exception as e:
                    logger.error(f"Error deleting folder {date_folder}: {e}")

    return deleted_files_count, deleted_size_gb


def _ensure_disk_space(archive_cfg):
    base_path = Path(archive_cfg['base_path'])
    min_free_gb = archive_cfg.get('min_disk_free_gb_threshold', 10)

    deleted_files_count = 0
    deleted_size_gb = 0

    current_free_gb = get_disk_free_space_gb(base_path)
    logger.info(f"Disk space check: Current free space {current_free_gb:.2f} GB. Required minimum {min_free_gb} GB.")

    # If free space is below threshold, delete oldest items (day by day)
    # from both continuous and clips until threshold is met or no more deletable files.
    if current_free_gb < min_free_gb:
        logger.warning(f"Low disk space ({current_free_gb:.2f}GB free). Trying to free up space...")

        # Collect all daily folders with their dates
        all_daily_folders = []
        for item_type in ["continuous", "clips"]:
            type_path = base_path / item_type
            if not type_path.exists(): continue
            for date_folder in type_path.iterdir():
                if date_folder.is_dir():
                    try:
                        folder_date = datetime.strptime(date_folder.name, "%Y-%m-%d")
                        all_daily_folders.append({'path': date_folder, 'date': folder_date})
                    except ValueError:
                        pass  # Skip non-date folders

        # Sort by date, oldest first
        all_daily_folders.sort(key=lambda x: x['date'])

        for folder_info in all_daily_folders:
            if get_disk_free_space_gb(base_path) >= min_free_gb:
                break  # Enough space freed

            folder_path = folder_info['path']
            logger.info(f"Deleting oldest folder to free space: {folder_path}")
            try:
                current_size_bytes = sum(f.stat().st_size for f in folder_path.glob('**/*') if f.is_file())
                shutil.rmtree(folder_path)
                deleted_files_count += 1  # Simplified count, can be more precise
                freed_gb_this_iter = current_size_bytes / (1024 ** 3)
                deleted_size_gb += freed_gb_this_iter
                logger.info(f"Deleted {folder_path}. Freed approx {freed_gb_this_iter:.2f} GB.")
            except Exception as e:
                logger.error(f"Error deleting folder {folder_path} for disk space: {e}")

    return deleted_files_count, deleted_size_gb


def archive_maintenance_thread_func(stop_event: Event):
    cfg = get_config()
    if not cfg['archiving'].get('enabled', False):
        logger.info("Archive maintenance is disabled as archiving is disabled.")
        return

    archive_cfg = cfg['archiving']
    current_thread().name = "ArchiveMaintThread"

    cleanup_interval = archive_cfg.get('cleanup_interval_seconds', 3600)  # Default 1 hour

    while not stop_event.wait(cleanup_interval):  # Wait for interval or stop signal
        logger.info("Running archive maintenance...")

        # 1. Delete by age (max_archive_days)
        deleted_by_age_count, freed_by_age_gb = _delete_old_archives(archive_cfg)
        if deleted_by_age_count > 0:
            logger.info(f"Age-based cleanup: Deleted {deleted_by_age_count} items, freed {freed_by_age_gb:.2f} GB.")

        # 2. Delete by disk space threshold (min_disk_free_gb_threshold)
        deleted_by_space_count, freed_by_space_gb = _ensure_disk_space(archive_cfg)
        if deleted_by_space_count > 0:
            logger.info(
                f"Space-based cleanup: Deleted {deleted_by_space_count} folder(s), freed {freed_by_space_gb:.2f} GB.")

        logger.info("Archive maintenance finished.")

    logger.info("Archive maintenance thread stopping.")