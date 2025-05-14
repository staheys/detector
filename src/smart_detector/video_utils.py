# src/smart_detector/video_utils.py
import threading

import cv2
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import time
import numpy as np

logger = logging.getLogger(__name__)

def create_video_capture(source: str | int, width: int, height: int, fps: int):
    """Creates and configures a cv2.VideoCapture object."""
    cap = None
    gst_pipeline_str = None

    if isinstance(source, str) and source.lower().startswith("rtsp://"):
        # Basic GStreamer pipeline for RTSP (H.264 assumed)
        # More robust pipelines might be needed depending on the camera/network
        gst_elements = [
            f"rtspsrc location={source} latency=200 drop-on-latency=true ! ",
            "queue ! rtph264depay ! h264parse ! avdec_h264",
            "! videoconvert ! video/x-raw,format=BGR",
        ]
        # Optional: Add capsfilter for desired resolution/fps if GStreamer should handle resizing/framerate
        # if width > 0 and height > 0 and fps > 0:
        #    gst_elements.append(f"! capsfilter caps=video/x-raw,width={width},height={height},framerate={fps}/1")
        gst_elements.append("! appsink name=sink sync=false max-buffers=3 drop=true emit-signals=true")
        gst_pipeline_str = " ".join(gst_elements)

        logger.info(f"Attempting GStreamer pipeline for {source}")
        logger.debug(f"GStreamer: {gst_pipeline_str}")
        cap = cv2.VideoCapture(gst_pipeline_str, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            logger.warning("GStreamer RTSP failed. Falling back to OpenCV default (FFmpeg).")
            cap = cv2.VideoCapture(source) # Fallback
    else:
        cap = cv2.VideoCapture(source) # For webcam (int) or local file (str)

    if not cap or not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return None

    # Set properties (best effort, may not always work, esp. with GStreamer post-creation)
    if width > 0: cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0: cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3) # Keep OpenCV internal buffer small

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) # Often unreliable for RTSP
    logger.info(f"Video source '{source}' opened. Requested: {width}x{height}@{fps}FPS. "
                f"Actual: {actual_w}x{actual_h}@{actual_fps:.2f}FPS (actual FPS may vary).")
    if actual_w == 0 or actual_h == 0:
        logger.error("Video source opened but frame dimensions are zero. Check stream/camera.")
        cap.release()
        return None
    return cap

def save_video_segment(frames_to_save: list[np.ndarray], filepath: Path, fps: float, frame_size: tuple[int, int]):
    """Saves a list of frames to a video file."""
    if not frames_to_save:
        logger.warning(f"No frames to save for {filepath}. Skipping video creation.")
        return False

    logger.info(f"Saving video segment: {filepath} ({len(frames_to_save)} frames, {fps} FPS, {frame_size[0]}x{frame_size[1]})")
    # Common codecs: 'XVID' (AVI), 'mp4v' (MP4 - ensure FFMPEG backend for OpenCV)
    # For MP4, filepath should end with .mp4
    # For AVI, filepath should end with .avi
    fourcc_str = 'XVID' if str(filepath).lower().endswith('.avi') else 'mp4v'
    if fourcc_str == 'mp4v' and not str(filepath).lower().endswith('.mp4'):
        logger.warning(f"Using 'mp4v' codec but filepath '{filepath}' does not end with .mp4. Output might be unexpected.")
        # Consider forcing .mp4 extension or using AVI as default. For simplicity, let's assume user names it well.

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, frame_size)

    if not writer.isOpened():
        logger.error(f"Failed to open VideoWriter for {filepath} with codec {fourcc_str}.")
        return False

    try:
        for frame in frames_to_save:
            if frame is not None and frame.shape[1] == frame_size[0] and frame.shape[0] == frame_size[1]:
                 writer.write(frame)
            else:
                logger.warning(f"Frame skipped for {filepath} due to None or mismatched dimensions. "
                               f"Expected {frame_size}, got {frame.shape if frame is not None else 'None'}")
        writer.release()
        logger.info(f"Successfully saved video segment: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error writing video segment {filepath}: {e}", exc_info=True)
        if writer.isOpened():
            writer.release() # Ensure writer is released on error
        return False

def draw_detections(frame: np.ndarray, detections: list, person_class_id: int, color_map_func):
    """Draws bounding boxes and labels for detections on a frame."""
    vis_frame = frame.copy()
    for det in detections:
        # Assuming det is a dict like {'class_id': id, 'class_name': name, 'confidence': conf, 'bbox_xyxy': [x1,y1,x2,y2]}
        # This structure will come from our _perform_inference adaptation
        if det['class_id'] == person_class_id: # Only draw persons, or adapt if more classes
            x1, y1, x2, y2 = map(int, det['bbox_xyxy'])
            conf = det['confidence']
            label = f"Person {conf:.2f}"
            color = color_map_func("person") # from utils

            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            # Text background
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x1, y1 - h - baseline - 3), (x1 + w, y1 - baseline +3), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return vis_frame

def archive_maintenance_thread_func(stop_event: threading.Event, cfg: dict):
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