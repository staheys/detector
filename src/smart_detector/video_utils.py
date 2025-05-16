# src/smart_detector/video_utils.py
import cv2
import logging
from pathlib import Path
from datetime import datetime  # Not directly used here now, but good for utils
import numpy as np

logger = logging.getLogger(__name__)


def create_video_capture(source: str | int, width: int, height: int, fps: int):
    cap = None
    gst_pipeline_str = None
    if isinstance(source, str) and source.lower().startswith("rtsp://"):
        gst_elements = [
            f"rtspsrc location={source} latency=200 drop-on-latency=true ! ",
            "queue ! rtph264depay ! h264parse ! avdec_h264",  # Assumes H.264
            "! videoconvert ! video/x-raw,format=BGR",
            "! appsink name=sink sync=false max-buffers=3 drop=true emit-signals=true"
        ]
        # Optional: Add capsfilter for GStreamer to handle resizing/framerate target
        # if width > 0 and height > 0 and fps > 0:
        #    gst_elements.insert(-1, f"! capsfilter caps=video/x-raw,width={width},height={height},framerate={fps}/1")
        gst_pipeline_str = " ".join(gst_elements)
        logger.info(f"Attempting GStreamer pipeline for {source}")
        logger.debug(f"GStreamer: {gst_pipeline_str}")
        cap = cv2.VideoCapture(gst_pipeline_str, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.warning("GStreamer RTSP failed. Falling back to OpenCV default (FFmpeg).")
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap or not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return None

    if width > 0: cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0: cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        f"Video source '{source}' opened. Requested: {width}x{height}@{fps}FPS. Actual: {actual_w}x{actual_h}@{actual_fps:.2f}FPS.")
    if actual_w == 0 or actual_h == 0:
        logger.error("Video source opened but frame dimensions are zero.");
        cap.release();
        return None
    return cap


def save_video_segment(frames_to_save: list[np.ndarray], filepath_base: Path, fps: float, frame_size: tuple[int, int]):
    """Saves a list of frames to an AVI video file."""
    if not frames_to_save:
        logger.warning(f"No frames to save for {filepath_base}.avi. Skipping video creation.")
        return False

    # Ensure filepath ends with .avi
    filepath = filepath_base.with_suffix('.avi')

    logger.info(
        f"Saving AVI segment: {filepath} ({len(frames_to_save)} frames, {fps} FPS, {frame_size[0]}x{frame_size[1]})")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID for AVI
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, frame_size)

    if not writer.isOpened():
        logger.error(f"Failed to open VideoWriter for {filepath} with XVID codec.")
        return False

    try:
        for frame in frames_to_save:
            if frame is not None and frame.shape[1] == frame_size[0] and frame.shape[0] == frame_size[1]:
                writer.write(frame)
            else:
                logger.warning(
                    f"Frame skipped for {filepath} due to None or mismatched dimensions. Expected {frame_size}, got {frame.shape if frame is not None else 'None'}")
        writer.release()
        logger.info(f"Successfully saved AVI segment: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error writing AVI segment {filepath}: {e}", exc_info=True)
        if writer.isOpened(): writer.release()
        return False


def draw_detections_on_frame_for_clip(frame_orig: np.ndarray, detections: list, person_class_id: int, color_map_func):
    """Draws detections on a frame. This is intended for frames being saved to clips."""
    if not detections:  # If no detections, return original frame
        return frame_orig.copy()  # Return a copy to avoid modifying the original if it's from a buffer

    vis_frame = frame_orig.copy()  # Work on a copy
    for det in detections:
        if det['class_id'] == person_class_id:
            x1, y1, x2, y2 = map(int, det['bbox_xyxy'])
            conf = det['confidence']
            label = f"Person {conf:.2f}"
            color = color_map_func("person")

            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x1, y1 - h - baseline - 3), (x1 + w, y1 - baseline + 3), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis_frame