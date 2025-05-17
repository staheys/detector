# video_recorder.py
import cv2
import time
import datetime
import os
import configparser
import threading
# Removed: import db_api (db_api module will be passed in __init__)
import logging

logger = logging.getLogger(__name__)


class VideoRecorder:
    def __init__(self, cfg: configparser.ConfigParser, db_api_module):  # db_api_module injected
        if 'RECORDING' not in cfg:
            raise ValueError("RECORDING section not found in configuration.")

        rec_cfg = cfg['RECORDING']
        self.rtsp_url = rec_cfg.get('rtsp_url')
        if not self.rtsp_url:
            raise ValueError("rtsp_url not specified in [RECORDING] section.")

        self.output_path = rec_cfg.get('output_path', './recordings/')
        self.duration_seconds = rec_cfg.getint('duration_seconds', 300)  # Default to 300s
        self.codec_str = rec_cfg.get('codec', 'XVID')
        self.config_fps = rec_cfg.getfloat('fps', 0)  # 0 means try to use stream FPS

        self.db_api = db_api_module  # Store injected db_api module

        self.is_recording_active = False
        self.stop_requested = False
        self.recording_thread = None

        if not os.path.exists(self.output_path):
            try:
                os.makedirs(self.output_path, exist_ok=True)
                logger.info(f"Created recording output directory: {self.output_path}")
            except OSError as e:
                logger.error(f"Error creating directory {self.output_path}: {e}")
                raise

    def _get_video_writer_details(self, cap) -> tuple[cv2.VideoWriter | None, str, float, datetime.datetime]:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_width == 0 or frame_height == 0:
            logger.error(
                f"Could not get frame dimensions from recording stream: {self.rtsp_url}. W: {frame_width}, H: {frame_height}")
            return None, "", 0.0, datetime.datetime.min  # Return a min datetime

        stream_fps = cap.get(cv2.CAP_PROP_FPS)
        fps_to_use = self.config_fps
        if fps_to_use <= 0:  # Config FPS not set or invalid
            if stream_fps > 0:
                fps_to_use = stream_fps
                logger.info(f"Using stream FPS for recording: {fps_to_use}")
            else:
                fps_to_use = 20.0  # Default fallback
                logger.warning(
                    f"Stream FPS is {stream_fps} and config FPS is {self.config_fps}. Using default FPS: {fps_to_use}")
        else:
            logger.info(f"Using configured FPS for recording: {fps_to_use}")

        # Generate filename using UTC datetime for start of segment
        segment_start_utc = datetime.datetime.utcnow()
        # Format: YYYY-MM-DD_HH-MM-SS.avi
        filename_dt_part = segment_start_utc.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{filename_dt_part}.avi"
        filepath = os.path.join(self.output_path, filename)

        try:
            fourcc = cv2.VideoWriter_fourcc(*self.codec_str)
            out = cv2.VideoWriter(filepath, fourcc, fps_to_use, (frame_width, frame_height))
            if not out.isOpened():
                logger.error(f"Could not open VideoWriter for {filepath}. Check codec, permissions, and disk space.")
                return None, filepath, fps_to_use, segment_start_utc
            return out, filepath, fps_to_use, segment_start_utc
        except Exception as e:
            logger.error(f"Error initializing VideoWriter for {filepath}: {e}")
            return None, filepath, fps_to_use, segment_start_utc

    def _continuous_record_loop(self):
        cap = None
        self.is_recording_active = True

        while not self.stop_requested:
            if cap is None or not cap.isOpened():
                logger.info(f"Attempting to open recording RTSP stream: {self.rtsp_url}")
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)  # Or cv2.CAP_GSTREAMER for some setups
                if not cap.isOpened():
                    logger.error(f"Failed to open recording RTSP stream. Retrying in 5 seconds...")
                    for _ in range(50):  # 5 seconds with 0.1s sleep
                        if self.stop_requested: break
                        time.sleep(0.1)
                    if self.stop_requested: break
                    continue  # Retry opening stream
                logger.info("Recording stream opened successfully.")

            out, filepath, fps_to_use, segment_start_utc = self._get_video_writer_details(cap)
            if out is None:
                logger.error("Failed to create video writer. Will attempt to re-establish stream and writer.")
                if cap: cap.release()  # Release faulty cap
                cap = None  # Force re-opening
                time.sleep(5)  # Wait before full retry
                continue

            clip_db_id = self.db_api.add_video_clip_start(os.path.basename(filepath), segment_start_utc)
            if clip_db_id is None:
                logger.error(f"Failed to log video clip {filepath} to DB. Skipping this segment recording.")
                out.release()  # Release writer for this failed segment
                # Keep cap open to try next segment, or it will be reopened if it also failed
                time.sleep(1)
                continue

            logger.info(
                f"Recording segment: {filepath}, Duration: {self.duration_seconds}s, FPS: {fps_to_use}, DB ID: {clip_db_id}")
            segment_start_wall_time = time.monotonic()  # For duration control
            frames_written_this_segment = 0

            # Inner loop for the current segment
            while not self.stop_requested and (time.monotonic() - segment_start_wall_time) < self.duration_seconds:
                if not cap.isOpened():
                    logger.warning("Recording stream closed unexpectedly during segment. Ending segment.")
                    break

                ret, frame = cap.read()
                if ret and frame is not None:
                    out.write(frame)
                    frames_written_this_segment += 1
                elif not ret:  # Frame grab failed
                    logger.warning(
                        "Failed to grab frame for recording. Stream might have issues. Trying to continue segment.")
                    # If stream is consistently failing, it might close itself, or we might need more robust error handling here.
                    time.sleep(0.1)  # Small pause before next attempt
                    # Check again if cap is opened; if read fails multiple times, cap.isOpened() might turn false.
                    if not cap.isOpened(): break

                    # Yield control slightly, helps responsiveness to stop_requested.
                # Approximate wait to manage CPU, not for precise frame timing.
                sleep_duration = (1.0 / (fps_to_use * 1.5)) if fps_to_use > 0 else 0.01
                time.sleep(max(0.001, sleep_duration))

            segment_end_utc = datetime.datetime.utcnow()
            elapsed_wall_time = time.monotonic() - segment_start_wall_time
            logger.info(
                f"Segment {filepath} finished. Wrote {frames_written_this_segment} frames in {elapsed_wall_time:.2f}s.")

            if out: out.release()
            self.db_api.update_video_clip_end(clip_db_id, segment_end_utc)

            if self.stop_requested:
                logger.info("Stop requested, exiting recording loop.")
                break

        # Cleanup when the main loop exits
        if cap and cap.isOpened():
            cap.release()
            logger.info("Released recording video capture.")
        self.is_recording_active = False
        logger.info("Continuous recording loop has fully ended.")

    def start_recording_session(self):
        if self.is_recording_active:
            logger.warning("Recording session is already active.")
            return

        self.stop_requested = False
        # self.is_recording_active = True # Set inside the thread's loop start
        self.recording_thread = threading.Thread(target=self._continuous_record_loop, daemon=True)
        self.recording_thread.name = "VideoRecorderThread"
        self.recording_thread.start()
        logger.info(f"Continuous video recording session initiated for {self.rtsp_url}. Output: {self.output_path}")

    def stop_recording_session(self):
        if not self.is_recording_active and (self.recording_thread is None or not self.recording_thread.is_alive()):
            logger.info("No active recording session to stop.")
            return

        logger.info("Attempting to stop recording session...")
        self.stop_requested = True

        if self.recording_thread and self.recording_thread.is_alive():
            # Generous timeout: current segment duration + buffer for closing cap/writer
            timeout_seconds = self.duration_seconds + 20
            logger.info(f"Waiting for recording thread to join (timeout: {timeout_seconds}s)...")
            self.recording_thread.join(timeout=timeout_seconds)
            if self.recording_thread.is_alive():
                logger.warning("Recording thread did not terminate gracefully within timeout.")
            else:
                logger.info("Recording thread terminated successfully.")

        self.is_recording_active = False  # Ensure it's false after stopping attempts
        self.recording_thread = None
        logger.info("Recording session stop process completed.")