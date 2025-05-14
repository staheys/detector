# src/smart_detector/cli.py
import argparse
import sys
import logging
import time
from datetime import datetime
from pathlib import Path
from threading import Thread, Event, current_thread
import signal  # For graceful shutdown

import yaml
import zmq
import cv2  # For display window
import pickle

from .utils import load_config, setup_logging, get_config
from .core import capture_frames_thread_func, process_frames_thread_func, draw_detections_on_frame
from .archiver import (
    continuous_archival_thread_func,
    clip_archival_thread_func,
    archive_maintenance_thread_func
)
from . import config as app_config  # For CV2 consts if needed

logger = logging.getLogger(app_config.APP_NAME)  # Use app name for logger

# --- Global stop event for all threads ---
stop_event_global = Event()


def signal_handler(signum, frame):
    logger.warning(f"Signal {signal.Signals(signum).name} received. Initiating shutdown...")
    stop_event_global.set()


def display_thread_func(stop_event: Event, zmq_context: zmq.Context):
    """
    Placeholder for GUI. Subscribes to frames and events, shows in OpenCV window.
    """
    cfg = get_config()
    if not cfg['display'].get('show_window', False):
        logger.info("Display window is disabled by config.")
        return

    zmq_cfg = cfg['zeromq']
    display_cfg = cfg['display']
    current_thread().name = "DisplayThread"

    # frame_subscriber = zmq_context.socket(zmq.SUB)
    # frame_subscriber.connect(zmq_cfg['frame_pub_url'])
    # frame_subscriber.subscribe(b"frames_raw")  # Get raw frames
    #
    # event_subscriber = zmq_context.socket(zmq.SUB)
    # event_subscriber.connect(zmq_cfg['event_pub_url'])
    # event_subscriber.subscribe(b"detection_events")  # Get detection events
    #
    # poller = zmq.Poller()
    # poller.register(frame_subscriber, zmq.POLLIN)
    # poller.register(event_subscriber, zmq.POLLIN)

    annotated_sub = zmq_context.socket(zmq.SUB)
    annotated_sub.connect(zmq_cfg['annotated_pub_url'])
    annotated_sub.subscribe(b"frames_annotated")

    poller = zmq.Poller()
    poller.register(annotated_sub, zmq.POLLIN)

    window_name = display_cfg.get('window_name', "Live Feed")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # Keep aspect ratio
    # cv2.resizeWindow(window_name, 960, 540) # Example resize

    last_detections_on_frame = {}  # Store detections per frame_id briefly
    last_frame_payload = None

    # For FPS calculation on display
    fps_display_interval = display_cfg.get('fps_display_interval', 1.0)
    frame_count_for_fps = 0
    last_fps_calc_time = time.time()
    display_fps = 0.0

    # For green border flash on new event
    new_event_flash_until = 0
    flash_duration = 0.5  # seconds

    # For event notification text
    last_event_notification = None  # (text, display_until_time)

    logger.info("Display thread started. Press 'q' in window to attempt shutdown.")

    while not stop_event.is_set():
        current_time_for_display = time.time()
        display_frame_this_loop = False

        # Process incoming messages
        socks = dict(poller.poll(timeout=10))  # Short timeout for responsiveness

        # if frame_subscriber in socks and socks[frame_subscriber] == zmq.POLLIN:
        #     try:
        #         _topic, frame_data_msg = frame_subscriber.recv_multipart(flags=zmq.NOBLOCK)  # Non-blocking
        #         last_frame_payload = pickle.loads(frame_data_msg)
        #         display_frame_this_loop = True
        #         frame_count_for_fps += 1
        #     except zmq.Again:  # No message
        #         pass
        #     except Exception as e:
        #         logger.error(f"Display: Error receiving frame: {e}", exc_info=True)
        #         last_frame_payload = None  # Invalidate
        #
        # if event_subscriber in socks and socks[event_subscriber] == zmq.POLLIN:
        #     try:
        #         _topic, event_data_msg = event_subscriber.recv_multipart(flags=zmq.NOBLOCK)
        #         event_payload = pickle.loads(event_data_msg)
        #         frame_id = event_payload['frame_id']
        #         last_detections_on_frame[frame_id] = event_payload.get('detections', [])
        #
        #         # Flash border and show notification for new events
        #         if event_payload.get('detections'):
        #             new_event_flash_until = current_time_for_display + flash_duration
        #             # Show first detection as notification
        #             first_det = event_payload['detections'][0]
        #             notif_text = f"[{datetime.fromtimestamp(first_det['timestamp']).strftime('%H:%M:%S')}] {first_det['class_name'].upper()} conf={first_det['confidence']:.2f}"
        #             last_event_notification = (notif_text, current_time_for_display + 2.0)  # Show for 2s
        #
        #         display_frame_this_loop = True  # Force redraw if event arrived
        #     except zmq.Again:
        #         pass
        #     except Exception as e:
        #         logger.error(f"Display: Error receiving event: {e}", exc_info=True)
        if annotated_sub in socks and socks[annotated_sub] == zmq.POLLIN:
            topic, msg = annotated_sub.recv_multipart()
            payload = pickle.loads(msg)
            vis_frame = payload['frame']
            # рисуем FPS, уведомления и т.п. прямо на vis_frame
            cv2.imshow(window_name, vis_frame)
        # Drawing logic
        if display_frame_this_loop and last_frame_payload:
            frame = last_frame_payload['frame']
            frame_id = last_frame_payload['id']

            # Get detections for this specific frame
            detections_to_draw = last_detections_on_frame.get(frame_id, [])
            vis_frame = draw_detections_on_frame(frame, detections_to_draw, cfg['inference']['classes_map'])

            # Clean up old detections from memory
            # A more robust way would be a timed cache or deque for last_detections_on_frame
            if len(last_detections_on_frame) > 50:  # Arbitrary limit
                oldest_key = next(iter(last_detections_on_frame))
                del last_detections_on_frame[oldest_key]

            # Calculate and display FPS
            if current_time_for_display - last_fps_calc_time >= fps_display_interval:
                display_fps = frame_count_for_fps / (current_time_for_display - last_fps_calc_time)
                frame_count_for_fps = 0
                last_fps_calc_time = current_time_for_display

            cv2.putText(vis_frame, f"{display_fps:.1f} FPS", app_config.CV2_FPS_POSITION, app_config.CV2_FONT,
                        app_config.CV2_FONT_SCALE_LARGE, app_config.CV2_COLOR_FPS, app_config.CV2_FONT_THICKNESS,
                        app_config.CV2_LINE_TYPE)

            # Green border flash
            if current_time_for_display < new_event_flash_until:
                cv2.rectangle(vis_frame, (0, 0), (vis_frame.shape[1] - 1, vis_frame.shape[0] - 1), (0, 255, 0),
                              3)  # Green, 3px thick

            # Event notification text
            if last_event_notification and current_time_for_display < last_event_notification[1]:
                notif_text = last_event_notification[0]
                (w, h), baseline = cv2.getTextSize(notif_text, app_config.CV2_FONT, app_config.CV2_FONT_SCALE_SMALL,
                                                   app_config.CV2_FONT_THICKNESS)
                # Bottom right corner
                rx, ry = vis_frame.shape[1] - w - 20, vis_frame.shape[0] - h - 20
                # Simple background rectangle for text
                cv2.rectangle(vis_frame, (rx - 5, ry - h - baseline - 2), (rx + w + 5, ry + baseline + 2),
                              app_config.CV2_EVENT_NOTIFICATION_COLOR_BG, -1)
                cv2.putText(vis_frame, notif_text, (rx, ry), app_config.CV2_FONT, app_config.CV2_FONT_SCALE_SMALL,
                            app_config.CV2_COLOR_WHITE, app_config.CV2_FONT_THICKNESS, app_config.CV2_LINE_TYPE)
            elif last_event_notification and current_time_for_display >= last_event_notification[1]:
                last_event_notification = None  # Clear expired notification

            try:
                cv2.imshow(window_name, vis_frame)
            except cv2.error as e:  # Handle cases where window might have been closed externally
                if "NULL window" in str(e) or "Invalid window handle" in str(e):
                    logger.warning("OpenCV window seems to have been closed. Stopping display.")
                    stop_event.set()  # Signal main to stop
                    break
                else:
                    raise e
        socks = dict(poller.poll(timeout=1000))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("'q' pressed in display window. Signaling shutdown.")
            stop_event.set()  # Signal global stop
            break
        elif key != 255:  # 255 is no key pressed
            # Placeholder for other hotkeys (L/E/C/S/T from requirements - GUI related)
            logger.debug(f"Key pressed: {chr(key) if 32 <= key <= 126 else key}")

    logger.info("Display thread stopping.")
    if cfg['display'].get('show_window', False):
        cv2.destroyAllWindows()
    # frame_subscriber.close()
    # event_subscriber.close()


def main(argv=None):
    p = argparse.ArgumentParser(description="Smart Detector System")
    p.add_argument(
        "--config", default="config.yml", help="Path to YAML configuration file (default: config.yml)"
    )
    args = p.parse_args(argv)

    try:
        config_data = load_config(args.config)
    except FileNotFoundError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL: Error parsing configuration file {args.config}: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup logging after config is loaded
    setup_logging()

    logger.info(f"Starting {app_config.APP_NAME}...")
    # Log some key config settings for verification
    logger.info(f"RTSP URL: {config_data.get('rtsp', {}).get('url')}")
    logger.info(f"Model Path: {config_data.get('inference', {}).get('model_path')}")
    logger.info(f"Archive Path: {config_data.get('archiving', {}).get('base_path')}")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill

    # Initialize ZeroMQ context
    # For inproc, context doesn't strictly need to be shared, but good practice.
    # For inter-process (tcp), one context per process.
    zmq_io_threads = config_data.get('zeromq', {}).get('context_io_threads', 1)
    zmq_ctx = zmq.Context(io_threads=zmq_io_threads)

    threads = []

    # 1. Capture Thread
    capture_thread = Thread(target=capture_frames_thread_func, args=(stop_event_global, zmq_ctx), daemon=True)
    threads.append(capture_thread)

    # 2. Processing Thread
    # Ensure model path is valid before starting thread, or handle error inside
    model_p = Path(config_data['inference']['model_path'])
    if not model_p.exists():
        logger.error(f"FATAL: Model file not found at {model_p}. Exiting.")
        # zmq_ctx.term() # Terminate context if exiting early
        sys.exit(1)

    processing_thread = Thread(target=process_frames_thread_func, args=(stop_event_global, zmq_ctx), daemon=True)
    threads.append(processing_thread)

    # 3. Archiving Threads (if enabled)
    if config_data['archiving'].get('enabled', False):
        continuous_arch_thread = Thread(target=continuous_archival_thread_func, args=(stop_event_global, zmq_ctx),
                                        daemon=True)
        threads.append(continuous_arch_thread)
        #
        # clip_arch_thread = Thread(target=clip_archival_thread_func, args=(stop_event_global, zmq_ctx), daemon=True)
        # threads.append(clip_arch_thread)

        maintenance_thread = Thread(target=archive_maintenance_thread_func, args=(stop_event_global,), daemon=True)
        threads.append(maintenance_thread)
    else:
        logger.info("Archiving is disabled in the configuration.")

    # 4. Display Thread (placeholder for GUI's live view)
    if config_data['display'].get('show_window', False):
        display_thread = Thread(target=display_thread_func, args=(stop_event_global, zmq_ctx),
                                daemon=True)  # Daemon=True might be an issue if cv2 window needs main thread focus on some OS.
        # For OpenCV GUI, it's often better to run it in the main thread or ensure it's not daemon if it manages its own lifecycle.
        # However, for a clean shutdown with Ctrl+C, daemon threads are easier.
        # Let's make it non-daemon and join it explicitly if it's the primary interaction loop.
        # For this setup, let's keep it daemon and rely on stop_event.
        threads.append(display_thread)
    else:
        logger.info("Display window is disabled. Running headless.")

    # Start all threads
    for t in threads:
        logger.info(f"Starting thread: {t.name if hasattr(t, 'name') else type(t).__name__}")
        t.start()

    # Keep main thread alive until stop_event is set
    # This loop also allows main thread to do other work or handle signals if not daemon.
    try:
        while not stop_event_global.is_set():
            # Check if any critical threads have died unexpectedly (optional)
            for t in threads:
                if not t.is_alive() and t.daemon is False:  # Only worry about non-daemon threads here
                    logger.error(f"Non-daemon thread {t.name} has died unexpectedly! Shutting down.")
                    stop_event_global.set()
                    break
            time.sleep(0.5)  # Main loop check interval
    except KeyboardInterrupt:  # Redundant if signal handler works, but good fallback
        logger.info("KeyboardInterrupt in main loop. Shutting down.")
        stop_event_global.set()

    logger.info("Shutdown initiated. Waiting for threads to complete...")
    # Wait for threads to finish (with a timeout)
    # Daemon threads will exit when main exits, but joining is cleaner.
    for t in threads:
        logger.debug(f"Joining thread {t.name if hasattr(t, 'name') else type(t).__name__}...")
        t.join(timeout=5.0)  # Add timeout to prevent hanging
        if t.is_alive():
            logger.warning(f"Thread {t.name if hasattr(t, 'name') else type(t).__name__} did not terminate in time.")

    logger.info("Terminating ZeroMQ context...")
    zmq_ctx.term()  # Gracefully terminate the ZeroMQ context

    logger.info(f"{app_config.APP_NAME} has shut down.")
    sys.exit(0)


if __name__ == "__main__":
    main()