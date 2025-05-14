# src/smart_detector/cli.py
import argparse
import sys
import logging
import time  # Keep for time.sleep if needed, or remove if not used
from threading import Thread, Event, current_thread as get_current_thread
import signal
from pathlib import Path

from .utils import load_config, setup_logging, get_config
from .person_detector_core import run_person_detection_loop
from .archive_manager import archive_maintenance_thread_func  # Import from new file
from . import APP_NAME, __version__

logger = logging.getLogger(APP_NAME)
stop_event_global = Event()


def signal_handler(signum, frame):
    signal_name = signal.Signals(signum).name
    logger.warning(f"Signal {signal_name} ({signum}) received. Initiating shutdown...")
    if not stop_event_global.is_set():
        stop_event_global.set()


def main(argv=None):
    get_current_thread().name = "MainCLIThread"
    p = argparse.ArgumentParser(
        description=f"{APP_NAME} - Person Detection & Recording System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--config", default="config.yml", help="Path to YAML configuration file."
    )
    args = p.parse_args(argv)

    config_data = None
    try:
        config_data = load_config(args.config)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)  # Basic log for early fail
        logger.fatal(f"Failed to load or parse configuration '{args.config}': {e}", exc_info=True)
        sys.exit(1)

    setup_logging()

    logger.info(f"Starting {APP_NAME} v{__version__}...")
    logger.info(f"Using configuration: {Path(args.config).resolve()}")
    logger.info(f"RTSP URL: {config_data.get('rtsp', {}).get('url')}")
    logger.info(
        f"Model: {config_data.get('inference', {}).get('model_path')}, Device: {config_data.get('inference', {}).get('device')}")
    logger.info(f"Recording to: {config_data.get('person_event_recording', {}).get('output_directory')}")
    logger.info(f"Confidence threshold for person: {config_data.get('inference', {}).get('confidence_threshold')}")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    maintenance_th = None
    if config_data.get('archive_maintenance', {}).get('enabled', False):
        maintenance_th = Thread(
            target=archive_maintenance_thread_func,
            args=(stop_event_global, config_data),
            name="ArchiveMaintThread",  # Name set inside the function now
            daemon=True
        )
        maintenance_th.start()
        # logger is now inside archive_maintenance_thread_func

    try:
        run_person_detection_loop(stop_event_global, config_data)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt in main execution. Signaling shutdown.")
        if not stop_event_global.is_set(): stop_event_global.set()
    except Exception as e:
        logger.critical(f"Unhandled exception in main CLI execution: {e}", exc_info=True)
        if not stop_event_global.is_set(): stop_event_global.set()
    finally:
        logger.info("Main CLI execution loop finished or interrupted. Finalizing shutdown...")

        if maintenance_th and maintenance_th.is_alive():
            logger.info("Waiting for archive maintenance thread to complete...")
            # stop_event_global is already set, maintenance_th should see it on its wait()
            maintenance_th.join(timeout=12.0)  # Wait a bit longer for it
            if maintenance_th.is_alive():
                logger.warning("Archive maintenance thread did not stop in time.")

        logger.info(f"{APP_NAME} has shut down.")
        logging.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()