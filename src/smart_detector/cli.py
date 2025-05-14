# src/smart_detector/cli.py
import argparse
import sys
import logging
import time
from threading import Thread, Event
import signal
from pathlib import Path

from .utils import load_config, setup_logging, get_config
# Import the main run function and maintenance thread function
from .core import run_person_detection_loop, archive_maintenance_thread_func
from . import APP_NAME, __version__ # From __init__.py

logger = logging.getLogger(APP_NAME)
stop_event_global = Event()

def signal_handler(signum, frame):
    signal_name = signal.Signals(signum).name
    logger.warning(f"Signal {signal_name} ({signum}) received. Initiating shutdown...")
    if not stop_event_global.is_set():
        stop_event_global.set()

def main(argv=None):
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
    except Exception as e: # Catch broader exceptions during initial load
        logging.basicConfig(level=logging.ERROR)
        logger.fatal(f"Failed to load or parse configuration '{args.config}': {e}", exc_info=True)
        sys.exit(1)

    setup_logging() # Setup logging based on loaded config

    logger.info(f"Starting {APP_NAME} v{__version__}...")
    logger.info(f"Using configuration: {Path(args.config).resolve()}")
    # Log some key settings
    logger.info(f"RTSP URL: {config_data.get('rtsp',{}).get('url')}")
    logger.info(f"Model: {config_data.get('inference',{}).get('model_path')}, Device: {config_data.get('inference',{}).get('device')}")
    logger.info(f"Recording to: {config_data.get('person_event_recording',{}).get('output_directory')}")


    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # --- Thread for Archive Maintenance ---
    maintenance_th = None
    if config_data.get('archive_maintenance', {}).get('enabled', False):
        maintenance_th = Thread(
            target=archive_maintenance_thread_func,
            args=(stop_event_global, config_data), # Pass full config
            name="ArchiveMaintThread",
            daemon=True
        )
        maintenance_th.start()
        logger.info("Archive maintenance thread started.")

    # --- Run main detection loop in the main thread ---
    # This simplifies state management for recording and display.
    # The capture itself is in a thread started by run_person_detection_loop.
    try:
        run_person_detection_loop(stop_event_global, config_data)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt in main execution. Signaling shutdown.")
        if not stop_event_global.is_set():
            stop_event_global.set()
    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
        if not stop_event_global.is_set():
            stop_event_global.set() # Ensure shutdown on critical error
    finally:
        logger.info("Main execution loop finished or interrupted. Finalizing shutdown...")

        if maintenance_th and maintenance_th.is_alive():
            logger.info("Waiting for archive maintenance thread to complete...")
            maintenance_th.join(timeout=10.0) # Give it time to finish current cycle if any
            if maintenance_th.is_alive():
                logger.warning("Archive maintenance thread did not stop in time.")

        logger.info(f"{APP_NAME} has shut down.")
        logging.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()