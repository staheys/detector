# src/smart_detector/cli.py
import argparse
import sys
import logging
from threading import Thread, Event, current_thread as get_current_thread
import signal
from pathlib import Path

from .utils import load_config, setup_logging
from .application_core import run_application_logic  # Имя изменено
from .archive_manager import archive_maintenance_thread_func
from . import APP_NAME, __version__

logger = logging.getLogger(APP_NAME)
stop_event_global = Event()


def signal_handler(signum, frame):
    signal_name = signal.Signals(signum).name
    logger.warning(f"Signal {signal_name} ({signum}) received. Initiating shutdown...")
    if not stop_event_global.is_set(): stop_event_global.set()


def main(argv=None):
    get_current_thread().name = "MainCLIThread"
    p = argparse.ArgumentParser(description=f"{APP_NAME}", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default="config.yml", help="Path to YAML configuration file.")
    args = p.parse_args(argv)

    config_data = None
    try:
        config_data = load_config(args.config)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR); logger.fatal(f"Config load error '{args.config}': {e}",
                                                               exc_info=True); sys.exit(1)
    setup_logging()

    logger.info(f"Starting {APP_NAME} v{__version__}...")
    logger.info(f"Config: {Path(args.config).resolve()}")
    if config_data.get('stream_display_archive', {}).get('enabled'):
        logger.info(f"Display/Archive Stream (101): {config_data['stream_display_archive']['url']}")
    if config_data.get('stream_detection', {}).get('enabled'):
        logger.info(f"Detection Stream (102): {config_data['stream_detection']['url']}")
    logger.info(
        f"Model: {config_data.get('inference', {}).get('model_path')}, Device: {config_data.get('inference', {}).get('device')}, NthFrame: {config_data.get('inference', {}).get('process_every_nth_frame')}")

    signal.signal(signal.SIGINT, signal_handler);
    signal.signal(signal.SIGTERM, signal_handler)

    maintenance_th = None
    if config_data.get('archive_maintenance', {}).get('enabled', False):
        maintenance_th = Thread(target=archive_maintenance_thread_func, args=(stop_event_global, config_data),
                                daemon=True)
        maintenance_th.start()

    try:
        run_application_logic(stop_event_global, config_data)  # Вызов измененной функции
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt. Signaling shutdown.")
    except Exception as e:
        logger.critical(f"Unhandled CLI exception: {e}", exc_info=True)
    finally:
        if not stop_event_global.is_set(): stop_event_global.set()
        logger.info("Main CLI finalizing shutdown...")
        if maintenance_th and maintenance_th.is_alive():
            logger.info("Waiting for archive maintenance thread...");
            maintenance_th.join(timeout=12.0)
            if maintenance_th.is_alive(): logger.warning("Archive maintenance thread did not stop.")
        logger.info(f"{APP_NAME} has shut down.");
        logging.shutdown();
        sys.exit(0)


if __name__ == "__main__":
    main()