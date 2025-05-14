# src/smart_detector/utils.py
import yaml
import logging
import logging.handlers
from pathlib import Path
import time
import os
import shutil

CONFIG = {}


def load_config(config_path: str | Path = 'config.yml') -> dict:
    global CONFIG
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)

    # Resolve project_root and ensure critical paths are absolute
    project_root_str = CONFIG.get('app', {}).get('project_root', '.')
    project_root = Path(project_root_str).resolve()
    CONFIG['app']['project_root'] = project_root

    if 'model_path' in CONFIG['inference']:
        model_path = Path(CONFIG['inference']['model_path'])
        if not model_path.is_absolute():
            CONFIG['inference']['model_path'] = (project_root / model_path).resolve()

    if 'base_path' in CONFIG['archiving']:
        archive_path = Path(CONFIG['archiving']['base_path'])
        if not archive_path.is_absolute():
            CONFIG['archiving']['base_path'] = (project_root / archive_path).resolve()
        CONFIG['archiving']['base_path'].mkdir(parents=True, exist_ok=True)

    log_dir_str = CONFIG.get('logging', {}).get('directory', 'logs')
    log_dir = Path(log_dir_str)
    if not log_dir.is_absolute():
        CONFIG['logging']['directory'] = (project_root / log_dir).resolve()
    CONFIG['logging']['directory'].mkdir(parents=True, exist_ok=True)

    return CONFIG


def get_config() -> dict:
    if not CONFIG:
        # Try loading default if not already loaded by main app
        # This is a fallback, prefer explicit loading
        try:
            load_config()
            logging.info("Configuration loaded implicitly.")
        except FileNotFoundError:
            logging.error("CONFIG NOT LOADED. Call load_config() first.")
            return {}  # Return empty dict to avoid crashing if accessed prematurely
    return CONFIG


def setup_logging():
    cfg = get_config()['logging']
    log_level_str = cfg.get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    log_dir = cfg['directory']
    log_file = log_dir / cfg.get('filename', 'app.log')

    # Basic config for console
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()]  # Default to console
    )

    # File handler with rotation
    if log_file:
        should_roll_over = False
        if log_file.exists():
            if log_file.stat().st_size >= cfg.get('rotation_max_bytes', 10 * 1024 * 1024):
                should_roll_over = True

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=cfg.get('rotation_max_bytes', 10 * 1024 * 1024),  # 10MB
            backupCount=cfg.get('rotation_backup_count', 5),
            encoding='utf-8'
        )
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] %(message)s",
                                      "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)

        # Add file handler to root logger
        logging.getLogger().addHandler(file_handler)
        if should_roll_over:  # Perform rollover if needed before logging starts
            file_handler.doRollover()
        logging.info(f"Logging initialized. Level: {log_level_str}. Log file: {log_file}")


def get_disk_free_space_gb(path: str | Path) -> float:
    """Returns free disk space in GB for the given path."""
    try:
        total, used, free = shutil.disk_usage(Path(path).anchor)
        return free / (1024 ** 3)
    except Exception as e:
        logging.error(f"Could not get disk usage for {path}: {e}")
        return float('inf')  # Assume plenty of space if check fails


def generate_event_id() -> str:
    return f"evt_{time.strftime('%Y%m%d%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{os.urandom(3).hex()}"


# Default colors for drawing, can be expanded or customized
CLASS_COLORS = {
    "person": (0, 255, 0),  # Green
    "ppe_helmet": (255, 0, 0),  # Blue
    "ppe_vest": (0, 0, 255),  # Red
    "cargo_box": (255, 255, 0),  # Yellow
    "default": (128, 128, 128)  # Grey
}


def get_class_color(class_name: str) -> tuple:
    return CLASS_COLORS.get(class_name.lower(), CLASS_COLORS["default"])