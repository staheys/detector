# src/smart_detector/utils.py
import yaml
import logging
import logging.handlers
from pathlib import Path
import time
import os
import shutil
from datetime import datetime

CONFIG = {}


def load_config(config_path: str | Path = 'config.yml') -> dict:
    global CONFIG
    p = Path(config_path)
    if not p.is_file():
        script_dir = Path(__file__).resolve().parent
        # Assuming config.yml is at project root, two levels up from src/smart_detector/
        p_rel_script = script_dir.parent.parent / config_path
        if p_rel_script.is_file():
            p = p_rel_script
        else:
            p_cwd = Path.cwd() / config_path
            if p_cwd.is_file():
                p = p_cwd
            else:
                raise FileNotFoundError(
                    f"Configuration file not found: {config_path} "
                    f"(Checked absolute, relative to script: {p_rel_script}, and CWD: {p_cwd})"
                )
    resolved_config_path = p.resolve()

    with open(resolved_config_path, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)

    project_root_str = CONFIG.get('app', {}).get('project_root', '.')
    if not Path(project_root_str).is_absolute():
        # project_root is relative to the *config file's directory*
        project_root = (resolved_config_path.parent / project_root_str).resolve()
    else:
        project_root = Path(project_root_str).resolve()
    CONFIG['app']['project_root'] = project_root

    def resolve_path_in_cfg(cfg_section_key, path_key, is_dir_to_create=False):
        if cfg_section_key in CONFIG and path_key in CONFIG[cfg_section_key]:
            path_val_str = CONFIG[cfg_section_key][path_key]
            path_val = Path(path_val_str)
            if not path_val.is_absolute():
                CONFIG[cfg_section_key][path_key] = (project_root / path_val).resolve()
            else:
                CONFIG[cfg_section_key][path_key] = path_val.resolve()

            if is_dir_to_create:
                CONFIG[cfg_section_key][path_key].mkdir(parents=True, exist_ok=True)

    resolve_path_in_cfg('inference', 'model_path', is_dir_to_create=False)
    resolve_path_in_cfg('person_event_recording', 'output_directory', is_dir_to_create=True)
    resolve_path_in_cfg('logging', 'directory', is_dir_to_create=True)
    if 'archive_maintenance' in CONFIG and 'output_directory' not in CONFIG['archive_maintenance']:
        # if maint needs to know about output_directory from person_event_recording
        CONFIG['archive_maintenance']['output_directory_to_clean'] = CONFIG['person_event_recording'][
            'output_directory']

    return CONFIG


def get_config() -> dict:
    if not CONFIG:
        try:
            load_config()
            logging.info("Configuration loaded implicitly by get_config().")
        except FileNotFoundError:
            logging.error("CONFIG NOT LOADED. Call load_config() explicitly first from main.")
            return {}
    return CONFIG


def setup_logging():
    cfg_all = get_config()
    if not cfg_all or 'logging' not in cfg_all:
        print("Cannot setup logging: config or logging section not loaded.", file=os.sys.stderr)
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
        logging.error("FALLBACK LOGGING. CONFIG/LOGGING SECTION NOT LOADED.")
        return

    cfg = cfg_all['logging']
    log_level_str = cfg.get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    log_dir = Path(cfg['directory'])
    log_file = log_dir / cfg.get('filename', 'app.log')

    try:
        from rich.logging import RichHandler
        console_handler = RichHandler(rich_tracebacks=True, show_path=False, show_level=True, show_time=True)
        log_format = "%(message)s"  # RichHandler handles its own formatting for levels/time
        date_format = "[%X]"
    except ImportError:
        console_handler = logging.StreamHandler()
        log_format = "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] [%(threadName)s] %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[console_handler]
    )

    if log_file:
        should_roll_over = False
        if log_file.exists():
            if log_file.stat().st_size >= cfg.get('rotation_max_bytes', 10 * 1024 * 1024):
                should_roll_over = True

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=cfg.get('rotation_max_bytes', 10 * 1024 * 1024),
            backupCount=cfg.get('rotation_backup_count', 3),
            encoding='utf-8'
        )
        # Explicit formatter for file handler
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] [%(threadName)s] %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)
        if should_roll_over:
            file_handler.doRollover()
        logging.info(f"File logging initialized. Level: {log_level_str}. Log file: {log_file}")
    else:
        logging.info(f"Console logging initialized. Level: {log_level_str}.")


def get_disk_free_space_gb(path: str | Path) -> float:
    try:
        stat_path = Path(path).resolve()
        check_path = stat_path if stat_path.is_dir() else stat_path.parent
        if not check_path.exists(): check_path = stat_path.anchor

        total, used, free = shutil.disk_usage(check_path)
        return free / (1024 ** 3)
    except Exception as e:
        logging.error(
            f"Could not get disk usage for {path} (checked {check_path if 'check_path' in locals() else ''}): {e}")
        return float('inf')


# CLASS_COLORS can be simplified or removed if only drawing "person"
PERSON_COLOR = (0, 255, 0)  # Green for person
DEFAULT_COLOR = (128, 128, 128)


def get_detection_color(class_name: str) -> tuple:
    if class_name.lower() == "person":
        return PERSON_COLOR
    return DEFAULT_COLOR