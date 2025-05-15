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
    # ... (логика поиска файла config.yml, как и ранее) ...
    p = Path(config_path)
    if not p.is_file():
        script_dir = Path(__file__).resolve().parent
        p_rel_script = script_dir.parent.parent / config_path
        if p_rel_script.is_file():
            p = p_rel_script
        else:
            p_cwd = Path.cwd() / config_path
            if p_cwd.is_file():
                p = p_cwd
            else:
                raise FileNotFoundError(
                    f"Config file not found: {config_path} (checked absolute, relative to script, CWD)")
    resolved_config_path = p.resolve()

    with open(resolved_config_path, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)

    project_root_str = CONFIG.get('app', {}).get('project_root', '.')
    project_root = (resolved_config_path.parent / project_root_str).resolve() if not Path(
        project_root_str).is_absolute() else Path(project_root_str).resolve()
    CONFIG['app']['project_root'] = project_root

    def _resolve_path(cfg_section, key, base_path, create_dir=False):
        if cfg_section in CONFIG and isinstance(CONFIG[cfg_section], dict) and key in CONFIG[cfg_section]:
            path_val = Path(CONFIG[cfg_section][key])
            resolved = (base_path / path_val).resolve() if not path_val.is_absolute() else path_val.resolve()
            CONFIG[cfg_section][key] = resolved
            if create_dir: resolved.mkdir(parents=True, exist_ok=True)

    _resolve_path('inference', 'model_path', project_root)

    # Resolve paths for stream-specific recordings
    if CONFIG.get('stream_display_archive', {}).get('enabled', False) and CONFIG.get('periodic_archiving', {}).get(
            'enabled', False):
        _resolve_path('periodic_archiving', 'output_directory', project_root, create_dir=True)

    if CONFIG.get('stream_detection', {}).get('enabled', False) and CONFIG.get('person_event_recording', {}).get(
            'enabled', False):
        _resolve_path('person_event_recording', 'output_directory', project_root, create_dir=True)

    _resolve_path('logging', 'directory', project_root, create_dir=True)

    # Populate directories_to_clean for archive_maintenance
    if CONFIG.get('archive_maintenance', {}).get('enabled', False):
        dirs_to_clean = []
        if CONFIG.get('periodic_archiving', {}).get('enabled', False):
            dirs_to_clean.append(str(CONFIG['periodic_archiving']['output_directory']))
        if CONFIG.get('person_event_recording', {}).get('enabled', False):
            dirs_to_clean.append(str(CONFIG['person_event_recording']['output_directory']))

        CONFIG['archive_maintenance']['directories_to_clean'] = list(set(dirs_to_clean))
        for dir_path_str in CONFIG['archive_maintenance']['directories_to_clean']:
            Path(dir_path_str).mkdir(parents=True, exist_ok=True)  # Ensure they exist

    # Ensure process_every_nth_frame is at least 1
    if 'inference' in CONFIG:
        CONFIG['inference']['process_every_nth_frame'] = max(1,
                                                             int(CONFIG['inference'].get('process_every_nth_frame', 1)))
    else:
        CONFIG['inference'] = {'process_every_nth_frame': 1}
    return CONFIG


# ... (get_config, setup_logging, get_disk_free_space_gb, get_detection_color - без изменений) ...
def get_config() -> dict:
    if not CONFIG:
        try:
            load_config(); logging.info("Config loaded implicitly by get_config().")
        except FileNotFoundError:
            logging.error("CONFIG NOT LOADED."); return {}
    return CONFIG


def setup_logging():  # Без изменений
    cfg_all = get_config()
    if not cfg_all or 'logging' not in cfg_all:
        print("Cannot setup logging: config or logging section not loaded.", file=os.sys.stderr)
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
        logging.error("FALLBACK LOGGING. CONFIG/LOGGING SECTION NOT LOADED.")
        return
    cfg = cfg_all['logging']
    log_level_str = cfg.get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_dir = Path(cfg['directory']);
    log_file = log_dir / cfg.get('filename', 'app.log')
    try:
        from rich.logging import RichHandler
        console_handler = RichHandler(rich_tracebacks=True, show_path=False, show_level=True, show_time=True)
        log_format = "%(message)s";
        date_format = "[%X]"
    except ImportError:
        console_handler = logging.StreamHandler()
        log_format = "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] [%(threadName)s] %(message)s";
        date_format = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format, handlers=[console_handler])
    if log_file:
        should_roll_over = False
        if log_file.exists() and log_file.stat().st_size >= cfg.get('rotation_max_bytes',
                                                                    10 * 1024 * 1024): should_roll_over = True
        file_handler = logging.handlers.RotatingFileHandler(log_file,
                                                            maxBytes=cfg.get('rotation_max_bytes', 10 * 1024 * 1024),
                                                            backupCount=cfg.get('rotation_backup_count', 3),
                                                            encoding='utf-8')
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] [%(threadName)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter);
        logging.getLogger().addHandler(file_handler)
        if should_roll_over: file_handler.doRollover()
        logging.info(f"File logging initialized. Level: {log_level_str}. Log file: {log_file}")
    else:
        logging.info(f"Console logging initialized. Level: {log_level_str}.")


def get_disk_free_space_gb(path: str | Path) -> float:  # Без изменений
    try:
        stat_path = Path(path).resolve();
        check_path = stat_path if stat_path.is_dir() else stat_path.parent
        if not check_path.exists(): check_path = stat_path.anchor
        total, used, free = shutil.disk_usage(check_path);
        return free / (1024 ** 3)
    except Exception as e:
        logging.error(f"Disk usage error for {path}: {e}"); return float('inf')


PERSON_COLOR = (0, 255, 0);
DEFAULT_COLOR = (128, 128, 128)  # Без изменений


def get_detection_color(class_name: str) -> tuple:  # Без изменений
    return PERSON_COLOR if class_name.lower() == "person" else DEFAULT_COLOR