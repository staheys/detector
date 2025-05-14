# src/smart_detector/archive_manager.py
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import threading  # For thread name

from .utils import get_disk_free_space_gb  # Assuming it's in utils

logger = logging.getLogger(__name__)


def archive_maintenance_thread_func(stop_event: threading.Event, cfg: dict):
    """Periodically cleans up old clips and manages disk space."""
    current_thread_name = "ArchiveMaintThread"
    threading.current_thread().name = current_thread_name

    maint_cfg = cfg.get('archive_maintenance')
    record_cfg = cfg.get('person_event_recording')

    if not maint_cfg or not maint_cfg.get('enabled', False) or not record_cfg:
        logger.info(f"[{current_thread_name}] Archive maintenance is disabled or recording config missing.")
        return

    clips_dir = Path(record_cfg['output_directory'])
    max_age_days = maint_cfg.get('max_clip_age_days', 7)
    min_disk_gb = maint_cfg.get('min_disk_free_gb_threshold', 5)
    interval_sec = maint_cfg.get('cleanup_interval_seconds', 3600)

    logger.info(
        f"[{current_thread_name}] Started. Cleaning '{clips_dir}'. "
        f"Max age: {max_age_days} days, Min disk: {min_disk_gb}GB. Interval: {interval_sec}s."
    )

    while not stop_event.wait(interval_sec):
        logger.info(f"[{current_thread_name}] Running cleanup cycle...")
        now_dt = datetime.now()
        files_deleted_age = 0
        files_deleted_space = 0
        total_freed_gb_space = 0.0

        # 1. Delete by age
        if max_age_days > 0:
            age_limit_ts = (now_dt - timedelta(days=max_age_days)).timestamp()
            try:
                for f_path in clips_dir.glob('*.mp4'):  # Adjust glob if other extensions used
                    if f_path.is_file():
                        try:
                            if f_path.stat().st_mtime < age_limit_ts:
                                file_size_gb = f_path.stat().st_size / (1024 ** 3)
                                logger.info(
                                    f"[{current_thread_name}] Deleting old clip (age): {f_path.name} ({file_size_gb:.3f} GB)")
                                f_path.unlink()
                                files_deleted_age += 1
                        except Exception as e_file_age:
                            logger.error(
                                f"[{current_thread_name}] Error processing {f_path} for age deletion: {e_file_age}")
            except Exception as e_glob_age:
                logger.error(f"[{current_thread_name}] Error globbing files for age deletion: {e_glob_age}")

        if files_deleted_age > 0:
            logger.info(f"[{current_thread_name}] Deleted {files_deleted_age} clips due to age.")

        # 2. Delete by disk space (oldest first if threshold breached)
        if min_disk_gb > 0:
            try:
                current_free_gb = get_disk_free_space_gb(clips_dir)
                logger.info(
                    f"[{current_thread_name}] Disk space: {current_free_gb:.2f}GB free. Threshold: {min_disk_gb}GB.")

                if current_free_gb < min_disk_gb:
                    logger.warning(
                        f"[{current_thread_name}] Low disk space. Attempting to free up by deleting oldest clips...")
                    all_clips = sorted(
                        [p for p in clips_dir.glob('*.mp4') if p.is_file()],
                        key=lambda p: p.stat().st_mtime  # Oldest first
                    )
                    for old_clip in all_clips:
                        # Re-check free space before each deletion
                        if get_disk_free_space_gb(clips_dir) >= min_disk_gb:
                            logger.info(
                                f"[{current_thread_name}] Disk space threshold met after deleting {files_deleted_space} files for space.")
                            break
                        try:
                            file_size_gb = old_clip.stat().st_size / (1024 ** 3)
                            logger.info(
                                f"[{current_thread_name}] Deleting oldest clip (space): {old_clip.name} ({file_size_gb:.3f} GB)")
                            old_clip.unlink()
                            files_deleted_space += 1
                            total_freed_gb_space += file_size_gb
                        except Exception as e_file_space:
                            logger.error(
                                f"[{current_thread_name}] Error deleting {old_clip} for disk space: {e_file_space}")

                    if files_deleted_space > 0:
                        logger.info(
                            f"[{current_thread_name}] Deleted {files_deleted_space} clips to free disk space (freed ~{total_freed_gb_space:.3f} GB).")
                    if get_disk_free_space_gb(clips_dir) < min_disk_gb:
                        logger.warning(
                            f"[{current_thread_name}] Still below disk space threshold after attempting cleanup.")
            except Exception as e_space_check:
                logger.error(f"[{current_thread_name}] Error during disk space check/cleanup: {e_space_check}")

        logger.info(f"[{current_thread_name}] Cleanup cycle finished.")
    logger.info(f"[{current_thread_name}] Stopping.")