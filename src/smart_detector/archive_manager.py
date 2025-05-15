# src/smart_detector/archive_manager.py
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import threading

from .utils import get_disk_free_space_gb  # Assuming it's in utils

logger = logging.getLogger(__name__)


def archive_maintenance_thread_func(stop_event: threading.Event, cfg: dict):
    current_thread_name = "ArchiveMaintThread"
    threading.current_thread().name = current_thread_name

    maint_cfg = cfg.get('archive_maintenance')

    if not maint_cfg or not maint_cfg.get('enabled', False):
        logger.info(f"[{current_thread_name}] Archive maintenance is disabled.")
        return

    dirs_to_clean_str = maint_cfg.get('directories_to_clean', [])
    if not dirs_to_clean_str:
        logger.info(f"[{current_thread_name}] No directories configured for cleaning. Maintenance thread stopping.")
        return

    dirs_to_clean = [Path(d) for d in dirs_to_clean_str]
    max_age_days = maint_cfg.get('max_clip_age_days', 7)
    min_disk_gb = maint_cfg.get('min_disk_free_gb_threshold', 5)
    interval_sec = maint_cfg.get('cleanup_interval_seconds', 3600)

    logger.info(
        f"[{current_thread_name}] Started. Cleaning: {dirs_to_clean_str}. "
        f"Max age: {max_age_days} days, Min disk: {min_disk_gb}GB. Interval: {interval_sec}s."
    )

    while not stop_event.wait(interval_sec):
        logger.info(f"[{current_thread_name}] Running cleanup cycle...")
        now_dt = datetime.now()

        for clips_dir in dirs_to_clean:
            if not clips_dir.exists():
                logger.warning(f"[{current_thread_name}] Directory for cleaning does not exist: {clips_dir}. Skipping.")
                continue

            logger.info(f"[{current_thread_name}] --- Cleaning directory: {clips_dir} ---")
            files_deleted_age = 0;
            files_deleted_space = 0;
            total_freed_gb_space = 0.0

            if max_age_days > 0:
                age_limit_ts = (now_dt - timedelta(days=max_age_days)).timestamp()
                try:
                    for f_path in clips_dir.glob('*.avi'):  # Changed to .avi
                        if f_path.is_file():
                            try:
                                if f_path.stat().st_mtime < age_limit_ts:
                                    file_size_gb = f_path.stat().st_size / (1024 ** 3)
                                    logger.info(
                                        f"[{current_thread_name}] Deleting old clip (age) from {clips_dir}: {f_path.name} ({file_size_gb:.3f} GB)")
                                    f_path.unlink()
                                    files_deleted_age += 1
                            except Exception as e_file_age:
                                logger.error(
                                    f"[{current_thread_name}] Error processing {f_path} for age deletion: {e_file_age}")
                except Exception as e_glob_age:
                    logger.error(
                        f"[{current_thread_name}] Error globbing files in {clips_dir} for age deletion: {e_glob_age}")
            if files_deleted_age > 0: logger.info(
                f"[{current_thread_name}] In {clips_dir}, deleted {files_deleted_age} clips due to age.")

            if min_disk_gb > 0:
                try:
                    current_free_gb = get_disk_free_space_gb(clips_dir)
                    logger.info(
                        f"[{current_thread_name}] Disk space for partition of {clips_dir}: {current_free_gb:.2f}GB free. Threshold: {min_disk_gb}GB.")
                    if current_free_gb < min_disk_gb:
                        logger.warning(
                            f"[{current_thread_name}] Low disk space on partition of {clips_dir}. Attempting to free up by deleting oldest clips from {clips_dir}...")
                        all_clips_in_dir = sorted([p for p in clips_dir.glob('*.avi') if p.is_file()],
                                                  key=lambda p: p.stat().st_mtime)
                        for old_clip in all_clips_in_dir:
                            if get_disk_free_space_gb(clips_dir) >= min_disk_gb:
                                logger.info(
                                    f"[{current_thread_name}] Disk space threshold met after deleting {files_deleted_space} files from {clips_dir} for space.")
                                break
                            try:
                                file_size_gb = old_clip.stat().st_size / (1024 ** 3)
                                logger.info(
                                    f"[{current_thread_name}] Deleting oldest clip (space) from {clips_dir}: {old_clip.name} ({file_size_gb:.3f} GB)")
                                old_clip.unlink();
                                files_deleted_space += 1;
                                total_freed_gb_space += file_size_gb
                            except Exception as e_file_space:
                                logger.error(
                                    f"[{current_thread_name}] Error deleting {old_clip} from {clips_dir} for disk space: {e_file_space}")
                        if files_deleted_space > 0: logger.info(
                            f"[{current_thread_name}] In {clips_dir}, deleted {files_deleted_space} clips to free disk space (freed ~{total_freed_gb_space:.3f} GB).")
                        if get_disk_free_space_gb(clips_dir) < min_disk_gb: logger.warning(
                            f"[{current_thread_name}] Still below disk space threshold for {clips_dir} after attempting cleanup.")
                except Exception as e_space_check:
                    logger.error(
                        f"[{current_thread_name}] Error during disk space check/cleanup for {clips_dir}: {e_space_check}")
            logger.info(f"[{current_thread_name}] --- Finished cleaning {clips_dir} ---")
        logger.info(f"[{current_thread_name}] Cleanup cycle finished for all configured directories.")
    logger.info(f"[{current_thread_name}] Stopping.")