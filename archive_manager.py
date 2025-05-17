# archive_manager.py
import os
import datetime
import shutil  # Для получения информации о дисковом пространстве
import logging
import time

logger = logging.getLogger(__name__)


# Вспомогательная функция для db_api (можно ее и там оставить, но для управления архивом она нужна здесь)
def ensure_aware_utc_for_archive(dt_obj: datetime.datetime, default_tz) -> datetime.datetime:
    """Делает naive datetime объект aware UTC, или конвертирует existing aware в UTC."""
    # default_tz должен быть объектом timezone (pytz.utc или ZoneInfo("UTC"))
    if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:  # Если naive
        return default_tz.localize(dt_obj) if hasattr(default_tz, 'localize') else dt_obj.replace(tzinfo=default_tz)
    else:  # Уже aware
        return dt_obj.astimezone(default_tz)


def get_disk_usage(path: str) -> tuple[int, int, int]:
    """Возвращает (total, used, free) в байтах для указанного пути."""
    try:
        total, used, free = shutil.disk_usage(path)
        return total, used, free
    except FileNotFoundError:
        logger.error(f"Путь для анализа дискового пространства не найден: {path}")
        # Пытаемся получить для корневого диска, если path это директория
        try:
            drive = os.path.splitdrive(os.path.abspath(path))[0] + os.sep
            total, used, free = shutil.disk_usage(drive)
            logger.warning(f"Используется корневой диск {drive} для анализа пространства.")
            return total, used, free
        except Exception as e:
            logger.error(f"Не удалось получить информацию о дисковом пространстве: {e}")
            return 0, 0, 0
    except Exception as e:
        logger.error(f"Не удалось получить информацию о дисковом пространстве: {e}")
        return 0, 0, 0


class ArchiveCleaner:
    def __init__(self, db_api_module, recordings_path: str, config_max_days: int = 30,
                 config_max_size_gb: float = 500.0):
        self.db_api = db_api_module
        self.recordings_path = recordings_path
        self.max_days = config_max_days
        self.max_size_bytes = int(config_max_size_gb * 1024 ** 3)  # GB в байты

        # Определение UTC timezone объекта для консистентности
        try:
            from zoneinfo import ZoneInfo
            self.utc_tz = ZoneInfo("UTC")
        except ImportError:
            import pytz
            self.utc_tz = pytz.utc

        logger.info(
            f"ArchiveCleaner инициализирован: путь={recordings_path}, max_days={self.max_days}, max_size_gb={config_max_size_gb}")

    def set_retention_days(self, days: int):
        self.max_days = days
        logger.info(f"Новый срок хранения архива установлен: {self.max_days} дней.")

    def get_current_archive_size(self) -> int:
        """Возвращает текущий размер всех файлов в папке recordings_path."""
        total_size = 0
        try:
            for entry in os.scandir(self.recordings_path):
                if entry.is_file() and entry.name.lower().endswith('.avi'):  # Считаем только .avi файлы
                    try:
                        total_size += entry.stat().st_size
                    except FileNotFoundError:
                        # Файл мог быть удален другим процессом
                        logger.warning(f"Файл {entry.name} не найден при подсчете размера архива.")
                        pass
            return total_size
        except FileNotFoundError:
            logger.error(f"Директория для подсчета размера архива не найдена: {self.recordings_path}")
            return 0
        except Exception as e:
            logger.error(f"Ошибка при подсчете размера архива: {e}")
            return 0

    def cleanup_archive(self) -> tuple[int, int]:
        """
        Выполняет очистку архива по дням и по размеру.
        Возвращает (количество удаленных файлов, общий освобожденный размер в байтах).
        """
        deleted_files_count = 0
        freed_space_bytes = 0

        # Шаг 1: Очистка по сроку хранения (дням)
        logger.info(f"Запуск очистки архива по сроку хранения (старше {self.max_days} дней).")
        now_utc = ensure_aware_utc_for_archive(datetime.datetime.utcnow(), self.utc_tz)
        cutoff_date_utc = now_utc - datetime.timedelta(days=self.max_days)

        files_to_delete_by_date = self.db_api.get_video_clips_older_than(cutoff_date_utc)
        logger.info(
            f"Найдено {len(files_to_delete_by_date)} файлов старше {cutoff_date_utc.isoformat()} для удаления по дате.")

        for clip_id, filename, start_time_utc_str in files_to_delete_by_date:
            file_path = os.path.join(self.recordings_path, filename)
            try:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    self.db_api.delete_video_clip_record(clip_id)  # Удаляем запись из БД
                    deleted_files_count += 1
                    freed_space_bytes += file_size
                    logger.info(f"Удален старый файл (по дате): {filename}, размер: {file_size} байт.")
                else:
                    logger.warning(
                        f"Файл {filename} (ID: {clip_id}) для удаления по дате не найден на диске. Удаление записи из БД.")
                    self.db_api.delete_video_clip_record(clip_id)
            except Exception as e:
                logger.error(f"Ошибка при удалении файла {filename} (по дате): {e}")

        # Шаг 2: Очистка по размеру, если все еще превышен лимит
        # Повторно получаем самый старый файл, пока размер не станет приемлемым
        # Этот цикл может быть долгим, если нужно удалить много файлов
        max_retries_size_cleanup = 1000  # Защита от бесконечного цикла
        retries = 0

        # После удаления по дате, проверяем размер
        current_archive_size_bytes = self.get_current_archive_size()
        logger.info(f"Текущий размер архива после очистки по дате: {current_archive_size_bytes / (1024 ** 3):.2f} GB.")

        while current_archive_size_bytes > self.max_size_bytes and retries < max_retries_size_cleanup:
            if retries == 0:
                logger.info(f"Размер архива ({current_archive_size_bytes / (1024 ** 3):.2f} GB) "
                            f"превышает лимит ({self.max_size_bytes / (1024 ** 3):.2f} GB). "
                            f"Начинаем удаление самых старых файлов по размеру.")

            oldest_clips = self.db_api.get_oldest_video_clips(limit=1)  # Получаем 1 самый старый файл
            if not oldest_clips:
                logger.warning(
                    "Лимит по размеру превышен, но в БД нет файлов для удаления. Остановка очистки по размеру.")
                break

            clip_id, filename, _ = oldest_clips[0]
            file_path = os.path.join(self.recordings_path, filename)
            try:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    self.db_api.delete_video_clip_record(clip_id)
                    deleted_files_count += 1
                    freed_space_bytes += file_size
                    current_archive_size_bytes -= file_size  # Обновляем текущий размер
                    logger.info(f"Удален старый файл (по размеру): {filename}, размер: {file_size} байт. "
                                f"Новый размер архива: {current_archive_size_bytes / (1024 ** 3):.2f} GB.")
                else:
                    logger.warning(
                        f"Самый старый файл {filename} (ID: {clip_id}) для удаления по размеру не найден на диске. Удаление записи из БД.")
                    self.db_api.delete_video_clip_record(clip_id)
                    # Размер не изменился, но запись удалена
            except Exception as e:
                logger.error(f"Ошибка при удалении файла {filename} (по размеру): {e}")
                break  # Прерываем цикл при ошибке удаления файла

            retries += 1
            time.sleep(0.01)  # Небольшая пауза, чтобы не нагружать диск сильно

        if retries == max_retries_size_cleanup:
            logger.warning(f"Достигнут лимит попыток ({max_retries_size_cleanup}) при очистке по размеру.")

        logger.info(
            f"Очистка архива завершена. Удалено файлов: {deleted_files_count}, освобождено: {freed_space_bytes / (1024 ** 3):.2f} GB.")
        return deleted_files_count, freed_space_bytes

    def run_periodic_cleanup(self, interval_seconds: int = 3600):  # По умолчанию раз в час
        """Запускает периодическую очистку в отдельном потоке."""
        import threading

        if hasattr(self, '_cleanup_thread') and self._cleanup_thread.is_alive():
            logger.info("Поток периодической очистки уже запущен.")
            return

        self._stop_periodic_cleanup_event = threading.Event()

        def _worker():
            logger.info(f"Запущен поток периодической очистки архива с интервалом {interval_seconds} секунд.")
            while not self._stop_periodic_cleanup_event.is_set():
                self.cleanup_archive()
                # Ждем указанный интервал или пока не придет сигнал остановки
                self._stop_periodic_cleanup_event.wait(interval_seconds)
            logger.info("Поток периодической очистки архива остановлен.")

        self._cleanup_thread = threading.Thread(target=_worker, daemon=True)
        self.cleanup_thread_name = "ArchiveCleanupThread"
        self._cleanup_thread.start()

    def stop_periodic_cleanup(self):
        if hasattr(self, '_stop_periodic_cleanup_event'):
            logger.info("Остановка потока периодической очистки архива...")
            self._stop_periodic_cleanup_event.set()
            if hasattr(self, '_cleanup_thread') and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=10)  # Даем время на завершение
                if self._cleanup_thread.is_alive():
                    logger.warning("Поток периодической очистки не завершился за 10 секунд.")