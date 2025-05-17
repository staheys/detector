# db_api.py
import configparser
import sqlite3
import datetime
import os
import logging
import threading
from typing import List, Dict

import pytz

# Configure logging for the DB API
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None # type: ignore

DB_NAME = "tracking_data.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=10) # Added timeout
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Table for video clips
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                start_time_utc TEXT NOT NULL,
                end_time_utc TEXT
            )
        ''')
        # Table for tracked objects
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracked_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tracker_object_id INTEGER NOT NULL, -- ID from CentroidTracker for this run
                class_name TEXT NOT NULL,
                first_seen_utc TEXT NOT NULL,
                last_seen_utc TEXT
                -- Consider adding session_id if persistence across app runs is important
            )
        ''')
        conn.commit()
        logger.info(f"Database '{DB_NAME}' initialized/verified successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise
    finally:
        if conn:
            conn.close()

# In-memory cache for active object database IDs: {tracker_object_id: db_primary_key}
_active_object_db_ids_cache = {}
_db_lock = threading.Lock() # Lock for cache modifications

def add_video_clip_start(filename: str, start_time_utc: datetime.datetime) -> int | None:
    start_time_str = start_time_utc.isoformat()
    clip_id = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO video_clips (filename, start_time_utc) VALUES (?, ?)",
            (filename, start_time_str)
        )
        conn.commit()
        clip_id = cursor.lastrowid
        logger.info(f"Video clip started: {filename}, DB ID: {clip_id} at {start_time_str}")
    except sqlite3.IntegrityError: # Handles UNIQUE constraint violation for filename
        logger.warning(f"Filename {filename} already exists in video_clips. Skipping insert.")
    except sqlite3.Error as e:
        logger.error(f"Error adding video clip start for {filename}: {e}")
    finally:
        if conn:
            conn.close()
    return clip_id

def update_video_clip_end(clip_id: int, end_time_utc: datetime.datetime):
    if clip_id is None:
        logger.warning("Attempted to update video clip end with no clip_id.")
        return

    end_time_str = end_time_utc.isoformat()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE video_clips SET end_time_utc = ? WHERE id = ?",
            (end_time_str, clip_id)
        )
        conn.commit()
        logger.info(f"Video clip ended: DB ID {clip_id} at {end_time_str}")
    except sqlite3.Error as e:
        logger.error(f"Error updating video clip end for ID {clip_id}: {e}")
    finally:
        if conn:
            conn.close()

def record_object_first_seen(tracker_object_id: int, class_name: str, first_seen_utc: datetime.datetime):
    with _db_lock:
        if tracker_object_id in _active_object_db_ids_cache:
            # logger.debug(f"Object {tracker_object_id} already active in cache. Not re-inserting.")
            return

    first_seen_str = first_seen_utc.isoformat()
    db_id = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Set last_seen_utc to first_seen_utc initially
        cursor.execute(
            "INSERT INTO tracked_objects (tracker_object_id, class_name, first_seen_utc, last_seen_utc) VALUES (?, ?, ?, ?)",
            (tracker_object_id, class_name, first_seen_str, first_seen_str)
        )
        conn.commit()
        db_id = cursor.lastrowid
        with _db_lock:
            _active_object_db_ids_cache[tracker_object_id] = db_id
        logger.info(f"Object {tracker_object_id} ({class_name}) first seen at {first_seen_str}, DB ID: {db_id}")
    except sqlite3.Error as e:
        logger.error(f"Error recording object first seen for {tracker_object_id} ({class_name}): {e}")
    finally:
        if conn:
            conn.close()


def update_object_last_seen(tracker_object_id: int, last_seen_utc: datetime.datetime):
    with _db_lock:
        db_id = _active_object_db_ids_cache.get(tracker_object_id)

    if not db_id:
        # This might happen if the app restarted and object was known before cache was repopulated.
        # Or if record_object_first_seen failed.
        # For simplicity now, we'll just log. A robust system might try to find existing record.
        logger.warning(f"Object {tracker_object_id} not in active cache. Cannot update last_seen without DB PK from cache.")
        return

    last_seen_str = last_seen_utc.isoformat()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tracked_objects SET last_seen_utc = ? WHERE id = ?",
            (last_seen_str, db_id)
        )
        conn.commit()
        # logger.debug(f"Object {tracker_object_id} (DB ID {db_id}) last seen updated to {last_seen_str}") # Verbose
    except sqlite3.Error as e:
        logger.error(f"Error updating object last seen for {tracker_object_id} (DB ID {db_id}): {e}")
    finally:
        if conn:
            conn.close()

def mark_object_disappeared(tracker_object_id: int, disappeared_time_utc: datetime.datetime):
    db_id = None
    with _db_lock:
        if tracker_object_id in _active_object_db_ids_cache:
            db_id = _active_object_db_ids_cache.pop(tracker_object_id) # Remove from active cache
        else:
            logger.warning(f"Object {tracker_object_id} not in active cache when marking as disappeared.")
            # Attempt to find the latest entry in DB if not in cache (e.g. after restart)
            # This is a fallback and might not always be correct if tracker_object_ids are reused rapidly.
            try:
                conn_temp = get_db_connection()
                cursor_temp = conn_temp.cursor()
                cursor_temp.execute(
                    "SELECT id FROM tracked_objects WHERE tracker_object_id = ? ORDER BY first_seen_utc DESC LIMIT 1",
                    (tracker_object_id,)
                )
                row = cursor_temp.fetchone()
                if row:
                    db_id = row['id']
                    logger.info(f"Found object {tracker_object_id} in DB (ID {db_id}) to mark as disappeared.")
            except sqlite3.Error as e:
                logger.error(f"Error finding object {tracker_object_id} in DB for disappearance: {e}")
            finally:
                if conn_temp: conn_temp.close()


    if db_id:
        # Ensure its last_seen_utc is current with the disappearance time
        last_seen_str = disappeared_time_utc.isoformat()
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE tracked_objects SET last_seen_utc = ? WHERE id = ?",
                (last_seen_str, db_id)
            )
            conn.commit()
            logger.info(f"Object {tracker_object_id} (DB ID {db_id}) marked as disappeared. Last seen: {last_seen_str}")
        except sqlite3.Error as e:
            logger.error(f"Error finalizing last_seen for disappeared object {tracker_object_id} (DB ID {db_id}): {e}")
        finally:
            if conn:
                conn.close()
    else:
        logger.warning(f"Could not find DB entry for object {tracker_object_id} to mark as disappeared.")


def clear_active_objects_cache():
    """Call this on application start to clear the in-memory cache from previous runs."""
    with _db_lock:
        _active_object_db_ids_cache.clear()
    logger.info("Active object cache cleared.")


def get_all_video_clips(limit=100):
    """Fetches all video clips, newest first, up to a limit."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, filename, start_time_utc, end_time_utc FROM video_clips ORDER BY start_time_utc DESC LIMIT ?",
            (limit,))
        clips = cursor.fetchall()
        return clips  # Returns a list of sqlite3.Row objects
    except sqlite3.Error as e:
        logger.error(f"Error fetching video clips: {e}")
        return []
    finally:
        if conn:
            conn.close()


def get_recent_tracked_objects(limit=100):
    """Fetches recent tracked objects, newest first, up to a limit."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Fetch distinct objects based on tracker_object_id, showing their latest record
        # This query might need refinement based on exact needs (e.g. all entries for an object)
        cursor.execute("""
            SELECT id, tracker_object_id, class_name, first_seen_utc, last_seen_utc
            FROM tracked_objects
            ORDER BY last_seen_utc DESC
            LIMIT ?
        """, (limit,))
        objects = cursor.fetchall()
        return objects  # Returns a list of sqlite3.Row objects
    except sqlite3.Error as e:
        logger.error(f"Error fetching tracked objects: {e}")
        return []
    finally:
        if conn:
            conn.close()


def get_config_rtsp_urls(config_path="config.conf") -> list:
    """Reads RTSP URLs from the config file."""
    cfg = configparser.ConfigParser()
    if not cfg.read(config_path):
        logger.error(f"GUI: Configuration file {config_path} not found.")
        return []

    urls = []
    if 'RTSP' in cfg and 'url' in cfg['RTSP']:
        main_yolo_url = cfg['RTSP']['url']
        if main_yolo_url:
            urls.append({"name": "YOLO Stream (main)", "url": main_yolo_url})

    if 'RECORDING' in cfg and 'rtsp_url' in cfg['RECORDING']:
        recording_url = cfg['RECORDING']['rtsp_url']
        if recording_url:
            # Avoid duplicates if URLs are the same
            is_duplicate = any(u["url"] == recording_url for u in urls)
            if not is_duplicate:
                urls.append({"name": "Recording Stream", "url": recording_url})
            elif len(urls) == 1 and urls[0]["url"] == recording_url:  # If it was the yolo stream
                urls[0]["name"] = "YOLO & Recording Stream"

    # You could add a section like [GUI_CAMERAS] to config.conf for more URLs
    # Example:
    # [GUI_CAMERAS]
    # camera1_name = Entrance Cam
    # camera1_url = rtsp://...
    # camera2_name = Back Office Cam
    # camera2_url = rtsp://...
    if 'GUI_CAMERAS' in cfg:
        i = 1
        while True:
            name_key = f'camera{i}_name'
            url_key = f'camera{i}_url'
            if name_key in cfg['GUI_CAMERAS'] and url_key in cfg['GUI_CAMERAS']:
                name = cfg['GUI_CAMERAS'][name_key]
                url = cfg['GUI_CAMERAS'][url_key]
                if name and url:
                    if not any(u["url"] == url for u in urls):  # Avoid duplicates
                        urls.append({"name": name, "url": url})
                i += 1
            else:
                break
    logger.info(f"GUI: Found {len(urls)} RTSP URLs from config for live view.")
    return urls


# db_api.py
# ... (существующие импорты) ...
from ultralytics.utils import YAML  # Для чтения YAML
from ultralytics.utils.checks import check_yaml  # Для проверки YAML


# ... (существующий код) ...

def get_available_classes_from_yaml(config_path="config.conf") -> List[str]:
    """
    Читает файл data.yaml, указанный в config.conf, и возвращает список имен классов.
    """
    cfg_parser = configparser.ConfigParser()
    if not cfg_parser.read(config_path):
        logger.error(f"Не удалось прочитать {config_path} для получения пути к YAML.")
        return []

    if 'YOLO' in cfg_parser and 'yaml_path' in cfg_parser['YOLO']:
        yaml_file_path_str = cfg_parser['YOLO']['yaml_path']
        try:
            checked_yaml_path = check_yaml(yaml_file_path_str)  # Проверяет существование и т.д.
            data_yaml = YAML.load(checked_yaml_path)
            if 'names' in data_yaml and isinstance(data_yaml['names'], list):
                logger.info(f"Загружены классы из {checked_yaml_path}: {data_yaml['names']}")
                return data_yaml['names']
            else:
                logger.error(f"Ключ 'names' не найден или не является списком в {checked_yaml_path}")
                return []
        except Exception as e:
            logger.error(f"Ошибка загрузки или парсинга YAML файла {yaml_file_path_str}: {e}")
            return []
    else:
        logger.error("Секция [YOLO] или ключ 'yaml_path' не найдены в config.conf.")
        return []


def find_object_occurrences_and_video_segments(
        class_name_filter: str,
        start_time_utc_filter: datetime.datetime,  # Aware UTC
        end_time_utc_filter: datetime.datetime  # Aware UTC
) -> List[Dict]:
    conn = None
    all_target_segments_for_composition = []  # Список всех сегментов со всех объектов

    start_utc_str_filter = start_time_utc_filter.isoformat()
    end_utc_str_filter = end_time_utc_filter.isoformat()

    logger.info(
        f"Поиск ИНДИВИДУАЛЬНЫХ вхождений класса '{class_name_filter}' с {start_utc_str_filter} по {end_utc_str_filter}")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 1. Найти все объекты заданного класса, которые были активны в заданном интервале
        query_objects = """
            SELECT tracker_object_id, class_name, first_seen_utc, last_seen_utc
            FROM tracked_objects
            WHERE class_name = ? 
              AND first_seen_utc <= ?  
              AND last_seen_utc >= ?   
            ORDER BY first_seen_utc ASC 
        """
        # Запрос остается тем же, он находит всех кандидатов
        cursor.execute(query_objects, (class_name_filter, end_utc_str_filter, start_utc_str_filter))
        relevant_objects_from_db = cursor.fetchall()

        if not relevant_objects_from_db:
            logger.info(f"Объекты класса '{class_name_filter}' не найдены в БД по первичному SQL-запросу.")
            return []

        logger.info(
            f"Найдено {len(relevant_objects_from_db)} потенциально релевантн(ых) объектов класса '{class_name_filter}'. Обработка каждого индивидуально...")

        # Обрабатываем КАЖДЫЙ найденный объект индивидуально
        for obj_row_idx, obj_row in enumerate(relevant_objects_from_db):
            obj_tracker_id = obj_row["tracker_object_id"]

            obj_first_seen_dt = datetime.datetime.fromisoformat(obj_row["first_seen_utc"])
            obj_last_seen_dt = datetime.datetime.fromisoformat(obj_row["last_seen_utc"])

            # Делаем datetime из БД "aware" UTC
            obj_first_seen_utc = ensure_aware_utc(obj_first_seen_dt)
            obj_last_seen_utc = ensure_aware_utc(obj_last_seen_dt)

            # Определяем фактический интервал активности этого КОНКРЕТНОГО объекта ВНУТРИ фильтра
            individual_event_start_utc = max(obj_first_seen_utc, start_time_utc_filter)
            individual_event_end_utc = min(obj_last_seen_utc, end_time_utc_filter)

            # Если интервал объекта после обрезки фильтром стал невалидным, пропускаем этот объект
            if individual_event_start_utc >= individual_event_end_utc:
                logger.debug(
                    f"Объект ID {obj_tracker_id} ({obj_first_seen_utc.isoformat()} - {obj_last_seen_utc.isoformat()}) "
                    f"не имеет активного периода внутри фильтра. Пропускается.")
                continue

            logger.info(f"Обработка объекта ID {obj_tracker_id}: активен с {individual_event_start_utc.isoformat()} "
                        f"по {individual_event_end_utc.isoformat()} (в рамках фильтра).")

            # 2. Для этого КОНКРЕТНОГО интервала [individual_event_start_utc, individual_event_end_utc]
            #    найти все пересекающиеся видеоклипы.
            query_clips_for_event = """
                SELECT id, filename, start_time_utc, end_time_utc
                FROM video_clips
                WHERE start_time_utc <= ? -- Клип начался до/во время конца этого события
                  AND (end_time_utc IS NULL OR end_time_utc >= ?) -- Клип закончился после/во время начала этого события
                ORDER BY start_time_utc ASC
            """
            individual_event_start_str = individual_event_start_utc.isoformat()
            individual_event_end_str = individual_event_end_utc.isoformat()

            cursor.execute(query_clips_for_event, (individual_event_end_str, individual_event_start_str))
            clips_for_this_event = cursor.fetchall()

            if not clips_for_this_event:
                logger.info(f"  Для объекта ID {obj_tracker_id} не найдено клипов, покрывающих его интервал активности "
                            f"{individual_event_start_str} - {individual_event_end_str}.")
                continue

            logger.info(
                f"  Для объекта ID {obj_tracker_id} найдено {len(clips_for_this_event)} клипов, покрывающих его интервал.")

            for clip_row in clips_for_this_event:
                clip_start_utc_dt_original = ensure_aware_utc(
                    datetime.datetime.fromisoformat(clip_row["start_time_utc"]))

                clip_end_utc_dt_original_or_event_end = None
                if clip_row["end_time_utc"] is not None:
                    clip_end_utc_dt_original_or_event_end = ensure_aware_utc(
                        datetime.datetime.fromisoformat(clip_row["end_time_utc"]))
                else:  # Клип еще идет
                    clip_end_utc_dt_original_or_event_end = individual_event_end_utc  # Ограничиваем концом текущего события
                    if clip_start_utc_dt_original > individual_event_end_utc:  # Активный клип начался после события
                        continue

                # Определяем пересечение интервала КЛИПА с интервалом АКТИВНОСТИ этого КОНКРЕТНОГО ОБЪЕКТА
                # [individual_event_start_utc, individual_event_end_utc] <- интервал текущего события
                # [clip_start_utc_dt_original, clip_end_utc_dt_original_or_event_end] <- интервал клипа

                segment_start_utc = max(individual_event_start_utc, clip_start_utc_dt_original)
                segment_end_utc = min(individual_event_end_utc, clip_end_utc_dt_original_or_event_end)

                if segment_start_utc >= segment_end_utc:  # Нет значимого пересечения
                    continue

                start_offset = segment_start_utc - clip_start_utc_dt_original
                end_offset = segment_end_utc - clip_start_utc_dt_original

                if start_offset.total_seconds() < 0: start_offset = datetime.timedelta(seconds=0)

                all_target_segments_for_composition.append({
                    'filename': clip_row['filename'],
                    'actual_segment_start_utc': segment_start_utc,  # Для сортировки
                    'actual_segment_end_utc': segment_end_utc,  # Для информации
                    'start_offset_in_clip': start_offset,
                    'end_offset_in_clip': end_offset,
                })
                logger.info(
                    f"    Добавлен сегмент из клипа {clip_row['filename']}: UTC {segment_start_utc.isoformat()} "
                    f"до {segment_end_utc.isoformat()} (смещения: {start_offset} до {end_offset}).")

        # Сортируем все найденные сегменты по их фактическому времени начала в UTC
        all_target_segments_for_composition.sort(key=lambda seg: seg['actual_segment_start_utc'])

        # ОПЦИОНАЛЬНО: Объединение перекрывающихся или смежных сегментов
        # Это более сложная логика. Пока пропустим, video_composer должен справиться с последовательной склейкой.
        # Если сегменты могут сильно перекрываться из-за разных объектов, это может привести к повторам.
        # Если задача - показать *любое* появление класса, то перекрытия нужно объединять.
        # Если для каждого объекта свой "фильм", то перекрытия не страшны.
        # Сейчас мы собираем все появления класса.

        if not all_target_segments_for_composition:
            logger.info(f"Не сформировано ни одного итогового сегмента для класса '{class_name_filter}'.")

        logger.info(
            f"ИТОГО сформировано {len(all_target_segments_for_composition)} индивидуальных сегментов для компоновки.")
        return all_target_segments_for_composition

    except Exception as e:
        logger.exception(f"Неожиданная ошибка при поиске ИНДИВИДУАЛЬНЫХ сегментов: {e}")
        return []
    finally:
        if conn:
            conn.close()

def ensure_aware_utc(dt_obj: datetime.datetime) -> datetime.datetime:
    """Делает naive datetime объект aware UTC, или конвертирует existing aware в UTC."""
    if dt_obj.tzinfo is None: # Если naive
        if ZoneInfo:
            return dt_obj.replace(tzinfo=ZoneInfo("UTC"))
        else:
            return pytz.utc.localize(dt_obj)
    else: # Уже aware
        return dt_obj.astimezone(ZoneInfo("UTC") if ZoneInfo else pytz.utc)