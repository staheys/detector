# db_api.py
import sqlite3
import datetime
import os
import logging
import threading

# Configure logging for the DB API
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

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
