# gui.py
import os
import subprocess
import tkinter as tk
from sys import platform
from tkinter import ttk, scrolledtext, messagebox, filedialog, HORIZONTAL
from typing import Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tzlocal import get_localzone
import archive_manager
import matplotlib


import cv2
import pytz
from PIL import Image, ImageTk
import threading
import time
import datetime
import logging

from tkcalendar import DateEntry

# Assuming db_api.py is in the same directory or Python path
import db_api
import configparser  # For direct config reading if needed, though db_api handles some

import video_composer
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    ZoneInfo = None # Если нет zoneinfo, будем полагаться на pytz
    ZoneInfoNotFoundError = None #

# Setup basic logging for the GUI module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("GUI")

# Global variable to signal camera threads to stop
stop_camera_threads = False


class CameraStreamWidget(ttk.Frame):
    def __init__(self, parent, stream_name, stream_url, width=640, height=480):
        super().__init__(parent)
        self.stream_name = stream_name
        self.stream_url = stream_url
        self.width = width
        self.height = height
        self.cap = None
        self.thread = None
        self.last_frame = None
        self.is_running = False

        self.label_name = ttk.Label(self, text=f"{self.stream_name}")
        self.label_name.pack(pady=2)

        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg="black")
        self.canvas.pack()
        self.canvas_image = None  # To keep a reference

        self.status_label = ttk.Label(self, text="Status: Initializing...")
        self.status_label.pack(pady=2)

        self.start_stream()

    def _stream_loop(self):
        global stop_camera_threads
        logger.info(f"Thread started for {self.stream_name} ({self.stream_url})")
        retry_delay = 5  # seconds
        max_retries = 3  # before showing a persistent error

        current_retries = 0
        while not stop_camera_threads and self.is_running:
            if self.cap is None or not self.cap.isOpened():
                if current_retries >= max_retries and max_retries > 0:
                    self.update_status(f"Error: Max retries exceeded. Stream offline.")
                    self.draw_error_frame(f"Stream Offline\n{self.stream_name}")
                    # Keep thread alive to potentially retry later if logic changes, or break
                    time.sleep(retry_delay * 5)  # Longer sleep if max retries hit
                    current_retries = 0  # Reset for next full attempt if desired
                    continue

                self.update_status(f"Status: Connecting (attempt {current_retries + 1})...")
                logger.info(f"Attempting to open stream: {self.stream_url} for {self.stream_name}")
                self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    logger.warning(f"Failed to open stream {self.stream_url} for {self.stream_name}")
                    self.cap = None
                    self.update_status(f"Status: Connection failed. Retrying in {retry_delay}s")
                    self.draw_error_frame(f"Connection Failed\n{self.stream_name}")
                    time.sleep(retry_delay)
                    current_retries += 1
                    continue
                else:
                    logger.info(f"Stream opened successfully: {self.stream_name}")
                    self.update_status("Status: Connected")
                    current_retries = 0  # Reset retries on successful connection

            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.last_frame = frame
                self.update_status("Статус: Эфир")  # Update status on good frame
                # Process and display frame (done in update_display typically)
            else:
                # Frame read failed or stream ended
                logger.warning(f"Failed to read frame from {self.stream_name}. Re-initializing capture.")
                if self.cap: self.cap.release()
                self.cap = None
                self.update_status("Статус: Переподключение...")
                self.draw_error_frame(f"Stream Interrupted\n{self.stream_name}")
                time.sleep(1)  # Brief pause before re-initializing in the loop
                continue  # Restart loop to re-initialize cap

            # Minimal delay to yield control, actual FPS is handled by camera/VideoCapture
            time.sleep(0.01)

        if self.cap:
            self.cap.release()
        logger.info(f"Thread finished for {self.stream_name}")
        self.update_status("Статус: Остановлено")
        self.draw_error_frame("Эфир остановлен")

    def start_stream(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.name = f"CamThread-{self.stream_name}"
            self.thread.start()
            self.update_display()  # Start the display update loop

    def stop_stream(self):
        self.is_running = False  # Signal thread to stop
        # stop_camera_threads will be set globally by the main app on close

    def update_status(self, message):
        if self.status_label.winfo_exists():
            self.status_label.config(text=message)

    def draw_error_frame(self, message):
        if not self.canvas.winfo_exists(): return
        # Create a black image with error text
        img = Image.new('RGB', (self.width, self.height), color='black')
        from PIL import ImageDraw  # Import here to avoid issues if PIL is not fully there
        draw = ImageDraw.Draw(img)
        try:
            # Attempt to load a font, fallback if not found
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        # Simple text centering
        text_bbox = draw.textbbox((0, 0), message, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (self.width - text_width) / 2
        text_y = (self.height - text_height) / 2
        draw.text((text_x, text_y), message, fill="white", font=font)

        self.photo = ImageTk.PhotoImage(image=img)
        if self.canvas_image:
            self.canvas.itemconfig(self.canvas_image, image=self.photo)
        else:
            self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def update_display(self):
        """Periodically updates the canvas with the latest frame."""
        if not self.is_running or not self.canvas.winfo_exists():
            return

        if self.last_frame is not None:
            try:
                frame_rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_resized = img.resize((self.width, self.height),
                                         Image.LANCZOS)  # Use Image.Resampling.LANCZOS in newer PIL
                self.photo = ImageTk.PhotoImage(image=img_resized)

                if self.canvas_image:
                    self.canvas.itemconfig(self.canvas_image, image=self.photo)
                else:
                    self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            except Exception as e:
                logger.error(f"Error updating display for {self.stream_name}: {e}")
                self.draw_error_frame(f"Display Error\n{self.stream_name}")

        # Schedule next update
        self.after(30, self.update_display)  # ~33 FPS target for GUI update


class App(tk.Tk):
    MATPLOTLIB_AVAILABLE = True
    def __init__(self):
        super().__init__()
        self.title("Наблюдение")
        self.geometry("1200x700")

        # Initialize DB and ensure tables exist
        try:
            db_api.initialize_database()
            self.recordings_path = "./recordings/"
            self.archive_cleaner_instance_gui = None  # Переименовал для ясности
            try:
                cfg_parser_gui = configparser.ConfigParser()
                if cfg_parser_gui.read("config.conf"):
                    self.recordings_path = cfg_parser_gui.get('RECORDING', 'output_path', fallback='./recordings/')
                    # Эти значения теперь в основном для отображения и ручного задания
                    default_days_gui = cfg_parser_gui.getint('ARCHIVE', 'retention_days', fallback=30)
                    default_size_gb_gui = cfg_parser_gui.getfloat('ARCHIVE', 'max_size_gb', fallback=500.0)
                    # cleanup_interval_hours больше не нужен GUI для автозапуска

                    self.archive_cleaner_instance_gui = archive_manager.ArchiveCleaner(
                        db_api_module=db_api,
                        recordings_path=self.recordings_path,
                        config_max_days=default_days_gui,  # Используется как начальное значение для слайдера
                        config_max_size_gb=default_size_gb_gui  # Используется для проверки лимита при ручной очистке
                    )
                    logger.info("ArchiveCleaner (GUI-экземпляр) инициализирован для ручного управления и отображения.")
                else:
                    logger.error("GUI: Не удалось прочитать config.conf для настроек архива (GUI-экземпляр).")
            except Exception as e:
                logger.error(f"GUI: Ошибка инициализации ArchiveCleaner (GUI-экземпляр): {e}")
        except Exception as e:
            logger.critical(f"GUI: Failed to initialize database: {e}")
            messagebox.showerror("Database Error", f"Could not initialize database: {e}\nApplication will exit.")
            self.destroy()
            return

        self.camera_widgets = []
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.notebook = ttk.Notebook(self)

        # Tab 1: Live Camera Feeds
        self.tab_cameras = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_cameras, text='Прямой эфир')
        self.setup_camera_tab()
        #
        # Tab 2: Recorded Video Clips
        # self.tab_clips = ttk.Frame(self.notebook)
        # self.notebook.add(self.tab_clips, text='Video Clips')
        # self.setup_clips_tab()
        #
        # Tab 3: Tracked Objects
        # self.tab_objects = ttk.Frame(self.notebook)
        # self.notebook.add(self.tab_objects, text='Tracked Objects')
        # self.setup_objects_tab()

        self.tab_composer = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_composer, text='Поиск объектов')
        self.setup_composer_tab()
        self.tab_archive = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_archive, text='Настрйока архива')
        self.setup_archive_tab()  # Новая функция для настройки вкладки

        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        self.load_data_periodically()

        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        self.load_data_periodically()

        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        self.load_data_periodically()

    def setup_camera_tab(self):
        camera_frame_container = ttk.Frame(self.tab_cameras)
        camera_frame_container.pack(expand=True, fill='both', padx=5, pady=5)

        rtsp_urls_data = db_api.get_config_rtsp_urls()
        rtsp_urls_data = [{'name': 'Entrance Camera', 'url': 'rtsp://admin:qwerty13579@192.168.0.3:554/Streaming/Channels/101'}]
        if not rtsp_urls_data:
            ttk.Label(camera_frame_container,
                      text="No camera streams configured in config.conf ([RTSP]url, [RECORDING]rtsp_url, or [GUI_CAMERAS]).").pack(
                padx=10, pady=10)
            return

        # Simple grid layout for cameras
        # Adjust num_cols as needed, e.g. based on number of cameras or desired layout
        num_cols = 1
        for i, cam_data in enumerate(rtsp_urls_data):
            row, col = divmod(i, num_cols)
            cam_widget = CameraStreamWidget(camera_frame_container, cam_data["name"], cam_data["url"], width=640,
                                            height=480)
            cam_widget.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            self.camera_widgets.append(cam_widget)
            camera_frame_container.grid_columnconfigure(col, weight=1)
            camera_frame_container.grid_rowconfigure(row, weight=1)

    def setup_clips_tab(self):
        clips_frame = ttk.LabelFrame(self.tab_clips, text="Recorded Video Clips")
        clips_frame.pack(expand=True, fill='both', padx=10, pady=10)

        cols = ("ID", "Filename", "Start Time (UTC)", "End Time (UTC)")
        self.clips_tree = ttk.Treeview(clips_frame, columns=cols, show='headings', selectmode="browse")
        for col_name in cols:
            self.clips_tree.heading(col_name, text=col_name)
            self.clips_tree.column(col_name, width=150, anchor='w')  # Adjust width as needed

        self.clips_tree.column("Filename", width=250)
        self.clips_tree.column("Start Time (UTC)", width=200)
        self.clips_tree.column("End Time (UTC)", width=200)

        vsb = ttk.Scrollbar(clips_frame, orient="vertical", command=self.clips_tree.yview)
        hsb = ttk.Scrollbar(clips_frame, orient="horizontal", command=self.clips_tree.xview)
        self.clips_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        self.clips_tree.pack(expand=True, fill='both')

        refresh_button = ttk.Button(clips_frame, text="Refresh Clips", command=self.load_clips_data)
        refresh_button.pack(pady=5)

        self.load_clips_data()

    def load_clips_data(self):
        logger.info("GUI: Refreshing video clips data.")
        for item in self.clips_tree.get_children():
            self.clips_tree.delete(item)

        clips_data = db_api.get_all_video_clips(limit=200)  # Fetch more if needed
        for clip in clips_data:
            # clip is a sqlite3.Row object, access by index or key
            self.clips_tree.insert("", "end", values=(
                clip["id"],
                clip["filename"],
                clip["start_time_utc"],
                clip["end_time_utc"] if clip["end_time_utc"] else "In Progress"
            ))

    def setup_objects_tab(self):
        objects_frame = ttk.LabelFrame(self.tab_objects, text="Tracked Objects (Recent)")
        objects_frame.pack(expand=True, fill='both', padx=10, pady=10)

        cols = ("DB ID", "Tracker ID", "Class Name", "First Seen (UTC)", "Last Seen (UTC)")
        self.objects_tree = ttk.Treeview(objects_frame, columns=cols, show='headings', selectmode="browse")
        for col_name in cols:
            self.objects_tree.heading(col_name, text=col_name)
            self.objects_tree.column(col_name, width=150, anchor='w')

        self.objects_tree.column("Class Name", width=100)
        self.objects_tree.column("First Seen (UTC)", width=200)
        self.objects_tree.column("Last Seen (UTC)", width=200)

        vsb = ttk.Scrollbar(objects_frame, orient="vertical", command=self.objects_tree.yview)
        hsb = ttk.Scrollbar(objects_frame, orient="horizontal", command=self.objects_tree.xview)
        self.objects_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        self.objects_tree.pack(expand=True, fill='both')

        refresh_button = ttk.Button(objects_frame, text="Refresh Objects", command=self.load_objects_data)
        refresh_button.pack(pady=5)

        self.load_objects_data()

    def load_objects_data(self):
        logger.info("GUI: Refreshing tracked objects data.")
        for item in self.objects_tree.get_children():
            self.objects_tree.delete(item)

        objects_data = db_api.get_recent_tracked_objects(limit=200)
        for obj in objects_data:
            self.objects_tree.insert("", "end", values=(
                obj["id"],
                obj["tracker_object_id"],
                obj["class_name"],
                obj["first_seen_utc"],
                obj["last_seen_utc"]
            ))

    def load_data_periodically(self):
        """Periodically refreshes data in non-camera tabs."""
        if self.notebook.index(self.notebook.select()) == 1:  # Clips tab is active
            self.load_clips_data()
        elif self.notebook.index(self.notebook.select()) == 2:  # Objects tab is active
            self.load_objects_data()

        self.after(30000, self.load_data_periodically)  # Refresh DB data every 30 seconds

    def setup_composer_tab(self):
        composer_main_frame = ttk.Frame(self.tab_composer)
        composer_main_frame.pack(padx=10, pady=10, fill='x')

        # --- Параметры выбора ---
        params_frame = ttk.LabelFrame(composer_main_frame, text="Parameters")
        params_frame.pack(fill='x', expand=True, pady=5)

        # Выбор класса
        ttk.Label(params_frame, text="Object Class:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.composer_class_var = tk.StringVar()
        self.composer_class_combo = ttk.Combobox(params_frame, textvariable=self.composer_class_var, state="readonly",
                                                 width=25)
        self.composer_class_combo.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        self.load_composer_classes()  # Загружаем классы

        # Выбор временного диапазона (локальное время)
        ttk.Label(params_frame, text="Timezone:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        initial_tz_name = "UTC (detecting...)"
        try:
            tz_obj_init = get_localzone()
            if hasattr(tz_obj_init, 'key'):  # ZoneInfo
                initial_tz_name = tz_obj_init.key
            elif hasattr(tz_obj_init, 'zone'):  # Pytz
                initial_tz_name = tz_obj_init.zone
            else:  # Неизвестный тип
                initial_tz_name = str(tz_obj_init) if tz_obj_init else "UTC (unknown type)"
        except Exception:
            pass  # initial_tz_name останется "UTC (detecting...)" или можно установить "UTC (detection failed)"
        self.composer_timezone_var = tk.StringVar(
            value=str(datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo))  # Текущая локальная таймзона
        self.composer_timezone_entry = ttk.Entry(params_frame, textvariable=self.composer_timezone_var, width=27,
                                                 state="readonly")
        self.composer_timezone_entry.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        ttk.Label(params_frame, text="Start DateTime:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.composer_start_date_entry = DateEntry(params_frame, width=12, background='darkblue', foreground='white',
                                                   borderwidth=2, date_pattern='yyyy-mm-dd')
        self.composer_start_date_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.composer_start_time_hour = ttk.Spinbox(params_frame, from_=0, to=23, width=3, format="%02.0f")
        self.composer_start_time_hour.grid(row=2, column=1, padx=(110, 0), pady=5, sticky='w')
        self.composer_start_time_min = ttk.Spinbox(params_frame, from_=0, to=59, width=3, format="%02.0f")
        self.composer_start_time_min.grid(row=2, column=1, padx=(160, 0), pady=5, sticky='w')
        # Установка текущего времени минус 1 час
        now = datetime.datetime.now()
        one_hour_ago = now - datetime.timedelta(hours=1)
        self.composer_start_date_entry.set_date(one_hour_ago.date())
        self.composer_start_time_hour.set(f"{one_hour_ago.hour:02}")
        self.composer_start_time_min.set(f"{one_hour_ago.minute:02}")

        ttk.Label(params_frame, text="End DateTime:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.composer_end_date_entry = DateEntry(params_frame, width=12, background='darkblue', foreground='white',
                                                 borderwidth=2, date_pattern='yyyy-mm-dd')
        self.composer_end_date_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        self.composer_end_time_hour = ttk.Spinbox(params_frame, from_=0, to=23, width=3, format="%02.0f")
        self.composer_end_time_hour.grid(row=3, column=1, padx=(110, 0), pady=5, sticky='w')
        self.composer_end_time_min = ttk.Spinbox(params_frame, from_=0, to=59, width=3, format="%02.0f")
        self.composer_end_time_min.grid(row=3, column=1, padx=(160, 0), pady=5, sticky='w')
        self.composer_end_date_entry.set_date(now.date())
        self.composer_end_time_hour.set(f"{now.hour:02}")
        self.composer_end_time_min.set(f"{now.minute:02}")

        params_frame.columnconfigure(1, weight=1)

        # --- Кнопка и статус ---
        action_frame = ttk.Frame(composer_main_frame)
        action_frame.pack(fill='x', expand=True, pady=5)

        self.compose_button = ttk.Button(action_frame, text="Create Composite Video",
                                         command=self.run_video_composition)
        self.compose_button.pack(side='left', padx=5)

        self.composer_status_label = ttk.Label(action_frame, text="Status: Ready")
        self.composer_status_label.pack(side='left', padx=5, fill='x', expand=True)

        # --- Результат ---
        result_frame = ttk.LabelFrame(composer_main_frame, text="Result")
        result_frame.pack(fill='both', expand=True, pady=5)
        self.result_text_area = scrolledtext.ScrolledText(result_frame, height=10, wrap=tk.WORD, state='disabled')
        self.result_text_area.pack(fill='both', expand=True, padx=5, pady=5)

        self.download_button = ttk.Button(result_frame, text="Download/Open Video", state='disabled',
                                          command=self.download_or_open_composed_video)
        self.download_button.pack(pady=5)
        self.composed_video_path = None

    def load_composer_classes(self):
        classes = db_api.get_available_classes_from_yaml()
        if classes:
            self.composer_class_combo['values'] = classes
            if classes:
                self.composer_class_combo.set(classes[0])  # Выбрать первый по умолчанию
        else:
            self.composer_class_combo['values'] = ["Error: No classes found"]
            self.composer_class_combo.set("Error: No classes found")
            logger.error("GUI Composer: Не удалось загрузить классы объектов.")

    def _set_composer_status(self, message, is_error=False):
        self.composer_status_label.config(text=f"Status: {message}",
                                          foreground='red' if is_error else 'black')
        logger.info(f"Composer Status: {message}")

    def _append_result_text(self, message):
        self.result_text_area.config(state='normal')
        self.result_text_area.insert(tk.END, message + "\n")
        self.result_text_area.see(tk.END)  # Прокрутка вниз
        self.result_text_area.config(state='disabled')

    def get_selected_datetime_utc(self, date_entry, hour_spinbox, min_spinbox) -> Optional[datetime.datetime]:
        try:
            date_val = date_entry.get_date()
            hour_val = int(hour_spinbox.get())
            min_val = int(min_spinbox.get())

            # Создаем наивный datetime объект из пользовательского ввода
            naive_local_dt = datetime.datetime(date_val.year, date_val.month, date_val.day, hour_val, min_val)

            aware_local_dt = None
            local_tz_name_for_display = "UTC (could not determine local)"

            try:
                local_tz_obj = get_localzone()  # Может вернуть zoneinfo.ZoneInfo или pytz.timezone

                if hasattr(local_tz_obj, 'key'):  # Это zoneinfo.ZoneInfo (IANA)
                    local_tz_name_for_display = local_tz_obj.key
                    aware_local_dt = naive_local_dt.replace(tzinfo=local_tz_obj)
                elif hasattr(local_tz_obj, 'zone'):  # Это pytz.timezone
                    local_tz_name_for_display = local_tz_obj.zone
                    aware_local_dt = local_tz_obj.localize(naive_local_dt, is_dst=None)
                else:  # Неожиданный тип объекта временной зоны
                    logger.warning(
                        f"tzlocal вернул неожиданный тип объекта зоны: {type(local_tz_obj)}. Используем UTC.")
                    # В этом случае self.composer_timezone_var.get() может не содержать имени зоны
                    # Лучше здесь явно присвоить UTC и обновить GUI, если возможно
                    if ZoneInfo:  # Предпочитаем встроенный ZoneInfo для UTC
                        aware_local_dt = naive_local_dt.replace(tzinfo=ZoneInfo("UTC"))
                    else:  # Fallback на pytz.utc
                        aware_local_dt = pytz.utc.localize(naive_local_dt)
                    local_tz_name_for_display = "UTC"

                logger.info(f"Определена локальная таймзона: {local_tz_name_for_display}")
                # Обновляем отображаемую таймзону в GUI, если это первый успешный вызов
                if self.composer_timezone_var.get() != local_tz_name_for_display and "UTC (could not detect local)" in self.composer_timezone_var.get():
                    self.composer_timezone_var.set(local_tz_name_for_display)

            except Exception as e_tz:
                logger.error(
                    f"Не удалось определить/использовать локальную таймзону: {e_tz}. Предполагается ввод в UTC.",
                    exc_info=True)
                self._set_composer_status("Warning: Could not determine local timezone. Assuming input is UTC.",
                                          is_error=True)
                if ZoneInfo:
                    aware_local_dt = naive_local_dt.replace(tzinfo=ZoneInfo("UTC"))
                else:
                    aware_local_dt = pytz.utc.localize(naive_local_dt)
                local_tz_name_for_display = "UTC"
                self.composer_timezone_var.set(local_tz_name_for_display)  # Обновить GUI

            if aware_local_dt is None:  # Если не удалось создать aware_local_dt
                raise ValueError("Не удалось создать aware datetime из локального времени.")

            # Конвертируем в UTC
            if ZoneInfo:  # Предпочитаем конвертацию с использованием ZoneInfo, если возможно
                dt_utc = aware_local_dt.astimezone(ZoneInfo("UTC"))
            else:  # Fallback на pytz.utc
                dt_utc = aware_local_dt.astimezone(pytz.utc)

            logger.info(
                f"Локальное время {naive_local_dt} ({local_tz_name_for_display}) конвертировано в UTC: {dt_utc.isoformat()}")
            return dt_utc

        except Exception as e:
            logger.error(f"Ошибка конвертации времени в UTC: {e}", exc_info=True)
            self._set_composer_status(f"Error parsing datetime: {e}", is_error=True)
            messagebox.showerror("Input Error", f"Invalid date/time input or timezone issue: {e}")
            return None

    def setup_archive_tab(self, MATPLOTLIB_AVAILABLE=True):
        archive_main_frame = ttk.Frame(self.tab_archive)
        archive_main_frame.pack(padx=10, pady=10, fill='both', expand=True)

        # --- Диаграмма использования диска (если Matplotlib доступен) ---
        self.disk_usage_figure_canvas = None
        if MATPLOTLIB_AVAILABLE:
            disk_frame = ttk.LabelFrame(archive_main_frame, text="Disk Usage")
            disk_frame.pack(pady=10, padx=10, fill='x')

            fig = Figure(figsize=(5, 3), dpi=100)  # Уменьшил размер для компактности
            self.ax_disk_usage = fig.add_subplot(111)
            self.ax_disk_usage.axis('equal')  # Для круговой диаграммы

            self.disk_usage_figure_canvas = FigureCanvasTkAgg(fig, master=disk_frame)
            self.disk_usage_figure_canvas.draw()
            self.disk_usage_figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        else:
            self.disk_text_info_var = tk.StringVar(value="Disk usage info unavailable (Matplotlib not found).")
            ttk.Label(archive_main_frame, textvariable=self.disk_text_info_var).pack(pady=10)

        # --- Настройки хранения ---
        settings_frame = ttk.LabelFrame(archive_main_frame, text="Storage Settings")
        settings_frame.pack(pady=10, padx=10, fill='x')

        ttk.Label(settings_frame, text="Retention Depth (days):").grid(row=0, column=0, padx=5, pady=5, sticky='w')

        self.retention_days_var = tk.IntVar()
        if self.archive_cleaner_instance_gui:
            self.retention_days_var.set(self.archive_cleaner_instance_gui.max_days)
        else:
            self.retention_days_var.set(30)  # Значение по умолчанию, если cleaner не инициализирован

        self.retention_slider = tk.Scale(settings_frame, from_=7, to_=60, orient=HORIZONTAL,
                                         variable=self.retention_days_var, length=300,
                                         command=self.on_retention_slider_change_display_only)  # Только отображение
        self.retention_slider.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.retention_days_label_val = ttk.Label(settings_frame, text=f"{self.retention_days_var.get()} days")
        self.retention_days_label_val.grid(row=0, column=2, padx=5, pady=5)

        apply_button = ttk.Button(settings_frame, text="Apply Retention & Clean",
                                  command=self.apply_and_clean_archive_manual)
        apply_button.grid(row=1, column=0, columnspan=3, pady=10)

        # --- Статус очистки ---
        self.archive_status_label = ttk.Label(archive_main_frame, text="Archive Status: Ready")
        self.archive_status_label.pack(pady=10, fill='x')

        self.update_disk_usage_display()  # Первоначальное отображение

    def on_retention_slider_change_display_only(self, value_str):
        # value_str приходит как строка от Scale
        self.retention_days_label_val.config(text=f"{int(value_str)} days")

    def apply_and_clean_archive_manual_thread(self):
        if not self.archive_cleaner_instance_gui:
            self.archive_status_label.config(text="Archive Status: Cleaner not initialized.")
            messagebox.showerror("Error", "Archive cleaner is not available.")
            return

        new_days = self.retention_days_var.get()
        self.archive_cleaner_instance_gui.set_retention_days(new_days)
        self.archive_status_label.config(
            text=f"Archive Status: Applying new retention ({new_days} days) and cleaning...")

        try:
            self.apply_retention_button.config(state="disabled")  # Если кнопка была бы отдельной
            # В нашем случае, apply_button - это кнопка, которая вызывает этот метод.
            # Если бы это была кнопка "Apply", то её бы блокировали.
            # Сейчас мы блокируем кнопку, которая запустила этот поток.
            # Найдем ее по имени или передадим как аргумент. Для простоты предположим, что она одна.
            # Можно найти виджет по имени, если оно задано, или передать self.tab_archive.winfo_children() и найти.
            # Для простоты, будем управлять активностью кнопки через состояние потока.

            logger.info(f"Ручной запуск очистки архива с новым сроком хранения: {new_days} дней.")
            deleted_count, freed_mb = self.archive_cleaner_instance_gui.cleanup_archive()
            freed_gb = freed_mb / (1024 ** 3)

            result_message = f"Cleanup complete. Deleted: {deleted_count} files, Freed: {freed_gb:.2f} GB."
            self.archive_status_label.config(text=f"Archive Status: {result_message}")
            logger.info(result_message)
            self.update_disk_usage_display()  # Обновить диаграмму
        except Exception as e:
            error_msg = f"Error during manual archive cleanup: {e}"
            self.archive_status_label.config(text=f"Archive Status: {error_msg}")
            logger.error(error_msg, exc_info=True)
        finally:
            # Разблокировать кнопку (нужно найти кнопку и сделать .config(state="normal"))
            # Это будет сделано в apply_and_clean_archive_manual после thread.start()
            # или через проверку is_alive, но лучше передавать кнопку или использовать флаг.
            # Пока оставим так, кнопка разблокируется после завершения потока в основном методе.
            pass

    def on_retention_slider_change_display_only(self, value_str):
        # value_str приходит как строка от Scale
        self.retention_days_label_val.config(text=f"{int(value_str)} days")

    def apply_and_clean_archive_manual_thread(self):
        if not self.archive_cleaner_instance_gui:
            self.archive_status_label.config(text="Archive Status: Cleaner not initialized.")
            messagebox.showerror("Error", "Archive cleaner is not available.")
            return

        new_days = self.retention_days_var.get()
        self.archive_cleaner_instance_gui.set_retention_days(new_days)
        self.archive_status_label.config(
            text=f"Archive Status: Applying new retention ({new_days} days) and cleaning...")

        try:
            self.apply_retention_button.config(state="disabled")  # Если кнопка была бы отдельной
            # В нашем случае, apply_button - это кнопка, которая вызывает этот метод.
            # Если бы это была кнопка "Apply", то её бы блокировали.
            # Сейчас мы блокируем кнопку, которая запустила этот поток.
            # Найдем ее по имени или передадим как аргумент. Для простоты предположим, что она одна.
            # Можно найти виджет по имени, если оно задано, или передать self.tab_archive.winfo_children() и найти.
            # Для простоты, будем управлять активностью кнопки через состояние потока.

            logger.info(f"Ручной запуск очистки архива с новым сроком хранения: {new_days} дней.")
            deleted_count, freed_mb = self.archive_cleaner_instance_gui.cleanup_archive()
            freed_gb = freed_mb / (1024 ** 3)

            result_message = f"Cleanup complete. Deleted: {deleted_count} files, Freed: {freed_gb:.2f} GB."
            self.archive_status_label.config(text=f"Archive Status: {result_message}")
            logger.info(result_message)
            self.update_disk_usage_display()  # Обновить диаграмму
        except Exception as e:
            error_msg = f"Error during manual archive cleanup: {e}"
            self.archive_status_label.config(text=f"Archive Status: {error_msg}")
            logger.error(error_msg, exc_info=True)
        finally:
            # Разблокировать кнопку (нужно найти кнопку и сделать .config(state="normal"))
            # Это будет сделано в apply_and_clean_archive_manual после thread.start()
            # или через проверку is_alive, но лучше передавать кнопку или использовать флаг.
            # Пока оставим так, кнопка разблокируется после завершения потока в основном методе.
            pass

    def apply_and_clean_archive_manual(self):
        # Найти кнопку и заблокировать ее
        # Это упрощенный пример, в реальном коде нужно быть аккуратнее с поиском виджетов
        # или передавать ссылку на кнопку.
        button_to_disable = None
        for widget_toplevel in self.tab_archive.winfo_children():  # Ищем LabelFrame
            if isinstance(widget_toplevel, ttk.LabelFrame) and "Storage Settings" in widget_toplevel.cget("text"):
                for widget_settings in widget_toplevel.winfo_children():  # Ищем кнопку внутри LabelFrame
                    if isinstance(widget_settings, ttk.Button) and "Apply Retention & Clean" in widget_settings.cget(
                            "text"):
                        button_to_disable = widget_settings
                        break
                if button_to_disable: break

        if button_to_disable:
            button_to_disable.config(state="disabled")

        cleanup_thread = threading.Thread(target=self.apply_and_clean_archive_manual_thread, daemon=True)
        cleanup_thread.name = "ManualArchiveCleanupThread"
        cleanup_thread.start()

        # Следим за потоком, чтобы разблокировать кнопку
        self.after(100, self.check_manual_cleanup_thread_status, cleanup_thread, button_to_disable)

    def check_manual_cleanup_thread_status(self, thread, button):
        if thread.is_alive():
            self.after(100, self.check_manual_cleanup_thread_status, thread, button)
        else:
            if button and button.winfo_exists():
                button.config(state="normal")
            logger.info("Поток ручной очистки завершен, кнопка разблокирована.")

    def update_disk_usage_display(self, MATPLOTLIB_AVAILABLE=True):
        if not self.archive_cleaner_instance_gui:
            logger.warning("ArchiveCleaner не инициализирован, не могу обновить использование диска.")
            if not MATPLOTLIB_AVAILABLE:
                self.disk_text_info_var.set("Disk usage: Archive cleaner not available.")
            return

        total_b, used_b_overall, free_b_overall = archive_manager.get_disk_usage(self.recordings_path)

        # Размер архива (занято нашими .avi файлами)
        archive_size_b = self.archive_cleaner_instance_gui.get_current_archive_size()

        # Зарезервированное место под архив (исходя из max_size_bytes)
        # Это не совсем "зарезервированное" системой, а скорее наш целевой максимум.
        # Если max_size_bytes больше, чем free_b_overall + archive_size_b, то у нас проблемы с местом.
        # Для диаграммы покажем: archive_size_b, (max_size_bytes - archive_size_b) -> как "планируемое свободное в рамках лимита", и остальное.

        # Более простая интерпретация для диаграммы:
        # 1. Место, занятое архивом (archive_size_b)
        # 2. Место, занятое ДРУГИМИ файлами на этом же диске/разделе (used_b_overall - archive_size_b)
        # 3. Свободное место на диске (free_b_overall)

        if total_b == 0:  # Не удалось получить инфо о диске
            if MATPLOTLIB_AVAILABLE and self.disk_usage_figure_canvas:
                self.ax_disk_usage.clear()
                self.ax_disk_usage.text(0.5, 0.5, "Disk info error", ha='center', va='center')
                self.disk_usage_figure_canvas.draw()
            elif not MATPLOTLIB_AVAILABLE:
                self.disk_text_info_var.set(f"Total: N/A, Used by archive: {archive_size_b / (1024 ** 3):.2f} GB")
            return

        occupied_by_archive_gb = archive_size_b / (1024 ** 3)
        # used_b_overall включает archive_size_b
        occupied_by_other_gb = (used_b_overall - archive_size_b) / (1024 ** 3)
        if occupied_by_other_gb < 0: occupied_by_other_gb = 0  # На случай погрешностей или если нет других файлов

        free_overall_gb = free_b_overall / (1024 ** 3)
        total_gb = total_b / (1024 ** 3)

        if MATPLOTLIB_AVAILABLE and self.disk_usage_figure_canvas:
            self.ax_disk_usage.clear()
            labels = [f'Archive ({occupied_by_archive_gb:.2f} GB)',
                      f'Other Used ({occupied_by_other_gb:.2f} GB)',
                      f'Free ({free_overall_gb:.2f} GB)']
            sizes = [archive_size_b, used_b_overall - archive_size_b, free_b_overall]
            # Убираем отрицательные или нулевые значения для корректного отображения
            valid_sizes = [s for s in sizes if s > 0]
            valid_labels = [labels[i] for i, s in enumerate(sizes) if s > 0]

            if valid_sizes:
                self.ax_disk_usage.pie(valid_sizes, labels=valid_labels, autopct='%1.1f%%', startangle=90,
                                       wedgeprops=dict(width=0.4, edgecolor='w'))  # Кольцевая диаграмма
                self.ax_disk_usage.set_title(f"Disk Usage (Total: {total_gb:.2f} GB)", fontsize=10)
            else:
                self.ax_disk_usage.text(0.5, 0.5, "No disk data to show", ha='center', va='center')

            self.ax_disk_usage.axis('equal')
            self.disk_usage_figure_canvas.draw()
        elif not MATPLOTLIB_AVAILABLE:
            self.disk_text_info_var.set(
                f"Total: {total_gb:.2f} GB, Used by Archive: {occupied_by_archive_gb:.2f} GB, "
                f"Other Used: {occupied_by_other_gb:.2f} GB, Free: {free_overall_gb:.2f} GB"
            )

        # Обновляем периодически (например, при обновлении других данных или по таймеру)
        # Сейчас это вызывается после ручной очистки и при инициализации вкладки.

    def load_data_periodically(self):
        """Periodically refreshes data in non-camera tabs."""
        active_tab_index = -1
        try:
            active_tab_index = self.notebook.index(self.notebook.select())
        except tk.TclError:  # Если вкладка еще не выбрана или notebook уничтожен
            pass

        if active_tab_index == 1:  # Clips tab
            self.load_clips_data()
        elif active_tab_index == 2:  # Objects tab
            self.load_objects_data()
        elif active_tab_index == 4:  # Archive Management tab
            self.update_disk_usage_display()  # Обновляем диаграмму, если она активна

        self.after(15000, self.load_data_periodically)  # Уменьшил интервал для диска

    def on_closing(self):
        global stop_camera_threads
        logger.info("GUI: Close button pressed. Shutting down threads.")
        stop_camera_threads = True

        if self.archive_cleaner_instance_gui:
            self.archive_cleaner_instance_gui.stop_periodic_cleanup()

        # ... (остальная часть on_closing для камер) ...
        self.destroy()
    
    def run_video_composition_thread(self, class_filter, start_utc, end_utc):
        try:
            self.compose_button.config(state='disabled')
            self.download_button.config(state='disabled')
            self.composed_video_path = None
            self.result_text_area.config(state='normal')
            self.result_text_area.delete(1.0, tk.END)  # Очистить предыдущие результаты
            self.result_text_area.config(state='disabled')

            self._set_composer_status(
                f"Searching for '{class_filter}' from {start_utc.strftime('%Y-%m-%d %H:%M')} to {end_utc.strftime('%Y-%m-%d %H:%M')} UTC...")
            self._append_result_text(f"Searching for object class: '{class_filter}'")
            self._append_result_text(f"Time range (UTC): {start_utc.isoformat()} to {end_utc.isoformat()}")

            video_segments = db_api.find_object_occurrences_and_video_segments(class_filter, start_utc, end_utc)

            if not video_segments:
                self._set_composer_status(f"No relevant video segments found for '{class_filter}'.", is_error=True)
                self._append_result_text("No video segments found matching criteria.")
                return

            self._append_result_text(f"Found {len(video_segments)} video segments to process.")
            for i, seg in enumerate(video_segments):
                self._append_result_text(
                    f"  Segment {i + 1}: {seg['filename']} from {seg['start_offset_in_clip']} to {seg['end_offset_in_clip']}")

            self._set_composer_status("Compositing video... This may take a while.")
            # Имя файла для композитного видео
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_class_name = "".join(
                c if c.isalnum() else "_" for c in class_filter)  # Делаем имя класса безопасным для файла
            output_filename = f"composite_{safe_class_name}_{timestamp_str}.avi"

            # Получаем путь к записям из config.conf (лучше бы через db_api, но для примера)
            cfg = configparser.ConfigParser()
            cfg.read("config.conf")
            recordings_path = cfg.get('RECORDING', 'output_path', fallback='./recordings/')

            # Запуск компоновки в самом VideoComposer
            # video_composer.create_composite_video_ffmpeg(...) # Если бы использовали ffmpeg
            self.composed_video_path = video_composer.create_composite_video_opencv(
                video_segments,
                output_filename,
                recordings_path
            )

            if self.composed_video_path and os.path.exists(self.composed_video_path):
                self._set_composer_status("Video composition successful!")
                self._append_result_text(f"Composite video created: {self.composed_video_path}")
                self.download_button.config(state='normal')
            else:
                self._set_composer_status("Video composition failed.", is_error=True)
                self._append_result_text("Failed to create composite video. Check logs.")

        except Exception as e:
            self._set_composer_status(f"Error during composition: {e}", is_error=True)
            self._append_result_text(f"ERROR: {e}")
            logger.exception("GUI: Error in video composition thread")
        finally:
            if self.compose_button.winfo_exists():  # Проверка, что виджет еще существует
                self.compose_button.config(state='normal')

    def run_video_composition(self):
        class_filter = self.composer_class_var.get()
        if not class_filter or "Error:" in class_filter:
            messagebox.showerror("Input Error", "Please select a valid object class.")
            return

        start_utc = self.get_selected_datetime_utc(self.composer_start_date_entry, self.composer_start_time_hour,
                                                   self.composer_start_time_min)
        end_utc = self.get_selected_datetime_utc(self.composer_end_date_entry, self.composer_end_time_hour,
                                                 self.composer_end_time_min)

        if not start_utc or not end_utc:
            return  # Ошибка уже показана в get_selected_datetime_utc

        if start_utc >= end_utc:
            messagebox.showerror("Input Error", "Start DateTime must be before End DateTime.")
            return

        # Запускаем тяжелую операцию в отдельном потоке, чтобы не блокировать GUI
        composition_thread = threading.Thread(
            target=self.run_video_composition_thread,
            args=(class_filter, start_utc, end_utc),
            daemon=True
        )
        composition_thread.name = "VideoCompositionThread"
        composition_thread.start()

    def download_or_open_composed_video(self):
        if self.composed_video_path and os.path.exists(self.composed_video_path):
            try:
                if platform.system() == "Windows":
                    os.startfile(self.composed_video_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.call(["open", self.composed_video_path])
                else:  # Linux and other UNIX-like
                    subprocess.call(["xdg-open", self.composed_video_path])
                self._append_result_text(f"Attempting to open video: {self.composed_video_path}")
            except Exception as e:
                logger.error(f"Failed to open video automatically: {e}")
                self._append_result_text(f"Could not open video automatically. Path: {self.composed_video_path}")
                # Предложить сохранить как...
                save_path = filedialog.asksaveasfilename(
                    initialdir=os.path.dirname(self.composed_video_path),  # Начать с директории, где он уже есть
                    initialfile=os.path.basename(self.composed_video_path),
                    defaultextension=".avi",
                    filetypes=[("AVI videos", "*.avi"), ("All files", "*.*")]
                )
                if save_path:
                    try:
                        import shutil
                        shutil.copy2(self.composed_video_path, save_path)
                        self._append_result_text(f"Video copied to: {save_path}")
                        messagebox.showinfo("Download Complete", f"Video saved to: {save_path}")
                    except Exception as ex_copy:
                        logger.error(f"Error copying video to {save_path}: {ex_copy}")
                        messagebox.showerror("Copy Error", f"Failed to copy video: {ex_copy}")
        else:
            messagebox.showwarning("No Video", "No composite video available to download/open.")

    def on_closing(self):
        global stop_camera_threads
        logger.info("GUI: Close button pressed. Shutting down camera threads.")
        stop_camera_threads = True  # Signal all camera threads to stop

        # Wait for threads to finish (optional, can make closing slower)
        # for cam_widget in self.camera_widgets:
        #     if cam_widget.thread and cam_widget.thread.is_alive():
        #         logger.info(f"GUI: Waiting for thread {cam_widget.thread.name} to join.")
        #         cam_widget.thread.join(timeout=2) # Short timeout
        #         if cam_widget.thread.is_alive():
        #             logger.warning(f"GUI: Thread {cam_widget.thread.name} did not stop in time.")

        logger.info("GUI: Destroying window.")
        self.destroy()


if __name__ == "__main__":
    # Ensure PIL (Pillow) is installed: pip install Pillow
    # Ensure OpenCV is installed: pip install opencv-python
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        logger.critical(f"GUI: Unhandled exception in main GUI loop: {e}", exc_info=True)
        messagebox.showerror("Critical Error", f"A critical error occurred: {e}\nSee logs for details.")
    finally:
        logger.info("GUI application finished.")