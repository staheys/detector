# gui.py
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import datetime
import logging

# Assuming db_api.py is in the same directory or Python path
import db_api
import configparser  # For direct config reading if needed, though db_api handles some

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
                self.update_status("Status: Streaming")  # Update status on good frame
                # Process and display frame (done in update_display typically)
            else:
                # Frame read failed or stream ended
                logger.warning(f"Failed to read frame from {self.stream_name}. Re-initializing capture.")
                if self.cap: self.cap.release()
                self.cap = None
                self.update_status("Status: Reconnecting...")
                self.draw_error_frame(f"Stream Interrupted\n{self.stream_name}")
                time.sleep(1)  # Brief pause before re-initializing in the loop
                continue  # Restart loop to re-initialize cap

            # Minimal delay to yield control, actual FPS is handled by camera/VideoCapture
            time.sleep(0.01)

        if self.cap:
            self.cap.release()
        logger.info(f"Thread finished for {self.stream_name}")
        self.update_status("Status: Stopped")
        self.draw_error_frame("Stream Stopped")

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
    def __init__(self):
        super().__init__()
        self.title("System Monitor GUI")
        self.geometry("1200x800")

        # Initialize DB and ensure tables exist
        try:
            db_api.initialize_database()
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
        self.notebook.add(self.tab_cameras, text='Live Feeds')
        self.setup_camera_tab()

        # Tab 2: Recorded Video Clips
        self.tab_clips = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_clips, text='Video Clips')
        self.setup_clips_tab()

        # Tab 3: Tracked Objects
        self.tab_objects = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_objects, text='Tracked Objects')
        self.setup_objects_tab()

        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        self.load_data_periodically()

    def setup_camera_tab(self):
        camera_frame_container = ttk.Frame(self.tab_cameras)
        camera_frame_container.pack(expand=True, fill='both', padx=5, pady=5)

        rtsp_urls_data = db_api.get_config_rtsp_urls()
        if not rtsp_urls_data:
            ttk.Label(camera_frame_container,
                      text="No camera streams configured in config.conf ([RTSP]url, [RECORDING]rtsp_url, or [GUI_CAMERAS]).").pack(
                padx=10, pady=10)
            return

        # Simple grid layout for cameras
        # Adjust num_cols as needed, e.g. based on number of cameras or desired layout
        num_cols = 2
        for i, cam_data in enumerate(rtsp_urls_data):
            row, col = divmod(i, num_cols)
            cam_widget = CameraStreamWidget(camera_frame_container, cam_data["name"], cam_data["url"], width=480,
                                            height=360)
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