# Smart Detector System

This project implements a smart detection system using RTSP video streams,
a YOLOv9-medium ONNX model for object detection, event-triggered and continuous
video archiving, and ZeroMQ for inter-module communication.

## Features

-   RTSP/Webcam video capture with resilience and auto-reconnect.
-   Object detection using a configurable ONNX model (YOLOv9-medium).
-   Configurable detection classes, confidence, and processing rate.
-   Continuous hourly video archiving.
-   Event-triggered video clip archiving (10-second clips with metadata).
-   Automatic archive maintenance (cleanup by age and disk space).
-   Modular design using threads and ZeroMQ `inproc` messaging.
-   Configuration via `config.yml`.
-   Logging to console and rotating files.
-   Placeholder live view using `cv2.imshow`.

## Setup

1.  **Clone/Download:** Get the project files.
2.  **Create `models` directory:**
    ```bash
    mkdir models
    ```
3.  **Place ONNX Model:** Download your `yolov9-medium.onnx` model and place it into the `smart-detector-project/models/` directory.
4.  **Configure `config.yml`:**
    *   Set `rtsp.url` to your camera's RTSP stream or `0` for a webcam.
    *   **Crucially, update `inference.classes_map` and `inference.active_class_ids` to match the classes and their integer IDs specific to YOUR `yolov9-medium.onnx` model.** The provided values are examples.
    *   Verify `inference.model_path` points to your model.
    *   Adjust other settings as needed (archive paths, logging levels, etc.).
5.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt # If you create one from pyproject.toml
    # OR directly:
    pip install ultralytics opencv-python pyzmq PyYAML rich numpy
    # For GPU support with ONNX, you might need:
    # pip install onnxruntime-gpu
    # And ensure CUDA/cuDNN are correctly set up. Then set `device: "cuda:0"` in config.yml.
    ```
    Alternatively, if you want to install the package itself (useful for the script):
    ```bash
    pip install .
    ```

## Running the Application

If you installed the package:
```bash
smart-detector --config config.yml