# Dual Stream Smart Detector & Recorder

This application processes two independent RTSP video streams:

1.  **Display & Archive Stream (e.g., Channel 101):**
    *   Provides a clean, real-time video feed to the user.
    *   Records continuous video segments periodically (e.g., every 30 minutes) into an archive.
2.  **Detection Stream (e.g., Channel 102):**
    *   Used exclusively for running person detection with a YOLOv9 ONNX model.
    *   When a person is detected, a video clip is recorded from this detection stream.
    *   Detection bounding boxes are drawn on these event clips.
    *   A metadata JSON file is saved alongside each event clip.

All recorded videos are saved in AVI format.

## Features

-   Independent capture and processing of two RTSP streams.
-   Clean live video display from the primary stream.
-   Periodic continuous video archiving (AVI) from the primary stream.
-   Person detection on a secondary stream using a configurable ONNX model.
-   Event-triggered recording of person clips (AVI) from the secondary stream, with detections drawn.
-   Metadata (`.meta.json`) generation for person event clips.
-   Archive maintenance for all recorded clips.
-   Configuration via `config.yml`.

## Setup

1.  **Project Files & `models` Directory:** As before.
2.  **ONNX Model:** Place your model in `models/`.
3.  **Configure `config.yml`:**
    *   **Crucially, set `stream_display_archive.url` and `stream_detection.url` to your correct RTSP stream URLs.**
    *   Verify `inference.person_class_id`.
    *   Adjust recording settings, paths, FPS, resolutions for each stream.
4.  **Install Dependencies:** `pip install ultralytics opencv-python PyYAML rich numpy`
    (Or `pip install .` for the script)

## Running

If installed: `dual-stream-detector --config config.yml`
Or: `python -m src.smart_detector.cli --config config.yml`

Press `Ctrl+C` or 'q' in the display window to shut down.