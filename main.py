# yolov8_stream.py
import argparse
import configparser
import time
from typing import List, Tuple, Dict
import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_requirements, check_yaml
import datetime  # For UTC timestamps
import logging  # For logging

import archive_manager
# Import custom modules
import db_api  # Our database API
from video_recorder import VideoRecorder

# Setup basic logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)  # Logger for this module


def load_config(path: str = "config.conf") -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if not cfg.read(path):
        logger.error(f"Configuration file {path} not found.")
        raise FileNotFoundError(f"Configuration file {path} not found.")
    logger.info(f"Configuration loaded from {path}")
    return cfg


# compute_iou and deduplicate remain the same
def compute_iou(box_a: List[int], box_b: List[int]) -> float:
    xa1, ya1, wa, ha = box_a
    xa2, ya2 = xa1 + wa, ya1 + ha
    xb1, yb1, wb, hb = box_b
    xb2, yb2 = xb1 + wb, yb1 + hb
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = iw * ih
    union = wa * ha + wb * hb - inter
    return inter / union if union else 0.0


def deduplicate(boxes: List[List[int]], scores: List[float], class_ids: List[int], iou_thresh: float):
    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    for i in idxs:
        if all(compute_iou(boxes[i], boxes[j]) < iou_thresh for j in keep):
            keep.append(i)
    return [boxes[i] for i in keep], [scores[i] for i in keep], [class_ids[i] for i in keep]


class CentroidTracker:
    def __init__(self, max_disappeared: int = 5, max_distance: float = 50.0,
                 db_api_module=None, class_names_map=None):
        self.next_id = 0
        # objects: { oid: (box, class_id, last_seen_utc_timestamp) }
        self.objects: Dict[int, Tuple[List[int], int, datetime.datetime]] = {}
        self.disappeared: Dict[int, int] = {}  # oid: consecutive_frames_disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.db_api = db_api_module
        self.class_names_map = class_names_map if class_names_map is not None else {}
        logger.info(f"CentroidTracker initialized. Max Disappeared: {max_disappeared}, Max Distance: {max_distance}")

    def register(self, box: List[int], cls_id: int):
        current_time_utc = datetime.datetime.utcnow()
        oid_to_register = self.next_id
        self.objects[oid_to_register] = (box, cls_id, current_time_utc)
        self.disappeared[oid_to_register] = 0

        if self.db_api:
            class_name = ""  # Initialize class_name
            # Check if cls_id is a valid index for self.class_names_map (which is a list)
            if self.class_names_map and 0 <= cls_id < len(self.class_names_map):
                class_name = self.class_names_map[cls_id]
            else:
                class_name = f"UnknownClassID_{cls_id}"
                logger.warning(
                    f"cls_id {cls_id} is out of bounds for class_names_map (length {len(self.class_names_map) if self.class_names_map else 0}). Using default name.")

            self.db_api.record_object_first_seen(oid_to_register, class_name, current_time_utc)
        # logger.debug(f"Registered new object ID {oid_to_register}")
        self.next_id += 1
        return oid_to_register

    def deregister(self, oid: int):
        # logger.debug(f"Deregistering object ID {oid}")
        # Get last known details before deleting from self.objects
        _, _, last_known_time_utc = self.objects.get(oid, (None, -1, datetime.datetime.utcnow()))

        if self.db_api:
            # Use the object's last recorded timestamp for disappearance time
            self.db_api.mark_object_disappeared(oid, last_known_time_utc)

        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, detected_boxes: List[List[int]], detected_class_ids: List[int]) -> Dict[
        int, Tuple[List[int], int, datetime.datetime]]:
        current_time_utc = datetime.datetime.utcnow()

        if not detected_boxes:  # No detections in the current frame
            for oid in list(self.disappeared.keys()):  # Iterate over a copy
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
                elif oid in self.objects:  # Object still exists, update its last_seen_utc (if it was seen recently)
                    # An object that is not detected but not yet deregistered will retain its last_seen_utc from its last detection.
                    # Only update its last_seen_utc if its box is updated.
                    # If we want to update last_seen even if not detected (but still tracked), this logic would change.
                    # For now, last_seen is updated only on a match.
                    pass
            return self.objects

        # If no objects are currently tracked, register all new detections
        if not self.objects:
            for i in range(len(detected_boxes)):
                self.register(detected_boxes[i], detected_class_ids[i])
            return self.objects

        # Prepare centroids for matching
        input_centroids = np.array([(box[0] + box[2] / 2, box[1] + box[3] / 2) for box in detected_boxes])

        existing_object_ids = list(self.objects.keys())
        existing_object_centroids = np.array(
            [(self.objects[oid][0][0] + self.objects[oid][0][2] / 2,
              self.objects[oid][0][1] + self.objects[oid][0][3] / 2)
             for oid in existing_object_ids]
        )

        # Compute distances between existing object centroids and new input centroids
        # D is shape (num_existing_objects, num_input_detections)
        if existing_object_centroids.size == 0 or input_centroids.size == 0:  # Should be caught by earlier checks
            if not existing_object_centroids.size and input_centroids.size > 0:  # No existing, but new inputs
                for i in range(len(detected_boxes)):
                    self.register(detected_boxes[i], detected_class_ids[i])
            # if existing but no new, handled by the "if not detected_boxes" block already.
            return self.objects

        D = np.linalg.norm(existing_object_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :], axis=2)

        # Greedily match: sort by smallest distance
        # `rows` are indices for existing_object_ids/centroids
        # `cols` are indices for input_centroids/detected_boxes
        # Sorted by distance for existing objects (rows)
        # This is a simple greedy assignment. For more complex scenarios, Hungarian algorithm could be used.

        # Find the minimum distance for each existing object (row)
        row_min_distances = D.min(axis=1)
        # Get the indices of existing objects sorted by their minimum distances
        sorted_row_indices = row_min_distances.argsort()
        # Get the column (input detection) index for each sorted existing object
        sorted_col_indices = D.argmin(axis=1)[sorted_row_indices]

        used_rows = set()  # Indices of existing_object_ids that have been matched
        used_cols = set()  # Indices of detected_boxes that have been matched

        for r_idx, c_idx in zip(sorted_row_indices, sorted_col_indices):
            if r_idx in used_rows or c_idx in used_cols:
                continue  # This existing object or input detection is already matched

            if D[r_idx, c_idx] > self.max_distance:
                continue  # Distance is too large

            oid = existing_object_ids[r_idx]
            self.objects[oid] = (detected_boxes[c_idx], detected_class_ids[c_idx], current_time_utc)
            self.disappeared[oid] = 0  # Reset disappeared counter

            if self.db_api:  # Update last seen time in DB
                self.db_api.update_object_last_seen(oid, current_time_utc)

            used_rows.add(r_idx)
            used_cols.add(c_idx)

        # Handle unmatched existing objects (potentially disappeared)
        unmatched_row_indices = set(range(len(existing_object_ids))) - used_rows
        for r_idx in unmatched_row_indices:
            oid = existing_object_ids[r_idx]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)
            # If not deregistered, its last_seen_utc remains from its last successful match.

        # Handle unmatched input boxes (new objects to register)
        unmatched_col_indices = set(range(len(detected_boxes))) - used_cols
        for c_idx in unmatched_col_indices:
            self.register(detected_boxes[c_idx], detected_class_ids[c_idx])

        return self.objects


class YOLOv8:
    def __init__(self, cfg: configparser.ConfigParser, db_api_module):  # Inject db_api
        yc = cfg['YOLO'];
        ic = cfg['INFERENCE']
        self.conf_thres = yc.getfloat('confidence_thres')
        self.iou_thres = yc.getfloat('iou_thres')
        self.frame_skip = max(1, yc.getint('frame_skip'))
        self.max_det = yc.getint('max_detections', fallback=100)
        self.dedupe_iou = yc.getfloat('dedupe_iou_thresh', fallback=0.5)

        self.db_api = db_api_module  # Store injected db_api

        try:
            yaml_path = check_yaml(yc.get('yaml_path'))
            self.classes = YAML.load(yaml_path)['names']
            logger.info(f"Loaded {len(self.classes)} classes from {yaml_path}")
        except Exception as e:
            logger.error(f"Error loading YAML class names from {yc.get('yaml_path')}: {e}")
            raise

        self.palette = np.random.uniform(0, 255, (len(self.classes), 3))

        # Initialize CentroidTracker with db_api and class_names map
        max_dis = cfg.getint('TRACKER', 'max_disappeared', fallback=30)  # Increased default
        max_dist = cfg.getfloat('TRACKER', 'max_distance', fallback=300.0)  # Increased default
        self.tracker = CentroidTracker(
            max_disappeared=max_dis,
            max_distance=max_dist,
            db_api_module=self.db_api,
            class_names_map=self.classes
        )

        model_path = yc.get('model_path')
        configured_provs_str = ic.get('providers', 'CPUExecutionProvider')  # Default to CPU if not specified
        available_ort_providers = ort.get_available_providers()

        provs_to_try = [p.strip() for p in configured_provs_str.split(',') if p.strip()]
        final_provs = []

        for prov_name in provs_to_try:
            if prov_name in available_ort_providers:
                if prov_name == 'CUDAExecutionProvider' and not torch.cuda.is_available():
                    logger.warning("CUDAExecutionProvider requested but CUDA not available. Skipping.")
                    continue
                final_provs.append(prov_name)
            else:
                logger.warning(
                    f"Requested provider '{prov_name}' not in available providers: {available_ort_providers}")

        if not final_provs:  # Fallback if no configured providers are suitable
            logger.warning("No suitable configured providers found. Attempting default fallback.")
            if 'CUDAExecutionProvider' in available_ort_providers and torch.cuda.is_available():
                final_provs = ['CUDAExecutionProvider']
            elif 'CPUExecutionProvider' in available_ort_providers:
                final_provs = ['CPUExecutionProvider']
            else:  # Should ideally not happen if onnxruntime is installed
                final_provs = available_ort_providers
                if not final_provs:
                    logger.critical("No ONNX Execution Provider found at all. Critical error.")
                    raise RuntimeError("No ONNX Execution Provider available. Check ONNXRuntime installation.")

        logger.info(f"Using ONNX Execution Providers: {final_provs}")
        try:

            self.session = ort.InferenceSession(model_path, providers=final_provs)

            inp = self.session.get_inputs()[0]
            self.in_name = inp.name
            self.w, self.h = int(inp.shape[2]), int(inp.shape[3])  # Model input dimensions
            logger.info(f"ONNX model {model_path} loaded. Input: {self.in_name}, Shape: (1, 3, {self.h}, {self.w})")
        except Exception as e:
            logger.error(f"Error loading ONNX model {model_path} with providers {final_provs}: {e}")
            raise

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int], float]:
        h0, w0 = img.shape[:2]  # Current image shape
        r = min(new_shape[0] / h0, new_shape[1] / w0)  # Resize ratio (new / old)

        # Compute padding
        new_unpad_w, new_unpad_h = int(round(w0 * r)), int(round(h0 * r))
        dw, dh = (new_shape[1] - new_unpad_w) / 2, (new_shape[0] - new_unpad_h) / 2  # wh padding

        if (w0, h0) != (new_unpad_w, new_unpad_h):  # Only resize if necessary
            img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, (top, left), r  # Return image, (padding_top, padding_left), ratio

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], int, int, float]:
        original_h, original_w = frame.shape[:2]
        # Letterbox expects new_shape as (height, width)
        img_letterboxed, pad_tl, ratio = self.letterbox(frame, (self.h, self.w))

        # Convert BGR to RGB, HWC to CHW, normalize to 0-1, add batch dimension
        img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
        img_tensor = np.transpose(img_rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
        return img_tensor, pad_tl, original_h, original_w, ratio

    def postprocess(self, frame_to_draw_on: np.ndarray, model_outputs: List[np.ndarray],
                    pad_tl: Tuple[int, int], original_h: int, original_w: int, ratio: float) -> np.ndarray:

        # model_outputs[0] shape is (1, num_classes + 4, num_detections) e.g. (1, 84, 8400) for YOLOv8n
        # Transpose to (1, num_detections, num_classes + 4) then squeeze
        predictions = np.squeeze(model_outputs[0].transpose(0, 2, 1))  # Shape (num_detections, num_classes + 4)

        pad_top, pad_left = pad_tl  # Unpack padding

        filtered_boxes = []
        filtered_scores = []
        filtered_class_ids = []

        for pred in predictions:  # Each pred is (xc, yc, w, h, class_scores...)
            box_coords_letterboxed = pred[:4]  # xc, yc, w, h in letterboxed image scale
            class_scores = pred[4:]

            max_score = np.max(class_scores)
            if max_score < self.conf_thres:
                continue

            class_id = np.argmax(class_scores)

            # Convert box from letterboxed scale to original image scale
            xc_lb, yc_lb, w_lb, h_lb = box_coords_letterboxed

            # 1. Remove padding effect (coordinates relative to unpadded resized image)
            x1_unpad_resized = xc_lb - w_lb / 2 - pad_left
            y1_unpad_resized = yc_lb - h_lb / 2 - pad_top
            # x2_unpad_resized = xc_lb + w_lb / 2 - pad_left # Not needed directly for x1,y1,w,h format
            # y2_unpad_resized = yc_lb + h_lb / 2 - pad_top

            # 2. Scale back to original image size
            x1_orig = int(x1_unpad_resized / ratio)
            y1_orig = int(y1_unpad_resized / ratio)
            w_orig = int(w_lb / ratio)
            h_orig = int(h_lb / ratio)

            # Clip to original image boundaries
            x1_orig = max(0, x1_orig)
            y1_orig = max(0, y1_orig)
            w_orig = min(original_w - x1_orig, w_orig)  # Ensure width doesn't exceed image bounds
            h_orig = min(original_h - y1_orig, h_orig)  # Ensure height doesn't exceed image bounds

            if w_orig > 0 and h_orig > 0:
                filtered_boxes.append([x1_orig, y1_orig, w_orig, h_orig])
                filtered_scores.append(float(max_score))
                filtered_class_ids.append(class_id)

        boxes_after_dedupe = []
        scores_after_dedupe = []
        class_ids_after_dedupe = []

        if filtered_boxes:
            # Optional deduplication (a simpler form of NMS, might be redundant if NMS is robust)
            # For now, let's assume deduplicate is useful for some edge cases or pre-filtering
            # If your model's NMS is very good, this might be skipped.
            boxes_after_dedupe, scores_after_dedupe, class_ids_after_dedupe = deduplicate(
                filtered_boxes, filtered_scores, filtered_class_ids, self.dedupe_iou
            )

        final_boxes_for_tracker = []
        final_class_ids_for_tracker = []

        if boxes_after_dedupe:
            # OpenCV DNN NMS (expects list of boxes, list of scores)
            # Returns indices of boxes to keep
            indices_to_keep = cv2.dnn.NMSBoxes(
                boxes_after_dedupe, scores_after_dedupe, self.conf_thres, self.iou_thres
            )

            # NMSBoxes returns a tuple of arrays if non-empty, or an empty tuple
            if isinstance(indices_to_keep, tuple) and len(indices_to_keep) > 0:
                indices_to_keep = indices_to_keep[0]  # Get the first array of indices
            elif not isinstance(indices_to_keep, np.ndarray):  # If it's an empty tuple or other non-array
                indices_to_keep = []

            if len(indices_to_keep) > 0:
                indices_to_keep_flat = np.array(indices_to_keep).flatten()
                if len(indices_to_keep_flat) > self.max_det:  # Apply max_detections limit
                    indices_to_keep_flat = indices_to_keep_flat[:self.max_det]

                for i in indices_to_keep_flat:
                    final_boxes_for_tracker.append(boxes_after_dedupe[i])
                    final_class_ids_for_tracker.append(class_ids_after_dedupe[i])

        # Update tracker with the final set of filtered and NMS'd boxes
        # tracked_objects_dict will be {oid: (box, cls_id, last_seen_utc)}
        tracked_objects_dict = self.tracker.update(final_boxes_for_tracker, final_class_ids_for_tracker)

        # Draw tracked objects on the frame
        for oid, (box, cls_id, _last_seen_time) in tracked_objects_dict.items():
            x, y, w_box, h_box = map(int, box)  # Ensure integer coordinates for drawing

            # Ensure class_id is within bounds of palette and classes
            safe_cls_id = cls_id % len(self.classes)
            color_val = self.palette[safe_cls_id]
            color = (int(color_val[0]), int(color_val[1]), int(color_val[2]))

            class_name = self.classes[safe_cls_id]
            label = f"ID {oid}:{class_name}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Label background rectangle
            label_bg_y1 = max(y - lh - baseline - 2, 0)  # Ensure not above frame
            # Bounding box rectangle
            cv2.rectangle(frame_to_draw_on, (x, label_bg_y1), (x + lw, y), color, thickness=cv2.FILLED)
            cv2.rectangle(frame_to_draw_on, (x, y), (x + w_box, y + h_box), color, thickness=2)
            # Label text (black for good contrast on colored background)
            cv2.putText(frame_to_draw_on, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                        thickness=1)

        return frame_to_draw_on

    def run_on_stream(self, stream_url: str):
        try:
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logger.critical(f"Failed to open stream \"{stream_url}\" for YOLO processing.")
                raise RuntimeError(f"Failed to open stream \"{stream_url}\" for YOLO processing.")
            logger.info(f"YOLO processing started for stream: {stream_url}")
        except Exception as e:
            logger.exception(f"Exception during VideoCapture initialization for {stream_url}")
            return  # Exit if cap can't be opened

        frame_id = 0
        window_name = "YOLOv8 Stream Processing"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning(
                        "Stream ended or frame could not be read for YOLO processing. Attempting to reconnect...")
                    cap.release()
                    time.sleep(5)  # Wait before retrying
                    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        logger.error("Failed to reconnect to stream. Exiting YOLO processing.")
                        break
                    logger.info("Reconnected to stream.")
                    continue

                display_frame = frame.copy()

                if frame_id % self.frame_skip == 0:
                    # Preprocess, Inference, Postprocess (which includes tracker update)
                    img_tensor, pad_tl, oh, ow, ratio = self.preprocess(frame)
                    model_outputs = self.session.run(None, {self.in_name: img_tensor})
                    display_frame = self.postprocess(display_frame, model_outputs, pad_tl, oh, ow, ratio)
                else:
                    # If skipping detection, still update tracker with empty detections
                    # This allows tracker to increment disappeared counters and deregister old objects.
                    # Then, draw currently tracked objects based on their last known positions.
                    current_tracked_objects = self.tracker.update([], [])
                    for oid, (box, cls_id, _last_seen) in current_tracked_objects.items():
                        x, y, w_box, h_box = map(int, box)
                        safe_cls_id = cls_id % len(self.classes)
                        color_val = self.palette[safe_cls_id]
                        color = (int(color_val[0]), int(color_val[1]), int(color_val[2]))
                        label = f"ID {oid}:{self.classes[safe_cls_id]}"
                        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        label_bg_y1 = max(y - lh - baseline - 2, 0)
                        cv2.rectangle(display_frame, (x, label_bg_y1), (x + lw, y), color, thickness=cv2.FILLED)
                        cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), color, thickness=2)
                        cv2.putText(display_frame, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                                    thickness=1)

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("'q' pressed, stopping YOLO stream processing.")
                    break
                elif key == 27:  # ESC key
                    logger.info("ESC pressed, stopping YOLO stream processing.")
                    break

                frame_id += 1
        except Exception as e:
            logger.exception("An error occurred during YOLO stream processing loop.")
        finally:
            if cap and cap.isOpened():
                cap.release()
            cv2.destroyWindow(window_name)  # Destroy only the specific window
            logger.info("YOLO stream processing resources released.")


if __name__ == "__main__":
    main_start_time = datetime.datetime.now()
    logger.info(f"Application starting at {main_start_time.isoformat()}")

    try:
        cfg = load_config()
    except FileNotFoundError:
        exit(1)  # load_config already logged the error
    except Exception as e:
        logger.exception(f"Failed to load configuration: {e}")
        exit(1)

    # --- Initialize DB API and database schema ---
    try:
        db_api.initialize_database()
        db_api.clear_active_objects_cache()  # Clear cache from any previous unclean shutdowns
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}. Exiting.")
        exit(1)
    # --- End DB API Setup ---
    archive_cleaner_main_instance = None
    try:
        recordings_path_main = cfg.get('RECORDING', 'output_path', fallback='./recordings/')
        default_days_main = cfg.getint('ARCHIVE', 'retention_days', fallback=30)
        default_size_gb_main = cfg.getfloat('ARCHIVE', 'max_size_gb', fallback=500.0)
        cleanup_interval_hours_main = cfg.getint('ARCHIVE', 'cleanup_interval_hours', fallback=1)

        archive_cleaner_main_instance = archive_manager.ArchiveCleaner(
            db_api_module=db_api,
            recordings_path=recordings_path_main,
            config_max_days=default_days_main,
            config_max_size_gb=default_size_gb_main
        )
        # Запуск периодической автоматической очистки
        archive_cleaner_main_instance.run_periodic_cleanup(interval_seconds=cleanup_interval_hours_main * 3600)
        logger.info("ArchiveCleaner (основной процесс) инициализирован и запущен периодический контроль архива.")
    except configparser.NoSectionError as e_cfg:
        logger.warning(
            f"Секция ARCHIVE или RECORDING не найдена в config.conf: {e_cfg}. Автоматическая очистка архива не будет запущена.")
    except configparser.NoOptionError as e_opt:
        logger.warning(
            f"Необходимая опция не найдена в config.conf для ArchiveCleaner: {e_opt}. Автоматическая очистка архива не будет запущена.")
    except Exception as e_ac:
        logger.error(f"Ошибка инициализации или запуска ArchiveCleaner (основной процесс): {e_ac}", exc_info=True)

    parser = argparse.ArgumentParser(
        description="YOLOv8 ONNX Stream Processing with Continuous Recording and DB Logging")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model file (overrides config)")
    parser.add_argument("--rtsp", type=str, default=None, help="RTSP URL for YOLO processing (overrides config)")
    parser.add_argument("--skip_recording", action='store_true', help="Skip video recording even if configured")
    args = parser.parse_args()

    if args.model:
        logger.info(f"Overriding model_path with command line arg: {args.model}")
        cfg['YOLO']['model_path'] = args.model
    if args.rtsp:
        logger.info(f"Overriding RTSP URL for YOLO with command line arg: {args.rtsp}")
        cfg['RTSP']['url'] = args.rtsp

    # Check Python package requirements for ONNX Runtime
    try:
        if torch.cuda.is_available():
            check_requirements("onnxruntime-gpu>=1.10")  # Specify a reasonable minimum version
        else:
            check_requirements("onnxruntime>=1.10")
        logger.info("ONNXRuntime requirement check passed.")
    except Exception as e:
        logger.warning(f"ONNXRuntime requirement check failed: {e}. Ensure a compatible version is installed.")
        # Not exiting, as YOLOv8 init will try to load it and fail more specifically if needed.

    # --- Video Recorder Setup ---
    recorder = None
    if not args.skip_recording:
        if 'RECORDING' in cfg and cfg['RECORDING'].get('rtsp_url'):
            try:
                logger.info("Initializing video recorder...")
                recorder = VideoRecorder(cfg, db_api_module=db_api)  # Pass the db_api module
                recorder.start_recording_session()
            except Exception as e:
                logger.exception("Failed to initialize or start video recorder.")
                recorder = None  # Ensure recorder is None if setup fails
        else:
            logger.info("RECORDING section or its rtsp_url not found/configured. Video recording will be skipped.")
    else:
        logger.info("Video recording skipped due to --skip_recording flag.")
    # --- End Video Recorder Setup ---

    yolo_processor = None
    try:
        logger.info("Initializing YOLOv8 processor...")
        yolo_processor = YOLOv8(cfg, db_api_module=db_api)  # Pass the db_api module

        yolo_rtsp_url = cfg.get('RTSP', 'url', fallback=None)
        if not yolo_rtsp_url:
            logger.critical("RTSP URL for YOLO processing not found in config ([RTSP] -> url).")
            raise ValueError("RTSP URL for YOLO processing not found in config.")

        yolo_processor.run_on_stream(yolo_rtsp_url)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C).")
    except RuntimeError as e:  # Catch specific runtime errors, e.g., ONNX model load
        logger.critical(f"Runtime Error during YOLO setup or processing: {e}", exc_info=True)
    except ValueError as e:  # Catch config errors
        logger.critical(f"Configuration or Value Error: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
    finally:
        shutdown_start_time = datetime.datetime.now()
        logger.info(f"Initiating application shutdown at {shutdown_start_time.isoformat()}...")

        if recorder and (
                recorder.is_recording_active or (recorder.recording_thread and recorder.recording_thread.is_alive())):
            logger.info("Signaling video recorder to stop...")
            recorder.stop_recording_session()  # This will block until thread joins or times out

        # Ensure tracker updates last_seen for any remaining active objects upon exit
        if yolo_processor and yolo_processor.tracker and yolo_processor.db_api:
            logger.info("Finalizing object tracking data before exit...")
            final_time_utc = datetime.datetime.utcnow()
            active_oids = list(
                yolo_processor.tracker.objects.keys())  # Get oids before cache is cleared by mark_object_disappeared
            if active_oids:
                logger.info(f"Marking {len(active_oids)} objects as disappeared at shutdown: {active_oids}")
            for oid in active_oids:
                yolo_processor.db_api.mark_object_disappeared(oid, final_time_utc)
            db_api.clear_active_objects_cache()  # Final clear of cache

        cv2.destroyAllWindows()  # Destroy any remaining OpenCV windows
        shutdown_duration = datetime.datetime.now() - shutdown_start_time
        logger.info(f"Application cleanup finished. Shutdown took {shutdown_duration.total_seconds():.2f}s.")
        total_run_time = datetime.datetime.now() - main_start_time
        logger.info(f"Total application runtime: {total_run_time.total_seconds():.2f}s.")