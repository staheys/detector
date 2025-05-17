# yolov8_stream.py
import argparse
import configparser
from typing import List, Tuple, Dict
import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_requirements, check_yaml


def load_config(path: str = "config.conf") -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


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
    def __init__(self, max_disappeared: int = 5, max_distance: float = 50.0):
        self.next_id = 0
        self.objects: Dict[int, Tuple[List[int], int]] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, box: List[int], cls: int):
        self.objects[self.next_id] = (box, cls)
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid: int):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, boxes: List[List[int]], class_ids: List[int]) -> Dict[int, Tuple[List[int], int]]:
        if not boxes:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.array([(x + w/2, y + h/2) for x, y, w, h in boxes])
        if not self.objects:
            for box, cls in zip(boxes, class_ids):
                self.register(box, cls)
            return self.objects

        oids = list(self.objects.keys())
        obj_data = list(self.objects.values())
        obj_centroids = np.array([(b[0] + b[2]/2, b[1] + b[3]/2) for b, _ in obj_data])

        D = np.linalg.norm(obj_centroids[:, None] - input_centroids[None, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols or D[r, c] > self.max_distance:
                continue
            oid = oids[r]
            self.objects[oid] = (boxes[c], class_ids[c])
            self.disappeared[oid] = 0
            used_rows.add(r)
            used_cols.add(c)

        unused_o = set(range(len(oids))) - used_rows
        for r in unused_o:
            oid = oids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        unused_i = set(range(len(boxes))) - used_cols
        for i in unused_i:
            self.register(boxes[i], class_ids[i])

        return self.objects


class YOLOv8:
    def __init__(self, cfg: configparser.ConfigParser):
        yc = cfg['YOLO']; ic = cfg['INFERENCE']
        self.conf_thres = yc.getfloat('confidence_thres')
        self.iou_thres = yc.getfloat('iou_thres')
        self.frame_skip = yc.getint('frame_skip')
        self.max_det = yc.getint('max_detections', fallback=100)
        self.dedupe_iou = yc.getfloat('dedupe_iou_thresh', fallback=0.5)

        max_dis = cfg.getint('TRACKER', 'max_disappeared', fallback=5)
        max_dist = cfg.getfloat('TRACKER', 'max_distance', fallback=50.0)
        self.tracker = CentroidTracker(max_disappeared=max_dis, max_distance=max_dist)

        self.classes = YAML.load(check_yaml(yc.get('yaml_path')))['names']
        self.palette = np.random.uniform(0, 255, (len(self.classes), 3))

        model_path = yc.get('model_path')
        provs = [p for p in ic.get('default_providers', '').split(',') if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(model_path, providers=provs or ['CPUExecutionProvider'])
        inp = self.session.get_inputs()[0]
        self.in_name = inp.name; self.w, self.h = inp.shape[2], inp.shape[3]

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
        h0, w0 = img.shape[:2]
        r = min(new_shape[0] / h0, new_shape[1] / w0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        img = cv2.resize(img, new_unpad, cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, (top, left)

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], int, int]:
        oh, ow = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img, pad = self.letterbox(img, (self.w, self.h))
        arr = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))[None]
        return arr, pad, oh, ow

    def postprocess(self, frame: np.ndarray, outs: List[np.ndarray], pad: Tuple[int, int], oh: int, ow: int) -> np.ndarray:
        preds = np.squeeze(outs[0]).T
        gain = min(self.h / oh, self.w / ow)
        px, py = pad[1], pad[0]
        raw_b, raw_s, raw_c = [], [], []
        for d in preds:
            score = float(d[4:].max())
            if score < self.conf_thres:
                continue
            cls_id = int(d[4:].argmax())
            x_c, y_c, bw, bh = d[:4]
            x1 = int((x_c - bw / 2 - px) / gain)
            y1 = int((y_c - bh / 2 - py) / gain)
            raw_b.append([x1, y1, int(bw / gain), int(bh / gain)])
            raw_s.append(score); raw_c.append(cls_id)
        bboxes, scores, class_ids = deduplicate(raw_b, raw_s, raw_c, self.dedupe_iou)
        idxs = cv2.dnn.NMSBoxes(bboxes, scores, self.conf_thres, self.iou_thres)
        if isinstance(idxs, tuple) and idxs:
            idxs = idxs[0]
        if idxs is None or len(idxs) == 0:
            tracked = self.tracker.update([], [])
        else:
            idxs_flat = np.array(idxs).flatten()
            sel_b = [bboxes[i] for i in idxs_flat]
            sel_c = [class_ids[i] for i in idxs_flat]
            tracked = self.tracker.update(sel_b, sel_c)
        for oid, (box, cls) in tracked.items():
            x, y, w, h = box
            color = (int(self.palette[cls][0]), int(self.palette[cls][1]), int(self.palette[cls][2]))
            label = f"ID {oid}:{self.classes[cls]}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # label background
            cv2.rectangle(frame, (x, y - lh - 2), (x + lw, y), color, thickness=cv2.FILLED)
            # box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
            # text
            cv2.putText(frame, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
        return frame

    def run_on_stream(self, url: str):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open \"{url}\"")
        fid = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            disp = frame.copy()
            if fid % self.frame_skip == 0:
                arr, pad, oh, ow = self.preprocess(frame)
                outs = self.session.run(None, {self.in_name: arr})
                disp = self.postprocess(disp, outs, pad, oh, ow)
            else:
                self.tracker.update([], [])
                for oid, (box, cls) in self.tracker.objects.items():
                    x, y, w, h = box
                    color = (int(self.palette[cls][0]), int(self.palette[cls][1]), int(self.palette[cls][2]))
                    label = f"ID {oid}:{self.classes[cls]}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(disp, (x, y - lh - 2), (x + lw, y), color, thickness=cv2.FILLED)
                    cv2.rectangle(disp, (x, y), (x + w, y + h), color, thickness=2)
                    cv2.putText(disp, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
            cv2.imshow("YOLOv8 RTSP", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fid += 1
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--rtsp", default=None)
    args = parser.parse_args()
    if args.model:
        cfg['YOLO']['model_path'] = args.model
    if args.rtsp:
        cfg['RTSP']['url'] = args.rtsp
    check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")
    YOLOv8(cfg).run_on_stream(cfg['RTSP']['url'])