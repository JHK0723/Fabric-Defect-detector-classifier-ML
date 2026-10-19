"""
FabricDetector - the ONLY thing other modules need from the ML side (team contract unchanged).
    det = FabricDetector("models/fabric_defect.pt", conf=0.25, imgsz=640)
    det.warmup()
    detections = det.detect(frame_bgr)   # list of {"class","class_id","confidence","bbox":[x1,y1,x2,y2]}

Changes vs. the original guide:
  * works with BOTH normal YOLO boxes and oriented boxes (OBB). For OBB models the returned "bbox" is the axis-aligned
    envelope of the rotated box, so the tracker / dashboard need no changes. The rotated corners are also returned in
    "obb_polygon" (4 x [x, y]) if you want to draw them.
  * to_gray=True (default): the models were TRAINED ON GRAYSCALE images, so frames are converted to gray (still 3-channel)
    before inference. Use to_gray=False only for a model you trained on colour.
Works with .pt, .onnx or .engine weights. Knows nothing about cameras or dashboards.
"""
from pathlib import Path

import cv2
import numpy as np


class FabricDetector:
    def __init__(self, weights="models/fabric_defect.pt", conf=0.25, imgsz=640, device=None, half=True, to_gray=True):
        import torch
        from ultralytics import YOLO

        if not Path(str(weights)).exists() and not str(weights).startswith(("yolo", "http")):
            raise FileNotFoundError(f"Model weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.half = bool(half) and self.device != "cpu"
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.to_gray = bool(to_gray)
        self.names = self.model.names            # {0: "hole", 1: "stain", 2: "broken_thread", 3: "irregular_yarn"}

    def warmup(self, n=3):
        dummy = np.full((self.imgsz, self.imgsz, 3), 127, np.uint8)
        for _ in range(n):
            self.detect(dummy)

    def _prep(self, frame):
        if not self.to_gray:
            return frame
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def detect(self, frame):
        """frame: BGR uint8 ndarray (OpenCV image). Returns list of detection dicts."""
        r = self.model.predict(self._prep(frame), conf=self.conf, imgsz=self.imgsz, device=self.device,
                               half=self.half, verbose=False)[0]
        out = []
        obb = getattr(r, "obb", None)
        if obb is not None and len(obb) > 0:                                  # oriented-box model
            env = obb.xyxy.cpu().numpy()
            poly = obb.xyxyxyxy.cpu().numpy()
            confs = obb.conf.cpu().numpy()
            clss = obb.cls.cpu().numpy().astype(int)
            for b, p, s, c in zip(env, poly, confs, clss):
                out.append({"class": self.names[c], "class_id": int(c), "confidence": float(s),
                            "bbox": [int(round(v)) for v in b],
                            "obb_polygon": [[int(round(x)), int(round(y))] for x, y in p]})
            return out
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        return [{"class": self.names[c], "class_id": int(c), "confidence": float(s),
                 "bbox": [int(round(v)) for v in b]} for b, s, c in zip(xyxy, confs, clss)]
