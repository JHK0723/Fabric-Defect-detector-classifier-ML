# Member 2 — Camera, Streaming & Real-Time Pipeline Guide
**Project:** Real-Time AI-Based Textile Fabric Defect Detection
**You own:** the physical rig, phone → laptop streaming, threaded frame capture, the real-time inference loop, temporal deduplication (one physical defect = one event), and all speed/latency measurements.

You can start on Day 1 without a trained fabric model: use the stock `yolo26n.pt` (detects people, cups, etc.) to build and test the whole loop, then swap in Member 1's `models/fabric_defect.pt` by changing one line in `config.yaml`.

---

## 0. What you hand over

| Deliverable | File | Used by |
|---|---|---|
| Camera interface + implementation | `streaming/camera.py`, `streaming/frame_queue.py` | Pipeline, Dashboard |
| Camera test tool | `streaming/test_camera.py` | Setup / debugging |
| ROI + drawing helpers | `vision/preprocessing.py`, `vision/postprocessing.py` | Pipeline |
| Duplicate-suppression tracker | `vision/tracker.py` | Pipeline |
| Real-time pipeline | `vision/pipeline.py` | Member 3's dashboard |
| Runtime config | `config/config.yaml` | Everyone |
| Stand-alone runner | `run_pipeline.py` | Phase 4 demo, debugging |
| Benchmarks | `benchmark_pipeline.py` + CSVs in `logs/` | Report |
| The physical rig | phone mount + lights + fabric board | Data capture (with M1) and demo |

---

## 1. Shared team contract (identical in all three guides)

### 1.1 Folder structure
```text
fabric-inspection/
├── config/config.yaml              (M2) all runtime settings
├── data/
│   ├── raw/<source_name>/          (M1) downloaded datasets + own captures, never edited
│   ├── yolo/                       (M1) GENERATED: images/{train,val,test}, labels/{...}, data.yaml
│   └── external/                   (M1) external test sets (AITEX tiles, unseen fabric)
├── models/fabric_defect.pt         (M1)
├── training/                       (M1) all training scripts
├── vision/                         detector.py (M1); preprocessing.py, postprocessing.py, tracker.py, pipeline.py (M2)
├── streaming/                      frame_queue.py, camera.py, test_camera.py (M2)
├── defect_logging/                 defect_logger.py (M3)   ← NOT "logging/": that name hides Python's built-in logging module
├── dashboard/                      app.py, runtime.py, mocks.py (M3)
├── logs/                           runtime output (SQLite DB, snapshots, benchmark CSVs)
├── run_pipeline.py                 (M2) OpenCV-window runner, no dashboard
├── benchmark_pipeline.py           (M2)
├── requirements.txt
└── README.md
```
Every folder with Python code (`vision/`, `streaming/`, `defect_logging/`, `dashboard/`) gets an empty `__init__.py`. Always run commands **from the repo root**.

### 1.2 Classes (final)
`0 hole · 1 stain · 2 broken_thread · 3 irregular_yarn`. No "normal" class — a frame with no confirmed detections is normal.

### 1.3 Data passed between modules
```python
# Detection — one box in one frame (output of FabricDetector.detect)
{"class": "hole", "class_id": 0, "confidence": 0.947, "bbox": [x1, y1, x2, y2]}   # ints, pixels, full-frame coords

# Event — one physical defect after deduplication (output of DefectEventTracker)
{"event_id": 12, "class": "hole", "confidence": 0.95, "bbox": [...],
 "first_frame": 100, "last_frame": 104, "first_seen": "2026-10-20T14:32:17",
 "last_seen": "2026-10-20T14:32:18", "frames_seen": 5, "snapshot": <BGR ndarray or None>}
```
Call chain: `Camera → InspectionPipeline → FabricDetector.detect() → DefectEventTracker.update() → on_event callback → DefectLogger + Dashboard`.

### 1.4 Team environment (everyone, Day 1)
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
:: Install PyTorch with CUDA FIRST: copy the command from https://pytorch.org/get-started/locally/
:: (Stable, Windows, Pip, Python, newest CUDA listed), then:
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Must print `True`. `requirements.txt`:
```text
ultralytics>=8.4.0
opencv-python
numpy
pandas
pyyaml
psutil
streamlit>=1.37
```

---

## 2. Your 4-week plan

| Week | Goal | Done when |
|---|---|---|
| 1 | Build rig, get phone → OpenCV working (USB and Wi-Fi), threaded camera, pipeline running with `yolo26n.pt` | `run_pipeline.py` shows live boxes at a measured FPS |
| 2 | Capture training videos with Member 1 on the rig; tracker + unit test; swap in fabric model v0 | One hole moving through the view = exactly one event |
| 3 | Tune tracker thresholds with M1's numbers; ROI; benchmarks; hand pipeline to M3 | Dashboard runs on your pipeline; benchmark CSVs saved |
| 4 | Movement-speed and lighting experiments, latency measurement, demo rehearsal, backup plans | Experiments 6–7 filled in; demo works on 3 dry runs in a row |

---

## 3. Physical rig (build this first — it affects data quality more than any code)

```text
      LED lamp (45°)          LED lamp (45°)
            \      📱 phone      /
             \     (lens down)  /
              \       |        /
               \  25–40 cm    /
                \     |      /
   ──────────────────────────────────────  flat board (matte, plain colour)
          ← fabric pulled by hand →
```

- **Mount:** overhead phone holder / tripod with a horizontal arm / a shelf with a hole. The phone must be **parallel to the fabric** and must not move between data capture and demo. Mark its position with tape.
- **Height:** 25–40 cm. Closer = small defects are bigger in the image (good) but the field of view is smaller. Decide once, with Member 1, and keep it.
- **Lighting:** two white LED lamps (desk lamps or LED strips) at ~45° from both sides — this avoids glare and shadows. Turn off/avoid mixing with tube light and sunlight. Same lights for data capture and demo.
- **Board:** a matte board under the fabric (a sheet of plain chart paper on a table works). Tape a strip along each side so the fabric runs straight.
- **Speed marks:** put a tape mark on the board edge every 10 cm; you'll use them to measure fabric speed (cm/s) in experiments.
- **Phone camera settings (in whichever app you stream with):** 1280×720, 30 fps; **lock focus and exposure** (tap-and-hold on most phones / app setting) so the image doesn't "breathe"; turn off beauty/HDR/filters; disable the phone's auto-sleep; keep it on the charger.
- **Motion blur is your enemy.** Moving fabric + slow shutter = blurred defects. Bright lighting makes the phone choose a short exposure automatically. If the app lets you set shutter/ISO manually, prefer a faster shutter (e.g. 1/250 s or faster) with higher ISO.

---

## 4. Getting the phone feed into OpenCV

Any method works as long as the laptop ends up with either **a webcam device index** (0, 1, 2…) or **a stream URL**. Your code won't care which — `OpenCVCamera` accepts both. Apps change over time, so verify on your actual phones.

| Option | How it reaches OpenCV | Notes |
|---|---|---|
| **USB "phone as webcam" app** (e.g. DroidCam, Iriun, Camo — Android/iOS clients + Windows client) | Virtual webcam → `cv2.VideoCapture(index)` | **Preferred.** Lowest latency, no Wi-Fi problems. Install the Windows client, connect via USB (Android may need USB debugging enabled). |
| **Windows 11 "use phone as a webcam"** (Phone Link, Android) | Shows up as a webcam | No extra app if it works on your phone/PC combination. |
| **Wi-Fi MJPEG/RTSP stream app** (e.g. "IP Webcam" on Android, DroidCam over Wi-Fi) | URL → `cv2.VideoCapture("http://<phone-ip>:<port>/video")` | The app shows the exact URL. Open it in a browser first to confirm. |

**College Wi-Fi often blocks device-to-device traffic.** If the URL doesn't load, turn on the **phone's hotspot and connect the laptop to it** (or use a personal router). Also allow Python through the Windows Firewall prompt.

### 4.1 Find your camera
```python
# File: streaming/test_camera.py
"""
Probe webcam indices or preview any source with live FPS.
  python streaming/test_camera.py --probe                     # list working webcam indices
  python streaming/test_camera.py --source 1                  # preview webcam 1
  python streaming/test_camera.py --source http://192.168.43.1:8080/video
Keys: s = save snapshot to logs/, q = quit
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streaming.camera import OpenCVCamera  # noqa: E402


def probe(max_index=6):
    api = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    for i in range(max_index):
        cap = cv2.VideoCapture(i, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = cap.read()
            shape = frame.shape if ok else None
            print(f"index {i}: OPEN  first frame ok={ok} shape={shape}")
        else:
            print(f"index {i}: -")
        cap.release()


def preview(source, width, height):
    cam = OpenCVCamera(source, width=width, height=height).start()
    Path("logs").mkdir(exist_ok=True)
    last_id = -1
    print("Press s to save a snapshot, q to quit")
    while True:
        pkt = cam.get(last_id, timeout=2.0)
        if pkt is None:
            print("No frame:", cam.last_error or "waiting...")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        last_id = pkt.frame_id
        img = pkt.frame.copy()
        h, w = img.shape[:2]
        cv2.putText(img, f"{w}x{h}  camera FPS {cam.fps:.1f}  dropped {cam.dropped_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("camera test", img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            p = f"logs/snapshot_{int(time.time())}.jpg"
            cv2.imwrite(p, pkt.frame)
            print("saved", p)
        elif k == ord("q"):
            break
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--source", default="0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    probe() if a.probe else preview(a.source, a.width, a.height)
```
(This uses `OpenCVCamera` from §5 — write §5 first.) Index 0 is usually the laptop's built-in webcam; the phone is often 1 or 2.

---

## 5. Threaded capture: why and how

If you read a frame and then run the model in the same loop, the camera's internal buffer fills up while the model works, and you end up processing **old** frames — latency grows by seconds. The fix:

```text
Camera thread ──writes──▶ [ latest frame only ] ──read──▶ Inference thread ──▶ latest result ──▶ Dashboard
  (30 FPS, never waits)       (size-1 buffer;            (as fast as the GPU allows,
                               old frames dropped)         always on the newest frame)
```
A size-1 "latest frame" buffer is better than a FIFO queue here: a queue would preserve old frames you don't care about. Frame skipping happens **automatically** — if inference runs at 20 FPS and the camera at 30, a third of frames are dropped and you can report exactly how many.

```python
# File: streaming/__init__.py
```
```python
# File: streaming/frame_queue.py
"""Thread-safe 'latest frame' buffer: producer overwrites, consumer always gets the newest frame."""
import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray        # BGR image
    frame_id: int            # increasing counter from the camera thread
    timestamp: float         # time.perf_counter() when the frame arrived on the laptop


class LatestFrameBuffer:
    def __init__(self):
        self._cond = threading.Condition()
        self._packet = None
        self._consumed = True
        self.dropped = 0     # frames overwritten before anyone read them

    def put(self, packet: FramePacket):
        with self._cond:
            if not self._consumed:
                self.dropped += 1
            self._packet = packet
            self._consumed = False
            self._cond.notify_all()

    def get(self, last_id: int = -1, timeout: float = 1.0):
        """Block until a frame newer than last_id exists (or timeout). Returns FramePacket or None."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._packet is not None and self._packet.frame_id > last_id, timeout)
            if not ok:
                return None
            self._consumed = True
            return self._packet
```

```python
# File: streaming/camera.py
"""
Camera interface + OpenCV implementation.

BaseCamera is the contract. Anything that implements start/stop/get/fps/is_connected
(phone, USB webcam, video file, future industrial camera, synthetic test camera) can be
plugged into InspectionPipeline without changing the detector, tracker or dashboard.
"""
import os
import threading
import time
from pathlib import Path

import cv2

from .frame_queue import FramePacket, LatestFrameBuffer


class BaseCamera:
    last_error = ""

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def get(self, last_id=-1, timeout=1.0):
        """Return the newest FramePacket with frame_id > last_id, or None on timeout."""
        raise NotImplementedError

    @property
    def fps(self):
        return 0.0

    @property
    def is_connected(self):
        return False

    @property
    def dropped_frames(self):
        return 0


def parse_source(src):
    if isinstance(src, int):
        return src
    s = str(src).strip()
    return int(s) if s.isdigit() else s


class OpenCVCamera(BaseCamera):
    """
    source: webcam index (0, 1, ...), stream URL ("http://...", "rtsp://..."), or a video file path.
    Video files are played back at their native FPS (to simulate a live camera) and loop if loop_file.
    """

    def __init__(self, source=0, width=1280, height=720, fps=30, backend="auto",
                 loop_file=True, reconnect_delay=1.0):
        self.source = parse_source(source)
        self.is_file = isinstance(self.source, str) and Path(self.source).exists()
        self.width, self.height, self.req_fps = int(width), int(height), int(fps)
        self.backend = backend
        self.loop_file = loop_file
        self.reconnect_delay = reconnect_delay
        self._buffer = LatestFrameBuffer()
        self._thread = None
        self._running = False
        self._connected = False
        self._fps = 0.0
        self._frame_id = 0
        self.last_error = ""

    # ---------- public API ----------
    def start(self, wait_s=5.0):
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        t0 = time.time()
        while self._running and self._frame_id == 0 and time.time() - t0 < wait_s:
            time.sleep(0.05)
        return self

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def get(self, last_id=-1, timeout=1.0):
        return self._buffer.get(last_id, timeout)

    @property
    def fps(self):
        return self._fps

    @property
    def is_connected(self):
        return self._connected

    @property
    def dropped_frames(self):
        return self._buffer.dropped

    # ---------- internals ----------
    def _open(self):
        if isinstance(self.source, int):
            if os.name == "nt" and self.backend in ("auto", "dshow"):
                api = cv2.CAP_DSHOW          # usually the most reliable on Windows for virtual webcams
            elif os.name == "nt" and self.backend == "msmf":
                api = cv2.CAP_MSMF
            else:
                api = cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, api)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep the driver buffer small (ignored by some backends)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _loop(self):
        cap, last_t, file_period = None, None, None
        while self._running:
            if cap is None:
                cap = self._open()
                if cap is None:
                    self._connected = False
                    self.last_error = f"Cannot open camera source {self.source!r}"
                    time.sleep(self.reconnect_delay)
                    continue
                self._connected, self.last_error = True, ""
                if self.is_file:
                    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    file_period = 1.0 / file_fps

            t_read = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                if self.is_file and self.loop_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self.is_file:
                    self.last_error = "End of video file"
                    break
                self._connected = False
                self.last_error = "Stream lost - reconnecting..."
                cap.release()
                cap = None
                time.sleep(self.reconnect_delay)
                continue

            now = time.perf_counter()
            self._frame_id += 1
            self._buffer.put(FramePacket(frame, self._frame_id, now))
            if last_t is not None and now > last_t:
                inst = 1.0 / (now - last_t)
                self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
            last_t = now
            if file_period:                       # pace video files like a live camera
                rest = file_period - (time.perf_counter() - t_read)
                if rest > 0:
                    time.sleep(rest)

        if cap is not None:
            cap.release()
        self._connected = False
        self._running = False
```
To support an industrial camera later, write `class IndustrialCamera(BaseCamera)` using the vendor SDK and push `FramePacket`s into a `LatestFrameBuffer` — nothing else changes.

---

## 6. ROI and drawing helpers

```python
# File: vision/__init__.py
```
```python
# File: vision/preprocessing.py
"""Region-of-interest cropping. ROI is given as fractions [x1, y1, x2, y2] so it works at any resolution."""


def crop_roi(frame, roi):
    if not roi:
        return frame, (0, 0)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    return frame[py1:py2, px1:px2], (px1, py1)


def offset_detections(detections, ox, oy):
    """Shift boxes found in the ROI crop back into full-frame coordinates."""
    if ox == 0 and oy == 0:
        return detections
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        d["bbox"] = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
    return detections
```
Use an ROI to cut away the board edges, your hands and anything that isn't fabric. A tighter ROI also makes defects relatively bigger for the model (they're resized to `imgsz`).

```python
# File: vision/postprocessing.py
"""Drawing boxes/labels/status on frames. Pure OpenCV, no model knowledge."""
import cv2

CLASS_COLORS = {
    "hole": (0, 0, 255),
    "stain": (0, 165, 255),
    "broken_thread": (255, 0, 255),
    "irregular_yarn": (255, 128, 0),
}


def color_for(name):
    return CLASS_COLORS.get(name, (0, 255, 255))


def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(th + 6, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_annotations(frame, detections, tracks=(), roi=None, stats=None):
    img = frame.copy()
    h, w = img.shape[:2]
    if roi:
        x1, y1, x2, y2 = roi
        cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (180, 180, 180), 1)
    # raw detections: thin boxes
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        c = color_for(d["class"])
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 1)
        _label(img, f"{d['class']} {d['confidence']:.2f}", x1, y1, c)
    # confirmed defects: thick boxes with event number
    for t in tracks:
        if t.get("confirmed"):
            x1, y1, x2, y2 = t["bbox"]
            c = color_for(t["class"])
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 4)
            _label(img, f"DEFECT #{t['event_id']} {t['class'].upper()}", x1, y2 + 24, c)
    if stats:
        txt = (f"cam {stats.get('camera_fps', 0):.0f} FPS | AI {stats.get('inference_fps', 0):.1f} FPS | "
               f"{stats.get('inference_ms', 0):.0f} ms | latency {stats.get('latency_ms', 0):.0f} ms")
        cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img
```

---

## 7. Duplicate suppression: the tracker

**Problem:** one hole stays in view for ~30–60 frames as the fabric moves. We want **one** event.

**Chosen approach (simplest robust one):** a small tracker that matches each new detection to an existing "track" if their boxes overlap (IoU) **or** their centres are close (the fabric moves only a little between frames). Then:

1. **Ignore** detections below `min_confidence`.
2. A track becomes a **confirmed defect event** once it has been seen in `confirm_frames` consecutive processed frames **and** its best confidence reached `confirm_confidence`. → alert fires **once**, logged **once**.
3. The track lives on while the defect is visible; after `max_missed` frames without a match it's closed (the logger updates its end time).
4. Matching is **class-agnostic** and the class is a confidence-weighted vote — if the model flickers between "hole" and "stain" on the same spot, you still get one event, labelled with the majority class.

Why not Ultralytics' built-in ByteTrack (`model.track`)? It would work, but ties dedup to one model API and needs re-tuning for jerky hand movement. This tracker is ~100 lines, model-independent, and easy to explain in a viva. Known limitation: if the operator pulls the fabric **back** so a defect leaves and re-enters after `max_missed` frames, it's counted again — document this.

```python
# File: vision/tracker.py
"""
DefectEventTracker: turns per-frame detections into one event per physical defect.
    tracker = DefectEventTracker(min_confidence=0.5, confirm_confidence=0.7, confirm_frames=3)
    new_events, closed_events = tracker.update(detections, frame_id, frame)
'confirm_frames' and 'max_missed' count PROCESSED (inference) frames, not camera frames.
"""
import math
from collections import defaultdict
from datetime import datetime


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


class _Track:
    def __init__(self, track_id, det, frame_id, now):
        self.track_id = track_id
        self.bbox = list(det["bbox"])
        self.best_bbox = list(det["bbox"])
        self.first_frame = self.last_frame = frame_id
        self.first_seen = self.last_seen = now
        self.hits = 1
        self.streak = 1
        self.missed = 0
        self.class_scores = defaultdict(float)
        self.class_scores[det["class"]] += det["confidence"]
        self.max_conf = det["confidence"]
        self.confirmed = False
        self.event_id = None
        self.snapshot = None

    @property
    def cls(self):
        return max(self.class_scores, key=self.class_scores.get)

    def as_event(self):
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "class": self.cls,
            "confidence": round(float(self.max_conf), 4),
            "bbox": list(self.best_bbox),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_seen": self.first_seen.isoformat(timespec="seconds"),
            "last_seen": self.last_seen.isoformat(timespec="seconds"),
            "frames_seen": self.hits,
            "snapshot": self.snapshot,
        }


class DefectEventTracker:
    def __init__(self, min_confidence=0.5, confirm_confidence=0.7, confirm_frames=3, max_missed=10,
                 iou_threshold=0.2, max_center_dist=0.15, keep_snapshot=True):
        self.min_confidence = min_confidence
        self.confirm_confidence = confirm_confidence
        self.confirm_frames = confirm_frames
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.max_center_dist = max_center_dist      # fraction of the frame diagonal
        self.keep_snapshot = keep_snapshot
        self.reset()

    def reset(self):
        self.tracks = []
        self._next_track = 1
        self._next_event = 1

    @property
    def total_events(self):
        return self._next_event - 1

    def _close(self, t, now=None):
        ev = t.as_event()
        ev["snapshot"] = None
        return ev

    def _match_score(self, box_a, box_b, diag):
        ov = iou(box_a, box_b)
        (ax, ay), (bx, by) = center(box_a), center(box_b)
        dist = math.hypot(ax - bx, ay - by) / diag
        if ov >= self.iou_threshold or dist <= self.max_center_dist:
            return ov + (1.0 - dist)
        return None

    def update(self, detections, frame_id, frame=None):
        now = datetime.now()
        dets = [d for d in detections if d["confidence"] >= self.min_confidence]
        diag = math.hypot(frame.shape[1], frame.shape[0]) if frame is not None else 1500.0

        # 1) greedy matching, best pairs first
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(dets):
                s = self._match_score(t.bbox, d["bbox"], diag)
                if s is not None:
                    pairs.append((s, ti, di))
        pairs.sort(key=lambda p: p[0], reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, d = self.tracks[ti], dets[di]
            t.bbox = list(d["bbox"])
            t.last_frame, t.last_seen = frame_id, now
            t.hits += 1
            t.streak += 1
            t.missed = 0
            t.class_scores[d["class"]] += d["confidence"]
            if d["confidence"] > t.max_conf:
                t.max_conf = d["confidence"]
                t.best_bbox = list(d["bbox"])

        # 2) unmatched tracks age
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
                t.streak = 0

        # 3) unmatched detections: merge into an overlapping track already matched this frame
        #    (e.g. two class boxes on one defect), otherwise start a new track
        matched_now = [self.tracks[ti] for ti in used_t]
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            host = next((t for t in matched_now if iou(t.bbox, d["bbox"]) > 0.3), None)
            if host is not None:
                host.class_scores[d["class"]] += d["confidence"]
                host.max_conf = max(host.max_conf, d["confidence"])
                continue
            self.tracks.append(_Track(self._next_track, d, frame_id, now))
            self._next_track += 1

        # 4) confirmation -> new events (fires once per track)
        new_events = []
        for t in self.tracks:
            if (not t.confirmed and t.streak >= self.confirm_frames
                    and t.max_conf >= self.confirm_confidence):
                t.confirmed = True
                t.event_id = self._next_event
                self._next_event += 1
                if self.keep_snapshot and frame is not None:
                    t.snapshot = frame.copy()
                new_events.append(t.as_event())

        # 5) remove dead tracks; confirmed ones are reported as closed
        closed, alive = [], []
        for t in self.tracks:
            if t.missed > self.max_missed:
                if t.confirmed:
                    closed.append(self._close(t))
            else:
                alive.append(t)
        self.tracks = alive
        return new_events, closed

    def active_tracks(self):
        """Tracks visible in the latest frame (for drawing)."""
        return [{"track_id": t.track_id, "class": t.cls, "confidence": round(t.max_conf, 3),
                 "bbox": list(t.bbox), "confirmed": t.confirmed, "event_id": t.event_id}
                for t in self.tracks if t.missed == 0]

    def flush(self):
        """Close all confirmed tracks (call when inspection stops)."""
        closed = [self._close(t) for t in self.tracks if t.confirmed]
        self.tracks = []
        return closed
```

### 7.1 Unit test (run this before trusting the tracker)
```python
# File: tests/test_tracker.py
"""Run: python -m pytest tests/test_tracker.py -q   (or: python tests/test_tracker.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.tracker import DefectEventTracker  # noqa: E402

FRAME = np.zeros((720, 1280, 3), np.uint8)


def det(x, cls="hole", conf=0.9, size=40):
    return {"class": cls, "class_id": 0, "confidence": conf, "bbox": [x, 300, x + size, 300 + size]}


def test_moving_defect_gives_one_event():
    tr = DefectEventTracker(confirm_frames=3, max_missed=5)
    events = []
    for f in range(30):                                   # hole slides 15 px per frame
        new, _ = tr.update([det(100 + 15 * f)], f, FRAME)
        events += new
    assert len(events) == 1


def test_flickering_class_still_one_event():
    tr = DefectEventTracker(confirm_frames=3)
    events = []
    for f in range(20):
        new, _ = tr.update([det(200 + 5 * f, cls="hole" if f % 3 else "stain")], f, FRAME)
        events += new
    assert len(events) == 1 and events[0]["class"] == "hole"


def test_weak_single_frame_is_ignored():
    tr = DefectEventTracker(confirm_frames=3, confirm_confidence=0.7)
    new, _ = tr.update([det(300, conf=0.55)], 0, FRAME)
    for f in range(1, 10):
        new += tr.update([], f, FRAME)[0]
    assert new == []


def test_two_far_apart_defects_two_events():
    tr = DefectEventTracker(confirm_frames=2)
    events = []
    for f in range(10):
        new, _ = tr.update([det(100 + 5 * f), det(1000 + 5 * f, cls="stain")], f, FRAME)
        events += new
    assert len(events) == 2


def test_event_closes_after_leaving_view():
    tr = DefectEventTracker(confirm_frames=2, max_missed=3)
    for f in range(5):
        tr.update([det(100)], f, FRAME)
    closed = []
    for f in range(5, 12):
        closed += tr.update([], f, FRAME)[1]
    assert len(closed) == 1 and closed[0]["last_frame"] == 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
```
(`pip install pytest` if you want the pytest form.)

---

## 8. The real-time pipeline

```python
# File: vision/pipeline.py
"""
InspectionPipeline: camera -> (ROI) -> detector -> tracker -> callbacks + latest annotated result.
Runs inference in its own thread. The UI (OpenCV window or Streamlit) just calls .latest().

    pipe = build_from_config(load_config("config/config.yaml"), on_event=print).start()
    result = pipe.latest()   # {"frame", "frame_id", "detections", "active_tracks", "stats"} or None
    pipe.stop()
"""
import threading
import time
from pathlib import Path

import yaml

from .postprocessing import draw_annotations
from .preprocessing import crop_roi, offset_detections
from .tracker import DefectEventTracker


def load_config(path="config/config.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class InspectionPipeline:
    def __init__(self, camera, detector, tracker, roi=None, on_event=None, on_event_closed=None, draw=True):
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.roi = roi
        self.on_event = on_event
        self.on_event_closed = on_event_closed
        self.draw = draw
        self._lock = threading.Lock()
        self._latest = None
        self._thread = None
        self._running = False
        self._stats = {"running": False, "camera_connected": False, "model_loaded": detector is not None,
                       "camera_fps": 0.0, "inference_fps": 0.0, "inference_ms": 0.0, "latency_ms": 0.0,
                       "frames_processed": 0, "frames_dropped": 0, "events_total": 0, "defect_in_view": False}

    # ---------- control ----------
    def start(self):
        if self._running:
            return self
        self.camera.start()
        if hasattr(self.detector, "warmup"):
            self.detector.warmup()
        self._running = True
        self._stats["running"] = True
        self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.camera.stop()
        for ev in self.tracker.flush():
            self._emit(self.on_event_closed, ev)
        with self._lock:
            self._stats["running"] = False
            self._stats["camera_connected"] = False

    @property
    def running(self):
        return self._running

    def latest(self):
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def stats(self):
        with self._lock:
            return dict(self._stats)

    # ---------- internals ----------
    @staticmethod
    def _emit(cb, ev):
        if cb is None:
            return
        try:
            cb(ev)
        except Exception as e:                       # never let a UI/logging bug kill inference
            print(f"[pipeline] callback error: {e!r}")

    @staticmethod
    def _ema(old, new, a=0.1):
        return new if old == 0 else (1 - a) * old + a * new

    def _loop(self):
        last_id, last_end = -1, None
        while self._running:
            pkt = self.camera.get(last_id, timeout=1.0)
            if pkt is None:
                with self._lock:
                    self._stats["camera_connected"] = self.camera.is_connected
                continue
            last_id = pkt.frame_id

            t0 = time.perf_counter()
            roi_img, (ox, oy) = crop_roi(pkt.frame, self.roi)
            dets = offset_detections(self.detector.detect(roi_img), ox, oy)
            t1 = time.perf_counter()

            new_events, closed = self.tracker.update(dets, pkt.frame_id, pkt.frame)
            for ev in new_events:
                self._emit(self.on_event, ev)
            for ev in closed:
                self._emit(self.on_event_closed, ev)
            tracks = self.tracker.active_tracks()

            s = self._stats
            s["camera_connected"] = self.camera.is_connected
            s["camera_fps"] = self.camera.fps
            s["inference_ms"] = self._ema(s["inference_ms"], (t1 - t0) * 1000)
            s["frames_processed"] += 1
            s["frames_dropped"] = self.camera.dropped_frames
            s["events_total"] = self.tracker.total_events
            s["defect_in_view"] = any(t["confirmed"] for t in tracks)

            frame_out = draw_annotations(pkt.frame, dets, tracks, self.roi, s) if self.draw else pkt.frame
            now = time.perf_counter()
            s["latency_ms"] = self._ema(s["latency_ms"], (now - pkt.timestamp) * 1000)
            if last_end is not None and now > last_end:
                s["inference_fps"] = self._ema(s["inference_fps"], 1.0 / (now - last_end))
            last_end = now

            with self._lock:
                self._latest = {"frame": frame_out, "frame_id": pkt.frame_id, "detections": dets,
                                "active_tracks": tracks, "stats": dict(s)}
        with self._lock:
            self._stats["running"] = False


def build_from_config(cfg, on_event=None, on_event_closed=None, camera=None, detector=None):
    """Build a pipeline from config.yaml. Pass camera/detector to override (e.g. mocks)."""
    if camera is None:
        from streaming.camera import OpenCVCamera
        c = cfg["camera"]
        camera = OpenCVCamera(c["source"], c.get("width", 1280), c.get("height", 720), c.get("fps", 30),
                              c.get("backend", "auto"), c.get("loop_file", True))
    if detector is None:
        from .detector import FabricDetector
        m = cfg["model"]
        detector = FabricDetector(m["weights"], conf=m.get("conf", 0.25), imgsz=m.get("imgsz", 640),
                                  half=m.get("half", True))
    t = cfg["tracker"]
    tracker = DefectEventTracker(t["min_confidence"], t["confirm_confidence"], t["confirm_frames"],
                                 t["max_missed"], t["iou_threshold"], t["max_center_dist"])
    return InspectionPipeline(camera, detector, tracker, roi=cfg.get("roi"),
                              on_event=on_event, on_event_closed=on_event_closed)
```

### 8.1 Config file
```yaml
# File: config/config.yaml
camera:
  source: 1            # webcam index of the phone (find with test_camera.py --probe),
                       # or a URL like "http://192.168.43.1:8080/video", or a video file path
  width: 1280
  height: 720
  fps: 30
  backend: auto        # auto | dshow | msmf   (Windows webcam backends)
  loop_file: true      # loop video files (for testing without the phone)

roi: null              # [x1, y1, x2, y2] as fractions of the frame, e.g. [0.10, 0.05, 0.90, 0.95]

model:
  weights: models/fabric_defect.pt   # use yolo26n.pt for plumbing tests before M1's model exists
  imgsz: 640           # MUST equal the training imgsz
  conf: 0.25           # raw model threshold (low on purpose); real decisions are in tracker
  half: true

tracker:               # tune with Member 1's threshold_sweep results
  min_confidence: 0.50
  confirm_confidence: 0.70
  confirm_frames: 3
  max_missed: 10
  iou_threshold: 0.20
  max_center_dist: 0.15

logging:
  db_path: logs/inspection.db
  snapshot_dir: logs/snapshots

alerts:
  beep: true
  banner_seconds: 3
```

### 8.2 Stand-alone runner (Phase 4 milestone — no dashboard needed)
```python
# File: run_pipeline.py
"""
Live detection in an OpenCV window.
  python run_pipeline.py                              # uses config/config.yaml
  python run_pipeline.py --source 1 --weights yolo26n.pt
  python run_pipeline.py --source data/videos/F01_v01.mp4
Press q in the window to quit.
"""
import argparse

import cv2

from vision.pipeline import build_from_config, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--source")
    ap.add_argument("--weights")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.source is not None:
        cfg["camera"]["source"] = args.source
    if args.weights:
        cfg["model"]["weights"] = args.weights

    def on_event(e):
        print(f"[DEFECT #{e['event_id']}] {e['class']} conf={e['confidence']:.2f} at {e['first_seen']} "
              f"(frame {e['first_frame']})")

    def on_closed(e):
        print(f"   #{e['event_id']} left view after {e['frames_seen']} frames")

    pipe = build_from_config(cfg, on_event=on_event, on_event_closed=on_closed)
    print("Starting camera and loading model...")
    pipe.start()
    print("Running. Press q in the video window to quit.")
    try:
        while True:
            r = pipe.latest()
            if r is not None:
                cv2.imshow("Fabric inspection", r["frame"])
            if cv2.waitKey(15) & 0xFF == ord("q"):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print("Final stats:", pipe.stats())


if __name__ == "__main__":
    main()
```

**First milestone test:** set `weights: yolo26n.pt`, run `python run_pipeline.py`, hold a cup or your hand in view → boxes + a single `[DEFECT #1] cup` event per object. That proves camera → model → tracker works before any fabric model exists.

---

## 9. Benchmarks (numbers for the report)

```python
# File: benchmark_pipeline.py
"""
Run the pipeline headless for N seconds and record system metrics once per second.
  python benchmark_pipeline.py --seconds 60 --tag usb_720p_n640
Writes logs/benchmark_<tag>.csv and prints a summary.
"""
import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil

from vision.pipeline import build_from_config, load_config


def gpu_stats():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
        util, mem = out.stdout.strip().splitlines()[0].split(",")
        return float(util), float(mem)
    except Exception:
        return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--source")
    ap.add_argument("--weights")
    ap.add_argument("--imgsz", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.source is not None:
        cfg["camera"]["source"] = args.source
    if args.weights:
        cfg["model"]["weights"] = args.weights
    if args.imgsz:
        cfg["model"]["imgsz"] = args.imgsz

    pipe = build_from_config(cfg).start()
    psutil.cpu_percent(None)
    rows = []
    time.sleep(3)                      # let EMAs settle
    for i in range(args.seconds):
        time.sleep(1)
        s = pipe.stats()
        gu, gm = gpu_stats()
        rows.append([i, s["camera_fps"], s["inference_fps"], s["inference_ms"], s["latency_ms"],
                     s["frames_processed"], s["frames_dropped"], psutil.cpu_percent(None), gu, gm])
    pipe.stop()

    Path("logs").mkdir(exist_ok=True)
    out = Path("logs") / f"benchmark_{args.tag}.csv"
    hdr = ["t", "camera_fps", "inference_fps", "inference_ms", "latency_ms", "frames_processed",
           "frames_dropped", "cpu_pct", "gpu_util_pct", "gpu_mem_mb"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    a = np.array([r[1:5] + r[7:9] for r in rows], dtype=float)
    names = ["camera_fps", "inference_fps", "inference_ms", "latency_ms", "cpu_pct", "gpu_util_pct"]
    print(f"\nSaved {out}")
    for j, n in enumerate(names):
        col = a[:, j]
        print(f"  {n:14s} mean {np.nanmean(col):7.1f}   p95 {np.nanpercentile(col, 95):7.1f}")
    last = rows[-1]
    print(f"  dropped frames: {last[6]} of {last[5] + last[6]} "
          f"({100 * last[6] / max(1, last[5] + last[6]):.0f}%)")


if __name__ == "__main__":
    main()
```

Run a small matrix and keep every CSV:

| Tag | Source | Model / imgsz |
|---|---|---|
| `usb_n640` | USB | yolo26n / 640 |
| `usb_s640` | USB | yolo26s / 640 |
| `usb_s960` | USB | yolo26s / 960 |
| `wifi_s640` | Wi-Fi | yolo26s / 640 |

**About latency:** `latency_ms` is measured from "frame arrived on the laptop" to "annotated result ready". It does **not** include the phone's own encoding + transmission delay. To measure true **glass-to-glass** latency: show a millisecond stopwatch on a second phone in front of the camera, film the laptop screen and the stopwatch together with a third phone, and compare the two times in one photo. Do it 5 times, report the mean. This is a good, honest number for the report.

**Decision rule for frame skipping:** don't add manual skipping. The latest-frame buffer already processes as many frames as the GPU allows. Target ≥ 15 inference FPS; if below, try (in order) a tighter ROI, `yolo26n`, smaller `imgsz`, then ONNX export (Member 1).

---

## 10. Tuning the tracker (Week 3)

1. Get `min_confidence` and `confirm_confidence` from Member 1's `threshold_sweep.py` (val set).
2. Record 3 test videos on the rig: clean fabric only; fabric with 5 known holes; with 5 stains. Pass a video as `--source` so tuning is repeatable.
3. For each setting, count: events on the clean video (false alarms), events on defect videos vs the true count (misses and duplicates).
4. Adjust: duplicates → raise `max_missed` or `max_center_dist`; flickery false alarms → raise `confirm_frames`; missed fast-moving defects → lower `confirm_frames` to 2.
5. Write the final values into `config.yaml` and a short table in the report.

---

## 11. Experiments you run (fill the table Member 3 keeps)

- **Exp 6 — Lighting:** your LED setup vs room light only vs one lamp off (side shadow). Count false alarms per 100 frames on clean fabric and hit rate on known defects.
- **Exp 7 — Movement speed:** slow (~2 cm/s), normal (~5 cm/s), fast (~10+ cm/s), timed with the 10 cm tape marks. For each: detection rate on 10 known defects, duplicates, blur visible?
- **System metrics:** the benchmark matrix above + glass-to-glass latency.

Expect detection rate to drop at high speed (blur + fewer frames per defect) — finding *where* it drops is a real result.

---

## 12. Demo-day backup plan
- Test USB **and** Wi-Fi/hotspot the day before; bring a spare USB cable.
- Record a good demo video on the rig in Week 4. If the phone fails on stage, `--source data/videos/demo.mp4` runs the exact same pipeline.
- Close other GPU apps (games, browsers with video) before the demo; plug in the laptop charger.

## 13. Common mistakes
- Reading frames and running the model in the same loop → growing lag.
- Using the laptop webcam (index 0) by accident — check with `--probe`.
- Auto-focus/auto-exposure left on → focus hunting → false alarms.
- Campus Wi-Fi client isolation → use phone hotspot.
- Changing the rig height after Member 1 collected training data.
- Measuring only model latency and calling it end-to-end latency.
