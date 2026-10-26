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
from .preprocessing import crop_roi, offset_detections, correct_lighting
from .tracker import DefectEventTracker


def load_config(path="config/config.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class InspectionPipeline:
    def __init__(self, camera, detector, tracker, roi=None, on_event=None, on_event_closed=None, draw=True, preprocessing=None):
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.roi = roi
        self.preprocessing = preprocessing or {}
        self.on_event = on_event
        self.on_event_closed = on_event_closed
        self.draw = draw
        self._lock = threading.Lock()
        self._latest = None
        self._thread = None
        self._running = False
        self._stats = {
            "running": False,
            "camera_connected": False,
            "model_loaded": detector is not None,
            "camera_fps": 0.0,
            "inference_fps": 0.0,
            "inference_ms": 0.0,
            "latency_ms": 0.0,
            "frames_processed": 0,
            "frames_dropped": 0,
            "events_total": 0,
            "defect_in_view": False,
        }

    # ---------- control ----------
    def start(self):
        if self._running:
            return self
        self.camera.start()
        if self.detector is not None and hasattr(self.detector, "warmup"):
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
            if self.preprocessing.get("lighting_correction", False):
                roi_img = correct_lighting(
                    roi_img,
                    method=self.preprocessing.get("method", "clahe"),
                    clip_limit=self.preprocessing.get("clip_limit", 2.0),
                    tile_grid_size=self.preprocessing.get("tile_grid_size", (8, 8)),
                    blur_kernel_size=self.preprocessing.get("blur_kernel_size", 51),
                    target_mean=self.preprocessing.get("target_mean", 128),
                    to_gray=self.preprocessing.get("to_gray", True),
                )
            if self.detector is not None:
                raw_dets = self.detector.detect(roi_img)
            else:
                raw_dets = []
            dets = offset_detections(raw_dets, ox, oy)
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
                self._latest = {
                    "frame": frame_out,
                    "frame_id": pkt.frame_id,
                    "detections": dets,
                    "active_tracks": tracks,
                    "stats": dict(s),
                }
        with self._lock:
            self._stats["running"] = False


def build_from_config(cfg, on_event=None, on_event_closed=None, camera=None, detector=None):
    """Build a pipeline from config.yaml. Pass camera/detector to override (e.g. mocks)."""
    if camera is None:
        from streaming.camera import OpenCVCamera
        c = cfg["camera"]
        camera = OpenCVCamera(
            c["source"],
            c.get("width", 1280),
            c.get("height", 720),
            c.get("fps", 30),
            c.get("backend", "auto"),
            c.get("loop_file", True),
        )
    if detector is None and "model" in cfg:
        try:
            from .detector import FabricDetector
            m = cfg["model"]
            detector = FabricDetector(
                m["weights"],
                conf=m.get("conf", 0.25),
                imgsz=m.get("imgsz", 640),
                half=m.get("half", False),
            )
        except Exception:
            detector = None


    t = cfg["tracker"]
    tracker = DefectEventTracker(
        t["min_confidence"],
        t["confirm_confidence"],
        t["confirm_frames"],
        t["max_missed"],
        t["iou_threshold"],
        t["max_center_dist"],
    )
    return InspectionPipeline(
        camera,
        detector,
        tracker,
        roi=cfg.get("roi"),
        on_event=on_event,
        on_event_closed=on_event_closed,
        preprocessing=cfg.get("preprocessing"),
    )
