"""
Synthetic test camera and mock detector for dashboard testing without hardware.
"""
import glob
import random
import threading
import time
from pathlib import Path
import cv2
import numpy as np

from streaming.camera import BaseCamera
from streaming.frame_queue import FramePacket, LatestFrameBuffer


class SyntheticFabricCamera(BaseCamera):
    """
    Generates moving fabric weave simulation or cycles through test dataset images
    to demonstrate live defect inspection without needing a physical camera.
    """
    def __init__(self, width=640, height=480, fps=25, use_test_images=True):
        self.width = width
        self.height = height
        self._target_fps = fps
        self.use_test_images = use_test_images
        
        self.queue = LatestFrameBuffer()
        self._running = False
        self._thread = None
        self._frame_id = 0
        self._last_time = time.perf_counter()
        self._fps = float(fps)
        self._connected = False
        
        # Collect test images if available
        self.test_images = []
        if use_test_images:
            patterns = [
                "data/built/cls/test/*/*.jpg",
                "data/built/cls/test/*/*.png",
                "data/raw/bd/*/*.jpg"
            ]
            for p in patterns:
                self.test_images.extend(glob.glob(p))
            random.seed(42)
            random.shuffle(self.test_images)
        self._img_idx = 0

    def start(self):
        if self._running:
            return self
        self._running = True
        self._connected = True
        self._thread = threading.Thread(target=self._capture_loop, name="synth_cam", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._connected = False

    def get(self, last_id=-1, timeout=1.0):
        return self.queue.get(last_id=last_id, timeout=timeout)

    @property
    def fps(self):
        return self._fps

    @property
    def is_connected(self):
        return self._connected

    @property
    def dropped_frames(self):
        return self.queue.dropped

    def _generate_synthetic_weave(self, offset):
        """Generate a procedural woven fabric texture."""
        h, w = self.height, self.width
        x = np.linspace(0, 80 * np.pi, w)
        y = np.linspace(0, 60 * np.pi, h)
        xx, yy = np.meshgrid(x, y)
        pattern = (np.sin(xx + offset) * np.cos(yy + offset) * 40 + 140).astype(np.uint8)
        frame = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
        
        # Periodic simulated defect (every ~100 frames)
        if (self._frame_id % 120) < 30:
            cx, cy = w // 2, h // 2
            cv2.circle(frame, (cx, cy), 28, (30, 30, 30), -1)  # Simulated hole
            cv2.putText(frame, "[SIMULATED HOLE]", (cx - 70, cy - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return frame

    def _capture_loop(self):
        interval = 1.0 / self._target_fps
        offset = 0.0
        
        while self._running:
            t0 = time.perf_counter()
            self._frame_id += 1
            
            # Use real test images if available (change image every ~40 frames to simulate moving fabric roll)
            if self.test_images:
                if self._frame_id % 35 == 0:
                    self._img_idx = (self._img_idx + 1) % len(self.test_images)
                img_path = self.test_images[self._img_idx]
                im = cv2.imread(img_path)
                if im is not None:
                    frame = cv2.resize(im, (self.width, self.height))
                else:
                    frame = self._generate_synthetic_weave(offset)
            else:
                frame = self._generate_synthetic_weave(offset)
                offset += 0.15
                
            now = time.perf_counter()
            dt = now - self._last_time
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            self._last_time = now
            
            packet = FramePacket(frame=frame, frame_id=self._frame_id, timestamp=now)
            self.queue.put(packet)
            
            elapsed = time.perf_counter() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)


class BlobDetector:
    """Mock detector that finds dark spots or returns simulated defect boxes."""
    def __init__(self, conf=0.5):
        self.conf = float(conf)
        self.names = {0: "hole", 1: "stain", 2: "broken_thread", 3: "irregular_yarn"}
        self.last_probs = {"hole": 0.0, "stain": 0.0, "broken_thread": 0.0, "irregular_yarn": 0.0, "non_defective": 1.0}

    def warmup(self, n=1):
        pass

    def detect(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Threshold for dark spots (holes)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        out = []
        for c in cnts:
            area = cv2.contourArea(c)
            if 300 < area < (w * h * 0.3):
                x, y, bw, bh = cv2.boundingRect(c)
                out.append({
                    "class": "hole",
                    "class_id": 0,
                    "confidence": 0.88,
                    "bbox": [x, y, x + bw, y + bh]
                })
        if out:
            self.last_probs = {"hole": 0.88, "stain": 0.05, "broken_thread": 0.03, "irregular_yarn": 0.02, "non_defective": 0.02}
        else:
            self.last_probs = {"hole": 0.02, "stain": 0.03, "broken_thread": 0.01, "irregular_yarn": 0.01, "non_defective": 0.93}
        return out
