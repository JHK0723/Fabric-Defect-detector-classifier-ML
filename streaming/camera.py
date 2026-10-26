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
