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
                lambda: self._packet is not None and self._packet.frame_id > last_id, timeout
            )
            if not ok:
                return None
            self._consumed = True
            return self._packet
