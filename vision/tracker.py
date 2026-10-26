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
