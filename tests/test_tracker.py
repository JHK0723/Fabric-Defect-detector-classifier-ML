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
