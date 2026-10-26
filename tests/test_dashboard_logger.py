"""Tests for SQLite DefectLogger and dashboard mock modules."""
import tempfile
import shutil
from pathlib import Path
import numpy as np

from defect_logging.defect_logger import DefectLogger
from dashboard.mocks import SyntheticFabricCamera, BlobDetector


def test_defect_logger_lifecycle():
    tmp_dir = tempfile.mkdtemp()
    try:
        db_path = str(Path(tmp_dir) / "test_inspection.db")
        snap_dir = str(Path(tmp_dir) / "snapshots")
        logger = DefectLogger(db_path=db_path, snapshot_dir=snap_dir)

        # 1. Start session
        sess_id = logger.start_session(fabric_id="Roll-Test", fabric_length_m=12.5)
        assert sess_id.startswith("sess_")

        # 2. Log event with dummy image
        dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        event = {
            "event_id": 1,
            "class": "hole",
            "confidence": 0.92,
            "bbox": [10, 10, 40, 40],
            "first_frame": 5,
            "last_frame": 8,
            "frames_seen": 4
        }
        snap_path = logger.log_event(sess_id, event, frame=dummy_frame)
        assert snap_path != ""
        assert Path(snap_path).exists()

        # 3. Verify event retrieval
        df = logger.events_df(sess_id)
        assert len(df) == 1
        assert df.iloc[0]["defect_class"] == "hole"
        assert df.iloc[0]["confidence"] == 0.92

        # 4. End session
        logger.end_session(sess_id, frames_processed=100)
        s_df = logger.sessions_df()
        assert len(s_df) == 1
        assert s_df.iloc[0]["frames_processed"] == 100
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_synthetic_camera_and_blob_detector():
    cam = SyntheticFabricCamera(width=320, height=240, fps=30, use_test_images=False)
    cam.start()
    try:
        packet = cam.get(timeout=2.0)
        assert packet is not None
        assert packet.frame.shape == (240, 320, 3)
        assert packet.frame_id > 0

        detector = BlobDetector()
        dets = detector.detect(packet.frame)
        assert isinstance(dets, list)
    finally:
        cam.stop()
