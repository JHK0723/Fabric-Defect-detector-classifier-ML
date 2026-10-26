import cv2
import numpy as np
import pytest
from pathlib import Path

from vision.detector import FabricDetector
from infer_classifier import load_gray


def test_fabric_detector_default_init():
    det = FabricDetector()
    assert det.model is not None
    assert det.imgsz == 224
    assert "hole" in det.names.values()
    assert "non_defective" in det.names.values()


def test_fabric_detector_on_clean_and_defect():
    det = FabricDetector(conf=0.3)
    
    # 1. Test clean plain fabric (uniform gray with slight texture)
    clean_patch = np.full((256, 256, 3), 160, dtype=np.uint8)
    dets_clean = det.detect(clean_patch)
    assert len(dets_clean) == 0  # Should be non_defective
    
    # 2. Test user sample image if present
    sample_path = Path("test_user_fabric.jpg")
    if sample_path.exists():
        im = cv2.imread(str(sample_path))
        dets = det.detect(im)
        assert len(dets) >= 1
        assert dets[0]["class"] in ["hole", "stain", "broken_thread", "irregular_yarn"]
        assert dets[0]["confidence"] > 0.5


def test_load_gray_preprocessing():
    sample_path = Path("test_user_fabric.jpg")
    if sample_path.exists():
        im_raw = load_gray(str(sample_path))
        assert im_raw.size == (1024, 1024)
        
        im_clahe = load_gray(str(sample_path), lighting_method="clahe")
        assert im_clahe.size == (1024, 1024)
