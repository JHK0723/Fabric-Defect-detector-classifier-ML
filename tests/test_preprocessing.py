import cv2
import numpy as np
import pytest

from vision.preprocessing import (
    crop_roi,
    offset_detections,
    apply_clahe,
    normalize_illumination,
    adaptive_gamma,
    correct_lighting,
)
from vision.pipeline import build_from_config


def test_crop_roi_and_offset():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    cropped, (ox, oy) = crop_roi(frame, [0.1, 0.2, 0.9, 0.8])
    assert ox == 20
    assert oy == 20
    assert cropped.shape == (60, 160, 3)

    dets = [{"bbox": [10, 10, 50, 50]}]
    shifted = offset_detections(dets, ox, oy)
    assert shifted[0]["bbox"] == [30, 30, 70, 70]


def test_clahe_shapes_and_types():
    # 2D grayscale input
    gray = np.random.randint(50, 200, (120, 160), dtype=np.uint8)
    res_2d_gray = apply_clahe(gray, to_gray=False)
    assert res_2d_gray.shape == (120, 160)
    assert res_2d_gray.dtype == np.uint8

    res_2d_bgr = apply_clahe(gray, to_gray=True)
    assert res_2d_bgr.shape == (120, 160, 3)

    # 3D BGR input
    bgr = np.random.randint(50, 200, (120, 160, 3), dtype=np.uint8)
    res_3d_gray = apply_clahe(bgr, to_gray=True)
    assert res_3d_gray.shape == (120, 160, 3)

    res_3d_bgr = apply_clahe(bgr, to_gray=False)
    assert res_3d_bgr.shape == (120, 160, 3)


def test_normalize_illumination():
    bgr = np.random.randint(80, 180, (120, 160, 3), dtype=np.uint8)
    res = normalize_illumination(bgr, blur_kernel_size=31, target_mean=128, to_gray=True)
    assert res.shape == (120, 160, 3)
    assert res.dtype == np.uint8
    # Mean of normalized output should be close to target_mean
    assert 115 < res.mean() < 140


def test_adaptive_gamma():
    # Dark image
    dark = np.full((100, 100, 3), 40, dtype=np.uint8)
    brightened = adaptive_gamma(dark, target_mean=128, to_gray=True)
    assert brightened.mean() > dark.mean()

    # Bright image
    bright = np.full((100, 100, 3), 220, dtype=np.uint8)
    dimmed = adaptive_gamma(bright, target_mean=128, to_gray=True)
    assert dimmed.mean() < bright.mean()


def test_correct_lighting_methods():
    sample = np.random.randint(40, 210, (100, 100, 3), dtype=np.uint8)
    for method in ["clahe", "illumination_norm", "combined", "gamma"]:
        out = correct_lighting(sample, method=method, to_gray=True)
        assert out.shape == (100, 100, 3)
        assert out.dtype == np.uint8

    # Edge cases
    assert correct_lighting(None) is None
    empty = np.array([])
    assert correct_lighting(empty).size == 0


def test_illumination_gradient_flattening():
    # Create an image with severe gradient (one side in heavy shadow, one side bright)
    h, w = 120, 160
    gradient = np.tile(np.linspace(0.2, 1.8, w, dtype=np.float32), (h, 1))
    base = np.full((h, w), 128, dtype=np.float32)
    uneven = np.clip(base * gradient, 0, 255).astype(np.uint8)
    uneven_bgr = cv2.cvtColor(uneven, cv2.COLOR_GRAY2BGR)

    left_raw = uneven[:, :30].mean()
    right_raw = uneven[:, -30:].mean()
    assert abs(right_raw - left_raw) > 100  # Raw image has huge disparity

    # Combined normalization flattens this gradient
    corrected = correct_lighting(uneven_bgr, method="combined", blur_kernel_size=31, to_gray=True)
    left_corr = corrected[:, :30].mean()
    right_corr = corrected[:, -30:].mean()

    # Disparity between sides should be dramatically reduced
    assert abs(right_corr - left_corr) < 20


def test_pipeline_config_integration():
    cfg = {
        "camera": {"source": 0, "width": 640, "height": 480, "fps": 30},
        "roi": [0.1, 0.1, 0.9, 0.9],
        "preprocessing": {
            "lighting_correction": True,
            "method": "combined",
            "clip_limit": 2.5,
            "tile_grid_size": [8, 8],
            "blur_kernel_size": 41,
            "to_gray": True,
        },
        "tracker": {
            "min_confidence": 0.5,
            "confirm_confidence": 0.7,
            "confirm_frames": 3,
            "max_missed": 5,
            "iou_threshold": 0.2,
            "max_center_dist": 0.15,
        },
    }

    # Test pipeline instantiates with preprocessing options
    class DummyCamera:
        fps = 30
        is_connected = True
        dropped_frames = 0
        def start(self): pass
        def stop(self): pass
        def get(self, *a, **kw): return None

    pipe = build_from_config(cfg, camera=DummyCamera(), detector=None)
    assert pipe.preprocessing["lighting_correction"] is True
    assert pipe.preprocessing["method"] == "combined"
    assert pipe.preprocessing["clip_limit"] == 2.5
