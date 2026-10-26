"""Vision package for preprocessing, postprocessing, tracking, detection, and real-time pipeline."""
from .postprocessing import CLASS_COLORS, color_for, draw_annotations
from .preprocessing import (
    crop_roi,
    offset_detections,
    correct_lighting,
    apply_clahe,
    normalize_illumination,
    adaptive_gamma,
)
from .tracker import DefectEventTracker
from .pipeline import InspectionPipeline, build_from_config, load_config

__all__ = [
    "CLASS_COLORS",
    "color_for",
    "draw_annotations",
    "crop_roi",
    "offset_detections",
    "correct_lighting",
    "apply_clahe",
    "normalize_illumination",
    "adaptive_gamma",
    "DefectEventTracker",
    "InspectionPipeline",
    "build_from_config",
    "load_config",
]

