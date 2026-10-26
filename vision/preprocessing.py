"""Region-of-interest cropping and lighting / illumination preprocessing."""
import cv2
import numpy as np


def crop_roi(frame, roi):
    if not roi:
        return frame, (0, 0)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    return frame[py1:py2, px1:px2], (px1, py1)


def offset_detections(detections, ox, oy):
    """Shift boxes found in the ROI crop back into full-frame coordinates."""
    if ox == 0 and oy == 0:
        return detections
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        d["bbox"] = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
    return detections


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8), to_gray=True):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances local contrast while avoiding over-amplification of noise in homogeneous textile areas.
    """
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))

    if len(img.shape) == 2:
        res = clahe.apply(img)
        return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR) if to_gray else res

    if to_gray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = clahe.apply(gray)
        return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def normalize_illumination(img, blur_kernel_size=51, target_mean=128, to_gray=True):
    """Illumination surface estimation and background division.
    Estimates the low-frequency illumination component via Gaussian smoothing and divides the image
    by it, effectively removing uneven lighting, side shadows, and vignetting across the fabric.
    """
    k = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    blur = cv2.GaussianBlur(gray, (k, k), 0)
    norm = cv2.divide(gray, blur, scale=target_mean)

    if to_gray:
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        ratio = (norm.astype(np.float32) + 1.0) / (gray.astype(np.float32) + 1.0)
        res = np.clip(img.astype(np.float32) * ratio[:, :, None], 0, 255).astype(np.uint8)
        return res
    return norm


def adaptive_gamma(img, target_mean=128.0, to_gray=True):
    """Adaptive dynamic range expansion / compression using automated gamma estimation."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    mean_val = float(np.mean(gray))
    if mean_val <= 1e-3 or mean_val >= 254.9:
        out_gray = gray
    else:
        gamma = float(np.clip(np.log(target_mean / 255.0) / np.log(mean_val / 255.0), 0.4, 2.5))
        lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).astype(np.uint8)
        out_gray = cv2.LUT(gray, lut)

    if to_gray:
        return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = out_gray
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out_gray


def correct_lighting(
    frame,
    method="clahe",
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    blur_kernel_size=51,
    target_mean=128,
    to_gray=True,
    **kwargs,
):
    """Unified lighting and illumination correction for fabric inspection frames.

    Args:
        frame: Input image (BGR or Grayscale, uint8).
        method: One of:
            - 'clahe': Adaptive contrast enhancement (best for micro-defects & low contrast).
            - 'illumination_norm': Background division (best for shadows, vignetting, uneven lamps).
            - 'combined': Illumination normalization followed by CLAHE (recommended for tough lighting).
            - 'gamma': Dynamic exposure tone adjustment.
        clip_limit: Contrast clip limit for CLAHE.
        tile_grid_size: Number of tiles (rows, cols) for CLAHE.
        blur_kernel_size: Gaussian kernel size for illumination surface estimation (must be odd).
        target_mean: Target intensity for background normalization.
        to_gray: When True, returns a 3-channel grayscale image (shape H, W, 3) matching YOLO model expectation.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return frame

    method = (method or "clahe").lower()

    if method == "clahe":
        return apply_clahe(frame, clip_limit=clip_limit, tile_grid_size=tile_grid_size, to_gray=to_gray)
    elif method == "illumination_norm":
        return normalize_illumination(
            frame, blur_kernel_size=blur_kernel_size, target_mean=target_mean, to_gray=to_gray
        )
    elif method == "combined":
        # Step 1: Flatten uneven lighting gradients
        flat = normalize_illumination(
            frame, blur_kernel_size=blur_kernel_size, target_mean=target_mean, to_gray=to_gray
        )
        # Step 2: Enhance micro-textures and defect boundaries
        return apply_clahe(flat, clip_limit=clip_limit, tile_grid_size=tile_grid_size, to_gray=to_gray)
    elif method == "gamma":
        return adaptive_gamma(frame, target_mean=target_mean, to_gray=to_gray)
    else:
        # Fallback if unknown method specified
        return apply_clahe(frame, clip_limit=clip_limit, tile_grid_size=tile_grid_size, to_gray=to_gray)

