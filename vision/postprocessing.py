"""Drawing boxes/labels/status on frames. Pure OpenCV, no model knowledge."""
import cv2

CLASS_COLORS = {
    "hole": (0, 0, 255),
    "stain": (0, 165, 255),
    "broken_thread": (255, 0, 255),
    "irregular_yarn": (255, 128, 0),
}


def color_for(name):
    return CLASS_COLORS.get(name, (0, 255, 255))


def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(th + 6, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_annotations(frame, detections, tracks=(), roi=None, stats=None):
    img = frame.copy()
    h, w = img.shape[:2]
    if roi:
        x1, y1, x2, y2 = roi
        cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (180, 180, 180), 1)
    # raw detections: thin boxes
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        c = color_for(d["class"])
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 1)
        _label(img, f"{d['class']} {d['confidence']:.2f}", x1, y1, c)
    # confirmed defects: thick boxes with event number
    for t in tracks:
        if t.get("confirmed"):
            x1, y1, x2, y2 = t["bbox"]
            c = color_for(t["class"])
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 4)
            _label(img, f"DEFECT #{t['event_id']} {t['class'].upper()}", x1, y2 + 24, c)
    if stats:
        txt = (f"cam {stats.get('camera_fps', 0):.0f} FPS | AI {stats.get('inference_fps', 0):.1f} FPS | "
               f"{stats.get('inference_ms', 0):.0f} ms | latency {stats.get('latency_ms', 0):.0f} ms")
        cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img
