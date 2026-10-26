"""
Probe webcam indices or preview any source with live FPS.
  python streaming/test_camera.py --probe                     # list working webcam indices
  python streaming/test_camera.py --source 1                  # preview webcam 1
  python streaming/test_camera.py --source http://192.168.43.1:8080/video
Keys: s = save snapshot to logs/, q = quit
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streaming.camera import OpenCVCamera  # noqa: E402


def probe(max_index=6):
    api = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    for i in range(max_index):
        cap = cv2.VideoCapture(i, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = cap.read()
            shape = frame.shape if ok else None
            print(f"index {i}: OPEN  first frame ok={ok} shape={shape}")
        else:
            print(f"index {i}: -")
        cap.release()


def preview(source, width, height):
    cam = OpenCVCamera(source, width=width, height=height).start()
    Path("logs").mkdir(exist_ok=True)
    last_id = -1
    print("Press s to save a snapshot, q to quit")
    while True:
        pkt = cam.get(last_id, timeout=2.0)
        if pkt is None:
            print("No frame:", cam.last_error or "waiting...")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        last_id = pkt.frame_id
        img = pkt.frame.copy()
        h, w = img.shape[:2]
        cv2.putText(img, f"{w}x{h}  camera FPS {cam.fps:.1f}  dropped {cam.dropped_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("camera test", img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            p = f"logs/snapshot_{int(time.time())}.jpg"
            cv2.imwrite(p, pkt.frame)
            print("saved", p)
        elif k == ord("q"):
            break
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--source", default="0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    probe() if a.probe else preview(a.source, a.width, a.height)
