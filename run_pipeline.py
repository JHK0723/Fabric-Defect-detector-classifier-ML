"""
Live detection in an OpenCV window.
  python run_pipeline.py                              # uses config/config.yaml
  python run_pipeline.py --source 1 --weights yolo26n.pt
  python run_pipeline.py --source data/videos/F01_v01.mp4
Press q in the video window to quit.
"""
import argparse

import cv2

from vision.pipeline import build_from_config, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--source")
    ap.add_argument("--weights")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.source is not None:
        cfg["camera"]["source"] = args.source
    if args.weights:
        cfg["model"]["weights"] = args.weights

    def on_event(e):
        print(f"[DEFECT #{e['event_id']}] {e['class']} conf={e['confidence']:.2f} at {e['first_seen']} "
              f"(frame {e['first_frame']})")

    def on_closed(e):
        print(f"   #{e['event_id']} left view after {e['frames_seen']} frames")

    pipe = build_from_config(cfg, on_event=on_event, on_event_closed=on_closed)
    print("Starting camera and loading model...")
    pipe.start()
    print("Running. Press q in the video window to quit.")
    try:
        while True:
            r = pipe.latest()
            if r is not None and r["frame"] is not None:
                cv2.imshow("Fabric inspection", r["frame"])
            if cv2.waitKey(15) & 0xFF == ord("q"):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print("Final stats:", pipe.stats())


if __name__ == "__main__":
    main()
