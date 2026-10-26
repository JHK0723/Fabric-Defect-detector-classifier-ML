"""
Run the pipeline headless for N seconds and record system metrics once per second.
  python benchmark_pipeline.py --seconds 60 --tag usb_720p_n640
Writes logs/benchmark_<tag>.csv and prints a summary.
"""
import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil

from vision.pipeline import build_from_config, load_config


def gpu_stats():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        util, mem = out.stdout.strip().splitlines()[0].split(",")
        return float(util), float(mem)
    except Exception:
        return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--source")
    ap.add_argument("--weights")
    ap.add_argument("--imgsz", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.source is not None:
        cfg["camera"]["source"] = args.source
    if args.weights:
        cfg["model"]["weights"] = args.weights
    if args.imgsz:
        cfg["model"]["imgsz"] = args.imgsz

    pipe = build_from_config(cfg).start()
    psutil.cpu_percent(None)
    rows = []
    time.sleep(3)                      # let EMAs settle
    for i in range(args.seconds):
        time.sleep(1)
        s = pipe.stats()
        gu, gm = gpu_stats()
        rows.append([
            i,
            s["camera_fps"],
            s["inference_fps"],
            s["inference_ms"],
            s["latency_ms"],
            s["frames_processed"],
            s["frames_dropped"],
            psutil.cpu_percent(None),
            gu,
            gm,
        ])
    pipe.stop()

    Path("logs").mkdir(exist_ok=True)
    out = Path("logs") / f"benchmark_{args.tag}.csv"
    hdr = [
        "t",
        "camera_fps",
        "inference_fps",
        "inference_ms",
        "latency_ms",
        "frames_processed",
        "frames_dropped",
        "cpu_pct",
        "gpu_util_pct",
        "gpu_mem_mb",
    ]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    a = np.array([r[1:5] + r[7:9] for r in rows], dtype=float)
    names = ["camera_fps", "inference_fps", "inference_ms", "latency_ms", "cpu_pct", "gpu_util_pct"]
    print(f"\nSaved {out}")
    for j, n in enumerate(names):
        col = a[:, j]
        print(f"  {n:14s} mean {np.nanmean(col):7.1f}   p95 {np.nanpercentile(col, 95):7.1f}")
    if rows:
        last = rows[-1]
        total_frames = last[5] + last[6]
        drop_pct = (100 * last[6] / total_frames) if total_frames > 0 else 0
        print(f"  dropped frames: {last[6]} of {total_frames} ({drop_pct:.0f}%)")


if __name__ == "__main__":
    main()
