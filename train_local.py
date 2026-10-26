import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
"""
Local GPU training script for Fabric Defect Detector & Classifier.
Downloads Bangladesh + TILDA datasets from Mendeley, builds datasets with
CLAHE lighting correction, trains YOLO11s-obb (detector) and YOLO11s-cls
(classifier), and reports full accuracy metrics.

Usage:
    python train_local.py --task all        # train both detector and classifier
    python train_local.py --task cls        # classifier only (~30-40 min on RTX 3050)
    python train_local.py --task det        # detector only (~60-90 min on RTX 3050)
    python train_local.py --task eval       # just run eval on existing best.pt files

Outputs:
    runs/det/weights/best.pt
    runs/cls/weights/best.pt
    logs/train_results.txt          <- printed accuracy summary
"""
import argparse, os, subprocess, sys, zipfile
from pathlib import Path

# ------------------- helpers -------------------------------------------------

def check_gpu():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available! Make sure you installed torch+cu121:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[GPU] {name}  ({vram:.1f} GB VRAM)")
    return vram


def download_mendeley(did, expected_size, label, dest_zip):
    dest_zip = Path(dest_zip)
    if dest_zip.exists() and dest_zip.stat().st_size == expected_size:
        print(f"[download] {label}: already downloaded.")
        return
    print(f"[download] {label}: downloading (~{expected_size // 1024 // 1024} MB)...")
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    curl_bin = "C:\\Windows\\System32\\curl.exe" if os.path.exists("C:\\Windows\\System32\\curl.exe") else "curl"
    subprocess.run(
        [curl_bin, "-L", "--retry", "10", "--retry-delay", "4",
         "-C", "-", "-o", str(dest_zip),
         f"https://data.mendeley.com/public-api/zip/{did}/download/1"],
        check=True,
    )
    if dest_zip.stat().st_size != expected_size:
        raise RuntimeError(f"{label}: incomplete download ({dest_zip.stat().st_size} of {expected_size}) -- rerun.")
    print(f"[download] {label}: done.")


def extract(zip_path, out_dir):
    out_dir = Path(out_dir)
    if out_dir.is_dir():
        return
    print(f"[extract] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        bad = [n for n in zf.namelist() if n.startswith("/") or ".." in n.split("/")]
        assert not bad, f"unsafe paths: {bad[:3]}"
        zf.extractall(out_dir)
    print("[extract] done.")


def find_tilda_root(base):
    for d, dirs, _ in os.walk(base):
        if os.path.basename(d) == "TILDA-OBB-ALL":
            return d
    raise FileNotFoundError("TILDA-OBB-ALL folder not found inside extracted archive.")


def print_section(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ------------------- dataset download ----------------------------------------

SOURCES = {
    "bd":    ("h2ytgy2pmb", 699967798, "Bangladesh fabric set (Hole / Broken End / Oil)"),
    "tilda": ("8y46pm2m6y", 193352808, "TILDA-OBB (oriented boxes)"),
}

DATA_DIR  = Path("data/raw")
BUILD_DIR = Path("data/built")
LOG_DIR   = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def prepare_data(lighting="clahe"):
    print_section("Downloading & extracting datasets")
    for key, (did, size, label) in SOURCES.items():
        z = DATA_DIR / f"{key}.zip"
        download_mendeley(did, size, label, z)
        extract(z, DATA_DIR / key)

    tilda_root = find_tilda_root(DATA_DIR / "tilda")
    bd_root    = str(DATA_DIR / "bd")
    pack_path  = "data/label_pack.json.gz"

    print_section(f"Building datasets (lighting='{lighting}')")
    sys.path.insert(0, str(Path(__file__).parent / "training"))
    import importlib, build_dataset
    importlib.reload(build_dataset)
    build_dataset.build(bd_root, tilda_root, pack_path, str(BUILD_DIR), gray=True, lighting=lighting)
    return str(BUILD_DIR)


# ------------------- training ------------------------------------------------

def train_classifier(built_dir, vram_gb, epochs=45):
    from ultralytics import YOLO
    print_section("Training CLASSIFIER (YOLO11s-cls)")
    batch = 64 if vram_gb >= 6 else 32
    print(f"[cls] batch={batch}  epochs={epochs}")
    model = YOLO("yolo11s-cls.pt")
    model.train(
        data=os.path.join(built_dir, "cls"),
        epochs=epochs,
        imgsz=224,
        batch=batch,
        patience=10,
        optimizer="AdamW",
        lr0=0.0006,
        weight_decay=0.01,
        label_smoothing=0.1,
        dropout=0.3,
        degrees=0,
        fliplr=0.5,
        flipud=0.5,
        scale=0.3,
        translate=0.1,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0.35,
        erasing=0.2,
        auto_augment=None,
        project="runs",
        name="cls",
        exist_ok=True,
        plots=True,
        seed=0,
        verbose=True,
        device=0,
    )
    print("[cls] Training done. Evaluating on test set...")
    best_file = find_or_copy_weights(
        [Path("runs/cls/weights/best.pt"), Path("runs/classify/runs/cls/weights/best.pt")],
        Path("runs/cls/weights/best.pt")
    )
    best = YOLO(str(best_file))
    metrics = best.val(data=os.path.join(built_dir, "cls"), split="test", device=0, verbose=True)
    return metrics


def find_or_copy_weights(candidates, standard_target):
    import shutil
    target = Path(standard_target)
    for c in candidates:
        cp = Path(c)
        if cp.exists():
            if cp.resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cp, target)
            return target
    # Recursive search under runs/
    for f in Path("runs").rglob("best.pt"):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        return target
    raise FileNotFoundError(f"Could not find best.pt in {candidates} or under runs/")


def train_detector(built_dir, vram_gb, epochs=80):
    from ultralytics import YOLO
    print_section("Training DETECTOR (YOLO11s-obb)")
    batch = 16 if vram_gb >= 6 else 8
    print(f"[det] batch={batch}  epochs={epochs}")
    model = YOLO("yolo11s-obb.pt")
    model.train(
        data=os.path.join(built_dir, "det/data.yaml"),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        patience=20,
        cos_lr=True,
        degrees=0,
        fliplr=0.5,
        flipud=0.5,
        scale=0.4,
        translate=0.1,
        mosaic=1.0,
        close_mosaic=10,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0.35,
        project="runs",
        name="det",
        exist_ok=True,
        plots=True,
        seed=0,
        verbose=True,
        device=0,
    )
    print("[det] Training done. Evaluating on test sets...")
    best_file = find_or_copy_weights(
        [Path("runs/det/weights/best.pt"), Path("runs/obb/runs/det/weights/best.pt")],
        Path("runs/det/weights/best.pt")
    )
    best = YOLO(str(best_file))
    results = {}
    for tag, yaml in [
        ("ALL",   "data.yaml"),
        ("OWN",   "data_own.yaml"),
        ("TILDA", "data_tilda.yaml"),
    ]:
        yaml_path = os.path.join(built_dir, "det", yaml)
        if not os.path.exists(yaml_path):
            continue
        m = best.val(data=yaml_path, split="test", device=0, verbose=False,
                     project="runs", name=f"test_{yaml.split('.')[0]}", exist_ok=True)
        results[tag] = m
    return results


# ------------------- eval only -----------------------------------------------

def eval_only(built_dir):
    from ultralytics import YOLO
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print_section("Evaluating existing models")
    summary = []

    cls_candidates = [Path("runs/cls/weights/best.pt"), Path("runs/classify/runs/cls/weights/best.pt")]
    cls_best = next((p for p in cls_candidates if p.exists()), None)
    if cls_best:
        print(f"\n[eval] Classifier: {cls_best}")
        m = YOLO(str(cls_best)).val(
            data=os.path.join(built_dir, "cls"), split="test",
            device=device, verbose=True
        )
        top1 = getattr(m, "top1", None)
        top5 = getattr(m, "top5", None)
        summary.append(f"Classifier top-1 acc: {top1:.3f}  top-5 acc: {top5:.3f}" if top1 else "Classifier: see above")
    else:
        print("[eval] No classifier best.pt found")

    det_candidates = [Path("runs/det/weights/best.pt"), Path("runs/obb/runs/det/weights/best.pt")]
    det_best = next((p for p in det_candidates if p.exists()), None)
    if det_best:
        print(f"\n[eval] Detector: {det_best}")
        for tag, yaml in [("ALL", "data.yaml"), ("OWN", "data_own.yaml"), ("TILDA", "data_tilda.yaml")]:
            yp = os.path.join(built_dir, "det", yaml)
            if not os.path.exists(yp):
                continue
            m = YOLO(str(det_best)).val(data=yp, split="test", device=device, verbose=False,
                                        project="runs", name=f"eval_{tag.lower()}", exist_ok=True)
            summary.append(f"Detector [{tag}] mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  P={m.box.mp:.3f}  R={m.box.mr:.3f}")
    else:
        print("[eval] No detector best.pt found at runs/det/weights/best.pt")

    return summary


# ------------------- main ----------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Local GPU training for fabric defect project")
    ap.add_argument("--task",     choices=["all", "cls", "det", "eval"], default="all")
    ap.add_argument("--lighting", default="clahe",
                    choices=["clahe", "illumination_norm", "combined", "gamma", "none"],
                    help="Lighting correction applied when building the dataset")
    ap.add_argument("--cls-epochs", type=int, default=45)
    ap.add_argument("--det-epochs", type=int, default=80)
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip downloading (if datasets already in data/raw/)")
    args = ap.parse_args()

    vram = check_gpu()
    lighting = None if args.lighting == "none" else args.lighting
    summary = []

    if args.task == "eval":
        # Just evaluate -- data must already be built
        built = str(BUILD_DIR)
    else:
        built = prepare_data(lighting=lighting) if not args.skip_download else str(BUILD_DIR)

    if args.task in ("all", "cls"):
        train_classifier(built, vram, epochs=args.cls_epochs)
    if args.task in ("all", "det"):
        train_detector(built, vram, epochs=args.det_epochs)

    # Final accuracy summary
    print_section("Evaluating best weights on test split")
    summary = eval_only(built)

    print_section("FINAL ACCURACY SUMMARY")
    for line in summary:
        print(" ", line)

    report = "\n".join(summary)
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / "train_results.txt").write_text(report, encoding="utf-8")
    print(f"\n[done] Results saved to logs/train_results.txt")


if __name__ == "__main__":
    main()
