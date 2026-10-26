import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os, time
from pathlib import Path
from ultralytics import YOLO

DATA_DIR = os.path.abspath("data/built/cls")

MODELS = [
    {
        "name": "YOLO11n-cls",
        "weights": "yolo11n-cls.pt",
        "save_name": "cls_yolo11n",
        "tag": "YOLO11 Nano (Ultra-lightweight Edge)"
    },
    {
        "name": "YOLOv8s-cls",
        "weights": "yolov8s-cls.pt",
        "save_name": "cls_yolov8s",
        "tag": "YOLOv8 Small (Prior Generation Benchmark)"
    },
]

def train_and_eval(model_cfg, epochs=45):
    name = model_cfg["name"]
    weights = model_cfg["weights"]
    save_name = model_cfg["save_name"]
    
    print("\n" + "=" * 60)
    print(f"  Training Model: {name} ({model_cfg['tag']})")
    print("=" * 60)
    
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = YOLO(weights)
    
    # Train
    model.train(
        data=DATA_DIR,
        epochs=epochs,
        imgsz=224,
        batch=32,
        workers=2,
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
        name=save_name,
        exist_ok=True,
        plots=True,
        seed=0,
        verbose=True,
        device=0,
    )
    
    # Find best.pt
    candidates = [
        Path("runs") / save_name / "weights" / "best.pt",
        Path("runs") / "classify" / "runs" / save_name / "weights" / "best.pt",
    ]
    best_file = None
    for c in candidates:
        if c.exists():
            best_file = c
            break
    if not best_file:
        for f in Path("runs").rglob("*.pt"):
            if save_name in str(f) and f.name == "best.pt":
                best_file = f
                break
                
    if not best_file:
        raise FileNotFoundError(f"Could not find best.pt for {save_name}")
        
    print(f"\n[eval] Evaluating best weights on test split: {best_file}")
    best_model = YOLO(str(best_file))
    metrics = best_model.val(data=DATA_DIR, split="test", device=0, verbose=True)
    return best_file, metrics

def evaluate_weights(weights_path):
    import gc, torch
    model = YOLO(str(weights_path))
    metrics = model.val(data=DATA_DIR, split="test", device=0, verbose=False)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics

def run_benchmark():
    results = []
    
    # 1. Existing YOLO11s-cls
    print("Evaluating trained baseline YOLO11s-cls...")
    yolo11s_path = Path("runs/cls/weights/best.pt")
    if not yolo11s_path.exists():
        yolo11s_path = Path("runs/classify/runs/cls/weights/best.pt")
    
    m_11s = evaluate_weights(yolo11s_path)
    model_11s = YOLO(str(yolo11s_path))
    
    # Extract params and flops if available
    n_params_11s = sum(p.numel() for p in model_11s.model.parameters()) / 1e6
    top1_11s = getattr(m_11s, "top1", 0.0)
    top5_11s = getattr(m_11s, "top5", 0.0)
    speed_11s = m_11s.speed.get("inference", 0.0) if hasattr(m_11s, "speed") else 0.0
    size_mb_11s = yolo11s_path.stat().st_size / (1024 * 1024)
    
    results.append({
        "name": "YOLO11s-cls",
        "tag": "YOLO11 Small (Current Baseline)",
        "params_m": f"{n_params_11s:.2f}M",
        "top1": f"{top1_11s*100:.2f}%",
        "top5": f"{top5_11s*100:.2f}%",
        "latency_ms": f"{speed_11s:.2f} ms",
        "size_mb": f"{size_mb_11s:.1f} MB",
        "weights": str(yolo11s_path)
    })
    
    # 2. Train and evaluate additional models
    for cfg in MODELS:
        best_pt, m = train_and_eval(cfg)
        model = YOLO(str(best_pt))
        n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
        top1 = getattr(m, "top1", 0.0)
        top5 = getattr(m, "top5", 0.0)
        speed = m.speed.get("inference", 0.0) if hasattr(m, "speed") else 0.0
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        
        results.append({
            "name": cfg["name"],
            "tag": cfg["tag"],
            "params_m": f"{n_params:.2f}M",
            "top1": f"{top1*100:.2f}%",
            "top5": f"{top5*100:.2f}%",
            "latency_ms": f"{speed:.2f} ms",
            "size_mb": f"{size_mb:.1f} MB",
            "weights": str(best_pt)
        })
        
    # Generate and print comparison table
    print("\n" + "=" * 70)
    print("                    MODEL BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Model':<14} | {'Description':<32} | {'Params':<8} | {'Top-1 Acc':<10} | {'Top-5 Acc':<10} | {'Latency':<9} | {'Size':<8}"
    print(header)
    print("-" * len(header))
    
    report_lines = [header, "-" * len(header)]
    for r in results:
        row = f"{r['name']:<14} | {r['tag']:<32} | {r['params_m']:<8} | {r['top1']:<10} | {r['top5']:<10} | {r['latency_ms']:<9} | {r['size_mb']:<8}"
        print(row)
        report_lines.append(row)
        
    out_file = Path("logs/benchmark_results.txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[done] Benchmark saved to {out_file}")

if __name__ == "__main__":
    run_benchmark()
