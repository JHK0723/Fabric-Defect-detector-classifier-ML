"""
Classify fabric defects with the trained patch classifier (models/fabric_classifier_best.pt).

The model was trained on 256x256 GRAYSCALE patches, so an image is converted to gray and scanned with a sliding window.

  python infer_classifier.py image.jpg                      # scan a whole image, print every window that is not "non_defective"
  python infer_classifier.py image.jpg --stride 128 --conf 0.6 --save out.jpg
  python infer_classifier.py patch.jpg --patch               # the file already is one 256x256 patch: print the class probabilities

Classes: hole, stain, broken_thread, irregular_yarn, non_defective
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

WIN = 256
COLORS = {"hole": (0, 255, 0), "stain": (255, 140, 0), "broken_thread": (255, 0, 0), "irregular_yarn": (255, 0, 255)}


def load_gray(path, lighting_method=None):
    im = Image.open(path).convert("L")
    if lighting_method:
        import numpy as np
        from vision.preprocessing import correct_lighting
        arr = np.array(im)
        corr = correct_lighting(arr, method=lighting_method, to_gray=False)
        return Image.fromarray(corr).convert("RGB")
    return im.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", default=str(Path(__file__).parent / "models" / "fabric_classifier_best.pt"))
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--conf", type=float, default=0.6, help="minimum probability for a window to count as a defect")
    ap.add_argument("--patch", action="store_true", help="input is a single 256x256 patch")
    ap.add_argument("--lighting", choices=["clahe", "illumination_norm", "combined", "gamma"], default=None,
                    help="apply lighting/illumination correction before inference")
    ap.add_argument("--save", help="write an annotated copy of the image here")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    names = model.names
    im = load_gray(a.image, lighting_method=a.lighting)

    if a.patch:
        r = model.predict(im, imgsz=224, verbose=False)[0]
        for i in r.probs.data.argsort(descending=True).tolist():
            print(f"{names[i]:15s} {float(r.probs.data[i]):.3f}")
        return

    W, H = im.size
    if min(W, H) < WIN:
        raise SystemExit(f"image is smaller than one {WIN}px window ({W}x{H})")
    xs = list(range(0, W - WIN + 1, a.stride)) or [0]
    ys = list(range(0, H - WIN + 1, a.stride)) or [0]
    if xs[-1] != W - WIN:
        xs.append(W - WIN)
    if ys[-1] != H - WIN:
        ys.append(H - WIN)
    boxes = [(x, y) for y in ys for x in xs]
    results = model.predict([im.crop((x, y, x + WIN, y + WIN)) for x, y in boxes], imgsz=224, verbose=False)
    out = im.copy()
    draw = ImageDraw.Draw(out)
    hits = []
    for (x, y), r in zip(boxes, results):
        k = int(r.probs.top1)
        p = float(r.probs.top1conf)
        cls = names[k]
        if cls != "non_defective" and p >= a.conf:
            hits.append((cls, p, x, y))
            color = COLORS.get(cls, (255, 255, 0))
            draw.rectangle([x, y, x + WIN, y + WIN], outline=color, width=3)
            draw.text((x + 6, y + 6), f"{cls} {p:.2f}", fill=color)
    print(f"{len(boxes)} windows scanned, {len(hits)} flagged as defects")
    for cls, p, x, y in sorted(hits, key=lambda t: -t[1]):
        print(f"  {cls:15s} p={p:.2f}  window at x={x}, y={y}")
    if a.save:
        out.save(a.save)
        print("saved", a.save)


if __name__ == "__main__":
    main()
