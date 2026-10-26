"""
Build the training datasets for the fabric defect project (works locally and in Colab).

Sources
  own   : Bangladesh dataset (Mendeley h2ytgy2pmb) images + the corrected/verified labels in label_pack.json.gz
  tilda : TILDA-OBB (Mendeley 8y46pm2m6y) with its own oriented boxes

Final classes
  0 hole  1 stain  2 broken_thread  3 irregular_yarn      (+ non_defective, classifier only)

Outputs
  <out>/det/            YOLO-OBB detector dataset (images/ labels/ {train,val,test}, data.yaml, test_own.txt, test_tilda.txt)
  <out>/cls/            image-classification dataset of 256x256 patches: <split>/<class>/*.jpg
  <out>/manifest.csv    one row per image with source / split / classes
Splits never mix near-duplicates: own images are grouped by neighbouring file number, TILDA keeps its authors' split.
"""
import csv, gzip, hashlib, json, os, random, re, shutil
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

CLASSES = ["hole", "stain", "broken_thread", "irregular_yarn"]
TILDA_MAP = {0: 0, 2: 0, 3: 1, 4: 3}      # Cut->hole, Hole->hole, Stain->stain, Thread Defect->irregular_yarn
TILDA_SKIP = {1}                          # Foreign Object: not one of our classes -> the whole image is skipped
WIN = 256                                 # patch size for the classifier
GRAY = True                               # train on grayscale (3-channel) images: removes the colour cue that separates the sources


def load_rgb(path, lighting=None):
    im = Image.open(path).convert("RGB")
    if lighting:
        import cv2
        arr = np.array(im)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if lighting == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            res = clahe.apply(gray)
        elif lighting == "illumination_norm":
            blur = cv2.GaussianBlur(gray, (51, 51), 0)
            res = cv2.divide(gray, blur, scale=128)
        elif lighting == "combined":
            blur = cv2.GaussianBlur(gray, (51, 51), 0)
            flat = cv2.divide(gray, blur, scale=128)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            res = clahe.apply(flat)
        else:
            res = gray
        return Image.fromarray(res).convert("RGB")
    return im.convert("L").convert("RGB") if GRAY else im



def split_for_group(group, seed=42):
    h = int(hashlib.md5(f"{seed}:{group}".encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else "val" if h < 90 else "test"


def hbb_to_poly(xc, yc, w, h):
    x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def clamp01(p):
    return [min(1.0, max(0.0, v)) for v in p]


def collect(bd_root, tilda_root, pack_path):
    pack = json.loads(gzip.open(pack_path).read())
    # index own images by stem
    own_files = {}
    for d, _, fs in os.walk(bd_root):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                own_files[os.path.splitext(f)[0]] = os.path.join(d, f)
    items, missing = [], 0
    for stem, lines in pack.items():
        if stem not in own_files:
            missing += 1
            continue
        boxes = []
        for ln in lines:
            c, xc, yc, w, h = ln.split()
            boxes.append((int(c), clamp01(hbb_to_poly(float(xc), float(yc), float(w), float(h)))))
        kind = stem.split(" ")[0].replace("Febric", "Oil")
        n = int(re.search(r"\((\d+)\)", stem).group(1))
        items.append(dict(src="own", name="own_" + re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_"), path=own_files[stem],
                          boxes=boxes, split=split_for_group(f"{kind}:{n // 10}")))
    print(f"own: {len(items)} labelled images ({missing} labels had no image file)")
    # TILDA-OBB
    nt = skipped = 0
    for sp_dir, sp in (("train", "train"), ("valid", "val"), ("test", "test")):
        for lf in sorted(os.listdir(os.path.join(tilda_root, sp_dir, "labels"))):
            if not lf.endswith(".txt") or lf.startswith("README"):
                continue
            ip = os.path.join(tilda_root, sp_dir, "images", lf[:-4] + ".jpg")
            if not os.path.exists(ip):
                continue
            rows = [l.split() for l in open(os.path.join(tilda_root, sp_dir, "labels", lf)) if l.strip()]
            if not rows or any(int(r[0]) in TILDA_SKIP for r in rows):
                skipped += 1
                continue
            boxes = [(TILDA_MAP[int(r[0])], clamp01([float(v) for v in r[1:9]])) for r in rows]
            items.append(dict(src="tilda", name="tilda_" + re.sub(r"[^A-Za-z0-9]+", "_", lf[:-4])[:60], path=ip, boxes=boxes, split=sp))
            nt += 1
    print(f"tilda: {nt} images kept ({skipped} skipped: contain 'Foreign Object' or no label)")
    return items


def write_detector(items, out, lighting=None):
    det = os.path.join(out, "det")
    shutil.rmtree(det, ignore_errors=True)
    for sp in ("train", "val", "test"):
        os.makedirs(f"{det}/images/{sp}"); os.makedirs(f"{det}/labels/{sp}")
    tests = defaultdict(list)
    for it in items:
        dst = f"{det}/images/{it['split']}/{it['name']}.jpg"
        load_rgb(it["path"], lighting=lighting).save(dst, quality=95)
        with open(f"{det}/labels/{it['split']}/{it['name']}.txt", "w") as f:
            for c, p in it["boxes"]:
                f.write(f"{c} " + " ".join(f"{v:.6f}" for v in p) + "\n")
        if it["split"] == "test":
            tests[it["src"]].append(os.path.abspath(dst))
    for src, paths in tests.items():
        open(f"{det}/test_{src}.txt", "w").write("\n".join(paths) + "\n")
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    for tag, test in (("", "images/test"), ("_own", "test_own.txt"), ("_tilda", "test_tilda.txt")):
        open(f"{det}/data{tag}.yaml", "w").write(
            f"path: {os.path.abspath(det)}\ntrain: images/train\nval: images/val\ntest: {test}\nnames:\n{names}\n")
    cnt = Counter((it["split"], CLASSES[c]) for it in items for c, _ in it["boxes"])
    print("detector boxes per split/class:")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s} imgs={sum(i['split']==sp for i in items):4d} ", {c: cnt[(sp, c)] for c in CLASSES})


def aabb(poly, W, H):
    xs, ys = poly[0::2], poly[1::2]
    return min(xs) * W, min(ys) * H, max(xs) * W, max(ys) * H


def write_classifier(items, out, seed=0, max_defect_per_img=3, clean_per_img=2, clean_frac=0.25, lighting=None):
    rnd = random.Random(seed)
    cls_dir = os.path.join(out, "cls")
    shutil.rmtree(cls_dir, ignore_errors=True)
    rows, clean_pool = [], []
    n_def = Counter()

    def crop_save(im, x0, y0, label, name, sp, src):
        d = f"{cls_dir}/{sp}/{label}"
        os.makedirs(d, exist_ok=True)
        im.crop((x0, y0, x0 + WIN, y0 + WIN)).save(f"{d}/{name}.jpg", quality=92)
        rows.append((f"{sp}/{label}/{name}.jpg", src, label, sp))

    for it in items:
        im = load_rgb(it["path"], lighting=lighting)
        W, H = im.size
        if min(W, H) < WIN:
            continue
        bxs = [(c, aabb(p, W, H)) for c, p in it["boxes"]]
        all_boxes = [b for _, b in bxs]
        # defect patches: centred on the largest boxes
        for k, (c, (x1, y1, x2, y2)) in enumerate(sorted(bxs, key=lambda t: -(t[1][2] - t[1][0]) * (t[1][3] - t[1][1]))[:max_defect_per_img]):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if c == 2:                                   # broken thread: a thin full-height line -> random height along it
                cy = rnd.uniform(WIN / 2, H - WIN / 2)
            x0 = int(min(max(cx - WIN / 2, 0), W - WIN)); y0 = int(min(max(cy - WIN / 2, 0), H - WIN))
            # skip windows whose centre lies in a box of another class (ambiguous label)
            ccx, ccy = x0 + WIN / 2, y0 + WIN / 2
            if any(cc != c and bx1 <= ccx <= bx2 and by1 <= ccy <= by2 for cc, (bx1, by1, bx2, by2) in bxs):
                continue
            crop_save(im, x0, y0, CLASSES[c], f"{it['name']}_d{k}", it["split"], it["src"]); n_def[(it["split"], CLASSES[c])] += 1
        # clean patches: windows that touch no box (12px margin)
        got = 0
        for _ in range(30):
            if got >= clean_per_img:
                break
            x0 = rnd.randint(0, W - WIN); y0 = rnd.randint(0, H - WIN)
            if any(not (x0 + WIN < bx1 - 12 or x0 > bx2 + 12 or y0 + WIN < by1 - 12 or y0 > by2 + 12) for bx1, by1, bx2, by2 in all_boxes):
                continue
            clean_pool.append((it, im.crop((x0, y0, x0 + WIN, y0 + WIN)), got)); got += 1
    # keep clean patches at ~clean_frac of each split's defect patches, spread over all sources
    by_split = defaultdict(list)
    for it, crop, k in clean_pool:
        by_split[it["split"]].append((it, crop, k))
    for sp, lst in by_split.items():
        n_defect = sum(v for (s, _), v in n_def.items() if s == sp)
        keep = int(n_defect * clean_frac / (1 - clean_frac))
        rnd.shuffle(lst)
        for it, crop, k in lst[:keep]:
            d = f"{cls_dir}/{sp}/non_defective"
            os.makedirs(d, exist_ok=True)
            crop.save(f"{d}/{it['name']}_c{k}.jpg", quality=92)
            rows.append((f"{sp}/non_defective/{it['name']}_c{k}.jpg", it["src"], "non_defective", sp))
    with open(os.path.join(out, "manifest_cls.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "source", "class", "split"]); w.writerows(rows)
    c = Counter((r[3], r[2]) for r in rows)
    print("classifier patches per split/class:")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s}", {k: c[(sp, k)] for k in CLASSES + ["non_defective"]})
    cs = Counter((r[1], r[2]) for r in rows)
    print("by source:", {s: {k: cs[(s, k)] for k in CLASSES + ["non_defective"] if cs[(s, k)]} for s in ("own", "tilda")})


def build(bd_root, tilda_root, pack_path, out, gray=True, lighting=None):
    global GRAY
    GRAY = bool(gray)
    os.makedirs(out, exist_ok=True)
    items = collect(bd_root, tilda_root, pack_path)
    write_detector(items, out, lighting=lighting)
    write_classifier(items, out, lighting=lighting)
    with open(os.path.join(out, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "source", "split", "classes"])
        for it in items:
            w.writerow([it["name"], it["src"], it["split"], "|".join(sorted({CLASSES[c] for c, _ in it["boxes"]}))])
    return items


if __name__ == "__main__":
    import sys
    build(*sys.argv[1:5])