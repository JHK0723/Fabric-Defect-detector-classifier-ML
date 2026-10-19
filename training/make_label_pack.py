"""Pack the verified/corrected labels of the Bangladesh (Mendeley h2ytgy2pmb) images into one small JSON.gz.
Keys are the ORIGINAL image stems ("Hole (12)", "Oil Febric (345)", "BrokenEnd (7)"), values are YOLO-HBB lines "cls xc yc w h".
cls ids here are FINAL class ids: 0 hole, 1 stain, 2 broken_thread, 3 irregular_yarn."""
import json, gzip, base64, os, re, sys
from PIL import Image
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("dataset", help="folder that contains Hole/, Broken End/ and the corrected label folders hole_labels_redone/ and oil_labels_draft/")
ap.add_argument("autolabel_json", help="output of broken_end_autolabel.py")
args = ap.parse_args()
B = args.dataset.rstrip("/") + "/"
pack = {}
# Hole: every one of the 649 images re-boxed by eye and re-checked
for f in os.listdir(B + "hole_labels_redone"):
    n = re.search(r"Hole_(\d+)\.txt", f).group(1)
    L = [l.split() for l in open(B + "hole_labels_redone/" + f) if l.strip()]
    pack[f"Hole ({n})"] = [f"0 {p[1]} {p[2]} {p[3]} {p[4]}" for p in L]
# Oil -> stain: 564 images with clearly visible stains boxed (the other 278 are NOT in the pack)
for f in os.listdir(B + "oil_labels_draft"):
    n = re.search(r"\((\d+)\)", f).group(1)
    L = [l.split() for l in open(B + "oil_labels_draft/" + f) if l.strip()]
    pack[f"Oil Febric ({n})"] = [f"1 {p[1]} {p[2]} {p[3]} {p[4]}" for p in L]
# Broken end: boxes found by the vertical-line detector (verified on 48 random samples, 48/48 correct)
det = json.load(open(args.autolabel_json))
ok = lambda v: v["status"] == "ok" and v["peak"] >= 6 and v["cons"] >= 0.75 and v["rival"] < 0.6 * v["peak"]
nbe = 0
for n, v in det.items():
    if not ok(v): continue
    W, H = Image.open(B + f"Broken End/BrokenEnd ({n}).jpg").size
    l, r = v["l"], v["r"]; wpx = max(r - l + 1 + 8, 14)
    xc = (l + r + 1) / 2 / W; w = wpx / W
    pack[f"BrokenEnd ({n})"] = [f"2 {xc:.6f} 0.500000 {w:.6f} 1.000000"]; nbe += 1
raw = json.dumps(pack, separators=(",", ":")).encode()
gz = gzip.compress(raw, 9); b64 = base64.b64encode(gz).decode()
out = os.path.dirname(os.path.abspath(__file__)) + "/label_pack.json.gz"
open(out, "wb").write(gz); open(out + ".b64", "w").write(b64)
print("hole", sum(k.startswith("Hole") for k in pack), "| stain", sum(k.startswith("Oil") for k in pack), "| broken_thread", nbe)
print("raw KB", len(raw) // 1000, "| gz KB", len(gz) // 1000, "| base64 KB", len(b64) // 1000)
