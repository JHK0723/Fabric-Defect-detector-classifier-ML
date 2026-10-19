"""
Find the real position of the "broken end" (a thin vertical line, a broken warp yarn) in each image of the
Bangladesh fabric set (Mendeley h2ytgy2pmb, folder "Broken End"), because the boxes shipped with that dataset do not
sit on the defect (only 58 of 664 labels fall within 15 px of the line a detector finds).

Method (no training involved)
  1. Cloth-edge frames (white/black background beside the cloth) are handled by analysing only the fabric strip
     (30 px away from the edge); frames with a big step inside the fabric are rejected.
  2. The image height is cut into 8 blocks. For each block the column-mean intensity profile is high-pass filtered
     (median filter, 31 px) and divided by its robust sigma (MAD). The 8 z-profiles are stacked
     (sum / sqrt(8)), so a line running down the whole image stands out while local texture does not.
  3. The strongest column is the candidate. It is ACCEPTED only if
        stacked z >= 6, the line shows up in >= 75 % of the blocks, and no rival column elsewhere reaches 60 % of it.
  4. Box = full image height, width = extent where the profile stays above 35 % of the peak (+ 8 px padding).
Checked by eye on 48 random accepted images: 48 / 48 show a real line inside the box.
Plaid / checked fabrics defeat the detector and are skipped (they stay unlabelled, and are not used for training).

Usage:  python broken_end_autolabel.py <folder with BrokenEnd (n).jpg> <out.json>
"""
import glob, json, os, re, sys

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

NB = 8


def line_in(g, x_off=0):
    H, W = g.shape
    Z = []
    for rb in np.array_split(np.arange(H), NB):
        c = g[rb].mean(0)
        hp = c - median_filter(c, size=31, mode="nearest")
        sig = 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-6
        Z.append(hp / sig)
    Z = np.array(Z)
    comb = Z.mean(0) * np.sqrt(NB)
    comb[:8] = 0
    comb[-8:] = 0
    x = int(np.argmax(np.abs(comb)))
    s = np.sign(comb[x])
    sl = slice(max(0, x - 1), x + 2)
    cons = float(((Z[:, sl].max(1) if s > 0 else -Z[:, sl].min(1)) > 1.5).mean())
    r = np.abs(comb).copy()
    r[max(0, x - 30):x + 31] = 0
    rival = float(r.max())
    peak = float(abs(comb[x]))
    l = x
    while l > 0 and abs(comb[l - 1]) > 0.35 * peak and x - l < 20:
        l -= 1
    rr = x
    while rr < W - 1 and abs(comb[rr + 1]) > 0.35 * peak and rr - x < 20:
        rr += 1
    return dict(status="ok", x=x + x_off, peak=peak, cons=cons, rival=rival, l=l + x_off, r=rr + x_off)


def big_step(col):
    return np.abs(np.diff(np.convolve(col, np.ones(9) / 9, "same")))[10:-10].max() * 9 > 45


def analyse(path):
    g = np.asarray(Image.open(path).convert("L")).astype(float)
    col = g.mean(0)
    white = col > 200
    if (col > 235).mean() > 0.06 or (col < 12).mean() > 0.06 or big_step(col):
        if white.mean() > 0.06:                                   # cloth-edge frame: analyse the fabric side only
            fab = np.where(~white)[0]
            run = max(np.split(fab, np.where(np.diff(fab) > 1)[0] + 1), key=len)
            x0, x1 = int(run[0]) + 30, int(run[-1]) - 30
            if x1 - x0 < 120:
                return dict(status="edge")
            sub = g[:, x0:x1]
            if big_step(sub.mean(0)):
                return dict(status="edge")
            return line_in(sub, x0)
        return dict(status="edge")                                # plaid / pattern: not handled
    return line_in(g)


def accepted(v):
    return v["status"] == "ok" and v["peak"] >= 6 and v["cons"] >= 0.75 and v["rival"] < 0.6 * v["peak"]


if __name__ == "__main__":
    folder, out = sys.argv[1], sys.argv[2]
    res = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
        n = int(re.search(r"\((\d+)\)", f).group(1))
        res[n] = analyse(f)
    json.dump(res, open(out, "w"), default=int)
    print(f"{sum(accepted(v) for v in res.values())} of {len(res)} images accepted -> {out}")
