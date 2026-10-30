"""Qualitative comparison of the top three teams with the reference annotation (Figure 3).
Run from the project root: python3 paper/figures/fig_qualitative.py
Inputs in paper/figures/data/qual/: <case>__ct.nii.gz, <case>__gt.nii.gz, <case>__{CIRCLE,AIMHI,fomofo}.nii.gz
(final submissions; masks are (findings, x, y, z), one channel per finding)."""
import os, json, textwrap
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy import ndimage
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Q = os.path.join(ROOT, "paper", "figures", "data", "qual")
INK, INK2 = "#0b0b0b", "#52514e"
GT_C, PR_C = "#1baf7a", "#2a78d6"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
test = {e["name"]: e for e in json.load(open(os.path.join(ROOT, "paper/figures/data/ReXGroundingCT_MICCAI2026_test.json")))["test"]}
RES = {"CIRCLE Team": "submissions/2572837347@qq.com/CIRCLE__HtYfefYk/results.json",
       "AIMHI": "submissions/khuhm@gachon.ac.kr/Grounder_1.4n__QZ7lRcdv/results.json",
       "fomofo": "submissions/r.vt.hull@nki.nl/TAGS_TAPv3__hfYYpTHz/results.json"}
FILE = {"CIRCLE Team": "CIRCLE", "AIMHI": "AIMHI", "fomofo": "fomofo"}
dice = {t: {c["file"]: c["findings"] for c in json.load(open(os.path.join(ROOT, p)))["cases"]} for t, p in RES.items()}
ROWS = [("train_13492_a_2", 0), ("train_3393_a_2", 3), ("train_13285_a_2", 2), ("train_3309_a_2", 0), ("train_3094_a_2", 0)]
lung = lambda a: np.clip((a + 1350) / 1500, 0, 1)

def body_mask(sl):
    b = ndimage.binary_fill_holes(sl > -500); lab, n = ndimage.label(b)
    return lab == (np.argmax(ndimage.sum(b, lab, range(1, n + 1))) + 1) if n else b

def crop(sl, m, aspect=1.25, margin=30, min_h=120):
    r, c = np.nonzero(m); H_, W_ = sl.shape
    rc, cc = (r.min() + r.max()) / 2, (c.min() + c.max()) / 2
    hh = max(r.max() - r.min() + 2 * margin, min_h); ww = max(c.max() - c.min() + 2 * margin, min_h * aspect)
    hh = max(hh, ww / aspect); ww = hh * aspect
    b = body_mask(sl); br, bc = np.nonzero(b)
    r0 = int(np.clip(rc - hh / 2, br.min(), max(br.max() - hh, br.min()))); c0 = int(np.clip(cc - ww / 2, bc.min(), max(bc.max() - ww, bc.min())))
    return r0, min(int(r0 + hh), H_), c0, min(int(c0 + ww), W_)

cols = ["Reference"] + list(RES)
NS = 3                                            # slices per method
W = 7.0
g_in, g_out = 0.025, 0.10                        # gap within a group of slices / between groups
tile_w = (W - 3 * g_out - 4 * (NS - 1) * g_in) / (4 * NS); tile_h = tile_w / 1.1
txt_h, row_gap, head, dice_h = 0.17, 0.06, 0.20, 0.14
row_h = txt_h + tile_h + dice_h + row_gap
H = head + row_h * len(ROWS)
fig = plt.figure(figsize=(W, H))
gx = lambda j, s_: j * (NS * tile_w + (NS - 1) * g_in + g_out) + s_ * (tile_w + g_in)
for j, name in enumerate(cols):
    fig.text((gx(j, 0) + (NS * tile_w + (NS - 1) * g_in) / 2) / W, 1 - 0.09 / H, name, ha="center", va="center", fontsize=7.5, color=INK)
for i, (case, k) in enumerate(ROWS):
    f = case + ".nii.gz"
    ct = np.asarray(nib.load(os.path.join(Q, f"{case}__ct.nii.gz")).dataobj).astype(np.float32)
    gt = np.asarray(nib.load(os.path.join(Q, f"{case}__gt.nii.gz")).dataobj)[k] > 0
    preds = {n: np.asarray(nib.load(os.path.join(Q, f"{case}__{FILE[n]}.nii.gz")).dataobj)[k] > 0 for n in RES}
    area = gt.sum((0, 1)); zz = np.nonzero(area > 0.3 * area.max())[0]
    zs = [int(np.percentile(zz, q, method="nearest")) for q in (12, 50, 88)]
    union = np.any([gt[:, :, z].T for z in zs], axis=0)
    small = union.sum() < 400
    r0, r1, c0, c1 = crop(ct[:, :, zs[1]].T, union, aspect=1.1, margin=18 if small else 30, min_h=60 if small else 120)
    y_top = 1 - (head + i * row_h) / H
    fig.text(0.0, y_top - 0.02 / H, test[f]["findings"][str(k)], ha="left", va="top", fontsize=6.8, color=INK)
    ty = y_top - (txt_h + tile_h) / H
    for j, name in enumerate(cols):
        for s_, z in enumerate(zs):
            ax = fig.add_axes([gx(j, s_) / W, ty, tile_w / W, tile_h / H])
            sl = ct[:, :, z].T
            ax.imshow(lung(sl[r0:r1, c0:c1]), cmap="gray", vmin=0, vmax=1, interpolation="bilinear", aspect="auto")
            gg = gt[:, :, z].T[r0:r1, c0:c1]
            m, col = (gg, GT_C) if name == "Reference" else (preds[name][:, :, z].T[r0:r1, c0:c1], PR_C)
            over = np.zeros(m.shape + (4,)); over[m] = (*to_rgb(col), 0.5)
            ax.imshow(over, interpolation="nearest", aspect="auto")
            if m.any(): ax.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=0.5)
            if name != "Reference" and gg.any():
                ax.contour(gg.astype(float), levels=[0.5], colors=["white"], linewidths=0.4, linestyles=[(0, (2, 1.5))])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_edgecolor("#c8c7c2"); sp.set_linewidth(0.5)
        if name != "Reference":
            d = dice[name][f][f"finding_{k}"]["global_dice"]
            fig.text((gx(j, 0) + (NS * tile_w + (NS - 1) * g_in) / 2) / W, ty - 0.03 / H, f"Dice {d:.2f}", ha="center", va="top", fontsize=6.6, color=INK2)
for ext in ("pdf", "svg"):
    fig.savefig(os.path.join(ROOT, "paper", "figures", f"fig_qualitative.{ext}"), bbox_inches="tight", dpi=600)
fig.savefig(os.path.join(ROOT, "paper", "figures", "fig_qualitative_preview.png"), dpi=200, bbox_inches="tight")
print("ok")
