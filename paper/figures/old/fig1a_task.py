"""Figure 1a: task illustration. One CT + one free-text finding -> one 3D mask.

Uses challenge test case train_3393_a_2. Inputs in paper/figures/data/case_test/:
  ct.nii.gz    CT-RATE volume (dataset/train_fixed/train_3393/train_3393_a/train_3393_a_2.nii.gz)
  mask.nii.gz  challenge test ground truth (ReXGroundingCT_MICCAI2026_test_gt/train_3393_a_2.nii.gz),
               shape (F, H, W, D)
Run from the project root:  python3 paper/figures/fig1a_task.py
"""
import os
import textwrap
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.colors import to_rgb
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "paper", "figures")
CASE = os.path.join(OUT, "data", "case_test")

INK, INK2, MUTED, PANEL = "#0b0b0b", "#52514e", "#8a8984", "#f0efec"
# (mask channel, finding text, colour, display slice z, zoom (x, y, half-size) or None).
# Validated all-pairs palette; the non-focal finding is orange to match panel d.
FINDINGS = [
    (2, "Subcentimeter calcified nodules in the upper lobes bilaterally", "#2a78d6", 296, (335, 204, 34)),
    (3, "Subpleural nodule 5 mm in the lateral basal segment of the right lower lobe", "#4a3aa7", 92, (72, 318, 34)),
    (1, "Fibrotic scarring in the lower lobes", "#1baf7a", 137, None),
    (0, "Emphysema in the upper lobes of both lungs", "#eb6834", 368, None),
]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none"})

ct = np.asarray(nib.load(os.path.join(CASE, "ct.nii.gz")).dataobj).astype(np.float32)
mk = np.asarray(nib.load(os.path.join(CASE, "mask.nii.gz")).dataobj) > 0
gray = lambda a: np.clip((a + 1350) / 1500, 0, 1)  # lung window (W 1500, L -600)

def body_mask(sl):
    b = ndimage.binary_fill_holes(sl > -500)
    lab, n = ndimage.label(b)
    return lab == (np.argmax(ndimage.sum(b, lab, range(1, n + 1))) + 1) if n else b

def axial(z):  # rows = anterior->posterior, cols = patient right->left (radiological convention)
    return ct[:, :, z].T

STACK = [400, 310, 220, 130]
bb = np.zeros_like(axial(STACK[0]), dtype=bool)
for z in STACK + [f[3] for f in FINDINGS]:
    bb |= body_mask(axial(z))
rows, cols = np.nonzero(bb)
R0, R1 = max(rows.min() - 4, 0), min(rows.max() + 4, bb.shape[0])
C0, C1 = max(cols.min() - 4, 0), min(cols.max() + 4, bb.shape[1])
h, w = R1 - R0, C1 - C0

def rgba_slice(z):
    sl = axial(z)
    b = body_mask(sl)[R0:R1, C0:C1]
    g = gray(sl)[R0:R1, C0:C1] * 0.92
    edge = b & ~ndimage.binary_erosion(b, iterations=3)
    g[edge] = 0.35
    return np.dstack([g, g, g, b.astype(float)])

W, H = 7.5, 2.75
fig = plt.figure(figsize=(W, H))
TOP = 0.90
fig.text(0.005, 0.985, "a", fontweight="bold", fontsize=8, va="top", color=INK)
for x, t in [(0.105, "Input CT volume"), (0.405, "Free-text finding (one per query)"),
             (0.885, "Output: 3D mask (axial slice shown)")]:
    fig.text(x, 0.955, t, ha="center", va="top", fontsize=7, color=INK2)

# ---- column 1: CT volume as a stack of axial slices, background removed
ax0 = fig.add_axes([0.0, 0.04, 0.23, 0.84]); ax0.axis("off"); ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
sx = 0.74 / w
for i, z in enumerate(STACK):  # superior (back) to inferior (front)
    tr = (Affine2D().scale(sx, sx * 0.55).skew_deg(-30, 0).translate(0.17 - i * 0.03, 0.70 - i * 0.165)
          + ax0.transData)
    ax0.imshow(rgba_slice(z), transform=tr, extent=(0, w, 0, h), interpolation="bilinear", clip_on=False)

# ---- columns 2-4: finding chip -> model -> output tile, one row per query
row_h = 0.19
ys = [TOP - row_h * (j + 0.5) - 0.03 for j in range(4)]
chip_x0, chip_w = 0.255, 0.30
model_x0, model_w = 0.615, 0.095
tile_x0 = 0.80
tile_w = (row_h - 0.016) * H * (w / h) / W
for j, (ch, text, col, z, zoom) in enumerate(FINDINGS):
    y = ys[j]
    fig.patches.append(FancyBboxPatch((chip_x0, y - 0.065), chip_w, 0.13, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor="white", edgecolor=col, linewidth=1.0))
    fig.patches.append(Rectangle((chip_x0 + 0.008, y - 0.03), 0.009, 0.06, transform=fig.transFigure, color=col))
    fig.text(chip_x0 + 0.025, y, textwrap.fill(text, 46), va="center", fontsize=6.2, color=INK, linespacing=1.25)
    for x_from, x_to in [(chip_x0 + chip_w + 0.004, model_x0 - 0.004), (model_x0 + model_w + 0.004, tile_x0 - 0.006)]:
        fig.add_artist(FancyArrowPatch((x_from, y), (x_to, y), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=7, color=MUTED, lw=0.7))
    axt = fig.add_axes([tile_x0, y - row_h / 2 + 0.008, tile_w, row_h - 0.016])
    sl, m = gray(axial(z)), mk[ch][:, :, z].T
    if zoom:
        c, r, half = zoom
        hw = int(round(half * w / h))
        sl, m = sl[r - half:r + half, c - hw:c + hw], m[r - half:r + half, c - hw:c + hw]
    else:
        sl, m = sl[R0:R1, C0:C1], m[R0:R1, C0:C1]
    axt.imshow(sl, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    over = np.zeros(m.shape + (4,)); over[m] = (*to_rgb(col), 0.55)
    axt.imshow(over, interpolation="nearest")
    axt.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=0.6)
    axt.set_xticks([]); axt.set_yticks([])
    for s in axt.spines.values():
        s.set_edgecolor(col); s.set_linewidth(0.8)
    if zoom:
        axt.text(0.05, 0.06, "zoomed", transform=axt.transAxes, fontsize=5, color="white", va="bottom")

# the same CT is the image input to every query
fig.patches.append(FancyBboxPatch((model_x0, ys[3] - 0.09), model_w, ys[0] - ys[3] + 0.18,
                                  boxstyle="round,pad=0,rounding_size=0.02", transform=fig.transFigure,
                                  facecolor=PANEL, edgecolor=MUTED, linewidth=0.8))
fig.text(model_x0 + model_w / 2, (ys[0] + ys[3]) / 2, "Grounding\nmodel", ha="center", va="center", fontsize=7,
         color=INK)
x_mid, y_lo = model_x0 + model_w / 2, 0.055
fig.add_artist(plt.Line2D([0.105, 0.105, x_mid], [0.255, y_lo, y_lo], transform=fig.transFigure, color=MUTED, lw=0.7))
fig.add_artist(FancyArrowPatch((x_mid, y_lo), (x_mid, ys[3] - 0.092), transform=fig.transFigure, arrowstyle="-|>",
                               mutation_scale=7, color=MUTED, lw=0.7))
fig.text(0.40, y_lo + 0.012, "the same CT is the image input to every query", ha="center", va="bottom",
         fontsize=5.5, color=MUTED)

for ext in ("pdf", "svg"):
    fig.savefig(os.path.join(OUT, f"fig1a_task.{ext}"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1a_task_preview.png"), dpi=250, bbox_inches="tight")
