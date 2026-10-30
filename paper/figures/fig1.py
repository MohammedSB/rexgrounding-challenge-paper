"""Figure 1: overview of the ReXGrounding Challenge (single composite figure).

a  task illustration (test case train_3393_a_2 volume; two extra queries drawn from test cases train_4079_a_1 and train_3747_a_1)
b  challenge workflow (development and evaluation phases)
c  participating teams by country
d  test-set finding categories
e  what the finding text specifies (test vs training findings)
f  finding volume and multiplicity per category (test set)

Run from the project root:  python3 paper/figures/fig1.py
Inputs: final_results_MICCAI2026.json, final_rankings_merged.csv, participant_countries.csv,
        submissions/<team>/<run>/results.json (any one run, for GT instance counts),
        paper/figures/data/{case_test*/ct.nii.gz, case_test*/mask.nii.gz, test_gt_voxels.json,
        test_spacing.json, ReXGroundingCT_MICCAI2026_test.json, dataset.json, ne_110m_*.shp}
"""
import os, re, csv, json, collections, textwrap, math
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Ellipse, Patch
from matplotlib.transforms import Affine2D
from matplotlib.colors import to_rgb
from scipy import ndimage
import shapefile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "paper", "figures")
DATA = os.path.join(OUT, "data")

# ---------------------------------------------------------------- style
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#c0399b"  # validated all-pairs on white (dataviz validator)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, LINE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#b9b8b3"
PANEL, PANEL2, LAND = "#f3f2ef", "#e9eef6", "#ebeae6"
FS_HEAD, FS_BODY, FS_NOTE = 7.0, 6.4, 5.8
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
VARIANT_C = os.environ.get("FIG1C", "map")                      # map | participation | leaderboard
W, H = 7.5, (7.5 if VARIANT_C == "map" else 6.55)
fig = plt.figure(figsize=(W, H))

def letter(x, y, s):
    fig.text(x, y, s, fontsize=9, fontweight="bold", color=INK, va="top", ha="left")

def heading(ax, text):
    bb = ax.get_position()
    fig.text(bb.x0, ax._row_top - 0.012, text, fontsize=7.5, color=INK, va="top", ha="left")

def canvas(x0, y0, w, h):
    """Axes in figure fractions whose data units are inches (isotropic)."""
    ax = fig.add_axes([x0, y0, w, h]); ax.set_xlim(0, w * W); ax.set_ylim(0, h * H)
    ax.set_aspect("equal"); ax.axis("off"); return ax

def arrow(ax, p, q, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=MUTED, lw=0.7,
                                 shrinkA=0, shrinkB=0, **kw))

def box(ax, x, y, w, h, text, fc="white", ec=LINE, fs=FS_BODY, lw=0.8, color=INK, weight="normal", r=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc,
                                edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight=weight,
            linespacing=1.3)

# ---------------------------------------------------------------- data
results = json.load(open(os.path.join(ROOT, "final_results_MICCAI2026.json")))["results"]
merged = list(csv.DictReader(open(os.path.join(ROOT, "final_rankings_merged.csv"))))
test_json = json.load(open(os.path.join(DATA, "ReXGroundingCT_MICCAI2026_test.json")))["test"]
train_json = json.load(open(os.path.join(DATA, "dataset.json")))["train"]
NAMES = {"1a": "Bronchial wall thickening", "1b": "Bronchiectasis", "1c": "Emphysema", "1d": "Septal thickening",
         "1e": "Micronodules", "1f": "Other non-focal", "2a": "Linear opacities", "2b": "Atelectasis / consolidation",
         "2c": "Ground-glass opacity", "2d": "Nodules / masses", "2e": "Pleural effusion / thickening",
         "2g": "Pneumothorax", "2h": "Other focal"}
cats = {k: v["n"] for k, v in results[0]["per_category_full_test_set"].items()}
focal = lambda k: k.startswith("2")
CAT_ORDER = ([k for k in sorted(cats, key=lambda k: -cats[k]) if focal(k)] +
             [k for k in sorted(cats, key=lambda k: -cats[k]) if not focal(k)])

# =================================================================== a: task
def panel_a(x0, y0, w, h):
    ax = canvas(x0, y0, w, h); CW, CH = w * W, h * H
    img = nib.load(os.path.join(DATA, "case_test", "ct.nii.gz"))
    ct = np.asarray(img.dataobj).astype(np.float32)
    mk = np.asarray(nib.load(os.path.join(DATA, "case_test", "mask.nii.gz")).dataobj) > 0
    CASES = {"case_test": (ct, mk)}
    def load_case(name):
        if name not in CASES:
            CASES[name] = (np.asarray(nib.load(os.path.join(DATA, name, "ct.nii.gz")).dataobj).astype(np.float32),
                           np.asarray(nib.load(os.path.join(DATA, name, "mask.nii.gz")).dataobj) > 0)
        return CASES[name]

    sxy, sz = float(img.header.get_zooms()[0]), float(img.header.get_zooms()[2])
    lung = lambda a: np.clip((a + 1350) / 1500, 0, 1)
    soft = lambda a: np.clip((a + 1000) / 1400, 0, 1)
    axial = lambda z: ct[:, :, z].T                       # rows: anterior->posterior, cols: patient R->L
    coronal = lambda y: ct[:, y, ::-1].T                  # rows: superior->inferior, cols: patient R->L
    sagittal = lambda x: ct[x, :, ::-1].T                 # rows: superior->inferior, cols: anterior->posterior

    def body_mask(sl):
        b = ndimage.binary_fill_holes(sl > -500); lab, n = ndimage.label(b)
        return lab == (np.argmax(ndimage.sum(b, lab, range(1, n + 1))) + 1) if n else b

    # --- 3D volume block (corner cut through the lungs)
    body = np.zeros(ct.shape[:2], bool)
    for z in range(60, ct.shape[2], 40):
        body |= body_mask(ct[:, :, z])
    xs, ys = np.nonzero(body); xa, xb, ya, yb = xs.min(), xs.max(), ys.min(), ys.max()
    lungz = []                                              # thorax only: slices whose largest air pocket is lung-sized
    for z in range(0, ct.shape[2], 3):
        sl = ct[:, :, z]; air = (sl < -400) & ndimage.binary_erosion(body_mask(sl), iterations=6)
        lab, n = ndimage.label(air)
        if n and ndimage.sum(air, lab, range(1, n + 1)).max() > 0.005 * sl.size:
            lungz.append(z)
    za, zb = max(min(lungz) - int(0.06 * ct.shape[2]) - 6, 0), min(max(lungz) + 6, ct.shape[2] - 1)
    ycut = ya + int(0.69 * (yb - ya)); zcut = za + int(0.78 * (zb - za))   # coronal plane through the lower-lobe scarring
    lungx = np.zeros(ct.shape[0], bool)                     # box width: lungs plus chest wall, not the arms
    for z in lungz[::2]:
        sl = ct[:, :, z]; air = (sl < -400) & ndimage.binary_erosion(body_mask(sl), iterations=6)
        lab_, n_ = ndimage.label(air); sz_ = ndimage.sum(air, lab_, range(1, n_ + 1))
        lungx |= np.isin(lab_, [i + 1 for i in np.argsort(sz_)[::-1][:2] if sz_[i] > 0.005 * sl.size]).any(1)
    cols = np.nonzero(lungx)[0]; m_ = int(22 / sxy)
    xa, xb = max(cols.min() - m_, 0), min(cols.max() + m_, ct.shape[0] - 1)
    xcut = xb                                               # full width: both lungs on the front face
    cube = lambda a: np.clip((a + 1200) / 1500, 0, 1)      # window between lung and soft tissue: lungs dark, vessels visible
    top = cube(axial(zcut)[ycut:yb, xa:xcut])[::-1]            # row 0 = posterior (back)
    front = cube(coronal(ycut)[ct.shape[2] - zcut:ct.shape[2] - za, xa:xcut])
    side = cube(sagittal(xcut)[ct.shape[2] - zcut:ct.shape[2] - za, ycut:yb])
    Wm, Dm, Hm = (xcut - xa) * sxy, (yb - ycut) * sxy, (zcut - za) * sz
    k = 1.0 / Wm                                           # inches per mm (cube width 1.0 in)
    dx, dy = 0.55, 0.34                                    # depth direction
    Wt, Ht = k * (Wm + dx * Dm), k * (Hm + dy * Dm)        # footprint of the drawn cube
    X0, Y0 = 0.28 + (1.25 / Wm * (Wm + dx * Dm) - Wt) / 2, 0.50 + (1.25 / Wm * (Hm + dy * Dm) - Ht) / 2
    T = lambda m: Affine2D(np.array(m + [[0, 0, 1]])) + ax.transData
    ax.imshow(front, cmap="gray", vmin=0, vmax=1, extent=(0, Wm, 0, Hm), interpolation="bilinear",
              transform=T([[k, 0, X0], [0, k, Y0]]))
    ax.imshow(top, cmap="gray", vmin=0, vmax=1, extent=(0, Wm, 0, Dm), interpolation="bilinear",
              transform=T([[k, k * dx, X0], [0, k * dy, Y0 + k * Hm]]))
    ax.imshow(side, cmap="gray", vmin=0, vmax=1, extent=(0, Dm, 0, Hm), interpolation="bilinear",
              transform=T([[k * dx, 0, X0 + k * Wm], [k * dy, k, Y0]]))
    P = lambda u, v, d: (X0 + k * (u + dx * d), Y0 + k * (v + dy * d))
    for pts in ([P(0, 0, 0), P(Wm, 0, 0), P(Wm, Hm, 0), P(0, Hm, 0)],
                [P(0, Hm, 0), P(Wm, Hm, 0), P(Wm, Hm, Dm), P(0, Hm, Dm)],
                [P(Wm, 0, 0), P(Wm, Hm, 0), P(Wm, Hm, Dm), P(Wm, 0, Dm)]):
        ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor="#3a3a38", linewidth=0.5))
    # hidden edges, for the cube feel
    for a_, b_ in ([P(0, 0, 0), P(0, 0, Dm)], [P(0, 0, Dm), P(Wm, 0, Dm)], [P(0, 0, Dm), P(0, Hm, Dm)]):
        ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color=LINE, lw=0.5, ls=(0, (2, 2)))
    ax.text(X0 + k * (Wm + dx * Dm) / 2, CH - 0.12, "Input: 3D chest CT volume", ha="center", va="top",
            fontsize=FS_HEAD, color=INK2)

    # --- queries, model, outputs
    # (case dir, mask channel, text, colour, tiles, side). The volume shown is case_test; other queries and their
    # tiles come from other test cases (case_test5 = train_13492_a_2, case_test6 = train_3249_a_1).
    # tiles: a list of axial slices (one crop box shared by the row) or "components:k" (one tile per nodule,
    # the k largest connected components, each at its own best slice and zoomed on that nodule).
    FIND = [("case_test", 3, "Subpleural nodule 5 mm in the lateral basal segment of the right lower lobe", BLUE, [92], "R"),
            ("case_test5", 0, "Peripheral patch-like consolidation in the superior segment of the left lower lobe",
             ORANGE, [297, 312, 327], "L"),
            ("case_test", 1, "Fibrotic scarring in the lower lobes", AQUA, [124, 172], None),
            ("case_test6", 0, "Subsegmental atelectasis in the medial segment of the right middle lobe",
             MAGENTA, [77, 82, 86], "R")]

    ASPECT, MARGIN, MIN_H = 1.3, 24, 110                  # tile w:h, context margin (px), min crop height (px)

    def zoom_crop(sl, fmask, min_h=MIN_H):
        """Crop box (r0, r1, c0, c1) around the finding on slice sl, fixed aspect, kept inside the body."""
        b = body_mask(sl); H_, W_ = sl.shape
        r, c = np.nonzero(fmask)
        rc, cc = (r.min() + r.max()) / 2, (c.min() + c.max()) / 2
        hh = max(r.max() - r.min() + 2 * MARGIN, min_h); ww = max(c.max() - c.min() + 2 * MARGIN, min_h * ASPECT)
        hh = max(hh, ww / ASPECT); ww = hh * ASPECT
        br, bc = np.nonzero(b); lim_r0, lim_r1, lim_c0, lim_c1 = br.min(), br.max(), bc.min(), bc.max()
        r0 = int(np.clip(rc - hh / 2, lim_r0, max(lim_r1 - hh, lim_r0))); c0 = int(np.clip(cc - ww / 2, lim_c0, max(lim_c1 - ww, lim_c0)))
        return r0, min(int(r0 + hh), H_), c0, min(int(c0 + ww), W_)
    chip_x, chip_w = 2.45, 2.35
    mod_x, mod_w = 5.05, 0.72
    tile_x, tgap = 6.02, 0.04
    avail = CW - tile_x - 0.06
    top_y, row_h = CH - 0.30, 0.44
    ys_ = [top_y - row_h * (i + 0.5) for i in range(len(FIND))]
    th0 = row_h - 0.08
    tw = avail                                             # width of the tile column (for the header)
    lighten = lambda c, f=0.25: tuple(1 - (1 - v) * (1 - f) for v in to_rgb(c))   # mix 25% toward white
    for (cs, ch, text, col, spec, side), y in zip(FIND, ys_):
        col = lighten(col)
        box(ax, chip_x, y - 0.185, chip_w, 0.37, "", ec=col, lw=0.9, r=0.05)
        ax.add_patch(Rectangle((chip_x + 0.07, y - 0.10), 0.07, 0.20, color=col))
        ax.text(chip_x + 0.21, y, textwrap.fill(text, 46), va="center", fontsize=FS_BODY, color=INK, linespacing=1.2)
        arrow(ax, (chip_x + chip_w + 0.04, y), (mod_x - 0.04, y))
        arrow(ax, (mod_x + mod_w + 0.04, y), (tile_x - 0.05, y))
        ct_c, mk_c = load_case(cs)
        tiles = []                                         # (slice, mask on that slice, crop box)
        if isinstance(spec, str) and spec.startswith("components"):
            ncomp = int(spec.split(":")[1]); lab, nc = ndimage.label(mk_c[ch]); sizes = ndimage.sum(mk_c[ch], lab, range(1, nc + 1))
            comps = sorted(np.argsort(sizes)[::-1][:ncomp], key=lambda o: -int(np.argmax((lab == o + 1).sum((0, 1)))))
            for o in comps:                                # superior to inferior
                comp = lab == o + 1; z = int(np.argmax(comp.sum((0, 1))))
                sl_, m_ = ct_c[:, :, z].T, mk_c[ch][:, :, z].T
                tiles.append((sl_, m_, zoom_crop(sl_, comp[:, :, z].T, min_h=80)))
        else:
            slices = [(ct_c[:, :, z].T, mk_c[ch][:, :, z].T) for z in spec]
            union = np.any([m_ for _, m_ in slices], axis=0)
            bx_ = zoom_crop(slices[0][0], union)
            tiles = [(sl_, m_, bx_) for sl_, m_ in slices]
        asps = [(c1 - c0) / (r1 - r0) for _, _, (r0, r1, c0, c1) in tiles]; n = len(tiles)
        th = min(th0, (avail - (n - 1) * tgap) / sum(asps)); xt = tile_x
        for (sl_, m_, (r0, r1, c0, c1)), asp in zip(tiles, asps):
            tw_i = th * asp
            tx = fig.add_axes([x0 + xt / W, y0 + (y - th / 2) / H, tw_i / W, th / H]); xt += tw_i + tgap
            sl, m = lung(sl_)[r0:r1, c0:c1], m_[r0:r1, c0:c1]
            tx.imshow(sl, cmap="gray", vmin=0, vmax=1, interpolation="bilinear", aspect="auto")
            over = np.zeros(m.shape + (4,)); over[m] = (*to_rgb(col), 0.55); tx.imshow(over, interpolation="nearest", aspect="auto")
            tx.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=0.6)
            tx.set_xticks([]); tx.set_yticks([])
            for sp_ in tx.spines.values():
                sp_.set_visible(True); sp_.set_edgecolor(col); sp_.set_linewidth(0.9)
    box(ax, mod_x, 0.04, mod_w, ys_[0] + 0.22 - 0.04, "Grounding\nmodel", fc=PANEL, ec=MUTED, fs=FS_HEAD, r=0.08)
    # same CT feeds every query
    xc, ylo = X0 + k * (Wm + dx * Dm) / 2, 0.12                 # volume enters the model from below-left: one bend
    ax.plot([xc, xc, mod_x - 0.14], [Y0 - 0.03, ylo, ylo], color=MUTED, lw=0.7, solid_joinstyle="miter")
    arrow(ax, (mod_x - 0.14, ylo), (mod_x - 0.04, ylo))
    for x, t in [(chip_x + chip_w / 2, "Set of free-text findings"),
                 (tile_x + tw / 2, "Output: set of masks per finding")]:
        ax.text(x, CH - 0.12, t, ha="center", va="top", fontsize=FS_HEAD, color=INK2)

# =================================================================== b: workflow
def cylinder(ax, x, y, w, h, fc, ec, hidden=False):
    e = w * 0.12
    ax.add_patch(Rectangle((x, y + e), w, h - 2 * e, facecolor=fc, edgecolor="none"))
    ax.add_patch(Ellipse((x + w / 2, y + e), w, 2 * e, facecolor=fc, edgecolor=ec, lw=0.7, ls=":" if hidden else "-"))
    ax.plot([x, x], [y + e, y + h - e], color=ec, lw=0.7, ls=":" if hidden else "-")
    ax.plot([x + w, x + w], [y + e, y + h - e], color=ec, lw=0.7, ls=":" if hidden else "-")
    ax.add_patch(Ellipse((x + w / 2, y + h - e), w, 2 * e, facecolor=fc, edgecolor=ec, lw=0.7, ls=":" if hidden else "-"))
    for yy in (y + h - e - (h - 2 * e) / 3, y + h - e - 2 * (h - 2 * e) / 3):
        ax.add_patch(Ellipse((x + w / 2, yy), w, 2 * e, facecolor="none", edgecolor=ec, lw=0.4, ls=":" if hidden else "-"))

def panel_b(x0, y0, w, h):
    ax = canvas(x0, y0, w, h); CW, CH = w * W, h * H
    split = 4.05
    panel_b.ax = ax
    dev = ax.add_patch(FancyBboxPatch((0.05, 0.05), split - 0.15, CH - 0.24, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=PANEL, edgecolor="none"))
    ev = ax.add_patch(FancyBboxPatch((split + 0.05, 0.05), CW - split - 0.12, CH - 0.24,
                                boxstyle="round,pad=0,rounding_size=0.08", facecolor=PANEL2, edgecolor="none"))
    ax._boxes = (dev, ev)
    ax.text(0.20, CH - 0.35, "Development phase", fontsize=FS_HEAD, color=INK, fontweight="bold", va="center")
    ax.text(1.42, CH - 0.35, "June – 14 September 2026", fontsize=FS_BODY, color=INK2, va="center")
    ax.text(split + 0.20, CH - 0.35, "Evaluation phase", fontsize=FS_HEAD, color=INK, fontweight="bold", va="center")
    ax.text(split + 1.30, CH - 0.35, "14 – 27 September 2026", fontsize=FS_BODY, color=INK2, va="center")
    yc = 1.02
    # development: training/validation data -> participants -> submissions on the public test half -> leaderboard
    cylinder(ax, 0.25, yc + 0.12, 0.36, 0.46, SEQ[1], INK2)
    ax.text(0.68, yc + 0.35, "Training set\n2,992 scans", fontsize=FS_BODY,
            color=INK, va="center", linespacing=1.25)
    cylinder(ax, 0.25, yc - 0.62, 0.36, 0.46, SEQ[1], INK2)
    ax.text(0.68, yc - 0.39, "Validation set\n200 scans", fontsize=FS_BODY, color=INK,
            va="center", linespacing=1.25)
    ax.plot([1.80, 1.88, 1.88, 1.80], [yc + 0.35, yc + 0.35, yc - 0.39, yc - 0.39], color=MUTED, lw=0.7, solid_joinstyle="miter")
    arrow(ax, (1.88, yc), (1.98, yc))
    box(ax, 1.98, yc - 0.27, 0.86, 0.54, "Participants\ntrain and tune\nmodels", fc="white", ec=MUTED)
    arrow(ax, (2.86, yc), (3.04, yc))
    box(ax, 3.04, yc - 0.27, 0.86, 0.54, "Submit\npredictions", fc="white", ec=MUTED)
    # public half of the test set: scans released, masks withheld
    cylinder(ax, 3.29, 0.14, 0.36, 0.38, SEQ[1], INK2)
    arrow(ax, (3.47, 0.54), (3.47, yc - 0.29))
    ax.text(3.20, 0.33, "Test set\n300 scans, masks withheld", fontsize=FS_BODY, color=INK,
            ha="right", va="center", linespacing=1.25)
    # leaderboard loop, above the boxes
    yl = yc + 0.44
    ax.plot([3.47, 3.47, 2.41], [yc + 0.27, yl, yl], color=MUTED, lw=0.7, solid_joinstyle="miter")
    arrow(ax, (2.41, yl), (2.41, yc + 0.29))
    ax.text(2.94, yl + 0.04, "Dice on 150 of the 300 scans", fontsize=FS_NOTE, color=INK2,
            ha="center", va="bottom", linespacing=1.15)
    # evaluation: final submission -> ranking on the full test set (public + hidden halves) -> session
    bx = split + 0.30
    arrow(ax, (3.90, yc), (bx, yc))
    box(ax, bx, yc - 0.27, 0.98, 0.54, "Final submission\nscored on all 300\ntest scans", fc="white", ec=MUTED)
    cylinder(ax, bx + 0.04, 0.22, 0.36, 0.36, SEQ[1], INK2)
    cylinder(ax, bx + 0.50, 0.22, 0.36, 0.36, "white", INK2, hidden=True)
    ax.plot([bx + 0.22, bx + 0.22, bx + 0.68, bx + 0.68], [0.58, 0.62, 0.62, 0.58], color=MUTED, lw=0.7, solid_joinstyle="miter")
    arrow(ax, (bx + 0.45, 0.62), (bx + 0.45, yc - 0.29))
    ax.text(bx + 0.98, 0.40, "Full test set, 300 scans\n150 scored during development\n150 scored only after the deadline",
            fontsize=FS_BODY, color=INK, va="center", linespacing=1.25)
    arrow(ax, (bx + 1.00, yc), (bx + 1.20, yc))
    px = bx + 1.24
    for i, (hh, off) in enumerate([(0.16, 0), (0.26, 0.13), (0.11, 0.26)]):
        ax.add_patch(Rectangle((px + off, yc - 0.14), 0.12, hh, facecolor=SEQ[3 if i == 1 else 2], edgecolor="none"))
    ax.text(px + 0.48, yc, "27 September at MICCAI 2026\nTop teams present their methods\nFinal results announced",
            fontsize=FS_BODY, color=INK, va="center", linespacing=1.25)

# =================================================================== c: map
def panel_c(x0, y0, w, h):
    """World map with one bubble per country, area proportional to the number of teams."""
    iso = {"China": "CN", "South Korea": "KR", "United States": "US", "United Kingdom": "GB", "Germany": "DE",
           "Netherlands": "NL", "India": "IN", "Japan": "JP", "Bangladesh": "BD", "France": "FR", "Qatar": "QA",
           "Austria": "AT", "Brazil": "BR", "United Arab Emirates": "AE", "Hong Kong SAR": "HK", "Chile": "CL"}
    NAME = {"CN": "China", "KR": "South Korea", "US": "United States", "GB": "United Kingdom", "DE": "Germany", "NL": "Netherlands",
            "IN": "India", "JP": "Japan", "BD": "Bangladesh", "FR": "France", "QA": "Qatar", "AT": "Austria", "BR": "Brazil",
            "AE": "UAE", "CL": "Chile"}
    country_of = {r["team"]: r["country"] for r in csv.DictReader(open(os.path.join(ROOT, "participant_countries.csv")))}
    per = collections.Counter()
    for r in merged:
        for part in country_of.get(r["team"], "Unknown").split(" / "):
            if part in iso:
                per[iso[part]] += 1
    per["CN"] += per.pop("HK", 0)
    sf = shapefile.Reader(os.path.join(DATA, "ne_110m_admin_0_countries.shp"))
    fields = [f[0] for f in sf.fields[1:]]; key = "ISO_A2_EH" if "ISO_A2_EH" in fields else "ISO_A2"
    shapes, where = [], {}
    for sr in sf.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        if rec.get("CONTINENT") == "Antarctica":
            continue
        pts, parts = sr.shape.points, list(sr.shape.parts) + [len(sr.shape.points)]
        shapes.append([pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)])
        where[rec[key]] = (rec["LABEL_X"], rec["LABEL_Y"])

    PW, PH = w * W, h * H
    head_h = 0.22
    map_h = PH - head_h - 0.06
    map_w = min(map_h * 350 / 136, PW)
    mx0 = x0 + (PW - map_w) / 2 / W
    ax = fig.add_axes([mx0, y0 + 0.03 / H, map_w / W, map_h / H])
    ax._row_top = y0 + h
    for rings in shapes:
        for ring in rings:
            ax.add_patch(Polygon(ring, closed=True, facecolor=LAND, edgecolor="white", linewidth=0.3, zorder=1))
    ax.set_xlim(-170, 180); ax.set_ylim(-56, 80); ax.set_aspect("equal"); ax.axis("off")

    K = 58.0                                                # bubble area (pt^2) per team
    FILL, RING = SEQ[3], SEQ[6]
    from matplotlib.patches import Ellipse
    # labels sit at a fixed gap outside the bubble, in the given direction (degrees), joined by a leader line
    ANG = {"US": 0, "BR": 0, "CL": 0, "JP": -20, "AT": -25, "DE": 25, "GB": 170, "NL": 95, "FR": 255,
           "QA": 160, "AE": 275, "IN": 265, "BD": -35, "CN": 185, "KR": 80}
    GAP = 8.0
    for c, n in sorted(per.items(), key=lambda kv: -kv[1]):
        lx, ly = where[c]
        ax.scatter(lx, ly, s=K * n, facecolors=FILL, edgecolors="none", alpha=0.42, zorder=5)
        ax.scatter(lx, ly, s=K * n, facecolors="none", edgecolors=RING, linewidths=0.75, zorder=6)
        r = math.sqrt(K * n / math.pi); th = math.radians(ANG[c]); cx, sy = math.cos(th), math.sin(th)
        dx, dy = (r + GAP) * cx, (r + GAP) * sy
        ha = "left" if cx > 0.35 else "right" if cx < -0.35 else "center"
        va = "bottom" if sy > 0.35 else "top" if sy < -0.35 else "center"
        ax.annotate(f"{NAME[c]} ({n})", (lx, ly), xytext=(dx, dy), textcoords="offset points", fontsize=5.8, color=INK2,
                    ha=ha, va=va, zorder=8, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5, shrinkA=1.5, shrinkB=r + 0.5))
    # nested-circle size legend: circles share a common bottom edge
    base_x, base_y = 0.075, 0.06
    for n in (10, 5, 1):
        r_pt = math.sqrt(K * n / math.pi)
        wx, hy = 2 * r_pt / (map_w * 72), 2 * r_pt / (map_h * 72)
        ax.add_patch(Ellipse((base_x, base_y + hy / 2), wx, hy, transform=ax.transAxes, facecolor=FILL, alpha=0.38,
                             edgecolor="none", clip_on=False, zorder=5))
        ax.add_patch(Ellipse((base_x, base_y + hy / 2), wx, hy, transform=ax.transAxes, facecolor="none",
                             edgecolor=RING, linewidth=0.8, clip_on=False, zorder=6))
        ax.plot([base_x, base_x + 0.085], [base_y + hy, base_y + hy], transform=ax.transAxes, color=MUTED, linewidth=0.5, clip_on=False, zorder=6)
        ax.text(base_x + 0.09, base_y + hy, str(n), transform=ax.transAxes, ha="left", va="center", fontsize=6, color=INK2)
    ax.text(base_x - 0.055, base_y + 2 * math.sqrt(K * 10 / math.pi) / (map_h * 72) + 0.045, "Teams per country",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=INK)
    heading(ax, f"{len(merged)} teams from {len(per)} countries")


# =================================================================== c (alternatives without countries)
SECTOR = [("academic / clinical", "Academic or clinical", SEQ[1]), ("academic + industry", "Academic and industry", SEQ[3]),
          ("industry", "Industry", SEQ[5]), ("unspecified", "Not stated", "#c9c8c3")]
def sector_of():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "participant_countries.csv"))))
    return {r["team"]: r["sector"].split(" (")[0] for r in rows}

def panel_c_participation(x0, y0, w, h):
    sec = sector_of()
    gap = 0.075; ww = (w - 2 * gap - 0.10) / 3
    axes = [fig.add_axes([x0 + 0.075 + i * (ww + gap) + (0.05 if i == 1 else 0), y0 + 0.30 * h, ww - (0.05 if i == 1 else 0), 0.52 * h]) for i in range(3)]
    for ax in axes: ax._row_top = y0 + h
    # funnel
    ax = axes[0]
    F = [("Registered", 120), ("Submitted", 53), ("Teams after\nmerging", len(merged))]
    ys = np.arange(len(F))[::-1]
    for y, (lab, n) in zip(ys, F):
        ax.barh(y, n, height=0.62, color=SEQ[3] if lab.startswith("Teams") else SEQ[1], edgecolor="none")
        ax.text(n + 2.5, y, str(n), va="center", fontsize=6, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([f[0] for f in F], fontsize=6, linespacing=1.05); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 135); ax.set_xlabel("Teams"); ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    heading(ax, "From registration to the final ranking")
    # sector
    ax = axes[1]
    cnt = collections.Counter(sec.get(r["team"], "unspecified") for r in merged)
    ys = np.arange(len(SECTOR))[::-1]
    for y, (k, lab, col) in zip(ys, SECTOR):
        ax.barh(y, cnt[k], height=0.62, color=col, edgecolor="none")
        ax.text(cnt[k] + 0.8, y, str(cnt[k]), va="center", fontsize=6, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([x[1] for x in SECTOR], fontsize=6); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 40); ax.set_xlabel("Teams"); ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    heading(ax, "Affiliation of the teams")
    # submissions per team
    ax = axes[2]
    n = np.array([int(r["n_submissions"]) for r in merged])
    bins = [(1, 1, "1"), (2, 5, "2–5"), (6, 10, "6–10"), (11, 25, "11–25"), (26, 60, "26–60"), (61, 10**6, ">60")]
    c = [int(((n >= a) & (n <= b)).sum()) for a, b, _ in bins]
    ax.bar(range(len(bins)), c, width=0.68, color=SEQ[3], edgecolor="none")
    for i, v in enumerate(c): ax.text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(range(len(bins))); ax.set_xticklabels([b[2] for b in bins], fontsize=6)
    ax.set_xlabel("Submissions per team"); ax.set_ylabel("Teams"); ax.set_ylim(0, max(c) * 1.18)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    heading(ax, f"{int(n.sum())} submissions in total")

def panel_c_leaderboard(x0, y0, w, h):
    sec = sector_of()
    ax = fig.add_axes([x0 + 0.075, y0 + 0.30 * h, w - 0.10, 0.52 * h]); ax._row_top = y0 + h
    d = [float(r["dice_full_test"]) for r in merged]; col = {k: c for k, _, c in SECTOR}
    for i, r in enumerate(merged):
        ax.bar(i + 1, d[i], width=0.72, color=col[sec.get(r["team"], "unspecified")], edgecolor="none")
    ax.set_xlim(0.3, len(merged) + 0.7); ax.set_ylim(0, 0.42)
    ax.set_xticks([1, 10, 20, 30, 40, len(merged)]); ax.set_xlabel("Final rank"); ax.set_ylabel("Dice (300 test scans)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=c, label=lab) for _, lab, c in SECTOR], loc="upper right", ncol=4, frameon=False,
              fontsize=6, handlelength=1.0, handleheight=0.9, columnspacing=1.2, borderaxespad=0.1)
    heading(ax, f"Final Dice of all {len(merged)} teams, by affiliation")

# =================================================================== d: categories
def panel_d(ax):
    order = CAT_ORDER[::-1]; ys = range(len(order))
    for y, k in zip(ys, order):
        ax.barh(y, cats[k], height=0.68, color=BLUE if focal(k) else ORANGE, edgecolor="white", linewidth=0.8)
        ax.text(cats[k] + 3, y, str(cats[k]), va="center", fontsize=6, color=INK2)
    ax.set_yticks(list(ys)); ax.set_yticklabels([NAMES[k] for k in order], fontsize=6); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Findings in the test set"); ax.xaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    nf = sum(v for k, v in cats.items() if not focal(k)); fo = sum(cats.values()) - nf
    ax.legend(handles=[Patch(color=BLUE, label=f"Focal (n={fo})"), Patch(color=ORANGE, label=f"Non-focal (n={nf})")],
              loc="lower right", frameon=False, fontsize=6, handlelength=1.0, handleheight=0.9, borderaxespad=0.2)
    heading(ax, f"{sum(cats.values())} test findings in {len(cats)} categories")

# =================================================================== e: text
def panel_e(ax):
    test = [t for e in test_json for t in e["findings"].values()]
    train = [t for e in train_json for t in e["findings"].values()]
    F = [("Laterality\n(left, right, bilateral)", r"\b(left|right|bilateral|both lungs|both sides|both)\b"),
         ("Lobe or segment", r"\b(lobe|lobes|lingula|lingular|segment|segments|apical|apex|apices|basal)\b"),
         ("Count or multiplicity", r"\b(multiple|several|a few|few|numerous|scattered|two|three|four|five|one or two|nodules|opacities|areas|foci|lesions)\b"),
         ("Size (mm or cm)", r"(\d+([.,]\d+)?\s?(mm|cm)\b|subcentimeter|centimeter|millimet)"),
         ("Adjacent structure\n(pleura, fissure, bronchi, ...)", r"\b(pleura|pleural|subpleural|fissure|diaphragm|mediastin|hilar|perihilar|peribronch|perivascular|adjacent|bronchovascular|paraseptal|centrilobular|interlobular)\b")]
    pct = lambda rx, L: 100 * sum(bool(re.search(rx, t, re.I)) for t in L) / len(L)
    ys = np.arange(len(F))[::-1]
    for y, (lab, rx) in zip(ys, F):
        pt, pr = pct(rx, test), pct(rx, train)
        ax.barh(y + 0.19, pt, height=0.36, color=BLUE, edgecolor="white", linewidth=0.6)
        ax.barh(y - 0.19, pr, height=0.36, color="#b9c6d6", edgecolor="white", linewidth=0.6)
        ax.text(pt + 1.5, y + 0.19, f"{pt:.0f}%", va="center", fontsize=5.8, color=INK2)
        ax.text(pr + 1.5, y - 0.19, f"{pr:.0f}%", va="center", fontsize=5.8, color=MUTED)
    ax.set_yticks(ys); ax.set_yticklabels([f[0] for f in F], fontsize=6, linespacing=1.1); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 108); ax.set_xlabel("Findings whose text specifies it (%)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.legend(handles=[Patch(color=BLUE, label=f"Test findings (n={len(test):,})"),
                       Patch(color="#b9c6d6", label=f"Training findings (n={len(train):,})")],
              loc="lower right", frameon=False, fontsize=5.8, handlelength=1.0, handleheight=0.9, borderaxespad=0.2)
    heading(ax, "What the finding text specifies")

# =================================================================== f: size & multiplicity
def panel_f(ax):
    vox = json.load(open(os.path.join(DATA, "test_gt_voxels.json")))
    spc = json.load(open(os.path.join(DATA, "test_spacing.json")))
    # GT instance counts from any scored run
    run = None
    for em in os.listdir(os.path.join(ROOT, "submissions")):
        d = os.path.join(ROOT, "submissions", em)
        if os.path.isdir(d):
            for r in os.listdir(d):
                p = os.path.join(d, r, "results.json")
                if os.path.exists(p):
                    run = json.load(open(p)); break
        if run:
            break
    inst = {c["file"]: {k.replace("finding_", ""): v["num_gt_instances"] for k, v in c["findings"].items()} for c in run["cases"]}
    vol, ninst = collections.defaultdict(list), collections.defaultdict(list)
    for e in test_json:
        n = e["name"]; sx, sz = spc[n]
        for k, cat in e["categories"].items():
            v = vox[n][int(k)] * sx * sx * sz / 1000.0
            if v > 0:
                vol[cat].append(v)
            if n in inst and k in inst[n]:
                ninst[cat].append(inst[n][k])
    order = [k for k in CAT_ORDER[::-1] if cats[k] >= 4]            # skip 2g (n=1), 2h (n=2)
    ys = range(len(order))
    for y, k in zip(ys, order):
        v = np.array(vol[k]); col = BLUE if focal(k) else ORANGE
        q1, med, q3 = np.percentile(v, [25, 50, 75]); lo, hi = v.min(), v.max()
        ax.plot([lo, hi], [y, y], color=col, lw=0.7, alpha=0.45, solid_capstyle="round")
        ax.plot([q1, q3], [y, y], color=col, lw=2.6, solid_capstyle="butt")
        ax.plot(med, y, marker="|", color="white", markersize=5, markeredgewidth=1.0, zorder=4)
        ax.text(1.0, y / len(order) + 0.5 / len(order), f"{np.median(ninst[k]):.0f}", transform=ax.transAxes,
                ha="left", va="center", fontsize=6, color=INK2, clip_on=False)
    ax.text(1.0, -0.03, "median\ninstances", transform=ax.transAxes, ha="left", va="top", fontsize=5.2, color=MUTED, linespacing=1.05)
    ax.set_xscale("log"); ax.set_xlim(0.02, 3000); ax.set_xticks([0.1, 1, 10, 100, 1000]); ax.set_xticklabels(["0.1", "1", "10", "100", "1000"])
    ax.set_xlabel("Finding volume (mL): median, IQR, range")
    ax.set_yticks(list(ys)); ax.set_yticklabels([NAMES[k] for k in order], fontsize=6); ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    heading(ax, "Finding size and multiplicity in the test set")

# =================================================================== layout
# Panels d-f (categories, text properties, finding size) are kept above but not used in Figure 1.
rows = {"a": (1 - 2.26 / H, 2.26 / H), "b": (1 - 4.40 / H, 2.06 / H), "c": (0.02, 1 - 4.55 / H - 0.02)}
def axes_at(x, y, w, h, row):
    ax = fig.add_axes([x, y, w, h]); ax._row_top = rows[row][0] + rows[row][1]; return ax
panel_a(0.0, rows["a"][0], 1.0, rows["a"][1]); letter(0.012, rows["a"][0] + rows["a"][1] + 0.004, "A")
panel_b(0.0, rows["b"][0], 1.0, rows["b"][1]); letter(0.012, rows["b"][0] + rows["b"][1] + 0.004, "B")
y, hh = rows["c"]
{"map": panel_c, "participation": panel_c_participation, "leaderboard": panel_c_leaderboard}[VARIANT_C](0.0, y, 1.0, hh)
letter(0.012, y + hh + 0.004, "C")

# Check: every text and patch in panel b must lie inside one of its two phase boxes.
fig.canvas.draw()
_r = fig.canvas.get_renderer(); _ax = panel_b.ax
_boxes = [b.get_window_extent(_r) for b in _ax._boxes]
_bad = []
for t in _ax.texts:
    e = t.get_window_extent(_r)
    if not any(bb.x0 <= e.x0 and e.x1 <= bb.x1 and bb.y0 <= e.y0 and e.y1 <= bb.y1 for bb in _boxes):
        _bad.append(t.get_text().split("\n")[0])
if _bad:
    raise SystemExit("panel b text outside its box: " + "; ".join(_bad))
print("panel b: all text inside boxes")
SUFFIX = "" if VARIANT_C == "map" else f"_{VARIANT_C}"
# Vector output; dpi only sets the resolution at which the embedded CT bitmaps are resampled.
for ext in ("pdf", "svg"):
    fig.savefig(os.path.join(OUT, f"fig1{SUFFIX}.{ext}"), bbox_inches="tight", dpi=900)
fig.savefig(os.path.join(OUT, f"fig1{SUFFIX}_preview.png"), dpi=200, bbox_inches="tight")
print("ok")
