"""3D qualitative comparison of the top three teams with the reference (Figure 3).
Each panel shows the whole lungs (translucent) seen from the front, so the location of each mask is visible.
Reference column: reference in aqua. Team columns: the predicted surface, blue where it lies inside the
reference and orange where it lies outside, with the reference shown translucent aqua.
Run from the project root: python3 paper/figures/fig_qualitative3d.py
"""
import os, json, textwrap
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy import ndimage
from skimage import measure
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Q = os.path.join(ROOT, "paper", "figures", "data", "qual")
BLUE, ORANGE, AQUA, LUNG, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#b8b8b8", "#0b0b0b", "#52514e"
# colour roles: reference / prediction agreeing / prediction outside. SCHEME=alt swaps to blue reference, green agree, red outside
if os.environ.get("SCHEME") == "alt":
    AQUA, BLUE, ORANGE = BLUE, AQUA, "#d64545"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
test = {e["name"]: e for e in json.load(open(os.path.join(ROOT, "paper/figures/data/ReXGroundingCT_MICCAI2026_test.json")))["test"]}
RES = {"CIRCLE Team": "submissions/2572837347@qq.com/CIRCLE__HtYfefYk/results.json",
       "AIMHI": "submissions/khuhm@gachon.ac.kr/Grounder_1.4n__QZ7lRcdv/results.json",
       "fomofo": "submissions/r.vt.hull@nki.nl/TAGS_TAPv3__hfYYpTHz/results.json"}
FILE = {"CIRCLE Team": "CIRCLE", "AIMHI": "AIMHI", "fomofo": "fomofo"}
dice = {t: {c["file"]: c["findings"] for c in json.load(open(os.path.join(ROOT, p)))["cases"]} for t, p in RES.items()}
import sys
SETS = {"success": [("train_13492_a_2", 0, False), ("train_3393_a_2", 3, False), ("train_13583_a_2", 0, False),
                    ("train_4092_a_2", 0, False), ("train_13285_a_2", 2, False)],
        "failure": [("train_13479_f_1", 0, True), ("train_3249_a_1", 0, False), ("train_3309_a_2", 0, False),
                    ("train_3841_a_2", 2, False), ("train_4079_a_1", 2, False)]}
WHICH = sys.argv[1] if len(sys.argv) > 1 else "success"
ROWS = SETS[WHICH]
OTHER = "#7d7d7d"
LIGHT = np.array([0.3, -0.8, 0.5]); LIGHT /= np.linalg.norm(LIGHT)

def mesh(vol, sp, off=(0, 0, 0), smooth=0.8, step=1):
    v = ndimage.gaussian_filter(vol.astype(np.float32), smooth) if smooth else vol.astype(np.float32)
    if v.max() < 0.5: return None
    verts, faces, _, _ = measure.marching_cubes(np.pad(v, 1), 0.5, spacing=sp, step_size=step)
    verts = verts - np.array(sp) + np.array(off) * np.array(sp)
    return verts[faces]

def shaded(tri, color, alpha):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    s = 0.55 + 0.45 * np.abs(n @ LIGHT)
    col = np.atleast_2d(color)
    rgb = np.clip(col * s[:, None], 0, 1)
    return np.c_[rgb, np.full(len(rgb), alpha)]

def add(ax, tri, fc):
    ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolors="none", linewidths=0))

def bbox(m, pad=3):
    idx = np.array(np.nonzero(m)); lo = np.maximum(idx.min(1) - pad, 0); hi = np.minimum(idx.max(1) + pad + 1, m.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi)), lo

def lungs(ct, sp):
    f = (4, 4, 2); c = ct[::f[0], ::f[1], ::f[2]]
    body = np.stack([ndimage.binary_fill_holes(c[:, :, z] > -500) for z in range(c.shape[2])], 2)
    air = (c < -500) & ndimage.binary_erosion(body, iterations=2)
    lab, n = ndimage.label(air); sz = ndimage.sum(air, lab, range(1, n + 1))
    order = list(np.argsort(sz)[::-1]); first = lab == order[0] + 1
    zr = np.nonzero(first.any((0, 1)))[0]; keep = first.copy()
    for o in order[1:3]:                                   # second lung: must overlap the first in height
        comp = lab == o + 1; zc = np.nonzero(comp.any((0, 1)))[0]
        ov = max(0, min(zr.max(), zc.max()) - max(zr.min(), zc.min()))
        if sz[o] > 0.05 * sz.max() and ov > 0.5 * (zc.max() - zc.min()):
            keep |= comp; break
    keep = ndimage.binary_closing(keep, iterations=2)
    return mesh(keep, tuple(s * k for s, k in zip(sp, f)), smooth=1.0), np.array(keep.shape) * np.array([s * k for s, k in zip(sp, f)])

cols = ["Reference"] + list(RES)
W = 7.0; pan_w = W / 4; pan_h = pan_w * 0.72; text_h = 0.17; dice_h = 0.13; head = 0.22; legend_h = 0.22
WRAP = 150
SHOW = lambda txt: txt.replace(", present on prior examination", "")   # display trims
lines = [len(textwrap.wrap(SHOW(test[c + '.nii.gz']['findings'][str(k)]), WRAP)) for c, k, _ in ROWS]
text_hs = [0.06 + 0.115 * n for n in lines]
row_hs = [th + pan_h + 0.06 for th in text_hs]
H = head + sum(row_hs) + legend_h
fig = plt.figure(figsize=(W, H))
for j, name in enumerate(cols):
    fig.text((j + 0.5) * pan_w / W, 1 - 0.1 / H, {"CIRCLE Team": "Team 1", "AIMHI": "Team 2", "fomofo": "Team 3"}.get(name, name),
             ha="center", va="center", fontsize=7.5, color=INK)
for i, (case, k, show_other) in enumerate(ROWS):
    f = case + ".nii.gz"
    img = nib.load(os.path.join(Q, f"{case}__ct.nii.gz")); ct = np.asarray(img.dataobj).astype(np.float32)
    sp = tuple(float(x) for x in img.header.get_zooms()[:3])
    gt_all = np.asarray(nib.load(os.path.join(Q, f"{case}__gt.nii.gz")).dataobj) > 0
    gt = gt_all[k]
    other_tri = []
    if show_other:
        for jj in range(gt_all.shape[0]):
            if jj != k and gt_all[jj].any():
                So, loo = bbox(gt_all[jj]); other_tri.append(mesh(gt_all[jj][So], sp, loo))
    lung_tri, ext = lungs(ct, sp)
    S_, lo = bbox(gt); gt_tri = mesh(gt[S_], sp, lo)
    dist = ndimage.distance_transform_edt(~gt, sampling=sp)          # mm from the reference
    TOL = 3.0                                                        # surface within 3 mm counts as agreeing
    text_h = text_hs[i]; y_top = 1 - (head + sum(row_hs[:i])) / H
    fig.text(0.0, y_top - 0.02 / H, "\n".join(textwrap.wrap(SHOW(test[f]["findings"][str(k)]), WRAP)), ha="left", va="top", fontsize=6.8, color=INK, linespacing=1.25)
    for j, name in enumerate(cols):
        ax = fig.add_axes([j * pan_w / W, y_top - (text_h + pan_h) / H, pan_w / W, pan_h / H], projection="3d")
        add(ax, lung_tri, shaded(lung_tri, to_rgb(LUNG), 0.05))
        for ot in other_tri:
            if ot is not None: add(ax, ot, shaded(ot, to_rgb(OTHER), 0.45))
        if gt.sum() * np.prod(sp) < 1000:                               # under 1 mL: ring the lesion so it can be found
            labc, ncc = ndimage.label(ndimage.binary_dilation(gt, iterations=6))   # group nearby pieces
            for cc_ in range(1, ncc + 1):
                part = gt & (labc == cc_)
                if part.sum() < 30: continue
                cz = np.array(np.nonzero(part)).mean(1) * np.array(sp)
                ax.scatter([cz[0]], [cz[1]], [cz[2]], s=70, facecolors='none', edgecolors=INK2, linewidths=0.6, depthshade=False)
        if name == "Reference":
            add(ax, gt_tri, shaded(gt_tri, to_rgb(AQUA), 1.0))
        else:
            add(ax, gt_tri, shaded(gt_tri, to_rgb(AQUA), 0.22))
            p = np.asarray(nib.load(os.path.join(Q, f"{case}__{FILE[name]}.nii.gz")).dataobj)[k] > 0
            if p.any():
                Sp, lop = bbox(p); tri = mesh(p[Sp], sp, lop)
                if tri is not None:
                    cen = np.clip((tri.mean(1) / np.array(sp)).round().astype(int), 0, np.array(gt.shape) - 1)
                    inside = dist[cen[:, 0], cen[:, 1], cen[:, 2]] <= TOL
                    colr = np.where(inside[:, None], np.array(to_rgb(BLUE))[None], np.array(to_rgb(ORANGE))[None])
                    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
                    s = 0.55 + 0.45 * np.abs(n @ LIGHT)
                    add(ax, tri, np.c_[np.clip(colr * s[:, None], 0, 1), np.ones(len(tri))])
            d = dice[name][f][f"finding_{k}"]["global_dice"]
            fig.text((j + 0.5) * pan_w / W, y_top - (text_h + pan_h * 0.90) / H, f"Dice {d:.2f}", ha="center", va="top", fontsize=6.6, color=INK2)
        ax.set_xlim(0, ext[0]); ax.set_ylim(0, ext[1]); ax.set_zlim(0, ext[2])
        ax.set_box_aspect(tuple(ext), zoom=2.0); ax.view_init(elev=8, azim=-90); ax.set_axis_off(); ax.set_rasterized(True)
fig.legend(handles=[Patch(color=AQUA, label="Reference"), Patch(color=BLUE, label="Prediction agreeing with the reference"),
                    Patch(color=ORANGE, label="Prediction outside the reference")] + ([Patch(color=OTHER, alpha=0.5, label="Another finding in the same scan")] if any(r[2] for r in ROWS) else []),
           loc="lower center", ncol=4, frameon=False, fontsize=6.8, handlelength=1.2, handleheight=0.9, bbox_to_anchor=(0.5, 0.0))
for e in ("pdf", "svg"):
    fig.savefig(os.path.join(ROOT, "paper", "figures", f"fig_qualitative3d_{WHICH}.{e}"), dpi=450)
fig.savefig(os.path.join(ROOT, "paper", "figures", f"fig_qualitative3d_{WHICH}_preview.png"), dpi=180)
print("ok")
