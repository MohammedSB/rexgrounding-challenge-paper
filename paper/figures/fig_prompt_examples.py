"""Prompt experiment, adversarial examples (Team 2): the same finding segmented with the original sentence and with edited
sentences. Each panel shows the lungs from the front (patient's right on the left), the reference translucent aqua and the
predicted surface blue where it lies within 3 mm of the reference and orange elsewhere.
Run from the project root: python3 paper/figures/fig_prompt_examples.py
"""
import os, csv, json, textwrap
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
R = os.path.join(ROOT, "prompt_variants", "returns", "AIMHI")
BLUE, ORANGE, AQUA, LUNG, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#b8b8b8", "#0b0b0b", "#52514e"
# colour roles: reference / prediction agreeing / prediction outside. SCHEME=alt swaps to blue reference, green agree, red outside
if os.environ.get("SCHEME") == "alt":
    AQUA, BLUE, ORANGE = BLUE, AQUA, "#d64545"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
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


TEXT = {(m["variant"], m["scan"], m["finding"]): m["variant_text"] for m in csv.DictReader(open(os.path.join(ROOT, "prompt_variants", "manifest.csv")))}
SC = {}
def score(v, scan, k):
    if v not in SC:
        d = json.load(open(os.path.join(R, "scores", f"{v}.json")))
        SC[v] = {(c["file"], kk.split("_")[1]): f["global_dice"] for c in d["cases"] for kk, f in c["findings"].items()}
    return SC[v][(scan, k)]
LABEL = {"original": "Original", "swap_left_right": "Left and right swapped", "wrong_lobe": "Wrong lobe",
         "wrong_size": "Wrong size", "finding_only": "Finding name only", "left_only": "Left side only", "right_only": "Right side only"}
ROWS = [("train_3393_a_2", "3", ["original", "swap_left_right", "wrong_lobe", "wrong_size"]),
        ("train_13583_a_2", "0", ["original", "swap_left_right", "wrong_lobe", "finding_only"]),
        ("train_3094_a_2", "1", ["original", "left_only", "right_only", "wrong_lobe"])]
W = 7.0; pan_w = W / 4; pan_h = pan_w * 0.72; head_h = 0.16; legend_h = 0.24; WRAP = 44
def prompt_lines(scan, k, v): return textwrap.wrap(TEXT[(v, scan + ".nii.gz", k)], WRAP)
text_hs = [0.07 + 0.105 * max(len(prompt_lines(s, k, v)) for v in vs) for s, k, vs in ROWS]
row_hs = [head_h + th + pan_h + 0.08 for th in text_hs]
H = sum(row_hs) + legend_h
fig = plt.figure(figsize=(W, H))
for i, (case, k, vs) in enumerate(ROWS):
    f = case + ".nii.gz"; ki = int(k)
    img = nib.load(os.path.join(Q, f"{case}__ct.nii.gz")); ct = np.asarray(img.dataobj).astype(np.float32)
    sp = tuple(float(x) for x in img.header.get_zooms()[:3])
    gt = np.asarray(nib.load(os.path.join(Q, f"{case}__gt.nii.gz")).dataobj)[ki] > 0
    lung_tri, ext = lungs(ct, sp)
    S_, lo = bbox(gt); gt_tri = mesh(gt[S_], sp, lo)
    dist = ndimage.distance_transform_edt(~gt, sampling=sp)
    y_top = 1 - sum(row_hs[:i]) / H
    for j, v in enumerate(vs):
        x0 = j * pan_w / W
        fig.text(x0 + 0.012, y_top - 0.02 / H, LABEL[v], ha="left", va="top", fontsize=7, color=INK, fontweight="bold")
        fig.text(x0 + 0.012, y_top - (head_h + 0.0) / H, "\n".join(prompt_lines(case, k, v)), ha="left", va="top",
                 fontsize=6.0, color=INK2, linespacing=1.2, style="italic")
        ax = fig.add_axes([x0, y_top - (head_h + text_hs[i] + pan_h) / H, pan_w / W, pan_h / H], projection="3d")
        add(ax, lung_tri, shaded(lung_tri, to_rgb(LUNG), 0.05))
        add(ax, gt_tri, shaded(gt_tri, to_rgb(AQUA), 0.25))
        if gt.sum() * np.prod(sp) < 1000:
            cz = np.array(np.nonzero(gt)).mean(1) * np.array(sp)   # single small reference lesion in these examples
            ax.scatter([cz[0]], [cz[1]], [cz[2]], s=70, facecolors="none", edgecolors=INK2, linewidths=0.6, depthshade=False)
        p = np.asarray(nib.load(os.path.join(R, "split", v, f)).dataobj)[ki] > 0
        if p.any():
            Sp, lop = bbox(p); tri = mesh(p[Sp], sp, lop)
            if tri is not None:
                cen = np.clip((tri.mean(1) / np.array(sp)).round().astype(int), 0, np.array(gt.shape) - 1)
                inside = dist[cen[:, 0], cen[:, 1], cen[:, 2]] <= 3.0
                colr = np.where(inside[:, None], np.array(to_rgb(BLUE))[None], np.array(to_rgb(ORANGE))[None])
                n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
                add(ax, tri, np.c_[np.clip(colr * (0.55 + 0.45 * np.abs(n @ LIGHT))[:, None], 0, 1), np.ones(len(tri))])
            if p.sum() * np.prod(sp) < 1000:                        # small prediction: ring each piece where it sits
                labp, npc = ndimage.label(ndimage.binary_dilation(p, iterations=6))
                for c_ in range(1, npc + 1):
                    part = p & (labp == c_)
                    if part.sum() < 10: continue                        # same minimum size as the official scorer
                    cz = np.array(np.nonzero(part)).mean(1) * np.array(sp)
                    ax.scatter([cz[0]], [cz[1]], [cz[2]], s=70, facecolors="none", edgecolors=ORANGE, linewidths=0.6, depthshade=False)
        fig.text(x0 + pan_w / W / 2, y_top - (head_h + text_hs[i] + pan_h * 0.93) / H, f"Dice {score(v, f, k):.2f}",
                 ha="center", va="top", fontsize=6.6, color=INK2)
        ax.set_xlim(0, ext[0]); ax.set_ylim(0, ext[1]); ax.set_zlim(0, ext[2])
        ax.set_box_aspect(tuple(ext), zoom=2.0); ax.view_init(elev=8, azim=-90); ax.set_axis_off(); ax.set_rasterized(True)
fig.legend(handles=[Patch(color=AQUA, alpha=0.5, label="Reference"), Patch(color=BLUE, label="Prediction agreeing with the reference"),
                    Patch(color=ORANGE, label="Prediction outside the reference")],
           loc="lower center", ncol=3, frameon=False, fontsize=6.8, handlelength=1.2, handleheight=0.9, bbox_to_anchor=(0.5, 0.0))
for e in ("pdf", "svg"):
    fig.savefig(os.path.join(ROOT, "paper", "figures", f"fig_prompt_examples.{e}"), dpi=450)
fig.savefig(os.path.join(ROOT, "paper", "figures", "fig_prompt_examples_preview.png"), dpi=180)
print("ok")
