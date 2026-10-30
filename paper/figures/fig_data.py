"""Challenge data (Figure 2): finding categories, reference volumes, findings per scan, sentence length.
Run from anywhere; paths are resolved relative to this file."""
import os, json, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FOCAL, NONFOCAL, INK, INK2, GREY = "#2a78d6", "#9ec5f4", "#0b0b0b", "#52514e", "#9a9a9a"   # Figure 1 blues
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "hatch.linewidth": 0.5, "svg.fonttype": "none", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
NAMES = {"1a": "Bronchial wall thickening", "1b": "Bronchiectasis", "1c": "Emphysema", "1d": "Septal thickening", "1e": "Micronodules",
         "1f": "Other non-focal", "2a": "Linear opacities", "2b": "Atelectasis and consolidation", "2c": "Ground-glass opacity",
         "2d": "Nodules and masses", "2e": "Pleural effusion and thickening", "2f": "Honeycombing", "2g": "Pneumothorax", "2h": "Other focal"}
focal = lambda k: k.startswith("2")

# ---- data ---------------------------------------------------------------------------------------------------------
test = json.load(open(os.path.join(DATA, "ReXGroundingCT_MICCAI2026_test.json")))["test"]
train = json.load(open(os.path.join(DATA, "dataset.json")))["train"]
voxels = json.load(open(os.path.join(DATA, "test_gt_voxels.json")))
spacing = json.load(open(os.path.join(DATA, "test_spacing.json")))

cat_test = collections.Counter(c for s in test for c in s["categories"].values())
cat_train = collections.Counter(c for s in train for c in s["categories"].values())
n_test, n_train = sum(cat_test.values()), sum(cat_train.values())
vols = collections.defaultdict(list)                       # reference volume (mL) of each test finding, by category
for s in test:
    vox, sp = voxels[s["name"]], spacing[s["name"]]
    for i, (k, cat) in enumerate(sorted(s["categories"].items(), key=lambda kv: int(kv[0]))):
        vols[cat].append(vox[i] * sp[0] * sp[0] * sp[1] / 1000.0)
per_scan = {"train": [len(s["findings"]) for s in train], "test": [len(s["findings"]) for s in test]}
words = {"train": [len(f.split()) for s in train for f in s["findings"].values()],
         "test": [len(f.split()) for s in test for f in s["findings"].values()]}
print(f"train {n_train} findings / {len(train)} scans; test {n_test} findings / {len(test)} scans")

order = sorted(NAMES, key=lambda k: (cat_test[k], cat_train[k]))     # rows, bottom to top
nonf = [k for k in order if not focal(k)]; foc = [k for k in order if focal(k)]
POS = {k: i for i, k in enumerate(nonf)}; POS.update({k: len(nonf) + 1 + i for i, k in enumerate(foc)})
HEAD = {len(nonf): "Non-focal", len(nonf) + len(foc) + 1: "Focal"}  # heading rows
NROW = len(nonf) + len(foc) + 2

# ---- layout -------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 2.55))
gs = fig.add_gridspec(2, 3, width_ratios=[2.05, 1.40, 1.25], height_ratios=[1, 1], wspace=0.26, hspace=0.80,
                      left=0.205, right=0.985, top=0.90, bottom=0.14)
axA = fig.add_subplot(gs[:, 0]); axB = fig.add_subplot(gs[:, 1], sharey=axA)
axC = fig.add_subplot(gs[0, 2]); axD = fig.add_subplot(gs[1, 2])
for ax in (axA, axB, axC, axD):
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2, labelcolor=INK)

# A: category share; filled = test set, outlined = training set
TRAIN = "#0d3b7a"                                           # deep blue for panel B summaries
SUFFIX = "" if os.environ.get("FIGDATA_A", "mirror") == "mirror" else "_" + os.environ["FIGDATA_A"]
VARIANT_A = os.environ.get("FIGDATA_A", "mirror")              # bars (paired bars) | tick (training as a tick) | mirror (back to back)
DEEP = "#0d3b7a"
if VARIANT_A == "tick":
    for y, k in enumerate(order):
        col = FOCAL if focal(k) else NONFOCAL
        pt, pr = 100 * cat_test[k] / n_test, 100 * cat_train[k] / n_train
        axA.barh(y, pt, height=0.62, color=col, edgecolor="none")
        axA.plot([pr, pr], [y - 0.42, y + 0.42], color=DEEP, linewidth=1.3, solid_capstyle="butt")
        axA.text(max(pt, pr) + 0.6, y, f"{cat_test[k]}", va="center", ha="left", fontsize=6, color=INK2)
    axA.set_xlim(0, 37); axA.set_xticks([0, 10, 20, 30])
    kx, kw = 21.0, 5.0
    axA.barh(3.0, kw, height=0.62, left=kx, color=FOCAL, edgecolor="none")
    axA.text(kx + kw + 0.8, 3.0, "Test set", va="center", ha="left", fontsize=6.2, color=INK)
    axA.plot([kx + kw / 2] * 2, [1.85, 2.55], color=DEEP, linewidth=1.3, solid_capstyle="butt")
    axA.text(kx + kw + 0.8, 2.2, "Training set", va="center", ha="left", fontsize=6.2, color=INK)
    for yk, c, lab in [(1.2, FOCAL, "Focal"), (0.5, NONFOCAL, "Non-focal")]:
        axA.barh(yk, kw, height=0.36, left=kx, color=c, edgecolor="none")
        axA.text(kx + kw + 0.8, yk, lab, va="center", ha="left", fontsize=6.2, color=INK)
elif VARIANT_A == "mirror":
    LIGHT = "#9ec5f4"
    for k in order:
        y = POS[k]; pt, pr = 100 * cat_test[k] / n_test, 100 * cat_train[k] / n_train
        axA.barh(y, pt, height=0.62, color=FOCAL, edgecolor="none")
        axA.barh(y, -pr, height=0.62, color=LIGHT, edgecolor="none")
        axA.text(pt + 0.6, y, f"{pt:.1f}", va="center", ha="left", fontsize=6, color=INK2)
        axA.text(-pr - 0.6, y, f"{pr:.1f}", va="center", ha="right", fontsize=6, color=INK2)
    axA.axvline(0, color=INK2, linewidth=0.6)
    axA.set_xlim(-31, 39); axA.set_xticks([-30, -20, -10, 0, 10, 20, 30]); axA.set_xticklabels(["30", "20", "10", "0", "10", "20", "30"])
    axA.text(-15, NROW - 0.55, "Training set", ha="center", va="bottom", fontsize=6.5, color=INK)
    axA.text(17, NROW - 0.55, "Test set", ha="center", va="bottom", fontsize=6.5, color=INK)
else:
    h = 0.36
    for y, k in enumerate(order):
        col = FOCAL if focal(k) else NONFOCAL
        pt, pr = 100 * cat_test[k] / n_test, 100 * cat_train[k] / n_train
        axA.barh(y + h / 2, pt, height=h, color=col, edgecolor="none")
        axA.barh(y - h / 2, pr, height=h, facecolor="none", edgecolor=col, linewidth=0.8)
        axA.text(max(pt, pr) + 0.6, y, f"{cat_test[k]}", va="center", ha="left", fontsize=6, color=INK2)
    axA.set_xlim(0, 37); axA.set_xticks([0, 10, 20, 30])
    # key drawn as a sample row in data coordinates, so the two splits look exactly as in the plot
    kx, kw = 21.0, 5.0                                          # left edge and bar length of the key
    for yk, lab, style in [(3.0, "Test set", dict(color=FOCAL, edgecolor="none")),
                           (2.3, "Training set", dict(facecolor="none", edgecolor=FOCAL, linewidth=0.8))]:
        axA.barh(yk, kw, height=h, left=kx, **style)
        axA.text(kx + kw + 0.8, yk, lab, va="center", ha="left", fontsize=6.2, color=INK)
    for yk, c, lab in [(1.2, FOCAL, "Focal"), (0.5, NONFOCAL, "Non-focal")]:
        axA.barh(yk, kw, height=h, left=kx, color=c, edgecolor="none")
        axA.text(kx + kw + 0.8, yk, lab, va="center", ha="left", fontsize=6.2, color=INK)
if VARIANT_A == "mirror":
    ticks = sorted(list(POS.values()) + list(HEAD))
    labs = {**{POS[k]: NAMES[k] for k in order}, **HEAD}
    axA.set_yticks(ticks); axA.set_yticklabels([labs[p] for p in ticks], fontsize=7, color=INK)
    for lab, p in zip(axA.get_yticklabels(), ticks):
        if p in HEAD: lab.set_text(HEAD[p].upper()); lab.set_color(INK2); lab.set_fontsize(6.2)
    axA.set_yticklabels([lab.get_text() for lab in axA.get_yticklabels()])
    for lab, p in zip(axA.get_yticklabels(), ticks):
        lab.set_fontsize(6.2 if p in HEAD else 7); lab.set_color(INK2 if p in HEAD else INK)
    for ax in (axA, axB):                                    # light rule between the focal and non-focal groups
        ax.axhline(len(nonf) + 0.5, color="#d6d5d0", linewidth=0.5, zorder=0)
    axA.set_ylim(-0.6, NROW - 0.4)
else:
    POS = {k: i for i, k in enumerate(order)}
    axA.set_yticks(range(len(order))); axA.set_yticklabels([NAMES[k] for k in order], fontsize=7, color=INK)
    axA.set_ylim(-0.6, len(order) - 0.4)
axA.tick_params(axis="y", length=0); axA.spines["left"].set_visible(False)
axA.set_xlabel("Share of findings (%)", color=INK, labelpad=2)

# B: reference volume per category (test set), log scale; dots per finding, IQR bar and median tick
rng = np.random.default_rng(0)
for x in (0.01, 0.1, 1, 10, 100, 1000):
    axB.axvline(x, color="#e6e6e3" if x != 1 else GREY, linewidth=0.5 if x != 1 else 0.6,
                linestyle="-" if x != 1 else (0, (2, 2)), zorder=0)
for k in order:
    y = POS[k]; v = np.asarray(vols.get(k, []))
    if len(v) == 0: continue
    col = FOCAL if focal(k) else "#6fa8ec"
    axB.scatter(v, y + rng.uniform(-0.17, 0.17, len(v)), s=4, color=col, alpha=0.55, linewidths=0, zorder=2)
    if len(v) >= 5:
        q25, q50, q75 = np.percentile(v, [25, 50, 75])
        axB.plot([q25, q75], [y, y], color=TRAIN, linewidth=2.2, alpha=0.9, solid_capstyle="butt", zorder=3)
        axB.plot([q50], [y], marker="o", ms=3.4, mfc="white", mec=TRAIN, mew=0.9, zorder=4)
axB.set_xscale("log"); axB.set_xlim(0.008, 3000)
axB.set_xticks([0.01, 0.1, 1, 10, 100, 1000]); axB.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1000"])
axB.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axB.set_xlabel("Reference volume per finding (mL)", color=INK, labelpad=2)
axB.tick_params(axis="y", length=0, labelleft=False); axB.spines["left"].set_visible(False)

# C / D: grouped bars, test (filled) next to training (light), in percent
def grouped(ax, data, edges, labels, xlabel, ylabel):
    w = 0.38
    for j, split in enumerate(("test", "train")):
        x = np.asarray(data[split]); pct = [100 * np.mean((x >= lo) & (x <= hi)) for lo, hi in edges]
        style = dict(color=FOCAL if split == "test" else "#9ec5f4", edgecolor="none")
        ax.bar(np.arange(len(edges)) + (j - 0.5) * w, pct, width=w * 0.92, zorder=2, **style)
    ax.set_xticks(range(len(edges))); ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0, labelsize=6.3)
    ax.yaxis.grid(True, color="#e6e6e3", linewidth=0.5, zorder=0); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel, color=INK, labelpad=2); ax.set_ylabel(ylabel, color=INK, labelpad=2)
grouped(axC, per_scan, [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 99)], ["1", "2", "3", "4", "5", "6+"],
        "Findings per scan", "Scans (%)")
axC.set_yticks([0, 20, 40])
grouped(axD, words, [(1, 5), (6, 10), (11, 15), (16, 20), (21, 99)], ["1–5", "6–10", "11–15", "16–20", "21+"],
        "Words per finding sentence", "Findings (%)")
axD.set_yticks([0, 20, 40])
axD.tick_params(axis="x", labelsize=5.6, pad=2)
axC.legend(handles=[Patch(facecolor=FOCAL, label="Test set"),
                    Patch(facecolor="#9ec5f4", label="Training set")],
           loc="upper right", frameon=False, fontsize=6.2, handlelength=1.0, handleheight=0.85, labelspacing=0.3, borderaxespad=0.1)
fig.align_ylabels([axC, axD])

# panel letters, aligned on one baseline above the panels
ytop = axA.get_position().y1 + 0.03
for ax, letter, dx in ((axA, "A", -0.19), (axB, "B", -0.012), (axC, "C", -0.055), (axD, "D", -0.055)):
    x = ax.get_position().x0 + dx
    y = ytop if ax is not axD else axD.get_position().y1 + 0.03
    fig.text(x, y, letter, fontsize=9, color=INK, ha="left", va="bottom")

for ext in ("pdf", "svg"):
    fig.savefig(os.path.join(HERE, f"fig_data{SUFFIX}.{ext}"))
fig.savefig(os.path.join(HERE, f"fig_data{SUFFIX}_preview.png"), dpi=220)
print("ok")
