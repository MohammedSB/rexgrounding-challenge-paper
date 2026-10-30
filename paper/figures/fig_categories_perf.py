"""Per-category performance of the ten best teams (Figure 4): mean Dice per finding on the full test set.
Inputs: final_results_MICCAI2026.json, final_rankings_merged.csv (project root). Run from anywhere."""
import os, json, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
LIGHT, FOCAL, DEEP, INK, INK2 = "#9ec5f4", "#2a78d6", "#0d3b7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
NAMES = {"1a": "Bronchial wall thickening", "1b": "Bronchiectasis", "1c": "Emphysema", "1d": "Septal thickening", "1e": "Micronodules",
         "1f": "Other non-focal", "2a": "Linear opacities", "2b": "Atelectasis and consolidation", "2c": "Ground-glass opacity",
         "2d": "Nodules and masses", "2e": "Pleural effusion and thickening", "2f": "Honeycombing", "2g": "Pneumothorax", "2h": "Other focal"}
DISPLAY = {"Raidium2": "Raidium"}
K, MIN_N = 10, 4                                                    # teams shown; smallest category shown

R = json.load(open(os.path.join(ROOT, "final_results_MICCAI2026.json")))["results"]
top = list(csv.DictReader(open(os.path.join(ROOT, "final_rankings_merged.csv"))))[:K]
pcs = [next(x for x in R if abs(x["metrics_full_test_set"]["dice"] - float(r["dice_full_test"])) < 5e-6)["per_category_full_test_set"]
       for r in top]
n = {k: v["n"] for k, v in pcs[0].items()}
def group(prefix):
    cats = sorted([k for k in n if k.startswith(prefix) and n[k] >= MIN_N], key=lambda k: -n[k])
    rows = [(f"{NAMES[k]} ({n[k]})", [pc[k]["dice"] for pc in pcs], False) for k in cats]
    tot = sum(n[k] for k in n if k.startswith(prefix))
    agg = [sum(pc[k]["n"] * pc[k]["dice"] for k in pc if k.startswith(prefix) and pc[k]["n"]) / tot for pc in pcs]
    return rows + [(f"All {'focal' if prefix == '2' else 'non-focal'} ({tot})", agg, True)]
rows = group("2") + group("1")
M = np.array([r[1] for r in rows])

fig = plt.figure(figsize=(3.45, 3.55))
ax = fig.add_axes([0.445, 0.10, 0.545, 0.645])
VARIANT = os.environ.get("VARIANT", "relative")
OUTSUFFIX = os.environ.get("OUTSUFFIX", "")
if VARIANT == "ylgnbu":                       # multi-hue sequential, light yellow -> green -> blue
    cmap = LinearSegmentedColormap.from_list("b", plt.get_cmap("YlGnBu")(np.linspace(0.03, 0.85, 256)))
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=0.50, clip=True); C = M
elif VARIANT == "mako":                       # seaborn-like dark teal-blue ramp, reversed (light = low)
    cmap = LinearSegmentedColormap.from_list("b", ["#f3f9f6", "#c9ebe0", "#8fd3c8", "#4fb3bf", "#3a8ab3", "#36639f", "#35427f"])
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=0.50, clip=True); C = M
elif VARIANT == "relative":                   # colour = difference from the category mean across the ten teams
    C = M - M.mean(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("d", ["#d9622b", "#f2b48f", "#f7f7f5", "#9ec5f4", "#2a78d6"])
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-0.10, vcenter=0, vmax=0.10)
ax.imshow(np.clip(C, norm.vmin, norm.vmax), cmap=cmap, norm=norm, aspect="auto")
def lum(c):
    r, g, b = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c[:3]]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
for i, (_, vals, is_sum) in enumerate(rows):
    for j, v in enumerate(vals):
        ax.text(j, i, f"{v:.2f}"[1:], ha="center", va="center", fontsize=5.4,
                color=INK)
GAP = 0.25                                                           # extra column: mean of the ten teams, uncoloured
xm = K + GAP
for i, (_, vals, is_sum) in enumerate(rows):
    ax.add_patch(plt.Rectangle((xm - 0.5, i - 0.5), 1, 1, facecolor="#ecebe8", edgecolor="none"))
    ax.text(xm, i, f"{np.mean(vals):.2f}"[1:], ha="center", va="center", fontsize=5.4, color=INK)
ax.set_xlim(-0.5, xm + 0.5)
ax.set_xticks(list(range(K)) + [xm]); ax.set_xticklabels([f"Team {r['rank']}" for r in top] + ["Mean"], rotation=90, fontsize=6.2)
ax.xaxis.set_ticks_position("top")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
for lab, (_, _, is_sum) in zip(ax.get_yticklabels(), rows):
    lab.set_color(INK if is_sum else INK2)
ax.tick_params(length=0, pad=2)
for sp in ax.spines.values(): sp.set_visible(False)
nf = len(group("2"))
for y in (nf - 1.5, nf - 0.5, len(rows) - 1.5):                     # thin white gaps around the summary rows
    ax.axhline(y, color="white", linewidth=1.6 if y == nf - 0.5 else 0.8)
cax = fig.add_axes([0.445 + 0.545 * 0.25, 0.045, 0.545 * 0.5, 0.012])
cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
cbar.outline.set_visible(False); cbar.ax.tick_params(labelsize=5.8, length=0, pad=1.5)
if VARIANT == "relative":
    cbar.set_ticks([-0.10, 0, 0.10]); cbar.ax.set_xticklabels(["\u2264\u22120.10", "0", "\u2265+0.10"])
else:
    cbar.set_ticks([0, 0.25, 0.5]); cbar.ax.set_xticklabels(["0", "0.25", "0.50"])
fig.text(0.445 + 0.545 * 0.25 - 0.015, 0.051, "Dice vs category mean" if VARIANT == "relative" else "Dice", ha="right", va="center", fontsize=6.2, color=INK)
for ext in ("pdf", "svg"):
    fig.savefig(os.path.join(HERE, f"fig_categories_perf{OUTSUFFIX}.{ext}"))
fig.savefig(os.path.join(HERE, f"fig_categories_perf{OUTSUFFIX}_preview.png"), dpi=250)
print("ok", M.shape)
