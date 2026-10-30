"""Finding categories in the challenge test set (Figure 2). Run from the project root."""
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FOCAL, NONFOCAL, INK, INK2 = "#2a78d6", "#9ec5f4", "#0b0b0b", "#52514e"   # Figure 1 blues
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
NAMES = {"1a": "Bronchial wall thickening", "1b": "Bronchiectasis", "1c": "Emphysema", "1d": "Septal thickening", "1e": "Micronodules",
         "1f": "Other non-focal", "2a": "Linear opacities", "2b": "Atelectasis and consolidation", "2c": "Ground-glass opacity",
         "2d": "Nodules and masses", "2e": "Pleural effusion and thickening", "2f": "Honeycombing", "2g": "Pneumothorax", "2h": "Other focal"}
results = json.load(open(os.path.join(ROOT, "final_results_MICCAI2026.json")))["results"]
cats = {k: v["n"] for k, v in results[0]["per_category_full_test_set"].items() if v["n"] > 0}   # categories absent from the test set are not shown
focal = lambda k: k.startswith("2")
order = sorted(cats, key=lambda k: cats[k])
fig, ax = plt.subplots(figsize=(3.4, 2.6))   # one column of the two-column layout
for y, k in enumerate(order):
    ax.barh(y, cats[k], height=0.62, color=FOCAL if focal(k) else NONFOCAL, edgecolor="none")
    ax.text(cats[k] + 2.5, y, str(cats[k]), va="center", fontsize=6.5, color=INK2)
ax.set_yticks(range(len(order))); ax.set_yticklabels([NAMES[k] for k in order], fontsize=7, color=INK)
ax.tick_params(axis="y", length=0); ax.set_xticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_xlim(0, max(cats.values()) * 1.1)
ax.legend(handles=[Patch(color=FOCAL, label="Focal"), Patch(color=NONFOCAL, label="Non-focal")], loc="lower right",
          frameon=False, fontsize=7, handlelength=1.0, handleheight=0.9)
for ext in ("pdf", "svg"):
    fig.savefig(os.path.join(ROOT, "paper", "figures", f"fig_categories.{ext}"), bbox_inches="tight")
fig.savefig(os.path.join(ROOT, "paper", "figures", "fig_categories_preview.png"), dpi=220, bbox_inches="tight")
print("ok")
