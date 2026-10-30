"""Ranking stability (Figure 3): bootstrap rank distribution and pairwise comparisons.
Reads paper/analysis/ranking_stats.json (written by paper/analysis/ranking_stats.py)."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "..", "analysis", "ranking_stats.json")))
FOCAL, LIGHT, DEEP, INK, INK2, GRID = "#2a78d6", "#9ec5f4", "#0d3b7a", "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
N, K = 20, 10                                                     # teams shown in A, teams in the pairwise matrix (B)
teams = sorted(S["teams"], key=lambda t: t["rank"])[:N]
B = S["B"]

FH = 3.6
fig = plt.figure(figsize=(7.2, FH))
axA = fig.add_axes([0.14, 0.11, 0.37, 0.79])
for side in ("top", "right", "left"): axA.spines[side].set_visible(False)
axA.tick_params(axis="y", length=0); axA.tick_params(colors=INK2, labelcolor=INK)
axA.xaxis.grid(True, color=GRID, linewidth=0.5); axA.set_axisbelow(True)
ys = np.arange(N)
DISPLAY = {"Raidium2": "Raidium"}                                  # match Table 1

# A: bootstrap rank distribution (blob area = share of replicates)
maxr = 1 + max(max(i for i, c in enumerate(t["rank_counts"], 1) if c) for t in teams)
for y, t in zip(ys, teams):
    r = np.arange(1, len(t["rank_counts"]) + 1); c = np.asarray(t["rank_counts"]) / B
    m = c > 0
    axA.scatter(r[m], np.full(m.sum(), y), s=c[m] * 70, color=FOCAL if t["rank"] <= 3 else LIGHT,
                edgecolors="none", alpha=0.95, zorder=2)
    axA.plot(t["rank_range_95"], [y, y], color=DEEP, linewidth=0.6, zorder=1)
axA.plot([0.5, N + 0.5], [-0.5, N - 0.5], color=INK2, linewidth=0.5, linestyle=(0, (2, 2)), zorder=0)
axA.set_yticks(ys); axA.set_yticklabels([f"Team {t['rank']}" for t in teams], fontsize=6.6)
axA.set_ylim(N - 0.4, -0.6)
axA.set_xlim(0.3, maxr); axA.set_xticks([x for x in (1, 5, 10, 15, 20, 25) if x < maxr])
axA.set_xlabel("Rank in bootstrap replicates", color=INK, labelpad=2)
axA.legend(handles=[Line2D([], [], linestyle="none", marker="o", ms=np.sqrt(f * 70), color=LIGHT, label=f"{int(f*100)}%")
                    for f in (0.1, 0.5, 1.0)] + [Line2D([], [], color=DEEP, linewidth=0.6, label="95% range")],
           title="Share of replicates", title_fontsize=6.2, loc="lower left", frameon=False, fontsize=6.2,
           handlelength=1.2, labelspacing=0.5, borderaxespad=0.2)

# B: pairwise comparisons among the top K; cell = share of replicates in which the row team beats the column team
axC = fig.add_axes([0.605, 0.11 + 0.79 - 0.375 * 7.2 / FH - 0.055, 0.375, 0.375 * 7.2 / FH])
pairs = {(p["a"], p["b"]): p for p in S["pairs_top10"]}
top = [t["team"] for t in teams[:K]]
M = np.full((K, K), np.nan)
for i, a in enumerate(top):
    for j, b in enumerate(top):
        if j <= i: continue
        p = pairs.get((a, b))
        win, sig = (p["boot_win"], p["significant"]) if p else (1 - pairs[(b, a)]["boot_win"], pairs[(b, a)]["significant"])
        M[i, j] = win
        axC.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, facecolor=FOCAL if sig else "#e8f1fc",
                                    edgecolor="none"))
        axC.text(j, i, f"{round(100 * win)}", ha="center", va="center", fontsize=5.6, color="white" if sig else INK2)
axC.set_xticks(range(K)); axC.set_xticklabels([str(i + 1) for i in range(K)], fontsize=6.4)
axC.set_yticks(range(K)); axC.set_yticklabels([str(i + 1) for i in range(K)], fontsize=6.4)
axC.xaxis.set_ticks_position("top"); axC.xaxis.set_label_position("top")
axC.tick_params(length=0, colors=INK2, labelcolor=INK)
for sp in axC.spines.values(): sp.set_visible(False)
axC.set_xlim(0.5, K - 0.5); axC.set_ylim(K - 1.5, -0.5)
axC.set_ylabel("Higher-ranked team", color=INK, labelpad=2)
axC.set_xlabel("Lower-ranked team", color=INK, labelpad=3)
axC.set_aspect("equal")
from matplotlib.patches import Patch
axC.legend(handles=[Patch(facecolor=FOCAL, label="Significant"), Patch(facecolor="#e8f1fc", label="Not significant")],
           loc="lower left", frameon=False, fontsize=6.2, handlelength=1.0, handleheight=0.85, labelspacing=0.4,
           borderaxespad=0.3, title="Numbers: % of replicates\nin which the row team wins", title_fontsize=6.2,
           alignment="left")

for ax, letter, dx in ((axA, "A", -0.115), (axC, "B", -0.055)):
    fig.text(ax.get_position().x0 + dx, 0.955, letter, fontsize=9, color=INK, ha="left", va="bottom")
for ext in ("pdf", "svg"):
    fig.savefig(os.path.join(HERE, f"fig_ranking.{ext}"))
fig.savefig(os.path.join(HERE, "fig_ranking_preview.png"), dpi=220)
print("ok", [t["team"] for t in teams[:K]])
