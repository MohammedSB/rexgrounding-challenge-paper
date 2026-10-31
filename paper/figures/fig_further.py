"""Further analyses (Figure 7): detection vs delineation, finding size, language cues.
Reads paper/analysis/further_analyses.json (written by paper/analysis/further_analyses.py)."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "..", "analysis", "further_analyses.json")))
FOCAL, LIGHT, DEEP, INK, INK2, GRID = "#2a78d6", "#9ec5f4", "#0d3b7a", "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
DISPLAY = {"Raidium2": "Raidium"}
fig = plt.figure(figsize=(7.2, 2.5))
axA = fig.add_axes([0.065, 0.17, 0.245, 0.71]); axB = fig.add_axes([0.395, 0.17, 0.20, 0.71]); axC = fig.add_axes([0.79, 0.17, 0.195, 0.71])
for ax in (axA, axB, axC):
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2, labelcolor=INK)

# A: hit rate vs Dice among localized findings, 20 best teams, labelled by rank
det = sorted(J["detection"], key=lambda x: x["rank"])[:10]
OFF = {2: (-7, 5), 3: (7, -5)}                                       # label offsets in points; default right of the dot
for x in det:
    top = x["rank"] <= 3
    axA.scatter(x["hit_rate"], x["dice_given_hit"], s=24, color=FOCAL if top else LIGHT, edgecolors="none", zorder=3)
    ox, oy = OFF.get(x["rank"], (4.5, 0))
    axA.annotate(str(x["rank"]), (x["hit_rate"], x["dice_given_hit"]), xytext=(ox, oy), textcoords="offset points",
                 fontsize=6, color=INK if top else INK2, ha="left" if ox > 0 else "right", va="center", zorder=4,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4, shrinkA=0, shrinkB=2.5) if x["rank"] in OFF else None)
axA.set_xlabel("Hit rate", color=INK, labelpad=2); axA.set_ylabel("Dice on localized findings", color=INK, labelpad=2)
axA.set_xlim(0.68, 0.79); axA.set_ylim(0.41, 0.47); axA.set_xticks([0.70, 0.74, 0.78]); axA.set_yticks([0.42, 0.44, 0.46])
axA.grid(True, color=GRID, linewidth=0.5); axA.set_axisbelow(True)

# B: Dice (filled) and hit rate (outlined) by reference volume, ten best teams pooled
S = J["size"]; x = np.arange(len(S)); w = 0.38
axB.bar(x - w / 2, [s["dice_mean_top10"] for s in S], width=w, color=FOCAL, edgecolor="none", zorder=2)
axB.bar(x + w / 2, [s["hit_mean_top10"] for s in S], width=w, facecolor="none", edgecolor=FOCAL, linewidth=0.8, zorder=2)
for i, s in enumerate(S): axB.text(i, 0.985, f"n={s['n']}", ha="center", va="top", fontsize=5.6, color=INK2)
axB.set_xticks(x); axB.set_xticklabels([s["bin"].replace("-", "–") for s in S], fontsize=6.3); axB.tick_params(axis="x", length=0)
axB.set_xlabel("Reference volume (mL)", color=INK, labelpad=2); axB.set_ylim(0, 1.0); axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axB.yaxis.grid(True, color=GRID, linewidth=0.5); axB.set_axisbelow(True); axB.spines["left"].set_visible(False); axB.tick_params(axis="y", length=0)
axB.legend(handles=[Patch(facecolor=FOCAL, label="Dice"), Patch(facecolor="none", edgecolor=FOCAL, linewidth=0.8, label="Hit rate")],
           loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=6.2, handlelength=1.0, handleheight=0.85, labelspacing=0.3, borderaxespad=0.1)

# C: controlled prompt experiment (Teams 2, 3 and 4): the same findings with the original and with the edited sentence.
# Figure 6C shows the mean over the three teams; the per-team version goes to the appendix (fig_prompt_teams).
RET = os.path.join(HERE, "..", "..", "prompt_variants", "returns")
TEAMS = [("Team 2", "AIMHI", FOCAL, 0.24), ("Team 3", "fomofo", "#6da7ec", 0.0), ("Team 4", "needpaper", DEEP, -0.24)]   # label, folder, colour, row offset
PE = {lab: json.load(open(os.path.join(RET, d, "prompt_experiment.json")))["variants"] for lab, d, _, _ in TEAMS}
C = [("Size removed", "no_size"), ("Wrong size", "wrong_size"), ("Laterality removed", "no_laterality"),
     ("Lobe or segment removed", "no_lobe_segment"), ("Finding name only", "finding_only"),
     ("Left and right swapped", "swap_left_right"), ("Wrong lobe", "wrong_lobe")]

def prompt_panel(ax, per_team):
    ys = np.arange(len(C))[::-1]
    for y, (nm, v) in zip(ys, C):
        if per_team:
            for lab, _, col, dy in TEAMS:
                o, e = PE[lab][v]["dice_original"], PE[lab][v]["dice_variant"]
                ax.plot([o, e], [y + dy, y + dy], color=col, linewidth=0.7, alpha=0.6, zorder=1)
                ax.scatter(o, y + dy, s=12, facecolors="white", edgecolors=col, linewidths=0.8, zorder=3)
                ax.scatter(e, y + dy, s=12, color=col, edgecolors="none", zorder=3)
        else:
            o = np.mean([PE[lab][v]["dice_original"] for lab, *_ in TEAMS]); e = np.mean([PE[lab][v]["dice_variant"] for lab, *_ in TEAMS])
            ax.plot([o, e], [y, y], color=INK2, linewidth=0.8, zorder=1)
            ax.scatter(o, y, s=18, facecolors="white", edgecolors=FOCAL, linewidths=0.9, zorder=3)
            ax.scatter(e, y, s=18, color=FOCAL, edgecolors="none", zorder=3)
        ax.text(0.535, y, str(PE["Team 2"][v]["n"]), va="center", ha="right", fontsize=5.6, color=INK2)
    ax.axhline(1.5, color=GRID, linewidth=0.6)
    ax.set_yticks(ys); ax.set_yticklabels([c[0] for c in C], fontsize=6.3); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0.0, 0.54); ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4]); ax.set_xticklabels(["0", "0.1", "0.2", "0.3", "0.4"])
    ax.text(0.535, len(C) - 0.45, "n", va="bottom", ha="right", fontsize=5.6, color=INK2)
    ax.set_xlabel("Dice on edited findings", color=INK, labelpad=2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    handles = [Line2D([], [], linestyle="none", marker="o", ms=3.8, mfc="white", mec=FOCAL, mew=0.8, label="Original"),
               Line2D([], [], linestyle="none", marker="o", ms=3.8, color=FOCAL, label="Edited")]
    if per_team:
        handles += [Line2D([], [], color=col, linewidth=1.4, label=lab) for lab, _, col, _ in TEAMS]
        ax.set_ylim(-2.6, len(C) - 0.4)
    else:
        ax.set_ylim(-1.6, len(C) - 0.4)
    ax.legend(handles=handles, loc="lower left", ncol=2, frameon=False, fontsize=6.0, handlelength=1.0, labelspacing=0.25,
              columnspacing=1.0, borderaxespad=0.1)

prompt_panel(axC, per_team=False)

for xpos, letter in ((0.01, "A"), (0.345, "B"), (0.635, "C")):
    fig.text(xpos, 0.93, letter, fontsize=9, color=INK, ha="left", va="bottom")
for ext in ("pdf", "svg"): fig.savefig(os.path.join(HERE, f"fig_further.{ext}"))
fig.savefig(os.path.join(HERE, "fig_further_preview.png"), dpi=220); print("ok")

# Appendix figure: per-team results of the controlled prompt experiment
fig2 = plt.figure(figsize=(3.45, 2.6)); ax2 = fig2.add_axes([0.42, 0.165, 0.555, 0.74])
prompt_panel(ax2, per_team=True)
for ext in ("pdf", "svg"): fig2.savefig(os.path.join(HERE, f"fig_prompt_teams.{ext}"))
fig2.savefig(os.path.join(HERE, "fig_prompt_teams_preview.png"), dpi=220)
