"""Submissions over the development phase (participation figure): scored submissions per week and the
best public-split Dice over time. Reads data/submission_timeline.csv (written by paper/analysis/submission_timeline.py)."""
import os, csv, json, datetime as dt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
FOCAL, LIGHT, DEEP, INK, INK2, MUTED, GRID = "#2a78d6", "#9ec5f4", "#0d3b7a", "#0b0b0b", "#52514e", "#8a8984", "#e6e6e3"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
rows = list(csv.DictReader(open(os.path.join(HERE, "data", "submission_timeline.csv"))))
P = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)      # UTC, naive
subs = sorted((P(r["created_at"]), float(r["public_dice"])) for r in rows)
DEADLINE = dt.datetime(2026, 9, 15, 3, 59)       # 14 September 2026, 11:59 PM ET, in UTC
OPEN = dt.datetime(2026, 6, 14)
# seven-day windows counted back from the deadline, so the last bar is the final week before it
weeks = [DEADLINE - dt.timedelta(days=7 * k) for k in range(14, 0, -1)]
cnt = [sum(1 for t, _ in subs if wk <= t < wk + dt.timedelta(days=7)) for wk in weeks]
assert sum(cnt) == len(subs), (sum(cnt), len(subs))
best_t, best_v, cur = [], [], 0.0
for t, v in subs:
    if v > cur: cur = v; best_t.append(t); best_v.append(v)
best_t.append(DEADLINE); best_v.append(cur)
last14 = sum(1 for t, _ in subs if t >= DEADLINE - dt.timedelta(days=14))
stats = {"n": len(subs), "first_date": subs[0][0].date().isoformat(), "first_public_dice": subs[0][1], "best_public_dice": cur,
         "best_reached": best_t[-2].date().isoformat(), "weekly": cnt, "weekly_sum": sum(cnt), "last_14_days": last14, "last_14_days_share": last14 / len(subs),
         "max_week": max(cnt)}
json.dump(stats, open(os.path.join(HERE, "..", "analysis", "submission_timeline_stats.json"), "w"), indent=1)

fig = plt.figure(figsize=(3.4, 3.5))
axA = fig.add_axes([0.16, 0.60, 0.80, 0.33]); axB = fig.add_axes([0.16, 0.11, 0.80, 0.33])
axA.bar(weeks, cnt, width=6.2, align="edge", color=LIGHT, edgecolor="none", zorder=2)
axA.set_ylabel("Submissions per week", color=INK, labelpad=3); axA.set_ylim(0, max(cnt) * 1.18); axA.set_yticks([0, 200, 400])
axB.scatter([t for t, _ in subs], [v for _, v in subs], s=3, color=LIGHT, alpha=0.6, linewidths=0, zorder=2)
axB.step(best_t, best_v, where="post", color=DEEP, linewidth=1.3, zorder=3)
axB.set_ylabel("Dice on the public half", color=INK, labelpad=3); axB.set_ylim(0, 0.45); axB.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
axB.text(best_t[-1], best_v[-1] + 0.012, f"{best_v[-1]:.3f} ", fontsize=6, color=DEEP, ha="right", va="bottom")
axB.text(best_t[0], best_v[0] - 0.018, f" {best_v[0]:.2f}", fontsize=6, color=DEEP, ha="left", va="top")
xlim = (dt.datetime(2026, 6, 12), dt.datetime(2026, 9, 19))
for a in (axA, axB):
    a.set_xlim(*xlim); a.xaxis.set_major_locator(mdates.MonthLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    a.tick_params(colors=INK2, labelcolor=INK); a.yaxis.grid(True, color=GRID, linewidth=0.5); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.axvline(DEADLINE, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    a.axvline(OPEN, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
axA.text(OPEN, max(cnt) * 1.14, " leaderboard opens", fontsize=5.8, color=INK2, ha="left", va="top")
axA.text(DEADLINE, max(cnt) * 1.14, "deadline ", fontsize=5.8, color=INK2, ha="right", va="top")
first_t = subs[0][0]
axA.annotate("first submission\n" + first_t.strftime("%-d %B"), xy=(first_t, 2), xytext=(first_t, max(cnt) * 0.42),
             ha="center", va="bottom", fontsize=5.8, color=INK2,
             arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=0.6, mutation_scale=6, shrinkA=1, shrinkB=1))
for a, letter in ((axA, "A"), (axB, "B")):
    fig.text(0.01, a.get_position().y1 + 0.005, letter, fontsize=9, color=INK, ha="left", va="bottom")
for ext in ("pdf", "svg"): fig.savefig(os.path.join(HERE, f"fig_timeline.{ext}"))
fig.savefig(os.path.join(HERE, "fig_timeline_preview.png"), dpi=220)
print(json.dumps(stats))
