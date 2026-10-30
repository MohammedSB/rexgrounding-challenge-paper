"""Figure 1 data panels (c-f) for the ReXGrounding challenge paper.

Run from the project root:  python3 paper/figures/fig1_panels.py
Writes vector PDFs (for assembly) and a PNG preview into paper/figures/.
Panels a (task illustration) and b (workflow schematic) are drawn separately.
"""
import csv, json, os, datetime, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import shapefile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "paper", "figures")

# Palette: reference dataviz palette (validated), light mode, print figure.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
LAND = "#efeeea"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none",
})

def title(ax, letter, text):
    ax.set_title(f"{letter}   {text}", loc="left", fontweight="bold", color=INK, pad=6)

# ---------------------------------------------------------------- data
results = json.load(open(os.path.join(ROOT, "final_results_MICCAI2026.json")))["results"]
merged = list(csv.DictReader(open(os.path.join(ROOT, "final_rankings_merged.csv"))))

# c: teams per country (merged competitors; joint CN/US team counted in both; 3 unplaced teams excluded)
iso = {"China": "CN", "South Korea": "KR", "United States": "US", "United Kingdom": "GB", "Germany": "DE",
       "Netherlands": "NL", "India": "IN", "Japan": "JP", "Bangladesh": "BD", "France": "FR", "Qatar": "QA",
       "Austria": "AT", "Brazil": "BR", "United Arab Emirates": "AE", "Hong Kong SAR": "HK", "Chile": "CL"}
country_of = {r["team"]: r["country"] for r in csv.DictReader(open(os.path.join(ROOT, "participant_countries.csv")))}
per = collections.Counter()
for r in merged:
    c = country_of.get(r["team"], "Unknown")
    for part in c.split(" / "):
        if part in iso:
            per[iso[part]] += 1
per["CN"] += per.pop("HK", 0)  # Hong Kong is not a separate polygon at 1:110m

# d: test-set finding categories
names = {"1a": "Bronchial wall thickening", "1b": "Bronchiectasis", "1c": "Emphysema", "1d": "Septal thickening",
         "1e": "Micronodules", "1f": "Other non-focal", "2a": "Linear opacities", "2b": "Atelectasis / consolidation",
         "2c": "Ground-glass opacity", "2d": "Nodules / masses", "2e": "Pleural effusion / thickening",
         "2g": "Pneumothorax", "2h": "Other focal"}
cats = {k: v["n"] for k, v in results[0]["per_category_full_test_set"].items()}

# e: submission upload times (scored runs up to the deadline)
deadline = datetime.datetime(2026, 9, 15, 3, 59)  # 14 Sep 11:59 PM ET
times = []
sub = os.path.join(ROOT, "submissions")
for em in os.listdir(sub):
    if not os.path.isdir(os.path.join(sub, em)):
        continue
    for d in os.listdir(os.path.join(sub, em)):
        z = os.path.join(sub, em, d, "submission.zip")
        if os.path.exists(z) and os.path.exists(os.path.join(sub, em, d, "results.json")):
            t = datetime.datetime.utcfromtimestamp(os.path.getmtime(z))
            if t <= deadline:
                times.append(t)

# f: final Dice per competitor
dice = [float(r["dice_full_test"]) for r in merged]

# ---------------------------------------------------------------- panels
def panel_map(ax):
    sf = shapefile.Reader(os.path.join(OUT, "data", "ne_110m_admin_0_countries.shp"))
    fields = [f[0] for f in sf.fields[1:]]
    key = "ISO_A2_EH" if "ISO_A2_EH" in fields else "ISO_A2"
    bins = [(1, 1, SEQ[2], "1"), (2, 3, SEQ[3], "2–3"), (4, 5, SEQ[4], "4–5"), (6, 99, SEQ[6], "10–12")]
    def color(n):
        for lo, hi, c, _ in bins:
            if lo <= n <= hi:
                return c
    small = {"QA", "AE", "BD", "NL", "AT"}  # too small to see at this scale: add a marker
    for sr in sf.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        if rec.get("CONTINENT") == "Antarctica":
            continue
        n = per.get(rec[key], 0)
        pts, parts = sr.shape.points, list(sr.shape.parts) + [len(sr.shape.points)]
        for i in range(len(parts) - 1):
            ax.add_patch(Polygon(pts[parts[i]:parts[i + 1]], closed=True,
                                 facecolor=color(n) if n else LAND, edgecolor="white", linewidth=0.25))
        if rec[key] in small and n:
            ax.scatter(rec["LABEL_X"], rec["LABEL_Y"], s=14, color=color(n), edgecolors="white",
                       linewidths=0.6, zorder=5)
    ax.set_xlim(-170, 180); ax.set_ylim(-58, 84); ax.set_aspect("equal"); ax.axis("off")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="none", label=lab) for _, _, c, lab in bins]
    ax.legend(handles=handles, title="Teams per country", loc="lower left", ncol=4, frameon=False,
              fontsize=6, title_fontsize=6, handlelength=1.2, handleheight=0.8, columnspacing=0.9,
              handletextpad=0.4, borderaxespad=0, bbox_to_anchor=(0.0, -0.08), alignment="left")
    title(ax, "c", f"{len(merged)} teams from 16 countries and regions")

def panel_categories(ax):
    order = [k for k in sorted(cats, key=lambda k: cats[k]) if k.startswith("1")] + \
            [k for k in sorted(cats, key=lambda k: cats[k]) if k.startswith("2")]
    ys = list(range(len(order)))
    for y, k in zip(ys, order):
        ax.barh(y, cats[k], height=0.7, color=ORANGE if k.startswith("1") else BLUE, edgecolor="white", linewidth=0.8)
        ax.text(cats[k] + 3, y, str(cats[k]), va="center", fontsize=6, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([names[k] for k in order], fontsize=6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Findings in test set")
    ax.xaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    nf = sum(v for k, v in cats.items() if k.startswith("1")); fo = sum(cats.values()) - nf
    for yy, lab, c in [(0.30, f"Focal (n={fo})", BLUE), (0.22, f"Non-focal (n={nf})", ORANGE)]:
        ax.add_patch(plt.Rectangle((0.60, yy), 0.025, 0.035, transform=ax.transAxes, color=c))
        ax.text(0.64, yy + 0.017, lab, transform=ax.transAxes, va="center", color=INK, fontsize=6.5)
    title(ax, "d", f"{sum(cats.values())} test findings in {len(cats)} categories")

def panel_timeline(ax):
    start = datetime.datetime(2026, 6, 29)  # Monday of the first week with uploads
    weeks = collections.Counter((t - start).days // 7 for t in times)
    xs = sorted(weeks)
    ax.bar(xs, [weeks[x] for x in xs], width=0.8, color=BLUE, edgecolor="white", linewidth=0.8)
    ticks = [x for x in xs if x % 2 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([(start + datetime.timedelta(weeks=x)).strftime("%d %b") for x in ticks], rotation=0)
    ax.set_ylabel("Submissions per week")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.text(max(xs), max(weeks.values()) + 8, "14 Sep\n(deadline)", ha="center", va="bottom", fontsize=6, color=INK2)
    title(ax, "e", "Submissions over time")

def panel_dice(ax):
    xs = list(range(1, len(dice) + 1))
    win = 4
    ax.vlines(xs, 0, dice, color=GRID, linewidth=0.8)
    ax.scatter(xs[win:], dice[win:], s=12, color=MUTED, zorder=3, edgecolors="white", linewidths=0.6)
    ax.scatter(xs[:win], dice[:win], s=16, color=BLUE, zorder=4, edgecolors="white", linewidths=0.6)
    ax.text(win + 1.5, dice[0], "Top 4 (winners)", fontsize=6, color=INK, va="center")
    ax.set_xlim(0, len(dice) + 1); ax.set_ylim(0, 0.42)
    ax.set_xlabel("Final rank"); ax.set_ylabel("Dice (300-scan test set)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    title(ax, "f", "Final Dice per team")

# ---------------------------------------------------------------- render
def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)

for name, fn, size in [("fig1c_map", panel_map, (3.7, 2.0)), ("fig1d_categories", panel_categories, (3.5, 2.4)),
                       ("fig1e_timeline", panel_timeline, (3.5, 2.0)), ("fig1f_dice", panel_dice, (3.5, 2.0))]:
    fig, ax = plt.subplots(figsize=size); fn(ax); save(fig, name)

# combined preview of c-f (double-column width, 7.5 in)
fig = plt.figure(figsize=(7.5, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.55, wspace=0.45)
panel_map(fig.add_subplot(gs[0, 0])); panel_categories(fig.add_subplot(gs[0, 1]))
panel_timeline(fig.add_subplot(gs[1, 0])); panel_dice(fig.add_subplot(gs[1, 1]))
fig.savefig(os.path.join(OUT, "fig1_cdef_preview.png"), dpi=220, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_cdef.pdf"), bbox_inches="tight")
print("teams per country:", dict(per), "| submissions:", len(times), "| teams:", len(dice))
