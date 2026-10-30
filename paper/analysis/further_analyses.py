"""Further analyses (Section 3.7): detection vs segmentation, finding size, language cues.
Uses only existing per-finding results (submissions/*/results.json) and the test JSON. Run from the project root."""
import json, csv, glob, re, collections
import numpy as np

merged = list(csv.DictReader(open("final_rankings_merged.csv")))
test = json.load(open("paper/figures/data/ReXGroundingCT_MICCAI2026_test.json"))["test"]
vox = json.load(open("paper/figures/data/test_gt_voxels.json")); sp = json.load(open("paper/figures/data/test_spacing.json"))
summ = {}
for p in glob.glob("submissions/**/results.json", recursive=True) + glob.glob("full_results/**/*.json", recursive=True):
    try: summ[p] = json.load(open(p))["summary"]
    except Exception: pass
def load(m):
    c = [p for p, v in summ.items() if abs(v["mean_global_dice_per_case"] - float(m["dice_full_test"])) < 5e-6
         and abs(v["hit_rate"] - float(m["hit_rate"])) < 5e-6 and abs(v["overall_instance_f1"] - float(m["instance_f1"])) < 5e-6]
    assert c, m["team"]; return json.load(open(c[0]))

# per-finding table: one row per (scan, finding) with sentence, category, volume
rows = []
for s in test:
    v, p = vox[s["name"]], sp[s["name"]]
    for i, (k, sent) in enumerate(sorted(s["findings"].items(), key=lambda kv: int(kv[0]))):
        rows.append({"scan": s["name"], "idx": int(k), "sent": sent, "cat": s["categories"][k], "ml": v[i] * p[0] * p[0] * p[1] / 1000})
N = len(rows); assert N == 582
# language cues
LAT = re.compile(r"\b(left|right)\b", re.I); BIL = re.compile(r"\b(bilateral|both|bibasilar|bibasal)\b", re.I)
LOBE = re.compile(r"\b(upper|middle|lower) lobe|\blingula|\bsegment\b|\bsegments\b|\bapic|\bbasal|\bbasilar|\blobes\b", re.I)
SIZE = re.compile(r"\d+(\.\d+)?\s*(mm|cm)\b", re.I)
MULTI = re.compile(r"\b(multiple|several|numerous|scattered|a few|few|many|nodules|opacities|areas|foci|lesions|densities|consolidations|bands)\b", re.I)
for r in rows:
    r["lat"] = bool(LAT.search(r["sent"])) or bool(BIL.search(r["sent"]))
    r["bil"] = bool(BIL.search(r["sent"])); r["lobe"] = bool(LOBE.search(r["sent"]))
    r["size"] = bool(SIZE.search(r["sent"])); r["multi"] = bool(MULTI.search(r["sent"])); r["words"] = len(r["sent"].split())
    r["focal"] = r["cat"].startswith("2")

K = 10
teams = merged[:K]; per = []     # per[t] = list aligned with rows: (dice, hit, n_gt_instances, mean_matched_dice)
for m in teams:
    d = load(m); by = {c["file"]: c["findings"] for c in d["cases"]}
    per.append([(by[r["scan"]][f"finding_{r['idx']}"]["global_dice"], by[r["scan"]][f"finding_{r['idx']}"]["global_hit"],
                 by[r["scan"]][f"finding_{r['idx']}"]["num_gt_instances"], by[r["scan"]][f"finding_{r['idx']}"]["mean_matched_dice"]) for r in rows])
D = np.array([[x[0] for x in t] for t in per]); H = np.array([[x[1] for x in t] for t in per], bool)
NI = np.array([x[2] for x in per[0]])
for r, n in zip(rows, NI): r["ninst"] = int(n)
out = {"n_findings": N, "teams": [m["team"] for m in teams]}

# (i) detection vs segmentation, all 48 teams: Dice per finding, hit rate, mean Dice among hits
det = []
for m in merged:
    d = load(m); dd = [f["global_dice"] for c in d["cases"] for f in c["findings"].values()]; hh = [f["global_hit"] for c in d["cases"] for f in c["findings"].values()]
    dd, hh = np.array(dd), np.array(hh, bool)
    det.append({"team": m["team"], "rank": int(m["rank"]), "dice_case": float(m["dice_full_test"]), "dice_finding": float(dd.mean()),
                "hit_rate": float(hh.mean()), "dice_given_hit": float(dd[hh].mean()), "dice_given_miss": float(dd[~hh].mean())})
out["detection"] = det
print("rank team            dice/case dice/finding hit  dice|hit")
for x in det[:12]: print(f"{x['rank']:2d} {x['team'][:14]:14s} {x['dice_case']:.3f} {x['dice_finding']:.3f} {x['hit_rate']:.3f} {x['dice_given_hit']:.3f}")
from scipy.stats import spearmanr
print("Spearman hit vs dice (48):", spearmanr([x["hit_rate"] for x in det], [x["dice_case"] for x in det]).statistic)
print("Spearman dice|hit vs dice (48):", spearmanr([x["dice_given_hit"] for x in det], [x["dice_case"] for x in det]).statistic)

# (ii) size bins, top 10 (pooled) and per-team
bins = [(0, 0.1), (0.1, 1), (1, 10), (10, 100), (100, 1e9)]; lab = ["<0.1", "0.1-1", "1-10", "10-100", ">100"]
ml = np.array([r["ml"] for r in rows]); foc = np.array([r["focal"] for r in rows])
size = []
for (lo, hi), l in zip(bins, lab):
    m_ = (ml >= lo) & (ml < hi)
    size.append({"bin": l, "n": int(m_.sum()), "dice_mean_top10": float(D[:, m_].mean()), "hit_mean_top10": float(H[:, m_].mean()),
                 "dice_per_team": D[:, m_].mean(1).tolist(), "hit_per_team": H[:, m_].mean(1).tolist(),
                 "dice_top10_focal": float(D[:, m_ & foc].mean()) if (m_ & foc).any() else None, "n_focal": int((m_ & foc).sum())})
    print(f"size {l:7s} n={m_.sum():3d}  dice {D[:, m_].mean():.3f}  hit {H[:, m_].mean():.3f}   focal n={(m_&foc).sum()} dice {D[:, m_&foc].mean() if (m_&foc).any() else float('nan'):.3f}")
out["size"] = size

# (iii) language cues, top 10 pooled; within focal findings to limit category confounding
def grp(mask, name):
    a = {"name": name, "n": int(mask.sum()), "dice": float(D[:, mask].mean()), "hit": float(H[:, mask].mean())}
    print(f"{name:40s} n={a['n']:3d} dice {a['dice']:.3f} hit {a['hit']:.3f}"); return a
lang = []
flags = {k: np.array([r[k] for r in rows]) for k in ("lat", "bil", "lobe", "size", "multi")}
ninst = np.array([r["ninst"] for r in rows]); words = np.array([r["words"] for r in rows])
print("--- all findings")
for k, nm in (("lat", "laterality stated"), ("lobe", "lobe/segment stated"), ("size", "size stated")):
    lang.append(grp(flags[k], nm)); lang.append(grp(~flags[k], "no " + nm))
print("--- focal only")
for k, nm in (("lat", "laterality stated"), ("lobe", "lobe/segment stated"), ("size", "size stated"), ("bil", "bilateral/both")):
    lang.append(grp(flags[k] & foc, nm + " (focal)")); lang.append(grp(~flags[k] & foc, "no " + nm + " (focal)"))
print("--- instances (reference), focal")
for lo, hi, nm in ((1, 1, "1 instance"), (2, 3, "2-3 instances"), (4, 10**6, ">=4 instances")):
    lang.append(grp((ninst >= lo) & (ninst <= hi) & foc, nm + " (focal)"))
print("--- sentence length")
for lo, hi, nm in ((1, 7, "<=7 words"), (8, 12, "8-12 words"), (13, 100, ">=13 words")):
    lang.append(grp((words >= lo) & (words <= hi), nm))
print("--- plural/multi phrasing, single reference instance vs multi")
lang.append(grp(flags["multi"] & foc, "multi phrasing (focal)")); lang.append(grp(~flags["multi"] & foc, "no multi phrasing (focal)"))
# instance-level: how often all instances found when >=2 instances
out["language"] = lang
# nodules: size stated vs not, hit/dice
nod = np.array([r["cat"] == "2d" for r in rows])
print("--- nodules only")
grp(flags["size"] & nod, "nodules with size"); grp(~flags["size"] & nod, "nodules without size")
bins_all = bins
# extra summaries used in the text and figure
out["focal_word_bins"] = [{"bin": nm, "n": int(((words >= lo) & (words <= hi) & foc).sum()), "dice": float(D[:, (words >= lo) & (words <= hi) & foc].mean()),
                           "hit": float(H[:, (words >= lo) & (words <= hi) & foc].mean())} for lo, hi, nm in ((1, 7, "<=7"), (8, 12, "8-12"), (13, 100, ">=13"))]
out["nodules_by_size"] = [{"bin": l, "n": int(((ml >= lo) & (ml < hi) & nod).sum()), "dice": float(D[:, (ml >= lo) & (ml < hi) & nod].mean()),
                           "hit": float(H[:, (ml >= lo) & (ml < hi) & nod].mean())} for (lo, hi), l in zip(bins, lab) if ((ml >= lo) & (ml < hi) & nod).any()]
out["size_bin_categories"] = {l: collections.Counter(r["cat"] for r, m_ in zip(rows, (ml >= lo) & (ml < hi)) if m_).most_common(4) for (lo, hi), l in zip(bins, lab)}
def pair(maskA, maskB, name):
    a, b = D[:, maskA].mean(1), D[:, maskB].mean(1)
    return {"name": name, "nA": int(maskA.sum()), "nB": int(maskB.sum()), "diceA": float(a.mean()), "diceB": float(b.mean()),
            "hitA": float(H[:, maskA].mean()), "hitB": float(H[:, maskB].mean()), "teams_A_gt_B": int((a > b).sum())}
out["cue_pairs_focal"] = [pair(flags["size"] & foc, ~flags["size"] & foc, "Size stated"),
                          pair(flags["lobe"] & foc, ~flags["lobe"] & foc, "Lobe or segment stated"),
                          pair((words >= 13) & foc, (words <= 7) & foc, "13+ words vs 7 or fewer"),
                          pair(flags["bil"] & foc, ~flags["bil"] & foc, "Bilateral or both lungs"),
                          pair((ninst >= 4) & foc, (ninst == 1) & foc, "4+ reference instances vs 1")]
out["max_hit_rate_team"] = max(det, key=lambda x: x["hit_rate"])["team"]
json.dump(out, open("paper/analysis/further_analyses.json", "w"), indent=1)
print("written")
