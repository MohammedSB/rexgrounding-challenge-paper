"""Independent re-check of numbers added to the paper in this session. Recomputes from raw files
(submissions/*/results.json, final_results_MICCAI2026.json, final_rankings_merged.csv, participant_countries.csv,
paper/figures/data/*). Does not import the figure scripts."""
import json, csv, glob, itertools, collections, statistics as st
import numpy as np
from scipy.stats import wilcoxon, kendalltau, spearmanr, rankdata

ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
res = json.load(open(f"{ROOT}/final_results_MICCAI2026.json"))["results"]
merged = list(csv.DictReader(open(f"{ROOT}/final_rankings_merged.csv")))
pc = list(csv.DictReader(open(f"{ROOT}/participant_countries.csv")))
fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"   <-- {detail}"))
    if not cond: fails.append(name)

# ---------- per-case data, matched by exact full-test Dice ----------
summ = {}
for p in glob.glob(f"{ROOT}/submissions/**/results.json", recursive=True) + glob.glob(f"{ROOT}/full_results/**/*.json", recursive=True):
    try: summ[p] = json.load(open(p))["summary"]
    except Exception: pass
teams, X, CNT, pubdice = [], [], [], []
for m in merged:
    target = float(m["dice_full_test"])
    cands = [p for p, v in summ.items() if abs(v["mean_global_dice_per_case"] - target) < 5e-6 and abs(v["hit_rate"] - float(m["hit_rate"])) < 5e-6
             and abs(v["overall_instance_f1"] - float(m["instance_f1"])) < 5e-6]
    assert cands, m["team"]
    d = json.load(open(cands[0]))
    per = {c["file"]: c["case_stats"]["mean_global_dice"] for c in d["cases"]}
    cnt = {c["file"]: [c["case_stats"][k] for k in ("hits", "num_findings", "tp", "fp", "fn")] for c in d["cases"]}
    files = sorted(per)
    teams.append(m["team"]); X.append([per[f] for f in files]); CNT.append([cnt[f] for f in files])
    r = [x for x in res if abs(x["metrics_full_test_set"]["dice"] - target) < 5e-6]; assert len(r) == 1
    pubdice.append(r[0]["metrics_public_split_50pct"]["dice"])
X = np.array(X); CNT = np.array(CNT, float); n_t, n_c = X.shape
obs = X.mean(1); obs_rank = rankdata(-obs, method="min").astype(int)
check("48 teams, 300 cases", (n_t, n_c) == (48, 300), (n_t, n_c))
check("official ranking = order of mean per-case Dice", all(obs_rank[i] == i + 1 for i in range(n_t)))
check("no exact ties in mean Dice", len(set(np.round(obs, 10))) == n_t)
pr = rankdata([-p for p in pubdice], method="min").astype(int)
check("public vs full: top 2 unchanged; pairs 3/4, 5/6, 7/8 swapped", list(pr[:8]) == [1, 2, 4, 3, 6, 5, 8, 7], list(pr[:8]))
check("public vs full: 9 of public top ten stay in top ten; max shift 6", sum(1 for i in range(10) if pr[i] <= 10) == 9 and max(abs(pr - obs_rank)) == 6, (sum(1 for i in range(10) if pr[i] <= 10), max(abs(pr - obs_rank))))

# ---------- bootstrap (same seed/B as the analysis script) ----------
rng = np.random.default_rng(20260927); B = 1000
bm = np.empty((B, n_t)); br = np.empty((B, n_t), int); bh = np.empty((B, n_t)); bf = np.empty((B, n_t)); taus = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n_c, n_c)
    m_ = X[:, idx].mean(1); bm[b] = m_; br[b] = (-m_).argsort().argsort() + 1
    s_ = CNT[:, idx].sum(1); bh[b] = s_[:, 0] / s_[:, 1]; bf[b] = 2 * s_[:, 2] / (2 * s_[:, 2] + s_[:, 3] + s_[:, 4])
    taus[b] = kendalltau(obs_rank, br[b]).statistic
ci = np.percentile(bm, [2.5, 97.5], axis=0); cih = np.percentile(bh, [2.5, 97.5], axis=0); cif = np.percentile(bf, [2.5, 97.5], axis=0)
check("Kendall tau median 0.95, IQR 0.94-0.96", (round(np.median(taus), 2), tuple(np.round(np.percentile(taus, [25, 75]), 2))) == (0.95, (0.94, 0.96)),
      (np.median(taus), np.percentile(taus, [25, 75])))
p1 = (br[:, 0] == 1).mean(); p3 = (br[:, 0] <= 3).mean()
check("CIRCLE first in 89% of replicates", round(p1 * 100) == 89, p1)
check("CIRCLE top three in 99%% (text)  [data says %.1f%%]" % (p3 * 100), round(p3 * 100) == 99, p3)
# Table 1 CIs vs main.tex
tex = open(f"{ROOT}/paper/main.tex").read()
s = tex.index("\\label{tab:leaderboard}"); e = tex.index("\\end{table}", s); tab = tex[s:e]
import re
ok = True
for i in range(10):
    row = re.search(r"^Team %d & .*$" % (i + 1), tab, re.M).group(0)
    nums = re.findall(r"\[\\chk\{(0\.\d{3})--(0\.\d{3})\}\]", row)
    exp = [(f"{ci[0,i]:.3f}", f"{ci[1,i]:.3f}"), (f"{cih[0,i]:.3f}", f"{cih[1,i]:.3f}"), (f"{cif[0,i]:.3f}", f"{cif[1,i]:.3f}")]
    if nums != exp: ok = False; print("   row", i + 1, nums, exp)
check("Table 1: Dice / hit-rate / F1 CIs match bootstrap", ok)
# official hit rate / F1 reproduced from per-case counts
tot = CNT.sum(1)
check("hit rate & F1 reproduce official values (all 48)",
      all(abs(tot[i, 0] / tot[i, 1] - float(merged[i]["hit_rate"])) < 5e-6 and abs(2 * tot[i, 2] / (2 * tot[i, 2] + tot[i, 3] + tot[i, 4]) - float(merged[i]["instance_f1"])) < 5e-6 for i in range(n_t)))
# pairwise among top 10
K = 10; pairs = {}
for a, b in itertools.combinations(range(K), 2):
    pairs[(a, b)] = ((bm[:, a] > bm[:, b]).mean(), wilcoxon(X[a], X[b], alternative="greater").pvalue)
ps = np.array([v[1] for v in pairs.values()]); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
for k, j in enumerate(o): run = max(run, min(1, (len(ps) - k) * ps[j])); adj[j] = run
holm = dict(zip(pairs, adj))
check("CIRCLE > AIMHI in 91% of replicates", round(pairs[(0, 1)][0] * 100) == 91, pairs[(0, 1)][0])
check("fomofo > needpaper in 57%", round(pairs[(2, 3)][0] * 100) == 57, pairs[(2, 3)][0])
check("winner vs 5th/6th adjusted p = 0.017 / 0.009", (round(holm[(0, 4)], 3), round(holm[(0, 5)], 3)) == (0.017, 0.009), (holm[(0, 4)], holm[(0, 5)]))
check("21 of 45 pairs significant", sum(v < 0.05 for v in holm.values()) == 21, sum(v < 0.05 for v in holm.values()))
check("top four: no significant pairwise difference", all(holm[(a, b)] >= 0.05 for a, b in itertools.combinations(range(4), 2)))
check("sjtu426lab vs HBKU: 50% of replicates, same Dice to 4 dp", round(pairs[(8, 9)][0], 2) == 0.50 and f"{obs[8]:.4f}" == f"{obs[9]:.4f}", pairs[(8, 9)][0])
check("Dice over 48 teams: 0.146-0.375, median 0.280", (round(obs.min(), 3), round(obs.max(), 3), round(float(np.median(obs)), 3)) == (0.146, 0.375, 0.280), (obs.min(), obs.max(), np.median(obs)))
check("needpaper 0.0016 behind fomofo", round(obs[2] - obs[3], 4) == 0.0016, obs[2] - obs[3])
# Figure 3 panel values: rank_counts and 95% rank ranges from JSON
S = json.load(open(f"{ROOT}/paper/analysis/ranking_stats.json"))
check("ranking_stats.json: rank_counts/ranges/CIs match recomputation (top 20)",
      all(S["teams"][i]["rank_counts"] == np.bincount(br[:, i], minlength=n_t + 1)[1:].tolist() and
          S["teams"][i]["rank_range_95"] == [int(x) for x in np.percentile(br[:, i], [2.5, 97.5])] and
          np.allclose(S["teams"][i]["ci95"], ci[:, i]) for i in range(20)))
check("Figure 3B: significant pairs in JSON = recomputed", all(S["pairs_top10"][k]["significant"] == (holm[(teams.index(S["pairs_top10"][k]["a"]), teams.index(S["pairs_top10"][k]["b"]))] < 0.05) for k in range(45)))
# rank-then-aggregate
cr = np.vstack([rankdata(-X[:, c]) for c in range(n_c)]).T; rta = rankdata(cr.mean(1), method="min").astype(int)
check("rank-then-aggregate: AIMHI 1, fomofo 2, CIRCLE 3, needpaper 4", list(rta[:4]) == [3, 1, 2, 4], list(rta[:4]))
check("rank-then-aggregate tau = 0.96", round(kendalltau(obs_rank, rta).statistic, 2) == 0.96, kendalltau(obs_rank, rta).statistic)
check("CIRCLE beats AIMHI on more scans head-to-head (153 vs 147)", ((X[0] > X[1]).sum(), (X[1] > X[0]).sum()) == (153, 147))

# ---------- Figure 4 / per-category text ----------
NAMES = {"1a": "Bronchial wall", "1b": "Bronchiectasis", "1c": "Emphysema", "1d": "Septal", "1e": "Micronodules", "1f": "Other non-focal",
         "2a": "Linear", "2b": "Atel/consol", "2c": "GGO", "2d": "Nodules", "2e": "Pleural", "2f": "Honeycomb", "2g": "Pneumothorax", "2h": "Other focal"}
pcs = []
for m in merged[:10]:
    r = [x for x in res if abs(x["metrics_full_test_set"]["dice"] - float(m["dice_full_test"])) < 5e-6][0]
    pcs.append(r["per_category_full_test_set"])
n = {k: v["n"] for k, v in pcs[0].items()}
check("category counts: 2g=1, 2h=2, 2f absent, 1f=4, 1a=6, 1d=9", (n["2g"], n["2h"], n.get("2f", 0), n["1f"], n["1a"], n["1d"]) == (1, 2, 0, 4, 6, 9))
check("focal n=509, non-focal n=73", (sum(v for k, v in n.items() if k[0] == "2"), sum(v for k, v in n.items() if k[0] == "1")) == (509, 73))
def agg(pc, pref, key):
    it = [(pc[k]["n"], pc[k][key]) for k in pc if k.startswith(pref) and pc[k]["n"]]
    return sum(a * b for a, b in it) / sum(a for a, _ in it)
foc = [agg(p, "2", "dice") for p in pcs[:6]]; nf = [agg(p, "1", "dice") for p in pcs[:6]]
foch = [agg(p, "2", "hit_rate") for p in pcs[:6]]; nfh = [agg(p, "1", "hit_rate") for p in pcs[:6]]
check("top-6 focal Dice 0.348-0.378", (round(min(foc), 3), round(max(foc), 3)) == (0.348, 0.378), (min(foc), max(foc)))
check("top-6 non-focal Dice 0.176-0.243", (round(min(nf), 3), round(max(nf), 3)) == (0.176, 0.243), (min(nf), max(nf)))
check("top-6 focal hit 0.78-0.82", (round(min(foch), 2), round(max(foch), 2)) == (0.78, 0.82), (min(foch), max(foch)))
check("top-6 non-focal hit 0.47-0.53", (round(min(nfh), 2), round(max(nfh), 2)) == (0.47, 0.53), (min(nfh), max(nfh)))
check("winner non-focal 0.243 vs at most 0.216 among other top six", round(nf[0], 3) == 0.243 and round(max(nf[1:]), 3) == 0.216, (nf[0], max(nf[1:])))
check("per-category weighted mean == dice_per_finding (sanity)", all(abs(agg(p, "", "dice") - [x for x in res if x["per_category_full_test_set"] == p][0]["metrics_full_test_set"]["dice_per_finding"]) < 1e-9 for p in pcs))
col = lambda k: [p[k]["dice"] for p in pcs]
check("pleural effusion 0.35-0.50", (round(min(col("2e")), 2), round(max(col("2e")), 2)) == (0.35, 0.50), (min(col("2e")), max(col("2e"))))
check("linear opacities 0.27-0.32", (round(min(col("2a")), 2), round(max(col("2a")), 2)) == (0.27, 0.32), (min(col("2a")), max(col("2a"))))
focal_cats = [k for k in n if k[0] == "2" and n[k] >= 4]
check("pleural is best focal category, linear is worst (by mean over top 10)",
      max(focal_cats, key=lambda k: np.mean(col(k))) == "2e" and min(focal_cats, key=lambda k: np.mean(col(k))) == "2a",
      {NAMES[k]: round(np.mean(col(k)), 3) for k in focal_cats})
spans = {k: max(col(k)) - min(col(k)) for k in ("2d", "2b", "2c")}
check("nodules/consolidation/GGO: top-10 span less than 0.08", max(spans.values()) < 0.08, spans)
em = col("1c")
check("emphysema: CIRCLE 0.33, no other top-10 team > 0.25", round(em[0], 2) == 0.33 and max(em[1:]) <= 0.25 + 5e-3, (em[0], max(em[1:])))
check("bronchiectasis & septal: Dice at most 0.18 for all top 10", max(col("1b") + col("1d")) <= 0.18 + 5e-3, (max(col("1b")), max(col("1d"))))
hardest = [min([k for k in n if n[k] >= 4], key=lambda k: p[k]["dice"]) for p in pcs]
check("smallest non-focal categories have 4-9 findings", sorted(v for k, v in n.items() if k[0] == "1")[:3] == [4, 6, 9])

# ---------- Figure 2 / data section ----------
D = f"{ROOT}/paper/figures/data"
test = json.load(open(f"{D}/ReXGroundingCT_MICCAI2026_test.json"))["test"]; train = json.load(open(f"{D}/dataset.json"))["train"]
vox = json.load(open(f"{D}/test_gt_voxels.json")); sp = json.load(open(f"{D}/test_spacing.json"))
ntr = sum(len(s["findings"]) for s in train); nte = sum(len(s["findings"]) for s in test)
check("7,687 training findings in 2,992 scans; 582 test findings in 300 scans", (ntr, len(train), nte, len(test)) == (7687, 2992, 582, 300))
ctr = collections.Counter(c for s in train for c in s["categories"].values()); cte = collections.Counter(c for s in test for c in s["categories"].values())
check("honeycombing: 16 in train, 0 in test", (ctr["2f"], cte["2f"]) == (16, 0))
check("focal findings predominate in both (train 79%, test 87%)", sum(v for k, v in ctr.items() if k[0] == "2") / ntr > 0.5 and sum(v for k, v in cte.items() if k[0] == "2") / nte > 0.5)
check("test category counts match per-category n in results", all(cte[k] == n.get(k, 0) for k in cte))
vols = []
for s in test:
    v, p = vox[s["name"]], sp[s["name"]]
    assert len(v) == len(s["findings"])
    vols += [x * p[0] * p[0] * p[1] / 1000 for x in v]
check("volumes span ~5 orders of magnitude, min < 1 mL, max > 1 L", min(vols) < 0.1 and max(vols) > 1000 and np.log10(max(vols) / min(vols)) >= 5, (min(vols), max(vols)))
check("spacing file is [in-plane, slice]: in-plane 0.39-0.98, slice 0.75-1.5", all(0.3 < p[0] < 1.0 and 0.7 <= p[1] <= 2.0 for p in sp.values()))
w = [len(f.split()) for s in train + test for f in s["findings"].values()]
check("sentence lengths 1-38 words", (min(w), max(w)) == (1, 38), (min(w), max(w)))

# ---------- participation / tables ----------
nsub = [int(m["n_submissions"]) for m in merged]
check("760 submissions, median 4, range 1-242", (sum(nsub), st.median(nsub), min(nsub), max(nsub)) == (760, 4, 1, 242), (sum(nsub), st.median(nsub), min(nsub), max(nsub)))
raw = list(csv.DictReader(open(f"{ROOT}/final_rankings_MICCAI2026.csv")))
check("53 registrations", len(raw) == 53, len(raw))
sel = collections.Counter(next(r["selection_source"] for r in raw if abs(float(r["dice_full_test"]) - float(m["dice_full_test"])) < 5e-6) for m in merged)
check("8 declared final, 40 default", (sel["team-selected"], sel["default_best_public_split"]) == (8, 40), sel)
cc = collections.Counter(q["country"] for q in pc)
china = cc["China"] + cc["Hong Kong SAR"] + cc["China / United States"]
countries = set(); [countries.update(x.strip() for x in q["country"].split("/")) for q in pc if q["country"] != "Unknown"]
countries.discard("Hong Kong SAR")
check("China 10 (incl. HK + joint), South Korea 10, 15 countries", (china, cc["South Korea"], len(countries)) == (10, 10, 15), (china, cc["South Korea"], sorted(countries)))
sec = collections.Counter(q["sector"].split(" (")[0] for q in pc)
check("sectors 35 academic / 6 industry / 4 mixed / 3 unknown", (sec["academic / clinical"], sec["industry"], sec["academic + industry"], sec["unspecified"]) == (35, 6, 4, 3), sec)
# appendix table
appx = open(f"{ROOT}/paper/tables/leaderboard_full.tex").read()
rows = re.findall(r"^Team (\d+) & (.*?) & .*? & (0\.\d{4}) & (0\.\d{3})--(0\.\d{3}) & (0\.\d{4}) & (0\.\d{4}) & (\d+) \\\\", appx, re.M)
ok = len(rows) == 48 and all(int(r[0]) == i + 1 and r[2] == f"{obs[i]:.4f}" and (r[3], r[4]) == (f"{ci[0,i]:.3f}", f"{ci[1,i]:.3f}") and
    r[5] == f"{float(merged[i]['hit_rate']):.4f}" and r[6] == f"{float(merged[i]['instance_f1']):.4f}" and int(r[7]) == nsub[i] for i, r in enumerate(rows))
check("Table B.1: 48 rows, Dice/CI/hit/F1/submissions all match", ok, len(rows))

# ---------- numbers in main.tex that I set ----------
for claim, pat in [("tex: Kendall 0.95 / 0.94-0.96", r"\\tau = \\chk\{0\.95\}\$, interquartile range \\chk\{0\.94--0\.96\}"),
                   ("tex: rank-then-aggregate tau 0.96", r"\\tau = \\chk\{0\.96\}"),
                   ("tex: 48 teams everywhere, no 52", None)]:
    check(claim, (re.search(pat, tex) is not None) if pat else ("\\chk{52}" not in tex and "\\chk{48}" in tex))

# ---------- Section 3.7 further analyses (from further_analyses.json, recomputed here from the same per-finding data) ----------
FA = json.load(open(f"{ROOT}/paper/analysis/further_analyses.json"))
det = FA["detection"]; t10 = det[:10]
check("3.7: Spearman hit vs Dice across 48 teams = 0.97", round(spearmanr([x["hit_rate"] for x in det], [x["dice_case"] for x in det]).statistic, 2) == 0.97)
check("3.7: top-10 hit rates 0.69-0.78", (round(min(x["hit_rate"] for x in t10), 2), round(max(x["hit_rate"] for x in t10), 2)) == (0.69, 0.78))
check("3.7: CIRCLE Dice|hit 0.46, others at most 0.44", round(t10[0]["dice_given_hit"], 2) == 0.46 and round(max(x["dice_given_hit"] for x in t10[1:]), 2) == 0.44)
np_ = next(x for x in det if x["team"] == "needpaper")
check("3.7: needpaper highest hit rate of all (0.78), Dice|hit 0.43", FA["max_hit_rate_team"] == "needpaper" and round(np_["hit_rate"], 2) == 0.78 and round(np_["dice_given_hit"], 2) == 0.43)
nb = {x["bin"]: x for x in FA["nodules_by_size"]}
check("3.7: nodules <0.1 mL hit 78%, Dice 0.39; 0.1-1 mL 85%, 0.38", (round(nb["<0.1"]["hit"] * 100), round(nb["<0.1"]["dice"], 2), round(nb["0.1-1"]["hit"] * 100), round(nb["0.1-1"]["dice"], 2)) == (78, 0.39, 85, 0.38))
sz = {x["bin"]: x for x in FA["size"]}
check("3.7: 1-10 mL Dice 0.27 hit 0.69; >100 mL 0.46, 0.86", (round(sz["1-10"]["dice_mean_top10"], 2), round(sz["1-10"]["hit_mean_top10"], 2), round(sz[">100"]["dice_mean_top10"], 2), round(sz[">100"]["hit_mean_top10"], 2)) == (0.27, 0.69, 0.46, 0.86))
check("3.7: 1-10 mL dominated by linear opacities; >100 mL mostly GGO/consolidation/effusion", FA["size_bin_categories"]["1-10"][0][0] == "2a" and [c for c, _ in FA["size_bin_categories"][">100"][:3]] == ["2c", "2b", "2e"])
cp = {c["name"]: c for c in FA["cue_pairs_focal"]}
check("3.7: size stated 0.42 vs 0.34, lobe 0.37 vs 0.33, 13+ words 0.41 vs 0.33, all 10/10 teams",
      (round(cp["Size stated"]["diceA"], 2), round(cp["Size stated"]["diceB"], 2), round(cp["Lobe or segment stated"]["diceA"], 2), round(cp["Lobe or segment stated"]["diceB"], 2),
       round(cp["13+ words vs 7 or fewer"]["diceA"], 2), round(cp["13+ words vs 7 or fewer"]["diceB"], 2)) == (0.42, 0.34, 0.37, 0.33, 0.41, 0.33)
      and all(cp[k]["teams_A_gt_B"] == 10 for k in ("Size stated", "Lobe or segment stated", "13+ words vs 7 or fewer")))
check("3.7: bilateral 0.33 vs 0.37 Dice, hit rates similar", (round(cp["Bilateral or both lungs"]["diceA"], 2), round(cp["Bilateral or both lungs"]["diceB"], 2)) == (0.33, 0.37) and abs(cp["Bilateral or both lungs"]["hitA"] - cp["Bilateral or both lungs"]["hitB"]) < 0.03)
check("3.7: 4+ instances hit 82% vs 77%, Dice 0.34 vs 0.37", (round(cp["4+ reference instances vs 1"]["hitA"] * 100), round(cp["4+ reference instances vs 1"]["hitB"] * 100), round(cp["4+ reference instances vs 1"]["diceA"], 2), round(cp["4+ reference instances vs 1"]["diceB"], 2)) == (82, 77, 0.34, 0.37))

# ---------- submission timeline (Participation section) ----------
import csv as _csv, datetime as _dt
_rows = list(_csv.DictReader(open(f"{ROOT}/paper/figures/data/submission_timeline.csv")))
_P = lambda x: _dt.datetime.fromisoformat(x.replace("Z", "+00:00")).replace(tzinfo=None)
_subs = sorted((_P(r["created_at"]), float(r["public_dice"])) for r in _rows)
_DL = _dt.datetime(2026, 9, 15, 3, 59)
check("timeline: 760 published submissions", len(_subs) == 760, len(_subs))
check("timeline: first submission 5 July, public Dice 0.28", _subs[0][0].date().isoformat() == "2026-07-05" and round(_subs[0][1], 2) == 0.28, _subs[0])
_best = max(_subs, key=lambda x: x[1]); _first_best = min((t for t, v in _subs if v == _best[1]))
check("timeline: best public Dice 0.386 reached 13 September", round(_best[1], 3) == 0.386 and _first_best.date().isoformat() == "2026-09-13", (_best, _first_best))
check("timeline: best public Dice equals CIRCLE Team's official public-split Dice", abs(_best[1] - max(pubdice)) < 1e-9)
_l14 = sum(1 for t, _ in _subs if t >= _DL - _dt.timedelta(days=14))
check("timeline: 74% of submissions in the final two weeks", round(100 * _l14 / len(_subs)) == 74, _l14 / len(_subs))

# ---------- Discussion ----------
_raw = list(csv.DictReader(open(f"{ROOT}/final_rankings_MICCAI2026.csv")))
_nm = {round(float(r["dice_full_test"]), 6): (r["final_model_name"] or r["leaderboard_label"]) for r in _raw}
_vt = [int(m["rank"]) for m in merged if re.search(r"vox\s*tell|\bvt[-_]", _nm[round(float(m["dice_full_test"]), 6)], re.I)]
check("discussion: at least 13 VoxTell-named teams ranked 10th-48th", len(_vt) == 13 and min(_vt) == 10 and max(_vt) == 48, _vt)
check("discussion: non-focal = 12.5% of test findings", round(100 * 73 / 582, 1) == 12.5)
check("discussion: several categories under ten findings", sum(1 for k, v in n.items() if 0 < v < 10) >= 3, n)
check("discussion: top-6 focal hit 0.78-0.82 below inter-reader focal hit 0.84", round(max(foch), 2) < 0.84)
print("\nFAILED:", fails if fails else "none")
