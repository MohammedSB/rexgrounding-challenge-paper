"""Ranking stability and significance for the ReXGrounding Challenge (registered analysis plan).

Uses only existing per-case results (no re-evaluation). For each of the 52 competitors, the final
submission's per-case mean Dice (300 cases) is taken from submissions/*/*/results.json.
Outputs paper/analysis/ranking_stats.json and prints a summary.
Run from the project root: python3 paper/analysis/ranking_stats.py
"""
import json, csv, glob, itertools
import numpy as np
from scipy.stats import wilcoxon, kendalltau

rng = np.random.default_rng(20260927)
B = 1000
res = json.load(open("final_results_MICCAI2026.json"))["results"]
merged = list(csv.DictReader(open("final_rankings_merged.csv")))
allp = sorted(set(glob.glob("submissions/**/results.json", recursive=True) + glob.glob("full_results/**/*.json", recursive=True)))
summ = {}
for p in allp:
    try:
        d = json.load(open(p)); summ[p] = d["summary"]["mean_global_dice_per_case"]
    except Exception:
        pass
emails = {}
for r in res:
    key = r["team"].rstrip("2")
    raw = r.get("all_emails") or r.get("contact_email") or ""
    raw = ";".join(raw) if isinstance(raw, list) else raw
    em = set(e.strip().lower() for e in raw.replace(",", ";").split(";") if "@" in e)
    emails.setdefault(key, set()).update(em)
teams, X, CNT = [], [], []          # CNT: per-case [hits, findings, tp, fp, fn]
case_order = None
for m in merged:
    target = float(m["dice_full_test"]); key = m["team"].rstrip("2")
    em = emails.get(key, set())
    def same(p):   # same submission: Dice, hit rate and instance F1 all match the official values
        d_ = json.load(open(p))["summary"]
        return (abs(d_["mean_global_dice_per_case"] - target) < 5e-6 and abs(d_["hit_rate"] - float(m["hit_rate"])) < 5e-6
                and abs(d_["overall_instance_f1"] - float(m["instance_f1"])) < 5e-6)
    cands = [p for p, v in summ.items() if abs(v - target) < 5e-6 and same(p)]
    if not cands:
        raise SystemExit(f"no per-case results matching {m['team']} ({target})")
    best = json.load(open(cands[0]))
    per = {c["file"]: c["case_stats"]["mean_global_dice"] for c in best["cases"]}
    if case_order is None: case_order = sorted(per)
    cnt = {c["file"]: [c["case_stats"][k] for k in ("hits", "num_findings", "tp", "fp", "fn")] for c in best["cases"]}
    teams.append(m["team"]); X.append([per[f] for f in case_order]); CNT.append([cnt[f] for f in case_order])
    s_ = best["summary"]; h_, n_, tp_, fp_, fn_ = np.array(CNT[-1]).sum(0)
    assert abs(h_ / n_ - float(m["hit_rate"])) < 5e-6 and abs(2 * tp_ / (2 * tp_ + fp_ + fn_) - float(m["instance_f1"])) < 5e-6, m["team"]
X = np.array(X)                     # teams x cases
CNT = np.array(CNT, dtype=float)    # teams x cases x 5
n_t, n_c = X.shape
obs = X.mean(1)
obs_rank = (-obs).argsort().argsort() + 1
boot_hit = np.empty((B, n_t)); boot_f1 = np.empty((B, n_t))
boot_means = np.empty((B, n_t)); boot_ranks = np.empty((B, n_t), int); taus = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n_c, n_c)
    m_ = X[:, idx].mean(1); boot_means[b] = m_
    s_ = CNT[:, idx].sum(1); boot_hit[b] = s_[:, 0] / s_[:, 1]
    boot_f1[b] = 2 * s_[:, 2] / np.maximum(2 * s_[:, 2] + s_[:, 3] + s_[:, 4], 1)
    boot_ranks[b] = (-m_).argsort().argsort() + 1
    taus[b] = kendalltau(obs_rank, boot_ranks[b]).statistic
ci = np.percentile(boot_means, [2.5, 97.5], axis=0)
ci_hit = np.percentile(boot_hit, [2.5, 97.5], axis=0); ci_f1 = np.percentile(boot_f1, [2.5, 97.5], axis=0)
out = {"n_cases": int(n_c), "n_teams": int(n_t), "B": B,
       "kendall_tau_median": float(np.median(taus)), "kendall_tau_iqr": [float(x) for x in np.percentile(taus, [25, 75])],
       "teams": []}
for i, t in enumerate(teams):
    out["teams"].append({"team": t, "rank": int(obs_rank[i]), "dice": float(obs[i]),
                         "ci95": [float(ci[0, i]), float(ci[1, i])],
                         "hit_rate_ci95": [float(ci_hit[0, i]), float(ci_hit[1, i])],
                         "instance_f1_ci95": [float(ci_f1[0, i]), float(ci_f1[1, i])],
                         "median_rank": float(np.median(boot_ranks[:, i])),
                         "rank_iqr": [float(x) for x in np.percentile(boot_ranks[:, i], [25, 75])],
                         "rank_range_95": [int(x) for x in np.percentile(boot_ranks[:, i], [2.5, 97.5])],
                         "p_rank1": float((boot_ranks[:, i] == 1).mean()),
                         "rank_counts": np.bincount(boot_ranks[:, i], minlength=n_t + 1)[1:].tolist()})
# pairwise among top K: bootstrap win proportion and one-sided paired Wilcoxon (A > B), Holm-corrected
K = 10
order = list(np.argsort(obs_rank))[:K]
pairs = []
for a, b in itertools.combinations(order, 2):
    win = float((boot_means[:, a] > boot_means[:, b]).mean())
    p = float(wilcoxon(X[a], X[b], alternative="greater").pvalue)
    pairs.append({"a": teams[a], "b": teams[b], "boot_win": win, "p": p})
ps = np.array([q["p"] for q in pairs]); m = len(ps); o = np.argsort(ps); adj = np.empty(m); run = 0
for k, j in enumerate(o):
    run = max(run, min(1, (m - k) * ps[j])); adj[j] = run
for q, a_ in zip(pairs, adj): q["p_holm"] = float(a_); q["significant"] = bool(a_ < 0.05)
out["pairs_top10"] = pairs
# alternative scheme: rank-then-aggregate (rank teams on each case, average the ranks; ties share the mean rank)
from scipy.stats import rankdata
case_ranks = np.vstack([rankdata(-X[:, c]) for c in range(n_c)]).T      # teams x cases
rta = case_ranks.mean(1); rta_rank = rankdata(rta, method="min").astype(int)
out["rank_then_aggregate"] = {"kendall_tau_vs_official": float(kendalltau(obs_rank, rta_rank).statistic),
                              "teams": [{"team": teams[i], "official_rank": int(obs_rank[i]), "rta_rank": int(rta_rank[i])}
                                        for i in np.argsort(obs_rank)]}
json.dump(out, open("paper/analysis/ranking_stats.json", "w"), indent=1)
print(f"Kendall tau (bootstrap vs observed ranking): median {out['kendall_tau_median']:.3f}, IQR {out['kendall_tau_iqr']}")
for r in out["teams"][:10]:
    print(f"{r['rank']:2d} {r['team']:14s} Dice {r['dice']:.4f} CI [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  median rank {r['median_rank']:.0f} IQR {r['rank_iqr']} 95% {r['rank_range_95']}  P(rank1) {r['p_rank1']:.2f}")
print("\nAdjacent pairs:")
for q in pairs:
    ra = next(t['rank'] for t in out['teams'] if t['team'] == q['a']); rb = next(t['rank'] for t in out['teams'] if t['team'] == q['b'])
    if rb - ra == 1 or (ra == 1 and rb <= 6):
        print(f"  {q['a']:12s} > {q['b']:12s} boot win {q['boot_win']:.2f}  p {q['p']:.4f}  Holm {q['p_holm']:.4f}  {'*' if q['significant'] else ''}")
sig = sum(q["significant"] for q in pairs); print(f"\nsignificant pairs among top {K}: {sig} of {len(pairs)}")
