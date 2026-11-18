"""Performance of the ten best teams by patient sex and age (Appendix F).

Uses only existing per-case results (no re-evaluation): the per-case mean Dice and the per-case hit counts of each team's
ranked submission, as selected in ranking_stats.py, and patient sex and age from the CT-RATE metadata
(train_metadata.csv and validation_metadata.csv from the gated ibrahimhamamci/CT-RATE dataset, dataset/metadata/).

usage (from the project root): python3 paper/analysis/sex_age_analysis.py <dir with the two CT-RATE metadata CSVs>
writes paper/analysis/sex_age_analysis.json
"""
import sys, os, csv, json, runpy, io, contextlib
import numpy as np
from scipy.stats import mannwhitneyu, kruskal

META = sys.argv[1]
meta = {}
for f in ("train_metadata.csv", "validation_metadata.csv"):
    for r in csv.DictReader(open(os.path.join(META, f))):
        a = r["PatientAge"].strip()                        # DICOM age string, e.g. 036Y (years) or 020M (months)
        meta[r["VolumeName"]] = (r["PatientSex"].strip(), int(a[:-1]) / {"Y": 1, "M": 12, "W": 52, "D": 365}[a[-1]] if a else np.nan)

with contextlib.redirect_stdout(io.StringIO()):          # reuse the exact per-case data behind the ranking
    g = runpy.run_path("paper/analysis/ranking_stats.py")
teams, X, CNT, cases = g["teams"][:10], g["X"][:10], g["CNT"][:10], g["case_order"]
missing = [c for c in cases if c not in meta]
assert not missing, missing[:5]
sex = np.array([meta[c][0] for c in cases]); age = np.array([meta[c][1] for c in cases])
AGE = [("<40", age < 40), ("40-59", (age >= 40) & (age < 60)), (">=60", age >= 60)]

def summ(mask):
    d = X[:, mask].mean(1)                                 # per-team mean Dice over these cases
    c = CNT[:, mask].sum(1); h = c[:, 0] / c[:, 1]         # per-team hit rate over these findings
    return {"n_scans": int(mask.sum()), "n_findings": int(CNT[0, mask, 1].sum()),
            "dice_per_team": d.round(4).tolist(), "dice_mean": float(d.mean()), "dice_range": [float(d.min()), float(d.max())],
            "hit_per_team": h.round(4).tolist(), "hit_mean": float(h.mean()), "hit_range": [float(h.min()), float(h.max())]}

out = {"teams": teams, "sex": {s: summ(sex == s) for s in ("F", "M")}, "age": {a: summ(m) for a, m in AGE}}
# per-team tests on per-case Dice: sex (Mann-Whitney U, two-sided) and age group (Kruskal-Wallis)
out["p_sex"] = [float(mannwhitneyu(X[i, sex == "F"], X[i, sex == "M"]).pvalue) for i in range(10)]
out["p_age"] = [float(kruskal(*[X[i, m] for _, m in AGE]).pvalue) for i in range(10)]
out["teams_F_gt_M"] = int(sum(X[i, sex == "F"].mean() > X[i, sex == "M"].mean() for i in range(10)))
json.dump(out, open("paper/analysis/sex_age_analysis.json", "w"), indent=1)

print(f"test scans {len(cases)}: F {int((sex=='F').sum())} ({(sex=='F').mean():.0%}), M {int((sex=='M').sum())}; "
      f"age median {np.median(age):.0f}")
for k in ("sex", "age"):
    for grp, v in out[k].items():
        print(f"{k:3s} {grp:6s} scans {v['n_scans']:3d} findings {v['n_findings']:3d}  Dice {v['dice_mean']:.3f} "
              f"[{v['dice_range'][0]:.3f}-{v['dice_range'][1]:.3f}]  hit {v['hit_mean']:.3f} [{v['hit_range'][0]:.3f}-{v['hit_range'][1]:.3f}]")
print("teams with F > M Dice:", out["teams_F_gt_M"], "/ 10")
print("p sex (per team):", [round(p, 3) for p in out["p_sex"]])
print("p age (per team):", [round(p, 3) for p in out["p_age"]])
