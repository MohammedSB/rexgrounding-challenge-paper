"""Controlled prompt experiment: the same finding on the same scan, original prompt vs edited prompt.

Reads the official per-finding scores of a team's per-variant runs (prompt_variants/returns/<team>/scores/<variant>.json,
written by rexrank_eval_fast.py) and prompt_variants/manifest.csv, and compares each variant with the same team's
original-prompt run on the findings that the variant actually edited. For swap_left_right and wrong_lobe it also reads
the predicted masks to measure whether the mask moved: overlap with the original-prompt mask, and the share of
predicted voxels on the side of the body that holds the reference finding.

usage: python3 paper/analysis/prompt_experiment.py AIMHI
writes prompt_variants/returns/<team>/prompt_experiment.json
"""
import sys, os, csv, json
import multiprocessing as mp
import numpy as np, nibabel as nib
from scipy.stats import wilcoxon

TEAM = sys.argv[1]
R = f"prompt_variants/returns/{TEAM}"
GT = sys.argv[2] if len(sys.argv) > 2 else None
VARIANTS = ["no_size", "no_lobe_segment", "no_laterality", "finding_only", "swap_left_right", "wrong_lobe", "wrong_size",
            "left_only", "right_only", "bilateral_split_union"]
B, SEED = 1000, 20260927

def scores(v):
    d = json.load(open(f"{R}/scores/{v}.json"))
    return {(c["file"], k.removeprefix("finding_")): (f["global_dice"], bool(f["global_hit"])) for c in d["cases"] for k, f in c["findings"].items()}

man = list(csv.DictReader(open("prompt_variants/manifest.csv")))
edited = {v: [(m["scan"], m["finding"]) for m in man if m["variant"] == v and m["edited"] == "1"] for v in VARIANTS}
# bilateral_split_union differs from original only where both left_only and right_only were edited
edited["bilateral_split_union"] = sorted(set(edited["left_only"]) & set(edited["right_only"]))
cat = {(m["scan"], m["finding"]): m["category"] for m in man if m["variant"] == "original"}

orig = scores("original")
rng = np.random.default_rng(SEED)

def paired(v):
    s = scores(v); keys = edited[v]
    o = np.array([orig[k][0] for k in keys]); e = np.array([s[k][0] for k in keys])
    oh = np.array([orig[k][1] for k in keys]); eh = np.array([s[k][1] for k in keys])
    # bootstrap over scans, keeping all edited findings of a scan together
    scans = sorted({k[0] for k in keys}); idx = {sc: [i for i, k in enumerate(keys) if k[0] == sc] for sc in scans}
    boot = []
    for _ in range(B):
        ii = np.concatenate([idx[sc] for sc in rng.choice(scans, len(scans))]); boot.append((e[ii] - o[ii]).mean())
    p = wilcoxon(e, o).pvalue if np.any(e != o) else 1.0
    return {"n": len(keys), "n_scans": len(scans), "dice_original": o.mean(), "dice_variant": e.mean(),
            "delta": (e - o).mean(), "delta_ci": list(np.percentile(boot, [2.5, 97.5])), "wilcoxon_p": p,
            "hit_original": oh.mean(), "hit_variant": eh.mean(),
            "share_worse_0.1": float(np.mean(e - o < -0.1)), "share_better_0.1": float(np.mean(e - o > 0.1))}

def by_category(v):
    s = scores(v); res = {}
    for grp, test in {"focal": lambda c: c.startswith("2"), "non_focal": lambda c: c.startswith("1")}.items():
        keys = [k for k in edited[v] if test(cat[k])]
        if keys: res[grp] = {"n": len(keys), "delta": float(np.mean([s[k][0] - orig[k][0] for k in keys]))}
    return res

out = {"team": TEAM, "original_overall": {"dice_per_case": json.load(open(f"{R}/scores/original.json"))["summary"]["mean_global_dice_per_case"]},
       "variants": {v: {**paired(v), "by_category": by_category(v)} for v in VARIANTS}}

# mask movement for the location and size edits: one job per scan, each file loaded once, results cached
MOVE = ["swap_left_right", "wrong_lobe", "wrong_size", "left_only", "right_only", "finding_only"]
CACHE = f"{R}/movement_rows.jsonl"

def movement(args):
    scan, need = args                                    # need: {variant: [finding, ...]}
    img = nib.load(f"{R}/split/original/{scan}")
    # first spatial axis: a higher index is the patient's left in all 300 test volumes (header code "R"). Checked
    # empirically: the reference mask of every single-sided finding ("left ..."/"right ...") lies on the side this implies
    left_high = nib.aff2axcodes(img.affine)[0] == "R"
    ks = sorted({int(k) for ks in need.values() for k in ks})
    a = np.asanyarray(img.dataobj); A = {i: a[i] > 0 for i in ks}; del a
    g = np.asanyarray(nib.load(f"{GT}/{scan}").dataobj); G = {i: g[i] > 0 for i in ks}; del g
    res = []
    for v, vks in need.items():
        b = np.asanyarray(nib.load(f"{R}/split/{v}/{scan}").dataobj)
        for k in vks:
            i = int(k); pa, pb, gi = A[i], b[i] > 0, G[i]
            agree = 2 * (pa & pb).sum() / max(pa.sum() + pb.sum(), 1)
            side = {}
            if gi.any():
                mid = int(gi.shape[0] / 2); gx = np.nonzero(gi)[0].mean()
                same = (lambda m: m[:mid].sum() if gx < mid else m[mid:].sum())
                side = {"orig_on_ref_side": same(pa) / max(pa.sum(), 1), "var_on_ref_side": same(pb) / max(pb.sum(), 1)}
            if v in ("left_only", "right_only"):
                h = int(pa.shape[0] / 2); want_high = (v == "left_only") == left_high
                named = (lambda m: m[h:].sum() if want_high else m[:h].sum())
                side["orig_on_named_side"] = named(pa) / max(pa.sum(), 1); side["var_on_named_side"] = named(pb) / max(pb.sum(), 1)
            res.append({"variant": v, "scan": scan, "finding": k, "agree_dice": float(agree), "orig_vox": int(pa.sum()),
                        "var_vox": int(pb.sum()), **{x: float(y) for x, y in side.items()}})
        del b
    return scan, res

if GT:
    need = {}
    for v in MOVE:
        for sc, k in edited[v]: need.setdefault(sc, {}).setdefault(v, []).append(k)
    done = {}
    if os.path.exists(CACHE):
        for line in open(CACHE): d = json.loads(line); done[d["scan"]] = d["rows"]
    todo = [(sc, nd) for sc, nd in sorted(need.items()) if sc not in done]
    print(f"movement: {len(done)} scans cached, {len(todo)} to do", flush=True)
    with mp.get_context("fork").Pool(2, maxtasksperchild=10) as pool, open(CACHE, "a") as fh:
        for n, (sc, res) in enumerate(pool.imap_unordered(movement, todo), 1):
            done[sc] = res; fh.write(json.dumps({"scan": sc, "rows": res}) + "\n"); fh.flush()
            if n % 25 == 0: print(f"  {n}/{len(todo)} scans", flush=True)
    allrows = [r for rs in done.values() for r in rs]
    for v in MOVE:
        rr = [r for r in allrows if r["variant"] == v]
        ag = np.array([r["agree_dice"] for r in rr])
        m = {"n": len(rr), "agree_dice_mean": ag.mean(), "agree_dice_median": float(np.median(ag)),
             "share_agree_below_0.5": float(np.mean(ag < 0.5)), "share_empty_variant": float(np.mean([r["var_vox"] == 0 for r in rr]))}
        sided = [r for r in rr if "var_on_ref_side" in r and r["orig_vox"] > 0 and r["var_vox"] > 0]
        if sided:
            m["orig_on_ref_side_mean"] = float(np.mean([r["orig_on_ref_side"] for r in sided]))
            m["var_on_ref_side_mean"] = float(np.mean([r["var_on_ref_side"] for r in sided]))
            m["share_moved_to_other_side"] = float(np.mean([r["var_on_ref_side"] < 0.5 for r in sided])); m["n_sided"] = len(sided)
        nm = [r for r in rr if "var_on_named_side" in r and r["var_vox"] > 0]
        if nm:
            m["orig_on_named_side_mean"] = float(np.mean([r["orig_on_named_side"] for r in nm]))
            m["var_on_named_side_mean"] = float(np.mean([r["var_on_named_side"] for r in nm]))
        vr = [r["var_vox"] / r["orig_vox"] for r in rr if r["orig_vox"] > 0]
        m["volume_ratio_median"] = float(np.median(vr)) if vr else None
        out["variants"][v]["movement"] = m; out["variants"][v]["movement_rows"] = rr

json.dump(out, open(f"{R}/prompt_experiment.json", "w"), indent=1, default=float)
for v, s in out["variants"].items():
    print(f"{v:22s} n={s['n']:3d}  Dice {s['dice_original']:.3f} -> {s['dice_variant']:.3f}  delta {s['delta']:+.3f} "
          f"[{s['delta_ci'][0]:+.3f}, {s['delta_ci'][1]:+.3f}]  p={s['wilcoxon_p']:.2g}  hit {s['hit_original']:.2f} -> {s['hit_variant']:.2f}")
    if "movement" in s: print("   movement:", {k: round(x, 3) if isinstance(x, float) else x for k, x in s["movement"].items()})
