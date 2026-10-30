"""Build the team file (each finding -> list of distinct prompts, original first) and the internal mapping
(list position -> variant names, plus which variants were really edited) from the single-variant files.
Run from the project root after make_prompt_variants.py."""
import json, csv, os, glob
V = ["original", "no_size", "no_lobe_segment", "no_laterality", "swap_left_right", "finding_only",
     "left_only", "right_only", "wrong_lobe", "wrong_size"]
files = {v: json.load(open(f"prompt_variants/ReXGrounding_test_{v}.json"))["test"] for v in V}
edited = {(r["variant"], r["scan"], r["finding"]) for r in csv.DictReader(open("prompt_variants/manifest.csv")) if r["edited"] == "1"}
team, mapping = [], []
for i, e in enumerate(files["original"]):
    te = {k: e[k] for k in ("name", "shape", "protocol")}; te["findings"] = {}; te["categories"] = e["categories"]
    me = {"name": e["name"], "findings": {}}
    for k in e["findings"]:
        prompts, slots = [], []
        for v in V:
            s = files[v][i]["findings"][k]
            if s in prompts: slots[prompts.index(s)]["variants"].append(v)
            else: prompts.append(s); slots.append({"index": len(prompts) - 1, "prompt": s, "variants": [v]})
        te["findings"][k] = prompts
        me["findings"][k] = {"slots": slots, "edited": [v for v in V[1:] if (v, e["name"], k) in edited]}
    team.append(te); mapping.append(me)
os.makedirs("prompt_variants/for_teams", exist_ok=True)
json.dump({"test": team}, open("prompt_variants/for_teams/ReXGrounding_test_prompts.json", "w"), indent=1, ensure_ascii=False)
json.dump({"variants": V, "test": mapping}, open("prompt_variants/mapping.json", "w"), indent=1, ensure_ascii=False)
n_pr = sum(len(p) for e in team for p in e["findings"].values()); n_f = sum(len(e["findings"]) for e in team)
lens = [len(p) for e in team for p in e["findings"].values()]
print(f"{n_f} findings, {n_pr} prompts (mean {n_pr / n_f:.2f}, max {max(lens)})")
