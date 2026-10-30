"""Split a team's prompt-experiment archive into one standard submission per variant.

A team returns, per scan, a 4-D NIfTI whose first axis has one mask per prompt (all prompts of finding "0" in list
order, then finding "1", ...), following prompt_variants/for_teams/ReXGrounding_test_prompts.json. Using
prompt_variants/mapping.json this script writes, for each variant, a folder in the ordinary challenge format
(one NIfTI per scan, first axis = findings) that can be scored with the official evaluation code. It also writes
"bilateral_split_union": for findings where both left_only and right_only were edited, the union of those two masks;
the original mask elsewhere.

usage: python3 paper/analysis/split_prompt_submission.py <team_dir_or_zip> <out_dir>
"""
import sys, os, json, zipfile, tempfile
import numpy as np, nibabel as nib

src, out = sys.argv[1], sys.argv[2]
M = json.load(open("prompt_variants/mapping.json")); V = M["variants"]
if src.endswith(".zip"):
    tmp = tempfile.mkdtemp(); zipfile.ZipFile(src).extractall(tmp); src = tmp
files = {}
for root, _, names in os.walk(src):
    for n in names:
        if n.endswith(".nii.gz"): files[n] = os.path.join(root, n)
missing = [e["name"] for e in M["test"] if e["name"] not in files]
if missing: sys.exit(f"missing {len(missing)} scans, e.g. {missing[:3]}")
OUTV = V + ["bilateral_split_union"]
for v in OUTV: os.makedirs(os.path.join(out, v), exist_ok=True)
problems = 0
for e in M["test"]:
    img = nib.load(files[e["name"]]); arr = np.asanyarray(img.dataobj)
    slots = [(k, s) for k in sorted(e["findings"], key=int) for s in e["findings"][k]["slots"]]
    if arr.ndim != 4 or arr.shape[0] != len(slots):
        print(f"{e['name']}: expected {len(slots)} masks on axis 0, got shape {arr.shape}"); problems += 1; continue
    pos = {}; i = 0
    for k, s in slots: pos[(k, s["index"])] = i; i += 1
    def mask(k, v):
        s = next(s for s in e["findings"][k]["slots"] if v in s["variants"]); return arr[pos[(k, s["index"])]]
    for v in OUTV:
        chans = []
        for k in sorted(e["findings"], key=int):
            if v == "bilateral_split_union":
                ed = e["findings"][k]["edited"]
                chans.append(np.maximum(mask(k, "left_only"), mask(k, "right_only")) if ("left_only" in ed and "right_only" in ed) else mask(k, "original"))
            else:
                chans.append(mask(k, v))
        nib.save(nib.Nifti1Image(np.stack(chans).astype(np.uint8), img.affine, img.header), os.path.join(out, v, e["name"]))
print("done;", problems, "scans with problems")
