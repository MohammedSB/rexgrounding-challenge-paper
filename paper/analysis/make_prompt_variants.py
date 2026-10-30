"""Prompt-variant test files for the controlled prompt experiment (inference only, no retraining).

Input : released challenge file (MICCAI_challenge_dataset.json, test split) from ~/Desktop/rexbackup_2026-10-05.zip,
        and language-model rewrites in prompt_variants/llm/out_*.json (no_lobe_segment, no_laterality, finding_only).
Output: prompt_variants/ReXGrounding_test_<variant>.json -- same entry format as the released test split, only the
        finding sentences differ. A finding is edited only if its rewrite passes the checks below; otherwise the
        original sentence is kept, so every file is complete and scored like a normal submission.
        prompt_variants/manifest.csv -- one row per (variant, finding): edited flag, reason if not edited, texts.
Run from the project root."""
import json, re, csv, os, zipfile, glob, collections

ZIP = os.path.expanduser("~/Desktop/rexbackup_2026-10-05.zip")
SRC = "rexbackup_2026-10-05/02_challenge_private_test_data/ReXGroundingCT_MICCAI2026/MICCAI_challenge_dataset.json"
test = json.loads(zipfile.ZipFile(ZIP).read(SRC))["test"]
OUT = "prompt_variants"
LLM = {}
for pat in ("out_*.json", "bilat_out_*.json", "wronglobe_out_*.json"):
    for p in sorted(glob.glob(f"{OUT}/llm/{pat}")):
        for r in json.load(open(p)): LLM.setdefault(r["id"], {}).update({k: v for k, v in r.items() if k != "id"})

# ---------- rule-based edits ----------
NUM = r"\d+(?:[.,]\d+)?(?:\s*(?:-|–|x|×|to)\s*\d+(?:[.,]\d+)?)*"
UNIT = r"(?:mm|cm|millimet(?:er|re)s?|centimet(?:er|re)s?)\b"
SIZE = re.compile(r",?\s*(?:\(\s*)?(?:each\s+|the\s+largest\s+|largest\s+)?(?:measuring|measured|with\s+a\s+(?:diameter|size)\s+of|of)?\s*"
                  r"(?:approximately|approx\.?|about|up\s+to|nearly|smaller\s+than|less\s+than|larger\s+than|greater\s+than|<|>|≤|≥)?\s*"
                  + NUM + r"\s*-?\s*" + UNIT + r"(?:\s+in\s+(?:diameter|size|thickness|length|width))?(?:\s*\))?,?", re.I)
SIZEWORD = re.compile(r"\b(?:subcentimet(?:er|ric)|sub-centimet(?:er|ric)|millimetric|centimetric)\s+", re.I)

def tidy(t):
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;)])", r"\1", t); t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r",\s*,", ",", t); t = re.sub(r"^\s*,\s*", "", t)
    t = re.sub(r"\s+", " ", t).strip(" ,;")
    return t[:1].upper() + t[1:] if t else t

def no_size(s):
    t = SIZE.sub(lambda m: ", " if m.group(0).strip().startswith(",") and m.group(0).strip().endswith(",") else " ", s)
    t = tidy(SIZEWORD.sub("", t))
    t = re.sub(r"(?:,\s*|\s+)(?:some|the\s+largest|largest|each|measuring|with\s+(?:a\s+)?maximal\s+thickness|with\s+a\s+thickness)\s*$", "", t, flags=re.I)
    t = re.sub(r",\s+(in|on|at|within)\b", r" \1", t)        # "nodules, in the" -> "nodules in the" after a removed parenthetical size
    return tidy(t)

def wrong_size(s):
    """Adversarial size: a single stated size is replaced by a clearly different one (small -> 30 mm, large -> 3 mm)."""
    ms = list(re.finditer(NUM + r"\s*-?\s*" + UNIT, s, re.I))
    if len(ms) != 1: return None
    if re.search(r"\b(?:largest|some|each|the\s+other|ranging|respectively)\b[^,;]*\d", s, re.I): return None
    m = ms[0]
    if re.search(r"(?:smaller|less|larger|greater|more)\s+than\s*$|up\s+to\s*$|under\s*$|over\s*$|<\s*$|>\s*$", s[:m.start()], re.I): return None
    vals = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", m.group(0))]
    mm = max(vals) * (10 if re.search(r"cm|centimet", m.group(0), re.I) else 1)
    return s[:m.start()] + ("30 mm" if mm < 15 else "3 mm") + s[m.end():]

def swap(s):
    return re.sub(r"\b(left|right)\b", lambda m: ({"left": "right", "right": "left"}[m.group(1).lower()]).capitalize()
                  if m.group(1)[0].isupper() else {"left": "right", "right": "left"}[m.group(1)], s, flags=re.I)

# ---------- checks ----------
LOBEWORDS = r"\b(?:lobes?|segments?|segmental|lingul\w*|apices|apex|apical|bases|basal|basilar|bibasilar|upper|lower|middle)\b"
SIDEWORDS = r"\b(?:left|right|bilateral\w*|both|bibasilar|bibasal|unilateral)\b"
LOCWORDS = r"\b(?:left|right|bilateral\w*|both|lobes?|segments?|lingul\w*|lungs?|apices|apex|apical|bases|basal|basilar|bibasilar|upper|lower|middle|hil\w+|fissur\w*|diaphragm\w*|rib\w*|mediastin\w*|hemithora\w*)\b"
def sides(t):
    t = t.lower(); out = set()
    if re.search(r"\bleft\b", t): out.add("L")
    if re.search(r"\bright\b", t): out.add("R")
    if re.search(r"\b(?:both|bilateral\w*|bibasilar|bibasal|biapical)\b", t): out |= {"L", "R"}
    return out

def check(v, s, new):
    if new is None or not str(new).strip(): return "no rewrite"
    new = str(new).strip()
    if new == s: return "unchanged"
    if v == "no_size" and (re.search(NUM + r"\s*-?\s*" + UNIT, new, re.I) or SIZEWORD.search(new)): return "size remains"
    if v == "no_size" and re.search(r"\b(?:largest|some|each|the\s+other|ranging|respectively)\b[^,;]*\d", s, re.I): return "size inside a qualifying clause"
    if v == "no_size" and re.search(r"\d\s*" + UNIT + r"[^,;]*\band\b[^,;]*\d\s*" + UNIT, s, re.I): return "several sizes for different sites"
    if v == "no_lobe_segment":
        if re.search(LOBEWORDS, new, re.I): return "lobe/segment/zone word remains"
        if sides(s) and sides(new) != sides(s): return "laterality changed"
    if v == "no_laterality":
        if re.search(SIDEWORDS, new, re.I): return "side word remains"
        if re.search(r"lingul|middle\s+lobe", new, re.I): return "side implied by lingula/middle lobe"
    if v == "swap_left_right":
        if re.search(r"\bleft\b[^,.;]*\bmiddle\s+lobe|middle\s+lobe\s+of\s+the\s+left|\bright\b[^,.;]*\blingul|lingul\w*[^,.;]*\bright\b", new, re.I):
            return "anatomically impossible after swap"
    if v == "finding_only":
        if re.search(LOCWORDS, new, re.I): return "location word remains"
        if re.search(r"\d\s*" + UNIT, new, re.I) or SIZEWORD.search(new): return "size remains"
    if v == "wrong_size":
        if len(re.findall(NUM + r"\s*-?\s*" + UNIT, new, re.I)) != 1: return "size count changed"
        if re.sub(NUM + r"\s*-?\s*" + UNIT, "S", new, flags=re.I) != re.sub(NUM + r"\s*-?\s*" + UNIT, "S", s, flags=re.I): return "other text changed"
    if v == "wrong_lobe":
        if sides(new) != sides(s): return "side changed"
        if not re.search(r"\blobes?\b", new, re.I): return "no lobe in rewrite"
        if re.search(r"\bsegments?\b|lingul", new, re.I): return "segment remains"
        lobes = lambda t: sorted(re.findall(r"\b(upper|middle|lower)\b", t.lower()))
        if set(lobes(new)) & set(lobes(re.sub(r"\b(?:\w+\s+)?segments?\b", "", s, flags=re.I))): return "lobe not changed"
    if v in ("left_only", "right_only"):
        want, other = ("left", "right") if v == "left_only" else ("right", "left")
        if not re.search(r"\b" + want + r"\b", new, re.I): return "side missing"
        if re.search(r"\b(?:" + other + r"|both|bilateral\w*|bibasilar|bibasal|biapical)\b", new, re.I): return "other side remains"
        if sides(s) != {"L", "R"}: return "not bilateral"
    if len(new.split()) < 1: return "empty"
    return ""

def variant_text(v, e, k, s):
    if v == "original": return s
    if v == "no_size": return no_size(s)
    if v == "swap_left_right": return swap(s) if re.search(r"\b(?:left|right)\b", s, re.I) else None
    if v == "wrong_size": return wrong_size(s)
    r = LLM.get(f"{e['name']}#{k}")
    return None if r is None else r.get(v)

VARIANTS = ["original", "no_size", "no_lobe_segment", "no_laterality", "swap_left_right", "finding_only",
            "left_only", "right_only", "wrong_lobe", "wrong_size"]
os.makedirs(OUT, exist_ok=True)
man, edited = [], collections.Counter()
for v in VARIANTS:
    out = []
    for e in test:
        e2 = dict(e); e2["findings"] = {}
        for k, s in e["findings"].items():
            cand = variant_text(v, e, k, s)
            why = "" if v == "original" else check(v, s, cand)
            ok = v != "original" and why == ""
            e2["findings"][k] = cand.strip() if ok else s
            edited[v] += ok
            man.append({"variant": v, "scan": e["name"], "finding": k, "category": e["categories"][k], "edited": int(ok),
                        "not_edited_reason": "" if ok or v == "original" else why, "original": s,
                        "variant_text": e2["findings"][k], "rejected_rewrite": "" if ok or cand in (None, s) else cand})
        out.append(e2)
    json.dump({"test": out}, open(f"{OUT}/ReXGrounding_test_{v}.json", "w"), indent=1, ensure_ascii=False)
with open(f"{OUT}/manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(man[0])); w.writeheader(); w.writerows(man)
n = sum(len(e["findings"]) for e in test)
print("LLM rewrites loaded:", len(LLM))
for v in VARIANTS[1:]:
    reasons = collections.Counter(r["not_edited_reason"] for r in man if r["variant"] == v and not int(r["edited"]))
    print(f"{v:16s} edited {edited[v]:3d}/{n}   not edited: {dict(reasons)}")
