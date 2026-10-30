# Prompt-variant experiment (internal notes)

**Send to teams:** `ReXGrounding_prompt_variants.zip` (= `for_teams/`): `ReXGrounding_test_prompts.json`, where each
finding maps to a list of distinct prompts (original first), plus `INSTRUCTIONS.txt`. Teams return one archive with a
4-D NIfTI per scan whose first axis has one mask per prompt, in file order.

**Keep here:**
- `mapping.json` — per finding, which list position corresponds to which variant(s). A position carries several
  variants when the edit left the sentence unchanged (then the original prompt stands in for them).
- `ReXGrounding_test_<variant>.json` — the six single-variant test files (one sentence per finding) used to build the
  prompt lists; also what `paper/analysis/split_prompt_submission.py` reproduces as per-variant submissions.
- `manifest.csv` — per (variant, finding): edited flag, reason if not edited, original and variant text.
- `llm/` — model-rewrite batches for no_lobe_segment, no_laterality, finding_only.

**Scoring a returned archive:** `python3 paper/analysis/split_prompt_submission.py TEAM_prompts.zip out/` writes
`out/<variant>/` in the ordinary submission format; score each folder with the official evaluation code and analyse
each variant on the findings marked `edited == 1` in `manifest.csv`, paired with the same team's `original` run.

Variants (findings edited of 582): original; information removed: no_size (163), no_lobe_segment (375),
no_laterality (446), finding_only (564); side tests: swap_left_right (222), left_only (297), right_only (297);
adversarial: wrong_lobe (299, same side, fixed lobe mapping, segment dropped), wrong_size (62, small -> 30 mm,
large -> 3 mm). The split script also writes bilateral_split_union (left_only OR right_only masks). Built by
`paper/analysis/make_prompt_variants.py` then `paper/analysis/build_prompt_lists.py`. 3,288 prompts in total.
