# ReXGrounding Challenge (MICCAI 2026): findings for model development

This summarizes what the challenge analysis found, so that its lessons can be used to build a new model. The full
argument and numbers are in `paper/main.tex` (compiled: `paper/main.pdf`).

## Task and evaluation

- Input: a non-contrast chest CT volume (CT-RATE, one hospital in Turkey) and one free-text finding sentence.
  Output: one 3D binary mask holding every instance of that finding.
- Data: training 2,992 scans (7,687 findings) annotated with up to 3 instances per finding (partial labels);
  validation 200 scans (381 findings) and test 300 scans (582 findings) annotated exhaustively.
- 14 categories. Focal: nodules and masses, linear opacities, atelectasis and consolidation, ground-glass opacity,
  pleural effusion and thickening, others. Non-focal: emphysema, micronodules, bronchiectasis, septal thickening,
  bronchial wall thickening, other. About 87% of test findings are focal.
- Metrics (paper Section 2.4): Dice per finding over the whole mask, averaged per scan, then over scans (primary).
  Hit rate = share of findings with Dice >= 0.1. Instance F1 from connected components matched at Dice >= 0.2.
  Components under 10 voxels are removed and at most the 50 largest kept before scoring.

## Where the field stands

- Pre-challenge best (VoxTell) Dice 0.285 on the original 100 test scans. The winner reached 0.390 on those scans
  and 0.375 on all 300. Off-the-shelf SAT, SegVol and BiomedParse-V scored near 0. SAT fine-tuned reached 0.205.
- Inter-radiologist agreement on 50 scans: Dice 0.32, hit rate 0.76, instance F1 0.47 (focal Dice 0.34, non-focal
  0.15). The top teams match or exceed this on Dice and hit rate but not on instance F1 (best 0.34).
- Full leaderboard: `paper/tables/leaderboard_full.tex`. Bootstrap ranking statistics:
  `paper/analysis/ranking_stats.json`.

## What the winning methods did (paper Section 3.2, Appendix B, `paper/team_writeups/`)

All three winners used the same recipe. A frozen Qwen3-Embedding encoder turns the sentence into one vector, and a 3D
segmentation network takes the CT volume plus that vector.

- Image encoder: CIRCLE chest-CT foundation model (Team 1), residual-encoder nnU-Net from scratch (Team 2),
  VoxTell encoder plus TAP-CT (Team 3).
- Text conditioning: the VoxTell decoder (text embedding as transformer query cross-attending to bottleneck features,
  then multiplied channel-wise into every decoder stage) for Teams 1 and 3. FiLM (per-channel scale and shift from
  the embedding, no cross-attention) for Team 2 and Team 3's TAP-CT branch.
- What helped, as reported by the teams:
  - Anatomical post-processing: restricting each mask to the lobe or side named in the sentence (Teams 3, 4, 5, 6).
  - Per-finding or per-category thresholds (Team 3).
  - Ensembling different architectures (Teams 3 and 5).
  - Extra training data: COVID-19-20, OmniAbnorm-CT, an ILD database and PatchChestCT (Team 4).
  - Cleaning training labels (one team corrected more than 100 masks, e.g. wrong side, findings not in the report).
  - Handling partial labels by excluding uncertain voxels from the loss (Team 5).

## Where models fail (paper Sections 3.5 to 3.7, `paper/analysis/further_analyses.json`)

- Diffuse, non-focal findings are segmented far worse than focal ones. The winner's margin came mostly from
  non-focal findings, especially emphysema. Per-team, per-category Dice: `paper/figures/fig_categories_perf.py`.
- Multi-instance findings: models find some lesions but not all. Findings with 4 or more instances score lower than
  single-instance ones. Partial training labels likely teach models to stop after a few instances.
- Location: models sometimes segment the right kind of abnormality at the wrong place, which overall Dice hides.

## Controlled prompt experiment (Teams 1 to 4)

The same scans were re-run with edited sentences, comparing Dice on only the findings each edit changed. Script:
`paper/analysis/prompt_experiment.py`. Results: `prompt_variants/returns/{CIRCLE,AIMHI,fomofo,needpaper}/prompt_experiment.json`
(CIRCLE = Team 1, AIMHI = Team 2, fomofo = Team 3, needpaper = Team 4). Teams 2, 3 and 4 behaved almost
identically. Team 1 (CIRCLE foundation-model encoder) differed: it relied less on location text and more often
returned an empty mask when the location was wrong.

| Edit | n | Team 1 | Team 2 | Team 3 | Team 4 |
|---|---|---|---|---|---|
| Size removed | 163 | 0.415 -> 0.409 | 0.390 -> 0.393 | 0.391 -> 0.384 | 0.382 -> 0.387 |
| Wrong size (small to 30 mm, large to 3 mm) | 62 | 0.463 -> 0.405 | 0.430 -> 0.433 | 0.426 -> 0.417 | 0.424 -> 0.419 |
| Laterality removed | 446 | 0.361 -> 0.355 | 0.339 -> 0.320 | 0.346 -> 0.313 | 0.343 -> 0.315 |
| Lobe or segment removed | 375 | 0.379 -> 0.368 | 0.363 -> 0.325 | 0.363 -> 0.318 | 0.363 -> 0.303 |
| Finding name only | 564 | 0.361 -> 0.341 | 0.341 -> 0.292 | 0.345 -> 0.286 | 0.347 -> 0.265 |
| Left and right swapped | 222 | 0.419 -> 0.140 | 0.397 -> 0.109 | 0.404 -> 0.103 | 0.411 -> 0.081 |
| Wrong lobe, same lung | 299 | 0.380 -> 0.167 | 0.368 -> 0.120 | 0.366 -> 0.073 | 0.370 -> 0.118 |
| Bilateral finding, left only | 297 | 0.327 -> 0.222 | 0.305 -> 0.201 | 0.316 -> 0.188 | 0.305 -> 0.171 |
| Bilateral finding, right only | 297 | 0.327 -> 0.263 | 0.305 -> 0.251 | 0.316 -> 0.238 | 0.305 -> 0.228 |
| Union of left-only and right-only | 297 | 0.327 -> 0.317 | 0.305 -> 0.310 | 0.316 -> 0.309 | 0.305 -> 0.304 |

What this means for a new model:
- Side is followed strongly. After a left/right swap, hit rate fell from 0.77-0.80 to 0.16-0.31. For Teams 2-4 the
  share of the mask on the reference side fell from 92-93% to 22-24% and about 75% of masks moved to the other side.
  Some masks vanished instead (12-21% for Teams 2-4, 31% for Team 1), especially large consolidations.
- Lobe is followed only partly. With a wrong lobe the mask mostly stays in the same lung (84-86%) but Dice collapses
  (0.37-0.38 to 0.07-0.17).
- Stated size is mostly ignored. Teams 2-4 change by less than 0.01 under both size edits. Team 1 is unchanged when
  size is removed but drops 0.06 with a wrong size (62 findings, Wilcoxon p = 0.44, so weak evidence).
- "Left only" / "right only" prompts put 91-98% / 96-100% of the mask on the named side for Teams 2-4, 85-86% for
  Team 1.
- Removing location details costs 0.02 to 0.08 Dice for Teams 2-4 and at most 0.02 for Team 1, so location text helps
  but most of the signal is the finding type plus image evidence. Team 1, the overall winner, depends least on it.

## Files

- `paper/analysis/`: ranking, bootstrap, further analyses, prompt-experiment scripts and their JSON outputs.
- `paper/figures/*.py`: figure scripts. Imaging data (`paper/figures/data/case*`, `qual/`) is not in the repo.
- `prompt_variants/`: edited test prompts (`ReXGrounding_test_<variant>.json`), `manifest.csv` (which findings each
  edit changed), `mapping.json`, LLM rewrite batches, and per-variant scores.
- Not in the repo: team submissions, predicted and reference masks, CT volumes, participant contact data.
