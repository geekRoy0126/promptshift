# Design decisions

Every choice not fixed by `CLAUDE.md`, and every human answer to an open question.
Format: date | question | choice | reason. Newest at the bottom.

| Date | Question | Choice | Reason |
|---|---|---|---|
| 2026-11-05 | Official Phase 1 spec? | None received; follow CLAUDE.md Section 3 defaults. P1-11 (classical) is recommended, owner C. | Human answer #1 |
| 2026-11-05 | Repo root and git | `D:\cs5330\promptshift\`, `git init -b main`; no git commands in `C:\Users\ruoyu` or `D:\cs5330`; check `git rev-parse --show-toplevel` before every commit. No remote until P1-8. | Human answer #2 (the home folder is itself a git repo) |
| 2026-11-05 | Private repo on Colab | Clone with the Colab secret `GH_TOKEN` via `google.colab.userdata`; never print or commit it. | Human answer #2 |
| 2026-11-05 | `lab1-Ruoyu wang.pdf` | Unrelated; stays outside the repo. | Human answer #3 |
| 2026-11-05 | `course_ref/` | Does not exist. Kvasir split = seed 42, 70/10/20; U-Net = smp default (CLAUDE.md 6.6). | Human answer #4 |
| 2026-11-05 | Owners B / C | B = Hao Wang, C = Jiani He. | Human answer #5 |
| 2026-11-05 | AI disclosure | Every commit keeps a `Co-Authored-By: Claude` trailer; `AI_USAGE_LOG.md` updated each session. Allowed level of AI use to be confirmed with the TA on 2026-11-06. | Human answer #6 |
| 2026-11-05 | Invalid prompts vs. "no missing rows" | Always write the row: `flag=invalid`, NaN metrics. P1-9 acceptance: rows = images × 24 per model, invalid count reported (expected ~0). | Human answer #7; keeps row counts checkable |
| 2026-11-05 | One invalid object in a multi-object image | The whole image row for that configuration is `flag=invalid`; count logged. | Human answer #8; a partial union would not be comparable to the full GT |
| 2026-11-05 | Kvasir count ≠ 1000 | Keep 70/10/20 fractions, log the actual count. | Human answer #9 |
| 2026-11-05 | MedSAM preprocessing in CLAUDE.md vs. official code | Official `MedSAM_Inference.py` wins; every difference logged here. | Human answer #10; CLAUDE.md 2.5 |
| 2026-11-05 | `third_party/medsam/` | Added only if the parity test fails; then update CLAUDE.md Section 4. | Human answer #11 |
| 2026-11-05 | Duplicate configurations (jitter r=0 × 3 seeds, scale s=1.0) | Keep all 24 configurations. | Human answer #12; sanity check that identical prompts give identical masks |
| 2026-11-05 | NSD with empty masks | Both empty → 1.0; exactly one empty → 0.0. | Human answer #13; same convention as Dice |
| 2026-11-05 | HD95 aggregation | NaN for empty predictions, excluded from means, empty-prediction count reported next to each mean. | Human answer #14 |
| 2026-11-05 | Per-object CSV | `results/raw/<exp_id>/<model>_objects.csv`: exp_id, dataset, split, model, image_id, object_idx, perturb, level, seed, dice, iou, box_iou, object_area_frac, flag. | Human answer #15 |
| 2026-11-05 | Box scale after clipping | Re-check validity (side ≥ 2); same invalid rule as jitter, no resampling (deterministic). | Human answer #16 |
| 2026-11-05 | argmax ties | First index in row-major order (numpy default). | Human answer #17 |
| 2026-11-05 | `git_sha` with uncommitted changes | Short SHA + `-dirty` suffix. | Human answer #18 |
| 2026-11-05 | Tests without weights | Default tests use a tiny mock segmenter (fast, offline); real-weight tests are `@pytest.mark.slow` and skip when weights are missing. | Human answer #19; local downloads may be blocked |
| 2026-11-05 | Python version | Local: conda env `promptshift`, Python 3.11. `requires-python = ">=3.10"`. Colab versions recorded after C1. | Human answer #20 |
| 2026-11-05 | Kaggle credentials | `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars or Colab secrets; `kaggle.json` is git-ignored. BUSI loader is C's task. | Human answer #21 |
| 2026-11-05 | Missing U-Net results / leakage labels | Show `TBD` until B's files arrive. seen/unseen label is read from `docs/leakage.md` (owner B); missing → `unknown`. Not filled in by A. | Human answer #22; CLAUDE.md 6.9 |
| 2026-11-05 | Level text in the rng key | `format(float(level), ".6g")`, so 0.1 / 0.10 / 1e-1 give the same key; `None`/NaN → `none`. | YAML may write 0.10 or 0.1; the key must not depend on that |
| 2026-11-05 | Storage root resolution | Explicit argument > `PROMPTSHIFT_ROOT` env var > Colab Drive (`/content/drive/MyDrive/promptshift`) > repo. On Colab without a mounted Drive, raise an error instead of silently writing to `/content`. | No silent failures (CLAUDE.md 8.5); env var lets tests use temp folders |
| 2026-11-05 | Colab detection | `COLAB_RELEASE_TAG` env var, else `google.colab` importable. | Simple and works on current Colab runtimes |
| 2026-11-05 | Where `CLAUDE.md` lives | The spec now lives in the repo (`promptshift/CLAUDE.md`, per Section 4) with the answers above applied. The original `D:\cs5330\CLAUDE.md` is outside the repo and outdated. | Section 4 puts CLAUDE.md at the repo root |
| 2026-11-05 | Line endings | `.gitattributes` with `* text=auto eol=lf` (PNG/PDF binary). | Local git has `autocrlf`; LF keeps files identical on Windows, Colab and teammates' machines |
| 2026-11-05 | Surface definition for NSD / HD95 | Mask minus one binary erosion with a 4-connected cross; pixels on the image border count as surface; Euclidean distances in pixels; "within tolerance" means `<=`. HD95 = max of the two directed 95th percentiles (numpy linear interpolation). | Matches MONAI 1.6.1 (`compute_surface_dice`, `compute_hausdorff_distance`) to 1e-5 on random blobs and on border-touching masks (`tests/test_metrics.py`) |
| 2026-11-05 | HD95 when the GT is empty | NaN (same as empty prediction). | Distance to an empty set is undefined; does not occur on Kvasir |
| 2026-11-05 | MONAI dependency | Optional, test-only (cross-check skips if MONAI is missing); not in `requirements.txt`. Local env has torch 2.15.0+cpu and monai 1.6.1. | Keeps the runtime light; the cross-check still runs where installed |
| 2026-11-05 | Jitter resampling count | `max_resample: 10` = at most 1 + 10 draws from the same rng. Rng draw order per attempt: dx0, dy0, dx1, dy1 via `U(-1, 1) × (r·w, r·h, r·w, r·h)`. | Spec says "resample up to 10 times"; fixed order keeps prompts reproducible |
| 2026-11-05 | Box geometry | Width = x1 − x0 (as in the spec's jitter formula); `box_iou` uses the same continuous areas. A GT box with side < 2 (e.g. a 1-px-wide object) is invalid at every level. | One convention for jitter, scale and box_iou |
| 2026-11-05 | Image border in the click DT | DT on the mask padded by one background pixel, so the image border is object boundary. | Otherwise a border-cut object's "center" lands on the border; matches the metrics' surface convention |
| 2026-11-05 | Click candidate order | Candidates are enumerated in row-major order and one is drawn with `rng.integers`; an empty depth band falls back to the closest-depth pixel (first in row-major order) and sets `fallback=True`. | Deterministic given the rng; fallback is visible for counting |
| 2026-11-05 | `debug_prompts.png` input | Synthetic masks (non-convex C shape, border-cut ellipse) until Kvasir is available (P1-4). | Exercises the hard cases without data; regenerate with `scripts/make_debug_prompts.py` |
