# CLAUDE.md — CS 5330 Team 04 (GPA 4.0) · Prompt Robustness of SAM/MedSAM · Phase 1

> **How to use**: place this file at the repository root as `CLAUDE.md` (Claude Code loads it automatically). Start the session with the kickoff prompt in Section 10.
> **Language**: all deliverables (code, comments, README, reports) are in English.

---

## 0. Official Phase 1 requirements (HIGHEST PRIORITY)

<!-- Paste the instructor's / TA's Phase 1 requirements here verbatim (deadline, rubric, submission format). -->

```
(Not pasted yet. If empty, follow the default scope in Section 3.)
```

Status 2026-11-05: no official spec received. Defaults from Section 3 apply; P1-11 (classical baseline) is treated as recommended, not mandatory.

**Rule**: if anything here conflicts with the rest of this file, the official requirements win. List every conflict first and wait for human confirmation before writing code.

---

## 1. Project and research questions

Promptable segmentation models (SAM, MedSAM) are almost always evaluated with near-perfect box prompts derived from the ground truth. We deliberately feed them imperfect prompts, measure how much Dice drops, and compare against a supervised U-Net that needs no prompt.

| ID | Research question | Answered in |
|---|---|---|
| RQ1 | How fast does Dice drop for SAM ViT-B, SAM ViT-H and MedSAM as box error (shift / scale) grows? | P1 pilot on Kvasir-SEG; P2 all datasets |
| RQ2 | Is the drop different on natural images (COCO) vs. medical images? | P2 |
| RQ3 | At what prompt-error level does the U-Net overtake the foundation models (crossover)? | P1 preliminary; P2 final |
| RQ4 | Effect of click offset and of 1–5 corrective clicks? | P2 |
| RQ5 (stretch) | Does fine-tuning only MedSAM's mask decoder with wider box noise improve robustness? | P3 |

**Phase 1 goals**: (1) build the full framework so that Phase 2 only needs new configs to cover all four datasets; (2) run an end-to-end pilot on Kvasir-SEG: box-perturbation curves for the three foundation models plus a U-Net reference line.

---

## 2. Hard constraints (MUST — violating any of these means the task is not done)

1. **Colab is the only supported platform.** Every experiment is delivered as a Colab notebook that runs top to bottom (`Run all`) on a fresh T4 runtime. The local machine (Windows) is only for writing code and CPU smoke tests.
2. **Never fabricate results.** Every number in the report, README or figures must come from a CSV produced by code in this repo. Anything not yet run is written as `TBD`. No estimates, no "illustrative" data.
3. **AI use must be disclosed.** Course policy: AI is allowed only as a learning aid and must be disclosed (1st violation = 0, 2nd = course failure). Before ending every session, append an entry to `AI_USAGE_LOG.md`: date, what was done, which files were generated or modified by AI, who reviewed them.
4. **Traceable contributions.** Every task has an owner (A / B / C, see Section 7). Commit messages start with `[A]`, `[B]` or `[C]`. `CONTRIBUTIONS.md` records who did what in each phase.
5. **Official preprocessing wins.** SAM uses the official `segment_anything` `SamPredictor`. MedSAM must be checked line by line against the official `MedSAM_Inference.py`. If this file disagrees with the official code, follow the official code and log it in `docs/decisions.md`.
6. **Reproducibility.** All randomness uses fixed seeds. All hyperparameters live in `configs/*.yaml`; nothing hard-coded.
7. **Identical prompts across models.** A perturbed prompt depends only on `(image_id, object_idx, perturb, level, seed)`, never on the model, so model comparisons are paired (see 6.3).
8. **No large files in git.** Datasets, weights, embeddings and predicted masks never go into git. Result CSVs may be committed if their total size is under 10 MB.
9. **Read-only directory.** `course_ref/` (course-provided code / notebooks, if present) is read-only. (As of 2026-11-05 there is no `course_ref/`.)
10. **Git location.** The repo root is `D:\cs5330\promptshift\`. Never run git commands in `C:\Users\ruoyu` or `D:\cs5330`. Before every commit, `git rev-parse --show-toplevel` must print `D:/cs5330/promptshift`. Every commit ends with a `Co-Authored-By: Claude …` trailer (part of the AI disclosure).
11. **Secrets.** The GitHub repo is private; Colab clones it with the token in the Colab secret `GH_TOKEN` (read via `google.colab.userdata`). Kaggle credentials come from `KAGGLE_USERNAME` / `KAGGLE_KEY` or Colab secrets. Never print or commit a token, and never commit `kaggle.json`.
12. **Tests stay offline.** Default tests use a tiny mock segmenter so `pytest -q` is fast and needs no weights or network. Tests that need real weights are marked `@pytest.mark.slow` and skip when the weights are missing.

---

## 3. Phases and Phase 1 scope

> The dates below follow the TA check-in schedule and are **assumptions**. If Section 0 gives official dates, use those.

| Phase | Due (assumed) | Content |
|---|---|---|
| **P1** | Before Round 2 check-in (Nov 16–20) | Infrastructure + end-to-end pilot on Kvasir-SEG |
| P2 | Before Round 3 check-in (Nov 30–Dec 4) | Full experiments on 4 datasets: box shift/scale, click offset, multi-click; U-Nets for ISIC and BUSI; statistical tests; crossover table |
| P3 | Final | Stretch goal (MedSAM decoder fine-tuning), failure-case gallery, Gradio demo, final report and presentation |

### Phase 1 in scope

| ID | Task | Owner |
|---|---|---|
| P1-1 | Repo skeleton, environment, test setup | A |
| P1-2 | Metrics: Dice, IoU, NSD, HD95, with unit tests | A |
| P1-3 | Prompt simulator: box jitter, box scale, click offset (clicks implemented and tested only; large-scale runs in P2) | A |
| P1-4 | Kvasir-SEG loader + fixed split + data statistics | A |
| P1-5 | Loaders and download/verification scripts for ISIC 2018 / BUSI / COCO (no experiments) | B (ISIC) / C (BUSI) / A (COCO) |
| P1-6 | Unified wrappers for SAM ViT-B, SAM ViT-H, MedSAM + MedSAM parity test against official output | A |
| P1-7 | Experiment runner: encode each image once, decode many times, resumable | A |
| P1-8 | Colab notebooks (P1_00–P1_04) | A / B |
| P1-9 | Pilot: Kvasir test split × 3 models × box jitter (6 levels × 3 seeds) + box scale (6 levels) | A |
| P1-10 | U-Net baseline on Kvasir (same test split as the foundation models) | B |
| P1-11 | Classical baseline on Kvasir (the course's classical track). **Recommended; mandatory if the official spec requires all three tracks** | C |
| P1-12 | Analysis and figures: bootstrap 95% CI, U-Net reference line, preliminary crossover | C |
| P1-13 | Data-leakage check: which datasets appear in MedSAM's training data | B |
| P1-14 | Phase 1 report + `CONTRIBUTIONS.md` + `AI_USAGE_LOG.md` | All |

### Phase 1 out of scope (but keep the interfaces ready)

Large-scale click-offset and multi-click experiments; experiments on ISIC, BUSI, COCO; U-Nets for ISIC and BUSI; any fine-tuning; Gradio demo; SAM 2.1; nnU-Net.

---

## 4. Repository layout

```
promptshift/
├── CLAUDE.md
├── README.md                 # overview, one-click Colab instructions, auto-generated results table
├── AI_USAGE_LOG.md
├── CONTRIBUTIONS.md
├── requirements.txt          # do NOT pin torch; use Colab's preinstalled version
├── pyproject.toml            # package config + ruff + pytest
├── .gitignore                # data/ weights/ embeddings/ *.pth *.pt *.npy etc.
├── configs/
│   ├── default.yaml
│   └── p1_kvasir_pilot.yaml
├── src/promptshift/
│   ├── paths.py              # detects Colab vs local; returns data / weights / results paths
│   ├── seeding.py            # stable hash → rng (see 6.3)
│   ├── data/{base,kvasir,isic,busi,coco,splits}.py
│   ├── models/{base,sam,medsam}.py
│   ├── prompts/{box,point}.py
│   ├── metrics.py
│   ├── runner.py
│   ├── baselines/{unet,classical}.py
│   └── analysis/{stats,plots,tables}.py
├── scripts/                  # Python only, no .sh, so they run on both Windows and Colab
│   ├── download_data.py
│   ├── download_weights.py
│   ├── check_data.py
│   ├── run_experiment.py     # python scripts/run_experiment.py --config configs/p1_kvasir_pilot.yaml
│   ├── train_unet.py
│   ├── run_classical.py
│   ├── make_figures.py
│   ├── make_report_tables.py # tables + renders reports/phase1_report.md from the template
│   ├── model_parity.py       # P1-6 acceptance: MedSAM vs official script, SAM vs SamPredictor
│   └── make_debug_prompts.py # figures/debug_prompts.png
├── notebooks/                # keep thin: clone repo → pip install -e . → call scripts
│   ├── P1_00_setup_colab.ipynb
│   ├── P1_01_data_check.ipynb
│   ├── P1_02_medsam_parity.ipynb
│   ├── P1_03_kvasir_box_pilot.ipynb
│   └── P1_04_unet_kvasir.ipynb
├── splits/kvasir_seg.json    # committed
├── tests/
├── results/{raw,summary}/
├── figures/
├── docs/{decisions,leakage,data_stats,progress}.md
├── external/                 # git-ignored; pinned clone of bowang-lab/MedSAM for the parity test
└── reports/
    ├── phase1_report.template.md   # edit this (prose only)
    ├── phase1_report.md            # generated; do not edit
    └── tables/*.md                 # generated
```

**Paths**: on Colab, data, weights and results live under `/content/drive/MyDrive/promptshift/` so a disconnect loses nothing. Locally they live under `data/`, `weights/`, `results/` in the repo. Only `paths.py` knows about locations; all other code uses `pathlib` and never hard-codes paths.

---

## 5. Core interfaces

```python
# src/promptshift/data/base.py
@dataclass
class Sample:
    image_id: str
    image: np.ndarray          # H×W×3, uint8, RGB
    mask: np.ndarray           # H×W, bool, full GT (union of all objects, specks removed)
    objects: list[np.ndarray]  # one H×W bool mask per connected component (8-connectivity, area ≥ min_object_frac·H·W)
    dataset: str
    meta: dict                 # original filename, size, class, etc.

class SegDataset(Protocol):
    name: str
    def ids(self, split: str) -> list[str]: ...
    def get(self, image_id: str) -> Sample: ...
```

```python
# src/promptshift/models/base.py
class PromptableSegmenter(Protocol):
    name: str                  # "sam_vit_b" | "sam_vit_h" | "medsam"
    supports_points: bool      # False for MedSAM
    def set_image(self, image: np.ndarray) -> None: ...                 # encode once per image
    def predict_box(self, box_xyxy: np.ndarray) -> np.ndarray: ...     # original-image coords → H×W bool
    def predict_points(self, coords: np.ndarray, labels: np.ndarray,
                       mask_input: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray]: ...          # → (H×W bool, low-res logits)
```

**Extensibility**: adding a dataset = one new loader file + one config. Adding a perturbation = one new function in `prompts/` + registration.

---

## 6. Technical specifications

### 6.1 Data

**Kvasir-SEG (fully done in P1)**
- Official page: https://datasets.simula.no/kvasir-seg/ . 1000 polyp images of varying size; trust the actual download count.
- **Masks are JPEG** with compression noise and must be binarized: `mask = gray > 127`.
- After binarization, connected components smaller than `min_object_frac` (default 0.001) of the image area are specks; remove them from both the GT mask and `objects`. Log how many were removed in `docs/data_stats.md`.
- Compute boxes from the masks; do not use the official `kavsir_bboxes.json`, so all four datasets follow the same convention: `x0 = min col, y0 = min row, x1 = max col, y1 = max row` (inclusive, same as MedSAM's `get_bbox`).
- **Split**: if `course_ref/` contains the course capstone's official split, use it. Otherwise split randomly with seed 42 into 700 / 100 / 200 (train / val / test), save to `splits/kvasir_seg.json` and commit it. **All methods (SAM, MedSAM, U-Net, classical) are evaluated on the same test split.**
- Write data statistics to `docs/data_stats.md`: image-size distribution, objects-per-image distribution, object-area-fraction distribution.

**ISIC 2018 Task 1 / BUSI / COCO (P1: loaders and verification only)**
- ISIC 2018 Task 1: https://challenge.isic-archive.com/data/ . Use the official training set; trust the official count.
- BUSI (Kaggle: `aryashah2k/breast-ultrasound-images-dataset`): drop the `normal` class (no lesion); when an image has several `*_mask*.png` files, take their union; deduplicate by file hash and log the number of duplicates.
- COCO val2017 (`http://images.cocodataset.org/zips/val2017.zip` + `annotations_trainval2017.zip`): keep only instances with `iscrowd=0` and area ≥ 1% of the image; sample 1000 instances with seed 42. Each instance is an independent sample.
- Acceptance: each dataset's count matches the official number (log the reason if not), plus one 5-sample visualization per dataset.

### 6.2 Models

**SAM** (https://github.com/facebookresearch/segment-anything)
- Weights: `sam_vit_b_01ec64.pth`, `sam_vit_h_4b8939.pth` from `https://dl.fbaipublicfiles.com/segment_anything/<name>`.
- Use `SamPredictor`; let it handle resizing, normalization and padding. Do not reimplement them.
- Box prompt: `predict(box=box[None], multimask_output=False)`.
- Single click: `multimask_output=True`, keep the mask with the highest predicted IoU (standard SAM protocol).
- Multi-click (P2): from the 2nd click on, `multimask_output=False` and pass the previous round's low-res logits as `mask_input`.
- Record every protocol detail in `docs/decisions.md`.

**MedSAM** (https://github.com/bowang-lab/MedSAM)
- Weights: `medsam_vit_b.pth`; use the download link in the official README (e.g. via `gdown`).
- Below is our understanding of the official `MedSAM_Inference.py` pipeline. **Verify it line by line against the source; the source wins on any difference.**
  1. Replicate grayscale images to 3 channels;
  2. Resize directly to 1024×1024 (no aspect-ratio preservation), bicubic, keep uint8;
  3. Min-max normalize to [0, 1];
  4. Call `image_encoder` directly, **without** SAM's pixel_mean/std normalization;
  5. Scale box coordinates: `box / [W, H, W, H] * 1024`;
  6. `prompt_encoder(points=None, boxes=box, masks=None)`, then `mask_decoder(..., multimask_output=False)`;
  7. Sigmoid → bilinear upsampling to (H, W) → threshold 0.5.
- Loading: first try loading the MedSAM weights with the facebookresearch `sam_model_registry["vit_b"]`. **If the parity test fails**, vendor the relevant MedSAM code into `third_party/medsam/` and load with the original code.
- **Parity test (acceptance criterion for P1-6)**: on 3 Kvasir images with GT boxes, compare our wrapper's output with the official script or the official `medsam_inference` function. Required: Dice ≥ 0.99 between the two. Record the result in `notebooks/P1_02_medsam_parity.ipynb` and in the report.

**General**
- Device selection: cuda > mps > cpu. Inference in fp32, no AMP (needed for parity).
- Local CPU smoke tests use only `sam_vit_b` and `medsam` (ViT-H weights are ~2.5 GB; run ViT-H on Colab only).

### 6.3 Prompt simulator

**Stable random numbers (critical)**: Python's built-in `hash()` on strings changes between processes. **Do not use it.**

```python
key = f"{global_seed}|{dataset}|{image_id}|{object_idx}|{perturb}|{level}|{seed}"
rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
```

The key does not include the model name, so all models receive exactly the same prompt for a given setting.

**Box jitter `box_jitter`**: levels `r ∈ {0, 0.05, 0.10, 0.20, 0.30, 0.50}`, seeds `{0, 1, 2}` (at r = 0 all seeds give the same box; record them anyway).

```
w = x1 - x0;  h = y1 - y0
dx0, dx1 ~ U(-r·w, r·w);  dy0, dy1 ~ U(-r·h, r·h)        # four edges sampled independently
box' = [x0+dx0, y0+dy0, x1+dx1, y1+dy1], clipped to [0, W-1] × [0, H-1]
if x1'-x0' < 2 or y1'-y0' < 2: resample with the same rng, up to 10 times; if still invalid, the prompt is invalid
```

**Box scale `box_scale`**: levels `s ∈ {0.8, 0.9, 1.0, 1.1, 1.25, 1.5}`. Scale width and height about the box center, clip to the image. Deterministic. Validity is re-checked after clipping with the same rule as jitter (width and height ≥ 2); there is no rng, so an invalid scaled box is invalid immediately.

**Invalid prompts (both perturbations)**: the row is never skipped. If any object's prompt in an image is invalid, the whole image row for that configuration gets `flag=invalid` with NaN metrics, and the per-object rows of that image/configuration also get `flag=invalid`. The runner counts and reports invalid rows.

**Level formatting in the key**: `level` is written as `format(float(level), ".6g")` (so `0.10`, `0.1` and `1e-1` give the same key); `object_idx` and `seed` as plain integers.

**Ties**: wherever an argmax is taken (e.g. the DT center), use the first index in row-major order.

**Resampling count**: "up to 10 times" means at most 1 + 10 draws in total (`max_resample: 10` in the config).

**Image border in the click DT**: the distance transform is computed on the object mask padded with one background pixel, so the image border counts as object boundary (DT = 1 on border pixels). Otherwise an object cut by the image border would get its "center" on the border. This matches the metrics, where border pixels are surface pixels.

**Record `box_iou` for every perturbation**: IoU between the perturbed box and the GT box, usable later as a model-independent x-axis.

**Click offset `point_offset`** (implemented + unit-tested in P1; experiments in P2): levels `d ∈ {0, 0.25, 0.5, 0.75, 1.0}`.
- `DT = scipy.ndimage.distance_transform_edt(object_mask)`; `center = argmax(DT)` (not the centroid, which can fall outside non-convex objects).
- d = 0: use `center`. For 0 < d < 1: target depth `t = 1 - d`; sample one pixel uniformly (stable rng) from `|DT/DT.max() - t| ≤ 0.05`; if that band is empty, take the closest pixel. d = 1: sample from pixels with `DT ≤ 1` (inner boundary). Label = 1 (foreground).

**Multi-object images**: prompt and predict each connected component separately, take the union of the predicted masks, and compare with the full GT to get image-level metrics (same basis as the U-Net). Save per-object metrics to a separate CSV for analysis.

**Required unit tests**: the same key always yields the same prompt; r = 0 and s = 1.0 return the original box; every output box is valid and inside the image; clicks always fall inside the object mask; generate `figures/debug_prompts.png` for visual inspection.

### 6.4 Metrics (`metrics.py`)

- **Dice** = 2|P∩G| / (|P|+|G|). Both empty → 1.0; exactly one empty → 0.0.
- **IoU**.
- **NSD** (normalized surface Dice) with tolerance `nsd_tolerance_px` (default 2 px), computed at original image resolution.
- **NSD** empty cases: both empty → 1.0; exactly one empty → 0.0 (same as Dice).
- **HD95**: NaN when the prediction is empty; count empty predictions separately. NaN HD95 values are excluded from means, and the number of empty predictions is reported next to every HD95 mean.
- Tests: identical masks → Dice = 1; disjoint masks → Dice = 0; two offset squares checked against the analytic value. If MONAI is installed, cross-check NSD and HD95 against `monai.metrics`.

### 6.5 Runner (`runner.py`)

```
for model in cfg.models:                 # load each model once
  for image_id in split:                 # skip image_ids already done (resume)
    model.set_image(sample.image)        # encode once
    for (perturb, level, seed) in grid:  # decode many times
      prompts = [make_prompt(obj, ...) for obj in sample.objects]
      pred = union(model.predict_box(p) for p in prompts)
      append one row to the CSV
    flush every 10 images
```

**Result CSV** (long format, `results/raw/<exp_id>/<model>.csv`):

```
exp_id, dataset, split, model, image_id, n_objects, perturb, level, seed,
dice, iou, nsd, hd95, box_iou_mean, pred_area_frac, gt_area_frac,
encode_ms, decode_ms, flag, git_sha, timestamp
```

The U-Net and the classical baseline write the same format with `perturb = "none"`, `level = NaN`.

- `flag` is `ok` or `invalid` (invalid prompt, see 6.3; metrics are NaN). Every (image, configuration) always gets exactly one row.
- `git_sha` is the short commit hash, with a `-dirty` suffix when the working tree has uncommitted changes.

**Per-object CSV** (`results/raw/<exp_id>/<model>_objects.csv`):

```
exp_id, dataset, split, model, image_id, object_idx, perturb, level, seed,
dice, iou, box_iou, object_area_frac, flag
```

**Acceptance**: a local CPU run on 5 images succeeds (with the mock segmenter in the default tests; with `sam_vit_b` as a `slow` test when weights are present); killing a run halfway and restarting produces no duplicate and no missing rows (covered by a test).

### 6.6 U-Net baseline (P1-10)

- If `course_ref/` contains the course's U-Net pipeline, port it faithfully and note it in `docs/decisions.md`.
- Otherwise use this default: `segmentation_models_pytorch.Unet("resnet34", encoder_weights="imagenet", classes=1)`, inputs resized to 352×352; augmentation: hflip, vflip, rotate90, light color jitter; loss = BCE + Dice; AdamW, lr 1e-4, weight decay 1e-4, batch 8, 60 epochs, cosine schedule; select the checkpoint with the best val Dice.
- At test time, upsample predictions to the original size, threshold at 0.5, compute metrics with the same `metrics.py`.
- P1: 1 seed. P2: 3 seeds. Save the training-curve plot; store the best checkpoint on Drive.

### 6.7 Classical baseline (P1-11)

Matches the course's Module 3 content: color-space conversion (Lab or HSV) → Gaussian smoothing → Otsu threshold → morphological opening and closing → keep the largest connected component. A small parameter grid search is allowed **on train/val only**; evaluate on test exactly once. Document the method and parameters in the report.

### 6.8 Analysis and figures (P1-12)

- **The statistical unit is the image.** Average the 3 seeds within each image first, then bootstrap over images (n = 2000, fixed seed, 95% percentile CI).
- At each level, compare models with a paired Wilcoxon signed-rank test; apply Holm correction across levels. Label all of this as preliminary in P1.
- **Two summary numbers**:
  - ΔDice(0 → 20%): the Dice drop of each model when box jitter goes from 0 to 20%;
  - Preliminary crossover: the first jitter level where a model's mean Dice falls below the U-Net's mean Dice, plus a continuous estimate by linear interpolation between the two neighboring levels; if it never falls below by 50%, report "> 50%".
- **Figures** (generated by `make_figures.py` from the CSVs; PNG at 300 dpi + PDF; colorblind-safe palette):
  - `figures/p1_kvasir_box_jitter.(png|pdf)`: x = jitter %, y = Dice; one line per model with CI band; U-Net as a dashed horizontal line with its CI band; classical baseline as a dotted line;
  - `figures/p1_kvasir_box_scale.(png|pdf)`;
  - `figures/p1_kvasir_dice_vs_boxiou.(png|pdf)`;
  - `figures/p1_kvasir_examples.png`: qualitative grid of 3 images × several levels (image, GT, prompt box, predictions of the three models).

### 6.9 Data-leakage check (P1-13)

Write `docs/leakage.md` with columns: Dataset | In MedSAM training data? | Evidence (paper supplementary table or repo docs, with link and location).

**If no reliable source can be found, write `unknown`. Never fill it in from memory.** Datasets seen by MedSAM are marked `seen` in all results and reported separately.

### 6.10 Phase 1 report (P1-14)

`reports/phase1_report.md`, about 2 pages:
1. Goal & RQs
2. Pipeline (one diagram: data → prompt simulator → models → metrics → analysis)
3. Sanity checks (MedSAM parity result, data statistics)
4. Pilot results on Kvasir-SEG (figures + table)
5. Issues & design decisions (from `docs/decisions.md`)
6. Plan for Phase 2
7. Contributions (who did what)
8. AI usage statement (summary of `AI_USAGE_LOG.md`)

**Every number in a table is generated by `make_report_tables.py` into `reports/tables/*.md` and then included in the report. No hand-typed numbers.**

---

## 7. Execution order and Definition of Done

> ⏸ HUMAN = a person must run something on Colab. When you reach such a step, stop, tell the human exactly which notebook to run and where to put the output files, then wait.

| # | Task | Owner | Acceptance criteria |
|---|---|---|---|
| 0 | Read this file and Section 0; output the plan, a conflict list and the human-only steps | — | Human confirms before any code is written |
| 1 | P1-1 Skeleton: `pyproject.toml`, `requirements.txt`, `.gitignore`, README skeleton, `paths.py`, `seeding.py` | A | `pip install -e .` succeeds; `ruff check .` and `pytest -q` pass |
| 2 | P1-2 Metrics + tests | A | All tests listed in 6.4 pass |
| 3 | P1-3 Prompt simulator + tests | A | All tests listed in 6.3 pass; `debug_prompts.png` generated |
| 4 | P1-4 Kvasir loader + split + statistics | A | 1000 image/mask pairs matched; `splits/kvasir_seg.json` committed; `docs/data_stats.md` written |
| 5 | P1-5 ISIC / BUSI / COCO loaders + download verification | B / C / A | Count checks logged; one 5-sample visualization each |
| 6 | P1-6 Model wrappers + parity test | A | MedSAM parity Dice ≥ 0.99 on 3 images; SAM wrapper output identical pixel-for-pixel to a direct `SamPredictor` call |
| 7 | P1-7 Runner + resume | A | Local CPU smoke test passes; resume test passes |
| 8 | P1-8 Colab notebooks | A / B | ⏸ HUMAN: `Run all` on a fresh T4 runtime with no errors |
| 9 | P1-9 Pilot experiment | A | ⏸ HUMAN runs it on Colab; for each of the 3 models, CSV rows = number of images × 24 configurations, and the number of `flag=invalid` rows is reported (expected ~0) |
| 10 | P1-10 U-Net | B | ⏸ HUMAN trains on Colab; test-metric CSV and training-curve plot produced |
| 11 | P1-11 Classical baseline | C | Test-metric CSV produced; parameters chosen on train/val only |
| 12 | P1-12 Analysis and figures | C | Every figure in 6.8 generated by script |
| 13 | P1-13 Leakage check | B | Every row has a source or is marked `unknown` |
| 14 | P1-14 Report + CONTRIBUTIONS + AI log | All | No hand-typed numbers in the report; every `TBD` resolved or explained |

**Owners**: A = Ruoyu Wang (framework, Kvasir, COCO). B = Hao Wang (ISIC, U-Net, leakage check). C = Jiani He (BUSI, classical baseline, analysis and figures). Each member should run Claude Code on their own tasks so the contribution record matches reality.

**Local environment**: conda env `promptshift`, Python 3.11; `pyproject.toml` declares `requires-python = ">=3.10"`.

**Before P1-8**: ask the human for the GitHub URL (no remote until then; P1-1 to P1-7 are local only).

**Leakage labels**: figures and tables read the seen/unseen label from `docs/leakage.md` (owner B); a dataset with no row, or a missing file, counts as `unknown`. Never fill it in for B.

---

## 8. Working rules for Claude Code

1. **Announce, then act.** Before each task, state in 3–5 lines which files you will create or modify. After finishing, run `ruff check . && pytest -q`; commit only if both pass.
2. **Small commits.** One task per commit, message format: `[A] P1-3: box/point prompt simulator + tests`.
3. **Unspecified choices**: pick the simplest reproducible option and log it in `docs/decisions.md` (date | question | choice | reason).
4. **When unsure how the official implementation works, read the official source code. Do not rely on memory.**
5. **No silent failures.** Every `try/except` logs and counts the error; the runner prints a summary at the end.
6. **Windows compatibility.** Use `pathlib`; no `.sh` scripts or Makefiles; the local shell is Git Bash.
7. **Session handoff.** Before ending each session, update `docs/progress.md` (current step, next step, blockers) and `AI_USAGE_LOG.md`, so the next session can pick up directly.
8. **Stop immediately at ⏸ HUMAN steps** and give concrete instructions. Never pretend a Colab run has happened.

---

## 9. Preview of Phases 2 and 3 (not in P1, but design for them)

- **P2**: full experiments on ISIC, BUSI, COCO; click offset; corrective multi-click (1–5 clicks; each click goes to the largest current error region: take the largest connected component of FN or FP, click at its distance-transform maximum, positive on FN, negative on FP); U-Nets for ISIC and BUSI (3 seeds); Wilcoxon + Holm; crossover table; report seen and unseen datasets separately.
- **P3**: freeze the MedSAM encoder and fine-tune only the decoder with box noise widened to 0–50 px (stretch goal); failure-case gallery (low contrast, blurry boundaries, small objects); Gradio demo (drag a box, compare SAM and MedSAM live); final report and presentation.

---

## 10. Kickoff prompt (paste into Claude Code)

```
Read CLAUDE.md fully. If Section 0 contains official Phase 1 requirements, reconcile them with Section 3 first.
Then output, without writing any code:
(1) a numbered plan mapped to tasks P1-1 … P1-14 for owner [A] (or the owner I specify),
(2) every conflict, ambiguity or open question you found,
(3) the exact steps that need me to run Colab (⏸ HUMAN).
Wait until I reply "go".
```
