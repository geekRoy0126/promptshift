# Role C: BUSI experiments, statistics, failure review and demo

This implementation extends the existing team pipeline. It reuses the BUSI loader,
committed splits, model wrappers, runner, metrics and U-Net training code. The completed BUSI box run is documented in [Role C results](role_c_results.md). Development used AI assistance (Codex);
the team should review the changes and record the actual contribution in its private log.

## Run the BUSI experiment

Recommended: run `P1_00_setup_colab.ipynb` first, then
`C_01_busi_analysis.ipynb`. The Role C notebook runs the box experiment, analysis,
failure-case rendering and downloads a summary archive. Existing setup notebooks
fetch `main`: to run an unmerged branch, set `REF` in the setup cell to that branch.

Equivalent commands from the repository root (after installing the project):

```bash
python scripts/download_data.py --dataset busi
python scripts/check_data.py --dataset busi
python scripts/run_experiment.py --config configs/c_busi_box.yaml
python scripts/analyze_results.py --config configs/c_busi_box.yaml
python scripts/make_failure_cases.py --config configs/c_busi_box.yaml --top 6
```

The runner resumes completed image/configuration blocks. Use `--max-images 2
--exp-id c_busi_smoke` for a quick execution check; never mix smoke output with
`c_busi_box` results. Model inference requires the downloaded official weights and
PyTorch. No GPU scores have been generated or claimed by this change.

BUSI cleaning already implemented by the team removes normal images and byte-identical
duplicates, merges masks and uses the committed image split. The current split is
452 train / 65 validation / 129 test images. It is not a patient-level split because
patient identities are unavailable. BUSI is marked **seen** by MedSAM in the team's
leakage review; do not present it as unseen medical generalization.

## Baseline integration

Use the shared Role B trainer for this dataset; do not implement a separate U-Net:

```bash
python scripts/train_unet.py --config configs/c_busi_box.yaml
python scripts/analyze_results.py --config configs/c_busi_box.yaml
```

Analysis reads `unet.csv` or `classical.csv` if present and records missing baselines
otherwise. A missing baseline means no corresponding crossover is reported.

## Analysis outputs

Outputs live under `<storage-root>/results/summary/c_busi_box/`.
`--root PATH` selects data/results storage; `--out-dir PATH` selects the analysis output.

- `integrity.csv`: expected image/configuration coverage and invalid-row counts.
- `metrics.csv`: seed-averaged, per-image Dice/IoU/NSD/HD95 means, bootstrap intervals
  and metric-specific image counts.
- `dice_drop.csv`: paired Dice decrease from jitter 0 to 0.2, with bootstrap interval.
- `paired_tests.csv`: image-paired Wilcoxon tests; Holm correction across levels
  separately within each model pair and perturbation family.
- `crossover.csv`: first tested jitter level below a baseline plus descriptive linear
  interpolation, using a fixed complete paired image cohort across levels.
- `box_jitter.*`, `box_scale.*`: PNG/PDF performance curves and confidence bands.
- `failures.csv`: seed-averaged image/model/level Dice drops, including improvements
  as negative drops. Missing references are retained as missing values.
- `analysis_status.json`: incomplete runs and missing methods/baselines are explicit.

The statistical unit is the image, not the seed or individual lesion. Invalid rows
are excluded from metrics but counted; missing HD95 values are omitted and counts
are exported. The CLI exits nonzero if the requested model grid is incomplete.
Do not interpret partial-run curves as final results. Confidence intervals do not
establish a statistically significant crossover.

`make_failure_cases.py` reruns the highest-drop cases using one explicitly recorded
seed (default 0). Each panel reports its own recomputed Dice; the ranking uses the
mean across seeds, so those numbers can differ. `failure_cases/review.csv` contains
blank reason/notes columns for manual review. Inspect small target size, low contrast,
clipped targets and ambiguous boundaries; do not infer causes from Dice alone.

## Interactive demo

```bash
pip install -e '.[demo]'
python scripts/demo.py
# UI-only check without weights:
python scripts/demo.py --mock
```

Open the printed local address. Upload an image, adjust the four box-edge sliders
(percent of image dimensions), and press Compare. Both real models receive the same
box. Yellow is the prompt and magenta is the prediction. The demo does not report
Dice without ground truth. Mock mode labels both panels as synthetic. The server
binds locally and does not create a public sharing link.

## Verification

`tests/test_role_c.py` exercises the real BUSI loader and experiment runner on tiny
synthetic fixtures, result integrity checks (including baseline NaN levels), paired
statistics, ranked case rendering, box validation and Gradio app construction.
Synthetic fixture scores must never be included in project results.
