# Role C: BUSI click protocol

This extension reuses the team's data loaders, mask cleaning, split, point-offset
simulator, SAM wrappers and metrics. It does not retrain SAM or MedSAM.

Run `notebooks/C_02_busi_clicks.ipynb` on Colab T4. The notebook defaults to the
feature branch until merged. Dataset/weights/results persist in the existing
project Drive directory. Full run: 129 BUSI test images, SAM ViT-B and ViT-H,
33 rows/image/model (4257 each; 8514 total). A separate three-image smoke run
runs first. MedSAM is excluded because the supplied wrapper is box-only.

## Prompt and refinement protocol

- Single positive click per object at offsets 0, .25, .5, .75, 1 from the deepest
  interior region towards the inner boundary; three deterministic seeds.
- Correction round 0 starts with the exact offset-.5 point for the same image,
  object and seed. Both models receive identical initial points.
- Rounds 1–5 add at most one click per object. Choose the largest false-negative
  or false-positive connected component (8-connectivity), then its deepest unused
  pixel. Ties prefer false negatives, then row-major order. FN clicks are positive;
  FP clicks are negative. Later clicks are adaptive and may differ across models.
- All previous points/labels and previous low-resolution logits are retained.
  A perfect prediction, or an error with no unused pixels, keeps its last mask.
  The trace records actual clicks; later levels do not imply fabricated clicks.
- Object predictions are unioned and evaluated at the image level. Round count is
  per object, not a total click budget per image. BUSI has 130 objects in 129 images.
- Ground truth is an oracle for simulated user clicks. Recovery is not guaranteed
  to be monotonic and is not an evaluation of real human interaction.

## Integrity and analysis

`run_click_experiment.py` saves a protocol/cohort fingerprint and atomic per-image
CSV blocks plus point traces. Resume repairs incomplete image blocks and rejects
changed protocols under the same experiment ID. Keep raw files together.
`analyze_click_results.py` rejects missing split IDs, duplicate keys, missing grid
entries, bad statuses and smoke runs. Seeds are averaged within images before
bootstrap confidence intervals or paired Wilcoxon tests. Recovery intervals use
paired per-image changes. Holm adjustment is within each model-pair/perturbation
family. Separate datasets must not be treated as paired images.

```bash
python scripts/run_click_experiment.py --config configs/c_busi_clicks.yaml --smoke-images 3
python scripts/run_click_experiment.py --config configs/c_busi_clicks.yaml
python scripts/analyze_click_results.py --config configs/c_busi_clicks.yaml
```

No click scores are claimed until a complete real-model run is validated.
