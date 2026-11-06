# Progress

## 2026-11-05 (A, with Claude Code)

- **Current step**: P1-4 Kvasir loader + split + statistics (done).
- **Next step**: P1-5 COCO loader + download verification (A part), then P1-6 model wrappers.
- **Blockers**: TA confirmation of the allowed AI-use level (2026-11-06). No official Phase 1 spec yet (defaults apply).
- **Environment**: Miniconda installed per-user at `C:\Users\ruoyu\miniconda3` (2026-11-05, via winget). Env `promptshift` = Python 3.11.18 from conda-forge (`--override-channels -c conda-forge`, avoids the Anaconda default-channel ToS). `pip install -e ".[dev]"` done.
- **Done**: P1-1 (`ruff check .` clean, `pytest -q` 15 passed).
- **Done**: P1-2 (`metrics.py`; matches MONAI 1.6.1; local env now also has torch 2.15.0+cpu and monai 1.6.1).
- **Done**: P1-3 (`prompts/`: box_jitter, box_scale, point_offset + registry; `figures/debug_prompts.png` from synthetic masks).
- **Done**: P1-4 (Kvasir-SEG downloaded locally to `data/kvasir_seg/`, 1000 pairs; `splits/kvasir_seg.json` 700/100/200; `docs/data_stats.md`; `figures/data_kvasir_seg_samples.png`).
- **Done**: P1-5 COCO part (annotations only locally; 1000 instances sampled; `splits/coco_val2017.json`; 5-sample figure). ISIC (B) and BUSI (C) still open.
- **Done**: P1-6 (SAM / MedSAM / mock wrappers; MedSAM parity Dice 1.0 on 3 images and SAM ViT-B pixel-identical, both on local CPU; ViT-H + GPU parity pending on Colab).
