# Progress

## 2026-11-05 (A, with Claude Code)

- **Current step**: A tasks P1-1 … P1-7 and the A part of P1-14 done; stopped before P1-8 (GitHub URL needed).
- **Next step**: P1-8 notebooks P1_00–P1_03 (needs GitHub URL + `GH_TOKEN` Colab secret), then C1–C5 on Colab, then P1-9 pilot.
- **Blockers**: GitHub URL (P1-8). TA confirmation of allowed AI use (2026-11-06). Open: U-Net / classical CSV paths (B, C); COCO instance = one object (team). Outer `D:5330CLAUDE.md` is outdated (not touched).
- **Environment**: Miniconda installed per-user at `C:\Users\ruoyu\miniconda3` (2026-11-05, via winget). Env `promptshift` = Python 3.11.18 from conda-forge (`--override-channels -c conda-forge`, avoids the Anaconda default-channel ToS). `pip install -e ".[dev]"` done.
- **Done**: P1-1 (`ruff check .` clean, `pytest -q` 15 passed).
- **Done**: P1-2 (`metrics.py`; matches MONAI 1.6.1; local env now also has torch 2.15.0+cpu and monai 1.6.1).
- **Done**: P1-3 (`prompts/`: box_jitter, box_scale, point_offset + registry; `figures/debug_prompts.png` from synthetic masks).
- **Done**: P1-4 (Kvasir-SEG downloaded locally to `data/kvasir_seg/`, 1000 pairs; `splits/kvasir_seg.json` 700/100/200; `docs/data_stats.md`; `figures/data_kvasir_seg_samples.png`).
- **Done**: P1-5 COCO part (annotations only locally; 1000 instances sampled; `splits/coco_val2017.json`; 5-sample figure). ISIC (B) and BUSI (C) still open.
- **Done**: P1-6 (SAM / MedSAM / mock wrappers; MedSAM parity Dice 1.0 on 3 images and SAM ViT-B pixel-identical, both on local CPU; ViT-H + GPU parity pending on Colab).
- **Done**: P1-7 (runner + resume; local CPU smoke 5 images × sam_vit_b + medsam, 120/120 rows each). **Next: P1-8 needs the GitHub URL from the human.**
- **Done**: P1-14 A part (`scripts/make_report_tables.py`, `reports/phase1_report.template.md` → generated `reports/phase1_report.md`; pilot sections TBD).
