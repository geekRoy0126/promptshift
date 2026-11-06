**MedSAM wrapper vs. official `MedSAM_Inference.py`** (GT boxes)

| image_id | dice_ours_vs_official | pixels_differ | pass |
|---|---|---|---|
| cju0qoxqj9q6s0835b43399p4 | 1.000000 | 0 | True |
| cju0s2a9ekvms080138tjjpxr | 1.000000 | 0 | True |
| cju0tl3uz8blh0993wxvn7ly3 | 1.000000 | 0 | True |

**SAM wrapper vs. direct `SamPredictor`**

| model | images | identical | pixels_differ |
|---|---|---|---|
| sam_vit_b | 3 | True | 0 |

Run on: device = cpu, gpu = none, torch = 2.15.0+cpu, python = 3.11.18, platform = Windows
