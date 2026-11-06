| method | images | Dice [CI] | IoU | NSD (2 px) | HD95 (px) | empty predictions |
|---|---|---|---|---|---|---|
| classical | 200 | 0.364 [0.329, 0.399] | 0.257 | 0.073 | 265.309 | 0 |

unet: TBD (not run yet).

HD95 means exclude empty predictions (NaN); their count is the last column.

Classical parameters (chosen by max mean Dice on train (700 images); test evaluated once): channel `lab_L`, object = `high` side of the Otsu threshold, Gaussian sigma = 1, open/close radius = 10 px; train Dice 0.351, val Dice 0.332. Grid: {'channel': ['lab_L', 'lab_a', 'lab_b', 'hsv_s', 'hsv_v'], 'polarity': ['high', 'low'], 'sigma': [1, 2, 4], 'radius': [2, 5, 10]}.
