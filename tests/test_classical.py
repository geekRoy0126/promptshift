import math

import numpy as np

from promptshift.baselines import baseline_row
from promptshift.baselines.classical import (
    ClassicalParams,
    grid,
    largest_component,
    segment,
    segment_grid,
)  # fmt: skip
from promptshift.data.base import Sample
from promptshift.runner import COLUMNS


def scene():
    """Bright background, dark square object, plus a small dark speck."""
    img = np.full((80, 100, 3), 200, np.uint8)
    img[20:50, 30:70] = 40
    img[70:73, 5:8] = 40
    gt = np.zeros((80, 100), bool)
    gt[20:50, 30:70] = True
    return img, gt


def test_segment_recovers_dark_object():
    img, gt = scene()
    pred = segment(img, ClassicalParams("lab_L", "low", 1, 2))
    assert (pred & gt).sum() / gt.sum() > 0.95
    assert pred[70:73, 5:8].sum() == 0  # speck removed (largest component only)
    # wrong polarity -> background; closing only rounds the square's corners into it
    assert segment(img, ClassicalParams("lab_L", "high", 1, 2))[gt].sum() < 0.01 * gt.sum()


def test_largest_component_tie_and_empty():
    m = np.zeros((10, 10), bool)
    m[0:2, 0:2] = True
    m[5:7, 5:7] = True  # same area: the first label wins
    out = largest_component(m)
    assert out[0, 0] and not out[5, 5]
    assert not largest_component(np.zeros((5, 5), bool)).any()


def test_grid_and_segment_grid_match_segment():
    cfg = {"channel": ["lab_L", "hsv_v"], "polarity": ["high", "low"], "sigma": [1, 2],
           "radius": [2]}  # fmt: skip
    params = grid(cfg)
    assert len(params) == 8
    img, _ = scene()
    out = segment_grid(img, params)
    for p in params:
        assert np.array_equal(out[p], segment(img, p))


def test_baseline_row_format():
    img, gt = scene()
    s = Sample("x", img, gt, [gt], "toy")
    row = baseline_row(exp_id="e", split="test", method="classical", sample=s, pred=gt, ms=1.0,
                       sha="abc", nsd_tolerance_px=2)  # fmt: skip
    assert set(row) == set(COLUMNS)
    assert row["perturb"] == "none" and math.isnan(row["level"]) and row["dice"] == 1.0
