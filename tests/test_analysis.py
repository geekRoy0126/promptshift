import math

import numpy as np
import pandas as pd
import pytest

from promptshift.analysis.stats import (
    baseline_summary,
    bootstrap_ci,
    crossover,
    curve,
    delta_dice,
    holm,
    per_image,
    wilcoxon_by_level,
)  # fmt: skip


def toy(model_shift=0.0):
    """3 images x 3 levels x 2 seeds of box_jitter rows."""
    rows = []
    for k, image_id in enumerate(["a", "b", "c"]):
        for level, base in [(0.0, 0.9), (0.2, 0.7), (0.5, 0.3)]:
            for seed in (0, 1):
                rows.append({"image_id": image_id, "perturb": "box_jitter", "level": level,
                             "seed": seed, "flag": "ok",
                             "dice": base - model_shift + 0.01 * k + 0.02 * seed})  # fmt: skip
    return pd.DataFrame(rows)


def test_per_image_averages_seeds_first():
    t = per_image(toy(), "box_jitter")
    assert t.shape == (3, 3)
    assert t.loc[0.0, "a"] == pytest.approx(0.91)


def test_bootstrap_ci_contains_mean_and_is_seeded():
    v = np.random.default_rng(0).normal(0.5, 0.1, 200)
    lo, hi = bootstrap_ci(v, 2000, 42)
    assert lo < v.mean() < hi
    assert (lo, hi) == bootstrap_ci(v, 2000, 42)
    assert all(math.isnan(x) for x in bootstrap_ci([np.nan]))


def test_curve_and_delta():
    c = curve(toy(), "box_jitter", n=200)
    assert list(c["level"]) == [0.0, 0.2, 0.5] and (c["lo"] <= c["mean"]).all()
    d = delta_dice(toy(), 0.0, 0.2, n=200)
    assert d["delta"] == pytest.approx(0.2)
    assert math.isnan(delta_dice(toy(), 0.0, 0.3)["delta"])  # level not run


def test_crossover_interpolates():
    c = crossover([0, 0.1, 0.2, 0.3], [0.9, 0.8, 0.6, 0.4], reference=0.7)
    assert c["first_level"] == 0.2 and c["interp"] == pytest.approx(0.15) and c["label"] == "20%"
    assert crossover([0, 0.5], [0.9, 0.8], 0.5)["label"] == "> 50%"
    assert crossover([0, 0.5], [0.4, 0.3], 0.5)["label"] == "below at 0%"


def test_holm_matches_definition():
    p = [0.01, 0.04, 0.03, np.nan]
    out = holm(p)
    assert out[0] == pytest.approx(0.03) and out[2] == pytest.approx(0.06)
    assert out[1] == pytest.approx(0.06) and math.isnan(out[3])


def test_wilcoxon_against_model_and_baseline():
    a, b = per_image(toy(), "box_jitter"), per_image(toy(0.1), "box_jitter")
    w = wilcoxon_by_level(a, b)
    assert list(w.columns) == ["level", "n", "mean_diff", "p", "p_holm"]
    assert w["mean_diff"].to_numpy() == pytest.approx([0.1, 0.1, 0.1])
    base = pd.DataFrame({"image_id": ["a", "b", "c"], "perturb": "none", "flag": "ok",
                         "dice": [0.5, 0.6, 0.7]})  # fmt: skip
    assert baseline_summary(base, n=100)["mean"] == pytest.approx(0.6)
    w2 = wilcoxon_by_level(a, base.set_index("image_id")["dice"])
    assert len(w2) == 3 and (w2["n"] == 3).all()
