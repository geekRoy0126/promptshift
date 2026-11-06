import inspect
import math
from itertools import product

import numpy as np
import pytest
import yaml

from promptshift.paths import REPO_ROOT
from promptshift.prompts import PERTURBATIONS, box_from_mask, make_prompt
from promptshift.prompts.box import box_iou, jitter_box, scale_box
from promptshift.prompts.debug import plot_debug_prompts, synthetic_masks
from promptshift.prompts.point import object_distance_transform

CFG = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
PCFG = CFG["prompts"]
MASKS = synthetic_masks()


def prompt(perturb, level, mask, seed=0, image_id="img", object_idx=0):
    return make_prompt(
        perturb, level, mask, global_seed=42, dataset="test", image_id=image_id,
        object_idx=object_idx, seed=seed, params=PCFG[perturb],
    )


def random_masks(n=6, shape=(90, 120)):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        m = np.zeros(shape, dtype=bool)
        y0, x0 = rng.integers(0, shape[0] - 10), rng.integers(0, shape[1] - 10)
        hh, ww = rng.integers(5, shape[0] - y0 + 1), rng.integers(5, shape[1] - x0 + 1)
        m[y0 : y0 + hh, x0 : x0 + ww] = True
        out.append(m)
    return out + list(MASKS.values())


ALL_GRID = [
    (perturb, level, seed)
    for perturb in PERTURBATIONS
    for level, seed in product(PCFG[perturb]["levels"], PCFG[perturb]["seeds"])
]


# --- same key -> same prompt ---------------------------------------------------


@pytest.mark.parametrize("perturb,level,seed", ALL_GRID)
def test_same_key_same_prompt(perturb, level, seed):
    mask = MASKS["c_shape"]
    a, b = prompt(perturb, level, mask, seed), prompt(perturb, level, mask, seed)
    if a.kind == "box":
        np.testing.assert_array_equal(a.box, b.box)
    else:
        np.testing.assert_array_equal(a.coords, b.coords)


def test_different_seed_or_object_gives_different_jitter():
    mask = MASKS["c_shape"]
    base = prompt("box_jitter", 0.2, mask, seed=0).box
    assert not np.array_equal(base, prompt("box_jitter", 0.2, mask, seed=1).box)
    assert not np.array_equal(base, prompt("box_jitter", 0.2, mask, object_idx=1).box)


def test_level_spelling_does_not_change_prompt():
    mask = MASKS["c_shape"]
    np.testing.assert_array_equal(
        prompt("box_jitter", 0.10, mask).box, prompt("box_jitter", 0.1000, mask).box
    )


def test_prompt_does_not_depend_on_model():
    assert not any("model" in p for p in inspect.signature(make_prompt).parameters)


# --- r = 0 and s = 1.0 give the GT box -----------------------------------------


@pytest.mark.parametrize("mask", random_masks())
def test_identity_levels_return_gt_box(mask):
    gt = box_from_mask(mask)
    for seed in PCFG["box_jitter"]["seeds"]:
        p = prompt("box_jitter", 0.0, mask, seed)
        np.testing.assert_array_equal(p.box, gt)
        assert p.valid and p.box_iou == 1.0
    p = prompt("box_scale", 1.0, mask)
    np.testing.assert_array_equal(p.box, gt)
    assert p.valid and p.box_iou == 1.0


def test_box_from_mask_is_inclusive():
    m = np.zeros((10, 12), dtype=bool)
    m[2:5, 3:9] = True
    np.testing.assert_array_equal(box_from_mask(m), [3, 2, 8, 4])


# --- every box valid and inside the image --------------------------------------


@pytest.mark.parametrize("mask", random_masks())
def test_boxes_valid_and_inside_image(mask):
    h, w = mask.shape
    for perturb in ("box_jitter", "box_scale"):
        cfg = PCFG[perturb]
        for level, seed in product(cfg["levels"], cfg["seeds"]):
            p = prompt(perturb, level, mask, seed)
            assert p.valid, (perturb, level, seed)
            x0, y0, x1, y1 = p.box
            assert 0 <= x0 < x1 <= w - 1 and 0 <= y0 < y1 <= h - 1
            assert x1 - x0 >= cfg["min_box_side"] and y1 - y0 >= cfg["min_box_side"]
            assert 0 < p.box_iou <= 1


def test_jitter_stays_within_r_of_each_edge():
    gt = np.array([100.0, 80.0, 200.0, 140.0])  # w = 100, h = 60
    rng = np.random.default_rng(3)
    for _ in range(200):
        p = jitter_box(gt, 0.3, rng, width=1000, height=1000)
        d = np.abs(p.box - gt)
        assert d[0] <= 30 and d[2] <= 30 and d[1] <= 18 and d[3] <= 18


def test_scale_box_iou_analytic():
    gt = np.array([100.0, 100.0, 200.0, 150.0])
    assert scale_box(gt, 0.8, 1000, 1000).box_iou == pytest.approx(0.64)
    assert scale_box(gt, 1.25, 1000, 1000).box_iou == pytest.approx(1 / 1.5625)


def test_scale_clips_to_image():
    gt = np.array([0.0, 0.0, 49.0, 29.0])
    p = scale_box(gt, 1.5, width=50, height=30)
    np.testing.assert_array_equal(p.box, [0, 0, 49, 29])


# --- invalid prompts -----------------------------------------------------------


def test_thin_object_gives_invalid_jitter_and_scale():
    m = np.zeros((50, 50), dtype=bool)
    m[10:40, 20] = True  # one column wide: box width 0 < 2
    j = prompt("box_jitter", 0.3, m)
    assert not j.valid and math.isnan(j.box_iou)
    assert j.n_resample == PCFG["box_jitter"]["max_resample"]
    s = prompt("box_scale", 1.5, m)
    assert not s.valid and math.isnan(s.box_iou)


def test_invalid_jitter_is_resampled():
    # Narrow box at the left border: large jitter often collapses x, resampling fixes it.
    gt = np.array([0.0, 0.0, 3.0, 40.0])
    resampled = 0
    for seed in range(200):
        p = jitter_box(gt, 0.5, np.random.default_rng(seed), width=100, height=100)
        if p.valid:
            assert p.box[2] - p.box[0] >= 2
        resampled += p.n_resample > 0
    assert resampled > 0


# --- clicks always inside the object -------------------------------------------


@pytest.mark.parametrize("mask", random_masks())
def test_clicks_inside_mask(mask):
    cfg = PCFG["point_offset"]
    for level, seed in product(cfg["levels"], cfg["seeds"]):
        p = prompt("point_offset", level, mask, seed)
        x, y = p.coords[0].astype(int)
        assert mask[y, x], (level, seed)
        assert p.labels.tolist() == [1]


def test_center_click_is_dt_argmax_not_centroid():
    mask = MASKS["c_shape"]
    ys, xs = np.nonzero(mask)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    assert not mask[cy, cx]  # the centroid falls outside the C shape
    p = prompt("point_offset", 0.0, mask)
    x, y = p.coords[0].astype(int)
    dt = object_distance_transform(mask)
    assert dt[y, x] == dt.max()
    assert p.depth == 1.0


def test_boundary_click_and_band_depths():
    mask = MASKS["c_shape"]
    dt = object_distance_transform(mask)
    x, y = prompt("point_offset", 1.0, mask).coords[0].astype(int)
    assert dt[y, x] <= 1.0
    for d in (0.25, 0.5, 0.75):
        p = prompt("point_offset", d, mask)
        assert not p.fallback
        assert abs(p.depth - (1 - d)) <= PCFG["point_offset"]["band"] + 1e-12


def test_image_border_counts_as_boundary():
    mask = MASKS["border_ellipse"]
    dt = object_distance_transform(mask)
    assert dt[mask[:, 0], 0].max() == 1.0  # pixels on the left image edge
    x, _ = prompt("point_offset", 0.0, mask).coords[0]
    assert x > 0


def test_empty_band_falls_back_to_closest_pixel():
    m = np.zeros((9, 9), dtype=bool)
    m[3:6, 3:6] = True  # padded DT: 1 on the ring, 2 at the center -> depths {0.5, 1}
    p = prompt("point_offset", 0.25, m)  # target depth 0.75: band empty
    assert p.fallback
    x, y = p.coords[0].astype(int)
    assert m[y, x]


# --- debug figure --------------------------------------------------------------


def test_debug_figure(tmp_path):
    out = plot_debug_prompts(tmp_path / "debug_prompts.png", PCFG)
    assert out.is_file() and out.stat().st_size > 10_000


def test_box_iou_of_disjoint_boxes_is_zero():
    assert box_iou(np.array([0, 0, 10, 10.0]), np.array([20, 20, 30, 30.0])) == 0.0
