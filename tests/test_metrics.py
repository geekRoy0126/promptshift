import math

import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial.distance import cdist

from promptshift.metrics import compute_metrics, dice, hd95, iou, nsd, surface


def square(shape=(40, 40), top=10, left=10, side=10):
    m = np.zeros(shape, dtype=bool)
    m[top : top + side, left : left + side] = True
    return m


def blob(seed, shape=(48, 64)):
    rng = np.random.default_rng(seed)
    smooth = ndimage.gaussian_filter(rng.normal(size=shape), sigma=4)
    return smooth > np.quantile(smooth, 0.7)


# --- required by CLAUDE.md 6.4 -------------------------------------------------


def test_identical_masks():
    m = square()
    assert dice(m, m) == 1.0
    assert iou(m, m) == 1.0
    assert nsd(m, m, 2) == 1.0
    assert hd95(m, m) == 0.0


def test_disjoint_masks():
    a, b = square(left=0), square(left=25)
    assert dice(a, b) == 0.0
    assert iou(a, b) == 0.0
    assert nsd(a, b, 2) == 0.0  # surfaces are > 2 px apart


def test_offset_squares_analytic():
    # 10x10 squares shifted by 5 px: overlap 5x10 = 50 px
    a, b = square(left=10), square(left=15)
    assert dice(a, b) == pytest.approx(2 * 50 / 200)
    assert iou(a, b) == pytest.approx(50 / 150)
    # 10 of the 36 surface pixels of each square are exactly 5 px from the other
    # surface and none is farther, so both directed 95th percentiles are 5.
    assert hd95(a, b) == pytest.approx(5.0)


def test_one_pixel_shift_within_tolerance():
    a, b = square(left=10), square(left=11)
    assert hd95(a, b) == pytest.approx(1.0)
    assert nsd(a, b, tolerance_px=2) == 1.0
    assert nsd(a, b, tolerance_px=0) < 1.0


# --- empty-mask conventions (docs/decisions.md) --------------------------------


def test_both_empty():
    z = np.zeros((8, 8), dtype=bool)
    assert dice(z, z) == iou(z, z) == nsd(z, z) == 1.0
    assert math.isnan(hd95(z, z))


def test_one_empty():
    z, m = np.zeros((40, 40), dtype=bool), square()
    for p, g in [(z, m), (m, z)]:
        assert dice(p, g) == iou(p, g) == nsd(p, g) == 0.0
        assert math.isnan(hd95(p, g))


def test_compute_metrics_flags_empty_prediction():
    out = compute_metrics(np.zeros((40, 40), bool), square(), nsd_tolerance_px=2)
    assert out["pred_empty"] is True
    assert math.isnan(out["hd95"])
    assert out["pred_area_frac"] == 0.0
    assert out["gt_area_frac"] == pytest.approx(100 / 1600)


# --- implementation checks -----------------------------------------------------


def test_surface_of_square_is_its_perimeter():
    s = surface(square())
    assert s.sum() == 36
    assert not s[11:19, 11:19].any()


def test_object_touching_image_border_has_surface_there():
    m = np.zeros((10, 10), dtype=bool)
    m[:, :4] = True
    assert surface(m)[:, 0].all()


@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force_surface_distances(seed):
    p, g = blob(seed), blob(seed + 100)
    sp = np.argwhere(surface(p))
    sg = np.argwhere(surface(g))
    d = cdist(sp, sg)
    d_pg, d_gp = d.min(axis=1), d.min(axis=0)
    tol = 2.0
    expected_nsd = ((d_pg <= tol).sum() + (d_gp <= tol).sum()) / (len(d_pg) + len(d_gp))
    expected_hd95 = max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))
    assert nsd(p, g, tol) == pytest.approx(expected_nsd)
    assert hd95(p, g) == pytest.approx(expected_hd95)


def test_nsd_increases_with_tolerance():
    p, g = blob(1), blob(2)
    values = [nsd(p, g, t) for t in (0, 1, 2, 4, 8)]
    assert values == sorted(values)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dice(np.zeros((4, 4)), np.zeros((4, 5)))


def test_accepts_non_bool_input():
    m = square().astype(np.uint8) * 255
    assert dice(m, square()) == 1.0


# --- optional cross-check against MONAI ---------------------------------------


def border_pair():
    # both objects touch the left image border (common in Kvasir)
    p = np.zeros((40, 50), dtype=bool)
    p[:25, :20] = True
    g = np.zeros((40, 50), dtype=bool)
    g[3:30, :24] = True
    return p, g


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("case", ["blob0", "blob1", "blob2", "border"])
def test_cross_check_monai(case):
    monai_metrics = pytest.importorskip("monai.metrics")
    torch = pytest.importorskip("torch")
    if case == "border":
        p, g = border_pair()
    else:
        seed = int(case[-1])
        p, g = blob(seed), blob(seed + 100)
    pt = torch.from_numpy(p[None, None].astype(np.float32))
    gt = torch.from_numpy(g[None, None].astype(np.float32))
    m_hd95 = monai_metrics.compute_hausdorff_distance(pt, gt, percentile=95).item()
    m_nsd = monai_metrics.compute_surface_dice(pt, gt, class_thresholds=[2.0]).item()
    # MONAI returns float32
    assert hd95(p, g) == pytest.approx(m_hd95, abs=1e-5)
    assert nsd(p, g, 2.0) == pytest.approx(m_nsd, abs=1e-5)
