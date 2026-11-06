import numpy as np
import pytest

from promptshift.metrics import dice
from promptshift.models import MockSegmenter, load_model
from promptshift.paths import get_paths

WEIGHTS = get_paths().weights


def has(fname):
    return (WEIGHTS / fname).is_file()


def test_mock_box_and_points():
    m = load_model("mock", WEIGHTS)
    m.set_image(np.zeros((60, 80, 3), np.uint8))
    mask = m.predict_box(np.array([10, 10, 50, 40.0]))
    assert mask.shape == (60, 80) and mask.dtype == bool
    assert mask[25, 30] and not mask[0, 0] and not mask[55, 75]
    pm, logits = m.predict_points(np.array([[30.0, 25.0]]), np.array([1]))
    assert pm[25, 30] and logits.ndim == 2


def test_mock_requires_set_image():
    with pytest.raises(RuntimeError):
        MockSegmenter().predict_box(np.array([0, 0, 5, 5.0]))


def test_unknown_model():
    with pytest.raises(KeyError):
        load_model("nope", WEIGHTS)


def test_missing_weights_message(tmp_path):
    pytest.importorskip("segment_anything")
    with pytest.raises(FileNotFoundError, match="download_weights"):
        load_model("medsam", tmp_path, "cpu")


def test_medsam_preprocess_matches_official_steps():
    pytest.importorskip("skimage")
    from promptshift.models.medsam import preprocess

    rng = np.random.default_rng(0)
    img = rng.integers(10, 200, size=(37, 53, 3), dtype=np.uint8)
    out = preprocess(img)
    assert out.shape == (1024, 1024, 3)
    assert out.min() == 0.0 and out.max() == 1.0
    gray = preprocess(img[:, :, 0])
    assert gray.shape == (1024, 1024, 3)
    np.testing.assert_array_equal(gray[..., 0], gray[..., 2])


# --- real weights (slow; skipped when weights are missing) -----------------------


def blob_image():
    yy, xx = np.mgrid[:240, :320]
    mask = ((yy - 120) / 60.0) ** 2 + ((xx - 170) / 80.0) ** 2 < 1
    img = np.full((240, 320, 3), 40, np.uint8)
    img[mask] = (220, 120, 90)
    return img, mask


@pytest.mark.slow
@pytest.mark.skipif(not has("sam_vit_b_01ec64.pth"), reason="sam_vit_b weights missing")
def test_sam_wrapper_identical_to_direct_predictor():
    import torch
    from segment_anything import SamPredictor

    img, gt = blob_image()
    m = load_model("sam_vit_b", WEIGHTS, "cpu")
    box = np.array([90.0, 60.0, 250.0, 180.0])
    m.set_image(img)
    ours = m.predict_box(box)
    direct = SamPredictor(m.model)
    with torch.no_grad():
        direct.set_image(img)
        ref, _, _ = direct.predict(box=box[None], multimask_output=False)
    np.testing.assert_array_equal(ours, ref[0])
    assert dice(ours, gt) > 0.9
    pm, logits = m.predict_points(np.array([[170.0, 120.0]]), np.array([1]))
    assert pm[120, 170] and logits.shape == (256, 256)


@pytest.mark.slow
@pytest.mark.skipif(not has("medsam_vit_b.pth"), reason="medsam weights missing")
def test_medsam_runs_on_simple_blob():
    img, gt = blob_image()
    m = load_model("medsam", WEIGHTS, "cpu")
    m.set_image(img)
    pred = m.predict_box(np.array([90.0, 60.0, 250.0, 180.0]))
    assert pred.shape == gt.shape and dice(pred, gt) > 0.8
