import json

import numpy as np
import pytest
from PIL import Image

from promptshift.data import get_dataset
from promptshift.data.base import split_objects
from promptshift.data.kvasir import KvasirSeg
from promptshift.data.splits import load_split, make_split, save_split
from promptshift.paths import REPO_ROOT

DATA_CFG = {"mask_threshold": 127, "min_object_frac": 0.001, "connectivity": 8}


# --- mask cleaning -------------------------------------------------------------


def test_split_objects_removes_specks_and_keeps_order():
    m = np.zeros((100, 100), dtype=bool)
    m[5:15, 60:80] = True      # object A (first in row-major order)
    m[50:70, 10:30] = True     # object B
    m[90, 90] = True           # speck: 1 px < 0.001 * 10000 = 10 px
    m[95:97, 2:5] = True       # speck: 6 px
    c = split_objects(m, 0.001, 8)
    assert c.n_components == 4 and c.n_specks == 2 and c.speck_pixels == 7
    assert len(c.objects) == 2
    assert c.objects[0][10, 70] and c.objects[1][60, 20]
    assert not c.mask[90, 90] and c.mask.sum() == 200 + 400


def test_diagonal_pixels_are_one_object_with_8_connectivity():
    m = np.zeros((40, 40), dtype=bool)
    m[0:10, 0:10] = True
    m[10:20, 10:20] = True     # touches only at a corner
    assert len(split_objects(m, 0.0, 8).objects) == 1
    assert len(split_objects(m, 0.0, 4).objects) == 2


def test_object_exactly_at_threshold_is_kept():
    m = np.zeros((100, 100), dtype=bool)
    m[0:2, 0:5] = True         # 10 px == 0.001 * 10000
    assert len(split_objects(m, 0.001).objects) == 1


# --- splits --------------------------------------------------------------------


def test_split_sizes_deterministic_and_disjoint():
    ids = [f"id{i:04d}" for i in range(1000)]
    a = make_split(ids, {"train": 0.7, "val": 0.1, "test": 0.2}, seed=42)
    b = make_split(list(reversed(ids)), {"train": 0.7, "val": 0.1, "test": 0.2}, seed=42)
    assert a == b
    assert [len(a[s]) for s in ("train", "val", "test")] == [700, 100, 200]
    assert set(a["train"]) | set(a["val"]) | set(a["test"]) == set(ids)
    assert not set(a["train"]) & set(a["test"])


def test_split_roundtrip_and_overlap_check(tmp_path):
    split = make_split([f"x{i}" for i in range(20)], {"train": 0.7, "val": 0.1, "test": 0.2}, 1)
    path = tmp_path / "s.json"
    save_split(path, "toy", split, 1, {"train": 0.7, "val": 0.1, "test": 0.2})
    assert load_split(path) == split
    bad = json.loads(path.read_text())
    bad["test"].append(bad["train"][0])
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="more than one split"):
        load_split(path)


# --- Kvasir loader on a synthetic folder ---------------------------------------


@pytest.fixture
def fake_kvasir(tmp_path):
    root = tmp_path / "kvasir_seg" / "Kvasir-SEG"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    rng = np.random.default_rng(0)
    for i in range(10):
        h, w = 60 + i, 80
        img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:40, 20:50] = 255
        mask[h - 1, w - 1] = 255                     # one-pixel speck
        Image.fromarray(img).save(root / "images" / f"img{i}.jpg", quality=95)
        Image.fromarray(mask).save(root / "masks" / f"img{i}.jpg", quality=75)  # JPEG noise
    Image.fromarray(np.zeros((5, 5, 3), np.uint8)).save(root / "images" / "orphan.jpg")
    return tmp_path


def test_kvasir_loader_binarizes_jpeg_masks(fake_kvasir):
    ds = KvasirSeg(fake_kvasir, **DATA_CFG)
    assert len(ds.all_ids()) == 10
    assert ds.problems["image_without_mask"] == ["orphan"]
    s = ds.get("img3")
    assert s.image.dtype == np.uint8 and s.image.shape == (63, 80, 3)
    assert s.mask.dtype == bool and s.mask.shape == (63, 80)
    assert len(s.objects) == 1 and s.meta["n_specks_removed"] >= 1
    # JPEG ringing must not leak far outside the 30x30 square after thresholding
    assert abs(int(s.mask.sum()) - 900) < 60
    assert s.dataset == "kvasir_seg"


def test_kvasir_split_ids(fake_kvasir, tmp_path):
    ds = KvasirSeg(fake_kvasir, **DATA_CFG)
    split = make_split(ds.all_ids(), {"train": 0.7, "val": 0.1, "test": 0.2}, 42)
    path = tmp_path / "split.json"
    save_split(path, "kvasir_seg", split, 42, {"train": 0.7, "val": 0.1, "test": 0.2})
    ds2 = get_dataset("kvasir_seg", fake_kvasir, DATA_CFG, path)
    assert ds2.ids("test") == split["test"]
    with pytest.raises(ValueError):
        ds2.ids("bogus")


# --- committed split (real file) -----------------------------------------------


def test_committed_kvasir_split():
    path = REPO_ROOT / "splits" / "kvasir_seg.json"
    if not path.exists():
        pytest.skip("splits/kvasir_seg.json not generated yet")
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = load_split(path)
    assert payload["seed"] == 42
    n = sum(len(v) for v in split.values())
    assert len(split["train"]) == round(0.7 * n) and len(split["val"]) == round(0.1 * n)


@pytest.mark.slow
def test_real_kvasir_counts():
    from promptshift.paths import get_paths

    try:
        ds = KvasirSeg(get_paths().data, **DATA_CFG)
    except FileNotFoundError:
        pytest.skip("Kvasir-SEG not downloaded")
    assert len(ds.all_ids()) == 1000
    assert not any(ds.problems.values())
