import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from promptshift.data import get_dataset
from promptshift.data.busi import Busi
from promptshift.data.isic import GT_DIR, IMG_DIR, INDEX_FILE, Isic2018
from promptshift.data.remote_zip import RangeFile

CFG = {"mask_threshold": 127, "min_object_frac": 0.001, "connectivity": 8}


def save_png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def blob(h=40, w=50, box=(10, 10, 30, 30)):
    m = np.zeros((h, w), np.uint8)
    y0, x0, y1, x1 = box
    m[y0:y1, x0:x1] = 255
    return m


# --- remote zip ---------------------------------------------------------------


def test_range_file_reads_zip_members():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a/x.txt", b"hello")
        z.writestr("a/y.bin", bytes(range(256)) * 100)
    data = buf.getvalue()
    f = RangeFile(len(data), lambda s, e: data[s : e + 1])
    with zipfile.ZipFile(io.BufferedReader(f, 64)) as z:
        assert z.namelist() == ["a/x.txt", "a/y.bin"]
        assert z.read("a/x.txt") == b"hello"
        assert z.read("a/y.bin") == bytes(range(256)) * 100
    assert f.requests > 1


# --- ISIC ---------------------------------------------------------------------


def make_isic(tmp_path, ids, with_images=(), index=None):
    root = tmp_path / "isic2018"
    for i in ids:
        save_png(root / GT_DIR / f"{i}_segmentation.png", blob())
    for i in with_images:
        save_png(root / IMG_DIR / f"{i}.jpg", np.full((40, 50, 3), 100, np.uint8))
    if index is not None:
        names = [f"{IMG_DIR}/{i}.jpg" for i in index] + [f"{IMG_DIR}/ATTRIBUTION.txt"]
        (root / INDEX_FILE).write_text(json.dumps(names))
    return tmp_path


def test_isic_pairs_from_index_and_mask_sample(tmp_path):
    ids = ["ISIC_0000000", "ISIC_0000001", "ISIC_0000003"]
    data = make_isic(tmp_path, ids, index=ids[:2] + ["ISIC_0000009"])
    ds = Isic2018(data)
    assert ds.all_ids() == ids
    assert ds.problems == {
        "image_without_mask": ["ISIC_0000009"], "mask_without_image": ["ISIC_0000003"],
    }  # fmt: skip
    s = ds.mask_sample("ISIC_0000000")
    assert s.image is None and len(s.objects) == 1 and s.mask.sum() == 400
    lines, ok = ds.count_check()
    assert not ok and "official: 2594" in lines[0]
    with pytest.raises(FileNotFoundError):
        ds.get("ISIC_0000000")  # image not downloaded


def test_isic_get_with_local_images(tmp_path):
    ids = ["ISIC_0000000", "ISIC_0000001"]
    ds = get_dataset("isic2018", make_isic(tmp_path, ids, with_images=ids), CFG)
    assert ds.image_source == "local folder" and not any(ds.problems.values())
    s = ds.get("ISIC_0000001")
    assert s.image.shape == (40, 50, 3) and s.mask.shape == (40, 50)


# --- BUSI ---------------------------------------------------------------------


def make_busi(tmp_path):
    root = tmp_path / "busi" / "Dataset_BUSI_with_GT"
    img = lambda v: np.full((40, 50, 3), v, np.uint8)  # noqa: E731
    save_png(root / "benign" / "benign (1).png", img(10))
    save_png(root / "benign" / "benign (1)_mask.png", blob(box=(0, 0, 10, 10)))
    save_png(root / "benign" / "benign (1)_mask_1.png", blob(box=(20, 20, 30, 40)))
    save_png(root / "benign" / "benign (2).png", img(20))
    save_png(root / "benign" / "benign (2)_mask.png", blob())
    save_png(root / "malignant" / "malignant (1).png", img(20))  # duplicate of benign (2)
    save_png(root / "malignant" / "malignant (1)_mask.png", blob())
    save_png(root / "malignant" / "malignant (2).png", img(30))
    save_png(root / "malignant" / "malignant (2)_mask.png", blob())
    save_png(root / "normal" / "normal (1).png", img(40))
    save_png(root / "normal" / "normal (1)_mask.png", np.zeros((40, 50), np.uint8))
    return tmp_path


def test_busi_drop_normal_dedup_and_union(tmp_path):
    ds = Busi(make_busi(tmp_path))
    assert ds.all_ids() == ["benign_0001", "benign_0002", "malignant_0002"]
    assert ds.duplicates == {"malignant_0001": "benign_0002"}
    assert ds.class_counts == {"benign": 2, "malignant": 2, "normal": 1}
    assert ds.dropped_normal == 1 and ds.n_multi_mask == 1
    s = ds.get("benign_0001")
    assert len(s.objects) == 2 and s.mask.sum() == 100 + 200  # union of two mask files
    assert s.meta["n_mask_files"] == 2 and s.meta["class"] == "benign"
    lines, ok = ds.count_check()
    assert not ok  # toy counts differ from the official ones
    assert any("duplicate" in line and "malignant_0001 = benign_0002" in line for line in lines)


def test_busi_keep_normal_via_config(tmp_path):
    ds = get_dataset("busi", make_busi(tmp_path), {**CFG, "busi": {"drop_normal": False}})
    assert "normal_0001" in ds.all_ids() and ds.dropped_normal == 0
    assert len(ds.get("normal_0001").objects) == 0
