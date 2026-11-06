import json

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from promptshift.data.coco import CocoVal2017, make_id  # noqa: E402


@pytest.fixture
def fake_coco(tmp_path):
    root = tmp_path / "coco_val2017" / "annotations"
    root.mkdir(parents=True)
    images = [{"id": i, "file_name": f"{i:012d}.jpg", "width": 100, "height": 80} for i in (1, 2)]

    def poly(x, y, s):
        return [[x, y, x + s, y, x + s, y + s, x, y + s]]

    anns = [
        # area >= 1% of 8000 px = 80
        {
            "id": 10,
            "image_id": 1,
            "category_id": 1,
            "iscrowd": 0,
            "area": 400.0,
            "segmentation": poly(10, 10, 20),
            "bbox": [10, 10, 20, 20],
        },
        {
            "id": 11,
            "image_id": 1,
            "category_id": 1,
            "iscrowd": 0,
            "area": 25.0,  # too small
            "segmentation": poly(50, 50, 5),
            "bbox": [50, 50, 5, 5],
        },
        {
            "id": 12,
            "image_id": 2,
            "category_id": 2,
            "iscrowd": 1,
            "area": 900.0,  # crowd
            "segmentation": poly(0, 0, 30),
            "bbox": [0, 0, 30, 30],
        },
        {
            "id": 13,
            "image_id": 2,
            "category_id": 2,
            "iscrowd": 0,
            "area": 100.0,
            "segmentation": poly(40, 20, 10),
            "bbox": [40, 20, 10, 10],
        },
    ]
    cats = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
    payload = {"images": images, "annotations": anns, "categories": cats}
    (root / "instances_val2017.json").write_text(json.dumps(payload))
    return tmp_path


def test_filtering_and_sampling(fake_coco):
    ds = CocoVal2017(fake_coco, min_area_frac=0.01, n_instances=1000, sample_seed=42)
    assert ds.counts() == {"images": 2, "annotations": 4, "crowd": 1, "eligible": 2}
    split = ds.make_split()
    assert split["test"] == sorted([make_id(1, 10), make_id(2, 13)])
    assert split["train"] == split["val"] == []
    assert (
        CocoVal2017(fake_coco, n_instances=1).make_split()
        == CocoVal2017(fake_coco, n_instances=1).make_split()
    )


def test_instance_mask_is_one_object(fake_coco):
    ds = CocoVal2017(fake_coco)
    st = ds.instance_stats(make_id(1, 10))
    assert st["n_fragments"] == 1 and st["category"] == "cat"
    assert st["gt_area_frac"] == pytest.approx(np.round(400 / 8000, 2), abs=0.02)
