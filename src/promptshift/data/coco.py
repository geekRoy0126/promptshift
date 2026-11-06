"""COCO val2017 instance loader (CLAUDE.md 6.1; P1: loader + verification only).

Layout after scripts/download_data.py --dataset coco_val2017:
    <data>/coco_val2017/annotations/instances_val2017.json
    <data>/coco_val2017/val2017/<file_name>.jpg       (skipped locally with --skip-images)

Each kept instance is an independent sample: image_id = "<coco image id:012d>_<ann id>".
Kept: iscrowd == 0 and ann["area"] >= min_area_frac * W * H; then `n_instances` are
sampled with a fixed seed. The instance mask is one object (not split into connected
components): a box prompt for an occluded, fragmented instance is still one box.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from promptshift.data.base import Sample, load_rgb
from promptshift.data.splits import SPLITS, load_split

NAME = "coco_val2017"
IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"


def make_id(image_id: int, ann_id: int) -> str:
    return f"{image_id:012d}_{ann_id}"


class CocoVal2017:
    name = NAME

    def __init__(
        self,
        data_dir: Path,
        split_file: Path | None = None,
        min_area_frac: float = 0.01,
        n_instances: int = 1000,
        sample_seed: int = 42,
    ) -> None:
        from pycocotools.coco import COCO

        self.root = Path(data_dir) / NAME
        ann_file = self.root / "annotations" / "instances_val2017.json"
        if not ann_file.is_file():
            raise FileNotFoundError(
                f"{ann_file} missing; run scripts/download_data.py --dataset coco_val2017"
            )
        self.coco = COCO(str(ann_file))
        self.images_dir = self.root / "val2017"
        self.min_area_frac = min_area_frac
        self.n_instances = n_instances
        self.sample_seed = sample_seed
        self.eligible = self._eligible()
        self.lookup = {make_id(a["image_id"], a["id"]): a for a in self.eligible}
        self.split = load_split(split_file) if split_file is not None else None
        self.problems: dict[str, list[str]] = {}

    @classmethod
    def from_config(cls, data_dir: Path, data_cfg: dict, split_file: Path | None = None):
        c = data_cfg["coco"]
        return cls(data_dir, split_file, c["min_area_frac"], c["n_instances"], c["sample_seed"])

    def _eligible(self) -> list[dict]:
        out = []
        for ann in self.coco.dataset["annotations"]:
            img = self.coco.imgs[ann["image_id"]]
            if (
                ann["iscrowd"] == 0
                and ann["area"] >= self.min_area_frac * img["width"] * img["height"]
            ):
                out.append(ann)
        return sorted(out, key=lambda a: a["id"])

    def counts(self) -> dict[str, int]:
        anns = self.coco.dataset["annotations"]
        return {
            "images": len(self.coco.imgs),
            "annotations": len(anns),
            "crowd": sum(a["iscrowd"] for a in anns),
            "eligible": len(self.eligible),
        }

    def make_split(self, fractions=None, seed=None) -> dict[str, list[str]]:
        """All sampled instances go to "test" (COCO is evaluation only)."""
        n = min(self.n_instances, len(self.eligible))
        idx = np.random.default_rng(self.sample_seed).choice(len(self.eligible), n, replace=False)
        test = sorted(make_id(self.eligible[i]["image_id"], self.eligible[i]["id"]) for i in idx)
        return {"train": [], "val": [], "test": test}

    def all_ids(self) -> list[str]:
        if self.split is not None:
            return sorted(i for s in SPLITS for i in self.split[s])
        return self.make_split()["test"]

    def ids(self, split: str) -> list[str]:
        if split == "all":
            return self.all_ids()
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        if self.split is None:
            raise RuntimeError("no split file given")
        return list(self.split[split])

    def image_path(self, image_id: str) -> Path:
        """val2017/<file> if the full zip is extracted, else val2017_samples/<file>."""
        name = self.coco.imgs[self.lookup[image_id]["image_id"]]["file_name"]
        full = self.images_dir / name
        return full if full.is_file() else self.root / "val2017_samples" / name

    def ensure_image(self, image_id: str) -> Path:
        """Download one image if missing (local checks without the 1 GB zip).

        Single images go to val2017_samples/, never val2017/, so a partial val2017/
        folder can't make download_data.py skip extracting the full zip.
        """
        path = self.image_path(image_id)
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(IMAGE_URL.format(file_name=path.name), path)
        return path

    def get(self, image_id: str) -> Sample:
        ann = self.lookup[image_id]
        info = self.coco.imgs[ann["image_id"]]
        image = load_rgb(self.image_path(image_id))
        mask = self.coco.annToMask(ann).astype(bool)
        if mask.shape != image.shape[:2]:
            raise ValueError(f"{image_id}: mask {mask.shape} != image {image.shape[:2]}")
        return Sample(
            image_id=image_id,
            image=image,
            mask=mask,
            objects=[mask],
            dataset=self.name,
            meta={
                "image_file": info["file_name"],
                "height": info["height"],
                "width": info["width"],
                "coco_image_id": ann["image_id"],
                "ann_id": ann["id"],
                "category": self.coco.cats[ann["category_id"]]["name"],
                "ann_area": ann["area"],
            },
        )

    def instance_stats(self, image_id: str) -> dict:
        """Mask-only statistics (no image file needed)."""
        ann = self.lookup[image_id]
        info = self.coco.imgs[ann["image_id"]]
        mask = self.coco.annToMask(ann).astype(bool)
        from scipy import ndimage

        _, n_frag = ndimage.label(mask, structure=np.ones((3, 3), bool))
        return {
            "image_id": image_id,
            "height": info["height"],
            "width": info["width"],
            "category": self.coco.cats[ann["category_id"]]["name"],
            "n_fragments": n_frag,
            "gt_area_frac": float(mask.sum()) / mask.size,
        }
