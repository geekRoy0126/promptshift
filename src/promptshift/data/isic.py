"""ISIC 2018 Task 1 (lesion segmentation) loader (CLAUDE.md 6.1). Official training set only.

Layout after scripts/download_data.py --dataset isic2018:
    <data>/isic2018/ISIC2018_Task1_Training_GroundTruth/ISIC_<7 digits>_segmentation.png
    <data>/isic2018/ISIC2018_Task1-2_Training_Input/ISIC_<7 digits>.jpg      (full zip, 11 GB)
With --skip-images only the masks are extracted, plus
    <data>/isic2018/ISIC2018_Task1-2_Training_Input.index.json   (member list of the image zip,
                                                                   read with range requests)
and single images are fetched on demand into ..._Training_Input_samples/ (never into the
full folder, so a partial folder can't stop the full extraction later).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from promptshift.data.base import Sample, load_gray, load_rgb, split_objects
from promptshift.data.splits import SPLITS, load_split

NAME = "isic2018"
IMAGES_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip"
)
IMG_DIR = "ISIC2018_Task1-2_Training_Input"
GT_DIR = "ISIC2018_Task1_Training_GroundTruth"
INDEX_FILE = f"{IMG_DIR}.index.json"
SAMPLES_DIR = f"{IMG_DIR}_samples"
# https://challenge.isic-archive.com/data/ : "2594 images and 12970 corresponding ground truth
# response masks (5 for each image)" (Task 1 uses one mask per image).
OFFICIAL_TRAINING_IMAGES = 2594
_MASK_RE = re.compile(r"^(ISIC_\d{7})_segmentation\.png$")
_IMG_RE = re.compile(r"^(ISIC_\d{7})\.jpg$")


def image_ids_from_names(names) -> set[str]:
    return {m.group(1) for n in names if (m := _IMG_RE.match(Path(n).name))}


class Isic2018:
    name = NAME

    def __init__(
        self,
        data_dir: Path,
        split_file: Path | None = None,
        mask_threshold: int = 127,
        min_object_frac: float = 0.001,
        connectivity: int = 8,
    ) -> None:
        self.root = Path(data_dir) / NAME
        gt = self.root / GT_DIR
        if not gt.is_dir():
            raise FileNotFoundError(
                f"{gt} not found; run scripts/download_data.py --dataset isic2018 [--skip-images]"
            )
        self.masks = {
            m.group(1): p for p in sorted(gt.iterdir()) if (m := _MASK_RE.match(p.name))
        }
        local = image_ids_from_names(p.name for p in (self.root / IMG_DIR).glob("*.jpg"))
        index = self.root / INDEX_FILE
        listed = (
            image_ids_from_names(json.loads(index.read_text(encoding="utf-8")))
            if index.is_file()
            else set()
        )
        self.image_ids = local | listed
        self.image_source = "index + local" if listed else "local folder"
        self.problems = {
            "image_without_mask": sorted(self.image_ids - self.masks.keys()),
            "mask_without_image": sorted(self.masks.keys() - self.image_ids),
        }
        self.split = load_split(split_file) if split_file is not None else None
        self.mask_threshold = mask_threshold
        self.min_object_frac = min_object_frac
        self.connectivity = connectivity
        self._remote = None

    @classmethod
    def from_config(cls, data_dir: Path, data_cfg: dict, split_file: Path | None = None):
        return cls(
            data_dir,
            split_file=split_file,
            mask_threshold=data_cfg["mask_threshold"],
            min_object_frac=data_cfg["min_object_frac"],
            connectivity=data_cfg["connectivity"],
        )

    def all_ids(self) -> list[str]:
        """Ids with a mask (the split is built from these; images are checked separately)."""
        return list(self.masks)

    def ids(self, split: str) -> list[str]:
        if split == "all":
            return self.all_ids()
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        if self.split is None:
            raise RuntimeError("no split file given")
        return list(self.split[split])

    def count_check(self) -> tuple[list[str], bool]:
        n_img, n_mask = len(self.image_ids), len(self.masks)
        lines = [
            f"- Training images in the image zip ({self.image_source}): **{n_img}** "
            f"(official: {OFFICIAL_TRAINING_IMAGES})",
            f"- Training masks: **{n_mask}** (official: {OFFICIAL_TRAINING_IMAGES})",
            "- Split: our seed-42 70/10/20 split of the official training set (CLAUDE.md 6.1)",
        ]
        return lines, n_img == n_mask == OFFICIAL_TRAINING_IMAGES

    def image_path(self, image_id: str) -> Path:
        full = self.root / IMG_DIR / f"{image_id}.jpg"
        return full if full.is_file() else self.root / SAMPLES_DIR / f"{image_id}.jpg"

    def ensure_image(self, image_id: str) -> Path:
        """Fetch one image from the remote zip if it is not on disk."""
        path = self.image_path(image_id)
        if not path.is_file():
            from promptshift.data.remote_zip import open_remote_zip

            if self._remote is None:
                self._remote = open_remote_zip(IMAGES_URL)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(self._remote.read(f"{IMG_DIR}/{image_id}.jpg"))
            tmp.replace(path)
        return path

    def _clean(self, image_id: str):
        gray = load_gray(self.masks[image_id])
        raw = gray > self.mask_threshold
        return gray, split_objects(raw, self.min_object_frac, self.connectivity)

    def _meta(self, image_id, gray, clean) -> dict:
        return {
            "mask_file": self.masks[image_id].name,
            "height": gray.shape[0],
            "width": gray.shape[1],
            "n_components_raw": clean.n_components,
            "n_specks_removed": clean.n_specks,
            "speck_pixels_removed": clean.speck_pixels,
        }

    def mask_sample(self, image_id: str) -> Sample:
        """Sample without the image (statistics from masks only)."""
        gray, clean = self._clean(image_id)
        return Sample(image_id, None, clean.mask, clean.objects, self.name,
                      self._meta(image_id, gray, clean))  # fmt: skip

    def get(self, image_id: str) -> Sample:
        path = self.image_path(image_id)
        if not path.is_file():
            raise FileNotFoundError(f"{path}: download the image zip or call ensure_image()")
        image = load_rgb(path)
        gray, clean = self._clean(image_id)
        if gray.shape != image.shape[:2]:
            raise ValueError(f"{image_id}: mask {gray.shape} != image {image.shape[:2]}")
        meta = {"image_file": path.name, **self._meta(image_id, gray, clean)}
        return Sample(image_id, image, clean.mask, clean.objects, self.name, meta)
