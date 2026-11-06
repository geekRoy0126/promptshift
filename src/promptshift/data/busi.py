"""BUSI (breast ultrasound) loader (spec 6.1).

Source: Kaggle aryashah2k/breast-ultrasound-images-dataset, layout
    <data>/busi/Dataset_BUSI_with_GT/<class>/<class> (<n>).png
    <data>/busi/Dataset_BUSI_with_GT/<class>/<class> (<n>)_mask.png, ..._mask_1.png, ...
Rules: drop the `normal` class (no lesion); several masks of one image are merged (union);
byte-identical images are deduplicated (the first id in sorted order is kept) and counted.
Image id: "<class>_<n:04d>", e.g. "benign_0001".
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from promptshift.data.base import Sample, load_gray, load_rgb, split_objects
from promptshift.data.splits import SPLITS, load_split

NAME = "busi"
DIR = "Dataset_BUSI_with_GT"
CLASSES = ("benign", "malignant", "normal")
LESION_CLASSES = ("benign", "malignant")
# Per-class image counts. Al-Dhabyani et al. 2020 (Data in Brief 28:104863) Table 1 prints
# benign 487, which contradicts its own total of 780; 437 + 210 + 133 = 780, and MedSAM's
# 647 lesion images (Supp. Table 3) = 437 + 210. We check against 437 (decision log).
OFFICIAL_COUNTS = {"benign": 437, "malignant": 210, "normal": 133}
_FILE_RE = re.compile(r"^(benign|malignant|normal) \((\d+)\)(_mask(?:_\d+)?)?\.png$")


def find_root(data_dir: Path) -> Path:
    for cand in (data_dir / NAME / DIR, data_dir / NAME, data_dir / DIR):
        if all((cand / c).is_dir() for c in LESION_CLASSES):
            return cand
    raise FileNotFoundError(
        f"BUSI not found under {data_dir}; run scripts/download_data.py --dataset busi"
    )


def scan(root: Path, classes) -> tuple[dict, dict, dict]:
    """Returns ({id: image path}, {id: [mask paths]}, {"unparsed": [...], per-class counts})."""
    images, masks, info = {}, {}, {"unparsed": [], "counts": {}}
    for cls in classes:
        n_img = 0
        for p in sorted((root / cls).iterdir()):
            m = _FILE_RE.match(p.name)
            if not m or m.group(1) != cls:
                info["unparsed"].append(f"{cls}/{p.name}")
                continue
            image_id = f"{cls}_{int(m.group(2)):04d}"
            if m.group(3):
                masks.setdefault(image_id, []).append(p)
            else:
                images[image_id] = p
                n_img += 1
        info["counts"][cls] = n_img
    return images, masks, info


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Busi:
    name = NAME

    def __init__(
        self,
        data_dir: Path,
        split_file: Path | None = None,
        mask_threshold: int = 127,
        min_object_frac: float = 0.001,
        connectivity: int = 8,
        drop_normal: bool = True,
    ) -> None:
        self.root = find_root(Path(data_dir))
        all_images, _, all_info = scan(self.root, [c for c in CLASSES if (self.root / c).is_dir()])
        self.class_counts = all_info["counts"]  # before dropping / dedup, for the count check
        classes = LESION_CLASSES if drop_normal else CLASSES
        images, masks, info = scan(self.root, classes)
        self.dropped_normal = self.class_counts.get("normal", 0) if drop_normal else 0
        self.problems = {
            "image_without_mask": sorted(images.keys() - masks.keys()),
            "mask_without_image": sorted(masks.keys() - images.keys()),
            "unparsed_files": info["unparsed"],
        }
        # deduplicate byte-identical images (keep the first id in sorted order)
        first: dict[str, str] = {}
        self.duplicates: dict[str, str] = {}  # removed id -> kept id
        for image_id in sorted(images):
            h = file_hash(images[image_id])
            if h in first:
                self.duplicates[image_id] = first[h]
            else:
                first[h] = image_id
        keep = sorted((images.keys() & masks.keys()) - self.duplicates.keys())
        self.images = {i: images[i] for i in keep}
        self.masks = {i: sorted(masks[i]) for i in keep}
        self.n_multi_mask = sum(len(v) > 1 for v in self.masks.values())
        self.split = load_split(split_file) if split_file is not None else None
        self.mask_threshold = mask_threshold
        self.min_object_frac = min_object_frac
        self.connectivity = connectivity

    @classmethod
    def from_config(cls, data_dir: Path, data_cfg: dict, split_file: Path | None = None):
        return cls(
            data_dir,
            split_file=split_file,
            mask_threshold=data_cfg["mask_threshold"],
            min_object_frac=data_cfg["min_object_frac"],
            connectivity=data_cfg["connectivity"],
            drop_normal=data_cfg.get("busi", {}).get("drop_normal", True),
        )

    def all_ids(self) -> list[str]:
        return list(self.images)

    def ids(self, split: str) -> list[str]:
        if split == "all":
            return self.all_ids()
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        if self.split is None:
            raise RuntimeError("no split file given")
        return list(self.split[split])

    def count_check(self) -> tuple[list[str], bool]:
        counts = ", ".join(
            f"{c} {self.class_counts.get(c, 0)} (official {OFFICIAL_COUNTS[c]})" for c in CLASSES
        )
        dups = ", ".join(f"{a} = {b}" for a, b in sorted(self.duplicates.items()))
        lines = [
            f"- Images per class: {counts}",
            f"- Dropped class `normal` (no lesion): {self.dropped_normal} images",
            f"- Byte-identical duplicate images removed: **{len(self.duplicates)}**"
            + (f" ({dups})" if dups else ""),
            f"- Images with several mask files (merged by union): {self.n_multi_mask}",
            f"- Lesion images kept: **{len(self.images)}**",
        ]
        ok = all(self.class_counts.get(c, 0) == n for c, n in OFFICIAL_COUNTS.items())
        return lines, ok

    def raw_mask(self, image_id: str) -> np.ndarray:
        parts = [load_gray(p) > self.mask_threshold for p in self.masks[image_id]]
        shapes = {m.shape for m in parts}
        if len(shapes) > 1:
            raise ValueError(f"{image_id}: masks have different shapes {shapes}")
        return np.logical_or.reduce(parts)

    def get(self, image_id: str) -> Sample:
        image = load_rgb(self.images[image_id])
        raw = self.raw_mask(image_id)
        if raw.shape != image.shape[:2]:
            raise ValueError(f"{image_id}: mask {raw.shape} != image {image.shape[:2]}")
        clean = split_objects(raw, self.min_object_frac, self.connectivity)
        return Sample(
            image_id=image_id,
            image=image,
            mask=clean.mask,
            objects=clean.objects,
            dataset=self.name,
            meta={
                "image_file": self.images[image_id].name,
                "class": image_id.split("_")[0],
                "n_mask_files": len(self.masks[image_id]),
                "height": image.shape[0],
                "width": image.shape[1],
                "n_components_raw": clean.n_components,
                "n_specks_removed": clean.n_specks,
                "speck_pixels_removed": clean.speck_pixels,
            },
        )
