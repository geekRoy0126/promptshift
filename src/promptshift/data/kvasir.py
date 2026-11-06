"""Kvasir-SEG loader (CLAUDE.md 6.1).

Layout after scripts/download_data.py:
    <data>/kvasir_seg/Kvasir-SEG/images/<id>.jpg
    <data>/kvasir_seg/Kvasir-SEG/masks/<id>.jpg     (JPEG: binarize with gray > threshold)
The official kavsir_bboxes.json is not used; boxes come from the masks.
"""

from __future__ import annotations

from pathlib import Path

from promptshift.data.base import Sample, load_gray, load_rgb, split_objects
from promptshift.data.splits import SPLITS, load_split

NAME = "kvasir_seg"


def find_root(data_dir: Path) -> Path:
    """The folder that contains images/ and masks/."""
    for cand in (data_dir / NAME / "Kvasir-SEG", data_dir / NAME, data_dir):
        if (cand / "images").is_dir() and (cand / "masks").is_dir():
            return cand
    raise FileNotFoundError(
        f"Kvasir-SEG not found under {data_dir}; run scripts/download_data.py --dataset kvasir_seg"
    )


def pair_files(root: Path) -> tuple[dict[str, tuple[Path, Path]], dict[str, list[str]]]:
    """Match images and masks by file stem. Returns (pairs, problems)."""
    images = {p.stem: p for p in (root / "images").iterdir() if p.is_file()}
    masks = {p.stem: p for p in (root / "masks").iterdir() if p.is_file()}
    pairs = {k: (images[k], masks[k]) for k in sorted(images.keys() & masks.keys())}
    problems = {
        "image_without_mask": sorted(images.keys() - masks.keys()),
        "mask_without_image": sorted(masks.keys() - images.keys()),
    }
    return pairs, problems


class KvasirSeg:
    name = NAME

    def __init__(
        self,
        data_dir: Path,
        split_file: Path | None = None,
        mask_threshold: int = 127,
        min_object_frac: float = 0.001,
        connectivity: int = 8,
    ) -> None:
        self.root = find_root(Path(data_dir))
        self.pairs, self.problems = pair_files(self.root)
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
        )

    def all_ids(self) -> list[str]:
        return list(self.pairs)

    def ids(self, split: str) -> list[str]:
        if split == "all":
            return self.all_ids()
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        if self.split is None:
            raise RuntimeError("no split file given")
        missing = [i for i in self.split[split] if i not in self.pairs]
        if missing:
            msg = f"{len(missing)} {split} ids missing on disk, e.g. {missing[:3]}"
            raise FileNotFoundError(msg)
        return list(self.split[split])

    def get(self, image_id: str) -> Sample:
        img_path, mask_path = self.pairs[image_id]
        image = load_rgb(img_path)
        gray = load_gray(mask_path)
        if gray.shape != image.shape[:2]:
            raise ValueError(f"{image_id}: mask {gray.shape} != image {image.shape[:2]}")
        clean = split_objects(gray > self.mask_threshold, self.min_object_frac, self.connectivity)
        return Sample(
            image_id=image_id,
            image=image,
            mask=clean.mask,
            objects=clean.objects,
            dataset=self.name,
            meta={
                "image_file": img_path.name,
                "mask_file": mask_path.name,
                "height": image.shape[0],
                "width": image.shape[1],
                "n_components_raw": clean.n_components,
                "n_specks_removed": clean.n_specks,
                "speck_pixels_removed": clean.speck_pixels,
            },
        )
