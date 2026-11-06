"""Common sample format and mask cleaning shared by every dataset loader (spec 5, 6.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image
from scipy import ndimage

_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class Sample:
    image_id: str
    image: np.ndarray          # H x W x 3, uint8, RGB
    mask: np.ndarray           # H x W, bool, union of kept objects (specks removed)
    objects: list[np.ndarray]  # one H x W bool mask per kept connected component
    dataset: str
    meta: dict = field(default_factory=dict)


class SegDataset(Protocol):
    name: str

    def ids(self, split: str) -> list[str]: ...

    def get(self, image_id: str) -> Sample: ...


def load_rgb(path) -> np.ndarray:
    """H x W x 3 uint8 RGB (grayscale images are replicated to 3 channels)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def load_gray(path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


@dataclass
class CleanMask:
    mask: np.ndarray
    objects: list[np.ndarray]
    n_components: int     # before speck removal
    n_specks: int         # components removed
    speck_pixels: int     # pixels removed


def split_objects(raw_mask: np.ndarray, min_object_frac: float, connectivity: int = 8) -> CleanMask:
    """Connected components of a binary mask; drop those smaller than min_object_frac * H * W.

    Objects are ordered by label, i.e. by the row-major position of each component's
    first pixel, so object_idx is stable.
    """
    raw_mask = np.asarray(raw_mask, dtype=bool)
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    structure = _EIGHT if connectivity == 8 else ndimage.generate_binary_structure(2, 1)
    labels, n = ndimage.label(raw_mask, structure=structure)
    min_area = min_object_frac * raw_mask.size
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    keep = [i + 1 for i in range(n) if areas[i] >= min_area]
    objects = [labels == k for k in keep]
    mask = np.isin(labels, keep) if keep else np.zeros_like(raw_mask)
    removed = [i for i in range(n) if areas[i] < min_area]
    return CleanMask(
        mask=mask,
        objects=objects,
        n_components=n,
        n_specks=len(removed),
        speck_pixels=int(sum(areas[i] for i in removed)),
    )
