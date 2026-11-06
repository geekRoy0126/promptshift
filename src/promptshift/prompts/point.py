"""Click prompts with controlled offset from the object center (spec 6.3).

The distance transform is computed on the mask padded with one background pixel on
every side, so the image border counts as object boundary. Without padding, an object
cut by the image border would get its "center" on the border itself.

Points are returned as SAM expects: coords (N, 2) as (x, y) = (col, row), labels (N,).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class PointPrompt:
    coords: np.ndarray     # float64 (1, 2) as (x, y)
    labels: np.ndarray     # int64 (1,), 1 = foreground
    valid: bool
    depth: float           # DT / DT.max() at the click (1 = center, ~0 = boundary)
    fallback: bool = False # True if the depth band was empty and the closest pixel was used

    kind = "point"


def object_distance_transform(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance to the nearest background pixel, image border = background."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("cannot place a click on an empty mask")
    padded = np.pad(mask, 1, constant_values=False)
    return ndimage.distance_transform_edt(padded)[1:-1, 1:-1]


def _point(flat_idx: int, shape, depth_map, fallback=False) -> PointPrompt:
    row, col = np.unravel_index(flat_idx, shape)
    return PointPrompt(
        coords=np.array([[col, row]], dtype=np.float64),
        labels=np.array([1], dtype=np.int64),
        valid=True,
        depth=float(depth_map.flat[flat_idx]),
        fallback=fallback,
    )


def offset_click(
    mask: np.ndarray,
    d: float,
    rng: np.random.Generator,
    band: float = 0.05,
) -> PointPrompt:
    """One foreground click at relative offset d from the center (0) to the boundary (1).

    - d == 0: argmax of the DT (first index in row-major order on ties).
    - 0 < d < 1: uniform over pixels with |DT/DT.max() - (1 - d)| <= band; if none,
      the pixel with the closest depth (first in row-major order).
    - d >= 1: uniform over pixels with DT <= 1 (inner boundary).
    Candidates are enumerated in row-major order, so the choice depends only on the rng.
    """
    mask = np.asarray(mask, dtype=bool)
    dt = object_distance_transform(mask)
    depth = dt / dt.max()
    if d <= 0:
        return _point(int(np.argmax(dt)), mask.shape, depth)
    if d >= 1:
        candidates = np.flatnonzero(mask & (dt <= 1.0))
    else:
        target = 1.0 - d
        candidates = np.flatnonzero(mask & (np.abs(depth - target) <= band))
        if candidates.size == 0:
            gap = np.where(mask, np.abs(depth - target), np.inf)
            return _point(int(np.argmin(gap)), mask.shape, depth, fallback=True)
    return _point(int(candidates[rng.integers(candidates.size)]), mask.shape, depth)
