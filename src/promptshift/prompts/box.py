"""Box prompts: GT box, jitter and scale perturbations (CLAUDE.md 6.3).

Box convention: [x0, y0, x1, y1] in original-image pixel coordinates, inclusive,
x = column, y = row (same as MedSAM's get_bbox). Perturbed boxes are float64.
Width is x1 - x0 and height y1 - y0, as in the spec's jitter formula; box_iou uses
the same continuous areas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxPrompt:
    box: np.ndarray        # float64 [x0, y0, x1, y1]; the last attempt if invalid
    gt_box: np.ndarray     # float64 [x0, y0, x1, y1]
    valid: bool
    box_iou: float         # IoU(box, gt_box); NaN if invalid
    n_resample: int = 0    # extra draws needed (jitter only)

    kind = "box"


def box_from_mask(mask: np.ndarray) -> np.ndarray:
    """Tight inclusive box of a non-empty mask: [min col, min row, max col, max row]."""
    mask = np.asarray(mask, dtype=bool)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        raise ValueError("cannot compute a box for an empty mask")
    return np.array([cols[0], rows[0], cols[-1], rows[-1]], dtype=np.float64)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two [x0, y0, x1, y1] boxes with continuous areas (x1 - x0) * (y1 - y0)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else math.nan


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip to [0, W-1] x [0, H-1]."""
    out = np.asarray(box, dtype=np.float64).copy()
    out[[0, 2]] = np.clip(out[[0, 2]], 0, width - 1)
    out[[1, 3]] = np.clip(out[[1, 3]], 0, height - 1)
    return out


def is_valid_box(box: np.ndarray, min_box_side: float = 2.0) -> bool:
    return bool(box[2] - box[0] >= min_box_side and box[3] - box[1] >= min_box_side)


def _result(box, gt_box, min_box_side, n_resample=0) -> BoxPrompt:
    valid = is_valid_box(box, min_box_side)
    return BoxPrompt(
        box=box,
        gt_box=gt_box,
        valid=valid,
        box_iou=box_iou(box, gt_box) if valid else math.nan,
        n_resample=n_resample,
    )


def jitter_box(
    gt_box: np.ndarray,
    r: float,
    rng: np.random.Generator,
    width: int,
    height: int,
    max_resample: int = 10,
    min_box_side: float = 2.0,
) -> BoxPrompt:
    """Move each edge independently by U(-r*w, r*w) (x) or U(-r*h, r*h) (y), then clip.

    An invalid box is redrawn with the same rng up to `max_resample` times
    (so at most 1 + max_resample draws); if all fail the prompt is invalid.
    """
    gt_box = np.asarray(gt_box, dtype=np.float64)
    w, h = gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]
    scale = np.array([r * w, r * h, r * w, r * h])
    box = gt_box
    for attempt in range(max_resample + 1):
        u = rng.uniform(-1.0, 1.0, size=4)  # order: dx0, dy0, dx1, dy1
        box = clip_box(gt_box + u * scale, width, height)
        if is_valid_box(box, min_box_side):
            return _result(box, gt_box, min_box_side, n_resample=attempt)
    return _result(box, gt_box, min_box_side, n_resample=max_resample)


def scale_box(
    gt_box: np.ndarray,
    s: float,
    width: int,
    height: int,
    min_box_side: float = 2.0,
) -> BoxPrompt:
    """Scale width and height by s about the box center, clip, re-check validity."""
    gt_box = np.asarray(gt_box, dtype=np.float64)
    cx, cy = (gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2
    hw, hh = s * (gt_box[2] - gt_box[0]) / 2, s * (gt_box[3] - gt_box[1]) / 2
    box = clip_box(np.array([cx - hw, cy - hh, cx + hw, cy + hh]), width, height)
    return _result(box, gt_box, min_box_side)
