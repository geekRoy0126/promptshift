"""Segmentation metrics on 2-D boolean masks at original image resolution (spec 6.4).

Conventions (logged in the decision log):
- Dice / IoU / NSD: both masks empty -> 1.0; exactly one empty -> 0.0.
- HD95: NaN when the prediction (or the GT) is empty. Callers count empty predictions.
- Surface = mask pixels removed by one binary erosion with a 4-connected (cross)
  structuring element; pixels on the image border count as surface.
- Surface distances are Euclidean, in pixels, from each surface pixel of one mask to
  the nearest surface pixel of the other.
- NSD = (#pred-surface within tol of GT surface + #GT-surface within tol of pred surface)
        / (#pred-surface + #GT-surface), "within" meaning distance <= tol.
- HD95 = max of the two directed 95th percentiles (numpy linear interpolation).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

_CROSS = ndimage.generate_binary_structure(2, 1)


def _as_pair(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred).astype(bool, copy=False)
    gt = np.asarray(gt).astype(bool, copy=False)
    if pred.ndim != 2 or pred.shape != gt.shape:
        raise ValueError(f"expected two 2-D masks of equal shape, got {pred.shape} and {gt.shape}")
    return pred, gt


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _as_pair(pred, gt)
    total = int(pred.sum()) + int(gt.sum())
    if total == 0:
        return 1.0
    return 2.0 * int(np.logical_and(pred, gt).sum()) / total


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _as_pair(pred, gt)
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    return int(np.logical_and(pred, gt).sum()) / union


def surface(mask: np.ndarray) -> np.ndarray:
    """Boolean map of the mask's surface (inner boundary) pixels."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask)
    eroded = ndimage.binary_erosion(mask, structure=_CROSS, border_value=0)
    return mask & ~eroded


def _directed_distances(src_surface: np.ndarray, dst_surface: np.ndarray) -> np.ndarray:
    """Distance from every src surface pixel to the nearest dst surface pixel.

    The EDT is computed on the bounding box of both surfaces only. This is exact:
    every dst pixel (the zeros of the EDT input) lies inside that box.
    """
    rows = np.flatnonzero(src_surface.any(axis=1) | dst_surface.any(axis=1))
    cols = np.flatnonzero(src_surface.any(axis=0) | dst_surface.any(axis=0))
    box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    dist_to_dst = ndimage.distance_transform_edt(~dst_surface[box])
    return dist_to_dst[src_surface[box]]


def surface_distances(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pred->gt, gt->pred) surface distances. Both masks must be non-empty."""
    pred, gt = _as_pair(pred, gt)
    if not pred.any() or not gt.any():
        raise ValueError("surface distances need two non-empty masks")
    sp, sg = surface(pred), surface(gt)
    return _directed_distances(sp, sg), _directed_distances(sg, sp)


def nsd(pred: np.ndarray, gt: np.ndarray, tolerance_px: float = 2.0) -> float:
    """Normalized surface Dice with a pixel tolerance."""
    pred, gt = _as_pair(pred, gt)
    p_any, g_any = bool(pred.any()), bool(gt.any())
    if not p_any and not g_any:
        return 1.0
    if p_any != g_any:
        return 0.0
    d_pg, d_gp = surface_distances(pred, gt)
    hits = int((d_pg <= tolerance_px).sum()) + int((d_gp <= tolerance_px).sum())
    return hits / (d_pg.size + d_gp.size)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance in pixels; NaN if either mask is empty."""
    pred, gt = _as_pair(pred, gt)
    if not pred.any() or not gt.any():
        return math.nan
    d_pg, d_gp = surface_distances(pred, gt)
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def compute_metrics(pred: np.ndarray, gt: np.ndarray, nsd_tolerance_px: float) -> dict:
    """All image-level metrics plus the fields the runner needs to count empty predictions."""
    pred, gt = _as_pair(pred, gt)
    n_pixels = pred.size
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "nsd": nsd(pred, gt, nsd_tolerance_px),
        "hd95": hd95(pred, gt),
        "pred_empty": not bool(pred.any()),
        "pred_area_frac": float(pred.sum()) / n_pixels,
        "gt_area_frac": float(gt.sum()) / n_pixels,
    }
