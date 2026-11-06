"""Classical segmentation baseline (spec 6.7, course Module 3 style):

    color-space channel (Lab or HSV) -> Gaussian smoothing -> Otsu threshold
    -> morphological opening, then closing -> keep the largest connected component

The free choices (channel, which side of the threshold is the object, sigma, kernel radius)
are chosen by a small grid search on the training split only.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import cv2
import numpy as np

CHANNELS = {
    "lab_L": (cv2.COLOR_RGB2LAB, 0), "lab_a": (cv2.COLOR_RGB2LAB, 1),
    "lab_b": (cv2.COLOR_RGB2LAB, 2), "hsv_s": (cv2.COLOR_RGB2HSV, 1),
    "hsv_v": (cv2.COLOR_RGB2HSV, 2),
}  # fmt: skip


@dataclass(frozen=True)
class ClassicalParams:
    channel: str  # key of CHANNELS
    polarity: str  # "high": object = values above the Otsu threshold; "low": below
    sigma: float  # Gaussian sigma in pixels
    radius: int  # elliptical kernel radius for opening / closing

    def as_dict(self) -> dict:
        return asdict(self)


def grid(cfg: dict) -> list[ClassicalParams]:
    keys = ("channel", "polarity", "sigma", "radius")
    return [ClassicalParams(*v) for v in itertools.product(*(cfg[k] for k in keys))]


def channel_image(rgb: np.ndarray, name: str) -> np.ndarray:
    code, idx = CHANNELS[name]
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), code)[..., idx]


def smooth(ch: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(ch, (0, 0), sigmaX=float(sigma)) if sigma > 0 else ch


def threshold(ch: np.ndarray, polarity: str) -> np.ndarray:
    _, high = cv2.threshold(ch, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = high.astype(bool)
    if polarity == "high":
        return high
    if polarity == "low":
        return ~high
    raise ValueError(f"polarity must be 'high' or 'low', got {polarity!r}")


def morph(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k).astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Largest 8-connected component; ties -> the lowest label (first in row-major order)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def segment(rgb: np.ndarray, p: ClassicalParams) -> np.ndarray:
    ch = smooth(channel_image(rgb, p.channel), p.sigma)
    return largest_component(morph(threshold(ch, p.polarity), p.radius))


def segment_grid(
    rgb: np.ndarray, params: list[ClassicalParams]
) -> dict[ClassicalParams, np.ndarray]:
    """All grid points for one image, reusing the smoothed channel per (channel, sigma)."""
    cache: dict[tuple[str, float], np.ndarray] = {}
    out = {}
    for p in params:
        key = (p.channel, p.sigma)
        if key not in cache:
            cache[key] = smooth(channel_image(rgb, p.channel), p.sigma)
        out[p] = largest_component(morph(threshold(cache[key], p.polarity), p.radius))
    return out
