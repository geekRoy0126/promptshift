"""Tiny deterministic segmenter for fast offline tests (no torch, no weights).

predict_box fills an ellipse inscribed in the box; predict_points fills a disc
around the first positive click. Good enough to exercise the runner and metrics.
"""

from __future__ import annotations

import numpy as np


class MockSegmenter:
    name = "mock"
    supports_points = True

    def __init__(self, radius_frac: float = 0.1) -> None:
        self.radius_frac = radius_frac
        self._shape: tuple[int, int] | None = None

    def set_image(self, image: np.ndarray) -> None:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected H x W x 3, got {image.shape}")
        self._shape = image.shape[:2]

    def _grid(self):
        if self._shape is None:
            raise RuntimeError("call set_image first")
        h, w = self._shape
        return np.mgrid[:h, :w]

    def predict_box(self, box_xyxy: np.ndarray) -> np.ndarray:
        yy, xx = self._grid()
        x0, y0, x1, y1 = np.asarray(box_xyxy, dtype=float)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = max((x1 - x0) / 2, 0.5), max((y1 - y0) / 2, 0.5)
        return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

    def predict_points(self, coords, labels, mask_input=None):
        yy, xx = self._grid()
        coords, labels = np.asarray(coords, float), np.asarray(labels)
        pos = coords[labels == 1]
        r = self.radius_frac * min(self._shape)
        mask = np.zeros(self._shape, dtype=bool)
        for x, y in pos:
            mask |= (xx - x) ** 2 + (yy - y) ** 2 <= r**2
        logits = np.where(mask, 10.0, -10.0)[::4, ::4].astype(np.float32)
        return mask, logits
