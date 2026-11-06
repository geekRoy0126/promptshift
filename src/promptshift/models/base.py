"""Common interface for promptable segmenters (spec 5)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PromptableSegmenter(Protocol):
    name: str  # "sam_vit_b" | "sam_vit_h" | "medsam" | "mock"
    supports_points: bool  # False for MedSAM

    def set_image(self, image: np.ndarray) -> None:
        """Encode one H x W x 3 uint8 RGB image (once per image)."""

    def predict_box(self, box_xyxy: np.ndarray) -> np.ndarray:
        """Box in original-image coords -> H x W bool mask."""

    def predict_points(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        mask_input: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(N, 2) xy coords + (N,) labels -> (H x W bool mask, low-res logits)."""


def select_device(preference: str = "auto") -> str:
    """cuda > mps > cpu when preference is "auto"; otherwise the given device."""
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_state_dict(path, device: str = "cpu") -> dict:
    """torch.load on CPU first (MedSAM-style map_location), works on CPU-only machines."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(path, map_location="cpu")
