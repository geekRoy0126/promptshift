"""Model registry: load_model(name, weights_dir, device)."""

from __future__ import annotations

from pathlib import Path

from promptshift.models.base import PromptableSegmenter, select_device
from promptshift.models.mock import MockSegmenter

MODEL_NAMES = ("sam_vit_b", "sam_vit_h", "medsam", "mock")


def load_model(name: str, weights_dir: Path, device: str = "auto") -> PromptableSegmenter:
    if name == "mock":
        return MockSegmenter()
    if name in ("sam_vit_b", "sam_vit_h"):
        from promptshift.models.sam import SamSegmenter

        return SamSegmenter(name, Path(weights_dir), device)
    if name == "medsam":
        from promptshift.models.medsam import MedSamSegmenter

        return MedSamSegmenter(Path(weights_dir), device)
    raise KeyError(f"unknown model {name!r}; known: {MODEL_NAMES}")


__all__ = ["MODEL_NAMES", "MockSegmenter", "PromptableSegmenter", "load_model", "select_device"]
