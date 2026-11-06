"""Prompt simulator registry.

Adding a perturbation = one function with the signature
    fn(obj_mask, level, rng, **params) -> BoxPrompt | PointPrompt
plus one `register(...)` call. `params` come from cfg["prompts"][name] minus
"levels" and "seeds".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from promptshift.prompts.box import BoxPrompt, box_from_mask, box_iou, jitter_box, scale_box
from promptshift.prompts.point import PointPrompt, offset_click
from promptshift.seeding import stable_rng

Prompt = BoxPrompt | PointPrompt
GRID_KEYS = ("levels", "seeds")


@dataclass(frozen=True)
class Perturbation:
    name: str
    kind: str  # "box" | "point"
    fn: Callable[..., Prompt]


PERTURBATIONS: dict[str, Perturbation] = {}


def register(name: str, kind: str, fn: Callable[..., Prompt]) -> None:
    if name in PERTURBATIONS:
        raise ValueError(f"perturbation {name!r} is already registered")
    if kind not in ("box", "point"):
        raise ValueError(f"unknown prompt kind {kind!r}")
    PERTURBATIONS[name] = Perturbation(name, kind, fn)


def _box_jitter(obj_mask, level, rng, max_resample=10, min_box_side=2.0):
    h, w = obj_mask.shape
    return jitter_box(box_from_mask(obj_mask), level, rng, w, h, max_resample, min_box_side)


def _box_scale(obj_mask, level, rng, min_box_side=2.0):
    h, w = obj_mask.shape
    return scale_box(box_from_mask(obj_mask), level, w, h, min_box_side)


def _point_offset(obj_mask, level, rng, band=0.05):
    return offset_click(obj_mask, level, rng, band)


register("box_jitter", "box", _box_jitter)
register("box_scale", "box", _box_scale)
register("point_offset", "point", _point_offset)


def make_prompt(
    perturb: str,
    level: float,
    obj_mask: np.ndarray,
    *,
    global_seed: int,
    dataset: str,
    image_id: str,
    object_idx: int,
    seed: int,
    params: dict | None = None,
) -> Prompt:
    """Prompt for one object. Depends only on the key fields and the mask, never on a model."""
    if perturb not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {perturb!r}; known: {sorted(PERTURBATIONS)}")
    rng = stable_rng(global_seed, dataset, image_id, object_idx, perturb, level, seed)
    kwargs = {k: v for k, v in (params or {}).items() if k not in GRID_KEYS}
    return PERTURBATIONS[perturb].fn(np.asarray(obj_mask, dtype=bool), level, rng, **kwargs)


__all__ = [
    "PERTURBATIONS",
    "BoxPrompt",
    "Perturbation",
    "PointPrompt",
    "Prompt",
    "box_from_mask",
    "box_iou",
    "make_prompt",
    "register",
]
