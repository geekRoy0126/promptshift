"""Stable, process-independent randomness (CLAUDE.md 6.3).

Python's built-in hash() on str is salted per process, so it is never used here.
A prompt's rng depends only on (global_seed, dataset, image_id, object_idx, perturb,
level, seed) and never on the model, so every model sees exactly the same prompt.
"""

from __future__ import annotations

import hashlib
import math
import os
import random

import numpy as np


def format_level(level: float | int | None) -> str:
    """Canonical text for a perturbation level: 0.1, 0.10 and 1e-1 all give "0.1"."""
    if level is None:
        return "none"
    value = float(level)
    if math.isnan(value):
        return "none"
    return format(value, ".6g")


def prompt_key(
    global_seed: int,
    dataset: str,
    image_id: str,
    object_idx: int,
    perturb: str,
    level: float | int | None,
    seed: int,
) -> str:
    """The exact key string from CLAUDE.md 6.3."""
    return (
        f"{int(global_seed)}|{dataset}|{image_id}|{int(object_idx)}|"
        f"{perturb}|{format_level(level)}|{int(seed)}"
    )


def seed_from_key(key: str) -> int:
    """First 64 bits of sha256(key) as a non-negative int."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def stable_rng(
    global_seed: int,
    dataset: str,
    image_id: str,
    object_idx: int,
    perturb: str,
    level: float | int | None,
    seed: int,
) -> np.random.Generator:
    """Generator seeded only by the prompt identity (no model name)."""
    key = prompt_key(global_seed, dataset, image_id, object_idx, perturb, level, seed)
    return np.random.default_rng(seed_from_key(key))


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch for training code."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
