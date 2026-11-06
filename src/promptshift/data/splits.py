"""Fixed, committed train/val/test splits (CLAUDE.md 6.1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


def make_split(ids: list[str], fractions: dict[str, float], seed: int) -> dict[str, list[str]]:
    """Shuffle sorted ids with a fixed seed; sizes = round(f * n) for train and val, rest test.

    For n = 1000 and 0.7 / 0.1 / 0.2 this gives exactly 700 / 100 / 200.
    """
    ids = sorted(set(ids))
    if len(ids) == 0:
        raise ValueError("no ids to split")
    if abs(sum(fractions[s] for s in SPLITS) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    order = np.random.default_rng(seed).permutation(len(ids))
    shuffled = [ids[i] for i in order]
    n_train = round(fractions["train"] * len(ids))
    n_val = round(fractions["val"] * len(ids))
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def save_split(path: Path, dataset: str, split: dict[str, list[str]], seed: int, fractions) -> None:
    payload = {
        "dataset": dataset,
        "seed": seed,
        "fractions": dict(fractions),
        "counts": {s: len(split[s]) for s in SPLITS},
        **{s: split[s] for s in SPLITS},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def load_split(path: Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    split = {s: list(payload[s]) for s in SPLITS}
    seen: set[str] = set()
    for s in SPLITS:
        overlap = seen.intersection(split[s])
        if overlap:
            raise ValueError(f"{path}: ids in more than one split, e.g. {sorted(overlap)[:3]}")
        seen.update(split[s])
    return split
