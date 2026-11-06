"""Experiment configs = configs/default.yaml deep-merged with an experiment YAML."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, default: str | Path | None = None) -> dict:
    """Load `path` on top of default.yaml (from the same folder unless given)."""
    path = Path(path)
    default = Path(default) if default is not None else path.parent / "default.yaml"
    base = yaml.safe_load(default.read_text(encoding="utf-8")) or {}
    if path.resolve() == default.resolve():
        return base
    exp = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return deep_merge(base, exp)
