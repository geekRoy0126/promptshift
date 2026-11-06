"""Result rows for prompt-free methods, in the runner's long CSV format."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from promptshift.metrics import compute_metrics
from promptshift.runner import COLUMNS


def baseline_row(
    *, exp_id: str, split: str, method: str, sample, pred: np.ndarray, ms: float, sha: str,
    nsd_tolerance_px: float,
) -> dict:  # fmt: skip
    """One image-level row: perturb "none", level NaN, seed 0, no prompt (box_iou_mean NaN)."""
    pred = np.asarray(pred, dtype=bool)
    if pred.shape != sample.mask.shape:
        raise ValueError(f"{sample.image_id}: prediction {pred.shape} != mask {sample.mask.shape}")
    m = compute_metrics(pred, sample.mask, nsd_tolerance_px)
    m.pop("pred_empty")
    return {
        "exp_id": exp_id, "dataset": sample.dataset, "split": split, "model": method,
        "image_id": sample.image_id, "n_objects": len(sample.objects),
        "perturb": "none", "level": math.nan, "seed": 0,
        **m, "box_iou_mean": math.nan,
        "encode_ms": ms, "decode_ms": 0.0, "flag": "ok", "git_sha": sha,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }  # fmt: skip


def write_rows(path: Path, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return df
