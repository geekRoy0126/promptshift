"""Rank observed failures; no inference or invented causal labels."""

from __future__ import annotations

import pandas as pd


def rank_failures(df: pd.DataFrame, perturb="box_jitter", reference=0.0) -> pd.DataFrame:
    """Seed-average per image, then rank by largest Dice drop from reference.

    Retain improvements (negative drop) and missing reference values transparently.
    Invalid rows are excluded here and counted separately by the analysis report.
    """
    ok = df[(df.flag == "ok") & (df.perturb == perturb)]
    keys = ["dataset", "model", "image_id"]
    cols = [c for c in ("dice", "iou", "gt_area_frac", "pred_area_frac", "box_iou_mean") if c in ok]
    avg = ok.groupby(keys + ["level"], as_index=False)[cols].mean()
    base = avg[avg.level.sub(reference).abs() < 1e-9][keys + ["dice"]]
    base = base.rename(columns={"dice": "reference_dice"})
    out = avg.merge(base, on=keys, how="left", validate="many_to_one")
    out = out[out.level.sub(reference).abs() >= 1e-9].copy()
    out["dice_drop"] = out.reference_dice - out.dice
    out["perturb"] = perturb
    return out.sort_values(
        ["dice_drop", "dice", "image_id"], ascending=[False, True, True], na_position="last"
    ).reset_index(drop=True)
