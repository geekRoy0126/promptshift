"""Statistics for P1-12 (spec 6.8). The statistical unit is the image: seeds are averaged
within each image first, then images are resampled (bootstrap) or paired (Wilcoxon)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def per_image(df: pd.DataFrame, perturb: str, metric: str = "dice") -> pd.DataFrame:
    """level x image_id table of seed-averaged metric (ok rows only)."""
    ok = df[(df["perturb"] == perturb) & (df["flag"] == "ok")]
    return ok.groupby(["level", "image_id"])[metric].mean().unstack("image_id")


def baseline_per_image(df: pd.DataFrame, metric: str = "dice") -> pd.Series:
    """Prompt-free method (perturb == "none"): one value per image."""
    ok = df[(df["perturb"] == "none") & (df["flag"] == "ok")]
    return ok.groupby("image_id")[metric].mean()


def bootstrap_ci(values, n: int = 2000, seed: int = 42, ci: float = 0.95) -> tuple[float, float]:
    """Percentile CI of the mean over images."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def curve(df: pd.DataFrame, perturb: str, n=2000, seed=42, ci=0.95) -> pd.DataFrame:
    """Per level: mean Dice over images and bootstrap CI."""
    t = per_image(df, perturb)
    rows = []
    for level, vals in t.iterrows():
        lo, hi = bootstrap_ci(vals.to_numpy(), n, seed, ci)
        rows.append({"level": float(level), "mean": float(vals.mean()), "lo": lo, "hi": hi,
                     "n_images": int(vals.notna().sum())})  # fmt: skip
    return pd.DataFrame(rows)


def baseline_summary(df: pd.DataFrame, n=2000, seed=42, ci=0.95) -> dict:
    v = baseline_per_image(df)
    lo, hi = bootstrap_ci(v.to_numpy(), n, seed, ci)
    return {"mean": float(v.mean()), "lo": lo, "hi": hi, "n_images": int(v.size)}


def delta_dice(df: pd.DataFrame, a: float = 0.0, b: float = 0.2, n=2000, seed=42, ci=0.95):
    """Paired per-image Dice drop from jitter a to b, with bootstrap CI."""
    t = per_image(df, "box_jitter")
    levels = {round(float(x), 9): x for x in t.index}
    if round(a, 9) not in levels or round(b, 9) not in levels:
        return {"d0": math.nan, "d1": math.nan, "delta": math.nan, "lo": math.nan, "hi": math.nan}
    d = (t.loc[levels[round(a, 9)]] - t.loc[levels[round(b, 9)]]).dropna()
    lo, hi = bootstrap_ci(d.to_numpy(), n, seed, ci)
    return {"d0": float(t.loc[levels[round(a, 9)]].mean()),
            "d1": float(t.loc[levels[round(b, 9)]].mean()),
            "delta": float(d.mean()), "lo": lo, "hi": hi}  # fmt: skip


def crossover(levels, means, reference: float) -> dict:
    """First level where the curve falls below the reference, plus linear interpolation.

    Returns {"first_level", "interp", "label"}; label is e.g. "20%", "> 50%" or "below at 0%".
    """
    lv = np.asarray(levels, dtype=float)
    mv = np.asarray(means, dtype=float)
    order = np.argsort(lv)
    lv, mv = lv[order], mv[order]
    below = np.nonzero(mv < reference)[0]
    if below.size == 0:
        return {"first_level": math.nan, "interp": math.nan, "label": f"> {lv[-1]:.0%}"}
    i = int(below[0])
    if i == 0:
        return {"first_level": float(lv[0]), "interp": float(lv[0]),
                "label": f"below at {lv[0]:.0%}"}  # fmt: skip
    x0, x1, y0, y1 = lv[i - 1], lv[i], mv[i - 1], mv[i]
    interp = x0 + (y0 - reference) * (x1 - x0) / (y0 - y1)
    return {"first_level": float(lv[i]), "interp": float(interp), "label": f"{lv[i]:.0%}"}


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as the input; NaN kept)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.nonzero(~np.isnan(p))[0]
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out


def wilcoxon_by_level(a: pd.DataFrame, b: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank per level (a vs b), Holm-corrected across levels.

    a: level x image table; b: same shape, or a per-image Series (prompt-free baseline).
    """
    rows = []
    for level in a.index:
        x = a.loc[level]
        y = b if isinstance(b, pd.Series) else b.loc[level]
        pair = pd.concat([x, y], axis=1, join="inner").dropna()
        d = pair.iloc[:, 0] - pair.iloc[:, 1]
        if len(pair) < 2 or np.allclose(d, 0):
            p = math.nan if len(pair) < 2 else 1.0
        else:
            p = float(stats.wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1]).pvalue)
        rows.append({"level": float(level), "n": len(pair), "mean_diff": float(d.mean()),
                     "p": p})  # fmt: skip
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    return out
