"""Classical baseline (P1-11, spec 6.7): grid search on train (val reported), then one
evaluation on test.

Usage: python scripts/run_classical.py [--config configs/p1_kvasir_pilot.yaml] [--root PATH]

Writes (under the storage root):
    results/raw/<exp_id>/classical.csv            test rows, runner format, perturb "none"
    results/raw/<exp_id>/classical_params.json    chosen parameters + selection record
    results/summary/classical_grid_<dataset>.csv  mean Dice of every grid point on train and val
Deterministic (OpenCV, no randomness), so a rerun gives the same numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from promptshift.baselines import baseline_row, write_rows
from promptshift.baselines.classical import grid, segment, segment_grid
from promptshift.config import load_config
from promptshift.data import get_dataset
from promptshift.metrics import dice
from promptshift.paths import get_paths
from promptshift.runner import git_sha

METHOD = "classical"


def grid_dice(ds, ids, params, desc) -> np.ndarray:
    """(n_images, n_params) Dice matrix."""
    out = np.zeros((len(ids), len(params)))
    for i, image_id in enumerate(tqdm(ids, desc=desc)):
        s = ds.get(image_id)
        preds = segment_grid(s.image, params)
        out[i] = [dice(preds[p], s.mask) for p in params]
    return out


def main() -> int:
    paths = get_paths()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=paths.configs / "p1_kvasir_pilot.yaml")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    paths = get_paths(args.root).ensure()
    cfg = load_config(args.config)
    ccfg = cfg["baselines"]["classical"]
    name, exp_id = cfg["dataset"], cfg["exp_id"]
    ds = get_dataset(name, paths.data, cfg["data"], paths.splits / f"{name}.json")
    params = grid(ccfg)
    sha = git_sha(paths.repo)
    print(f"{name}: {len(params)} grid points; select on {ccfg['select_on']}; git {sha}")

    scores = {s: grid_dice(ds, ds.ids(s), params, s) for s in ("train", "val")}
    table = pd.DataFrame([p.as_dict() for p in params])
    for s, d in scores.items():
        table[f"{s}_dice"] = d.mean(axis=0)
    best_i = int(np.argmax(table[f"{ccfg['select_on']}_dice"].to_numpy()))  # ties: first
    best = params[best_i]
    paths.summary_results.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths.summary_results / f"classical_grid_{name}.csv", index=False)
    print(table.sort_values("train_dice", ascending=False).head(5).to_string(index=False))
    print(f"chosen: {best.as_dict()}")

    rows = []
    test_ids = ds.ids(cfg["split"])
    for image_id in tqdm(test_ids, desc="test"):
        s = ds.get(image_id)
        t0 = time.perf_counter()
        pred = segment(s.image, best)
        ms = (time.perf_counter() - t0) * 1000
        rows.append(baseline_row(
            exp_id=exp_id, split=cfg["split"], method=METHOD, sample=s, pred=pred, ms=ms,
            sha=sha, nsd_tolerance_px=cfg["metrics"]["nsd_tolerance_px"],
        ))  # fmt: skip
    out_dir = paths.raw_results / exp_id
    df = write_rows(out_dir / f"{METHOD}.csv", rows)
    record = {
        "method": METHOD, "dataset": name, "params": best.as_dict(),
        "selection": f"max mean Dice on {ccfg['select_on']} "
                     f"({len(ds.ids(ccfg['select_on']))} images); test evaluated once",
        "train_dice": float(table.loc[best_i, "train_dice"]),
        "val_dice": float(table.loc[best_i, "val_dice"]),
        "test_dice": float(df["dice"].mean()), "n_test": len(df), "git_sha": sha,
        "grid": {k: ccfg[k] for k in ("channel", "polarity", "sigma", "radius")},
    }  # fmt: skip
    (out_dir / f"{METHOD}_params.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
