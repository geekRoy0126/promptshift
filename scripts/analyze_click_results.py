"""Analyze full point runs only: image bootstrap, recovery and paired model tests."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from promptshift.analysis import plots
from promptshift.analysis.stats import bootstrap_ci, curve, per_image, wilcoxon_by_level
from promptshift.clicks import click_grid, validate_click_config, validate_click_trace
from promptshift.config import load_config
from promptshift.paths import get_paths


def analyze(cfg, paths, out=None):
    validate_click_config(cfg)
    if cfg.get("max_images") is not None:
        raise ValueError("smoke runs cannot be analyzed as full experiments")
    out = out or paths.summary_results / cfg["exp_id"]
    source = paths.raw_results / cfg["exp_id"]
    protocol = json.loads((source / "protocol.json").read_text())
    if protocol["config"] != cfg:
        raise ValueError("Analysis config differs from the run protocol")
    ids = set(json.loads((paths.splits / f"{cfg['dataset']}.json").read_text())[cfg["split"]])
    wanted = set(click_grid(cfg))
    raw = {}
    for model in cfg["models"]:
        df = pd.read_csv(source / f"{model}.csv", dtype={"image_id": str})
        keys = ["image_id", "perturb", "level", "seed"]
        if set(df.image_id) != ids or df.duplicated(keys).any() or not df.flag.eq("ok").all():
            raise ValueError(f"{model}: invalid IDs, duplicates, or row statuses")
        for col, expected in [
            ("model", model),
            ("dataset", cfg["dataset"]),
            ("split", cfg["split"]),
            ("exp_id", cfg["exp_id"]),
        ]:
            if not df[col].eq(expected).all():
                raise ValueError(f"{model}: mixed {col}")
        for _, g in df.groupby("image_id"):
            if set(g[["perturb", "level", "seed"]].itertuples(index=False, name=None)) != wanted:
                raise ValueError(f"{model}: incomplete click grid")
        if not np.isfinite(df.dice).all() or not df.dice.between(0, 1).all():
            raise ValueError(f"{model}: invalid Dice")
        trace = pd.read_csv(source / f"{model}_clicks.csv", dtype={"image_id": str})
        validate_click_trace(df, trace)
        raw[model] = df
    out.mkdir(parents=True, exist_ok=True)
    a = cfg["analysis"]
    boot = dict(n=a["bootstrap_n"], seed=a["bootstrap_seed"], ci=a["ci"])
    metrics, recovery, tests = [], [], []
    for perturb in ["point_offset", "corrective_clicks"]:
        curves = {m: curve(df, perturb, **boot) for m, df in raw.items()}
        for m, c in curves.items():
            metrics.extend(c.assign(model=m, perturb=perturb).to_dict("records"))
        fig = plots.curve_figure(
            curves,
            {},
            xlabel="Click offset (center to inner boundary)"
            if perturb == "point_offset"
            else "Additional corrective clicks (0 = initial click)",
            title=f"{cfg['dataset']} {cfg['split']}: {perturb.replace('_', ' ')}",
            percent_x=perturb == "point_offset",
        )
        plots.save(fig, out / perturb)
        tables = {m: per_image(df, perturb) for m, df in raw.items()}
        for left, right in itertools.combinations(tables, 2):
            w = wilcoxon_by_level(tables[left], tables[right])
            tests.extend(w.assign(left=left, right=right, perturb=perturb).to_dict("records"))
        if perturb == "corrective_clicks":
            for model, table in tables.items():
                for level in table.index[table.index > 0]:
                    gain = table.loc[level] - table.loc[0]
                    lo, hi = bootstrap_ci(gain, **boot)
                    recovery.append(
                        dict(
                            model=model,
                            corrections=int(level),
                            mean_dice_gain=float(gain.mean()),
                            lo=lo,
                            hi=hi,
                            n_images=len(gain),
                        )
                    )
    pd.DataFrame(metrics).to_csv(out / "dice_curves.csv", index=False)
    pd.DataFrame(recovery).to_csv(out / "paired_recovery.csv", index=False)
    pd.DataFrame(
        tests, columns=["level", "n", "mean_diff", "p", "p_holm", "left", "right", "perturb"]
    ).to_csv(out / "paired_tests.csv", index=False)
    status = dict(
        complete=True,
        images=len(ids),
        models=cfg["models"],
        rows_per_model=len(ids) * len(wanted),
        initial_offset=cfg["clicks"]["initial_offset"],
        correction_protocol="Largest error component, deepest unused pixel; GT oracle",
        comparison="Same initial clicks; adaptive later clicks can differ across models",
        caution="Image-level split; no MedSAM point experiment; no claim of clinical use",
    )
    (out / "analysis_status.json").write_text(json.dumps(status, indent=2))
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path)
    args = p.parse_args()
    print(json.dumps(analyze(load_config(args.config), get_paths(args.root)), indent=2))


if __name__ == "__main__":
    main()
