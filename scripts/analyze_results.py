"""Dataset-independent Role C analysis; consumes real runner CSVs only.

python scripts/analyze_results.py --config configs/c_busi_box.yaml [--root PATH]
Missing models/baselines are recorded, never filled with synthetic scores.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from promptshift.analysis import plots
from promptshift.analysis.failures import rank_failures
from promptshift.analysis.leakage import leakage_label
from promptshift.analysis.stats import (
    baseline_per_image,
    baseline_summary,
    bootstrap_ci,
    crossover,
    curve,
    delta_dice,
    per_image,
    wilcoxon_by_level,
)
from promptshift.config import load_config
from promptshift.paths import get_paths
from promptshift.runner import KEY, build_grid

METHODS = ("sam_vit_b", "sam_vit_h", "medsam", "unet", "classical")


def read_results(path, cfg, method):
    df = pd.read_csv(path, dtype={"image_id": str})
    required = {
        "exp_id",
        "dataset",
        "split",
        "model",
        "image_id",
        "perturb",
        "level",
        "seed",
        "flag",
        "dice",
        "iou",
        "nsd",
        "hd95",
    }
    if required - set(df):
        raise ValueError(f"{path}: missing columns {sorted(required - set(df))}")
    if df.empty:
        raise ValueError(f"{path}: empty results")
    for col, expected in (
        ("dataset", cfg["dataset"]),
        ("split", cfg["split"]),
        ("model", method),
        ("exp_id", cfg["exp_id"]),
    ):
        if col in df and not df[col].eq(expected).all():
            raise ValueError(f"{path}: {col} does not match {expected}")
    if df.image_id.isna().any() or df.duplicated(KEY).any():
        raise ValueError(f"{path}: missing IDs or duplicate image/configuration rows")
    if not df.flag.isin(["ok", "invalid"]).all():
        raise ValueError(f"{path}: unrecognized row status")
    for metric in ("dice", "iou", "nsd"):
        v = df.loc[df.flag.eq("ok"), metric]
        if v.isna().any() or not v.between(0, 1).all():
            raise ValueError(f"{path}: invalid {metric} in ok rows")
    return df


def analyze(cfg, paths, out):
    out.mkdir(parents=True, exist_ok=True)
    source = paths.raw_results / cfg["exp_id"]
    raw = {
        m: read_results(source / f"{m}.csv", cfg, m)
        for m in METHODS
        if (source / f"{m}.csv").is_file()
    }
    if not raw:
        raise ValueError(f"No result CSVs found in {source}")
    # Avoid retaining a curve from an earlier run when all current rows are invalid.
    for perturb in cfg["perturbations"]:
        for ext in ("png", "pdf"):
            (out / f"{perturb}.{ext}").unlink(missing_ok=True)
    a = cfg["analysis"]
    boot = dict(n=a["bootstrap_n"], seed=a["bootstrap_seed"], ci=a["ci"])
    expected_ids = set(
        json.loads((paths.splits / f"{cfg['dataset']}.json").read_text())[cfg["split"]]
    )
    if cfg.get("max_images") is not None:
        raise ValueError("Analysis needs a full split config; smoke outputs are not final results")
    grid = {(p, round(lv, 9), sd) for p, lv, sd in build_grid(cfg)}
    integrity, summaries, drops, tests, crossings, failures = [], [], [], [], [], []
    for m, df in raw.items():
        wanted = {("none", 0.0, 0)} if m in ("unet", "classical") else grid
        complete = (
            all(
                {
                    (p, 0.0 if pd.isna(lv) and p == "none" else round(float(lv), 9), int(sd))
                    for p, lv, sd in g[["perturb", "level", "seed"]].itertuples(
                        index=False, name=None
                    )
                }
                == wanted
                for _, g in df.groupby("image_id")
            )
            and set(df.image_id) == expected_ids
        )
        integrity.append(
            dict(
                model=m,
                rows=len(df),
                images=df.image_id.nunique(),
                expected_images=len(expected_ids),
                complete=complete,
                invalid_rows=int(df.flag.eq("invalid").sum()),
                valid_rows=int(df.flag.eq("ok").sum()),
            )
        )
        for (perturb, level), group in df[df.flag.eq("ok")].groupby(["perturb", "level"]):
            for metric in ("dice", "iou", "nsd", "hd95"):
                v = group.groupby("image_id")[metric].mean()
                v = v[np.isfinite(v)]
                lo, hi = bootstrap_ci(v, **boot)
                summaries.append(
                    dict(
                        model=m,
                        perturb=perturb,
                        level=level,
                        metric=metric,
                        mean=v.mean(),
                        lo=lo,
                        hi=hi,
                        n_images=len(v),
                    )
                )
        if m not in ("unet", "classical"):
            drops.append(dict(model=m, **delta_dice(df, **boot)))
            for p in cfg["perturbations"]:
                if p in ("box_jitter", "box_scale"):
                    failures.append(rank_failures(df, p, 0.0 if p == "box_jitter" else 1.0))
    bases = {m: baseline_summary(df, **boot) for m, df in raw.items() if m in ("unet", "classical")}
    prompted = {m: df for m, df in raw.items() if m not in bases}
    for p in cfg["perturbations"]:
        curves = {m: curve(df, p, **boot) for m, df in prompted.items() if p in set(df.perturb)}
        curves = {m: c for m, c in curves.items() if not c.empty}
        if not curves:
            continue
        label = leakage_label(
            paths.docs, "BUSI" if cfg["dataset"] == "busi" else cfg["dataset"].split("_")[0]
        )
        title = f"{cfg['dataset']} {cfg['split']} (MedSAM: {label})"
        fig = plots.curve_figure(
            curves,
            bases,
            xlabel=p.replace("_", " "),
            title=title,
            percent_x=p != "box_scale",
            ref_x=1.0 if p == "box_scale" else None,
        )
        plots.save(fig, out / p)
        tables = {m: per_image(df, p) for m, df in prompted.items() if m in curves}
        for left, right in itertools.combinations(tables, 2):
            # Missing levels are excluded explicitly; image matching is done by ID.
            common = tables[left].index.intersection(tables[right].index)
            w = wilcoxon_by_level(tables[left].loc[common], tables[right].loc[common])
            tests.extend(w.assign(left=left, right=right, perturb=p).to_dict("records"))
        for m, table in tables.items():
            for b in bases:
                series = baseline_per_image(raw[b])
                w = wilcoxon_by_level(table, series)
                tests.extend(w.assign(left=m, right=b, perturb=p).to_dict("records"))
                if p == "box_jitter":
                    # Crossover uses the same complete paired cohort at every level.
                    ids = table.dropna(axis=1).columns.intersection(series.dropna().index)
                    if len(ids):
                        c = crossover(table.index, table[ids].mean(axis=1), series[ids].mean())
                        crossings.append(dict(model=m, baseline=b, n_images=len(ids), **c))
    for filename, rows in (
        ("integrity", integrity),
        ("metrics", summaries),
        ("dice_drop", drops),
        ("paired_tests", tests),
        ("crossover", crossings),
    ):
        columns = {
            "dice_drop": ["model", "d0", "d1", "delta", "lo", "hi"],
            "paired_tests": ["level", "n", "mean_diff", "p", "p_holm", "left", "right", "perturb"],
            "crossover": ["model", "baseline", "n_images", "first_level", "interp", "label"],
            "metrics": ["model", "perturb", "level", "metric", "mean", "lo", "hi", "n_images"],
        }
        pd.DataFrame(rows, columns=columns.get(filename)).to_csv(
            out / f"{filename}.csv", index=False
        )
    if failures:
        pd.concat(failures, ignore_index=True).sort_values("dice_drop", ascending=False).to_csv(
            out / "failures.csv", index=False
        )
    status = dict(
        dataset=cfg["dataset"],
        experiment=cfg["exp_id"],
        missing_methods=[m for m in cfg["models"] if m not in raw],
        missing_baselines=[m for m in ("unet", "classical") if m not in raw],
        complete=all(x["complete"] and x["valid_rows"] > 0 for x in integrity)
        and all(m in raw for m in cfg["models"]),
        notes=[
            "Seeds averaged within images before inference statistics.",
            "Holm correction is across levels within each model pair/perturbation.",
            "HD95 omits non-finite values; sample counts are exported.",
            "Crossover interpolation is descriptive, not a significance threshold.",
            "Failure rankings show drops; explanations require review.",
        ],
    )
    (out / "analysis_status.json").write_text(json.dumps(status, indent=2))
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--out-dir", type=Path)
    args = p.parse_args()
    cfg, paths = load_config(args.config), get_paths(args.root)
    status = analyze(cfg, paths, args.out_dir or paths.summary_results / cfg["exp_id"])
    print(json.dumps(status, indent=2))
    return 0 if status["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
