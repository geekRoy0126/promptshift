"""Reproduce BUSI point-versus-baseline figures and paired statistical tables."""

from __future__ import annotations

import json

import pandas as pd

from promptshift.analysis import plots
from promptshift.analysis.stats import (
    baseline_per_image,
    baseline_summary,
    bootstrap_ci,
    curve,
    per_image,
    wilcoxon_by_level,
)
from promptshift.config import load_config
from promptshift.paths import get_paths


def main():
    paths = get_paths()
    cfg = load_config(paths.configs / "c_busi_clicks.yaml")
    out = paths.summary_results / "c_busi_clicks"
    for folder in ["c_busi_clicks", "c_busi_box"]:
        status = json.loads((paths.summary_results / folder / "analysis_status.json").read_text())
        if not status["complete"] or status.get("missing_baselines", []):
            raise ValueError("Run complete click and box/baseline analyses first")
    ids = set(json.loads((paths.splits / "busi.json").read_text())["test"])
    methods = {}
    for name in ["unet", "classical"]:
        df = pd.read_csv(paths.raw_results / "c_busi_box" / f"{name}.csv", dtype={"image_id": str})
        if (
            len(df) != len(ids)
            or set(df.image_id) != ids
            or not df.image_id.is_unique
            or not df.flag.eq("ok").all()
            or not df.dice.between(0, 1).all()
        ):
            raise ValueError(f"Incomplete baseline: {name}")
        methods[name] = df
    boot = dict(
        n=cfg["analysis"]["bootstrap_n"],
        seed=cfg["analysis"]["bootstrap_seed"],
        ci=cfg["analysis"]["ci"],
    )
    references = {name: baseline_summary(df, **boot) for name, df in methods.items()}
    pd.DataFrame([dict(model=name, **r) for name, r in references.items()]).to_csv(
        out / "baselines.csv", index=False
    )
    points = {
        name: pd.read_csv(
            paths.raw_results / "c_busi_clicks" / f"{name}.csv", dtype={"image_id": str}
        )
        for name in cfg["models"]
    }
    tests, differences = [], []
    for perturb in ["point_offset", "corrective_clicks"]:
        curves = {name: curve(df, perturb, **boot) for name, df in points.items()}
        fig = plots.curve_figure(
            curves,
            references,
            xlabel="Click offset (center to inner boundary)"
            if perturb == "point_offset"
            else "Additional corrective clicks (0 = initial click)",
            title=f"BUSI test: {perturb.replace('_', ' ')} vs. prompt-free baselines",
            percent_x=perturb == "point_offset",
        )
        plots.save(fig, out / f"{perturb}_with_baselines")
        for model, df in points.items():
            table = per_image(df, perturb)
            for baseline, reference in methods.items():
                series = baseline_per_image(reference)
                w = wilcoxon_by_level(table, series)
                tests.extend(
                    w.assign(model=model, baseline=baseline, perturb=perturb).to_dict("records")
                )
                for level in table.index:
                    paired = table.loc[level].reindex(sorted(ids)) - series.reindex(sorted(ids))
                    lo, hi = bootstrap_ci(paired, **boot)
                    differences.append(
                        dict(
                            model=model,
                            baseline=baseline,
                            perturb=perturb,
                            level=float(level),
                            mean_dice_difference=float(paired.mean()),
                            lo=lo,
                            hi=hi,
                            n_images=len(paired),
                        )
                    )
    pd.DataFrame(tests).to_csv(out / "paired_tests_vs_baselines.csv", index=False)
    pd.DataFrame(differences).to_csv(out / "paired_differences_vs_baselines.csv", index=False)
    history = pd.read_csv(paths.raw_results / "c_busi_box/unet_history.csv")
    plots.save(
        plots.training_curve_figure(history), paths.summary_results / "c_busi_box/unet_training"
    )


if __name__ == "__main__":
    main()
