"""Make the P1-12 figures from the result CSVs (spec 6.8). PNG (300 dpi) + PDF.

Usage:
    python scripts/make_figures.py [--exp-id p1_kvasir_pilot] [--root PATH]
    python scripts/make_figures.py --examples-only --out-dir <dir> [--models ...]  (needs weights)

From CSVs (no models needed):
    figures/p1_kvasir_box_jitter, p1_kvasir_box_scale, p1_kvasir_dice_vs_boxiou,
    p1_kvasir_unet_curve (from unet_history.csv)
With --examples / --examples-only (runs the models on 3 test images):
    p1_kvasir_examples.png
Missing inputs are skipped with a message, never faked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from promptshift.analysis import plots
from promptshift.analysis.leakage import leakage_label
from promptshift.analysis.stats import baseline_summary, curve
from promptshift.config import load_config
from promptshift.data import get_dataset
from promptshift.models import load_model, select_device
from promptshift.paths import get_paths
from promptshift.prompts import make_prompt

MODELS = ["sam_vit_b", "sam_vit_h", "medsam"]
BASELINES = ["unet", "classical"]
EXAMPLE_LEVELS = [0.0, 0.1, 0.3, 0.5]


def read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, dtype={"image_id": str}) if path.is_file() else None


def pick_example_ids(ds, ids, n=3, seed=42) -> list[str]:
    single = [i for i in ids if len(ds.get(i).objects) == 1]
    pick = np.random.default_rng(seed).choice(len(single), size=min(n, len(single)), replace=False)
    return sorted(single[i] for i in pick)


def examples(cfg, paths, models, out_stem: Path) -> None:
    name = cfg["dataset"]
    ds = get_dataset(name, paths.data, cfg["data"], paths.splits / f"{name}.json")
    ids = pick_example_ids(ds, ds.ids(cfg["split"]), seed=cfg["global_seed"])
    panels = []
    for image_id in ids:
        s = ds.get(image_id)
        boxes = {}
        for level in EXAMPLE_LEVELS:
            boxes[level] = [
                make_prompt("box_jitter", level, obj, global_seed=cfg["global_seed"],
                            dataset=s.dataset, image_id=s.image_id, object_idx=k, seed=0,
                            params=cfg["prompts"]["box_jitter"]).box
                for k, obj in enumerate(s.objects)
            ]  # fmt: skip
        panels.append({"image_id": image_id, "image": s.image, "gt": s.mask, "boxes": boxes,
                       "preds": {}})  # fmt: skip
    for m in models:
        model = load_model(m, paths.weights, select_device("auto"))
        for p in panels:
            model.set_image(p["image"])
            for level in EXAMPLE_LEVELS:
                preds = [model.predict_box(np.asarray(b)) for b in p["boxes"][level]]
                pred = np.logical_or.reduce(preds)
                p["preds"][(m, level)] = pred
        del model
    fig = plots.examples_figure(panels, EXAMPLE_LEVELS, models)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plots.plt.close(fig)
    print(f"wrote {out_stem.with_suffix('.png')} (images {ids})")


def main() -> int:
    paths = get_paths()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--exp-id", default="p1_kvasir_pilot")
    parser.add_argument("--config", type=Path, default=paths.configs / "p1_kvasir_pilot.yaml")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: repo figures/")
    parser.add_argument("--examples", action="store_true", help="also run models for examples")
    parser.add_argument("--examples-only", action="store_true")
    parser.add_argument("--models", nargs="+", default=MODELS, help="models for the examples")
    args = parser.parse_args()
    paths = get_paths(args.root)
    cfg = load_config(args.config)
    out = args.out_dir or paths.figures
    a = cfg["analysis"]
    boot = {"n": a["bootstrap_n"], "seed": a["bootstrap_seed"], "ci": a["ci"]}
    exp = paths.raw_results / args.exp_id

    if not args.examples_only:
        raw = {m: df for m in MODELS if (df := read(exp / f"{m}.csv")) is not None}
        base = {b: baseline_summary(df, **boot) for b in BASELINES
                if (df := read(exp / f"{b}.csv")) is not None}  # fmt: skip
        print(f"models: {sorted(raw)}; baselines: {sorted(base)}")
        seen = leakage_label(paths.docs, "kvasir")
        if raw:
            for perturb, xlabel, pct, ref in [
                ("box_jitter", "box jitter r (fraction of box size)", True, None),
                ("box_scale", "box scale s", False, 1.0),
            ]:
                curves = {m: curve(df, perturb, **boot) for m, df in raw.items()}
                fig = plots.curve_figure(
                    curves, base, xlabel=xlabel, percent_x=pct, ref_x=ref,
                    title=f"Kvasir-SEG test (MedSAM: {seen}): Dice vs. "
                          f"{perturb.replace('_', ' ')} (preliminary)",
                )  # fmt: skip
                print("wrote", *plots.save(fig, out / f"p1_kvasir_{perturb}"))
            print("wrote", *plots.save(plots.dice_vs_boxiou_figure(raw),
                                       out / "p1_kvasir_dice_vs_boxiou"))  # fmt: skip
        else:
            print(f"skip curves: no model CSVs in {exp}")
        hist = read(exp / "unet_history.csv")
        if hist is not None and len(hist):
            print("wrote", *plots.save(plots.training_curve_figure(hist),
                                       out / "p1_kvasir_unet_curve"))  # fmt: skip
        else:
            print("skip U-Net curve: no unet_history.csv")
    if args.examples or args.examples_only:
        examples(cfg, paths, args.models, out / "p1_kvasir_examples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
