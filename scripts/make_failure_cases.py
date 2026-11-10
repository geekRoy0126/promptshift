"""Render ranked failures with real model inference; needs dataset and weights.

The ranking averages seeds; each panel reproduces one seed (default 0), and
reports its own Dice. Reasons are left blank for human review, not inferred.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from promptshift.analysis import plots
from promptshift.config import load_config
from promptshift.data import get_dataset
from promptshift.demo import overlay
from promptshift.metrics import dice
from promptshift.models import load_model
from promptshift.paths import get_paths
from promptshift.prompts import make_prompt


def render_cases(cfg, paths, ranking, out, top=6, seed=0):
    if top < 1:
        raise ValueError("top must be positive")
    ds = get_dataset(
        cfg["dataset"], paths.data, cfg["data"], paths.splits / f"{cfg['dataset']}.json"
    )
    ranked = pd.read_csv(ranking, dtype={"image_id": str})
    ranked = ranked[ranked.dataset.eq(cfg["dataset"])]
    ranked = ranked[ranked.dice_drop.notna()].sort_values("dice_drop", ascending=False)
    ranked = ranked.drop_duplicates(["model", "image_id", "perturb"]).head(top)
    if ranked.empty:
        raise ValueError("No ranked failure cases available")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name, group in ranked.groupby("model", sort=False):
        model = load_model(name, paths.weights, "auto")
        for row in group.itertuples():
            s = ds.get(row.image_id)
            model.set_image(s.image)
            fig, axes = plots.plt.subplots(1, 3, figsize=(10, 3.5))
            axes[0].imshow(s.image)
            axes[0].contour(s.mask, levels=[0.5], colors=["lime"])
            axes[0].set_title("Ground truth (green)")
            scores = []
            reference = 0.0 if row.perturb == "box_jitter" else 1.0
            for ax, level, title in zip(
                axes[1:], [reference, row.level], ["Reference", "Perturbed"], strict=True
            ):
                prompts = [
                    make_prompt(
                        row.perturb,
                        level,
                        obj,
                        global_seed=cfg["global_seed"],
                        dataset=s.dataset,
                        image_id=s.image_id,
                        object_idx=k,
                        seed=seed,
                        params=cfg["prompts"][row.perturb],
                    )
                    for k, obj in enumerate(s.objects)
                ]
                if not prompts or any(not p.valid for p in prompts):
                    ax.imshow(s.image)
                    ax.set_title(f"{title}: invalid prompt")
                    scores.append(np.nan)
                    continue
                pred = np.logical_or.reduce([model.predict_box(p.box) for p in prompts])
                rgb = s.image
                for k, p in enumerate(prompts):
                    rgb = overlay(rgb, pred if k == 0 else np.zeros_like(pred), p.box)
                ax.imshow(rgb)
                score = dice(pred, s.mask)
                scores.append(score)
                ax.set_title(f"{title} {level:g}; Dice={score:.3f}")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{name} | {s.image_id} | {row.perturb} | seed {seed}")
            fig.tight_layout()
            filename = f"case_{len(records) + 1:02d}.png"
            fig.savefig(out / filename, dpi=180, bbox_inches="tight")
            plots.plt.close(fig)
            records.append(
                dict(
                    image_id=s.image_id,
                    model=name,
                    perturb=row.perturb,
                    level=row.level,
                    seed=seed,
                    seed_mean_drop=row.dice_drop,
                    reference_dice=scores[0],
                    perturbed_dice=scores[1],
                    figure=filename,
                    reviewer_reason="",
                    reviewer_notes="",
                )
            )
        del model
    pd.DataFrame(records).to_csv(out / "review.csv", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--ranking", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg, paths = load_config(args.config), get_paths(args.root)
    summary = paths.summary_results / cfg["exp_id"]
    render_cases(
        cfg,
        paths,
        args.ranking or summary / "failures.csv",
        args.out_dir or summary / "failure_cases",
        args.top,
        args.seed,
    )


if __name__ == "__main__":
    main()
