"""Descriptive comparison of validated box runs; datasets are not paired cohorts."""

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from promptshift.analysis.stats import curve, delta_dice
from promptshift.paths import REPO_ROOT


def main():
    root = REPO_ROOT
    out = root / "results/summary/busi_kvasir"
    out.mkdir(parents=True, exist_ok=True)
    records, drops, sources = [], [], []
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for row, (dataset, exp, n) in enumerate(
        [("BUSI", "c_busi_box", 129), ("Kvasir-SEG", "p1_kvasir_pilot", 200)]
    ):
        split_name = "busi" if dataset == "BUSI" else "kvasir_seg"
        ids = set(json.loads((root / f"splits/{split_name}.json").read_text())["test"])
        for model, color in [("sam_vit_b", "#2a78d6"), ("medsam", "#1baf7a")]:
            path = root / f"results/raw/{exp}/{model}.csv"
            df = pd.read_csv(path, dtype={"image_id": str})
            expected = {
                ("box_jitter", level, s)
                for level in [0, 0.05, 0.1, 0.2, 0.3, 0.5]
                for s in [0, 1, 2]
            } | {("box_scale", level, 0) for level in [0.8, 0.9, 1, 1.1, 1.25, 1.5]}
            assert len(df) == n * 24 and set(df.image_id) == ids and len(ids) == n
            assert not df.duplicated(["image_id", "perturb", "level", "seed"]).any()
            assert (
                df.flag.eq("ok").all() and df.model.eq(model).all() and df.dice.between(0, 1).all()
            )
            assert all(
                set(g[["perturb", "level", "seed"]].itertuples(index=False, name=None)) == expected
                for _, g in df.groupby("image_id")
            )
            sources.append(
                dict(
                    dataset=dataset,
                    model=model,
                    path=str(path.relative_to(root)),
                    rows=len(df),
                    images=n,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
            drops.append(dict(dataset=dataset, model=model, n_images=n, **delta_dice(df)))
            for col, perturb in enumerate(["box_jitter", "box_scale"]):
                c = curve(df, perturb)
                records.extend(
                    c.assign(dataset=dataset, model=model, perturb=perturb).to_dict("records")
                )
                ax = axes[row, col]
                ax.plot(c.level, c["mean"], marker="o", color=color, label=model)
                ax.fill_between(c.level, c.lo, c.hi, color=color, alpha=0.15)
                ax.set_title(f"{dataset} (n={n})")
                ax.set_xlabel("Box jitter" if col == 0 else "Box scale factor")
                ax.set_ylim(0, 1.03)
                ax.grid(alpha=0.2)
                if col == 0:
                    ax.set_ylabel("Image-mean Dice")
                if row == 0:
                    ax.legend(fontsize=9)
    fig.suptitle("SAM / MedSAM box robustness — image bootstrap 95% intervals")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(out / "curves.csv", index=False)
    pd.DataFrame(drops).to_csv(out / "dice_drop.csv", index=False)
    (out / "sources.json").write_text(
        json.dumps(
            dict(
                sources=sources,
                comparison="Descriptive only; no between-dataset paired test",
                seeds="Averaged within image before image bootstrap",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
