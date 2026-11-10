"""Figures for P1-12 (spec 6.8). Static PNG (300 dpi) + PDF for the report.

Colors: the first three slots of the validated reference categorical palette (they pass the
all-pairs colorblind check), one per foundation model in a fixed order. The prompt-free baselines
are reference lines in neutral ink, told apart by line style (U-Net dashed, classical dotted)
and direct labels, so identity never depends on color alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODEL_COLORS = {"sam_vit_b": "#2a78d6", "sam_vit_h": "#eb6834", "medsam": "#1baf7a"}
MODEL_LABELS = {"sam_vit_b": "SAM ViT-B", "sam_vit_h": "SAM ViT-H", "medsam": "MedSAM"}
BASELINE_STYLE = {
    "unet": {"color": "#3d3c3a", "ls": "--", "label": "U-Net (no prompt)"},
    "classical": {"color": "#7a7974", "ls": ":", "label": "Classical (no prompt)"},
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def save(fig, out_stem: Path) -> list[Path]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_stem.with_suffix(".png"), out_stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def curve_figure(curves: dict[str, pd.DataFrame], baselines: dict[str, dict], *, xlabel: str,
                 title: str, percent_x: bool, ref_x: float | None = None):  # fmt: skip
    """curves: model -> DataFrame(level, mean, lo, hi); baselines: method -> {mean, lo, hi}."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs_all = [c["level"].to_numpy() for c in curves.values()] or [np.array([0.0, 1.0])]
    x_min, x_max = min(x.min() for x in xs_all), max(x.max() for x in xs_all)
    for method, b in baselines.items():
        st = BASELINE_STYLE[method]
        ax.axhspan(b["lo"], b["hi"], color=st["color"], alpha=0.08, lw=0)
        ax.axhline(b["mean"], color=st["color"], ls=st["ls"], lw=1.5)
        ax.annotate(f"{st['label']} {b['mean']:.2f}", (x_min, b["mean"]),
                    xytext=(2, -12 if b["mean"] > 0.9 else 3),
                    textcoords="offset points", color=MUTED, fontsize=8)  # fmt: skip
    for model in MODEL_COLORS:
        if model not in curves:
            continue
        c = curves[model].sort_values("level")
        col = MODEL_COLORS[model]
        ax.fill_between(c["level"], c["lo"], c["hi"], color=col, alpha=0.18, lw=0)
        ax.plot(c["level"], c["mean"], color=col, lw=2, marker="o", ms=4,
                label=MODEL_LABELS[model])  # fmt: skip
        ax.annotate(MODEL_LABELS[model], (c["level"].iloc[-1], c["mean"].iloc[-1]),
                    xytext=(5, 0), textcoords="offset points", va="center", color=INK,
                    fontsize=8)  # fmt: skip
    if ref_x is not None:
        ax.axvline(ref_x, color=GRID, lw=1)
    ax.set_xlim(x_min - 0.02 * (x_max - x_min), x_max + 0.18 * (x_max - x_min))
    ax.set_ylim(0, 1)
    if percent_x:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _style(ax, xlabel, "Dice (mean over images, 95% CI)", title)
    if curves:
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def dice_vs_boxiou_figure(raw: dict[str, pd.DataFrame], n_bins: int = 10):
    """Binned mean Dice against the prompt box IoU (box jitter + box scale rows)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bins = np.linspace(0, 1, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    for model in MODEL_COLORS:
        if model not in raw:
            continue
        df = raw[model]
        ok = df[(df["flag"] == "ok") & df["perturb"].isin(["box_jitter", "box_scale"])]
        idx = np.clip(np.digitize(ok["box_iou_mean"], bins) - 1, 0, len(centers) - 1)
        g = ok.groupby(idx)["dice"].agg(["mean", "count"])
        g = g[g["count"] >= 5]  # hide bins with almost no rows
        ax.plot(centers[g.index], g["mean"], color=MODEL_COLORS[model], lw=2, marker="o", ms=4,
                label=MODEL_LABELS[model])  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax, "Prompt box IoU with the GT box (bins of 0.1)", "Dice (mean of rows in bin)",
           "Dice vs. prompt quality (box jitter + box scale)")  # fmt: skip
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return fig


def training_curve_figure(history: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    ax[0].plot(history["epoch"], history["train_loss"], color=INK, lw=1.5)
    _style(ax[0], "epoch", "train loss (BCE + Dice)", "U-Net training loss")
    ax[1].plot(history["epoch"], history["val_dice"], color=INK, lw=1.5)
    best = history.loc[history["val_dice"].idxmax()]
    ax[1].plot(best["epoch"], best["val_dice"], "o", color=INK, ms=6)
    ax[1].annotate(f"best {best['val_dice']:.3f} @ {int(best['epoch'])}",
                   (best["epoch"], best["val_dice"]), xytext=(4, -12),
                   textcoords="offset points", color=MUTED, fontsize=8)  # fmt: skip
    _style(ax[1], "epoch", "val Dice", "U-Net validation Dice")
    fig.tight_layout()
    return fig


def examples_figure(panels: list[dict], levels: list[float], models: list[str]):
    """panels: one dict per image with image, gt, boxes{level: [box,...]}, preds{(model, level)}."""
    n_rows, n_cols = len(panels), 2 + len(levels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows), squeeze=False)
    for r, p in enumerate(panels):
        axes[r, 0].imshow(p["image"])
        axes[r, 0].set_title(p["image_id"][:12], fontsize=7, color=MUTED)
        axes[r, 1].imshow(p["image"])
        axes[r, 1].contour(p["gt"], levels=[0.5], colors=["white"], linewidths=1.2)
        axes[r, 1].set_title("GT", fontsize=8, color=INK)
        for c, level in enumerate(levels, start=2):
            ax = axes[r, c]
            ax.imshow(p["image"])
            for x0, y0, x1, y1 in p["boxes"][level]:
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white",
                                           lw=1.0, ls="--"))  # fmt: skip
            for m in models:
                pred = p["preds"].get((m, level))
                if pred is not None and pred.any():
                    ax.contour(pred, levels=[0.5], colors=[MODEL_COLORS[m]], linewidths=1.2)
            ax.set_title(f"jitter {level:.0%}", fontsize=8, color=INK)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Line2D([], [], color=MODEL_COLORS[m], lw=2, label=MODEL_LABELS[m])
               for m in models]  # fmt: skip
    handles += [plt.Line2D([], [], color=MUTED, lw=1, ls="--", label="prompt box (white)"),
                plt.Line2D([], [], color=MUTED, lw=1.2, label="GT outline (white)")]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig
