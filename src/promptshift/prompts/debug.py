"""Debug figure of simulated prompts (CLAUDE.md 6.3: figures/debug_prompts.png)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from promptshift.prompts import make_prompt


def synthetic_masks(height: int = 200, width: int = 260) -> dict[str, np.ndarray]:
    """Shapes that stress the simulator: a non-convex C shape, and an ellipse cut by the border."""
    yy, xx = np.mgrid[:height, :width]
    ring = (np.hypot(yy - 100, xx - 120) < 70) & (np.hypot(yy - 100, xx - 120) > 35)
    c_shape = ring & ~((xx > 120) & (np.abs(yy - 100) < 25))
    border = ((yy - 60) / 50.0) ** 2 + ((xx - 10) / 90.0) ** 2 < 1
    return {"c_shape": c_shape, "border_ellipse": border}


def plot_debug_prompts(out_path: str | Path, prompts_cfg: dict, global_seed: int = 42) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masks = synthetic_masks()
    panels = ["box_jitter", "box_scale", "point_offset"]
    fig, axes = plt.subplots(len(masks), len(panels), figsize=(4.2 * len(panels), 3.4 * len(masks)))
    cmap = plt.get_cmap("viridis")
    for i, (name, mask) in enumerate(masks.items()):
        for j, perturb in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(mask, cmap="gray", vmin=0, vmax=1.6)
            cfg = prompts_cfg[perturb]
            levels = cfg["levels"]
            for k, level in enumerate(levels):
                color = cmap(k / max(1, len(levels) - 1))
                for seed in cfg["seeds"]:
                    p = make_prompt(
                        perturb, level, mask, global_seed=global_seed, dataset="debug",
                        image_id=name, object_idx=0, seed=seed, params=cfg,
                    )
                    label = f"{level:g}" if seed == cfg["seeds"][0] else None
                    if p.kind == "box":
                        x0, y0, x1, y1 = p.box
                        ls = "-" if p.valid else ":"
                        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                                   ec=color, lw=1.2, ls=ls, label=label))
                    else:
                        ax.scatter(*p.coords[0], s=28, color=color, edgecolors="white",
                                   linewidths=0.6, label=label, zorder=3)
            ax.set_title(f"{name} · {perturb}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(fontsize=6, loc="lower right", title="level", title_fontsize=6)
    fig.suptitle("Simulated prompts (all seeds; dotted box = invalid)", fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
