"""Generate figures/debug_prompts.png for visual inspection of the prompt simulator.

Usage: python scripts/make_debug_prompts.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from promptshift.paths import get_paths
from promptshift.prompts.debug import plot_debug_prompts


def main() -> None:
    paths = get_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=paths.configs / "default.yaml")
    parser.add_argument("--out", type=Path, default=paths.figures / "debug_prompts.png")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = plot_debug_prompts(args.out, cfg["prompts"], global_seed=cfg["global_seed"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
