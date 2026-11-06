"""Run an experiment config. Resumable: rerun the same command after a disconnect.

Usage:
    python scripts/run_experiment.py --config configs/p1_kvasir_pilot.yaml
        [--models sam_vit_b medsam] [--max-images 5] [--device auto] [--root PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from promptshift.config import load_config
from promptshift.paths import get_paths
from promptshift.runner import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=None, help="override cfg models")
    parser.add_argument("--max-images", type=int, default=None, help="first N images of the split")
    parser.add_argument("--exp-id", default=None, help="override cfg exp_id (e.g. for smoke runs)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--root", type=Path, default=None, help="storage root (default: paths.py)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.exp_id:
        cfg["exp_id"] = args.exp_id
    paths = get_paths(args.root).ensure()
    summaries = run_experiment(
        cfg, paths, models=args.models, max_images=args.max_images, device=args.device
    )
    return 0 if all(s["complete"] and s["errors"] == 0 for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
