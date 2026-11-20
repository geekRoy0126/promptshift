"""Run/resume Role C point experiments. Use --smoke-images 3 before a GPU run.

Writes metrics and click traces to results/raw/<exp_id>/. No MedSAM point results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from promptshift.clicks import click_grid, evaluate_click_image, validate_click_config
from promptshift.config import load_config
from promptshift.data import get_dataset
from promptshift.models import load_model
from promptshift.paths import get_paths
from promptshift.runner import _free_gpu, git_sha

KEY = ["image_id", "perturb", "level", "seed"]


def atomic_csv(df, path):
    temporary = path.with_suffix(".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def run(cfg, paths, ds=None, factory=None, device="auto"):
    validate_click_config(cfg)
    if ds is None:
        ds = get_dataset(
            cfg["dataset"], paths.data, cfg["data"], paths.splits / f"{cfg['dataset']}.json"
        )
    ids = ds.ids(cfg["split"])
    if cfg.get("max_images") is not None:
        ids = ids[: cfg["max_images"]]
    if not ids:
        raise ValueError("empty image split")
    folder = paths.raw_results / cfg["exp_id"]
    folder.mkdir(parents=True, exist_ok=True)
    # A run's protocol and cohort are immutable when resuming.
    encoded = json.dumps(dict(config=cfg, image_ids=ids), sort_keys=True).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    manifest = folder / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Protocol/cohort changed: use a new experiment ID")
    if not manifest.exists():
        manifest.write_text(
            json.dumps(dict(config=cfg, image_ids=ids, fingerprint=fingerprint), indent=2)
        )
    wanted = set(click_grid(cfg))
    summaries = []
    for name in cfg["models"]:
        file, trace_file = folder / f"{name}.csv", folder / f"{name}_clicks.csv"
        previous = pd.read_csv(file, dtype={"image_id": str}) if file.exists() else pd.DataFrame()
        done = set()
        if len(previous):
            if previous.duplicated(KEY).any():
                raise ValueError(f"Duplicate keys in {file}")
            for image_id, group in previous.groupby("image_id"):
                keys = set(group[["perturb", "level", "seed"]].itertuples(index=False, name=None))
                if keys == wanted and group.flag.eq("ok").all():
                    done.add(image_id)
            previous = previous[previous.image_id.isin(done)]
        traces = (
            pd.read_csv(trace_file, dtype={"image_id": str})
            if trace_file.exists()
            else pd.DataFrame()
        )
        if len(traces):
            traces = traces[traces.image_id.isin(done)]
        # Missing trace blocks invalidate a completed result block too.
        if done:
            traced = set(traces.image_id) if len(traces) else set()
            done &= traced
            previous = previous[previous.image_id.isin(done)]
            traces = traces[traces.image_id.isin(done)] if len(traces) else traces
        todo = [i for i in ids if i not in done]
        model = (
            (factory(name) if factory else load_model(name, paths.weights, device))
            if todo
            else None
        )
        errors = []
        started = time.perf_counter()
        try:
            for number, image_id in enumerate(todo, 1):
                meta = dict(
                    exp_id=cfg["exp_id"],
                    dataset=cfg["dataset"],
                    split=cfg["split"],
                    model=name,
                    git_sha=git_sha(paths.repo),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                try:
                    rows, points = evaluate_click_image(model, ds.get(image_id), cfg, meta)
                except Exception as e:
                    errors.append(dict(image_id=image_id, error=f"{type(e).__name__}: {e}"))
                    print(name, image_id, errors[-1]["error"], flush=True)
                    continue
                previous = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
                traces = pd.concat([traces, pd.DataFrame(points)], ignore_index=True)
                # Write trace first. A crash before metrics will safely rerun this image.
                atomic_csv(traces, trace_file)
                atomic_csv(previous, file)
                print(f"{name}: {number}/{len(todo)} new images; {len(previous)} rows", flush=True)
        finally:
            del model
            _free_gpu()
        summary = dict(
            model=name,
            n_images=len(ids),
            configs_per_image=len(wanted),
            rows=len(previous),
            expected_rows=len(ids) * len(wanted),
            errors=errors,
            already_done=len(done),
            seconds=time.perf_counter() - started,
            complete=len(previous) == len(ids) * len(wanted) and not errors,
        )
        (folder / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke-images", type=int)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.smoke_images is not None:
        if args.smoke_images < 1:
            p.error("smoke-images must be positive")
        cfg.update(exp_id=cfg["exp_id"] + "_smoke", max_images=args.smoke_images)
    summaries = run(cfg, get_paths(args.root).ensure(), device=args.device)
    print(json.dumps(summaries, indent=2))
    return 0 if all(s["complete"] for s in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
