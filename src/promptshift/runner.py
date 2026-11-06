"""Experiment runner (CLAUDE.md 6.5): encode each image once, decode every configuration,
write one long-format row per (image, configuration), resume after interruption.

Files per model in <results>/raw/<exp_id>/:
    <model>.csv            image-level rows (COLUMNS)
    <model>_objects.csv    per-object rows (OBJECT_COLUMNS)
    <model>_summary.json   counts from the last run

Resume: an image is "done" when its rows are complete (one valid row per configuration).
Rows of incomplete images, duplicated rows and truncated lines (no timestamp) are removed
on start, then those images are redone. Rows are written in whole-image blocks.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from promptshift.metrics import compute_metrics, dice, iou
from promptshift.models import load_model
from promptshift.prompts import PERTURBATIONS, make_prompt

log = logging.getLogger("promptshift.runner")

COLUMNS = [
    "exp_id", "dataset", "split", "model", "image_id", "n_objects", "perturb", "level", "seed",
    "dice", "iou", "nsd", "hd95", "box_iou_mean", "pred_area_frac", "gt_area_frac",
    "encode_ms", "decode_ms", "flag", "git_sha", "timestamp",
]  # fmt: skip
OBJECT_COLUMNS = [
    "exp_id", "dataset", "split", "model", "image_id", "object_idx", "perturb", "level", "seed",
    "dice", "iou", "box_iou", "object_area_frac", "flag",
]  # fmt: skip
KEY = ["image_id", "perturb", "level", "seed"]
OBJECT_KEY = ["image_id", "object_idx", "perturb", "level", "seed"]
NAN = math.nan


def build_grid(cfg: dict) -> list[tuple[str, float, int]]:
    grid = []
    for perturb in cfg["perturbations"]:
        if perturb not in PERTURBATIONS:
            raise KeyError(f"unknown perturbation {perturb!r}")
        pcfg = cfg["prompts"][perturb]
        grid += [(perturb, float(lv), int(sd)) for lv in pcfg["levels"] for sd in pcfg["seeds"]]
    return grid


def git_sha(repo: Path) -> str:
    """Short SHA, with -dirty if the working tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError) as e:
        log.warning("git sha unavailable: %s", e)
        return "unknown"


def _level_key(series: pd.Series) -> pd.Series:
    return series.astype(float).round(9)


def repair_and_find_done(csv_path: Path, obj_path: Path, n_configs: int) -> tuple[set[str], int]:
    """Drop broken / duplicate / incomplete rows; return (complete image ids, rows dropped)."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        csv_path.unlink(missing_ok=True)
        obj_path.unlink(missing_ok=True)
        return set(), 0
    df = pd.read_csv(csv_path, dtype={"image_id": str}, on_bad_lines="skip")
    n0 = len(df)
    df = df[df["timestamp"].notna() & df["flag"].isin(["ok", "invalid"])]
    df = df.assign(_lv=_level_key(df["level"])).drop_duplicates(
        ["image_id", "perturb", "_lv", "seed"], keep="first"
    )
    counts = df.groupby("image_id").size()
    done = set(counts[counts == n_configs].index)
    df = df[df["image_id"].isin(done)].drop(columns="_lv")
    dropped = n0 - len(df)
    if dropped:
        log.warning("%s: removed %d broken/duplicate/incomplete rows", csv_path.name, dropped)
        _atomic_write(df, csv_path, COLUMNS)
    if obj_path.exists():
        odf = pd.read_csv(obj_path, dtype={"image_id": str}, on_bad_lines="skip")
        odf = odf[odf["flag"].isin(["ok", "invalid"]) & odf["image_id"].isin(done)]
        odf = odf.assign(_lv=_level_key(odf["level"])).drop_duplicates(
            ["image_id", "object_idx", "perturb", "_lv", "seed"], keep="first"
        )
        _atomic_write(odf.drop(columns="_lv"), obj_path, OBJECT_COLUMNS)
    return done, dropped


INT_COLUMNS = ("n_objects", "seed", "object_idx")


def _atomic_write(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    """Rewrite via a temp file; restore integer columns (a truncated line makes them float)."""
    df = df.reindex(columns=columns)
    for c in INT_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype("int64")
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _append(rows: list[dict], path: Path, columns: list[str]) -> None:
    if not rows:
        return
    pd.DataFrame(rows, columns=columns).to_csv(
        path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False
    )


def _predict(model, prompt) -> np.ndarray:
    if prompt.kind == "box":
        return model.predict_box(prompt.box)
    mask, _ = model.predict_points(prompt.coords, prompt.labels)
    return mask


def evaluate_image(model, sample, grid, cfg, meta) -> tuple[list[dict], list[dict], dict]:
    """All configurations for one encoded image -> (image rows, object rows, counters)."""
    h, w = sample.mask.shape
    n_px = h * w
    tol = cfg["metrics"]["nsd_tolerance_px"]
    t0 = time.perf_counter()
    model.set_image(sample.image)
    encode_ms = (time.perf_counter() - t0) * 1000
    rows, obj_rows = [], []
    counts = {"invalid": 0, "empty_pred": 0}
    base = {**meta, "dataset": sample.dataset, "image_id": sample.image_id}
    for perturb, level, seed in grid:
        prompts = [
            make_prompt(
                perturb,
                level,
                obj,
                global_seed=cfg["global_seed"],
                dataset=sample.dataset,
                image_id=sample.image_id,
                object_idx=k,
                seed=seed,
                params=cfg["prompts"][perturb],
            )  # fmt: skip
            for k, obj in enumerate(sample.objects)
        ]
        cfg_key = {"perturb": perturb, "level": level, "seed": seed}
        box_ious = [getattr(p, "box_iou", NAN) for p in prompts]
        invalid = not prompts or any(not p.valid for p in prompts)
        if invalid:
            counts["invalid"] += 1
            m = {"dice": NAN, "iou": NAN, "nsd": NAN, "hd95": NAN, "pred_area_frac": NAN}
            decode_ms, obj_preds = NAN, [None] * len(prompts)
        else:
            t1 = time.perf_counter()
            obj_preds = [_predict(model, p) for p in prompts]
            decode_ms = (time.perf_counter() - t1) * 1000
            union = np.logical_or.reduce(obj_preds)
            m = compute_metrics(union, sample.mask, tol)
            counts["empty_pred"] += int(m.pop("pred_empty"))
            m.pop("gt_area_frac")
        rows.append({
            **base, **cfg_key, **m, "n_objects": len(sample.objects),
            "box_iou_mean": float(np.mean(box_ious)) if prompts else NAN,
            "gt_area_frac": float(sample.mask.sum()) / n_px,
            "encode_ms": encode_ms, "decode_ms": decode_ms,
            "flag": "invalid" if invalid else "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })  # fmt: skip
        for k, (obj, p, pred) in enumerate(zip(sample.objects, prompts, obj_preds, strict=True)):
            obj_rows.append({
                **{c: base[c] for c in ("exp_id", "dataset", "split", "model", "image_id")},
                **cfg_key, "object_idx": k,
                "dice": NAN if pred is None else dice(pred, obj),
                "iou": NAN if pred is None else iou(pred, obj),
                "box_iou": getattr(p, "box_iou", NAN),
                "object_area_frac": float(obj.sum()) / n_px,
                "flag": "invalid" if invalid else "ok",
            })  # fmt: skip
    return rows, obj_rows, counts


def run_model(
    model_name: str,
    model,
    ds,
    image_ids: list[str],
    cfg: dict,
    out_dir: Path,
    sha: str,
    progress: bool = True,
) -> dict:
    grid_all = build_grid(cfg)
    grid = [g for g in grid_all if PERTURBATIONS[g[0]].kind == "box" or model.supports_points]
    skipped_cfgs = len(grid_all) - len(grid)
    if skipped_cfgs:
        log.warning(
            "%s: skipping %d point configurations (no point support)", model_name, skipped_cfgs
        )
    csv_path = out_dir / f"{model_name}.csv"
    obj_path = out_dir / f"{model_name}_objects.csv"
    done, dropped = repair_and_find_done(csv_path, obj_path, len(grid))
    todo = [i for i in image_ids if i not in done]
    meta = {"exp_id": cfg["exp_id"], "split": cfg["split"], "model": model_name, "git_sha": sha}
    summary = {
        "model": model_name, "n_images": len(image_ids), "n_configs": len(grid),
        "skipped_point_configs": skipped_cfgs, "already_done": len(done & set(image_ids)),
        "repaired_rows_dropped": dropped, "processed": 0, "errors": 0, "error_ids": [],
        "invalid_rows": 0, "empty_predictions": 0,
    }  # fmt: skip
    buf, obj_buf = [], []
    flush_every = int(cfg["runner"]["flush_every"])
    iterator = todo
    if progress:
        from tqdm import tqdm

        iterator = tqdm(todo, desc=model_name, unit="img")
    for n, image_id in enumerate(iterator, 1):
        try:
            sample = ds.get(image_id)
            rows, obj_rows, counts = evaluate_image(model, sample, grid, cfg, meta)
        except Exception:  # noqa: BLE001 - logged and counted, run continues
            summary["errors"] += 1
            summary["error_ids"].append(image_id)
            log.error("%s %s failed:\n%s", model_name, image_id, traceback.format_exc())
            continue
        buf += rows
        obj_buf += obj_rows
        summary["processed"] += 1
        summary["invalid_rows"] += counts["invalid"]
        summary["empty_predictions"] += counts["empty_pred"]
        if n % flush_every == 0:
            _append(buf, csv_path, COLUMNS)
            _append(obj_buf, obj_path, OBJECT_COLUMNS)
            buf, obj_buf = [], []
    _append(buf, csv_path, COLUMNS)
    _append(obj_buf, obj_path, OBJECT_COLUMNS)

    final = pd.read_csv(csv_path, dtype={"image_id": str}) if csv_path.exists() else pd.DataFrame()
    summary["rows_in_csv"] = len(final)
    summary["expected_rows"] = len(image_ids) * len(grid)
    summary["invalid_rows_in_csv"] = (
        int((final.get("flag") == "invalid").sum()) if len(final) else 0
    )
    summary["complete"] = summary["rows_in_csv"] == summary["expected_rows"]
    (out_dir / f"{model_name}_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    return summary


def run_experiment(
    cfg: dict,
    paths,
    ds=None,
    models: list[str] | None = None,
    max_images: int | None = None,
    model_factory: Callable | None = None,
    device: str = "auto",
    progress: bool = True,
) -> list[dict]:
    from promptshift.data import get_dataset

    if ds is None:
        split_file = paths.splits / f"{cfg['dataset']}.json"
        ds = get_dataset(cfg["dataset"], paths.data, cfg["data"], split_file)
    image_ids = ds.ids(cfg["split"])
    limit = max_images if max_images is not None else cfg.get("max_images")
    if limit is not None:
        image_ids = image_ids[: int(limit)]
    out_dir = paths.raw_results / cfg["exp_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = git_sha(paths.repo)
    factory = model_factory or (lambda name: load_model(name, paths.weights, device))
    summaries = []
    for name in models or cfg["models"]:
        model = factory(name)
        summaries.append(run_model(name, model, ds, image_ids, cfg, out_dir, sha, progress))
        del model
        _free_gpu()
    print_summary(summaries)
    return summaries


def _free_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def print_summary(summaries: list[dict]) -> None:
    print("\n=== run summary ===")
    for s in summaries:
        status = "COMPLETE" if s["complete"] else "INCOMPLETE"
        print(
            f"{s['model']}: {status} rows {s['rows_in_csv']}/{s['expected_rows']} | "
            f"processed {s['processed']} (already done {s['already_done']}) | "
            f"invalid rows {s['invalid_rows_in_csv']} | "
            f"empty predictions {s['empty_predictions']} | "
            f"errors {s['errors']}" + (f" {s['error_ids'][:5]}" if s["errors"] else "")
        )
