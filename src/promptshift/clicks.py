"""Oracle click simulation using the existing SAM wrappers and image metrics.

Initial prompts are paired across models. Later corrections are adaptive and may
differ across models. Level 0 means one initial click; levels 1..5 add corrections.
Ground truth selects clicks for this controlled experiment, not for deployment.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from promptshift.metrics import compute_metrics
from promptshift.prompts.point import object_distance_transform, offset_click
from promptshift.seeding import stable_rng


def correction_click(pred, gt, used=()):
    """Deepest unused pixel in the largest FN/FP component (8-connectivity).

    Ties prefer false negatives, then row-major component/pixel order. A positive
    click corrects a false negative; a negative click corrects a false positive.
    Returns None when there is no unused error pixel.
    """
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("pred and gt must be equal-sized 2-D masks")
    available = np.ones_like(gt)
    for x, y in used:
        available[int(y), int(x)] = False
    best = None
    for label, errors in [(1, gt & ~pred), (0, pred & ~gt)]:
        components, n = ndimage.label(errors, np.ones((3, 3), bool))
        areas = np.bincount(components.ravel(), minlength=n + 1)
        for k in range(1, n + 1):
            component = components == k
            candidates = component & available
            if not candidates.any():
                continue
            if best is None or areas[k] > best[0]:
                depth = object_distance_transform(component)
                index = int(np.argmax(np.where(candidates, depth, -1)))
                y, x = np.unravel_index(index, gt.shape)
                best = (int(areas[k]), int(x), int(y), label)
    return None if best is None else dict(area=best[0], x=best[1], y=best[2], label=best[3])


def click_grid(cfg):
    c = cfg["clicks"]
    return [
        ("point_offset", float(lv), int(sd)) for lv in c["offset_levels"] for sd in c["seeds"]
    ] + [
        ("corrective_clicks", float(k), int(sd))
        for k in range(c["max_corrections"] + 1)
        for sd in c["seeds"]
    ]


def validate_click_config(cfg):
    c = cfg["clicks"]
    if not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"]):
        raise ValueError("click seeds must be nonempty and unique")
    if any(not isinstance(s, int) for s in c["seeds"]):
        raise ValueError("click seeds must be integers")
    levels = c["offset_levels"]
    if not levels or len(set(levels)) != len(levels) or any(not 0 <= x <= 1 for x in levels):
        raise ValueError("offset levels must be unique values in [0,1]")
    if not 0 <= c["initial_offset"] <= 1:
        raise ValueError("initial offset must be in [0,1]")
    if not isinstance(c["max_corrections"], int) or not 1 <= c["max_corrections"] <= 5:
        raise ValueError("max_corrections must be an integer from 1 to 5")
    if not cfg["models"] or any(m not in ("sam_vit_b", "sam_vit_h", "mock") for m in cfg["models"]):
        raise ValueError("Click experiments support SAM only; MedSAM is box-only")


def evaluate_click_image(model, sample, cfg, meta):
    """Encode once; produce image-level metrics and an auditable click trace."""
    if not model.supports_points:
        raise ValueError("model does not support point prompts")
    if not sample.objects:
        raise ValueError("click experiment needs at least one target object")
    c = cfg["clicks"]
    model.set_image(sample.image)
    rows, traces = [], []

    def initial(obj, obj_idx, perturb, level, seed):
        rng = stable_rng(
            cfg["global_seed"], sample.dataset, sample.image_id, obj_idx, perturb, level, seed
        )
        return offset_click(obj, level, rng, band=c["band"])

    def trace(point, obj_idx, perturb, level, seed, step, area=None):
        x, y, label = point
        traces.append(
            dict(
                **meta,
                image_id=sample.image_id,
                object_idx=obj_idx,
                perturb=perturb,
                level=level,
                seed=seed,
                step=step,
                x=float(x),
                y=float(y),
                label=int(label),
                error_region_area=area,
            )
        )

    def record(perturb, level, seed, preds, counts):
        union = np.logical_or.reduce(preds)
        metrics = compute_metrics(union, sample.mask, cfg["metrics"]["nsd_tolerance_px"])
        rows.append(
            dict(
                **meta,
                image_id=sample.image_id,
                perturb=perturb,
                level=float(level),
                seed=seed,
                flag="ok",
                **metrics,
                n_objects=len(sample.objects),
                clicks_total=sum(counts),
                corrections_used_max=max(counts) - 1,
            )
        )

    for level in c["offset_levels"]:
        for seed in c["seeds"]:
            predictions = []
            for k, obj in enumerate(sample.objects):
                prompt = initial(obj, k, "point_offset", level, seed)
                pred, _ = model.predict_points(prompt.coords, prompt.labels)
                predictions.append(pred)
                trace((*prompt.coords[0], 1), k, "point_offset", level, seed, 0)
            record("point_offset", level, seed, predictions, [1] * len(predictions))

    for seed in c["seeds"]:
        states = []
        for k, obj in enumerate(sample.objects):
            prompt = initial(obj, k, "point_offset", c["initial_offset"], seed)
            coords, labels = prompt.coords.copy(), prompt.labels.copy()
            pred, logits = model.predict_points(coords, labels)
            states.append(dict(coords=coords, labels=labels, pred=pred, logits=logits))
            trace((*coords[0], 1), k, "corrective_clicks", 0, seed, 0)
        record("corrective_clicks", 0, seed, [s["pred"] for s in states], [1] * len(states))
        for step in range(1, c["max_corrections"] + 1):
            for k, (obj, state) in enumerate(zip(sample.objects, states, strict=True)):
                correction = correction_click(state["pred"], obj, state["coords"])
                if correction is None:
                    continue  # keep the last prediction; never fabricate another click
                state["coords"] = np.vstack([state["coords"], [correction["x"], correction["y"]]])
                state["labels"] = np.append(state["labels"], correction["label"])
                state["pred"], state["logits"] = model.predict_points(
                    state["coords"], state["labels"], mask_input=state["logits"]
                )
                trace(
                    (correction["x"], correction["y"], correction["label"]),
                    k,
                    "corrective_clicks",
                    step,
                    seed,
                    step,
                    correction["area"],
                )
            record(
                "corrective_clicks",
                step,
                seed,
                [s["pred"] for s in states],
                [len(s["coords"]) for s in states],
            )
    return rows, traces


def validate_click_trace(metrics, traces):
    """Reject missing/duplicated clicks and disagreement with actual click counts."""
    if len(traces) == 0:
        raise ValueError("Missing click trace")
    key = ["image_id", "object_idx", "perturb", "level", "seed", "step"]
    if traces.duplicated(key).any():
        raise ValueError("Duplicate click trace")
    if not np.isfinite(traces[["x", "y", "step"]].to_numpy()).all():
        raise ValueError("Non-finite click trace")
    if not traces.label.isin([0, 1]).all() or not traces.step.ge(0).all():
        raise ValueError("Invalid click label/step")
    for col in ["image_id", "model", "dataset", "split", "exp_id"]:
        if col in metrics and col in traces and set(traces[col]) != set(metrics[col]):
            raise ValueError(f"Mixed trace {col}")
    for row in metrics.itertuples():
        t = traces[
            (traces.image_id == row.image_id)
            & (traces.perturb == row.perturb)
            & (traces.seed == row.seed)
        ]
        if row.perturb == "point_offset":
            t = t[t.level == row.level]
            if not t.step.eq(0).all() or not t.label.eq(1).all():
                raise ValueError("Invalid offset trace")
        else:
            if not t.level.eq(t.step).all():
                raise ValueError("Invalid correction round")
            t = t[t.step <= row.level]
        initial = t[t.step == 0]
        if (
            len(t) != row.clicks_total
            or len(initial) != row.n_objects
            or set(initial.object_idx) != set(range(row.n_objects))
        ):
            raise ValueError("Incomplete click trace/count mismatch")
