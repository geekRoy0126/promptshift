"""Role C integration checks use synthetic images, never reported as experiment results."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from promptshift.analysis.failures import rank_failures
from promptshift.analysis.stats import delta_dice
from promptshift.baselines.common import baseline_row, write_rows
from promptshift.config import load_config
from promptshift.data import get_dataset
from promptshift.demo import Comparison, build_app, pixel_box
from promptshift.models.mock import MockSegmenter
from promptshift.paths import REPO_ROOT, Paths
from promptshift.runner import run_model


def script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def experiment(tmp_path):
    paths = Paths(root=tmp_path / "storage", repo=tmp_path / "repo").ensure()
    paths.splits.mkdir(parents=True)
    paths.docs.mkdir()
    folder = paths.data / "busi" / "Dataset_BUSI_with_GT"
    for cls in ("benign", "malignant"):
        (folder / cls).mkdir(parents=True)
    ids = []
    for n in range(1, 4):
        rgb = np.zeros((40, 48, 3), dtype=np.uint8)
        rgb[..., 0] = 30 * n  # distinct files: no deduplication
        mask = np.zeros((40, 48), dtype=np.uint8)
        mask[10:30, 12:36] = 255
        Image.fromarray(rgb).save(folder / "benign" / f"benign ({n}).png")
        Image.fromarray(mask).save(folder / "benign" / f"benign ({n})_mask.png")
        ids.append(f"benign_{n:04d}")
    (paths.splits / "busi.json").write_text(json.dumps(dict(train=[], val=[], test=ids)))
    cfg = load_config(REPO_ROOT / "configs/c_busi_box.yaml")
    cfg["models"] = ["sam_vit_b"]
    cfg["analysis"]["bootstrap_n"] = 30
    ds = get_dataset("busi", paths.data, cfg["data"], paths.splits / "busi.json")
    out = paths.raw_results / cfg["exp_id"]
    out.mkdir()
    summary = run_model("sam_vit_b", MockSegmenter(), ds, ids, cfg, out, "test", False)
    assert summary["complete"]
    return cfg, paths, ds, out


def test_busi_runner_analysis_and_incomplete_status(experiment):
    cfg, paths, ds, source = experiment
    mod = script("analyze_results")
    out = paths.summary_results / cfg["exp_id"]
    status = mod.analyze(cfg, paths, out)
    assert status["complete"]
    assert status["missing_baselines"] == ["unet", "classical"]
    rows = [
        baseline_row(
            exp_id=cfg["exp_id"],
            split="test",
            method="unet",
            sample=(sample := ds.get(i)),
            pred=sample.mask,
            ms=0,
            sha="test",
            nsd_tolerance_px=2,
        )
        for i in ds.ids("test")
    ]
    write_rows(source / "unet.csv", rows)
    assert mod.analyze(cfg, paths, out)["complete"]
    assert len(pd.read_csv(out / "paired_tests.csv")) > 0
    assert len(pd.read_csv(out / "crossover.csv")) == 1
    assert (out / "box_jitter.png").is_file()
    assert (out / "box_scale.pdf").is_file()
    assert len(pd.read_csv(out / "failures.csv")) > 0
    # A partial image/configuration grid must not be presented as complete.
    p = source / "sam_vit_b.csv"
    df = pd.read_csv(p)
    df.iloc[:-1].to_csv(p, index=False)
    assert not mod.analyze(cfg, paths, out)["complete"]
    # Entirely invalid configurations remain visible in integrity counts, without a plot crash.
    invalid = df.copy()
    invalid["flag"] = "invalid"
    invalid[["dice", "iou", "nsd", "hd95"]] = np.nan
    invalid.to_csv(p, index=False)
    mod.analyze(cfg, paths, out)
    # A complete grid can contain invalid rows: they must be counted, not invented.
    integrity = pd.read_csv(out / "integrity.csv")
    assert integrity.loc[integrity.model.eq("sam_vit_b"), "invalid_rows"].iloc[0] == len(df)
    # Mixing experiments or duplicating rows is a hard error.
    pd.concat([df, df.iloc[:1]]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        mod.analyze(cfg, paths, out)


def test_failure_ranking_and_visual_export(experiment):
    cfg, paths, ds, source = experiment
    df = pd.read_csv(source / "sam_vit_b.csv")
    ranked = rank_failures(df)
    assert ranked.dice_drop.is_monotonic_decreasing
    ranked["model"] = "mock"
    ranking = paths.root / "ranking.csv"
    ranked.to_csv(ranking, index=False)
    out = paths.root / "failures"
    script("make_failure_cases").render_cases(cfg, paths, ranking, out, top=2)
    review = pd.read_csv(out / "review.csv")
    assert len(review) == 2
    assert review.reviewer_reason.isna().all()
    assert all((out / name).is_file() for name in review.figure)


def test_delta_endpoints_are_paired():
    rows = [
        dict(image_id=i, perturb="box_jitter", level=lv, flag="ok", dice=d)
        for i, lv, d in [("a", 0.0, 0.9), ("a", 0.2, 0.6), ("b", 0.0, 0.1)]
    ]
    d = delta_dice(pd.DataFrame(rows), n=20)
    assert d["d0"] == pytest.approx(0.9)
    assert d["d1"] == pytest.approx(0.6)
    assert d["delta"] == pytest.approx(d["d0"] - d["d1"])


def test_demo_shared_box_mock_and_app(tmp_path):
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    assert np.array_equal(pixel_box(rgb, 10, 20, 90, 80), [8, 8, 72, 32])
    for box in [(90, 20, 10, 80), (0, 0, np.nan, 80), (-1, 0, 90, 80)]:
        with pytest.raises(ValueError):
            pixel_box(rgb, *box)
    comparison = Comparison(tmp_path, mock=True)
    left, right, status = comparison.predict(rgb, 10, 20, 90, 80)
    assert left.shape == rgb.shape and np.array_equal(left, right)
    assert "synthetic" in status
    app = build_app(tmp_path, mock=True)
    assert app is not None
