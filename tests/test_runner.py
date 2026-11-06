import json

import numpy as np
import pandas as pd
import pytest

from promptshift.config import load_config
from promptshift.data.base import Sample
from promptshift.models import MockSegmenter
from promptshift.paths import REPO_ROOT, Paths
from promptshift.runner import COLUMNS, OBJECT_COLUMNS, build_grid, run_experiment

CFG = load_config(REPO_ROOT / "configs" / "p1_kvasir_pilot.yaml")
N_CFG = 24
VOLATILE = ["encode_ms", "decode_ms", "timestamp", "git_sha"]


class ToyDataset:
    """6 images: single object, two objects, and one 1-px-wide object (always invalid)."""

    name = "toy"

    def __init__(self):
        self._ids = [f"im{i}" for i in range(6)]

    def ids(self, split):
        return list(self._ids)

    def get(self, image_id):
        i = int(image_id[2:])
        h, w = 64 + 4 * i, 80
        objs = []
        a = np.zeros((h, w), bool)
        a[10:30, 10 + i : 40 + i] = True
        objs.append(a)
        if i % 2 == 1:
            b = np.zeros((h, w), bool)
            b[40:55, 50:70] = True
            objs.append(b)
        if i == 4:
            c = np.zeros((h, w), bool)
            c[40:60, 60] = True  # width 0 -> invalid box at every level
            objs.append(c)
        img = np.zeros((h, w, 3), np.uint8)
        return Sample(image_id, img, np.logical_or.reduce(objs), objs, "toy", {})


class Killed(MockSegmenter):
    """Simulates the process being killed (KeyboardInterrupt is not caught by the runner)."""

    def __init__(self, kill_at):
        super().__init__()
        self.kill_at, self.n = kill_at, 0

    def set_image(self, image):
        self.n += 1
        if self.n == self.kill_at:
            raise KeyboardInterrupt
        super().set_image(image)


def cfg(**kw):
    c = {**CFG, "exp_id": "toy_exp", "split": "test", "dataset": "toy", **kw}
    c["runner"] = {**CFG["runner"], "flush_every": 2}
    return c


def paths_for(tmp_path):
    return Paths(root=tmp_path, repo=REPO_ROOT)


def run(tmp_path, factory=lambda n: MockSegmenter(), models=("mock",), **kw):
    return run_experiment(
        cfg(**kw), paths_for(tmp_path), ds=ToyDataset(), models=list(models),
        model_factory=factory, progress=False,
    )  # fmt: skip


def read(tmp_path, name="mock"):
    d = tmp_path / "results" / "raw" / "toy_exp"
    return pd.read_csv(d / f"{name}.csv"), pd.read_csv(d / f"{name}_objects.csv")


def test_grid_has_24_configurations():
    assert len(build_grid(CFG)) == N_CFG


def test_full_run_rows_columns_and_invalid_flag(tmp_path):
    (s,) = run(tmp_path)
    df, odf = read(tmp_path)
    assert list(df.columns) == COLUMNS and list(odf.columns) == OBJECT_COLUMNS
    assert len(df) == 6 * N_CFG and s["complete"] and s["errors"] == 0
    assert not df.duplicated(["image_id", "perturb", "level", "seed"]).any()
    bad = df[df["image_id"] == "im4"]
    assert (bad["flag"] == "invalid").all() and bad["dice"].isna().all()
    assert (df[df["image_id"] != "im4"]["flag"] == "ok").all()
    assert s["invalid_rows_in_csv"] == N_CFG
    n_objects = sum(len(ToyDataset().get(i).objects) for i in ToyDataset().ids("test"))
    assert len(odf) == n_objects * N_CFG
    assert (odf[odf["image_id"] == "im4"]["flag"] == "invalid").all()
    assert df["dice"].between(0, 1).all() or df["dice"].isna().any()
    summary = json.loads((tmp_path / "results/raw/toy_exp/mock_summary.json").read_text())
    assert summary["rows_in_csv"] == 6 * N_CFG


def test_identity_configs_give_identical_predictions(tmp_path):
    run(tmp_path)
    df, _ = read(tmp_path)
    ok = df[df["flag"] == "ok"]
    ident = ok[((ok["perturb"] == "box_jitter") & (ok["level"] == 0)) | (
        (ok["perturb"] == "box_scale") & (ok["level"] == 1.0)
    )]  # fmt: skip
    assert (ident.groupby("image_id")["dice"].nunique() == 1).all()
    assert (ident["box_iou_mean"] == 1.0).all()


def test_resume_after_kill_no_duplicates_no_missing(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, factory=lambda n: Killed(kill_at=4))
    df, _ = read(tmp_path)
    assert 0 < len(df) < 6 * N_CFG  # 2 images flushed before the kill
    # simulate a write cut off mid-line
    csv = tmp_path / "results/raw/toy_exp/mock.csv"
    with csv.open("a") as f:
        f.write("toy_exp,toy,test,mock,im3,1,box_jitter,0.1")
    (s,) = run(tmp_path)
    assert s["already_done"] == 2 and s["processed"] == 4 and s["complete"]
    df, odf = read(tmp_path)
    assert len(df) == 6 * N_CFG
    assert not df.duplicated(["image_id", "perturb", "level", "seed"]).any()
    assert not odf.duplicated(["image_id", "object_idx", "perturb", "level", "seed"]).any()

    clean = tmp_path / "clean"
    run(clean)
    ref, _ = read(clean)
    key = ["image_id", "perturb", "level", "seed"]
    a = df.drop(columns=VOLATILE).sort_values(key).reset_index(drop=True)
    b = ref.drop(columns=VOLATILE).sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_rerun_of_complete_experiment_is_noop(tmp_path):
    run(tmp_path)
    (s,) = run(tmp_path)
    assert s["processed"] == 0 and s["already_done"] == 6 and s["complete"]


def test_prompts_are_model_independent(tmp_path):
    run(
        tmp_path,
        factory=lambda n: MockSegmenter(radius_frac=0.1 if n == "mock" else 0.3),
        models=("mock", "mock2"),
    )
    a, _ = read(tmp_path, "mock")
    b, _ = read(tmp_path, "mock2")
    key = ["image_id", "perturb", "level", "seed"]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    assert len(m) == len(a)
    pd.testing.assert_series_equal(m["box_iou_mean_a"], m["box_iou_mean_b"], check_names=False)


def test_errors_are_counted_not_silent(tmp_path):
    class Broken(MockSegmenter):
        def predict_box(self, box):
            raise RuntimeError("boom")

    (s,) = run(tmp_path, factory=lambda n: Broken())
    assert s["errors"] == 5 and not s["complete"]  # im4 is invalid -> never decoded
