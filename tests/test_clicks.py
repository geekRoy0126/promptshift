"""Protocol and crash recovery checks; synthetic masks are never reported as results."""

import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from promptshift.clicks import correction_click, evaluate_click_image, validate_click_config
from promptshift.config import load_config
from promptshift.paths import Paths, REPO_ROOT


def script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cfg():
    c = load_config(REPO_ROOT / "configs/c_busi_clicks.yaml")
    c["models"] = ["sam_vit_b"]
    c["analysis"]["bootstrap_n"] = 20
    return c


def test_error_click_polarity_depth_and_no_repeat():
    gt = np.zeros((12, 14), bool)
    gt[2:5, 2:5] = True
    pred = np.zeros_like(gt)
    pred[7:10, 8:11] = True
    # Equal-area FN and FP regions: positive FN wins, deepest pixel is the center.
    assert correction_click(pred, gt) == dict(area=9, x=3, y=3, label=1)
    other = correction_click(pred, gt, [(3, 3)])
    assert (other["x"], other["y"]) != (3, 3)
    pred[6:11, 7:12] = True
    assert correction_click(pred, gt)["label"] == 0
    assert correction_click(gt, gt) is None


class RecordingModel:
    supports_points = True

    def __init__(self, gt):
        self.gt = gt
        self.calls = []

    def set_image(self, image):
        pass

    def predict_points(self, coords, labels, mask_input=None):
        self.calls.append((coords.copy(), labels.copy(), mask_input))
        # Initial prediction empty; one correction recovers the object exactly.
        pred = self.gt.copy() if mask_input is not None else np.zeros_like(self.gt)
        return pred, np.full((4, 4), len(coords), np.float32)


def sample():
    gt = np.zeros((16, 16), bool)
    gt[3:13, 3:13] = True
    return SimpleNamespace(
        image_id="one", dataset="busi", image=np.zeros((16, 16, 3), np.uint8), mask=gt, objects=[gt]
    )


def test_cumulative_history_stop_and_paired_initial(cfg):
    s = sample()
    m = RecordingModel(s.mask)
    rows, points = evaluate_click_image(m, s, cfg, {})
    assert len(rows) == 33
    df, trace = pd.DataFrame(rows), pd.DataFrame(points)
    start = trace[(trace.perturb == "corrective_clicks") & (trace.step == 0)].sort_values("seed")
    offset = trace[(trace.perturb == "point_offset") & (trace.level == 0.5)].sort_values("seed")
    np.testing.assert_array_equal(start[["x", "y", "label"]], offset[["x", "y", "label"]])
    # Stops after perfect correction; later levels carry actual click count 2.
    tail = df[(df.perturb == "corrective_clicks") & (df.level > 0)]
    assert tail.dice.eq(1).all() and tail.clicks_total.eq(2).all()
    refined = [c for c in m.calls if c[2] is not None]
    assert len(refined) == 3
    assert all(len(c[0]) == 2 and np.all(c[2] == 1) for c in refined)
    other = RecordingModel(s.mask)
    _, points2 = evaluate_click_image(other, s, cfg, {})
    assert points == points2


def test_medsam_point_results_rejected(cfg):
    cfg["models"] = ["medsam"]
    with pytest.raises(ValueError, match="box-only"):
        validate_click_config(cfg)


def test_resume_repairs_incomplete_block_and_rejects_changed_protocol(tmp_path, cfg):
    paths = Paths(tmp_path / "storage", tmp_path / "repo").ensure()
    paths.splits.mkdir(parents=True)
    (paths.splits / "busi.json").write_text(json.dumps({"test": ["one"]}))
    s = sample()
    ds = SimpleNamespace(ids=lambda split: ["one"], get=lambda image_id: s)
    runner = script("run_click_experiment")
    factory = lambda name: RecordingModel(s.mask)
    assert runner.run(cfg, paths, ds, factory)[0]["complete"]
    assert (
        runner.run(cfg, paths, ds, lambda n: pytest.fail("completed model reloaded"))[0][
            "already_done"
        ]
        == 1
    )
    file = paths.raw_results / cfg["exp_id"] / "sam_vit_b.csv"
    pd.read_csv(file).iloc[:-1].to_csv(file, index=False)
    assert runner.run(cfg, paths, ds, factory)[0]["complete"]
    df = pd.read_csv(file)
    assert len(df) == 33 and not df.duplicated(["image_id", "perturb", "level", "seed"]).any()
    analyzer = script("analyze_click_results")
    assert analyzer.analyze(cfg, paths)["complete"]
    df.iloc[:-1].to_csv(file, index=False)
    with pytest.raises(ValueError, match="incomplete"):
        analyzer.analyze(cfg, paths)
    cfg["clicks"]["initial_offset"] = 0.25
    with pytest.raises(ValueError, match="Protocol/cohort changed"):
        runner.run(cfg, paths, ds, factory)
