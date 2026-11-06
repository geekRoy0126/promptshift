import importlib.util

import pandas as pd

from promptshift.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "make_report_tables", REPO_ROOT / "scripts" / "make_report_tables.py"
)
mrt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mrt)


def test_leakage_label(tmp_path):
    assert mrt.leakage_label(tmp_path, "kvasir") == "unknown"  # no file
    (tmp_path / "leakage.md").write_text(
        "| Dataset | In MedSAM training data? | Evidence |\n|---|---|---|\n"
        "| Kvasir-SEG | yes | paper Table S1 |\n| BUSI | no | ... |\n| ISIC 2018 | unknown | |\n"
    )
    assert mrt.leakage_label(tmp_path, "kvasir") == "seen"
    assert mrt.leakage_label(tmp_path, "busi") == "unseen"
    assert mrt.leakage_label(tmp_path, "isic") == "unknown"
    assert mrt.leakage_label(tmp_path, "coco") == "unknown"


def test_render_replaces_includes_and_marks_missing(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "a.md").write_text("| x |\n|---|\n| 1 |\n")
    tpl = tmp_path / "t.md"
    tpl.write_text("A\n<!-- include: tables/a.md -->\nB\n<!-- include: tables/missing.md -->\n")
    out = tmp_path / "out.md"
    mrt.render(tpl, tables, out)
    text = out.read_text()
    assert "| 1 |" in text and "TBD" in text and "include:" not in text


def test_seed_average_before_image_mean():
    df = pd.DataFrame(
        {
            "perturb": "box_jitter", "flag": "ok", "level": 0.1,
            "image_id": ["a", "a", "a", "b"], "seed": [0, 1, 2, 0],
            "dice": [0.0, 0.0, 0.0, 1.0],
        }
    )  # fmt: skip
    pi = mrt.per_image(df, "box_jitter")
    assert pi.groupby("level")["dice"].mean().iloc[0] == 0.5  # not 0.25
