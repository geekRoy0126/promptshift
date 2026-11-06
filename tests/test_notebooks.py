import ast
import importlib.util
import json
import re

import pytest

from promptshift.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "build_notebooks", REPO_ROOT / "scripts" / "build_notebooks.py"
)
bn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bn)

NAMES = list(bn.NOTEBOOKS)


def load(name):
    return json.loads((bn.NB_DIR / f"{name}.ipynb").read_text(encoding="utf-8"))


def test_expected_notebooks():
    assert NAMES == [
        "P1_00_setup_colab", "P1_01_data_check", "P1_02_medsam_parity", "P1_03_kvasir_box_pilot",
        "P1_04_unet_kvasir",
    ]  # fmt: skip


@pytest.mark.parametrize("name", NAMES)
def test_in_sync_with_builder(name):
    text = (bn.NB_DIR / f"{name}.ipynb").read_text(encoding="utf-8")
    assert text == bn.render(name), "run python scripts/build_notebooks.py"


@pytest.mark.parametrize("name", NAMES)
def test_cells_parse_and_have_no_outputs(name):
    nb = load(name)
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))


@pytest.mark.parametrize("name", NAMES)
def test_params_then_shared_setup(name):
    code = [c for c in load(name)["cells"] if c["cell_type"] == "code"]
    assert code[0]["metadata"].get("tags") == ["parameters"]
    assert "".join(code[1]["source"]) == bn.SETUP.rstrip("\n")


def test_setup_never_exposes_token():
    s = bn.SETUP
    assert 'userdata.get("GH_TOKEN")' in s
    assert "print(_token" not in s and "x-access-token:{_token}@" not in s  # not in a URL
    assert "del _token, _auth" in s
    assert bn.REPO_URL in s and "@github.com" not in s


@pytest.mark.parametrize("name", NAMES)
def test_no_secrets_committed(name):
    text = (bn.NB_DIR / f"{name}.ipynb").read_text(encoding="utf-8")
    assert not re.search(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}", text)
    assert "kaggle.json" not in text
