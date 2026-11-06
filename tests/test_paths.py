from pathlib import Path

import pytest

from promptshift import paths


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv(paths.ROOT_ENV_VAR, raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.setattr(paths, "is_colab", lambda: False)


def test_local_default_is_repo(no_env):
    p = paths.get_paths()
    assert p.root == paths.REPO_ROOT
    assert p.data == paths.REPO_ROOT / "data"
    assert p.weights == paths.REPO_ROOT / "weights"
    assert p.raw_results == paths.REPO_ROOT / "results" / "raw"


def test_repo_root_contains_pyproject():
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()


def test_explicit_root_wins(no_env, monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(tmp_path / "env"))
    assert paths.get_paths(tmp_path / "arg").root == tmp_path / "arg"


def test_env_root(no_env, monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(tmp_path))
    p = paths.get_paths()
    assert p.root == tmp_path
    # repo-tracked folders stay in the repo
    assert p.splits == paths.REPO_ROOT / "splits"


def test_ensure_creates_storage_dirs(tmp_path):
    p = paths.get_paths(tmp_path).ensure()
    for d in (p.data, p.weights, p.raw_results, p.summary_results):
        assert d.is_dir()


def test_colab_without_drive_raises(no_env, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "is_colab", lambda: True)
    monkeypatch.setattr(paths, "COLAB_DRIVE_MOUNT", tmp_path / "not_mounted")
    with pytest.raises(RuntimeError, match="Drive is not mounted"):
        paths.get_paths()


def test_colab_with_drive(no_env, monkeypatch, tmp_path):
    mount = tmp_path / "MyDrive"
    mount.mkdir()
    monkeypatch.setattr(paths, "is_colab", lambda: True)
    monkeypatch.setattr(paths, "COLAB_DRIVE_MOUNT", mount)
    monkeypatch.setattr(paths, "COLAB_ROOT", mount / "promptshift")
    assert paths.get_paths().root == Path(mount / "promptshift")
