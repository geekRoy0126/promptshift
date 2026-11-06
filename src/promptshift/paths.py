"""The only module that knows where data, weights and results live.

- Colab: everything under /content/drive/MyDrive/promptshift/ (survives disconnects).
- Local: data/, weights/, results/ inside the repo.
- Override: set PROMPTSHIFT_ROOT (used by tests and for custom locations).

Repo-tracked outputs (splits/, figures/, docs/, reports/) always live in the repo.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COLAB_DRIVE_MOUNT = Path("/content/drive/MyDrive")
COLAB_ROOT = COLAB_DRIVE_MOUNT / "promptshift"
ROOT_ENV_VAR = "PROMPTSHIFT_ROOT"


def is_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    if "COLAB_RELEASE_TAG" in os.environ:
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        # find_spec raises if the parent package "google" is missing or broken.
        return False


@dataclass(frozen=True)
class Paths:
    """Storage root for large artifacts plus repo-tracked output folders."""

    root: Path
    repo: Path = REPO_ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def weights(self) -> Path:
        return self.root / "weights"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def raw_results(self) -> Path:
        return self.results / "raw"

    @property
    def summary_results(self) -> Path:
        return self.results / "summary"

    @property
    def splits(self) -> Path:
        return self.repo / "splits"

    @property
    def figures(self) -> Path:
        return self.repo / "figures"

    @property
    def docs(self) -> Path:
        return self.repo / "docs"

    @property
    def configs(self) -> Path:
        return self.repo / "configs"

    def ensure(self) -> Paths:
        """Create the storage folders (not the repo folders) if missing."""
        for p in (self.data, self.weights, self.raw_results, self.summary_results):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(root: str | os.PathLike[str] | None = None) -> Paths:
    """Resolve the storage root: explicit arg > PROMPTSHIFT_ROOT > Colab Drive > repo."""
    if root is not None:
        return Paths(root=Path(root))
    env_root = os.environ.get(ROOT_ENV_VAR)
    if env_root:
        return Paths(root=Path(env_root))
    if is_colab():
        if not COLAB_DRIVE_MOUNT.is_dir():
            raise RuntimeError(
                "Running on Colab but Google Drive is not mounted. Run "
                "`from google.colab import drive; drive.mount('/content/drive')` first."
            )
        return Paths(root=COLAB_ROOT)
    return Paths(root=REPO_ROOT)
