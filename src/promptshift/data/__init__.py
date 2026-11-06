"""Dataset loaders. Adding a dataset = one loader module + one entry in LOADERS + one config."""

from __future__ import annotations

from pathlib import Path

from promptshift.data.base import Sample, SegDataset, split_objects
from promptshift.data.kvasir import KvasirSeg


class _Lazy:
    """Imports a loader only when used, so optional deps are not needed for other datasets."""

    def __init__(self, module: str, cls: str) -> None:
        self.module, self.cls = module, cls

    def from_config(self, *args, **kwargs):
        import importlib

        return getattr(importlib.import_module(self.module), self.cls).from_config(*args, **kwargs)


LOADERS = {
    "kvasir_seg": KvasirSeg,
    "coco_val2017": _Lazy("promptshift.data.coco", "CocoVal2017"),
    "isic2018": _Lazy("promptshift.data.isic", "Isic2018"),
    "busi": _Lazy("promptshift.data.busi", "Busi"),
}


def get_dataset(name: str, data_dir: Path, data_cfg: dict, split_file: Path | None = None):
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(LOADERS)}")
    return LOADERS[name].from_config(Path(data_dir), data_cfg, split_file)


__all__ = ["LOADERS", "KvasirSeg", "Sample", "SegDataset", "get_dataset", "split_objects"]
