"""Seen / unseen label of a dataset for MedSAM, read from docs/leakage.md (owner B, P1-13).

A missing file or a dataset without a row counts as "unknown" (spec 7).
"""

from __future__ import annotations

from pathlib import Path


def leakage_label(docs: Path, dataset_name: str) -> str:
    """'seen' / 'unseen' / 'unknown' from the row whose first cell mentions the dataset."""
    path = docs / "leakage.md"
    if not path.exists():
        return "unknown"
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip().lower() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and dataset_name.lower() in cells[0]:
            if cells[1].startswith("yes") or cells[1] == "seen":
                return "seen"
            if cells[1].startswith("no") or cells[1] == "unseen":
                return "unseen"
            return "unknown"
    return "unknown"
