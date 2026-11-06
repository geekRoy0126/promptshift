"""Prompt-free baselines (U-Net, classical). They write the runner's CSV format with
perturb = "none" and level = NaN (CLAUDE.md 6.5), so analysis code treats all methods alike."""

from promptshift.baselines.common import baseline_row, write_rows

__all__ = ["baseline_row", "write_rows"]
