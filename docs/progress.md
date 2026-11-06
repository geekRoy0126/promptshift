# Progress

## 2026-11-05 (A, with Claude Code)

- **Current step**: P1-1 repo skeleton.
- **Next step**: P1-2 metrics + tests.
- **Blockers**: TA confirmation of the allowed AI-use level (2026-11-06). No official Phase 1 spec yet (defaults apply).
- **Environment**: Miniconda installed per-user at `C:\Users\ruoyu\miniconda3` (2026-11-05, via winget). Env `promptshift` = Python 3.11.18 from conda-forge (`--override-channels -c conda-forge`, avoids the Anaconda default-channel ToS). `pip install -e ".[dev]"` done.
- **Done**: P1-1 (`ruff check .` clean, `pytest -q` 15 passed).
