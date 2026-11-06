# AI usage log

Course policy: AI is allowed only as a learning aid and must be disclosed. Every session that uses AI adds an entry here. Commits written with AI carry a `Co-Authored-By: Claude` trailer.

| Date | Member | Tool | What was done | Files generated / modified by AI | Reviewed by |
|---|---|---|---|---|---|
| 2026-11-05 | A (Ruoyu Wang) | Claude Code (Claude Opus 5.5) | Read the spec and listed conflicts / open questions; the member answered them. Set up P1-1: repo skeleton, path resolution, stable seeding, default config, tests. Applied the answers to `CLAUDE.md` and logged them in `docs/decisions.md`. | `CLAUDE.md` (edited copy), `.gitattributes`, `pyproject.toml`, `requirements.txt`, `.gitignore`, `README.md`, `configs/default.yaml`, `src/promptshift/{__init__,paths,seeding}.py`, `tests/test_{paths,seeding,config}.py`, `docs/{decisions,progress}.md`, `CONTRIBUTIONS.md`, `AI_USAGE_LOG.md` | Pending review by Ruoyu Wang |
| 2026-11-05 | A (Ruoyu Wang) | Claude Code (Claude Opus 5.5) | P1-2: Dice / IoU / NSD / HD95 with empty-mask conventions, brute-force and MONAI cross-check tests. Installed CPU torch + MONAI in the local env. | `src/promptshift/metrics.py`, `tests/test_metrics.py`, `docs/{decisions,progress}.md`, `CONTRIBUTIONS.md`, `AI_USAGE_LOG.md` | Pending review by Ruoyu Wang |

Note: the allowed level of AI use is to be confirmed with the TA on 2026-11-06.
