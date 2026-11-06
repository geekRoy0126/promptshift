"""Generate every number in the Phase 1 report from CSVs (CLAUDE.md 6.10).

Writes reports/tables/*.md, renders reports/phase1_report.md from
reports/phase1_report.template.md (each `<!-- include: tables/X.md -->` line is replaced
by that table), and refreshes the results block in README.md.
Anything whose source CSV is missing is written as TBD.

Usage: python scripts/make_report_tables.py [--exp-id p1_kvasir_pilot] [--root PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from promptshift.config import load_config
from promptshift.paths import get_paths
from promptshift.runner import build_grid

TBD = "TBD"
MODEL_ORDER = ["sam_vit_b", "sam_vit_h", "medsam"]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "NaN" if pd.isna(v) else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def tbd(why: str) -> str:
    return f"{TBD} — {why}"


# --- leakage label (docs/leakage.md, owner B) ----------------------------------


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


# --- tables ----------------------------------------------------------------------


def table_data(paths) -> str:
    rows = []
    for name, label in [("kvasir_seg", "Kvasir-SEG"), ("coco_val2017", "COCO val2017")]:
        f = paths.splits / f"{name}.json"
        if not f.exists():
            rows.append(
                {"dataset": label, "train": TBD, "val": TBD, "test": TBD, "objects (test)": TBD}
            )
            continue
        sp = json.loads(f.read_text(encoding="utf-8"))
        stats = paths.summary_results / f"data_stats_{name}.csv"
        n_obj = TBD
        if stats.exists() and name == "kvasir_seg":
            st = pd.read_csv(stats, dtype={"image_id": str})
            n_obj = int(st.loc[st["split"] == "test", "n_objects"].sum())
        elif name == "coco_val2017":
            n_obj = len(sp["test"])
        rows.append({
            "dataset": label, "train": len(sp["train"]), "val": len(sp["val"]),
            "test": len(sp["test"]), "objects (test)": n_obj,
        })  # fmt: skip
    return md_table(pd.DataFrame(rows))


def table_parity(paths) -> str:
    out = []
    f = paths.results / "parity" / "medsam_parity.csv"
    if f.exists():
        df = pd.read_csv(f)
        out.append("**MedSAM wrapper vs. official `MedSAM_Inference.py`** (GT boxes)\n")
        out.append(
            md_table(df[["image_id", "dice_ours_vs_official", "pixels_differ", "pass"]], "{:.6f}")
        )
    else:
        out.append(tbd("run scripts/model_parity.py (notebook P1_02)"))
    f = paths.results / "parity" / "sam_parity.csv"
    out.append("\n**SAM wrapper vs. direct `SamPredictor`**\n")
    if f.exists():
        df = pd.read_csv(f)
        agg = df.groupby("model").agg(
            images=("image_id", "count"),
            identical=("identical", "all"),
            pixels_differ=("pixels_differ", "sum"),
        )
        agg = agg.reset_index()
        out.append(md_table(agg))
    else:
        out.append(tbd("run scripts/model_parity.py"))
    env = paths.results / "parity" / "environment.json"
    if env.exists():
        e = json.loads(env.read_text(encoding="utf-8"))
        out.append(f"\nRun on: {', '.join(f'{k} = {v}' for k, v in e.items())}")
    return "\n".join(out)


def load_raw(paths, exp_id: str) -> dict[str, pd.DataFrame]:
    d = paths.raw_results / exp_id
    return {
        m: pd.read_csv(d / f"{m}.csv", dtype={"image_id": str})
        for m in MODEL_ORDER
        if (d / f"{m}.csv").exists()
    }


def table_integrity(raw, cfg, n_images) -> str:
    if not raw:
        return tbd("pilot not run yet (notebook P1_03)")
    n_cfg = len(build_grid(cfg))
    rows = []
    for m, df in raw.items():
        rows.append({
            "model": m, "images": df["image_id"].nunique(), "rows": len(df),
            "expected rows": n_images * n_cfg if n_images else TBD,
            "invalid rows": int((df["flag"] == "invalid").sum()),
            "empty predictions": int(((df["flag"] == "ok") & (df["pred_area_frac"] == 0)).sum()),
            "git_sha": ",".join(sorted(df["git_sha"].astype(str).unique())),
        })  # fmt: skip
    return md_table(pd.DataFrame(rows))


def per_image(df: pd.DataFrame, perturb: str) -> pd.DataFrame:
    """Statistical unit = image: average seeds within each image first."""
    ok = df[(df["perturb"] == perturb) & (df["flag"] == "ok")]
    return ok.groupby(["level", "image_id"])["dice"].mean().reset_index()


def table_dice(raw, perturb: str, label: str) -> str:
    if not raw:
        return tbd("pilot not run yet")
    cols = {}
    for m, df in raw.items():
        pi = per_image(df, perturb)
        cols[m] = pi.groupby("level")["dice"].mean()
    t = pd.DataFrame(cols).reset_index().rename(columns={"level": label})
    note = "Mean Dice over images (seeds averaged within each image first); CIs: P1-12 (owner C)."
    return md_table(t) + "\n\n" + note


def table_delta(raw) -> str:
    if not raw:
        return tbd("pilot not run yet")
    rows = []
    for m, df in raw.items():
        pi = per_image(df, "box_jitter").groupby("level")["dice"].mean()
        d0, d20 = pi.get(0.0), pi.get(0.2)
        rows.append({"model": m, "Dice @ 0%": d0, "Dice @ 20%": d20, "ΔDice (0 → 20%)": d0 - d20})
    return md_table(pd.DataFrame(rows))


def results_block(raw) -> str:
    if not raw:
        return TBD
    return "**Kvasir-SEG pilot, box jitter (Phase 1, preliminary)**\n\n" + table_delta(raw)


# --- rendering ------------------------------------------------------------------


def render(template: Path, tables_dir: Path, out: Path) -> None:
    text = template.read_text(encoding="utf-8")

    def sub(m):
        f = tables_dir / m.group(1)
        return f.read_text(encoding="utf-8").strip() if f.exists() else tbd(f"missing {m.group(1)}")

    text = re.sub(r"<!-- include: tables/([\w.\-]+) -->", sub, text)
    out.write_text(
        "<!-- Generated by scripts/make_report_tables.py from "
        f"{template.name}; edit the template, not this file. -->\n\n" + text,
        encoding="utf-8",
    )


def update_readme(readme: Path, block: str) -> None:
    t = readme.read_text(encoding="utf-8")
    pat = re.compile(r"(<!-- RESULTS_TABLE_START[^>]*-->\n).*?(\n<!-- RESULTS_TABLE_END -->)", re.S)
    readme.write_text(pat.sub(lambda m: m.group(1) + block + m.group(2), t), encoding="utf-8")


def main() -> int:
    paths = get_paths()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--exp-id", default="p1_kvasir_pilot")
    parser.add_argument("--config", type=Path, default=paths.configs / "p1_kvasir_pilot.yaml")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    paths = get_paths(args.root)
    cfg = load_config(args.config)
    tables = paths.repo / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    split = json.loads((paths.splits / "kvasir_seg.json").read_text(encoding="utf-8"))
    n_images = len(split[cfg["split"]])
    raw = load_raw(paths, args.exp_id)
    label = leakage_label(paths.docs, "kvasir")
    out = {
        "data.md": table_data(paths),
        "parity.md": table_parity(paths),
        "p1_pilot_integrity.md": table_integrity(raw, cfg, n_images),
        "p1_pilot_box_jitter.md": table_dice(raw, "box_jitter", "jitter r"),
        "p1_pilot_box_scale.md": table_dice(raw, "box_scale", "scale s"),
        "p1_pilot_delta.md": table_delta(raw),
        "leakage_label.md": (
            f"Kvasir-SEG in MedSAM training data (from `docs/leakage.md`): **{label}**"
        ),
    }
    for name, body in out.items():
        (tables / name).write_text(body + "\n", encoding="utf-8")
        print(f"wrote {tables / name}")
    template = paths.repo / "reports" / "phase1_report.template.md"
    if template.exists():
        render(template, tables, paths.repo / "reports" / "phase1_report.md")
        print("rendered reports/phase1_report.md")
    update_readme(paths.repo / "README.md", results_block(raw))
    print("updated README.md results block")
    return 0


if __name__ == "__main__":
    sys.exit(main())
