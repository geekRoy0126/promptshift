"""Generate every number in the Phase 1 report from CSVs (spec 6.10).

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

from promptshift.analysis import stats
from promptshift.analysis.leakage import leakage_label  # noqa: F401  (used + re-exported)
from promptshift.config import load_config
from promptshift.paths import get_paths
from promptshift.runner import build_grid

TBD = "TBD"
MODEL_ORDER = ["sam_vit_b", "sam_vit_h", "medsam"]
BASELINES = ["unet", "classical"]
BOOT: dict = {"n": 2000, "seed": 42, "ci": 0.95}  # replaced from the config in main()


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


# --- tables ----------------------------------------------------------------------


def table_data(paths) -> str:
    rows = []
    for name, label in [
        ("kvasir_seg", "Kvasir-SEG"), ("isic2018", "ISIC 2018 (official train set)"),
        ("busi", "BUSI (lesion images)"), ("coco_val2017", "COCO val2017"),
    ]:
        f = paths.splits / f"{name}.json"
        if not f.exists():
            rows.append(
                {"dataset": label, "train": TBD, "val": TBD, "test": TBD, "objects (test)": TBD}
            )
            continue
        sp = json.loads(f.read_text(encoding="utf-8"))
        stats = paths.summary_results / f"data_stats_{name}.csv"
        n_obj = TBD
        if stats.exists() and name != "coco_val2017":
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


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return "NaN" if pd.isna(mean) else f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def table_dice(raw, perturb: str, label: str, boot: dict) -> str:
    if not raw:
        return tbd("pilot not run yet")
    cols, index = {}, None
    for m, df in raw.items():
        c = stats.curve(df, perturb, **boot).set_index("level")
        cols[m] = [fmt_ci(r.mean, r.lo, r.hi) for r in c.itertuples()]
        index = c.index
    t = pd.DataFrame(cols, index=index).reset_index().rename(columns={"level": label})
    note = (
        "Mean Dice over test images (seeds averaged within each image first) with a "
        f"{boot['ci']:.0%} percentile bootstrap CI over images ({boot['n']} resamples, seed "
        f"{boot['seed']}). Preliminary."
    )
    return md_table(t) + "\n\n" + note


def table_delta(raw, boot: dict) -> str:
    if not raw:
        return tbd("pilot not run yet")
    rows = []
    for m, df in raw.items():
        d = stats.delta_dice(df, 0.0, 0.2, **boot)
        rows.append({
            "model": m, "Dice @ 0%": d["d0"], "Dice @ 20%": d["d1"],
            "ΔDice (0 → 20%) [CI]": fmt_ci(d["delta"], d["lo"], d["hi"]),
        })  # fmt: skip
    return md_table(pd.DataFrame(rows))


def load_baselines(paths, exp_id: str) -> dict[str, pd.DataFrame]:
    d = paths.raw_results / exp_id
    return {
        b: pd.read_csv(d / f"{b}.csv", dtype={"image_id": str})
        for b in BASELINES
        if (d / f"{b}.csv").exists()
    }


def table_baselines(base, boot: dict, paths, exp_id: str) -> str:
    """Prompt-free methods on the test split."""
    if not base:
        return tbd("U-Net (P1_04) and classical (scripts/run_classical.py) not run yet")
    rows = []
    for b, df in base.items():
        s = stats.baseline_summary(df, **boot)
        ok = df[df["flag"] == "ok"]
        rows.append({
            "method": b, "images": s["n_images"],
            "Dice [CI]": fmt_ci(s["mean"], s["lo"], s["hi"]),
            "IoU": ok["iou"].mean(), "NSD (2 px)": ok["nsd"].mean(),
            "HD95 (px)": ok["hd95"].mean(),
            "empty predictions": int((ok["pred_area_frac"] == 0).sum()),
        })  # fmt: skip
    out = md_table(pd.DataFrame(rows))
    missing = [b for b in BASELINES if b not in base]
    if missing:
        out += f"\n\n{', '.join(missing)}: {TBD} (not run yet)."
    out += "\n\nHD95 means exclude empty predictions (NaN); their count is the last column."
    f = paths.raw_results / exp_id / "classical_params.json"
    if f.exists():
        c = json.loads(f.read_text(encoding="utf-8"))
        p = c["params"]
        out += (
            f"\n\nClassical parameters (chosen by {c['selection']}): channel `{p['channel']}`, "
            f"object = `{p['polarity']}` side of the Otsu threshold, Gaussian sigma = "
            f"{p['sigma']}, open/close radius = {p['radius']} px; train Dice "
            f"{c['train_dice']:.3f}, val Dice {c['val_dice']:.3f}. Grid: {c['grid']}."
        )
    return out


def table_crossover(raw, base, boot: dict) -> str:
    if not raw:
        return tbd("pilot not run yet")
    if "unet" not in base:
        return tbd("U-Net not run yet (P1_04)")
    ref = stats.baseline_summary(base["unet"], **boot)["mean"]
    rows = []
    for m, df in raw.items():
        c = stats.curve(df, "box_jitter", **boot)
        x = stats.crossover(c["level"], c["mean"], ref)
        interp = "—" if pd.isna(x["interp"]) else f"{x['interp']:.1%}"
        rows.append({
            "model": m, "first jitter level below U-Net": x["label"],
            "interpolated crossover": interp,
        })  # fmt: skip
    note = (
        f"U-Net test Dice = {ref:.3f}. Crossover = first box-jitter level whose mean Dice is "
        "below the U-Net mean, interpolated linearly between the two neighbouring levels. "
        "Preliminary (one U-Net seed; see the Wilcoxon table for significance)."
    )
    return md_table(pd.DataFrame(rows)) + "\n\n" + note


def table_wilcoxon(raw, base) -> str:
    """Paired Wilcoxon per box-jitter level, Holm across levels, for each comparison."""
    if not raw:
        return tbd("pilot not run yet")
    tabs = {m: stats.per_image(df, "box_jitter") for m, df in raw.items()}
    present = [m for m in MODEL_ORDER if m in tabs]
    pairs = [(a, b) for i, a in enumerate(present) for b in present[i + 1 :]]
    rows = []
    for a, b in pairs:
        rows += [(f"{a} − {b}", r) for r in stats.wilcoxon_by_level(tabs[a], tabs[b]).itertuples()]
    if "unet" in base:
        u = stats.baseline_per_image(base["unet"])
        for a in present:
            rows += [(f"{a} − unet", r) for r in stats.wilcoxon_by_level(tabs[a], u).itertuples()]
    out = pd.DataFrame([
        {"comparison": c, "jitter r": r.level, "n": r.n, "mean ΔDice": r.mean_diff,
         "p (Holm)": r.p_holm}
        for c, r in rows
    ])  # fmt: skip
    note = (
        "Two-sided paired Wilcoxon signed-rank test per level on per-image Dice (seeds "
        "averaged); Holm correction across the levels of each comparison. Preliminary."
    )
    return md_table(out, "{:.3g}") + "\n\n" + note


def results_block(raw) -> str:
    if not raw:
        return TBD
    return "**Kvasir-SEG pilot, box jitter (Phase 1, preliminary)**\n\n" + table_delta(raw, BOOT)


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
    a = cfg["analysis"]
    boot = {"n": a["bootstrap_n"], "seed": a["bootstrap_seed"], "ci": a["ci"]}
    BOOT.update(boot)
    base = load_baselines(paths, args.exp_id)
    out = {
        "data.md": table_data(paths),
        "parity.md": table_parity(paths),
        "p1_pilot_integrity.md": table_integrity(raw, cfg, n_images),
        "p1_pilot_box_jitter.md": table_dice(raw, "box_jitter", "jitter r", boot),
        "p1_pilot_box_scale.md": table_dice(raw, "box_scale", "scale s", boot),
        "p1_pilot_delta.md": table_delta(raw, boot),
        "p1_baselines.md": table_baselines(base, boot, paths, args.exp_id),
        "p1_crossover.md": table_crossover(raw, base, boot),
        "p1_wilcoxon.md": table_wilcoxon(raw, base),
        "leakage_label.md": (
            f"Kvasir-SEG in MedSAM training data (from `docs/leakage.md`): **{label}**"
        ),
    }
    for name, body in out.items():
        (tables / name).write_text(body + "\n", encoding="utf-8", newline="\n")
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
