"""P1-6 acceptance checks (CLAUDE.md 6.2).

1. MedSAM parity: on 3 single-object Kvasir test images with GT boxes, compare our
   MedSamSegmenter with the official MedSAM_Inference.py, run unmodified as a
   subprocess from a pinned clone of bowang-lab/MedSAM (it then imports MedSAM's own
   bundled segment_anything). Required: Dice >= 0.99 for every image.
2. SAM parity: for each available SAM variant, our SamSegmenter.predict_box must be
   pixel-identical to a direct SamPredictor.predict(box=..., multimask_output=False).

Usage:
    python scripts/model_parity.py [--sam sam_vit_b --sam sam_vit_h] [--device auto]
Writes results/parity/medsam_parity.csv and results/parity/sam_parity.csv
(under the storage root). Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from promptshift.data import get_dataset
from promptshift.metrics import dice
from promptshift.models import load_model, select_device
from promptshift.paths import get_paths
from promptshift.prompts import box_from_mask

MEDSAM_REPO = "https://github.com/bowang-lab/MedSAM.git"
MEDSAM_COMMIT = "d71e8a1a99ad751840a22a7fa3ecfb4166fb1488"
N_IMAGES = 3
THRESHOLD = 0.99


def ensure_medsam_repo(dest: Path) -> Path:
    """Clone bowang-lab/MedSAM at the pinned commit (idempotent)."""
    script = dest / "MedSAM_Inference.py"
    if not script.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", MEDSAM_REPO], check=True)
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", MEDSAM_COMMIT],
            check=True,
        )
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"], check=True)
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != MEDSAM_COMMIT:
        print(f"WARNING: {dest} is at {head}, expected {MEDSAM_COMMIT}")
    return dest


def pick_images(ds, n: int) -> list[str]:
    """First n single-object images of the test split (sorted ids)."""
    out = []
    for image_id in ds.ids("test"):
        if len(ds.get(image_id).objects) == 1:
            out.append(image_id)
        if len(out) == n:
            return out
    raise RuntimeError("not enough single-object test images")


def run_official(
    repo: Path, image: np.ndarray, box: np.ndarray, ckpt: Path, device: str
) -> np.ndarray:
    """Run MedSAM_Inference.py unmodified on a lossless PNG copy of the image.

    The script saves its mask as "seg_" + input file name. With a .jpg input the 0/1
    mask would be written as a lossy JPEG and come back with compression noise, so the
    input is passed as PNG (the decoded pixels are identical to the JPEG's).
    """
    from skimage import io

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "input.png"
        io.imsave(src, image, check_contrast=False)
        box_arg = "[" + ",".join(str(int(v)) for v in box) + "]"
        cmd = [
            sys.executable, "MedSAM_Inference.py", "-i", str(src), "-o", str(tmp),
            "--box", box_arg, "-chk", str(ckpt), "--device", device,
        ]  # fmt: skip
        env = {**os.environ, "MPLBACKEND": "Agg"}
        # cwd = repo so `import segment_anything` resolves to MedSAM's bundled fork
        subprocess.run(cmd, cwd=repo, env=env, check=True, capture_output=True, text=True)
        return io.imread(tmp / "seg_input.png") > 0


def medsam_parity(paths, ds, image_ids, device, repo) -> pd.DataFrame:
    model = load_model("medsam", paths.weights, device)
    rows = []
    for image_id in image_ids:
        s = ds.get(image_id)
        box = box_from_mask(s.objects[0])
        t0 = time.perf_counter()
        ckpt = paths.weights / "medsam_vit_b.pth"
        official = run_official(repo, s.image, box, ckpt, device)
        t_official = time.perf_counter() - t0
        model.set_image(s.image)
        ours = model.predict_box(box)
        d = dice(ours, official)
        rows.append({
            "image_id": image_id, "box": box.astype(int).tolist(), "dice_ours_vs_official": d,
            "pixels_differ": int((ours != official).sum()), "dice_ours_vs_gt": dice(ours, s.mask),
            "dice_official_vs_gt": dice(official, s.mask), "pass": d >= THRESHOLD,
            "official_seconds": round(t_official, 1),
        })  # fmt: skip
        print(f"  medsam {image_id}: dice(ours, official) = {d:.6f}")
    return pd.DataFrame(rows)


def sam_parity(paths, ds, image_ids, variants, device) -> pd.DataFrame:
    import torch
    from segment_anything import SamPredictor

    rows = []
    for name in variants:
        try:
            model = load_model(name, paths.weights, device)
        except FileNotFoundError as e:
            print(f"  skip {name}: {e}")
            continue
        direct = SamPredictor(model.model)  # same weights, plain official API
        for image_id in image_ids:
            s = ds.get(image_id)
            box = box_from_mask(s.objects[0])
            model.set_image(s.image)
            ours = model.predict_box(box)
            with torch.no_grad():
                direct.set_image(s.image)
                ref, _, _ = direct.predict(box=box[None], multimask_output=False)
            same = bool(np.array_equal(ours, ref[0]))
            rows.append({
                "model": name, "image_id": image_id, "identical": same,
                "pixels_differ": int((ours != ref[0]).sum()), "dice_vs_gt": dice(ours, s.mask),
            })  # fmt: skip
            print(f"  {name} {image_id}: identical = {same}")
    return pd.DataFrame(rows)


def write_environment(path: Path, device: str) -> None:
    import platform

    import torch

    env = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.system(),
    }
    path.write_text(json.dumps(env, indent=1), encoding="utf-8")


def main() -> int:
    paths = get_paths()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=paths.configs / "default.yaml")
    parser.add_argument("--sam", action="append", default=None, help="SAM variants to check")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-medsam", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = select_device(args.device)
    ds = get_dataset("kvasir_seg", paths.data, cfg["data"], paths.splits / "kvasir_seg.json")
    image_ids = pick_images(ds, N_IMAGES)
    out_dir = paths.results / "parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_environment(out_dir / "environment.json", device)
    ok = True
    print(f"device: {device}; images: {image_ids}")

    if not args.skip_medsam:
        repo = ensure_medsam_repo(paths.repo / "external" / "MedSAM")
        df = medsam_parity(paths, ds, image_ids, device, repo)
        df.to_csv(out_dir / "medsam_parity.csv", index=False)
        ok &= bool(df["pass"].all())
        verdict = "PASS" if df["pass"].all() else "FAIL"
        worst = df["dice_ours_vs_official"].min()
        print(f"MedSAM parity: {verdict} (min dice {worst:.6f}) -> {out_dir / 'medsam_parity.csv'}")

    df = sam_parity(paths, ds, image_ids, args.sam or ["sam_vit_b", "sam_vit_h"], device)
    if len(df):
        df.to_csv(out_dir / "sam_parity.csv", index=False)
        ok &= bool(df["identical"].all())
        verdict = "PASS" if df["identical"].all() else "FAIL"
        print(f"SAM parity: {verdict} -> {out_dir / 'sam_parity.csv'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
