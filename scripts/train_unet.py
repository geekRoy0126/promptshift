"""Train the U-Net baseline and evaluate it on the test split (P1-10, CLAUDE.md 6.6).

Usage:
    python scripts/train_unet.py [--config configs/p1_kvasir_pilot.yaml] [--device auto]
        [--epochs N] [--max-train N] [--max-val N] [--max-test N] [--encoder-weights none]
        [--exp-id ID] [--root PATH]
Resumable: the last checkpoint is saved after every epoch; rerunning continues from it.

Writes (under the storage root; on Colab = Drive):
    checkpoints/<exp_id>/unet_last.pt, unet_best.pt     (best = highest val Dice)
    results/raw/<exp_id>/unet_history.csv               per-epoch loss, val Dice, lr, time
    results/raw/<exp_id>/unet.csv                       test rows, runner format, perturb "none"
    results/raw/<exp_id>/unet_train.json                training record
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from promptshift.baselines import baseline_row, write_rows
from promptshift.baselines.unet import (
    TrainState,
    augment,
    batch_tensors,
    bce_dice_loss,
    build_model,
    normalize,
    predict,
    resize_image,
    resize_mask,
    seed_everything,
)  # fmt: skip
from promptshift.config import load_config
from promptshift.data import get_dataset
from promptshift.metrics import dice
from promptshift.models import select_device
from promptshift.paths import get_paths
from promptshift.runner import git_sha

METHOD = "unet"


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_resized(ds, ids, size):
    xs, ys = [], []
    for image_id in tqdm(ids, desc="load train"):
        s = ds.get(image_id)
        xs.append(resize_image(s.image, size))
        ys.append(resize_mask(s.mask, size))
    return np.stack(xs), np.stack(ys)


def val_dice(model, ds, ids, size, device, thr) -> float:
    return float(np.mean([dice(predict(model, (s := ds.get(i)).image, size, device, thr), s.mask)
                          for i in ids]))  # fmt: skip


def main() -> int:
    paths = get_paths()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=paths.configs / "p1_kvasir_pilot.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None, help="smoke runs only")
    parser.add_argument("--max-val", type=int, default=None, help="smoke runs only")
    parser.add_argument("--max-test", type=int, default=None, help="smoke runs only")
    parser.add_argument("--encoder-weights", default=None, help="'none' = random init (offline)")
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    paths = get_paths(args.root).ensure()
    cfg = load_config(args.config)
    u = dict(cfg["baselines"]["unet"])
    if args.epochs is not None:
        u["epochs"] = args.epochs
    if args.encoder_weights is not None:
        u["encoder_weights"] = None if args.encoder_weights == "none" else args.encoder_weights
    exp_id = args.exp_id or cfg["exp_id"]
    name, size, thr = cfg["dataset"], u["input_size"], u["threshold"]
    device = select_device(args.device)
    sha = git_sha(paths.repo)
    ds = get_dataset(name, paths.data, cfg["data"], paths.splits / f"{name}.json")
    train_ids = ds.ids("train")[: args.max_train]
    val_ids = ds.ids("val")[: args.max_val]
    test_ids = ds.ids(cfg["split"])[: args.max_test]
    out_dir = paths.raw_results / exp_id
    ckpt_dir = paths.root / "checkpoints" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last, best = ckpt_dir / f"{METHOD}_last.pt", ckpt_dir / f"{METHOD}_best.pt"
    print(f"{name}: train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}; "
          f"device {device}; git {sha}; {u}")  # fmt: skip

    seed_everything(u["seed"])
    gen = torch.Generator().manual_seed(u["seed"])
    model = build_model(u["encoder"], u["encoder_weights"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=u["lr"], weight_decay=u["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=u["epochs"])
    state, history = TrainState(), []
    if last.is_file():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"])
        state, history = TrainState(**ck["state"]), ck["history"]
        print(f"resumed after epoch {state.epoch} (best val Dice {state.best_val_dice:.4f})")

    if state.epoch < u["epochs"]:
        x_all, y_all = load_resized(ds, train_ids, size)
    bs = min(u["batch_size"], len(train_ids))
    for epoch in range(state.epoch, u["epochs"]):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(train_ids), generator=gen).numpy()
        losses = []
        for k in range(0, len(perm) - bs + 1, bs):  # drop the last partial batch
            idx = np.sort(perm[k : k + bs])
            x, y = batch_tensors(x_all[idx], y_all[idx], "cpu")
            x, y = augment(x, y, gen, u["color_jitter"])
            x, y = normalize(x.to(device)), y.to(device)
            loss = bce_dice_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        lr = opt.param_groups[0]["lr"]
        sched.step()
        vd = val_dice(model, ds, val_ids, size, device, thr)
        if vd > state.best_val_dice:
            state.best_val_dice, state.best_epoch = vd, epoch + 1
            atomic_save({"model": model.state_dict(), "epoch": epoch + 1, "val_dice": vd}, best)
        state.epoch = epoch + 1
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_dice": vd,
                        "lr": lr, "seconds": time.time() - t0})  # fmt: skip
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "gen": gen.get_state(),
                     "state": asdict(state), "history": history}, last)  # fmt: skip
        pd.DataFrame(history).to_csv(out_dir / f"{METHOD}_history.csv", index=False)
        print(f"epoch {epoch + 1}/{u['epochs']}: loss {np.mean(losses):.4f}, val Dice {vd:.4f}"
              f"{' *' if state.best_epoch == epoch + 1 else ''} ({time.time() - t0:.0f} s)",
              flush=True)  # fmt: skip

    ck = torch.load(best, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    rows = []
    for image_id in tqdm(test_ids, desc="test"):
        s = ds.get(image_id)
        t0 = time.perf_counter()
        pred = predict(model, s.image, size, device, thr)
        ms = (time.perf_counter() - t0) * 1000
        rows.append(baseline_row(
            exp_id=exp_id, split=cfg["split"], method=METHOD, sample=s, pred=pred, ms=ms,
            sha=sha, nsd_tolerance_px=cfg["metrics"]["nsd_tolerance_px"],
        ))  # fmt: skip
    df = write_rows(out_dir / f"{METHOD}.csv", rows)
    record = {
        "method": METHOD, "dataset": name, "config": u, "device": str(device),
        "torch": torch.__version__, "git_sha": sha, "n_train": len(train_ids),
        "n_val": len(val_ids), "n_test": len(df), "best_epoch": state.best_epoch,
        "best_val_dice": state.best_val_dice, "test_dice": float(df["dice"].mean()),
        "selection": "checkpoint with the highest val Dice (original resolution)",
    }  # fmt: skip
    (out_dir / f"{METHOD}_train.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
