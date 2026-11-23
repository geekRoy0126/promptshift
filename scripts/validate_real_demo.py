"""Exercise the Gradio HTTP API with real weights and two boxes on one BUSI image."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from promptshift.config import load_config
from promptshift.data import get_dataset
from promptshift.demo import build_app
from promptshift.paths import get_paths


def main():
    from gradio_client import Client, handle_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--image-id", default="benign_0330")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", default=7869, type=int)
    args = parser.parse_args()
    paths = get_paths(args.root).ensure()
    cfg = load_config(paths.configs / "c_busi_box.yaml")
    ds = get_dataset("busi", paths.data, cfg["data"], paths.splits / "busi.json")
    if args.image_id not in ds.ids("test"):
        raise ValueError("Use an image in the committed test split")
    sample = ds.get(args.image_id)
    out = paths.results / "validation/real_demo"
    out.mkdir(parents=True, exist_ok=True)
    # A temporary upload source is kept outside exported results.
    image = paths.root / "demo_validation_input.png"
    Image.fromarray(sample.image).save(image)
    app = build_app(paths.weights, device=args.device, mock=False)
    records = []
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=args.port,
            share=False,
            prevent_thread_lock=True,
            quiet=True,
        )
        client = Client(f"http://127.0.0.1:{args.port}", verbose=False)
        for label, box in [("wide", [20, 20, 80, 80]), ("shifted", [40, 10, 95, 65])]:
            started = time.perf_counter()
            result = client.predict(handle_file(str(image)), *box, api_name="/predict")
            assert "SAM ViT-B and MedSAM" in result[2] and "Mock" not in result[2]
            arrays = []
            for name, file in zip(["sam_vit_b", "medsam"], result[:2], strict=True):
                array = np.asarray(Image.open(file).convert("RGB"))
                assert array.shape == sample.image.shape
                arrays.append(array)
                shutil.copy(file, out / f"{label}_{name}.png")
            records.append(
                dict(
                    condition=label,
                    box_percent=box,
                    seconds=time.perf_counter() - started,
                    status=result[2],
                    model_outputs_differ=not np.array_equal(*arrays),
                )
            )
            print(records[-1], flush=True)
        (out / "validation.json").write_text(
            json.dumps(
                dict(
                    image_id=args.image_id,
                    device=args.device,
                    mode="Real pretrained weights through Gradio HTTP API; no public share link",
                    ground_truth_used=False,
                    runs=records,
                ),
                indent=2,
            )
        )
    finally:
        app.close()
        image.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
