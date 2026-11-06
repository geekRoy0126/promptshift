"""SAM ViT-B / ViT-H through the official SamPredictor (CLAUDE.md 6.2).

SamPredictor does all resizing, normalization and padding; nothing is reimplemented.
Protocol:
- box: predict(box=box[None], multimask_output=False)
- single click (no mask_input): multimask_output=True, keep the highest predicted IoU
- later clicks (mask_input given): multimask_output=False with the previous low-res logits
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from promptshift.models.base import load_state_dict, select_device

WEIGHT_FILES = {
    "sam_vit_b": ("vit_b", "sam_vit_b_01ec64.pth"),
    "sam_vit_h": ("vit_h", "sam_vit_h_4b8939.pth"),
}


class SamSegmenter:
    supports_points = True

    def __init__(self, name: str, weights_dir: Path, device: str = "auto") -> None:
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        if name not in WEIGHT_FILES:
            raise KeyError(f"unknown SAM variant {name!r}")
        arch, fname = WEIGHT_FILES[name]
        ckpt = Path(weights_dir) / fname
        if not ckpt.is_file():
            msg = f"{ckpt} missing; run scripts/download_weights.py --model {name}"
            raise FileNotFoundError(msg)
        self.name = name
        self.device = select_device(device)
        model = sam_model_registry[arch](checkpoint=None)
        model.load_state_dict(load_state_dict(ckpt))
        model.to(self.device).eval()
        self.model = model
        self.predictor = SamPredictor(model)
        self._torch = torch

    def set_image(self, image: np.ndarray) -> None:
        with self._torch.no_grad():
            self.predictor.set_image(np.asarray(image, dtype=np.uint8), image_format="RGB")

    def predict_box(self, box_xyxy: np.ndarray) -> np.ndarray:
        box = np.asarray(box_xyxy, dtype=np.float32)
        with self._torch.no_grad():
            masks, _, _ = self.predictor.predict(box=box[None], multimask_output=False)
        return masks[0].astype(bool)

    def predict_points(self, coords, labels, mask_input=None):
        coords = np.asarray(coords, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        with self._torch.no_grad():
            if mask_input is None:
                masks, scores, logits = self.predictor.predict(
                    point_coords=coords, point_labels=labels, multimask_output=True
                )
                best = int(np.argmax(scores))
            else:
                masks, scores, logits = self.predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    mask_input=np.asarray(mask_input, dtype=np.float32)[None],
                    multimask_output=False,
                )
                best = 0
        return masks[best].astype(bool), logits[best]
