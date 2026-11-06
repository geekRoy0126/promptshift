"""MedSAM wrapper, line by line from the official MedSAM_Inference.py.

Source checked: https://github.com/bowang-lab/MedSAM, MedSAM_Inference.py at commit d71e8a1.
Official pipeline (each step below cites the source line):
  1. grayscale -> repeat to 3 channels                          (L115-118)
  2. skimage.transform.resize(img, (1024, 1024), order=3,
     preserve_range=True, anti_aliasing=True).astype(np.uint8)  (L121-123; truncation, not rounding)
  3. global min-max over the whole image, clip denominator 1e-8 (L124-126)
  4. image_encoder on the (1, 3, 1024, 1024) float tensor,
     no pixel_mean / pixel_std                                  (L128-136)
  5. box_1024 = box / [W, H, W, H] * 1024                       (L134)
  6. prompt_encoder(points=None, boxes=box, masks=None) ->
     mask_decoder(..., multimask_output=False)                  (L44-60)
  7. sigmoid -> bilinear (align_corners=False) to (H, W) -> > 0.5 (L62-71)
Difference from the official CLI: it parses the box as ints (L132). We pass float boxes
unchanged so perturbed boxes are not rounded; GT boxes are integers anyway.
Model: facebookresearch sam_model_registry["vit_b"] with the MedSAM state dict (same
architecture; the MedSAM fork only adds map_location="cpu" when loading, done here too).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from promptshift.models.base import load_state_dict, select_device

WEIGHT_FILE = "medsam_vit_b.pth"


def preprocess(image: np.ndarray) -> np.ndarray:
    """Steps 1-3: H x W (x 3) -> 1024 x 1024 x 3 float64 in [0, 1]."""
    from skimage import transform

    img = np.asarray(image)
    img_3c = np.repeat(img[:, :, None], 3, axis=-1) if img.ndim == 2 else img
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    return (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )


class MedSamSegmenter:
    name = "medsam"
    supports_points = False

    def __init__(self, weights_dir: Path, device: str = "auto") -> None:
        import torch
        from segment_anything import sam_model_registry

        ckpt = Path(weights_dir) / WEIGHT_FILE
        if not ckpt.is_file():
            msg = f"{ckpt} missing; run scripts/download_weights.py --model medsam"
            raise FileNotFoundError(msg)
        self.device = select_device(device)
        model = sam_model_registry["vit_b"](checkpoint=None)
        model.load_state_dict(load_state_dict(ckpt))
        model.to(self.device).eval()
        self.model = model
        self._torch = torch
        self._embedding = None
        self._hw: tuple[int, int] | None = None

    def set_image(self, image: np.ndarray) -> None:
        torch = self._torch
        image = np.asarray(image)
        self._hw = image.shape[:2]
        img_1024 = preprocess(image)
        tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self._embedding = self.model.image_encoder(tensor)

    def predict_box(self, box_xyxy: np.ndarray) -> np.ndarray:
        torch = self._torch
        import torch.nn.functional as F

        if self._embedding is None:
            raise RuntimeError("call set_image first")
        h, w = self._hw
        box_1024 = np.asarray(box_xyxy, dtype=np.float64)[None] / np.array([w, h, w, h]) * 1024
        box_t = torch.as_tensor(box_1024, dtype=torch.float, device=self.device)[:, None, :]
        with torch.no_grad():
            sparse, dense = self.model.prompt_encoder(points=None, boxes=box_t, masks=None)
            low_res_logits, _ = self.model.mask_decoder(
                image_embeddings=self._embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            prob = F.interpolate(
                torch.sigmoid(low_res_logits), size=(h, w), mode="bilinear", align_corners=False
            )
        return prob.squeeze().cpu().numpy() > 0.5

    def predict_points(self, coords, labels, mask_input=None):
        raise NotImplementedError("MedSAM is box-only (supports_points = False)")
