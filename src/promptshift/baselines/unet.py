"""Supervised U-Net baseline (CLAUDE.md 6.6 default; there is no course_ref/ pipeline).

smp.Unet(resnet34, imagenet) on 352x352 inputs; augmentation hflip / vflip / rot90 / light color
jitter; loss BCE + soft Dice; AdamW lr 1e-4, wd 1e-4, batch 8, cosine schedule; the checkpoint
with the best val Dice (at original resolution, same metric as test) is kept.

Training images are resized once and kept in memory; augmentation runs on the batch tensor with a
seeded torch.Generator, so there are no DataLoader workers to seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(encoder: str = "resnet34", encoder_weights: str | None = "imagenet"):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def normalize(x: torch.Tensor) -> torch.Tensor:
    """float N x 3 x H x W in [0, 1] -> ImageNet-normalized."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """uint8 H x W x 3 -> uint8 size x size x 3 (area interpolation, i.e. anti-aliased)."""
    import cv2

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)


def batch_tensors(images_u8: np.ndarray, masks_u8: np.ndarray, device):
    """N x S x S x 3 uint8, N x S x S uint8 -> float images in [0, 1] and N x 1 x S x S targets."""
    x = torch.from_numpy(images_u8).to(device).permute(0, 3, 1, 2).float() / 255.0
    y = torch.from_numpy(masks_u8).to(device).float()[:, None]
    return x, y


def to_input(image: np.ndarray, size: int) -> torch.Tensor:
    """One uint8 H x W x 3 image -> normalized 1 x 3 x size x size."""
    x = torch.from_numpy(resize_image(image, size)).permute(2, 0, 1).float()[None] / 255.0
    return normalize(x)


def augment(x: torch.Tensor, y: torch.Tensor, g: torch.Generator, jitter: float):
    """Random hflip, vflip, rot90 (same for image and mask) + brightness/contrast/saturation.

    x: float images in [0, 1] (before normalization); g: CPU generator (fixed seed)."""
    n = x.shape[0]
    out_x, out_y = [], []
    for i in range(n):
        xi, yi = x[i], y[i]
        r = torch.rand(3, generator=g)
        if r[0] < 0.5:
            xi, yi = xi.flip(-1), yi.flip(-1)
        if r[1] < 0.5:
            xi, yi = xi.flip(-2), yi.flip(-2)
        k = int(torch.randint(0, 4, (1,), generator=g))
        xi, yi = torch.rot90(xi, k, (-2, -1)), torch.rot90(yi, k, (-2, -1))
        if jitter > 0:
            b, c, s = (1 + (torch.rand(3, generator=g) * 2 - 1) * jitter).tolist()
            mean = xi.mean()
            xi = (xi - mean) * c + mean * b  # contrast about the mean, then brightness
            gray = xi.mean(0, keepdim=True)
            xi = (gray + (xi - gray) * s).clamp(0, 1)  # saturation
        out_x.append(xi)
        out_y.append(yi)
    return torch.stack(out_x), torch.stack(out_y)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return bce + (1 - (2 * inter + eps) / (den + eps)).mean()


@torch.no_grad()
def predict(model, image: np.ndarray, size: int, device, threshold: float = 0.5) -> np.ndarray:
    """Original-resolution boolean mask: sigmoid, bilinear upsample, > threshold."""
    model.eval()
    logits = model(to_input(image, size).to(device))
    prob = torch.sigmoid(logits)
    prob = F.interpolate(prob, size=image.shape[:2], mode="bilinear", align_corners=False)
    return (prob[0, 0] > threshold).cpu().numpy()


@dataclass
class TrainState:
    epoch: int = 0  # epochs finished
    best_val_dice: float = -math.inf
    best_epoch: int = -1
