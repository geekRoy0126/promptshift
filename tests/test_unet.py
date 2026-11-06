import numpy as np
import pytest
import torch

pytest.importorskip("segmentation_models_pytorch")

from promptshift.baselines.unet import (  # noqa: E402
    augment,
    batch_tensors,
    bce_dice_loss,
    build_model,
    predict,
    resize_mask,
    to_input,
)  # fmt: skip


def test_augment_moves_image_and_mask_together():
    g = torch.Generator().manual_seed(0)
    m = np.zeros((1, 16, 16), np.uint8)
    m[0, 2:6, 3:9] = 1
    img = np.repeat(m[..., None] * 255, 3, axis=-1).astype(np.uint8)
    x, y = batch_tensors(img, m, "cpu")
    for _ in range(10):
        xa, ya = augment(x, y, g, jitter=0.0)
        assert torch.equal((xa[:, :1] > 0.5).float(), ya)


def test_augment_is_seeded():
    x = torch.rand(2, 3, 8, 8)
    y = (torch.rand(2, 1, 8, 8) > 0.5).float()
    a = augment(x, y, torch.Generator().manual_seed(1), 0.1)
    b = augment(x, y, torch.Generator().manual_seed(1), 0.1)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert a[0].min() >= 0 and a[0].max() <= 1


def test_loss_prefers_correct_logits():
    y = torch.zeros(1, 1, 8, 8)
    y[..., 2:6, 2:6] = 1
    good = bce_dice_loss((y * 2 - 1) * 10, y)
    bad = bce_dice_loss(-(y * 2 - 1) * 10, y)
    assert good < 0.01 < bad


def test_predict_returns_original_resolution():
    torch.manual_seed(0)
    model = build_model("resnet34", None)
    img = np.random.default_rng(0).integers(0, 255, (50, 70, 3), dtype=np.uint8)
    pred = predict(model, img, 64, "cpu")
    assert pred.shape == (50, 70) and pred.dtype == bool
    assert to_input(img, 64).shape == (1, 3, 64, 64)
    assert resize_mask(np.ones((50, 70), bool), 64).shape == (64, 64)
