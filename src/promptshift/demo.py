"""Interactive box comparison. Models stay server-side; no GT required."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from promptshift.models import load_model


def pixel_box(image, x0, y0, x1, y1):
    h, w = image.shape[:2]
    coords = np.asarray([x0, y0, x1, y1], dtype=float)
    if not np.isfinite(coords).all() or (coords < 0).any() or (coords > 100).any():
        raise ValueError("Box coordinates must be percentages between 0 and 100.")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("The bottom-right corner must be below and right of the top-left corner.")
    return coords / 100 * np.array([w, h, w, h])


def overlay(image, mask, box):
    rgb = np.asarray(image, dtype=np.uint8).copy()
    rgb[mask] = (0.55 * rgb[mask] + 0.45 * np.array([240, 40, 170])).astype(np.uint8)
    out = Image.fromarray(rgb)
    ImageDraw.Draw(out).rectangle(tuple(box), outline=(255, 210, 30), width=3)
    return np.asarray(out)


class Comparison:
    def __init__(self, weights: Path, device="auto", mock=False):
        self.weights, self.device = weights, device
        self.names = ["mock"] if mock else ["sam_vit_b", "medsam"]
        self.models = {}
        self.lock = threading.Lock()

    def predict(self, image, x0, y0, x1, y1):
        if image is None:
            raise ValueError("Upload an image first.")
        image = np.asarray(image, dtype=np.uint8)
        box = pixel_box(image, x0, y0, x1, y1)
        outputs = []
        # Wrappers retain embeddings; lock the full request to prevent cross-session races.
        with self.lock:
            for name in self.names:
                if name not in self.models:
                    self.models[name] = load_model(name, self.weights, self.device)
                model = self.models[name]
                model.set_image(image)
                pred = model.predict_box(box)
                outputs.append(overlay(image, pred, box))
        if len(outputs) == 1:
            outputs *= 2
        return (
            *outputs,
            "Mock mode: synthetic masks, not model results."
            if self.names == ["mock"]
            else "SAM ViT-B and MedSAM; same box. No ground truth supplied, so no Dice score.",
        )


def build_app(weights, device="auto", mock=False):
    import gradio as gr

    comparison = Comparison(Path(weights), device, mock)
    with gr.Blocks(title="PromptShift") as app:
        gr.Markdown(
            "# PromptShift\nCompare SAM and MedSAM as you adjust the prompt box. "
            "Yellow: prompt box. Magenta: predicted region."
        )
        if mock:
            gr.Markdown("**MOCK MODE: both panels show synthetic masks for UI testing.**")
        image = gr.Image(type="numpy", image_mode="RGB", label="Upload an image")
        with gr.Row():
            x0 = gr.Slider(0, 100, value=20, label="Left (%)")
            y0 = gr.Slider(0, 100, value=20, label="Top (%)")
            x1 = gr.Slider(0, 100, value=80, label="Right (%)")
            y1 = gr.Slider(0, 100, value=80, label="Bottom (%)")
        run = gr.Button("Compare", variant="primary")
        with gr.Row():
            sam = gr.Image(label="SAM ViT-B" if not mock else "Mock A")
            med = gr.Image(label="MedSAM" if not mock else "Mock B")
        status = gr.Textbox(label="Status", interactive=False)

        def predict(*args):
            try:
                return comparison.predict(*args)
            except (ValueError, FileNotFoundError) as e:
                raise gr.Error(str(e)) from e

        run.click(predict, [image, x0, y0, x1, y1], [sam, med, status], concurrency_limit=1)
    return app
