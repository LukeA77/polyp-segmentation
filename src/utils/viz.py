"""Overlay rendering and shared figure helpers, used from Stage 2 (data inspection)
onward through evaluation, explainability, and inference.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a binary `mask` onto `image` (HxWx3 uint8) and return the annotated copy."""
    mask_bool = mask.astype(bool)
    overlay = image.copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool].astype(np.float32) + alpha * color_arr
    ).astype(np.uint8)
    return overlay


def save_grid(images: list[np.ndarray], out_path: str | Path, n_cols: int = 3) -> None:
    """Save a grid of images (e.g. image+mask overlays) to `out_path`."""
    n_rows = math.ceil(len(images) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, image in zip(axes, images, strict=False):
        ax.imshow(image)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
