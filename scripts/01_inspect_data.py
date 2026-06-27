"""Print split sizes, image-size distribution, and mask-coverage histogram; save a 3x3
image+mask overlay grid to outputs.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from src.data.splits import load_manifest
from src.utils.config import load_config
from src.utils.viz import overlay_mask, save_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["paths"]["data_root"])
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"

    manifest = load_manifest(cfg["paths"]["split_manifest"])
    print(
        f"Split sizes -> train: {len(manifest['train'])}, val: {len(manifest['val'])}, "
        f"test: {len(manifest['test'])}"
    )

    all_filenames = manifest["train"] + manifest["val"] + manifest["test"]

    widths, heights, coverages = [], [], []
    for filename in all_filenames:
        image = cv2.imread(str(images_dir / filename))
        h, w = image.shape[:2]
        widths.append(w)
        heights.append(h)

        mask = cv2.imread(str(masks_dir / filename), cv2.IMREAD_GRAYSCALE)
        coverages.append((mask > 127).mean())

    widths_arr, heights_arr, coverage_arr = np.array(widths), np.array(heights), np.array(coverages)
    print(
        f"Image width:  min={widths_arr.min()} max={widths_arr.max()} mean={widths_arr.mean():.1f}"
    )
    print(
        f"Image height: min={heights_arr.min()} max={heights_arr.max()} "
        f"mean={heights_arr.mean():.1f}"
    )
    resolutions = set(zip(widths, heights, strict=True))
    print(f"Distinct image resolutions: {len(resolutions)}")
    print(
        f"Polyp coverage (fraction of pixels): min={coverage_arr.min():.4f} "
        f"max={coverage_arr.max():.4f} mean={coverage_arr.mean():.4f}"
    )

    bins = np.linspace(0, coverage_arr.max() + 1e-6, 6)
    hist, edges = np.histogram(coverage_arr, bins=bins)
    print("Coverage histogram:")
    for count, lo, hi in zip(hist, edges[:-1], edges[1:], strict=True):
        print(f"  [{lo:.3f}, {hi:.3f}): {count}")

    sample_filenames = random.Random(cfg["seed"]).sample(all_filenames, min(9, len(all_filenames)))
    overlays = []
    for filename in sample_filenames:
        image = cv2.cvtColor(cv2.imread(str(images_dir / filename)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(masks_dir / filename), cv2.IMREAD_GRAYSCALE)
        overlays.append(overlay_mask(image, mask > 127))

    grid_path = Path(cfg["paths"]["output_root"]) / "data_inspection_grid.png"
    save_grid(overlays, grid_path, n_cols=3)
    print(f"Saved overlay grid to {grid_path}")


if __name__ == "__main__":
    main()
