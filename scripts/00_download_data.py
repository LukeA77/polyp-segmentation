"""Download Kvasir-SEG into `paths.data_root` and verify integrity. Idempotent."""

from __future__ import annotations

import argparse

from src.data.download import ensure_kvasir_seg
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, (n_images, n_masks, n_unreadable) = ensure_kvasir_seg(cfg)
    print(
        f"Verified: {n_images} images, {n_masks} masks, filenames matched, "
        f"{n_unreadable} unreadable."
    )


if __name__ == "__main__":
    main()
