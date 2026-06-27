"""GradCAM overlays + calibration reliability diagram (pre/post temperature scaling).

Implemented in Stage 7.
"""

from __future__ import annotations

import argparse

from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    load_config(args.config)
    raise NotImplementedError("Implemented in Stage 7.")


if __name__ == "__main__":
    main()
