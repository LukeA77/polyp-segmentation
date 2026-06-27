"""Export best checkpoint to ONNX, verify parity, and benchmark CPU vs GPU latency.

Writes model.onnx + latency.md. Implemented in Stage 8.
"""

from __future__ import annotations

import argparse

from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    load_config(args.config)
    raise NotImplementedError("Implemented in Stage 8.")


if __name__ == "__main__":
    main()
