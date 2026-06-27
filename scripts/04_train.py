"""Full training run via src.engine.trainer.Trainer. Implemented in Stage 5."""

from __future__ import annotations

import argparse

from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")
    args = parser.parse_args()

    load_config(args.config)
    raise NotImplementedError("Implemented in Stage 5.")


if __name__ == "__main__":
    main()
