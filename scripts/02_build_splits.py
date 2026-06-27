"""Generate and freeze the deterministic split manifest (configs/splits.json)."""

from __future__ import annotations

import argparse

from src.data.splits import build_splits
from src.utils.config import load_config
from src.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    manifest = build_splits(cfg)
    print(
        f"Train: {len(manifest['train'])}, Val: {len(manifest['val'])}, "
        f"Test: {len(manifest['test'])} -> manifest written to {cfg['paths']['split_manifest']}"
    )


if __name__ == "__main__":
    main()
