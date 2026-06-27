"""Deterministic image-level train/val/test split + manifest I/O.

Kvasir-SEG ships no patient IDs, so a true patient-level split is impossible here; we
use a fixed-seed image-level split (honest limitation, stated in the README). The
optional `groups` mapping (filename -> group id) lets Projects 3-5, which do have
case/patient grouping, reuse this exact manifest mechanism with grouped splitting
instead of a fresh implementation.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def build_splits(
    cfg: dict[str, Any],
    groups: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """Build a deterministic train/val/test split of filenames and write the manifest.

    Splits at the image level by default. If `groups` is given (filename -> group id),
    splits at the group level instead, so every file sharing a group lands in the same
    split — unused for Kvasir-SEG, reserved for grouped-split projects.
    """
    images_dir = Path(cfg["paths"]["data_root"]) / "images"
    filenames = sorted(p.name for p in images_dir.iterdir() if p.is_file())

    subset = cfg["data"].get("subset")
    if subset is not None:
        rng = random.Random(cfg["seed"])
        filenames = sorted(rng.sample(filenames, min(subset, len(filenames))))

    val_fraction = cfg["data"]["val_fraction"]
    test_fraction = cfg["data"]["test_fraction"]

    units = list(filenames) if groups is None else sorted({groups[f] for f in filenames})
    random.Random(cfg["seed"]).shuffle(units)

    n_val = round(len(units) * val_fraction)
    n_test = round(len(units) * test_fraction)
    val_units = set(units[:n_val])
    test_units = set(units[n_val : n_val + n_test])
    train_units = set(units[n_val + n_test :])

    if groups is None:
        manifest = {
            "train": sorted(train_units),
            "val": sorted(val_units),
            "test": sorted(test_units),
        }
    else:
        manifest = {
            "train": sorted(f for f in filenames if groups[f] in train_units),
            "val": sorted(f for f in filenames if groups[f] in val_units),
            "test": sorted(f for f in filenames if groups[f] in test_units),
        }

    manifest_path = Path(cfg["paths"]["split_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_manifest(path: str | Path) -> dict[str, list[str]]:
    """Load a previously written split manifest."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
