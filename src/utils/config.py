"""Config loading: YAML parsing, deep-merge, and minimal schema validation.

`configs/default.yaml` is the single source of truth. Override files such as
`configs/smoke.yaml` contain only the handful of keys that differ and are deep-merged
on top of the default config by `load_config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REQUIRED_SECTIONS = (
    "project",
    "seed",
    "paths",
    "data",
    "augment",
    "model",
    "loss",
    "train",
    "eval",
    "export",
    "logging",
)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base`, returning a new dict.

    Nested dicts are merged key-by-key; any other value type in `override` replaces
    the corresponding value in `base` outright.
    """
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(value, dict) and isinstance(base_value, dict):
            merged[key] = deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def _validate_config(cfg: dict[str, Any]) -> None:
    """Raise ValueError if any required top-level section is missing."""
    missing = [section for section in _REQUIRED_SECTIONS if section not in cfg]
    if missing:
        raise ValueError(f"Config is missing required section(s): {missing}")


def load_config(
    path: str | Path,
    base_path: str | Path = "configs/default.yaml",
) -> dict[str, Any]:
    """Load a YAML config, deep-merged onto the base default config.

    If `path` resolves to the same file as `base_path`, the base config is returned
    directly. Otherwise `base_path` is loaded first and `path`'s keys are deep-merged
    on top of it — this is how an override file like `smoke.yaml`, which only contains
    the keys that change, produces a complete, validated config.
    """
    base_path = Path(base_path)
    path = Path(path)

    with open(base_path, encoding="utf-8") as f:
        base_cfg: dict[str, Any] = yaml.safe_load(f) or {}

    if path.resolve() == base_path.resolve():
        cfg = base_cfg
    else:
        with open(path, encoding="utf-8") as f:
            override_cfg: dict[str, Any] = yaml.safe_load(f) or {}
        cfg = deep_merge(base_cfg, override_cfg)

    _validate_config(cfg)
    return cfg
