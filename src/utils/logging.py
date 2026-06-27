"""Project-wide logger factory. `src/` modules log through this, never bare `print()`."""

from __future__ import annotations

import logging
import sys

_CONFIGURED: set[str] = set()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured `logging.Logger` for `name`, idempotent across repeat calls."""
    logger = logging.getLogger(name)
    if name not in _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED.add(name)
    logger.setLevel(level)
    return logger
