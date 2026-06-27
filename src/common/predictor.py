"""BasePredictor protocol every sub-project implements.

Frozen interface — see POLYP_SEG_BUILD_SPEC.md section 4. The flagship router calls
predict() without knowing which model it's talking to.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .result import InferenceResult


class BasePredictor(Protocol):
    """Every sub-project implements this. The flagship router calls predict()
    without knowing which model it is talking to.
    """

    def predict(self, image: np.ndarray) -> InferenceResult: ...

    def warmup(self) -> None: ...  # optional pre-run for fair latency
