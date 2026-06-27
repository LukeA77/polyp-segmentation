"""Global determinism: seed every RNG the pipeline touches."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python's `random`, NumPy, and Torch (CPU + CUDA), and force deterministic cuDNN.

    Call once at the start of any script that trains, evaluates, or builds splits, so
    runs are reproducible across CPU/GPU and across machines.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
