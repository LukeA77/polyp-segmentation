"""Model factory: builds a segmentation_models_pytorch architecture from config."""

from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp
import torch.nn as nn

_ARCHS = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build `smp.Unet` / `smp.DeepLabV3Plus` (per `cfg['model']`) with a pretrained
    encoder and a single output channel (raw logits, no activation).
    """
    model_cfg = cfg["model"]
    arch = model_cfg["arch"]
    if arch not in _ARCHS:
        raise ValueError(f"Unknown model.arch: {arch!r} (expected one of {list(_ARCHS)})")

    return _ARCHS[arch](
        encoder_name=model_cfg["encoder"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
    )
