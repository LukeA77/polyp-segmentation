"""DiceBCELoss: weighted sum of soft Dice loss and BCE-with-logits."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

_REDUCE_DIMS = (1, 2, 3)  # sum over C,H,W per sample; mean over the batch


class DiceBCELoss(nn.Module):
    """`dice_weight * (1 - soft_dice) + bce_weight * bce`, computed from raw logits."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.dice_weight = cfg["loss"]["dice_weight"]
        self.bce_weight = cfg["loss"]["bce_weight"]
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=_REDUCE_DIMS)
        union = probs.sum(dim=_REDUCE_DIMS) + targets.sum(dim=_REDUCE_DIMS)
        dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (union + 1e-6)).mean()

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
