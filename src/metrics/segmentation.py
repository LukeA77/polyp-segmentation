"""Core segmentation metrics: dice, iou, precision, recall.

All four take raw logits — sigmoid + threshold are applied internally — so callers
never have to remember to binarise first. Each returns the batch-mean of the
per-sample metric.
"""

from __future__ import annotations

import torch

_REDUCE_DIMS = (1, 2, 3)
_EPS = 1e-6


def _binarize(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).float()


def dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    preds = _binarize(logits, threshold)
    intersection = (preds * targets).sum(dim=_REDUCE_DIMS)
    union = preds.sum(dim=_REDUCE_DIMS) + targets.sum(dim=_REDUCE_DIMS)
    return ((2.0 * intersection + _EPS) / (union + _EPS)).mean()


def iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    preds = _binarize(logits, threshold)
    intersection = (preds * targets).sum(dim=_REDUCE_DIMS)
    union = preds.sum(dim=_REDUCE_DIMS) + targets.sum(dim=_REDUCE_DIMS) - intersection
    return ((intersection + _EPS) / (union + _EPS)).mean()


def precision(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    preds = _binarize(logits, threshold)
    true_positive = (preds * targets).sum(dim=_REDUCE_DIMS)
    predicted_positive = preds.sum(dim=_REDUCE_DIMS)
    return ((true_positive + _EPS) / (predicted_positive + _EPS)).mean()


def recall(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    preds = _binarize(logits, threshold)
    true_positive = (preds * targets).sum(dim=_REDUCE_DIMS)
    actual_positive = targets.sum(dim=_REDUCE_DIMS)
    return ((true_positive + _EPS) / (actual_positive + _EPS)).mean()
