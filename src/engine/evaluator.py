"""Test-set evaluation: threshold sweep, boundary/calibration metrics.

Worst-K failure-gallery rendering and results.json/markdown writing are added on top
of this in Stage 6 (`scripts/05_evaluate.py`) — this module covers the metric
computation that the Stage-4 smoke run also exercises.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import KvasirSEGDataset
from src.metrics.boundary import boundary_iou, hd95
from src.metrics.calibration import expected_calibration_error
from src.metrics.segmentation import dice, iou, precision, recall
from src.utils.logging import get_logger

logger = get_logger(__name__)

_THRESHOLD_GRID = np.arange(0.3, 0.71, 0.05)


class Evaluator:
    """Sweeps the binarisation threshold on val, applies it to test, and reports
    Dice/mIoU/precision/recall (+ optional boundary/calibration metrics).
    """

    def __init__(self, cfg: dict[str, Any], model: nn.Module) -> None:
        self.cfg = cfg
        self.eval_cfg = cfg["eval"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

        loader_kwargs = {
            "num_workers": cfg["data"]["num_workers"],
            "pin_memory": cfg["data"]["pin_memory"] and self.device.type == "cuda",
        }
        batch_size = cfg["train"]["batch_size"]
        self.val_loader = DataLoader(
            KvasirSEGDataset(cfg, "val"), batch_size=batch_size, shuffle=False, **loader_kwargs
        )
        self.test_loader = DataLoader(
            KvasirSEGDataset(cfg, "test"), batch_size=batch_size, shuffle=False, **loader_kwargs
        )

    @torch.no_grad()
    def _collect_logits(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        all_logits, all_masks = [], []
        for images, masks in loader:
            logits = self.model(images.to(self.device))
            all_logits.append(logits.cpu())
            all_masks.append(masks)
        return torch.cat(all_logits), torch.cat(all_masks)

    def _select_threshold(self, logits: torch.Tensor, masks: torch.Tensor) -> float:
        configured = self.eval_cfg["threshold"]
        if configured != "auto":
            return float(configured)

        best_threshold, best_dice = 0.5, -1.0
        for t in _THRESHOLD_GRID:
            score = float(dice(logits, masks, threshold=float(t)))
            if score > best_dice:
                best_dice, best_threshold = score, float(t)
        return best_threshold

    def evaluate(self) -> dict[str, Any]:
        """Run the full evaluation and return the results dict written to results.json."""
        val_logits, val_masks = self._collect_logits(self.val_loader)
        threshold = self._select_threshold(val_logits, val_masks)

        test_logits, test_masks = self._collect_logits(self.test_loader)
        results: dict[str, Any] = {
            "threshold": threshold,
            "dice": float(dice(test_logits, test_masks, threshold=threshold)),
            "iou": float(iou(test_logits, test_masks, threshold=threshold)),
            "precision": float(precision(test_logits, test_masks, threshold=threshold)),
            "recall": float(recall(test_logits, test_masks, threshold=threshold)),
        }

        probs = torch.sigmoid(test_logits)
        preds = (probs > threshold).numpy().astype(np.uint8)
        gts = test_masks.numpy().astype(np.uint8)

        if self.eval_cfg.get("boundary_metrics"):
            hd95_scores = [hd95(preds[i, 0], gts[i, 0]) for i in range(len(preds))]
            biou_scores = [boundary_iou(preds[i, 0], gts[i, 0]) for i in range(len(preds))]
            results["hd95"] = float(np.nanmean(hd95_scores))
            results["boundary_iou"] = float(np.nanmean(biou_scores))

        if self.eval_cfg.get("calibration"):
            results["ece"] = expected_calibration_error(probs.numpy(), gts.astype(np.float32))

        logger.info("Evaluation results: %s", results)
        return results
