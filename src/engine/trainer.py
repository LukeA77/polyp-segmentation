"""Trainer: fit loop, AMP, checkpointing, early stopping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import KvasirSEGDataset
from src.losses.losses import DiceBCELoss
from src.metrics.segmentation import dice, iou
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Train/val loop with AMP, cosine/plateau LR, best-checkpoint-on-`monitor`, early
    stopping, and per-epoch checkpointing (Colab-disconnect insurance: every epoch's
    weights land on disk, not just the best one).
    """

    def __init__(self, cfg: dict[str, Any], model: nn.Module) -> None:
        self.cfg = cfg
        self.train_cfg = cfg["train"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        loader_kwargs = {
            "num_workers": cfg["data"]["num_workers"],
            "pin_memory": cfg["data"]["pin_memory"] and self.device.type == "cuda",
        }
        self.train_loader = DataLoader(
            KvasirSEGDataset(cfg, "train"),
            batch_size=self.train_cfg["batch_size"],
            shuffle=True,
            **loader_kwargs,
        )
        self.val_loader = DataLoader(
            KvasirSEGDataset(cfg, "val"),
            batch_size=self.train_cfg["batch_size"],
            shuffle=False,
            **loader_kwargs,
        )

        self.criterion = DiceBCELoss(cfg)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.amp_enabled = bool(self.train_cfg["amp"]) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.amp_enabled)

        self.ckpt_dir = Path(cfg["paths"]["output_root"]) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.last_ckpt_path = self.ckpt_dir / "last.ckpt"
        self.best_ckpt_path = self.ckpt_dir / "best.ckpt"

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.train_cfg["optimizer"]
        if name != "adamw":
            raise ValueError(f"Unsupported train.optimizer: {name!r}")
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg["lr"],
            weight_decay=self.train_cfg["weight_decay"],
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        name = self.train_cfg["scheduler"]
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.train_cfg["epochs"]
            )
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max")
        raise ValueError(f"Unsupported train.scheduler: {name!r}")

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train)
        totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0}

        for images, masks in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            with torch.set_grad_enabled(train):
                with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            totals["loss"] += loss.item()
            totals["dice"] += float(dice(logits.detach(), masks))
            totals["iou"] += float(iou(logits.detach(), masks))

        return {k: v / len(loader) for k, v in totals.items()}

    def fit(self) -> dict[str, Any]:
        """Run training to completion (or early stop); returns the best-epoch summary."""
        monitor = self.train_cfg["monitor"]
        patience = self.train_cfg["early_stop_patience"]
        best_score = -float("inf")
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, self.train_cfg["epochs"] + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["dice"])
            else:
                self.scheduler.step()

            logger.info(
                "epoch %d/%d train_loss=%.4f val_loss=%.4f val_dice=%.4f val_iou=%.4f",
                epoch,
                self.train_cfg["epochs"],
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["dice"],
                val_metrics["iou"],
            )
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            self.save_checkpoint(self.last_ckpt_path, epoch)

            score = val_metrics["dice"] if monitor == "val_dice" else -val_metrics["loss"]
            if score > best_score:
                best_score = score
                epochs_without_improvement = 0
                self.save_checkpoint(self.best_ckpt_path, epoch)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs).",
                        epoch,
                        patience,
                    )
                    break

        return {
            "best_val_dice": best_score,
            "history": history,
            "best_ckpt_path": str(self.best_ckpt_path),
            "last_ckpt_path": str(self.last_ckpt_path),
        }

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            path,
        )

    @staticmethod
    def load_checkpoint(path: str | Path, model: nn.Module, map_location: str | None = None) -> int:
        """Load a checkpoint's weights into `model` in place; returns the saved epoch."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint["epoch"]
