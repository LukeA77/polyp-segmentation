"""KvasirSEGDataset: paired image/mask loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.splits import load_manifest
from src.data.transforms import build_train_transforms, build_val_transforms

_SPLITS = ("train", "val", "test")


class KvasirSEGDataset(Dataset):
    """Returns (image: float32 CxHxW, mask: float32 1xHxW in {0,1})."""

    def __init__(self, cfg: dict[str, Any], split: str) -> None:
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")

        data_root = Path(cfg["paths"]["data_root"])
        self.images_dir = data_root / "images"
        self.masks_dir = data_root / "masks"

        manifest = load_manifest(cfg["paths"]["split_manifest"])
        self.filenames = manifest[split]
        self.transform = (
            build_train_transforms(cfg) if split == "train" else build_val_transforms(cfg)
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[index]

        image = cv2.imread(str(self.images_dir / filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Kvasir masks are JPEG and may have near-but-not-exactly {0,255} values from
        # compression artifacts; binarise before any resizing/augmentation.
        raw_mask = cv2.imread(str(self.masks_dir / filename), cv2.IMREAD_GRAYSCALE)
        mask = (raw_mask > 127).astype(np.uint8) * 255

        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"].unsqueeze(0).float() / 255.0

        return image_tensor, mask_tensor
