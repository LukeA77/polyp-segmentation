"""Albumentations train/val pipelines, driven by the `augment:` config block.

Masks are resized nearest-neighbour, images bilinear; ImageNet normalisation on images
only (the `Normalize` defaults already are the ImageNet mean/std).
"""

from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_transform(cfg: dict[str, Any]) -> A.BasicTransform:
    mode = cfg["data"]["normalize"]
    if mode == "imagenet":
        return A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    if mode == "none":
        return A.NoOp()
    raise ValueError(f"Unknown data.normalize mode: {mode!r}")


def _resize_transform(image_size: int) -> A.BasicTransform:
    return A.Resize(
        image_size,
        image_size,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
    )


def build_train_transforms(cfg: dict[str, Any]) -> A.Compose:
    """Build the training augmentation pipeline from `cfg['augment']` + `cfg['data']`.

    Albumentations 2.x owns its own RNG, seeded via `Compose(seed=...)`, decoupled
    from `random`/`numpy.random` (so `seed_everything` alone does not make this
    deterministic) — passing `cfg['seed']` here is what makes two freshly-built
    pipelines draw the same augmentations.
    """
    aug = cfg["augment"]
    return A.Compose(
        [
            _resize_transform(cfg["data"]["image_size"]),
            A.HorizontalFlip(p=aug["hflip"]),
            A.VerticalFlip(p=aug["vflip"]),
            A.RandomRotate90(p=aug["rotate90"]),
            A.RandomBrightnessContrast(p=aug["brightness_contrast"]),
            A.ElasticTransform(p=aug["elastic"]),
            A.GridDistortion(p=aug["grid_distortion"]),
            A.CoarseDropout(p=aug["coarse_dropout"]),
            _normalize_transform(cfg),
            ToTensorV2(),
        ],
        seed=cfg["seed"],
    )


def build_val_transforms(cfg: dict[str, Any]) -> A.Compose:
    """Build the (no-augmentation) val/test resize+normalise pipeline."""
    return A.Compose(
        [
            _resize_transform(cfg["data"]["image_size"]),
            _normalize_transform(cfg),
            ToTensorV2(),
        ],
        seed=cfg["seed"],
    )
