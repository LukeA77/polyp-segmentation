"""Segmentation GradCAM via pytorch-grad-cam's SemanticSegmentationTarget, targeting
the last encoder stage.

Targets `model.encoder.layer4[-1]` — the last residual block of a resnet-family smp
encoder. This is specific to the resnet encoders this project configures; a different
encoder family (e.g. an efficientnet) would need a different target-layer path.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def generate_gradcam_overlay(model: nn.Module, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render a GradCAM heatmap overlay (HxWx3 uint8) over the predicted polyp region.

    `image` is an HxWx3 uint8 RGB array already resized to the model's input size;
    `mask` is the corresponding HxW binary array (0/1) used to weight the target
    region (typically the model's own predicted mask).
    """
    model = model.eval()
    image_float = image.astype(np.float32) / 255.0

    normalized = (image_float - _IMAGENET_MEAN) / _IMAGENET_STD
    input_tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()

    target_layers = [model.encoder.layer4[-1]]
    targets = [SemanticSegmentationTarget(0, mask.astype(np.float32))]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    return show_cam_on_image(image_float, grayscale_cam, use_rgb=True)
