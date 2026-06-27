"""PolypSegPredictor: frozen predict() contract over the exported ONNX model.

Implements `BasePredictor` (src/common/predictor.py). `explanation` (GradCAM) is left
unpopulated for now — wiring it into predict() is a Stage 9 addition, kept separate so
the latency-benchmarked inference path (Stage 8) isn't paying for a backward pass it
doesn't need.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.common.result import InferenceResult
from src.utils.viz import overlay_mask

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PolypSegPredictor:
    """Wraps an ONNX Runtime session; `predict()` takes an HxWx3 RGB uint8 image and
    returns a fully populated `InferenceResult` (mask, confidence_map, overlay,
    report, latency_ms).
    """

    def __init__(
        self,
        onnx_path: str | Path,
        image_size: int = 352,
        threshold: float = 0.5,
    ) -> None:
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.image_size = image_size
        self.threshold = threshold

    def warmup(self) -> None:
        """Optional pre-run so the first timed `predict()` call isn't penalised."""
        dummy = np.zeros((1, 3, self.image_size, self.image_size), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})

    def predict(self, image: np.ndarray) -> InferenceResult:
        start = time.perf_counter()
        original_h, original_w = image.shape[:2]

        resized = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        normalized = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        input_tensor = normalized.transpose(2, 0, 1)[None].astype(np.float32)

        logits = self.session.run(None, {self.input_name: input_tensor})[0]
        probs_small = 1.0 / (1.0 + np.exp(-logits[0, 0]))
        mask_small = (probs_small > self.threshold).astype(np.uint8)

        mask = cv2.resize(mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        confidence_map = cv2.resize(
            probs_small, (original_w, original_h), interpolation=cv2.INTER_LINEAR
        )

        has_polyp = bool(mask.any())
        coverage = float(mask.mean())
        confidence = float(confidence_map[mask.astype(bool)].mean()) if has_polyp else 0.0
        report = (
            f"Polyp detected, area {coverage * 100:.1f}% of frame, mean confidence {confidence:.2f}"
            if has_polyp
            else "No polyp detected."
        )

        latency_ms = (time.perf_counter() - start) * 1000.0

        return InferenceResult(
            task="polyp_segmentation",
            modality="endoscopy_image",
            prediction_type="mask",
            mask=mask,
            confidence=confidence,
            confidence_map=confidence_map,
            overlay=overlay_mask(image, mask),
            report=report,
            latency_ms=latency_ms,
            meta={"image_size": self.image_size, "threshold": self.threshold},
        )
