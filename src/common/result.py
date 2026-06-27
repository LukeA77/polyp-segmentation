"""Unified inference result contract for every flagship sub-project.

Frozen interface — see POLYP_SEG_BUILD_SPEC.md section 4. Every one of the five
sub-projects returns this object, and the flagship router/demo depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InferenceResult:
    """Unified output for every model in the flagship. Fields not relevant to a
    given task stay None. The flagship demo renders from these fields alone.
    """

    task: str  # "polyp_segmentation"
    modality: str  # "endoscopy_image"
    prediction_type: str  # "mask" | "grade" | "labels" | "boxes"

    # --- payload (task fills the relevant ones) ---
    mask: np.ndarray | None = None  # HxW uint8 in {0,1}
    boxes: np.ndarray | None = None  # Nx4 xyxy (future tasks)
    labels: dict | None = None  # {class: score} (future tasks)
    grade: int | None = None  # ordinal grade (future tasks)

    # --- always populated ---
    confidence: float = 0.0  # scalar summary in [0,1]
    confidence_map: np.ndarray | None = None  # HxW float prob map (seg)
    overlay: np.ndarray | None = None  # HxWx3 uint8 annotated render
    explanation: np.ndarray | None = None  # HxWx3 uint8 GradCAM overlay
    report: str = ""  # short human-readable summary
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)  # model name, input size, threshold, etc.
