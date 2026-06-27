"""ONNX export and PyTorch-vs-ONNX parity check.

CPU/GPU latency benchmarking (`benchmark_latency`) is added in Stage 8 — the
Stage-4 smoke run only needs export + parity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


def export_onnx(model: nn.Module, cfg: dict[str, Any], out_path: str | Path) -> Path:
    """Export `model` to ONNX (opset + dynamic batch axis per `cfg['export']`)."""
    export_cfg = cfg["export"]
    image_size = cfg["data"]["image_size"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval().cpu()
    dummy_input = torch.randn(1, cfg["model"]["in_channels"], image_size, image_size)
    dynamic_axes = (
        {"input": {0: "batch"}, "output": {0: "batch"}} if export_cfg["dynamic_batch"] else None
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=export_cfg["opset"],
        dynamic_axes=dynamic_axes,
    )
    return out_path


def check_parity(model: nn.Module, onnx_path: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Compare PyTorch vs ONNX Runtime outputs on a random batch.

    `dice_match` compares the binarised (logit > 0, i.e. prob > 0.5) masks rather than
    raw values, since that's the quantity that actually matters downstream.
    """
    export_cfg = cfg["export"]
    image_size = cfg["data"]["image_size"]

    model = model.eval().cpu()
    sample = torch.randn(2, cfg["model"]["in_channels"], image_size, image_size)

    with torch.no_grad():
        torch_logits = model(sample).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: sample.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
    dice_match = bool(np.array_equal(torch_logits > 0, onnx_logits > 0))

    result = {
        "max_abs_diff": max_abs_diff,
        "dice_match": dice_match,
        "parity_ok": max_abs_diff < export_cfg["parity_atol"] and dice_match,
    }
    logger.info("ONNX parity: %s", result)
    return result
