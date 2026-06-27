"""Mandatory pre-training smoke test: exercises every stage of the pipeline once
(split -> load -> train 2 epochs -> checkpoint round-trip -> eval -> ONNX export ->
parity -> gradcam -> predict) on a 16-image subset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data.splits import build_splits
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.explain.gradcam import generate_gradcam_overlay
from src.export.onnx_export import check_parity, export_onnx
from src.inference.predictor import PolypSegPredictor
from src.models.build import build_model
from src.utils.config import load_config
from src.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    output_root = Path(cfg["paths"]["output_root"])
    image_size = cfg["data"]["image_size"]

    print("[1/8] Building splits...")
    manifest = build_splits(cfg)
    print(
        f"      train={len(manifest['train'])} val={len(manifest['val'])} "
        f"test={len(manifest['test'])}"
    )

    print("[2/8] Building model...")
    model = build_model(cfg)

    print("[3/8] Training (smoke)...")
    trainer = Trainer(cfg, model)
    fit_result = trainer.fit()
    print(f"      best_val_dice={fit_result['best_val_dice']:.4f}")

    print("[4/8] Checkpoint round-trip...")
    reloaded_model = build_model(cfg)
    epoch = Trainer.load_checkpoint(fit_result["best_ckpt_path"], reloaded_model)
    print(f"      reloaded checkpoint from epoch {epoch}")

    print("[5/8] Evaluating...")
    eval_result = Evaluator(cfg, reloaded_model).evaluate()
    print(f"      test_dice={eval_result['dice']:.4f} test_iou={eval_result['iou']:.4f}")

    print("[6/8] Exporting ONNX + parity check...")
    onnx_path = output_root / "model.onnx"
    export_onnx(reloaded_model, cfg, onnx_path)
    parity_result = check_parity(reloaded_model, onnx_path, cfg)
    if not parity_result["parity_ok"]:
        raise RuntimeError(f"ONNX parity check failed: {parity_result}")
    print(f"      parity OK (max|delta|={parity_result['max_abs_diff']:.2e})")

    print("[7/8] GradCAM overlay...")
    images_dir = Path(cfg["paths"]["data_root"]) / "images"
    masks_dir = Path(cfg["paths"]["data_root"]) / "masks"
    sample_filename = manifest["test"][0]

    sample_image = cv2.cvtColor(cv2.imread(str(images_dir / sample_filename)), cv2.COLOR_BGR2RGB)
    sample_image = cv2.resize(sample_image, (image_size, image_size))
    sample_mask = cv2.imread(str(masks_dir / sample_filename), cv2.IMREAD_GRAYSCALE)
    sample_mask = cv2.resize(sample_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    sample_mask = (sample_mask > 127).astype(np.uint8)

    gradcam_overlay = generate_gradcam_overlay(reloaded_model, sample_image, sample_mask)
    gradcam_path = output_root / "smoke_gradcam.png"
    cv2.imwrite(str(gradcam_path), cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))
    print(f"      saved {gradcam_path}")

    print("[8/8] PolypSegPredictor.predict()...")
    predictor = PolypSegPredictor(onnx_path, image_size=image_size)
    predictor.warmup()
    raw_image = cv2.cvtColor(cv2.imread(str(images_dir / sample_filename)), cv2.COLOR_BGR2RGB)
    result = predictor.predict(raw_image)
    print(f"      {result.task} | {result.report} | latency={result.latency_ms:.1f}ms")

    print("\nSMOKE RUN PASSED.")


if __name__ == "__main__":
    main()
