"""Regression suite mirroring the Stage-4 smoke run: shapes, loss-goes-down, seed
determinism, and ONNX parity. Implemented in Stage 4; inherited unchanged by Projects
2-5.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.data.dataset import KvasirSEGDataset
from src.data.splits import build_splits
from src.export.onnx_export import check_parity, export_onnx
from src.losses.losses import DiceBCELoss
from src.models.build import build_model
from src.utils.config import load_config
from src.utils.seeding import seed_everything

# Prefer the gitignored local-runtime override if present (this machine), otherwise
# fall back to the spec's real configs/smoke.yaml (hosted Colab + Drive) — so this
# exact `pytest tests/test_smoke.py -q` command works unmodified in both places.
_CONFIG_PATH = (
    "configs/local_smoke.yaml"
    if Path("configs/local_smoke.yaml").exists()
    else "configs/smoke.yaml"
)


def _load_cfg() -> dict:
    cfg = load_config(_CONFIG_PATH)
    seed_everything(cfg["seed"])
    build_splits(cfg)
    return cfg


def test_batch_shapes() -> None:
    cfg = _load_cfg()
    image, mask = KvasirSEGDataset(cfg, "train")[0]

    image_size = cfg["data"]["image_size"]
    assert image.shape == (3, image_size, image_size)
    assert image.dtype == torch.float32
    assert mask.shape == (1, image_size, image_size)
    assert mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_loss_decreases_on_fixed_batch() -> None:
    cfg = _load_cfg()
    seed_everything(cfg["seed"])

    model = build_model(cfg)
    dataset = KvasirSEGDataset(cfg, "train")
    images = torch.stack([dataset[i][0] for i in range(4)])
    masks = torch.stack([dataset[i][1] for i in range(4)])

    criterion = DiceBCELoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_seed_determinism() -> None:
    cfg = _load_cfg()

    seed_everything(cfg["seed"])
    image_a, mask_a = KvasirSEGDataset(cfg, "train")[0]

    seed_everything(cfg["seed"])
    image_b, mask_b = KvasirSEGDataset(cfg, "train")[0]

    assert torch.equal(image_a, image_b)
    assert torch.equal(mask_a, mask_b)


def test_onnx_parity(tmp_path: Path) -> None:
    cfg = _load_cfg()
    model = build_model(cfg).eval()

    onnx_path = tmp_path / "model.onnx"
    export_onnx(model, cfg, onnx_path)
    result = check_parity(model, onnx_path, cfg)

    assert result["max_abs_diff"] < cfg["export"]["parity_atol"]
    assert result["dice_match"]
