# Polyp Segmentation (Kvasir-SEG) — Build Spec for Claude Code

**Project 1 of the AI Diagnostic & Surgical Assistant flagship.**
Owner: Luke A. (`github.com/LukeA77`) · Environment: Google Colab Pro + Google Drive
Implementer: Claude Code (VS Code or Colab) · Status: ready to build

---

## 0. End goal (read this first)

Build a **modular, config-driven, production-standard polyp segmentation system** on the
Kvasir-SEG dataset that:

1. Trains a segmentation model (U-Net / DeepLabV3+ with a pretrained encoder) to **Dice ≥ 0.85,
   mIoU ≥ 0.78** on a held-out test set.
2. Exports to **ONNX with verified parity** and reports **CPU vs GPU inference latency** (the same
   credibility move as the MOT project's 1.84× speedup).
3. Produces **GradCAM explainability overlays**, **confidence calibration** (reliability diagram +
   ECE), and an honest **failure-case gallery**.
4. Exposes a single stable `predict(image) -> InferenceResult` contract.

That last point is the one that matters most for the bigger picture. This is **Project 1 of 5**.
Projects 2–5 (DR grading, chest X-ray, tumour 3D seg, surgical video) and the final flagship
wrapper will all consume the **same `InferenceResult` object and the same repo skeleton** defined
here. So this repo is not just "a polyp segmenter" — it is the **reference template** every later
component is cloned from, and the inference contract the flagship router will call. Build it as if
four more modules depend on these exact interfaces, because they do.

**Definition of done:** a clean standalone repo (`polyp-segmentation`) with a top-level README, a
benchmark table, ONNX weights, reproducible configs, and a 20-second demo GIF — that another
engineer could clone, run end-to-end on Colab, and reproduce the headline metrics.

---

## 1. How to use this spec (workflow — important)

**Do not write the whole project in one pass.** Build it in the **10 stages** in §6. Each stage has:

- an **Objective** (what to produce),
- **Deliverables** (the exact files/functions),
- **Verification gate** (a concrete check that must pass).

**At the end of every stage: STOP.** Run the verification gate, show Luke the output, and wait for a
"continue" before starting the next stage. Do **not** proceed past a failing gate. The single most
expensive failure mode we are avoiding is writing 2,000 lines across the whole pipeline and only
discovering at training time that the masks were misaligned or the dataloader was returning the
wrong shapes. We catch that at Stage 2, not Stage 5.

**Stage 4 is a mandatory smoke run** (full pipeline, tiny data, 2 epochs) that must go green before
any real training is attempted in Stage 5.

When in doubt, prefer the smaller, verifiable step. Ask before introducing a dependency or design
choice not specified here.

---

## 2. Environment & infrastructure

| Concern | Decision |
|---|---|
| Runtime | Google Colab Pro (T4/L4 is ample — Kvasir-SEG is tiny) |
| Code | Lives in a **GitHub repo** `LukeA77/polyp-segmentation`, cloned fresh each Colab session |
| Data + outputs | Live on **Google Drive**, mounted at `/content/drive/MyDrive/` |
| Dataset on Drive | `/content/drive/MyDrive/datasets/kvasir-seg/` |
| Checkpoints/logs/exports | `/content/drive/MyDrive/polyp-seg-outputs/` (survives runtime disconnects) |

**Why this split:** code in git (versioned, diffable, editable in VS Code), heavy/persistent
artifacts on Drive (survive Colab's ephemeral disk). The dataset is only ~46 MB so it sits on Drive
comfortably; copy it to local `/content/` disk at session start if you want faster I/O (optional —
1,000 images is not an I/O bottleneck).

**All filesystem paths come from the config, never hardcoded.** A `paths:` block in the config maps
`data_root`, `output_root`, etc. so the same code runs unchanged on Colab, the RTX 5090 server, or a
laptop. This matters: Projects 4 and 5 will run on the 5090 server, so path-portability is built in
from day one.

A `notebooks/colab_runner.ipynb` is the only thing run interactively. It does:
`mount Drive → clone/pull repo → pip install -r requirements.txt → run the stage script`. All real
logic lives in `src/` and `scripts/`, never in notebook cells.

---

## 3. Target repository structure

Build toward exactly this layout. Folders marked **[shared]** are written so they can later be
lifted, unchanged, into the flagship's common package — keep them free of polyp-specific assumptions.

```
polyp-segmentation/
├── configs/
│   ├── default.yaml              # full config (single source of truth)
│   └── smoke.yaml                # overrides for the Stage-4 smoke run
├── src/
│   ├── common/                   # [shared] flagship-facing contract — keep generic
│   │   ├── result.py             #   InferenceResult dataclass (see §4)
│   │   └── predictor.py          #   BasePredictor protocol (see §4)
│   ├── data/
│   │   ├── dataset.py            #   KvasirSEGDataset
│   │   ├── transforms.py        #   albumentations train/val pipelines
│   │   └── splits.py            #   deterministic split + manifest I/O
│   ├── models/
│   │   └── build.py             #   build_model(cfg) -> nn.Module (smp factory)
│   ├── losses/
│   │   └── losses.py           #   DiceBCELoss
│   ├── metrics/
│   │   ├── segmentation.py     #   dice, iou, precision, recall
│   │   ├── boundary.py         #   HD95 / boundary IoU
│   │   └── calibration.py      #   ECE + reliability bins
│   ├── engine/
│   │   ├── trainer.py          #   Trainer: fit loop, AMP, ckpt, early stop
│   │   └── evaluator.py        #   test eval, threshold sweep, error analysis
│   ├── explain/
│   │   └── gradcam.py          #   segmentation GradCAM overlays
│   ├── export/
│   │   └── onnx_export.py      #   export + parity check + latency benchmark
│   ├── inference/
│   │   └── predictor.py        #   PolypSegPredictor implements BasePredictor
│   └── utils/
│       ├── config.py           #   load + deep-merge + validate config
│       ├── seeding.py          #   seed_everything(seed)
│       ├── logging.py          #   get_logger()
│       └── viz.py              #   overlay rendering, figure helpers
├── scripts/                      # thin CLI wrappers; argparse → src/ calls
│   ├── 00_download_data.py
│   ├── 01_inspect_data.py
│   ├── 02_build_splits.py
│   ├── 03_smoke_run.py
│   ├── 04_train.py
│   ├── 05_evaluate.py
│   ├── 06_explain.py
│   └── 07_export_onnx.py
├── tests/
│   └── test_smoke.py            # pytest: shapes, determinism, ONNX parity
├── notebooks/
│   └── colab_runner.ipynb
├── outputs/                      # gitignored; real artifacts go to Drive
├── requirements.txt
├── .gitignore
├── pyproject.toml               # ruff + black config, package metadata
└── README.md
```

**Design rules that make this reusable (enforce these):**

- `scripts/` files contain **no logic** — they parse args, load config, and call into `src/`. Every
  script is a 10–30 line wrapper. This is what lets Projects 2–5 reuse the same CLI surface.
- Anything polyp-specific (mask handling, single-channel output, endoscopy normalisation) lives in
  `src/data/` and `src/inference/predictor.py`. Everything in `src/engine/`, `src/export/`,
  `src/common/`, `src/utils/` must be **task-agnostic** so it is copy-paste reusable.
- No global state. Config is passed explicitly. No magic constants outside the config.

---

## 4. The flagship contract (build this in Stage 1, freeze it early)

This is the most important interface in the whole portfolio. Every one of the five sub-projects will
return this object, and the flagship router/demo will depend on it. Define it once, here, and keep it
generic enough to describe a mask **or** a class grade **or** bounding boxes.

`src/common/result.py`:

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class InferenceResult:
    """Unified output for every model in the flagship. Fields not relevant to a
    given task stay None. The flagship demo renders from these fields alone."""
    task: str                                   # "polyp_segmentation"
    modality: str                               # "endoscopy_image"
    prediction_type: str                        # "mask" | "grade" | "labels" | "boxes"

    # --- payload (task fills the relevant ones) ---
    mask: Optional[np.ndarray] = None           # HxW uint8 in {0,1}
    boxes: Optional[np.ndarray] = None          # Nx4 xyxy (future tasks)
    labels: Optional[dict] = None               # {class: score} (future tasks)
    grade: Optional[int] = None                 # ordinal grade (future tasks)

    # --- always populated ---
    confidence: float = 0.0                     # scalar summary in [0,1]
    confidence_map: Optional[np.ndarray] = None # HxW float prob map (seg)
    overlay: Optional[np.ndarray] = None        # HxWx3 uint8 annotated render
    explanation: Optional[np.ndarray] = None    # HxWx3 uint8 GradCAM overlay
    report: str = ""                            # short human-readable summary
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)    # model name, input size, threshold, etc.
```

`src/common/predictor.py`:

```python
from typing import Protocol
import numpy as np
from .result import InferenceResult

class BasePredictor(Protocol):
    """Every sub-project implements this. The flagship router calls predict()
    without knowing which model it is talking to."""
    def predict(self, image: np.ndarray) -> InferenceResult: ...
    def warmup(self) -> None: ...               # optional pre-run for fair latency
```

`PolypSegPredictor` (in `src/inference/predictor.py`) implements `BasePredictor`, wraps the ONNX
session, and takes a raw `HxWx3` RGB uint8 image → returns a fully populated `InferenceResult`
(`mask`, `confidence_map`, `overlay`, `report`, `latency_ms`). **Treat this signature as frozen** —
later modules and the flagship are written against it.

---

## 5. Config schema (single source of truth)

`configs/default.yaml` — everything tunable lives here. No hyperparameter or path is hardcoded in
code. `smoke.yaml` only overrides the handful of keys needed for the smoke run.

```yaml
project: polyp-segmentation
seed: 42

paths:
  data_root: /content/drive/MyDrive/datasets/kvasir-seg
  output_root: /content/drive/MyDrive/polyp-seg-outputs
  split_manifest: configs/splits.json      # committed to git for reproducibility

data:
  image_size: 352          # PraNet-era standard for polyp seg
  val_fraction: 0.10
  test_fraction: 0.10
  num_workers: 2
  pin_memory: true
  normalize: imagenet      # mean/std for the pretrained encoder

augment:                   # albumentations, train only
  hflip: 0.5
  vflip: 0.5
  rotate90: 0.5
  brightness_contrast: 0.2
  elastic: 0.2             # endoscopy frames deform well under elastic/grid distort
  grid_distortion: 0.2
  coarse_dropout: 0.2

model:
  arch: unet               # unet | deeplabv3plus | (segformer = optional later)
  encoder: resnet34        # timm/smp encoder name
  encoder_weights: imagenet
  in_channels: 3
  classes: 1               # binary -> single logit channel

loss:
  name: dice_bce
  dice_weight: 0.5
  bce_weight: 0.5

train:
  epochs: 80
  batch_size: 16
  optimizer: adamw
  lr: 3.0e-4
  weight_decay: 1.0e-4
  scheduler: cosine        # cosine | plateau
  amp: true                # mixed precision
  early_stop_patience: 15  # on val Dice
  monitor: val_dice

eval:
  threshold: auto          # 'auto' = sweep on val; or a float
  tta: false               # flip-TTA toggle
  boundary_metrics: true   # HD95 / boundary IoU
  calibration: true        # ECE + reliability diagram

export:
  opset: 17
  dynamic_batch: true
  parity_atol: 1.0e-4
  benchmark_runs: 50

logging:
  backend: wandb           # wandb | none ; falls back to offline if no key
  project: polyp-segmentation
```

---

## 6. The staged build plan

### Stage 0 — Scaffolding, config, utilities
**Objective:** repo skeleton that imports cleanly and parses the config. No ML yet.
**Deliverables:** the full folder tree (empty modules with docstrings + signatures), `requirements.txt`,
`pyproject.toml` (ruff + black), `.gitignore`, `src/utils/{config,seeding,logging}.py` fully
implemented, `configs/default.yaml` and `configs/smoke.yaml`.
**Verification gate:**
```bash
python -c "from src.utils.config import load_config; c=load_config('configs/default.yaml'); print(c['model']['arch'], c['seed'])"
# -> prints: unet 42      (config loads, deep-merge with smoke.yaml works)
ruff check src/            # passes clean
```
**STOP.**

---

### Stage 1 — The flagship contract + data download
**Objective:** freeze the shared interfaces; get the data onto Drive and verified.
**Deliverables:**
- `src/common/result.py` and `src/common/predictor.py` exactly as in §4.
- `scripts/00_download_data.py`: downloads Kvasir-SEG into `paths.data_root`. **Strategy:** try the
  Kaggle API first (`debeshjha1/kvasirseg` — Luke uses Kaggle, credentials via `kaggle.json`); fall
  back to the official Simula zip (`https://datasets.simula.no/kvasir-seg/` — resolve the current
  download link; do not hardcode a guessed URL, fetch the page and follow the download). Extract, then
  **verify integrity**: assert exactly 1000 images in `images/` and 1000 masks in `masks/`, matching
  filenames, no corrupt reads. Idempotent (skip if already present and verified).
**Verification gate:**
```bash
python scripts/00_download_data.py --config configs/default.yaml
# -> "Verified: 1000 images, 1000 masks, filenames matched, 0 unreadable."
python -c "from src.common.result import InferenceResult; print(InferenceResult(task='t',modality='m',prediction_type='mask'))"
```
**STOP.**

---

### Stage 2 — Data pipeline (dataset, transforms, splits)
**Objective:** a dataloader that returns correctly shaped, correctly aligned image/mask tensors.
This is the highest-risk correctness stage — give it real scrutiny.
**Deliverables:**
- `src/data/splits.py`: deterministic image-level split (seeded) into train/val/test using the
  fractions in config; **writes a split manifest (`configs/splits.json`) and commits it** so the exact
  split is reproducible forever. *Honest note to surface in the README:* Kvasir-SEG ships no patient
  IDs, so a true patient-level split is impossible here; we use a fixed-seed image-level split and say
  so. (Projects 3–5 *do* have case/patient grouping and will reuse `splits.py`'s manifest mechanism
  with grouped splitting — design the function signature to accept an optional `groups` argument now.)
- `src/data/transforms.py`: albumentations pipelines from the `augment:` config. **Masks resized with
  nearest-neighbour; images with bilinear.** ImageNet normalisation on images only.
- `src/data/dataset.py`: `KvasirSEGDataset` returning `(image: float32 CxHxW, mask: float32 1xHxW in
  {0,1})`.
- `scripts/01_inspect_data.py`: prints split sizes, image-size distribution, mask area / polyp-coverage
  histogram; saves a 3×3 grid of `image + mask overlay` to outputs.
- `scripts/02_build_splits.py`: generates and freezes the manifest.
**Verification gate:**
```bash
python scripts/02_build_splits.py --config configs/default.yaml   # -> 800/100/100, manifest written
python scripts/01_inspect_data.py --config configs/default.yaml    # -> stats + saved overlay grid
# Then assert a batch is correct:
#   images (B,3,352,352) float, range ~normalised; masks (B,1,352,352) in {0,1};
#   overlay grid visually shows mask aligned on the polyp (Luke eyeballs this).
```
The overlay grid is the human check: **the mask must sit on the polyp.** If it's offset or inverted,
stop and fix before anything else.
**STOP.**

---

### Stage 3 — Model, loss, metrics
**Objective:** the learnable pieces, each unit-checked on dummy tensors. No training yet.
**Deliverables:**
- `src/models/build.py`: `build_model(cfg)` → `smp.Unet`/`smp.DeepLabV3Plus` per config, 1 output
  channel, ImageNet-pretrained encoder.
- `src/losses/losses.py`: `DiceBCELoss` (weighted sum, logits in).
- `src/metrics/segmentation.py`: `dice`, `iou`, `precision`, `recall` (threshold-aware, work on
  logits or probs consistently — document which).
- `src/metrics/boundary.py`: `hd95`, `boundary_iou`.
- `src/metrics/calibration.py`: `expected_calibration_error`, `reliability_bins`.
**Verification gate:**
```bash
python -c "
import torch; from src.utils.config import load_config; from src.models.build import build_model
from src.losses.losses import DiceBCELoss; from src.metrics.segmentation import dice
c=load_config('configs/default.yaml'); m=build_model(c).eval()
x=torch.randn(2,3,352,352); y=(torch.rand(2,1,352,352)>0.5).float()
logits=m(x); print('out', tuple(logits.shape))          # -> (2,1,352,352)
print('loss', float(DiceBCELoss(c)(logits,y)))           # finite scalar
print('dice', float(dice(logits,y)))                     # in [0,1]
"
```
**STOP.**

---

### Stage 4 — SMOKE RUN (mandatory gate before real training)
**Objective:** prove the **entire** pipeline runs end-to-end on a tiny slice, fast, with no crashes —
*before* committing to an 80-epoch run. This is the safety net.
**Deliverables:**
- `scripts/03_smoke_run.py` driven by `configs/smoke.yaml`, which overrides: `data.subset: 16`
  (8 train / 4 val / 4 test), `train.epochs: 2`, `train.batch_size: 4`, `logging.backend: none`,
  tiny image size optional.
- It must exercise **every** downstream component once: build splits → dataset → model → train 2
  epochs (AMP on) → checkpoint write+reload → evaluate → ONNX export → parity check → one GradCAM
  overlay → one `PolypSegPredictor.predict()` call returning a populated `InferenceResult`.
- `tests/test_smoke.py`: a `pytest` version asserting (a) batch shapes, (b) loss decreases over 2
  steps on a fixed batch, (c) seeding reproduces identical first-batch tensors, (d) ONNX parity within
  `atol`. This converts the smoke run into a **reusable regression test** that Projects 2–5 inherit.
**Verification gate:**
```bash
python scripts/03_smoke_run.py --config configs/smoke.yaml   # completes in 1–2 min, no errors
pytest tests/test_smoke.py -q                                # all green
```
Smoke metrics will be garbage (4 images) — **that's fine**. We are checking *plumbing*, not accuracy.
Green here = the full Stage-5 run is safe to launch.
**STOP.**

---

### Stage 5 — Full training
**Objective:** train the real model to target and log everything.
**Deliverables:**
- `src/engine/trainer.py`: `Trainer.fit()` with AMP, cosine/plateau LR, gradient-safe loop, per-epoch
  train/val Dice+IoU, **best-checkpoint on `val_dice`**, early stopping, W&B logging (offline fallback
  if no API key), resumable from checkpoint (Colab disconnect insurance — save every epoch to Drive).
- `scripts/04_train.py`.
**Verification gate:**
```bash
python scripts/04_train.py --config configs/default.yaml
# -> training curves logged; best val Dice printed; best.ckpt on Drive.
# Target signal: val Dice climbing past ~0.80 and still improving. If it stalls < 0.75,
# stop and diagnose (LR, aug strength, threshold) rather than burning all 80 epochs.
```
**STOP.**

---

### Stage 6 — Evaluation, threshold tuning, error analysis
**Objective:** honest held-out numbers + where the model fails.
**Deliverables:**
- `src/engine/evaluator.py`: loads best checkpoint, **sweeps the binarisation threshold on val** (don't
  assume 0.5), applies the chosen threshold to **test**, reports Dice, mIoU, precision, recall, and (if
  enabled) HD95 / boundary IoU. Optional flip-TTA. Saves a **worst-K failure gallery** (lowest-Dice
  test cases with image / GT / pred / error map).
- `scripts/05_evaluate.py` → writes `results.json` + a markdown results table.
**Verification gate:**
```bash
python scripts/05_evaluate.py --config configs/default.yaml
# -> test Dice ≥ 0.85, mIoU ≥ 0.78 (targets). results.json + failure gallery saved.
```
If targets are missed, report the real numbers and the top suspected causes — **do not inflate.**
Calibrated honesty is the explicit standard for this portfolio.
**STOP.**

---

### Stage 7 — Explainability + calibration
**Objective:** show the model attends to clinically plausible regions, and quantify confidence honesty.
**Deliverables:**
- `src/explain/gradcam.py`: segmentation GradCAM via `pytorch-grad-cam`'s `SemanticSegmentationTarget`,
  targeting the last encoder stage; renders overlay heatmaps for a sample of test cases (include some
  failures from Stage 6).
- Calibration: pixel-level **ECE + reliability diagram** saved as a figure; optional **temperature
  scaling** fit on val with before/after ECE reported. (Calibration is rare in portfolios and is called
  out in the plan as a standout — make it a clean, labelled figure.)
- `scripts/06_explain.py`.
**Verification gate:**
```bash
python scripts/06_explain.py --config configs/default.yaml
# -> GradCAM overlays + reliability diagram + ECE (pre/post temp-scaling) saved to outputs.
# Human check: heatmaps concentrate on polyp tissue, not borders/specular glare.
```
**STOP.**

---

### Stage 8 — ONNX export, parity, latency benchmark
**Objective:** deployable artifact + the headline speed table (mirrors the MOT 1.84× framing).
**Deliverables:**
- `src/export/onnx_export.py`: export best model (opset 17, dynamic batch axis), **parity check**
  (max abs logit diff < `parity_atol` AND Dice agreement PyTorch-vs-ONNX on a test batch), then
  **benchmark CPU vs GPU** latency over N runs (warmup excluded) → mean / p50 / p95 and the **CPU→GPU
  and PyTorch→ONNX speedup factors**.
- `scripts/07_export_onnx.py` → writes `model.onnx` to Drive + a `latency.md` table.
**Verification gate:**
```bash
python scripts/07_export_onnx.py --config configs/default.yaml
# -> "Parity OK (max|Δ|=…, Dice match=…)"; latency table CPU vs GPU printed + saved.
```
**STOP.**

---

### Stage 9 — Inference contract, packaging, README
**Objective:** the frozen `predict()` path + a repo that reads as engineering maturity.
**Deliverables:**
- `src/inference/predictor.py`: `PolypSegPredictor` implementing `BasePredictor`, wrapping the ONNX
  session. `predict(rgb_uint8)` → full `InferenceResult` (mask, confidence_map, overlay, GradCAM
  explanation, a short `report` string e.g. *"Polyp detected, area 4.1% of frame, mean confidence
  0.91"*, and `latency_ms`).
- A 20-second **demo GIF**: a few images in → overlay + heatmap out.
- `README.md`: one-paragraph overview, architecture diagram, results table (auto-pulled from
  `results.json`), latency table, repro instructions (the exact Colab steps), honest limitations
  (image-level split, single-dataset), and a **"role in the flagship"** note explaining the
  `InferenceResult` contract. A short **model card** (data, intended use, metrics, limitations).
- Final `colab_runner.ipynb` that reproduces the whole thing from a clean runtime.
**Verification gate:**
```bash
python -c "
import numpy as np, cv2
from src.inference.predictor import PolypSegPredictor
p=PolypSegPredictor('…/model.onnx'); r=p.predict(cv2.cvtColor(cv2.imread('sample.jpg'),cv2.COLOR_BGR2RGB))
print(r.task, r.prediction_type, r.confidence, r.mask.shape, r.report)
"
# -> populated InferenceResult; overlay + explanation arrays present.
```
**DONE.** Tag `v1.0`, push, pin the repo.

---

## 7. Smoke-run spec (detail)

The smoke run (Stage 4) is the contract that protects every later stage. Requirements:

- Runs from `configs/smoke.yaml` — **same code paths** as the real run, only the data subset and epoch
  count change. (If the smoke run uses a different code path, it's worthless.)
- Touches **every** component exactly once: split → load → train → checkpoint round-trip → eval →
  export → parity → gradcam → predict.
- Finishes in **≤ 2 minutes** on a T4.
- Asserts plumbing, not accuracy: shapes, dtypes, finite losses, loss-goes-down-on-fixed-batch,
  seed-determinism, ONNX parity.
- Mirrored as `pytest tests/test_smoke.py` so it becomes CI-able and is **inherited by Projects 2–5**.

---

## 8. Targets (definition of done for the metrics)

| Metric | Target | Source |
|---|---|---|
| Test Dice | ≥ 0.85 | plan §3 |
| Test mIoU | ≥ 0.78 | plan §3 |
| ONNX parity | max|Δ logit| < 1e-4 & Dice match | this spec |
| Latency table | CPU + GPU, mean/p50/p95 + speedup | mirrors MOT |
| Calibration | ECE reported, reliability diagram | standout |

Report real numbers. If short of target, say so and explain — that's the standard.

---

## 9. Improvements I'm adding (beyond the base plan) and why

These are deliberately included above; here's the rationale so you can keep or drop each:

1. **Frozen `InferenceResult` + `BasePredictor` contract, built in Stage 1.** The biggest
   engineering-maturity signal in the whole flagship. Defining it now (not bolting it on at
   integration) is what makes five modules "snap together" later instead of needing a rewrite.
2. **Committed split manifest (`splits.json`).** Exact reproducibility, and an honest, explicit
   statement about the image-level-vs-patient-level limitation rather than hand-waving it.
3. **Confidence calibration (ECE + reliability diagram + optional temperature scaling).** Called out
   in the plan as rare-in-portfolios; this makes it concrete for segmentation (pixel-level).
4. **Boundary-aware metrics (HD95 / boundary IoU).** Clinically meaningful for polyp margins and
   distinguishes the work from a generic Dice-only notebook.
5. **Threshold sweep on val** instead of assuming 0.5 — a real, cheap metric gain and good practice.
6. **Worst-K failure gallery.** Honest error analysis reads as maturity; reviewers trust it more than
   cherry-picked wins.
7. **CPU vs GPU + PyTorch vs ONNX latency table.** Gives a consistent portfolio narrative with the MOT
   1.84× speedup — "this person ships fast, measured inference."
8. **`pytest` smoke suite + W&B offline fallback + per-epoch Drive checkpointing.** Colab-disconnect
   insurance and a reusable regression net inherited by Projects 2–5.
9. **Optional flip-TTA toggle.** A free, honest metric bump when you want it, off by default.

**Things I deliberately did NOT add** (scope discipline): no FastAPI server, no multi-dataset
training, no SOTA architecture chasing (PraNet/Polyp-PVT). Project 1 is the *template*; keep it lean.
Note those as "future work" in the README.

---

## 10. Coding standards (enforce throughout)

- Python 3.10+, type hints on all public functions, module + function docstrings.
- `ruff` + `black` clean. No unused imports, no dead code.
- **No hardcoded paths or hyperparameters** — everything via config.
- Pure functions where possible; explicit dependency passing; no hidden globals.
- Deterministic: `seed_everything()` covers `random`, `numpy`, `torch`, cudnn flags.
- Logging via `get_logger()`, not bare `print()`, in `src/` (scripts may print summaries).
- Small, single-responsibility modules. If a file exceeds ~200 lines, reconsider the split.
- Every `scripts/*.py` is a thin argparse wrapper over `src/`.

---

## 11. Known pitfalls (Kvasir-SEG + Colab specific)

- **Mask values:** Kvasir masks are JPEG and may contain near-but-not-exactly {0,255} values from
  compression. Binarise with `> 127`, don't assume clean binary. Verify in Stage 2.
- **Variable resolutions** (332×487 → 1920×1072): always resize to `image_size`; masks
  nearest-neighbour, images bilinear. Never letterbox masks with interpolation that creates grey edges.
- **Filename pairing:** images and masks share filenames in separate folders — pair by name, assert
  1:1, fail loudly on any mismatch.
- **Colab disconnects:** checkpoint to Drive every epoch and support `--resume`. Never keep the only
  copy of weights on local `/content/`.
- **Drive small-file I/O:** fine for 1k images, but if dataloading is slow, copy the extracted dataset
  to `/content/kvasir-seg/` at session start.
- **ONNX + dynamic shapes:** export with a dynamic batch axis only (fixed H/W = `image_size`) for a
  clean, fast graph; document the fixed input size in the model card.
- **GradCAM for segmentation** needs a scalar target — use `SemanticSegmentationTarget` over the
  predicted polyp region, not a classification target.

---

*End of spec. Build stage by stage, stop at every gate, keep the shared contract frozen.*
