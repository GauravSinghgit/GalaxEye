# EO-SAR Change Detection

Binary semantic segmentation for multimodal change detection using pre-event EO (Electro-Optical) RGB imagery and post-event SAR (Synthetic Aperture Radar) grayscale imagery.  
Predicts pixel-wise building damage / change masks (`0` = no change, `1` = change).

---

## Overview

| Property | Value |
|---|---|
| Task | Binary semantic segmentation |
| Input | EO RGB (pre-event) + SAR grayscale (post-event) → 4-channel tensor |
| Model | UNet++ with EfficientNet-B0 encoder |
| Resolution | 1024 × 1024 raw → 256 × 256 training crops |
| Loss | BCE + Dice (configurable) |
| Metrics | IoU, F1, Precision, Recall |
| Evaluation | Scene-wise (val: scene_01–08, test: scene_09–10) |

### Key Engineering Features
- Config-driven — all hyperparameters in `configs/config.yaml`; nothing hardcoded
- Mixed-precision training (`torch.cuda.amp`)
- Gradient clipping for training stability
- Automatic best/latest checkpointing with safe `KeyboardInterrupt` recovery
- Early stopping with configurable patience
- TensorBoard + CSV logging
- Reproducible via fixed seeds and deterministic CUDA

---

## Project Structure

```
GalaxyAI/
├── configs/
│   └── config.yaml          # All hyperparameters — single source of truth
│
├── datasets/
│   └── dataset.py           # EOSARDataset + get_transforms()
│
├── models/
│   └── model.py             # build_model() — SMP UNet++ wrapper
│
├── losses/
│   └── losses.py            # BCE+Dice, Focal, Tversky, FocalTversky
│
├── utils/
│   ├── metrics.py           # SegmentationMetrics (IoU / F1 / P / R)
│   ├── visualization.py     # Prediction grids + error-analysis maps
│   ├── checkpoint.py        # CheckpointManager (best + latest)
│   ├── logger.py            # TensorBoardLogger + CSVLogger
│   └── seed.py              # set_seed() for full reproducibility
│
├── experiments/             # Auto-created; one sub-dir per run
│   └── run_YYYYMMDD_HHMMSS/
│       ├── checkpoints/     # best.pth, latest.pth
│       ├── logs/            # TensorBoard events + metrics.csv
│       ├── visualizations/  # Epoch-level prediction grids
│       └── config.yaml      # Snapshot of config used for this run
│
├── notebooks/
│   └── train_kaggle.ipynb   # Kaggle experiment launcher
│
├── train.py                 # Training pipeline
├── eval.py                  # Evaluation pipeline
├── inference.py             # Single-pair / batch inference
├── requirements.txt
└── README.md
```

---

## Dataset Structure

```
data/
├── train/
│   ├── pre-event/    # EO RGB  (1024×1024×3, uint8)
│   ├── post-event/   # SAR     (1024×1024,   uint8)
│   └── target/       # Mask    (1024×1024,   binary 0/1)
├── val/
│   └── ...           # scene_01 – scene_08
└── test/
    └── ...           # scene_09 – scene_10
```

| Split | Samples |
|---|---|
| train | 2 781 |
| val | 334 |
| test | 77 |

Set `data.root_dir` in `configs/config.yaml` to the folder containing `train/`, `val/`, `test/`.

---

## Setup

### 1. Create environment

```bash
conda create -n eosar python=3.10 -y
conda activate eosar
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure data path

Edit `configs/config.yaml`:

```yaml
data:
  root_dir: "/path/to/your/data"   # folder containing train/ val/ test/
```

---

## Training

### Sanity check (2 epochs, no GPU memory risk)

```bash
python train.py --config configs/config.yaml --debug
```

### Full training

```bash
python train.py --config configs/config.yaml
```

### Named run

```bash
python train.py --config configs/config.yaml --name my_experiment_v1
```

### Resume from checkpoint

```bash
python train.py \
    --config configs/config.yaml \
    --resume experiments/run_20240101_120000/checkpoints/latest.pth
```

---

## Evaluation

### Validate on val set (use during hyperparameter search)

```bash
python eval.py \
    --checkpoint experiments/run_X/checkpoints/best.pth \
    --split val
```

### Final evaluation on test set (run once only)

```bash
python eval.py \
    --checkpoint experiments/run_X/checkpoints/best.pth \
    --split test \
    --n-vis 16
```

Results are saved to `experiments/run_X/eval_test/`:
- `metrics.json` — full numeric report
- `predictions/predictions.png` — EO | SAR | GT | Pred grid
- `error_analysis/error_analysis.png` — TP/FP/FN overlay

---

## Inference

### Single image pair

```bash
python inference.py \
    --checkpoint experiments/run_X/checkpoints/best.pth \
    --pre-event  /path/to/pre.tif \
    --post-event /path/to/post.tif \
    --output-dir inference_output
```

### Batch directory

```bash
python inference.py \
    --checkpoint experiments/run_X/checkpoints/best.pth \
    --pre-dir  /path/to/pre_dir \
    --post-dir /path/to/post_dir \
    --output-dir inference_output
```

Each image produces:
- `<stem>_mask.tif` — binary prediction
- `<stem>_prob.tif` — raw probability map (float32)
- `<stem>_vis.png` — EO | SAR | probability overlay

---

## TensorBoard

```bash
tensorboard --logdir experiments
```

Tracks: train/val loss, IoU, F1, Precision, Recall, learning rate.

---

## Configuration Reference

Key sections in `configs/config.yaml`:

```yaml
model:
  architecture: "UnetPlusPlus"    # UnetPlusPlus | Unet | DeepLabV3Plus | FPN
  encoder:      "efficientnet-b0"
  in_channels:  4                 # 3 EO + 1 SAR

training:
  epochs:                  40
  batch_size:               8
  learning_rate:         1e-4
  mixed_precision:        true
  early_stopping_patience:  10
  gradient_clip:           1.0

loss:
  primary: "bce_dice"             # bce_dice | dice | focal | tversky

scheduler:
  name: "cosine"                  # cosine | step | plateau
```

---

## Results

| Split | Loss | IoU | F1 | Precision | Recall |
|---|---|---|---|---|---|
| val | — | — | — | — | — |
| test | — | — | — | — | — |

*Fill in after training.*

---

## Design Notes

### EO-SAR Modality Gap
EO captures surface reflectance in optical wavelengths; SAR captures microwave backscatter regardless of cloud cover. The two modalities have fundamentally different noise characteristics and spatial statistics. Early fusion (channel concatenation) is used here so the encoder learns cross-modal correspondences jointly.

### SAR Speckle
SAR imagery exhibits multiplicative speckle noise. `GaussNoise` augmentation during training acts as a partial proxy for this; in production, multi-look or Lee filtering could be applied as a pre-processing step.

### Scene-wise Evaluation
The val/test split is scene-based, meaning the model is evaluated on entirely unseen geographic regions. This tests cross-scene domain generalisation rather than in-distribution performance — a stricter and more realistic protocol for disaster response applications.

### Class Imbalance
Change pixels are a small fraction of total pixels. BCE+Dice loss is used to balance pixel-level cross-entropy with region-level overlap. Tversky loss (higher `beta`) is an alternative when recall is prioritised over precision.

---

## Checkpoint Format

```python
{
    "epoch":                int,
    "model_state_dict":     OrderedDict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "best_metric":          float,
    "scores":               {"iou": ..., "f1": ..., "precision": ..., "recall": ...},
    "cfg":                  dict,   # full config snapshot
}
```

Load manually:

```python
import torch
state = torch.load("experiments/run_X/checkpoints/best.pth", map_location="cpu")
model.load_state_dict(state["model_state_dict"])
```
