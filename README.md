# EO-SAR Change Detection

Binary pixel-level change detection on co-registered pre-event Electro-Optical (EO) and post-event Synthetic Aperture Radar (SAR) image pairs.  
The model classifies each pixel as **changed (1)** or **unchanged (0)** — targeting disaster response, urban monitoring, and environmental surveillance use cases.

**Architecture:** UNet++ with EfficientNet-B3 encoder, early-fusion 4-channel input (3-ch EO + 1-ch SAR).

---

## Requirements

- Python 3.10
- CUDA 12.8 / PyTorch 2.10

```
torch==2.10.0+cu128
segmentation-models-pytorch==0.3.3
timm==0.9.2
albumentations==1.3.1
tifffile==2024.2.12
tensorboard==2.14.0
pyyaml>=6.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
```

Full pinned list:

```bash
pip install -r requirements.txt
```

---

## Environment Setup

```bash
# 1. Create and activate environment
conda create -n eosar python=3.10 -y
conda activate eosar

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Dataset Structure

Place the provided dataset so it matches this layout:

```
Galaxy Data/
├── train/
│   ├── pre-event/      # EO RGB  (.tif, uint8, 3-channel)
│   ├── post-event/     # SAR     (.tif, uint8, 1-channel)
│   └── target/         # Mask    (.tif, binary 0/1)
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

| Split | Scenes | Samples |
|-------|--------|---------|
| train | scene_01 – scene_08 | 2 781 |
| val   | scene_01 – scene_08 |   334 |
| test  | scene_09 – scene_10 |    77 |

**Label remapping** (applied automatically in `datasets/dataset.py`):

| Original Class | Original Value | Remapped Value | Remapped Class |
|----------------|---------------|----------------|----------------|
| Background     | 0             | 0              | No-Change      |
| Intact         | 1             | 0              | No-Change      |
| Damaged        | 2             | 1              | Change         |
| Destroyed      | 3             | 1              | Change         |

Set `data.root_dir` in `configs/config.yaml` to the folder containing `train/`, `val/`, `test/`.

---

## Project Structure

```
Galaxy AI/
├── configs/
│   └── config.yaml          # All hyperparameters — single source of truth
├── datasets/
│   └── dataset.py           # EOSARDataset + label remapping + get_transforms()
├── models/
│   └── model.py             # build_model() — SMP UNet++ wrapper
├── losses/
│   └── losses.py            # BCE+Dice, Focal, Tversky, FocalTversky
├── utils/
│   ├── metrics.py           # SegmentationMetrics (IoU / F1 / Precision / Recall)
│   ├── visualization.py     # Prediction grids + error-analysis maps
│   ├── checkpoint.py        # CheckpointManager (best + latest)
│   ├── logger.py            # TensorBoard + CSV logger
│   └── seed.py              # set_seed() for full reproducibility
├── experiments/             # Auto-created; one sub-dir per run
│   └── run_YYYYMMDD_HHMMSS/
│       ├── checkpoints/     # best.pth, latest.pth
│       ├── logs/            # TensorBoard events + metrics.csv
│       ├── visualizations/  # Epoch-level prediction grids
│       └── config.yaml      # Config snapshot for this run
├── notebooks/
│   └── train_kaggle.ipynb   # Kaggle experiment launcher
├── train.py
├── eval.py
├── inference.py
├── requirements.txt
└── README.md
```

---

## Training

```bash
# Full training
python train.py --config configs/config.yaml --name kaggle_run_v1

# 2-epoch sanity check
python train.py --config configs/config.yaml --debug

# Resume from checkpoint
python train.py \
    --config configs/config.yaml \
    --resume experiments/run_X/checkpoints/latest.pth
```

---

## Evaluation

```bash
# Validation set
python eval.py \
    --config configs/config.yaml \
    --checkpoint experiments/kaggle_run_v1/checkpoints/best.pth \
    --split val \
    --n-vis 8

# Test set (with TTA — enabled automatically for --split test)
python eval.py \
    --config configs/config.yaml \
    --checkpoint experiments/kaggle_run_v1/checkpoints/best.pth \
    --split test \
    --n-vis 16
```

Outputs saved to `experiments/run_X/eval_<split>/`:
- `metrics.json` — full numeric report
- `predictions/predictions.png` — EO | SAR | Ground Truth | Prediction grid
- `error_analysis/error_analysis.png` — TP / FP / FN overlay

---

## Inference

```bash
# Single image pair
python inference.py \
    --checkpoint experiments/kaggle_run_v1/checkpoints/best.pth \
    --pre-event  /path/to/pre.tif \
    --post-event /path/to/post.tif \
    --output-dir inference_output

# Batch directory
python inference.py \
    --checkpoint experiments/kaggle_run_v1/checkpoints/best.pth \
    --pre-dir  /path/to/pre_dir \
    --post-dir /path/to/post_dir \
    --output-dir inference_output
```

---

## Model Weights

Download the final trained checkpoint (epoch 28, best val IoU):

**[Download best.pth — Google Drive](https://drive.google.com/file/d/1nT2dcNnQQaU5NovIaeoamhDojotROlHl/view?usp=sharing)**

---

## Results

Metrics computed for the **change class (label = 1)** using threshold = 0.94 with Test-Time Augmentation (TTA) on the test split.

| Split   | Loss   | IoU    | F1 Score | Precision | Recall |
|---------|--------|--------|----------|-----------|--------|
| **Val** | 0.2042 | 0.7990 | 0.8830   | 0.8330    | 0.9390 |
| **Test**| 0.4792 | 0.4900 | 0.6580   | 0.6150    | 0.7060 |

> **Note on val/test gap:** Train and val share the same 8 scenes (patch-level split); test uses 2 entirely unseen scenes (scene_09–10). The val IoU is therefore inflated by scene-level familiarity. The test score (~0.48 IoU) represents true cross-scene generalisation performance, which is the more honest number.

### Confusion Matrix — Test Set (threshold = 0.99, with TTA)

|                    | Predicted No-Change | Predicted Change |
|--------------------|--------------------:|----------------:|
| **Actual No-Change** | TN (high — model conservative at t=0.94) | FP |
| **Actual Change**    | FN | TP |

**Error profile:** Precision (0.61) is the binding constraint — the model over-predicts change in test scenes. Recall (0.70) is reasonable, meaning most true change pixels are found. The false positive pattern suggests the model fires on SAR texture differences that are noise/speckle in the unseen scenes rather than structural change.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Fusion strategy | Early fusion (channel concat) | Encoder learns cross-modal features jointly; simpler than late/mid fusion |
| Architecture | UNet++ + EfficientNet-B3 | Dense skip connections capture fine change boundaries; B3 balances capacity vs. compute |
| Loss | Focal Tversky (α=0.25, β=0.75) | Penalises false negatives harder — appropriate for sparse change class |
| Threshold | 0.94 (tuned on val) | Raises precision without large recall drop; default 0.5 was too aggressive on test |
| TTA | 4-flip averaging (H, V, HV, orig) | ~0.03 IoU improvement on test at zero training cost |
| Augmentation | Elastic, CLAHE, multiplicative noise, coarse dropout | Improve SAR generalisation across unseen scene conditions |

---

## Configuration Reference

```yaml
model:
  architecture: UnetPlusPlus
  encoder:       efficientnet-b3
  in_channels:   4               # 3 EO + 1 SAR
  classes:       1

training:
  epochs:                   40
  batch_size:                8
  learning_rate:           1e-4
  mixed_precision:          true
  early_stopping_patience:  10
  gradient_clip:            1.0
  weight_decay:           1e-4

loss:
  primary:        focal_tversky
  tversky_alpha:  0.25            # FP weight
  tversky_beta:   0.75            # FN weight (recall-biased)

metrics:
  threshold: 0.94

scheduler:
  name:  cosine
  T_max: 40
  min_lr: 1e-6
```

---

## TensorBoard

```bash
tensorboard --logdir experiments
```

Tracks: train/val loss, IoU, F1, Precision, Recall, learning rate per epoch.

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
    "cfg":                  dict,
}
```

```python
import torch
state = torch.load("best.pth", map_location="cpu")
model.load_state_dict(state["model_state_dict"])
```

---

## Citations / References

- **segmentation-models-pytorch**: Iakubovskii, P. (2019). [https://github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- **UNet++**: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", DLMIA 2018.
- **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019.
- **Focal Tversky Loss**: Abraham & Khan, "A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation", ISBI 2019.
- **Albumentations**: Buslaev et al., "Albumentations: Fast and Flexible Image Augmentations", Information 2020.
- **TTA**: Shanmugam et al., "Better Aggregation in Test-Time Augmentation", ICCV 2021.
