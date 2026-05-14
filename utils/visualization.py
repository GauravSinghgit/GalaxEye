import os
from typing import List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized (C,H,W) tensor → uint8 numpy array."""
    arr = tensor.numpy()
    return (arr * 255).clip(0, 255).astype(np.uint8)


def save_prediction_grid(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    fnames: List[str],
    save_dir: str,
    n_samples: int = 4,
) -> None:
    """Save a grid: EO | SAR | Ground Truth | Prediction."""
    os.makedirs(save_dir, exist_ok=True)
    n = min(n_samples, images.shape[0])

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(["EO (Pre-event)", "SAR (Post-event)", "Ground Truth", "Prediction"]):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for i in range(n):
        img = _to_numpy(images[i])            # (4, H, W) uint8
        eo  = img[:3].transpose(1, 2, 0)      # (H, W, 3)
        sar = img[3]                           # (H, W)
        gt  = masks[i, 0].numpy()
        pr  = preds[i, 0].numpy()

        axes[i, 0].imshow(eo)
        axes[i, 1].imshow(sar, cmap="gray")
        axes[i, 2].imshow(gt,  cmap="binary_r", vmin=0, vmax=1)
        axes[i, 3].imshow(pr,  cmap="binary_r", vmin=0, vmax=1)

        label = os.path.splitext(os.path.basename(fnames[i]))[0]
        axes[i, 0].set_ylabel(label, fontsize=7, rotation=0, labelpad=55, ha="right", va="center")

        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predictions.png"), dpi=120, bbox_inches="tight")
    plt.close()


def save_failure_analysis(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    fnames: List[str],
    save_dir: str,
    n_samples: int = 4,
) -> None:
    """Save EO | SAR | GT | Pred | Error-map (TP=green, FP=red, FN=blue)."""
    os.makedirs(save_dir, exist_ok=True)
    n = min(n_samples, images.shape[0])

    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(["EO", "SAR", "Ground Truth", "Prediction", "Error Map"]):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for i in range(n):
        img = _to_numpy(images[i])
        eo  = img[:3].transpose(1, 2, 0)
        sar = img[3]
        gt  = masks[i, 0].numpy().astype(bool)
        pr  = preds[i, 0].numpy().astype(bool)

        error = np.zeros((*gt.shape, 3), dtype=np.uint8)
        error[pr &  gt] = [0,   200, 0]    # TP — green
        error[pr & ~gt] = [220, 0,   0]    # FP — red
        error[~pr & gt] = [0,   0,   220]  # FN — blue

        axes[i, 0].imshow(eo)
        axes[i, 1].imshow(sar, cmap="gray")
        axes[i, 2].imshow(gt.astype(np.uint8), cmap="binary_r", vmin=0, vmax=1)
        axes[i, 3].imshow(pr.astype(np.uint8), cmap="binary_r", vmin=0, vmax=1)
        axes[i, 4].imshow(error)

        for j in range(5):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_analysis.png"), dpi=120, bbox_inches="tight")
    plt.close()
