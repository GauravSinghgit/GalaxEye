"""
inference.py — Single-pair or batch inference for EO-SAR Change Detection

Usage:
    # Single pair
    python inference.py \\
        --checkpoint experiments/run_X/checkpoints/best.pth \\
        --pre-event  /path/to/pre.tif \\
        --post-event /path/to/post.tif \\
        --output-dir inference_output

    # Directory batch (all files in pre-event dir matched by name)
    python inference.py \\
        --checkpoint experiments/run_X/checkpoints/best.pth \\
        --pre-dir  /path/to/pre_dir \\
        --post-dir /path/to/post_dir \\
        --output-dir inference_output
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from models.model import build_model
from utils.checkpoint import CheckpointManager
from utils.seed import set_seed


SUPPORTED_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EO-SAR Change Detection — Inference")
    p.add_argument("--config",      default="configs/config.yaml")
    p.add_argument("--checkpoint",  required=True)
    # Single-pair mode
    p.add_argument("--pre-event",   default=None, help="Pre-event EO image path")
    p.add_argument("--post-event",  default=None, help="Post-event SAR image path")
    # Batch mode
    p.add_argument("--pre-dir",     default=None, help="Directory of pre-event images")
    p.add_argument("--post-dir",    default=None, help="Directory of post-event images")
    p.add_argument("--output-dir",  default="inference_output")
    p.add_argument("--threshold",   type=float, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_eo(path: str) -> np.ndarray:
    img = tifffile.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def _load_sar(path: str) -> np.ndarray:
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.max() > 255.0 or img.min() < 0.0:
        lo, hi = img.min(), img.max()
        img = (img - lo) / (hi - lo + 1e-6) * 255.0
    return img.astype(np.uint8)


def preprocess(eo: np.ndarray, sar: np.ndarray, image_size: int) -> torch.Tensor:
    image = np.concatenate([eo, sar[:, :, np.newaxis]], axis=2)
    tf = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.0]*4, std=[1.0]*4, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return tf(image=image)["image"].unsqueeze(0)   # (1, 4, H, W)


def save_visualization(eo, sar, prob_map, save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Pre-event (EO)", "Post-event (SAR)", "Predicted Change"]
    data   = [eo, sar, prob_map]
    cmaps  = [None, "gray", "Reds"]

    for ax, title, d, cmap in zip(axes, titles, data, cmaps):
        kw = dict(cmap=cmap, vmin=0, vmax=1) if cmap else {}
        ax.imshow(d, **kw)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_inference(model, device, eo_path: str, sar_path: str, image_size: int,
                  threshold: float, output_dir: str, use_amp: bool) -> None:
    eo  = _load_eo(eo_path)
    sar = _load_sar(sar_path)

    tensor = preprocess(eo, sar, image_size).to(device)

    with torch.no_grad():
        if use_amp:
            with autocast():
                logits = model(tensor)
        else:
            logits = model(tensor)

    prob_map  = torch.sigmoid(logits)[0, 0].cpu().numpy()   # (H, W) float32
    pred_mask = (prob_map > threshold).astype(np.uint8)

    stem = Path(eo_path).stem
    tifffile.imwrite(os.path.join(output_dir, f"{stem}_mask.tif"),  pred_mask)
    tifffile.imwrite(os.path.join(output_dir, f"{stem}_prob.tif"),  prob_map.astype(np.float32))
    save_visualization(eo, sar, prob_map, os.path.join(output_dir, f"{stem}_vis.png"))

    n_changed = pred_mask.sum()
    pct       = 100 * n_changed / pred_mask.size
    print(f"  {stem}: {n_changed:,} changed px ({pct:.2f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    os.makedirs(args.output_dir, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = args.threshold or cfg["metrics"]["threshold"]
    size      = cfg["data"]["image_size"]
    use_amp   = cfg["training"]["mixed_precision"] and device.type == "cuda"

    model = build_model(cfg).to(device)
    state = CheckpointManager.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Model loaded (epoch {state.get('epoch', '?')})\n")

    # ── Single pair ───────────────────────────────────────────────────────
    if args.pre_event and args.post_event:
        run_inference(model, device, args.pre_event, args.post_event,
                      size, threshold, args.output_dir, use_amp)

    # ── Batch directory ───────────────────────────────────────────────────
    elif args.pre_dir and args.post_dir:
        pre_files = sorted(
            f for f in os.listdir(args.pre_dir)
            if Path(f).suffix.lower() in SUPPORTED_EXT
        )
        print(f"Found {len(pre_files)} images in {args.pre_dir}\n")
        for fname in pre_files:
            stem     = Path(fname).stem
            pre_path = os.path.join(args.pre_dir, fname)
            post_path = next(
                (os.path.join(args.post_dir, stem + ext)
                 for ext in SUPPORTED_EXT
                 if os.path.exists(os.path.join(args.post_dir, stem + ext))),
                None,
            )
            if post_path is None:
                print(f"  [SKIP] No post-event match for {fname}")
                continue
            run_inference(model, device, pre_path, post_path,
                          size, threshold, args.output_dir, use_amp)
    else:
        raise ValueError(
            "Provide either --pre-event / --post-event (single) "
            "or --pre-dir / --post-dir (batch)."
        )

    print(f"\nOutputs saved → {args.output_dir}")


if __name__ == "__main__":
    main()
