"""
eval.py — EO-SAR Change Detection Evaluation

Usage:
    python eval.py --checkpoint experiments/run_X/checkpoints/best.pth
    python eval.py --checkpoint experiments/run_X/checkpoints/best.pth --split val
    python eval.py --checkpoint experiments/run_X/checkpoints/best.pth --split test --n-vis 16
"""

import os
import json
import argparse
import time
import yaml

import torch
from torch.utils.data import DataLoader
try:
    from torch.amp import autocast as _autocast
    def autocast(enabled=True):
        return _autocast("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast  # type: ignore[assignment]

from datasets.dataset import EOSARDataset, get_transforms
from models.model import build_model
from losses.losses import build_loss
from utils.metrics import SegmentationMetrics
from utils.checkpoint import CheckpointManager
from utils.seed import set_seed
from utils.visualization import save_prediction_grid, save_failure_analysis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EO-SAR Change Detection — Evaluation")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--save-dir",   default=None,
                   help="Output directory (default: <exp>/eval_<split>)")
    p.add_argument("--n-vis",      type=int, default=8,
                   help="Number of samples to visualise")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])

    # ── Save directory ────────────────────────────────────────────────────
    save_dir = args.save_dir or os.path.normpath(
        os.path.join(os.path.dirname(args.checkpoint), "..", f"eval_{args.split}")
    )
    os.makedirs(save_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    print(f"Device : {device}")

    # ── Dataset ───────────────────────────────────────────────────────────
    _, val_tf = get_transforms(cfg)
    dataset   = EOSARDataset(
        cfg["data"]["root_dir"], args.split, cfg["data"]["image_size"], val_tf
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"] and device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    state = CheckpointManager.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    trained_epoch = state.get("epoch", "?")
    print(f"Checkpoint epoch: {trained_epoch}\n")

    # ── Evaluation loop ───────────────────────────────────────────────────
    criterion       = build_loss(cfg)
    metrics_tracker = SegmentationMetrics(cfg["metrics"]["threshold"])
    threshold       = cfg["metrics"]["threshold"]
    total_loss      = 0.0

    all_imgs, all_masks, all_preds, all_fnames = [], [], [], []

    t0 = time.time()
    with torch.no_grad():
        for images, masks, fnames in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            if use_amp:
                with autocast():
                    logits = model(images)
                    loss   = criterion(logits, masks)
            else:
                logits = model(images)
                loss   = criterion(logits, masks)

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > threshold).float()
            metrics_tracker.update(preds, masks)

            all_imgs.append(images.cpu())
            all_masks.append(masks.cpu())
            all_preds.append(preds.cpu())
            all_fnames.extend(list(fnames))

    elapsed = time.time() - t0
    scores  = metrics_tracker.compute()
    avg_loss = total_loss / len(loader)

    # ── Report ────────────────────────────────────────────────────────────
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Evaluation Results  [{args.split.upper()}]")
    print(sep)
    print(f"  Samples    : {len(dataset)}")
    print(f"  Loss       : {avg_loss:.4f}")
    print(f"  IoU        : {scores['iou']:.4f}")
    print(f"  F1 Score   : {scores['f1']:.4f}")
    print(f"  Precision  : {scores['precision']:.4f}")
    print(f"  Recall     : {scores['recall']:.4f}")
    print(f"  Time       : {elapsed:.1f}s")
    print(sep)

    # ── Save metrics JSON ─────────────────────────────────────────────────
    results = {
        "split":      args.split,
        "checkpoint": args.checkpoint,
        "epoch":      trained_epoch,
        "n_samples":  len(dataset),
        "loss":       avg_loss,
        **scores,
        "eval_time_s": round(elapsed, 2),
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics  → {os.path.join(save_dir, 'metrics.json')}")

    # ── Visualisations ────────────────────────────────────────────────────
    all_imgs  = torch.cat(all_imgs)
    all_masks = torch.cat(all_masks)
    all_preds = torch.cat(all_preds)
    n_vis     = min(args.n_vis, len(all_imgs))

    save_prediction_grid(
        images=all_imgs[:n_vis], masks=all_masks[:n_vis], preds=all_preds[:n_vis],
        fnames=all_fnames[:n_vis],
        save_dir=os.path.join(save_dir, "predictions"), n_samples=n_vis,
    )
    save_failure_analysis(
        images=all_imgs[:n_vis], masks=all_masks[:n_vis], preds=all_preds[:n_vis],
        fnames=all_fnames[:n_vis],
        save_dir=os.path.join(save_dir, "error_analysis"), n_samples=n_vis,
    )
    print(f"Visuals  → {save_dir}\n")


if __name__ == "__main__":
    main()
