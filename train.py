"""
train.py — EO-SAR Change Detection Training Pipeline

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --resume experiments/run_X/checkpoints/latest.pth
    python train.py --config configs/config.yaml --debug
"""

import os
import sys
import argparse
import shutil
import time
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    # PyTorch >= 2.4 preferred API
    from torch.amp import GradScaler, autocast as _autocast
    def autocast(enabled=True):           # thin wrapper keeps call-sites identical
        return _autocast("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[assignment]

from datasets.dataset import EOSARDataset, get_transforms
from models.model import build_model
from losses.losses import build_loss
from utils.metrics import SegmentationMetrics
from utils.checkpoint import CheckpointManager
from utils.logger import TensorBoardLogger, CSVLogger
from utils.seed import set_seed
from utils.visualization import save_prediction_grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EO-SAR Change Detection — Training")
    p.add_argument("--config",  default="configs/config.yaml", help="YAML config path")
    p.add_argument("--resume",  default=None, help="Checkpoint path to resume from")
    p.add_argument("--name",    default=None, help="Override experiment run name")
    p.add_argument("--debug",   action="store_true", help="2-epoch sanity check mode")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Optimizer / Scheduler builders
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg["optimizer"]["name"].lower()
    lr   = cfg["training"]["learning_rate"]
    wd   = cfg["training"]["weight_decay"]
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    sc   = cfg["scheduler"]
    name = sc["name"].lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sc.get("T_max", cfg["training"]["epochs"]),
            eta_min=sc.get("min_lr", 1e-6),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sc.get("step_size", 10), gamma=sc.get("gamma", 0.5)
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=sc.get("patience", 5),
            factor=sc.get("factor", 0.5),
            min_lr=sc.get("min_lr", 1e-6),
        )
    # fallback
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )


# ---------------------------------------------------------------------------
# Train / Validate loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch):
    model.train()
    total_loss  = 0.0
    n_batches   = len(loader)
    log_every   = cfg["logging"]["log_interval"]
    use_amp     = cfg["training"]["mixed_precision"] and device.type == "cuda"
    clip_norm   = cfg["training"]["gradient_clip"]

    for i, (images, masks, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                logits = model(images)
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        total_loss += loss.item()

        if (i + 1) % log_every == 0 or i == n_batches - 1:
            print(f"    [{i+1:4d}/{n_batches}]  loss={loss.item():.4f}")

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, criterion, metrics_tracker, device, cfg, epoch, exp_dir):
    model.eval()
    total_loss  = 0.0
    use_amp     = cfg["training"]["mixed_precision"] and device.type == "cuda"
    threshold   = cfg["metrics"]["threshold"]
    save_vis    = (epoch % cfg["logging"]["save_vis_every"] == 0)
    metrics_tracker.reset()

    vis_imgs, vis_masks, vis_preds, vis_fnames = [], [], [], []

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

        if save_vis and len(vis_imgs) < 4:
            vis_imgs.append(images.cpu())
            vis_masks.append(masks.cpu())
            vis_preds.append(preds.cpu())
            vis_fnames.extend(list(fnames))

    if save_vis and vis_imgs:
        save_prediction_grid(
            images=torch.cat(vis_imgs)[:4],
            masks=torch.cat(vis_masks)[:4],
            preds=torch.cat(vis_preds)[:4],
            fnames=vis_fnames[:4],
            save_dir=os.path.join(exp_dir, "visualizations", f"epoch_{epoch:03d}"),
            n_samples=4,
        )

    return total_loss / len(loader), metrics_tracker.compute()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.debug:
        cfg["training"]["epochs"]               = 2
        cfg["data"]["num_workers"]              = 0
        cfg["logging"]["save_vis_every"]        = 1
        cfg["training"]["early_stopping_patience"] = 999
        print("[DEBUG] 2-epoch sanity check — workers=0")

    # ── Experiment directory ──────────────────────────────────────────────
    run_name = args.name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir  = os.path.join(cfg["project"]["experiment_dir"], run_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir  = os.path.join(exp_dir, "logs")

    for d in (exp_dir, ckpt_dir, log_dir, os.path.join(exp_dir, "visualizations")):
        os.makedirs(d, exist_ok=True)

    shutil.copy(args.config, os.path.join(exp_dir, "config.yaml"))
    print(f"\nExperiment : {exp_dir}")

    # ── Reproducibility ───────────────────────────────────────────────────
    set_seed(cfg["project"]["seed"])

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Device     : {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms(cfg)
    train_ds = EOSARDataset(cfg["data"]["root_dir"], "train", cfg["data"]["image_size"], train_tf)
    val_ds   = EOSARDataset(cfg["data"]["root_dir"], "val",   cfg["data"]["image_size"], val_tf)

    bs, nw   = cfg["training"]["batch_size"], cfg["data"]["num_workers"]
    pin_mem  = cfg["data"]["pin_memory"] and device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,  num_workers=nw,
        pin_memory=pin_mem, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw,
        pin_memory=pin_mem,
    )

    # ── Model / loss / optim ──────────────────────────────────────────────
    model     = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    _amp_on   = cfg["training"]["mixed_precision"] and device.type == "cuda"
    try:
        scaler = GradScaler("cuda", enabled=_amp_on)   # torch >= 2.4
    except TypeError:
        scaler = GradScaler(enabled=_amp_on)            # torch < 2.4

    # ── Utilities ─────────────────────────────────────────────────────────
    metrics_tracker = SegmentationMetrics(cfg["metrics"]["threshold"])
    ckpt_manager    = CheckpointManager(
        ckpt_dir,
        monitor=cfg["checkpointing"]["monitor"],
        mode=cfg["checkpointing"]["mode"],
    )
    tb_logger  = TensorBoardLogger(log_dir)
    csv_logger = CSVLogger(os.path.join(log_dir, "metrics.csv"))

    # ── Optional resume ───────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        state = CheckpointManager.load(args.resume, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch             = state.get("epoch", 0) + 1
        ckpt_manager.best_metric = state.get("best_metric", ckpt_manager.best_metric)
        print(f"Resumed from epoch {start_epoch}\n")

    # ── Training loop ─────────────────────────────────────────────────────
    patience   = cfg["training"]["early_stopping_patience"]
    no_improve = 0
    max_epochs = cfg["training"]["epochs"]

    print(f"Training epochs {start_epoch+1} → {max_epochs}\n")
    print(f"{'Epoch':>6} | {'TrLoss':>7} | {'VaLoss':>7} | {'IoU':>6} | {'F1':>6} | {'Prec':>6} | {'Rec':>6} | {'LR':>9}")
    print("─" * 78)

    try:
        for epoch in range(start_epoch, max_epochs):
            t0 = time.time()

            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, cfg, epoch
            )
            val_loss, scores = validate(
                model, val_loader, criterion, metrics_tracker, device, cfg, epoch, exp_dir
            )

            # Scheduler step
            sched_name = cfg["scheduler"]["name"].lower()
            if sched_name == "plateau":
                scheduler.step(scores["iou"])
            else:
                scheduler.step()

            lr      = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # ── Log ───────────────────────────────────────────────────────
            row = {
                "epoch":         epoch + 1,
                "train_loss":    train_loss,
                "val_loss":      val_loss,
                "val_iou":       scores["iou"],
                "val_f1":        scores["f1"],
                "val_precision": scores["precision"],
                "val_recall":    scores["recall"],
                "lr":            lr,
                "epoch_time_s":  round(elapsed, 1),
            }
            tb_logger.log(row, epoch)
            csv_logger.log(row)

            print(
                f"{epoch+1:6d} | {train_loss:7.4f} | {val_loss:7.4f} | "
                f"{scores['iou']:6.4f} | {scores['f1']:6.4f} | "
                f"{scores['precision']:6.4f} | {scores['recall']:6.4f} | "
                f"{lr:9.2e}  {elapsed:.0f}s"
            )

            # ── Checkpoint ────────────────────────────────────────────────
            state = {
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric":         ckpt_manager.best_metric,
                "scores":              scores,
                "cfg":                 cfg,
            }
            improved = ckpt_manager.save(state, scores[cfg["checkpointing"]["monitor"]])

            if improved:
                no_improve = 0
                print(f"  ★ New best {cfg['checkpointing']['monitor']}: {ckpt_manager.best_metric:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\nEarly stopping — no improvement for {patience} epochs.")
                    break

    except KeyboardInterrupt:
        print("\n[Interrupted]  Saving emergency checkpoint …")
        emergency_path = os.path.join(ckpt_dir, "interrupt.pth")
        torch.save(
            {
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric":         ckpt_manager.best_metric,
                "cfg":                 cfg,
            },
            emergency_path,
        )
        print(f"Saved: {emergency_path}")

    finally:
        tb_logger.close()
        print(f"\nBest {cfg['checkpointing']['monitor']}: {ckpt_manager.best_metric:.4f}")
        print(f"Results  : {exp_dir}")


if __name__ == "__main__":
    main()
