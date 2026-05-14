import os
import csv
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, metrics: dict, step: int) -> None:
        loss_scalars = {}
        if metrics.get("train_loss") is not None:
            loss_scalars["train"] = metrics["train_loss"]
        if metrics.get("val_loss") is not None:
            loss_scalars["val"] = metrics["val_loss"]
        if loss_scalars:
            self.writer.add_scalars("Loss", loss_scalars, step)

        for key in ("val_iou", "val_f1", "val_precision", "val_recall"):
            if metrics.get(key) is not None:
                tag = "Metrics/" + key.replace("val_", "").upper()
                self.writer.add_scalar(tag, metrics[key], step)

        if metrics.get("lr") is not None:
            self.writer.add_scalar("Training/LR", metrics["lr"], step)

    def close(self) -> None:
        self.writer.close()


class CSVLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._initialized = False
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    def log(self, metrics: dict) -> None:
        if not self._initialized:
            self._fieldnames = list(metrics.keys())
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fieldnames).writeheader()
            self._initialized = True

        row = {
            k: f"{v:.6f}" if isinstance(v, float) else v
            for k, v in metrics.items()
        }
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore").writerow(row)
