import os
import shutil
import torch


class CheckpointManager:
    """Saves latest and best checkpoints; tracks the monitored metric."""

    def __init__(self, save_dir: str, monitor: str = "iou", mode: str = "max"):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        os.makedirs(save_dir, exist_ok=True)

    def save(self, state: dict, current_metric: float) -> bool:
        """Save latest checkpoint; copy to best if metric improved. Returns True if improved."""
        latest_path = os.path.join(self.save_dir, "latest.pth")
        torch.save(state, latest_path)

        improved = (
            (self.mode == "max" and current_metric > self.best_metric)
            or (self.mode == "min" and current_metric < self.best_metric)
        )
        if improved:
            self.best_metric = current_metric
            shutil.copy(latest_path, os.path.join(self.save_dir, "best.pth"))
        return improved

    @staticmethod
    def load(path: str, map_location="cpu") -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location=map_location)
