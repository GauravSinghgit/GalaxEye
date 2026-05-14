from typing import Dict
import torch


class SegmentationMetrics:
    """Accumulates TP/FP/FN/TN counts across batches for binary segmentation."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        preds   : (B, 1, H, W) — already thresholded binary float tensor
        targets : (B, 1, H, W) — binary float tensor (0 or 1)
        """
        p = preds.bool()
        t = targets.bool()
        self.tp += int((p & t).sum())
        self.fp += int((p & ~t).sum())
        self.tn += int((~p & ~t).sum())
        self.fn += int((~p & t).sum())

    def compute(self) -> Dict[str, float]:
        eps = 1e-7
        precision = self.tp / (self.tp + self.fp + eps)
        recall    = self.tp / (self.tp + self.fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        iou       = self.tp / (self.tp + self.fp + self.fn + eps)
        return {
            "iou":       float(iou),
            "f1":        float(f1),
            "precision": float(precision),
            "recall":    float(recall),
        }
