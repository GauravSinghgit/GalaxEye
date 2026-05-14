import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        tgt   = targets.view(-1)
        inter = (probs * tgt).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (probs.sum() + tgt.sum() + self.smooth)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_w  = bce_weight
        self.dice_w = dice_weight
        self.bce    = nn.BCEWithLogitsLoss()
        self.dice   = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_w * self.bce(logits, targets) + self.dice_w * self.dice(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss for hard-example mining in imbalanced change detection."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class TverskyLoss(nn.Module):
    """
    Tversky loss — asymmetric Dice that penalises FN more than FP.
    Useful when missed changes (FN) are more costly than false alarms (FP).
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha  = alpha   # FP weight
        self.beta   = beta    # FN weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        tgt   = targets.view(-1)
        tp    = (probs * tgt).sum()
        fp    = (probs * (1 - tgt)).sum()
        fn    = ((1 - probs) * tgt).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """Non-linear variant of Tversky loss that focuses on very hard examples."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.gamma   = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.tversky(logits, targets) ** self.gamma


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_loss(cfg: dict) -> nn.Module:
    lc   = cfg["loss"]
    name = lc["primary"].lower()

    if name == "bce_dice":
        return BCEDiceLoss(lc.get("bce_weight", 0.5), lc.get("dice_weight", 0.5))
    if name == "dice":
        return DiceLoss()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "focal":
        return FocalLoss(lc.get("focal_alpha", 0.25), lc.get("focal_gamma", 2.0))
    if name == "tversky":
        return TverskyLoss(lc.get("tversky_alpha", 0.3), lc.get("tversky_beta", 0.7))
    if name == "focal_tversky":
        return FocalTverskyLoss(lc.get("tversky_alpha", 0.3), lc.get("tversky_beta", 0.7))
    raise ValueError(f"Unknown loss '{name}'. Options: bce_dice, dice, bce, focal, tversky, focal_tversky")
