import os
from typing import List, Optional, Tuple

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


SUPPORTED_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


class EOSARDataset(Dataset):
    """
    Loads paired EO (RGB pre-event) + SAR (grayscale post-event) tiles
    and their binary change masks.

    All three folders (pre-event, post-event, target) contain files with
    IDENTICAL filenames. The folder name determines the modality.

    Each sample returns:
        image  : (4, H, W) float32 tensor — EO(3) + SAR(1), normalised to [0,1]
        mask   : (1, H, W) float32 tensor — binary {0, 1}
        fname  : str — source filename for traceability
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 256,
        transform: Optional[A.Compose] = None,
    ):
        self.root_dir   = root_dir
        self.split      = split
        self.image_size = image_size
        self.transform  = transform

        self.pre_dir    = os.path.join(root_dir, split, "pre-event")
        self.post_dir   = os.path.join(root_dir, split, "post-event")
        self.target_dir = os.path.join(root_dir, split, "target")

        self._verify_dirs()
        self.filenames = self._discover_files()
        print(f"[{split:5s}] {len(self.filenames)} samples  →  {self.pre_dir}")

    # ------------------------------------------------------------------
    def _verify_dirs(self) -> None:
        for d in (self.pre_dir, self.post_dir, self.target_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Dataset directory not found: {d}")

    def _discover_files(self) -> List[str]:
        """
        List files from pre-event dir, keep only those whose identical
        filename also exists in post-event and target.
        """
        candidates = sorted(
            f for f in os.listdir(self.pre_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        )
        valid = [
            f for f in candidates
            if os.path.exists(os.path.join(self.post_dir,   f))
            and os.path.exists(os.path.join(self.target_dir, f))
        ]
        return valid

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname = self.filenames[idx]

        pre_path  = os.path.join(self.pre_dir,    fname)
        post_path = os.path.join(self.post_dir,   fname)
        tgt_path  = os.path.join(self.target_dir, fname)

        eo   = self._load_eo(pre_path)       # (H, W, 3) uint8
        sar  = self._load_sar(post_path)     # (H, W, 1) uint8
        mask = self._load_mask(tgt_path)     # (H, W)    uint8

        image = np.concatenate([eo, sar], axis=2)   # (H, W, 4)

        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out["image"]              # (4, H, W) float32 tensor
            mask  = out["mask"]               # (H, W) tensor
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask  = torch.from_numpy(mask).float()

        return image, mask.unsqueeze(0).float(), fname

    # ------------------------------------------------------------------
    @staticmethod
    def _load_eo(path: str) -> np.ndarray:
        img = tifffile.imread(path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        return img.astype(np.uint8)

    @staticmethod
    def _load_sar(path: str) -> np.ndarray:
        img = tifffile.imread(path).astype(np.float32)
        if img.ndim == 3:
            img = img[:, :, 0]
        # Rescale to [0, 255] if sensor delivers values outside uint8 range
        if img.max() > 255.0 or img.min() < 0.0:
            lo, hi = img.min(), img.max()
            img = (img - lo) / (hi - lo + 1e-6) * 255.0
        return img.astype(np.uint8)[:, :, np.newaxis]

    @staticmethod
    def _load_mask(path: str) -> np.ndarray:
        mask = tifffile.imread(path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return (mask > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def _make_gauss_noise(tr_cfg: dict) -> A.BasicTransform:
    """Build GaussNoise compatible with albumentations 1.3 – 2.x."""
    var_limit = tuple(tr_cfg.get("noise_var_limit", [10, 50]))
    try:
        # albumentations < 1.4 / pinned 1.3.x
        return A.GaussNoise(var_limit=var_limit, p=0.3)
    except TypeError:
        pass
    try:
        # albumentations 1.4.x — std_range replaces var_limit
        import math
        std = (math.sqrt(var_limit[0]) / 255.0, math.sqrt(var_limit[1]) / 255.0)
        return A.GaussNoise(std_range=std, p=0.3)
    except TypeError:
        # fallback: use defaults, no kwargs
        return A.GaussNoise(p=0.3)


def get_transforms(cfg: dict) -> Tuple[A.Compose, A.Compose]:
    size    = cfg["data"]["image_size"]
    aug_cfg = cfg.get("augmentation", {})
    tr_cfg  = aug_cfg.get("train", {})

    # Normalise all 4 channels to [0, 1]
    _normalize = A.Normalize(
        mean=[0.0, 0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    )

    def _maybe(flag_key: str, transform: A.BasicTransform) -> A.BasicTransform:
        return transform if tr_cfg.get(flag_key, True) else A.NoOp()

    train_transforms = A.Compose([

    A.Resize(size, size),

    A.HorizontalFlip(p=0.5),

    A.VerticalFlip(p=0.5),

    A.RandomRotate90(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.15,
        rotate_limit=20,
        border_mode=0,
        p=0.5,
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.25,
        contrast_limit=0.25,
        p=0.5,
    ),

    A.RandomGamma(
        gamma_limit=(80, 120),
        p=0.3,
    ),

    A.GaussianBlur(
        blur_limit=3,
        p=0.2,
    ),

    A.GaussNoise(
        std_range=(0.02, 0.08),
        p=0.3,
    ),

    _normalize,

    ToTensorV2(),
])

    val_transforms = A.Compose([
        A.Resize(size, size),
        _normalize,
        ToTensorV2(),
    ])

    return train_transforms, val_transforms

