from PIL import Image
import numpy as np
import os

base = r"C:\Users\gaura\OneDrive\Desktop\Galaxy Data"

for split in ["train", "val", "test"]:
    tgt_dir = os.path.join(base, split, "target")
    files = sorted(os.listdir(tgt_dir))

    change_pcts = []
    zero_tiles  = 0

    for f in files:
        mask = np.array(Image.open(os.path.join(tgt_dir, f)))
        mask_binary = (mask > 0)          # collapse 0,1,2,3 → binary
        pct = 100 * mask_binary.mean()
        change_pcts.append(pct)
        if pct == 0:
            zero_tiles += 1

    change_pcts = np.array(change_pcts)

    print(f"\n=== {split.upper()} ({len(files)} tiles) ===")
    print(f"  Mean change %    : {change_pcts.mean():.2f}%")
    print(f"  No-change %      : {100 - change_pcts.mean():.2f}%")
    print(f"  Min change %     : {change_pcts.min():.2f}%")
    print(f"  Max change %     : {change_pcts.max():.2f}%")
    print(f"  Zero-change tiles: {zero_tiles} / {len(files)}")
    print(f"  Ratio (no:change): {(100-change_pcts.mean())/change_pcts.mean():.1f} : 1")