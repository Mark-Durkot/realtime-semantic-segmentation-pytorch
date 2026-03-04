import argparse
import os

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Compute class weights for VDD masks.")
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/content/drive/MyDrive/VDD/train/gt",
        help="Path to VDD train/gt directory.",
    )
    parser.add_argument(
        "--palette",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6],
        help="Grayscale mask values for classes in order.",
    )
    parser.add_argument(
        "--min_weight",
        type=float,
        default=0.5,
        help="Minimum clipped weight.",
    )
    parser.add_argument(
        "--max_weight",
        type=float,
        default=4.0,
        help="Maximum clipped weight.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    if not os.path.isdir(args.gt_dir):
        raise RuntimeError(f"Ground-truth directory does not exist: {args.gt_dir}")

    counts = np.zeros(len(args.palette), dtype=np.float64)
    mask_files = [f for f in os.listdir(args.gt_dir) if f.lower().endswith(valid_exts)]

    if len(mask_files) == 0:
        raise RuntimeError(f"No mask files found in: {args.gt_dir}")

    for file_name in sorted(mask_files):
        mask = np.asarray(Image.open(os.path.join(args.gt_dir, file_name)))
        if mask.ndim == 3:
            mask = mask[..., 0]

        for i, value in enumerate(args.palette):
            counts[i] += np.sum(mask == value)

    total = counts.sum()
    freq = counts / (total + 1e-12)

    nonzero = freq > 0
    if not np.any(nonzero):
        raise RuntimeError("All class frequencies are zero. Check mask encoding.")

    median_freq = np.median(freq[nonzero])
    weights = median_freq / (freq + 1e-12)
    weights = np.clip(weights, args.min_weight, args.max_weight)

    print("palette:", args.palette)
    print("counts:", counts.astype(np.int64).tolist())
    print("freq:", [round(x, 6) for x in freq.tolist()])
    print("weights:", [round(x, 4) for x in weights.tolist()])


if __name__ == "__main__":
    main()
