import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models import get_model
from configs.vdd_config import VDDConfig


VDD_GRAY_PALETTE = np.array([0, 38, 75, 113, 150, 188, 225], dtype=np.uint8)
VDD_COLOR_PALETTE = np.array(
    [
        [0, 0, 0],        # other
        [255, 0, 0],      # wall
        [128, 64, 128],   # road
        [0, 255, 0],      # vegetation
        [0, 0, 255],      # vehicle
        [255, 255, 0],    # roof
        [0, 255, 255],    # water
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run VDD inference from best checkpoint.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/content/MyDrive/VDD/results/save/best.pth",
        help="Path to trained checkpoint (best.pth).",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="/content/MyDrive/VDD/test/src",
        help="Directory containing test RGB images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/MyDrive/VDD/results",
        help="Directory to save predicted outputs.",
    )
    parser.add_argument("--tile_size", type=int, default=512, help="Sliding-window tile size.")
    parser.add_argument("--stride", type=int, default=384, help="Sliding-window stride.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay alpha for mask blend.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    return parser.parse_args()


def normalize_to_tensor(image_np: np.ndarray, device: torch.device) -> torch.Tensor:
    image = image_np.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    return tensor


@torch.no_grad()
def sliding_window_predict(model, image_np: np.ndarray, num_class: int, tile_size: int, stride: int, device):
    h, w = image_np.shape[:2]
    logits_sum = torch.zeros((1, num_class, h, w), device=device)
    count = torch.zeros((1, 1, h, w), device=device)

    y_list = list(range(0, max(h - tile_size + 1, 1), stride))
    x_list = list(range(0, max(w - tile_size + 1, 1), stride))
    if not y_list or y_list[-1] != max(h - tile_size, 0):
        y_list.append(max(h - tile_size, 0))
    if not x_list or x_list[-1] != max(w - tile_size, 0):
        x_list.append(max(w - tile_size, 0))

    for y in y_list:
        for x in x_list:
            crop = image_np[y:y + tile_size, x:x + tile_size]
            ch, cw = crop.shape[:2]

            # Pad only edge tiles to full tile_size for model forward.
            if ch != tile_size or cw != tile_size:
                pad_crop = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                pad_crop[:ch, :cw] = crop
            else:
                pad_crop = crop

            inp = normalize_to_tensor(pad_crop, device)
            pred = model(inp)[:, :, :ch, :cw]
            logits_sum[:, :, y:y + ch, x:x + cw] += pred
            count[:, :, y:y + ch, x:x + cw] += 1

    logits = logits_sum / torch.clamp(count, min=1.0)
    pred_cls = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred_cls


def ensure_input(path: str, fallback: str) -> str:
    if os.path.exists(path):
        return path
    if os.path.exists(fallback):
        return fallback
    return path


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    ckpt_path = ensure_input(args.ckpt_path, "/content/drive/MyDrive/VDD/results/save/best.pth")
    test_dir = ensure_input(args.test_dir, "/content/drive/MyDrive/VDD/test/src")
    output_dir = ensure_input(args.output_dir, "/content/drive/MyDrive/VDD/results")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test image directory not found: {test_dir}")

    pred_dir = os.path.join(output_dir, "pred_masks")
    color_dir = os.path.join(output_dir, "pred_color_masks")
    overlay_dir = os.path.join(output_dir, "pred_overlays")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    config = VDDConfig()
    model = get_model(config).to(device).eval()

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    del checkpoint

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = sorted([p for p in Path(test_dir).iterdir() if p.suffix.lower() in valid_exts])
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {test_dir}")

    print(f"Loaded {len(image_paths)} test images.")
    print(f"Saving outputs to: {output_dir}")

    for img_path in image_paths:
        image = np.asarray(Image.open(img_path).convert("RGB"))
        pred_cls = sliding_window_predict(
            model=model,
            image_np=image,
            num_class=config.num_class,
            tile_size=args.tile_size,
            stride=args.stride,
            device=device,
        )

        pred_gray = VDD_GRAY_PALETTE[pred_cls]
        pred_color = VDD_COLOR_PALETTE[pred_cls]
        overlay = (image.astype(np.float32) * (1.0 - args.alpha) + pred_color.astype(np.float32) * args.alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        stem = img_path.stem
        Image.fromarray(pred_gray).save(os.path.join(pred_dir, f"{stem}.png"))
        Image.fromarray(pred_color).save(os.path.join(color_dir, f"{stem}.png"))
        Image.fromarray(overlay).save(os.path.join(overlay_dir, f"{stem}.png"))

    print("Inference finished.")


if __name__ == "__main__":
    main()
