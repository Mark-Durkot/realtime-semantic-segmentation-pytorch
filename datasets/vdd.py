import os
from collections import namedtuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as AT
from albumentations.pytorch import ToTensorV2

from utils import transforms
from .dataset_registry import register_dataset


@register_dataset
class Vdd(Dataset):
    Label = namedtuple(
        "Label",
        [
            "name",
            "id",
            "trainId",
            "category",
            "color",
        ],
    )

    labels = [
        Label("other", 0, 0, "other", 0),
        Label("wall", 1, 1, "wall", 1),
        Label("road", 2, 2, "road", 2),
        Label("vegetation", 3, 3, "vegetation", 3),
        Label("vehicle", 4, 4, "vehicle", 4),
        Label("roof", 5, 5, "roof", 5),
        Label("water", 6, 6, "water", 6),
    ]
    color_to_train_id = {label.color: label.trainId for label in labels}

    def __init__(self, config, mode="train"):
        if mode not in ["train", "val"]:
            raise ValueError(f"Unsupported mode for Vdd dataset: {mode}")

        data_root = os.path.expanduser(config.dataroot)
        img_dir = os.path.join(data_root, mode, "src")
        msk_dir = os.path.join(data_root, mode, "gt")

        if not os.path.isdir(img_dir):
            raise RuntimeError(f"Image directory: {img_dir} does not exist.")
        if not os.path.isdir(msk_dir):
            raise RuntimeError(f"Mask directory: {msk_dir} does not exist.")

        if mode == "train":
            # Dataset is relatively small (~280 images), so use spatial augmentation only.
            self.transform = AT.Compose(
                [
                    transforms.Scale(scale=config.scale),
                    AT.RandomScale(scale_limit=config.randscale),
                    AT.PadIfNeeded(
                        min_height=config.crop_h,
                        min_width=config.crop_w,
                        value=(114, 114, 114),
                        mask_value=0,
                    ),
                    AT.RandomCrop(height=config.crop_h, width=config.crop_w),
                    AT.HorizontalFlip(p=config.h_flip),
                    # AT.VerticalFlip(p=config.v_flip),
                    # AT.RandomRotate90(p=0.5),
                    AT.ColorJitter(
                        brightness=config.brightness,
                        contrast=config.contrast,
                        saturation=config.saturation,
                    ),
                    AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = AT.Compose(
                [
                    transforms.Scale(scale=config.scale),
                    AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

        valid_img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.images = []
        self.masks = []

        for file_name in sorted(os.listdir(img_dir)):
            if not file_name.lower().endswith(valid_img_exts):
                continue

            img_path = os.path.join(img_dir, file_name)
            base = os.path.splitext(file_name)[0]

            mask_path = None
            for ext in valid_img_exts:
                candidate = os.path.join(msk_dir, f"{base}{ext}")
                if os.path.isfile(candidate):
                    mask_path = candidate
                    break

            if mask_path is None:
                raise RuntimeError(f"Could not find mask for image: {img_path}")

            self.images.append(img_path)
            self.masks.append(mask_path)

        if len(self.images) == 0:
            raise RuntimeError(f"No valid images found in {img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert("RGB"))
        mask = np.asarray(Image.open(self.masks[index]))

        mask = self.encode_target(mask)

        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]
        return image, mask

    @classmethod
    def encode_target(cls, mask):
        # Supports both grayscale label maps and RGB masks where classes are encoded in any channel.
        if mask.ndim == 3:
            mask = mask[..., 0]

        encoded = np.full(mask.shape, 255, dtype=np.uint8)
        for color_value, train_id in cls.color_to_train_id.items():
            encoded[mask == color_value] = train_id
        return encoded
