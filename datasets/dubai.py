import os
import torch
from collections import namedtuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as AT
from albumentations.pytorch import ToTensorV2
import cv2
from patchify import patchify

from utils import transforms
from .dataset_registry import register_dataset


@register_dataset
class Dubai(Dataset):
    # Label definition for Dubai dataset
    Label = namedtuple('Label', [
        'name',        # The identifier of this label
        'id',          # An integer ID for the label
        'trainId',     # Training ID for the label
        'category',    # The category name
        'color',       # The color of this label
    ])

    # Labels based on the Dubai dataset classes with correct hex colors converted to RGB
    labels = [
        # name                id    trainId   category        color
        Label('building',      0,       0,    'building',     (60, 16, 152)),    # #3C1098
        Label('land',          1,       1,    'land',         (132, 41, 246)),   # #8429F6
        Label('road',          2,       2,    'road',         (110, 193, 228)),  # #6EC1E4
        Label('vegetation',    3,       3,    'vegetation',   (254, 221, 58)),   # #FEDD3A
        Label('water',         4,       4,    'water',        (226, 169, 41)),   # #E2A929
        Label('unlabeled',     5,       255,  'void',         (155, 155, 155)),  # #9B9B9B
    ]

    id_to_train_id = np.array([label.trainId for label in labels])
    patch_size = 256

    def __init__(self, config, mode='train'):
        data_root = os.path.expanduser(config.dataroot)
        print(f"Loading dataset from: {data_root}")
        
        if not os.path.isdir(data_root):
            raise RuntimeError(f'Data root directory: {data_root} does not exist.')

        if mode == 'train':
            self.transform = AT.Compose([
                transforms.Scale(scale=config.scale),
                AT.RandomScale(scale_limit=config.randscale),
                AT.PadIfNeeded(min_height=config.crop_h, min_width=config.crop_w, value=(114,114,114), mask_value=(0,0,0)),
                AT.RandomCrop(height=config.crop_h, width=config.crop_w),
                AT.ColorJitter(brightness=config.brightness, contrast=config.contrast, saturation=config.saturation),
                AT.HorizontalFlip(p=config.h_flip),
                AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),                
            ])
        elif mode == 'val':
            self.transform = AT.Compose([
                transforms.Scale(scale=config.scale),
                AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

        self.images = []
        self.masks = []

        # Process each tile folder
        for tile_num in range(1, 9):  # Tile 1 to Tile 8
            tile_folder = os.path.join(data_root, f'Tile {tile_num}')  # Note the space after 'Tile'
            print(f"Checking tile folder: {tile_folder}")
            
            if not os.path.isdir(tile_folder):
                print(f"Warning: Tile folder {tile_folder} does not exist")
                continue

            img_dir = os.path.join(tile_folder, 'images')
            msk_dir = os.path.join(tile_folder, 'masks')

            if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
                print(f"Warning: Missing images or masks directory in {tile_folder}")
                continue

            # Process images and create patches
            image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

            for file_name in image_files:
                img_path = os.path.join(img_dir, file_name)
                mask_path = os.path.join(msk_dir, file_name.replace('.jpg', '.png'))
                
                if not os.path.exists(mask_path):
                    print(f"Warning: Missing mask for {img_path}")
                    continue
                
                # Read and process image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue

                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Ensure image dimensions are divisible by patch_size
                h, w = image.shape[:2]
                new_h = (h // self.patch_size) * self.patch_size
                new_w = (w // self.patch_size) * self.patch_size
                image = image[:new_h, :new_w]
                
                # Create patches
                patches_img = patchify(image, (self.patch_size, self.patch_size, 3), step=self.patch_size)
                
                # Read and process mask
                mask = cv2.imread(mask_path)
                if mask is None:
                    print(f"Warning: Could not read mask {mask_path}")
                    continue

                # Convert BGR to RGB
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                mask = mask[:new_h, :new_w]
                
                # Create mask patches
                patches_mask = patchify(mask, (self.patch_size, self.patch_size, 3), step=self.patch_size)
                
                # Store patches
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        img_patch = patches_img[i,j,:,:][0]
                        mask_patch = patches_mask[i,j,:,:][0]
                        
                        self.images.append(img_patch)
                        self.masks.append(mask_patch)

        if len(self.images) == 0:
            raise RuntimeError("No valid images found in the dataset")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]

        # Convert RGB mask to label
        label_seg = np.zeros(mask.shape[:2], dtype=np.uint8)
        for i, label in enumerate(self.labels):
            label_seg[np.all(mask == label.color, axis=-1)] = label.trainId

        # Perform augmentation and normalization
        augmented = self.transform(image=image, mask=label_seg)
        image, mask = augmented['image'], augmented['mask']

        return image, mask

    @classmethod
    def encode_target(cls, mask):
        return cls.id_to_train_id[np.array(mask)]
