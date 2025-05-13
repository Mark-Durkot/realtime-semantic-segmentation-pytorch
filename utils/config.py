import os
import torch
import argparse
from utils.logger import get_logger

def get_config():
    class Config:
        def __init__(self):
            # Basic settings
            self.task = 'train'  # 'train', 'val', or 'predict'
            self.dataroot = 'datasets/dubai'
            self.save_dir = 'checkpoints'
            self.load_ckpt = False
            self.load_ckpt_path = ''
            
            # Model settings
            self.model = 'bisenetv2'
            self.num_classes = 6
            
            # Training settings
            self.batch_size = 8
            self.num_workers = 4
            self.total_epoch = 100
            self.optimizer = 'adam'
            self.lr = 1e-4
            self.weight_decay = 1e-4
            self.momentum = 0.9
            self.scheduler = 'poly'
            self.power = 0.9
            
            # Data augmentation
            self.scale = 1.0
            self.randscale = 0.5
            self.crop_h = 512
            self.crop_w = 512
            self.brightness = 0.5
            self.contrast = 0.5
            self.saturation = 0.5
            self.h_flip = 0.5
            
            # Distributed training
            self.DDP = True
            self.gpu_num = torch.cuda.device_count()
            
            # Mixed precision training
            self.amp_training = True
            
            # Logging
            self.use_tb = True
            self.log_interval = 10
            self.save_ckpt = True
            self.save_interval = 1
            
            # Initialize dependent configs
            self.init_dependent_config()
            
        def init_dependent_config(self):
            # Calculate iterations per epoch
            self.iters_per_epoch = len(os.listdir(os.path.join(self.dataroot, 'Tile 1/images'))) // self.batch_size
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create save directory
            os.makedirs(self.save_dir, exist_ok=True)

    def parse_args():
        parser = argparse.ArgumentParser(description='Semantic Segmentation Training')
        parser.add_argument('--model', type=str, help='Model name to use (e.g., bisenetv2)')
        return parser.parse_args()

    # Create config instance
    config = Config()
    
    # Parse command line arguments
    args = parse_args()
    
    # Override config with command line arguments if provided
    if args.model:
        config.model = args.model
    
    return config 