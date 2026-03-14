from .base_config import BaseConfig


class CityscapesConfig(BaseConfig):
    def __init__(self,):
        super().__init__()
        # Task
        self.task = 'train'

        # Dataset
        self.dataset = 'cityscapes'
        self.data_root = '/root/.cache/kagglehub/datasets/xiaose/cityscapes/versions/1/Cityscapes'
        # Keep both attributes for compatibility across datasets/loaders.
        self.dataroot = self.data_root
        self.num_class = 19
        self.ignore_index = 255

        # Model
        self.model = 'ppliteseg'

        # Training
        self.total_epoch = 200
        self.base_lr = 1e-3
        self.train_bs = 8
        self.loss_type = 'ohem'
        self.optimizer_type = 'adam'
        self.logger_name = 'seg_trainer'
        self.use_aux = False

        # Validating
        self.val_bs = 8

        # Testing
        self.test_bs = 8
        self.save_mask = True

        # Training setting
        self.use_ema = True
        self.base_workers = 4

        # Augmentation
        self.crop_size = 512
        self.randscale = [-0.25, 0.5]
        self.scale = 1.0
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.h_flip = 0.5
