from .base_config import BaseConfig


class CityscapesConfig(BaseConfig):
    def __init__(self,):
        super().__init__()
        # Task
        self.task = 'train'

        # Dataset
        self.dataset = 'cityscapes'
        self.data_root = '/root/.cache/kagglehub/datasets/xiaose/cityscapes/versions/1/Cityspaces'
        # self.data_root = '/kaggle/input/cityscapes/Cityspaces'
        # Keep both attributes for compatibility across datasets/loaders.
        self.dataroot = self.data_root
        self.num_class = 19
        self.ignore_index = 255

        # Model
        self.model = 'ppliteseg'

        # Training
        self.total_epoch = 100
        self.max_iters = 160000
        self.base_lr = 0.005
        self.train_bs = 16
        self.loss_type = 'ohem'
        self.optimizer_type = 'sgd'
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.lr_policy = 'poly'
        self.poly_power = 0.9
        self.warmup_epochs = 5
        self.logger_name = 'seg_trainer'
        self.use_aux = False
        self.load_ckpt = True
        self.load_ckpt_path = '/content/drive/MyDrive/Cityscapes/results/save/best.pth'
        self.resume_training = True
        self.save_dir = '/content/drive/MyDrive/Cityscapes/results/save'

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
        # Albumentations RandomScale uses scale_limit with factor range [1+low, 1+high].
        # This corresponds to an effective random scale range of [0.125, 1.5].
        self.randscale = (0.125, 1.5)
        self.scale = 1.0
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.h_flip = 0.5
        self.v_flip = 0.0
