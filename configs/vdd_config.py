from .base_config import BaseConfig


class VDDConfig(BaseConfig):
    def __init__(self,):
        super().__init__()
        # Task
        self.task = 'train'

        # Dataset
        self.dataset = 'vdd'
        self.dataroot = '/content/drive/MyDrive/VDD'
        self.num_class = 7

        # Model
        self.model = 'ppliteseg'

        # Training
        self.total_epoch = 200
        self.max_iters = 160000
        self.base_lr = 1e-3
        self.train_bs = 8
        self.loss_type = 'ohem'
        self.class_weights = [1.0, 1.2, 1.0, 1.0, 1.6, 1.2, 1.8]
        self.optimizer_type = 'adam'
        self.logger_name = 'seg_trainer'
        self.use_aux = False
        self.DDP = False
        self.gpu_num = 4

        # Validating
        self.val_bs = 8

        # Testing
        self.test_bs = 8
        self.test_data_folder = '/content/drive/MyDrive/VDD/test'
        self.load_ckpt = False
        self.resume_training = False
        self.save_dir = '/content/drive/MyDrive/VVD/results/save'
        self.load_ckpt_path = '/content/drive/MyDrive/VVD/results/save/vvd_best.pth'
        self.save_mask = True

        # Training setting
        self.use_ema = True
        self.base_workers = 2

        # Augmentation
        self.crop_size = 512
        # RandomScale scale_limit: offsets around 1; (-0.875, 0.5) -> factors in [0.125, 1.5]
        self.randscale = (0.125, 1.5)
        self.scale = 1.0
        self.brightness = 0.0
        self.contrast = 0.1
        self.saturation = 0.1
        self.h_flip = 0.5
        self.v_flip = 0.5

        # Knowledge Distillation
        self.kd_training = False
        self.teacher_ckpt = '/content/drive/MyDrive/VDD/checkpoints/smp_resnet101_deeplabv3p_vdd_best.pth'
        self.teacher_model = 'smp'
        self.teacher_encoder = 'resnet101'
        self.teacher_decoder = 'deeplabv3p'