from .base_config import BaseConfig


class MyConfig(BaseConfig):
    def __init__(self,):
        super().__init__()
        # Task
        self.task = 'train'

        # Dataset
        self.dataset = 'dubai'
        self.dataroot = 'dubai'
        self.num_class = 6

        # Model
        self.model = 'ppliteseg'

        # Training
        self.total_epoch = 200
        self.train_bs = 4
        self.loss_type = 'ohem'
        self.optimizer_type = 'adam'
        self.logger_name = 'seg_trainer'
        self.use_aux = False

        # Validating
        self.val_bs = 4

        # Testing
        self.test_bs = 4
        self.test_data_folder = '/path/to/your/test/folder'
        self.load_ckpt_path = '/path/to/your/inference/checkpoint'
        self.save_mask = True

        # Training setting
        self.use_ema = False
        self.base_workers = 2

        # Augmentation
        self.crop_size = 512
        self.randscale = [-0.5, 1.0]
        self.scale = 1.0
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.h_flip = 0.5

        # Knowledge Distillation
        self.kd_training = False
        self.teacher_ckpt = '/path/to/your/teacher/checkpoint'
        self.teacher_model = 'smp'
        self.teacher_encoder = 'resnet101'
        self.teacher_decoder = 'deeplabv3p'