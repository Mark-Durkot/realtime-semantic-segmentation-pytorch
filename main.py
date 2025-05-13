from core import SegTrainer
from utils.config import get_config

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # Get configuration with command line arguments
    config = get_config()

    # Initialize trainer
    trainer = SegTrainer(config)

    if config.task == 'train':
        trainer.run(config)
    elif config.task == 'val':
        trainer.validate(config)
    elif config.task == 'predict':
        trainer.predict(config)
    else:    
        raise ValueError(f'Unsupported task type: {config.task}.\n')