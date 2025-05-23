from torch.utils.data import DataLoader

from .cityscapes import Cityscapes
from .dubai import Dubai
from .dataset_registry import dataset_hub


def get_dataset(config):
    if config.dataset in dataset_hub.keys():
        train_dataset = dataset_hub[config.dataset](config=config, mode='train')
        val_dataset = dataset_hub[config.dataset](config=config, mode='val')
    else:
        raise NotImplementedError('Unsupported dataset!')

    return train_dataset, val_dataset


def get_loader(config, rank, pin_memory=True):
    train_dataset, val_dataset = get_dataset(config)

    # Make sure train number is divisible by batch size
    print(f"Total number of training samples: {len(train_dataset)}")
    config.train_num = len(train_dataset)
    print(f"Train number: {config.train_num}")
    print(f"Train batch size: {config.train_bs}")
    if config.train_num % config.train_bs != 0:
        config.train_num = (config.train_num // config.train_bs) * config.train_bs
    config.val_num = len(val_dataset)

    # For M1 Mac, we'll use single-GPU/CPU mode
    config.DDP = False
    config.gpu_num = 1

    train_loader = DataLoader(train_dataset, batch_size=config.train_bs, 
                                shuffle=True, num_workers=config.num_workers, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=config.val_bs, 
                                shuffle=False, num_workers=config.num_workers)

    return train_loader, val_loader


def get_test_loader(config): 
    from .test_dataset import TestDataset
    dataset = TestDataset(config)

    config.test_num = len(dataset)

    if config.DDP:
        raise NotImplementedError()

    else:
        test_loader = DataLoader(dataset, batch_size=config.test_bs, 
                                    shuffle=False, num_workers=config.num_workers)

    return test_loader


def list_available_datasets():
    dataset_list = list(dataset_hub.keys())

    return dataset_list