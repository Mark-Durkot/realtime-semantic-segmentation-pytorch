from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR
from math import ceil


def get_scheduler(config, optimizer):
    if config.DDP:
        config.iters_per_epoch = ceil(config.train_num/config.train_bs/config.gpu_num)
    else:
        config.iters_per_epoch = ceil(config.train_num/config.train_bs)
    if getattr(config, 'max_iters', None) is not None:
        config.total_itrs = int(config.max_iters)
        config.total_epoch = ceil(config.total_itrs / config.iters_per_epoch)
    else:
        config.total_itrs = int(config.total_epoch*config.iters_per_epoch)

    if config.lr_policy == 'cos_warmup':
        warmup_ratio = config.warmup_epochs / config.total_epoch
        scheduler = OneCycleLR(optimizer, max_lr=config.lr, total_steps=config.total_itrs, 
                                pct_start=warmup_ratio)

    elif config.lr_policy == 'linear':
        scheduler = OneCycleLR(optimizer, max_lr=config.lr, total_steps=config.total_itrs, 
                                pct_start=0., anneal_strategy='linear')

    elif config.lr_policy == 'step':
        scheduler = StepLR(optimizer, step_size=config.step_size, gamma=0.1)

    elif config.lr_policy == 'poly':
        warmup_iters = getattr(config, 'warmup_iters', None)
        if warmup_iters is None:
            warmup_iters = int(config.warmup_epochs * config.iters_per_epoch)
        warmup_iters = max(0, min(int(warmup_iters), config.total_itrs - 1))
        poly_power = getattr(config, 'poly_power', 0.9)

        def lr_lambda(cur_iter: int) -> float:
            if warmup_iters > 0 and cur_iter < warmup_iters:
                return float(cur_iter + 1) / float(warmup_iters)

            progress = (cur_iter - warmup_iters) / max(1, (config.total_itrs - warmup_iters))
            progress = min(max(progress, 0.0), 1.0)
            return (1.0 - progress) ** poly_power

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    else:
        raise NotImplementedError(f'Unsupported scheduler type: {config.lr_policy}')
    return scheduler