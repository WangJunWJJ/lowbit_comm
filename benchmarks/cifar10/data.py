from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)


def build_loaders(data_root: Path, config, rank: int, world_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )
    val_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
    )
    if rank == 0:
        train_set = datasets.CIFAR10(
            data_root, train=True, download=True, transform=train_transform
        )
        val_set = datasets.CIFAR10(
            data_root, train=False, download=True, transform=val_transform
        )
    dist.barrier()
    if rank != 0:
        train_set = datasets.CIFAR10(
            data_root, train=True, download=False, transform=train_transform
        )
        val_set = datasets.CIFAR10(
            data_root, train=False, download=False, transform=val_transform
        )
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed
    )
    val_sampler = DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False
    )
    common = dict(
        batch_size=config.batch_size_per_rank,
        num_workers=config.workers_per_rank,
        pin_memory=True,
        persistent_workers=config.workers_per_rank > 0,
    )
    return (
        DataLoader(train_set, sampler=train_sampler, **common),
        DataLoader(val_set, sampler=val_sampler, **common),
        train_sampler,
    )
