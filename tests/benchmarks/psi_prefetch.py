"""Deterministic, positional CPU data loading for the PSI benchmark.

The caller owns epoch/rank sharding and checkpoints *consumed* sample positions.
Dataset transforms must use Python, NumPy's global RNG or Torch's CPU RNG, not
private generators, CUDA, stateful counters or concurrent RNG-consuming threads.
Collation is preserved unchanged and is expected to be deterministic. Workers
are fresh spawned processes; neither their scheduling nor speculative fetching
changes sample seeds or the parent's augmentation stream.
"""

from __future__ import annotations

from hashlib import sha256
import random
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _sample_seed(base_seed: int, epoch: int, position: int) -> int:
    key = f"psi-positional-cpu-v1:{base_seed}:{epoch}:{position}"
    return int.from_bytes(sha256(key.encode("ascii")).digest()[:8], "big")


class PositionalSeedDataset(Dataset):
    """Map absolute local sampler positions to independently seeded samples."""

    def __init__(self, dataset: Dataset, indices: tuple[int, ...], *,
                 base_seed: int, epoch: int) -> None:
        _integer(base_seed, "base_seed")
        _integer(epoch, "epoch")
        if isinstance(dataset, IterableDataset):
            raise ValueError("a map-style dataset is required")
        for index in indices:
            _integer(index, "sample index")
            if index >= len(dataset):
                raise ValueError("sample index exceeds dataset length")
        self.dataset = dataset
        self.indices = tuple(indices)
        self.base_seed = base_seed
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        _integer(position, "sample position")
        index = self.indices[position]
        seed = _sample_seed(self.base_seed, self.epoch, position)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            # torch.manual_seed also seeds CUDA. Only touch the CPU generator.
            torch.random.default_generator.manual_seed(seed)
            return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


def build_prefetch_loader(
    source_loader: DataLoader,
    local_indices: Iterable[int],
    *,
    base_seed: int,
    epoch: int,
    start_position: int = 0,
    num_workers: int = 0,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Replay a rank-local epoch from its consumed sample offset.

    Pass the original loader template and the full epoch's materialized local
    indices, even on resume. Do not slice indices before passing start_position.
    The caller increments its checkpoint offset only after consuming a batch;
    DataLoader's internal sampler cursor can be ahead due to prefetch.
    ``prefetch_factor`` bounds queued batches per worker and is validated but
    unused in synchronous mode. No persistent worker or generator state needs
    to be checkpointed.
    """
    _integer(start_position, "start_position")
    _integer(num_workers, "num_workers")
    _integer(prefetch_factor, "prefetch_factor", 1)
    if source_loader.batch_size is None:
        raise ValueError("a batched source loader is required")
    indices = tuple(local_indices)
    if start_position > len(indices):
        raise ValueError("start_position exceeds local epoch length")
    dataset = PositionalSeedDataset(
        source_loader.dataset, indices, base_seed=base_seed, epoch=epoch,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_sample_seed(base_seed, epoch, 0))
    options = {}
    if num_workers:
        options.update(multiprocessing_context="spawn", prefetch_factor=prefetch_factor)
    return DataLoader(
        dataset,
        batch_size=source_loader.batch_size,
        sampler=range(start_position, len(indices)),
        num_workers=num_workers,
        collate_fn=source_loader.collate_fn,
        pin_memory=source_loader.pin_memory,
        drop_last=source_loader.drop_last,
        generator=generator,
        persistent_workers=False,
        timeout=0,
        **options,
    )
