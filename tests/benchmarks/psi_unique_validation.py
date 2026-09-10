"""Unpadded, auditable validation for offline checkpoint quality comparisons.

This module does not reinterpret legacy training losses or qualify a release.
The caller controls model mode, precision and RNG; batches carry actual dataset
indices so coverage is verified from observations, not merely a sampler length.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math


def unique_validation_indices(dataset_size: int, rank: int, world_size: int) -> tuple[int, ...]:
    """Visit every window once globally; permit empty ranks, not empty datasets."""
    if (type(dataset_size) is not int or dataset_size <= 0
            or type(world_size) is not int or world_size <= 0
            or type(rank) is not int or not 0 <= rank < world_size):
        raise ValueError("invalid unique validation scope")
    return tuple(range(rank, dataset_size, world_size))


def _indices_sha256(indices: tuple[int, ...]) -> str:
    return sha256(json.dumps(indices, separators=(",", ":")).encode("ascii")).hexdigest()


def _finite_loss_sum(values: list[float]) -> float:
    try:
        total = math.fsum(values)
    except OverflowError as error:
        raise ValueError("validation loss sum overflow") from error
    if not math.isfinite(total):
        raise ValueError("validation loss sum is not finite")
    return total


class _IndexedDataset:
    def __init__(self, dataset: object) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, object]:
        return index, self.dataset[index]


class _IndexedCollator:
    def __init__(self, collate: object) -> None:
        self.collate = collate

    def __call__(self, samples: list[tuple[int, object]]) -> tuple[tuple[int, ...], object]:
        return tuple(index for index, _ in samples), self.collate([value for _, value in samples])


def build_unique_validation_loader(
    source_loader: object, *, rank: int, world_size: int, seed: int,
    epoch: int = 0, num_workers: int = 0, prefetch_factor: int = 2,
) -> object:
    """Wrap the original dataset/collator, ignoring its padded sampler/drop_last."""
    from torch.utils.data import DataLoader
    from tests.benchmarks.psi_prefetch import build_prefetch_loader
    from tests.benchmarks.psi_training_runtime import data_loader_seed

    indices = unique_validation_indices(len(source_loader.dataset), rank, world_size)
    if source_loader.batch_size is None:
        raise ValueError("unique validation requires a batched source loader")
    template = DataLoader(
        _IndexedDataset(source_loader.dataset), batch_size=source_loader.batch_size,
        collate_fn=_IndexedCollator(source_loader.collate_fn),
        pin_memory=source_loader.pin_memory, drop_last=False, num_workers=0,
    )
    return build_prefetch_loader(
        template, indices, base_seed=data_loader_seed(seed, rank, 1), epoch=epoch,
        num_workers=num_workers, prefetch_factor=prefetch_factor,
    )


def evaluate_validation_shard(
    batches: object, *, dataset_size: int, rank: int, world_size: int,
    loss_fn: object, sample_count_fn: object = len,
) -> dict[str, object]:
    """Accumulate scalar batch-mean losses with observed, exact index coverage."""
    expected = unique_validation_indices(dataset_size, rank, world_size)
    offset, weighted_losses, counts = 0, [], []
    for indices, batch in batches:
        indices = tuple(indices)
        if (not indices or any(type(index) is not int for index in indices)
                or indices != expected[offset:offset + len(indices)]):
            raise ValueError("validation batch indices differ from unique plan")
        sample_count = sample_count_fn(batch)
        if type(sample_count) is not int or sample_count != len(indices):
            raise ValueError("validation batch sample count differs from its indices")
        loss = loss_fn(batch)
        if type(loss) not in (int, float) or not math.isfinite(loss):
            raise ValueError("validation batch loss must be finite numeric")
        weighted = float(loss) * len(indices)
        if not math.isfinite(weighted):
            raise ValueError("validation weighted loss overflow")
        weighted_losses.append(weighted)
        counts.append(len(indices))
        offset += len(indices)
    if offset != len(expected):
        raise ValueError("validation loader coverage is incomplete")
    return {
        "rank": rank, "sample_count": offset, "batch_counts": counts,
        "indices_sha256": _indices_sha256(expected),
        "weighted_loss_sum": _finite_loss_sum(weighted_losses),
    }


def merge_validation_shards(
    shards: object, *, dataset_size: int, world_size: int,
) -> dict[str, object]:
    """Validate all rank records before forming a global sample-weighted mean."""
    unique_validation_indices(dataset_size, 0, world_size)
    shards = list(shards)
    if len(shards) != world_size:
        raise ValueError("validation rank inventory is incomplete")
    ranks, sums = set(), []
    fields = {"rank", "sample_count", "batch_counts", "indices_sha256", "weighted_loss_sum"}
    for shard in shards:
        if type(shard) is not dict or set(shard) != fields:
            raise ValueError("validation rank record fields are invalid")
        rank = shard["rank"]
        expected = unique_validation_indices(dataset_size, rank, world_size)
        if rank in ranks:
            raise ValueError("duplicate validation rank")
        ranks.add(rank)
        if (type(shard["sample_count"]) is not int or shard["sample_count"] != len(expected)
                or shard["indices_sha256"] != _indices_sha256(expected)):
            raise ValueError("validation rank count or index hash mismatch")
        counts = shard["batch_counts"]
        if (type(counts) is not list or any(type(n) is not int or n <= 0 for n in counts)
                or sum(counts) != len(expected)):
            raise ValueError("validation rank batch counts mismatch")
        value = shard["weighted_loss_sum"]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or (not expected and value != 0)):
            raise ValueError("validation rank loss is invalid")
        sums.append(float(value))
    total = _finite_loss_sum(sums)
    return {
        "evaluation_protocol": "unique_windows_v1",
        "qualification_eligible": False,
        "sample_count": dataset_size, "validation_loss": total / dataset_size,
        "rank_execution": sorted(shards, key=lambda shard: shard["rank"]),
    }
