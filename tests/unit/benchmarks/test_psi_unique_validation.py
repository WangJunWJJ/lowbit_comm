"""Unique-window coverage and sample-weighted validation regression tests."""
import math

import pytest

from tests.benchmarks import psi_unique_validation as validation


@pytest.mark.parametrize("size,world", [(2266, 2), (7, 3), (2, 8), (1, 1), (9, 4)])
def test_indices_cover_every_window_once(size, world):
    parts = [validation.unique_validation_indices(size, r, world) for r in range(world)]
    assert sorted(i for part in parts for i in part) == list(range(size))
    assert all(part == tuple(range(r, size, world)) for r, part in enumerate(parts))


@pytest.mark.parametrize("args", [(0, 0, 2), (-1, 0, 2), (2, -1, 2), (2, 2, 2),
                                   (2, 0, 0), (True, 0, 1), (2, False, 2), (2, 0, 2.0)])
def test_indices_reject_invalid_scope(args):
    with pytest.raises(ValueError):
        validation.unique_validation_indices(*args)


def _collate_samples(samples):
    import torch
    return torch.stack(samples).double()


def _records(size=7, world=3, batch=2):
    import torch
    source = torch.utils.data.DataLoader(
        torch.arange(size, dtype=torch.float32), batch_size=batch,
        drop_last=True, collate_fn=_collate_samples,
    )
    records = []
    for rank in range(world):
        loader = validation.build_unique_validation_loader(source, rank=rank, world_size=world, seed=42)
        records.append(validation.evaluate_validation_shard(
            loader, dataset_size=size, rank=rank, world_size=world,
            loss_fn=lambda batch: float(batch.square().mean()),
        ))
    return records


@pytest.mark.parametrize("size,world,batch", [(7, 3, 2), (2, 8, 3), (2266, 2, 256)])
def test_real_loaders_include_partial_batches_and_merge_sample_weighted(size, world, batch):
    shards = _records(size, world, batch)
    merged = validation.merge_validation_shards(shards[::-1], dataset_size=size, world_size=world)
    expected = sum(i*i for i in range(size))/size
    assert merged["validation_loss"] == pytest.approx(expected)
    assert merged["sample_count"] == size
    assert sum(s["sample_count"] for s in shards) == size
    assert merged["evaluation_protocol"] == "unique_windows_v1"
    assert merged["qualification_eligible"] is False
    assert all(sum(s["batch_counts"]) == s["sample_count"] for s in shards)
    if size == 2266:
        assert shards[0]["batch_counts"] == [256, 256, 256, 256, 109]
        assert shards[1]["batch_counts"] == [256, 256, 256, 256, 109]
    if world > size:
        assert all(s["weighted_loss_sum"] == 0 for s in shards[size:])


@pytest.mark.parametrize("indices", [(0, 0), (1, 2), (0, 2, 4), (0.0, 2)])
def test_actual_batch_indices_are_checked(indices):
    with pytest.raises(ValueError, match="indices"):
        validation.evaluate_validation_shard(
            [(indices, object())], dataset_size=4, rank=0, world_size=2, loss_fn=lambda _: 1.0,
        )


def test_missing_loader_tail_is_rejected():
    with pytest.raises(ValueError, match="coverage"):
        validation.evaluate_validation_shard(
            [((0,), [0])], dataset_size=5, rank=0, world_size=2, loss_fn=lambda _: 1.0,
        )


@pytest.mark.parametrize("loss", [math.nan, math.inf, -math.inf, True])
def test_nonfinite_or_boolean_batch_loss_is_rejected(loss):
    with pytest.raises(ValueError, match="loss"):
        validation.evaluate_validation_shard(
            [((0,), [0])], dataset_size=1, rank=0, world_size=1, loss_fn=lambda _: loss,
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "count", "hash", "nan", "batch_count"])
def test_merge_rejects_inconsistent_rank_evidence(mutation):
    shards = _records()
    if mutation == "missing":
        shards.pop()
    elif mutation == "duplicate":
        shards[1] = shards[0].copy()
    elif mutation == "count":
        shards[0]["sample_count"] += 1
    elif mutation == "hash":
        shards[0]["indices_sha256"] = "0"*64
    elif mutation == "nan":
        shards[0]["weighted_loss_sum"] = math.nan
    else:
        shards[0]["batch_counts"] = [True, 2]
    with pytest.raises(ValueError):
        validation.merge_validation_shards(shards, dataset_size=7, world_size=3)


def test_empty_global_dataset_is_not_zero_loss():
    with pytest.raises(ValueError):
        validation.merge_validation_shards([], dataset_size=0, world_size=1)


def test_batch_payload_count_must_match_observed_indices():
    with pytest.raises(ValueError, match="sample count"):
        validation.evaluate_validation_shard(
            [((0, 1), [5])], dataset_size=2, rank=0, world_size=1, loss_fn=lambda _: 5.0,
        )


def test_structured_batch_uses_explicit_sample_count_callback():
    record = validation.evaluate_validation_shard(
        [((0, 1), {"values": [3, 5]})], dataset_size=2, rank=0, world_size=1,
        loss_fn=lambda batch: sum(batch["values"])/2,
        sample_count_fn=lambda batch: len(batch["values"]),
    )
    assert record["weighted_loss_sum"] == 8.0


class _StochasticDataset:
    def __len__(self):
        return 7

    def __getitem__(self, index):
        import random
        import numpy as np
        import torch
        return torch.tensor([index, random.random(), float(np.random.rand()), float(torch.rand(()))])


def test_unique_loader_spawn_workers_preserve_indices_and_stochastic_samples():
    import torch
    source = torch.utils.data.DataLoader(_StochasticDataset(), batch_size=2)
    batches = []
    for workers in (0, 2):
        batches.append(list(validation.build_unique_validation_loader(
            source, rank=0, world_size=2, seed=42, epoch=2, num_workers=workers,
        )))
    assert len(batches[0]) == len(batches[1]) == 2
    for (idx0, tensor0), (idx1, tensor1) in zip(*batches):
        assert idx0 == idx1
        assert torch.equal(tensor0, tensor1)


def test_finite_batch_losses_overflow_is_a_validation_error():
    with pytest.raises(ValueError, match="loss sum"):
        validation.evaluate_validation_shard(
            [((0,), [1]), ((1,), [1])], dataset_size=2, rank=0, world_size=1,
            loss_fn=lambda _: 1e308,
        )


def test_finite_rank_loss_sum_overflow_is_a_validation_error():
    shards = [validation.evaluate_validation_shard(
        [((rank,), [1])], dataset_size=2, rank=rank, world_size=2,
        loss_fn=lambda _: 1e308,
    ) for rank in (0, 1)]
    with pytest.raises(ValueError, match="loss sum"):
        validation.merge_validation_shards(shards, dataset_size=2, world_size=2)
