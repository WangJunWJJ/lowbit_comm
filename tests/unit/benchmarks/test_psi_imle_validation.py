import math

import pytest
import torch

from tests.benchmarks.psi_imle_validation import merge_imle_validation_shards, observe_imle_reduction
from tests.benchmarks.psi_unique_validation import unique_validation_indices


class FrozenModel:
    def rs_imle_loss(self, real, fake, epsilon=0.03):
        distances = torch.cdist(real.reshape(len(real), 1, -1), fake.reshape(len(real), fake.shape[1], -1)).squeeze(1)
        valid = (distances > float(epsilon)).float()
        maximum = distances.max()
        minimum, _ = (distances + (1 - valid) * maximum).min(dim=1)
        valid_real = (minimum < maximum).float()
        numerator = (minimum * valid_real).sum()
        denominator = valid_real.sum()
        loss = numerator / denominator.clamp_min(1.0)
        return torch.where(denominator > 0, loss, torch.zeros_like(loss))


def batch(real_values, fake_values):
    return torch.tensor(real_values, dtype=torch.float32).reshape(-1, 1, 1), torch.tensor(fake_values, dtype=torch.float32).reshape(len(real_values), 2, 1, 1)


def test_observation_uses_actual_valid_window_denominator_and_restores_method():
    model = FrozenModel()
    real, fake = batch([0.0, 0.0], [[1.0, 2.0], [0.0, 0.0]])
    with observe_imle_reduction(model) as facts:
        actual = model.rs_imle_loss(real, fake)
    assert actual.item() == pytest.approx(1.0)
    assert facts == [{"sample_count": 2, "valid_sample_count": 1, "numerator": 1.0, "scalar_loss": 1.0}]
    assert "rs_imle_loss" not in model.__dict__


def test_observation_restores_instance_override_after_error_and_calls_once():
    model = FrozenModel()
    calls = []
    old = model.rs_imle_loss

    def override(real, fake, epsilon=0.03):
        calls.append(1)
        return old(real, fake, epsilon) + 1

    model.rs_imle_loss = override
    with pytest.raises(ValueError):
        with observe_imle_reduction(model):
            model.rs_imle_loss(torch.ones(2, 1, 1), torch.zeros(2, 2, 1, 1))
    assert len(calls) == 1 and model.__dict__["rs_imle_loss"] is override


def test_observation_has_no_rng_side_effect_and_accepts_all_invalid_window():
    model = FrozenModel()
    real, fake = batch([0.0], [[0.0, 0.0]])
    old_randn = torch.randn
    torch.randn = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected RNG"))
    try:
        with observe_imle_reduction(model) as facts:
            value = model.rs_imle_loss(real, fake)
    finally:
        torch.randn = old_randn
    assert value.item() == 0 and facts[0]["valid_sample_count"] == 0


def test_observation_rejects_invalid_shapes_and_epsilon():
    model = FrozenModel()
    with observe_imle_reduction(model):
        with pytest.raises(ValueError):
            model.rs_imle_loss(torch.zeros(2, 1), torch.zeros(2, 2, 1, 1))
        with pytest.raises(ValueError):
            model.rs_imle_loss(torch.zeros(2, 1, 1), torch.zeros(2, 2, 1, 1), epsilon=float("nan"))


def shard(rank, dataset_size, numerators, valids, counts=None, world_size=1):
    indices = unique_validation_indices(dataset_size, rank, world_size)
    counts = ([len(indices)] if indices else []) if counts is None else counts
    facts = [{"sample_count": n, "valid_sample_count": v, "numerator": float(num), "scalar_loss": float(num / v) if v else 0.0} for n, v, num in zip(counts, valids, numerators)]
    return {"rank": rank, "sample_count": len(indices), "batch_counts": counts, "indices_sha256": __import__("hashlib").sha256(__import__("json").dumps(indices, separators=(",", ":")).encode()).hexdigest(), "weighted_loss_sum": float(sum(numerators)), "imle_batches": facts}


def test_merge_uses_true_numerators_and_reports_both_means():
    records = [shard(0, 3, [1.0, 3.0], [1, 1], [2, 1])]
    result = merge_imle_validation_shards(records, dataset_size=3, world_size=1)
    assert result["evaluation_protocol"] == "unique_windows_imle_valid_v2"
    assert result["sample_count"] == 3 and result["valid_sample_count"] == 2
    assert result["validation_loss"] == pytest.approx(2.0)
    assert result["zero_filled_all_window_loss"] == pytest.approx(4 / 3)
    assert result["invalid_sample_count"] == 1 and not result["qualification_eligible"]


@pytest.mark.parametrize("mutation", ["counts", "nonfinite", "empty"])
def test_merge_rejects_bad_reduction_facts(mutation):
    records = [shard(0, 2, [1.0], [1], [2])]
    assert merge_imle_validation_shards(records, dataset_size=2, world_size=1)["validation_loss"] == 1.0
    if mutation == "counts":
        records[0]["imle_batches"][0]["valid_sample_count"] = 3
    elif mutation == "nonfinite":
        records[0]["imle_batches"][0]["numerator"] = math.nan
    else:
        records[0]["imle_batches"][0]["valid_sample_count"] = 0
    with pytest.raises(ValueError, match="IMLE"):
        merge_imle_validation_shards(records, dataset_size=2, world_size=1)


@pytest.mark.parametrize("field, value", [("sample_count", True), ("scalar_loss", 9.0)])
def test_merge_rejects_tampered_fact_types_or_scalar(field, value):
    records = [shard(0, 2, [1.0], [1], [2])]
    records[0]["imle_batches"][0][field] = value
    with pytest.raises(ValueError, match="IMLE"):
        merge_imle_validation_shards(records, dataset_size=2, world_size=1)


def test_merge_rejects_zero_global_effective_denominator():
    records = [shard(0, 2, [0.0], [0], [2])]
    with pytest.raises(ValueError, match="zero"):
        merge_imle_validation_shards(records, dataset_size=2, world_size=1)


def test_merge_retains_local_empty_rank_and_rejects_duplicate_rank():
    from copy import deepcopy
    records = [shard(0, 2, [3.0], [1], world_size=3),
               shard(1, 2, [5.0], [1], world_size=3),
               shard(2, 2, [], [], counts=[], world_size=3)]
    result = merge_imle_validation_shards(records, dataset_size=2, world_size=3)
    assert result["sample_count"] == result["valid_sample_count"] == 2
    assert result["validation_loss"] == 4.0
    assert result["rank_execution"][2]["imle_batches"] == []
    records[2] = deepcopy(records[0])
    with pytest.raises(ValueError, match="duplicate"):
        merge_imle_validation_shards(records, dataset_size=2, world_size=3)


def test_merge_rejects_global_numerator_overflow_with_valid_rank_records():
    records = [shard(0, 2, [1e308], [1], world_size=2),
               shard(1, 2, [1e308], [1], world_size=2)]
    with pytest.raises(ValueError, match="overflow|invalid"):
        merge_imle_validation_shards(records, dataset_size=2, world_size=2)
