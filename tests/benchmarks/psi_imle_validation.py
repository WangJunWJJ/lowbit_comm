"""Auditable observation and reduction of the frozen PSI RS-IMLE loss."""
from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterator


_FACT_FIELDS = {"sample_count", "valid_sample_count", "numerator", "scalar_loss"}
_SHARD_FIELDS = {"rank", "sample_count", "batch_counts", "indices_sha256", "weighted_loss_sum", "imle_batches"}


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _reconstruct(real_samples, fake_samples, epsilon):
    import torch

    if (not isinstance(real_samples, torch.Tensor) or not isinstance(fake_samples, torch.Tensor)
            or real_samples.ndim != 3 or fake_samples.ndim != 4
            or real_samples.shape[0] < 1 or real_samples.shape[1] < 1 or real_samples.shape[2] < 1
            or fake_samples.shape[0] != real_samples.shape[0]
            or fake_samples.shape[2:] != real_samples.shape[1:]
            or fake_samples.shape[1] < 1
            or not real_samples.is_floating_point() or not fake_samples.is_floating_point()
            or real_samples.device != fake_samples.device):
        raise ValueError("invalid IMLE reduction tensor shapes")
    if type(epsilon) not in (int, float) or not math.isfinite(float(epsilon)) or epsilon < 0:
        raise ValueError("IMLE epsilon must be a finite nonnegative number")
    batch_size = real_samples.shape[0]
    real_flat = real_samples.reshape(batch_size, 1, -1)
    fake_flat = fake_samples.reshape(batch_size, fake_samples.shape[1], -1)
    distances = torch.cdist(real_flat, fake_flat).squeeze(1)
    valid_samples = (distances > float(epsilon)).float()
    max_dist = distances.max()
    min_dist, _ = (distances + (1 - valid_samples) * max_dist).min(dim=1)
    valid_real = (min_dist < max_dist).float()
    numerator = (min_dist * valid_real).sum()
    denominator = valid_real.sum()
    loss = torch.where(denominator > 0, numerator / denominator.clamp_min(1.0), torch.zeros_like(numerator))
    return numerator, denominator, loss


@contextmanager
def observe_imle_reduction(model: object) -> Iterator[list[dict[str, float | int]]]:
    """Temporarily observe each bound ``rs_imle_loss`` call and restore it exactly."""
    import torch

    if not hasattr(model, "__dict__") or not callable(getattr(model, "rs_imle_loss", None)):
        raise ValueError("model must provide a callable rs_imle_loss method")
    had_instance = "rs_imle_loss" in vars(model)
    previous = vars(model).get("rs_imle_loss")
    original = getattr(model, "rs_imle_loss")
    facts: list[dict[str, float | int]] = []

    def wrapped(real_samples, fake_samples, epsilon=0.03):
        actual = original(real_samples, fake_samples, epsilon=epsilon)
        if not isinstance(actual, torch.Tensor) or actual.numel() != 1:
            raise ValueError("rs_imle_loss must return one scalar tensor")
        numerator, denominator, reconstructed = _reconstruct(real_samples, fake_samples, epsilon)
        if (not torch.isfinite(actual.detach()).all().item()
                or not torch.isfinite(numerator.detach()).all().item()
                or not torch.isfinite(reconstructed.detach()).all().item()):
            raise ValueError("IMLE reduction values must be finite")
        actual_value = float(actual.detach().item())
        reconstructed_value = float(reconstructed.detach().item())
        tolerance = 4.0 * max(float(torch.finfo(actual.dtype).eps), float(torch.finfo(reconstructed.dtype).eps))
        if abs(actual_value - reconstructed_value) > tolerance * max(1.0, abs(reconstructed_value)):
            raise ValueError("rs_imle_loss reduction is not the frozen IMLE reduction")
        facts.append({
            "sample_count": int(real_samples.shape[0]),
            "valid_sample_count": int(denominator.detach().item()),
            "numerator": float(numerator.detach().item()),
            "scalar_loss": actual_value,
        })
        return actual

    setattr(model, "rs_imle_loss", wrapped)
    try:
        yield facts
    finally:
        if had_instance:
            setattr(model, "rs_imle_loss", previous)
        else:
            del vars(model)["rs_imle_loss"]


def merge_imle_validation_shards(shards: object, *, dataset_size: int, world_size: int) -> dict[str, object]:
    """Validate unique coverage and aggregate observed IMLE numerators globally."""
    import math
    from tests.benchmarks.psi_unique_validation import merge_validation_shards

    shards = list(shards)
    stripped = []
    total_numerator, total_valid = [], 0
    enriched = []
    for shard in shards:
        if type(shard) is not dict or set(shard) != _SHARD_FIELDS or type(shard["imle_batches"]) is not list:
            raise ValueError("IMLE validation rank record fields are invalid")
        counts = shard["batch_counts"]
        batch_facts = shard["imle_batches"]
        if len(batch_facts) != len(counts):
            raise ValueError("IMLE batch facts do not match batch counts")
        numerators = []
        for count, fact in zip(counts, batch_facts):
            if type(fact) is not dict or set(fact) != _FACT_FIELDS:
                raise ValueError("IMLE batch fact fields are invalid")
            if (type(count) is not int or count <= 0 or type(fact["sample_count"]) is not int
                    or fact["sample_count"] != count):
                raise ValueError("IMLE batch sample count mismatch")
            valid = fact["valid_sample_count"]
            if type(valid) is not int or not 0 <= valid <= count:
                raise ValueError("IMLE valid sample count is invalid")
            if not _finite_number(fact["numerator"]) or not _finite_number(fact["scalar_loss"]):
                raise ValueError("IMLE batch values must be finite numbers")
            expected_scalar = 0.0 if valid == 0 else float(fact["numerator"]) / valid
            if (fact["numerator"] < 0 or fact["scalar_loss"] < 0
                    or not math.isclose(float(fact["scalar_loss"]), expected_scalar, rel_tol=1e-6, abs_tol=1e-7)
                    or (valid == 0 and (fact["numerator"] != 0 or fact["scalar_loss"] != 0))):
                raise ValueError("IMLE batch reduction values are invalid")
            numerators.append(float(fact["numerator"]))
            total_valid += valid
        weighted = shard["weighted_loss_sum"]
        try:
            numerator_sum = math.fsum(numerators)
        except (OverflowError, ValueError) as error:
            raise ValueError("IMLE numerator sum is invalid") from error
        if (not _finite_number(weighted) or not math.isfinite(numerator_sum)
                or not math.isclose(numerator_sum, float(weighted), rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError("IMLE numerator sum differs from shard loss sum")
        total_numerator.extend(numerators)
        stripped.append({key: value for key, value in shard.items() if key != "imle_batches"})
        enriched.append(shard)
    base = merge_validation_shards(stripped, dataset_size=dataset_size, world_size=world_size)
    try:
        total = math.fsum(total_numerator)
    except (OverflowError, ValueError) as error:
        raise ValueError("IMLE numerator sum is invalid") from error
    if not math.isfinite(total):
        raise ValueError("IMLE numerator sum is invalid")
    sample_count = base["sample_count"]
    if total_valid == 0:
        raise ValueError("global IMLE valid sample count is zero")
    return {
        "evaluation_protocol": "unique_windows_imle_valid_v2",
        "qualification_eligible": False,
        "sample_count": sample_count,
        "valid_sample_count": total_valid,
        "invalid_sample_count": sample_count - total_valid,
        "validation_loss": total / total_valid,
        "zero_filled_all_window_loss": total / sample_count,
        "rank_execution": sorted(enriched, key=lambda item: item["rank"]),
    }


__all__ = ["observe_imle_reduction", "merge_imle_validation_shards"]
