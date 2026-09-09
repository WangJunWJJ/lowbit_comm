"""Regressions for fair PSI warmup and trustworthy timing/oracle checks."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.benchmarks import distributed_psi_v040_worker as worker
from tests.benchmarks.psi_v040_training import StepTiming, validate_step_record


class _WarmupModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.register_buffer("running", torch.tensor(7.0))
        self.forward_count = 0

    def forward(self, batch: torch.Tensor, *, training: bool) -> torch.Tensor:
        self.forward_count += 1
        self.running.add_(1.0)
        return (self.weight * batch).sum()


class _Telemetry:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset_after_warmup(self) -> None:
        self.reset_count += 1


def _rng_observation() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(3)


@pytest.mark.parametrize("route", ("native", "cag", "rsag_qwd"))
def test_all_routes_run_identical_state_preserving_model_warmup(route: str) -> None:
    random.seed(811)
    np.random.seed(811)
    torch.manual_seed(811)
    expected_rng = _rng_observation()
    random.seed(811)
    np.random.seed(811)
    torch.manual_seed(811)

    model = _WarmupModel()
    model.weight.grad = torch.tensor(13.0)
    saved_grad = model.weight.grad.clone()
    saved_buffer = model.running.clone()
    telemetry = _Telemetry() if route in {"native", "cag"} else None
    workspace = SimpleNamespace(
        _apply_train_augmentation=lambda batch: batch
        + random.random()
        + float(np.random.random())
        + torch.rand(())
    )

    worker._stabilize_ddp_bucket_layout(
        model, workspace, torch.tensor([3.0]), torch.device("cpu"), telemetry
    )

    assert model.forward_count == worker._DDP_BUCKET_WARMUP_BACKWARDS
    assert torch.equal(model.running, saved_buffer)
    assert torch.equal(model.weight.grad, saved_grad)
    actual_rng = _rng_observation()
    assert actual_rng[:2] == expected_rng[:2]
    assert torch.equal(actual_rng[2], expected_rng[2])
    if telemetry is not None:
        assert telemetry.reset_count == 1


def test_oracle_setup_is_read_only_for_force_refresh_schedule() -> None:
    helper = getattr(worker, "_prepare_resume_oracle_observation", None)
    assert helper is not None
    engine = SimpleNamespace(force_refresh=False)

    observation = helper(
        path=worker.Path("checkpoint.oracle.json"),
        next_batch_indices=(4, 8),
        learning_rate=1.0e-4,
        amp_scale=65536.0,
        optimizer_state_sha256="a" * 64,
        model_sha256="b" * 64,
    )

    assert observation[1:] == ((4, 8), 1.0e-4, 65536.0, "a" * 64, "b" * 64)
    assert engine.force_refresh is False


def test_step_timing_left_associative_measurement_validates_exactly() -> None:
    timing = StepTiming(
        forward_s=1.0,
        backward_s=2.0**-53,
        update_s=2.0**-53,
        communication_s=2.0**-53,
    )
    record = {
        "schema_version": 1,
        "task_id": "task",
        "attempt_id": "attempt",
        "route": "native",
        "seed": 20260821,
        "epoch": 0,
        "step": 0,
        "batch_indices": [0],
        "timing": timing.to_dict(),
        "communication": {
            "gradient_route": "ddp_nccl",
            "parameter_route": "full_adamw",
            "bytes": 0,
            "qwd_s": 0.0,
            "refresh_s": 0.0,
            "decision": "native",
        },
        "quality": {
            "loss": 1.0,
            "amp_scale": 1.0,
            "learning_rate": 1.0e-4,
            "model_sha256": "c" * 64,
            "rank_parameter_gap": 0.0,
            "optimizer_step": 0,
            "finite": True,
            "audit_performed": True,
        },
    }

    assert validate_step_record(record) == record
    corrupted = dict(record)
    corrupted["timing"] = dict(record["timing"])
    corrupted["timing"]["measured_s"] += 1.0e-6
    with pytest.raises(ValueError, match="measured_s is inconsistent"):
        validate_step_record(corrupted)
