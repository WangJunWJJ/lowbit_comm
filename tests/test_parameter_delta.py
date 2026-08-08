from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from ccdl_comm.communication.parameter_delta import (
    ParameterCommunicationDecision,
    ParameterDeltaShard,
    SafeInt8QWDPolicy,
    TorchParameterDeltaProvider,
)


class FakeTensor:
    def __init__(
        self,
        values: tuple[float, ...],
        *,
        dtype: str = "fp32",
        device: str = "cuda:0",
        contiguous: bool = True,
    ) -> None:
        self.values = list(values)
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous

    def numel(self) -> int:
        return len(self.values)

    def is_contiguous(self) -> bool:
        return self._contiguous

    def copy_(self, other: "FakeTensor") -> "FakeTensor":
        self.values[:] = other.values
        return self

    def sub_(self, other: "FakeTensor") -> "FakeTensor":
        self.values[:] = [
            left - right
            for left, right in zip(self.values, other.values, strict=True)
        ]
        return self

    def narrow(self, dimension: int, start: int, length: int) -> "FakeTensorView":
        assert dimension == 0
        return FakeTensorView(self, start, length)


class FakeTensorView:
    def __init__(self, tensor: FakeTensor, start: int, length: int) -> None:
        self._tensor = tensor
        self._start = start
        self._length = length

    def zero_(self) -> "FakeTensorView":
        end = self._start + self._length
        self._tensor.values[self._start : end] = [0.0] * self._length
        return self


def test_delta_provider_computes_master_minus_model_and_zeros_padding() -> None:
    master = FakeTensor((1.125, 2.25, 99.0))
    model = FakeTensor((1.0, 2.0, 7.0), dtype="fp16")
    output = FakeTensor((0.0, 0.0, 0.0))

    result = TorchParameterDeltaProvider().prepare_delta(
        master,
        model,
        out=output,
        valid_numel=2,
    )

    assert result is output
    assert output.values == pytest.approx((0.125, 0.25, 0.0))


def test_delta_provider_accepts_real_torch_mixed_precision_tensors() -> None:
    torch = pytest.importorskip("torch")
    master = torch.tensor([1.125, 2.25, 99.0], dtype=torch.float32)
    model = torch.tensor([1.0, 2.0, 7.0], dtype=torch.float16)
    output = torch.empty_like(master)

    result = TorchParameterDeltaProvider().prepare_delta(
        master,
        model,
        out=output,
        valid_numel=2,
    )

    assert result is output
    torch.testing.assert_close(output, torch.tensor([0.125, 0.25, 0.0]))


def test_parameter_delta_shard_copies_metadata_and_is_immutable() -> None:
    metadata = {"step": 101}
    delta = ParameterDeltaShard(
        shard=FakeTensor((0.1, 0.2, 0.0)),
        shard_index=1,
        shard_numel=3,
        valid_numel=2,
        original_numel=5,
        padded_numel=6,
        world_size=2,
        layout_version=7,
        metadata=metadata,
    )
    metadata["step"] = 102

    assert delta.metadata == {"step": 101}
    assert isinstance(delta.metadata, MappingProxyType)
    with pytest.raises(FrozenInstanceError):
        delta.valid_numel = 3


@pytest.mark.parametrize(
    ("step", "role", "error", "capability", "mode", "reason"),
    (
        (1, "weight", None, True, "fp_refresh", "warmup"),
        (100, "weight", None, True, "fp_refresh", "warmup"),
        (101, "weight", None, True, "qwd", "int8_qwd"),
        (512, "weight", None, True, "fp_refresh", "periodic_refresh"),
        (129, "weight", 0.02, True, "fp_refresh", "error_threshold"),
        (129, "sensitive", None, True, "fp_refresh", "sensitive_tensor"),
        (129, "weight", None, False, "fp_refresh", "capability"),
    ),
)
def test_safe_policy_decisions(
    step: int,
    role: str,
    error: float | None,
    capability: bool,
    mode: str,
    reason: str,
) -> None:
    decision = SafeInt8QWDPolicy().decide(
        step=step,
        tensor_role=role,
        numel=4096,
        relative_error=error,
        capability=capability,
    )

    assert isinstance(decision, ParameterCommunicationDecision)
    assert decision.mode == mode
    assert decision.bit == 8
    assert decision.reason == reason


def test_policy_configuration_packet_is_stable_and_integer_only() -> None:
    first = SafeInt8QWDPolicy(
        warmup_steps=7,
        refresh_interval=31,
        relative_error_threshold=0.0125,
        error_check_interval=11,
    )
    second = SafeInt8QWDPolicy(
        warmup_steps=7,
        refresh_interval=31,
        relative_error_threshold=0.0125,
        error_check_interval=11,
    )

    assert first.configuration_packet() == second.configuration_packet()
    assert first.configuration_packet() == (7, 31, 12_500_000, 11)
    assert all(isinstance(value, int) for value in first.configuration_packet())


@pytest.mark.parametrize(
    "kwargs",
    (
        {"warmup_steps": -1},
        {"refresh_interval": 0},
        {"relative_error_threshold": 0.0},
        {"relative_error_threshold": float("nan")},
        {"error_check_interval": 0},
    ),
)
def test_safe_policy_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        SafeInt8QWDPolicy(**kwargs)


@pytest.mark.parametrize(
    "mutation",
    (
        {"step": 0},
        {"tensor_role": ""},
        {"numel": 0},
        {"relative_error": -0.1},
        {"relative_error": float("inf")},
        {"capability": 1},
    ),
)
def test_safe_policy_rejects_invalid_decision_input(
    mutation: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "step": 101,
        "tensor_role": "weight",
        "numel": 4096,
        "relative_error": None,
        "capability": True,
    }
    values.update(mutation)

    with pytest.raises((TypeError, ValueError)):
        SafeInt8QWDPolicy().decide(**values)
