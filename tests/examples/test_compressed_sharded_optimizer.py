from __future__ import annotations

import pytest

from examples.training.compressed_sharded_optimizer import (
    PIPELINE_STAGE_NAMES,
    TorchFlatParameterStorage,
    build_parser,
    run_fake_step,
)

torch = pytest.importorskip("torch")


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        self.bias = torch.nn.Parameter(torch.tensor([1.0, -1.0]))

    def forward(self, inputs):
        return inputs @ self.weight.transpose(0, 1) + self.bias


class MixedDtypeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.second = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))


def test_flat_storage_rebinds_parameters_to_one_padded_buffer() -> None:
    model = TinyModel()
    original = tuple(parameter.detach().clone() for parameter in model.parameters())

    storage = TorchFlatParameterStorage.from_parameters(
        model.parameters(),
        rank=0,
        world_size=4,
        group_size=64,
    )

    assert storage.layout.shard_numel % 512 == 0
    assert storage.padded_flat.numel() == storage.layout.padded_numel
    assert model.weight.data_ptr() == storage.padded_flat.data_ptr()
    assert model.bias.data_ptr() == (
        storage.padded_flat.data_ptr() + model.weight.numel() * model.weight.element_size()
    )
    torch.testing.assert_close(model.weight, original[0])
    torch.testing.assert_close(model.bias, original[1])

    storage.padded_flat[: storage.original_numel].add_(1)
    torch.testing.assert_close(model.weight, original[0] + 1)
    torch.testing.assert_close(model.bias, original[1] + 1)
    assert torch.count_nonzero(storage.padded_flat[storage.original_numel :]) == 0


def test_incompatible_parameter_storage_is_rejected_before_rebinding() -> None:
    model = MixedDtypeModel()
    before = tuple(parameter.data_ptr() for parameter in model.parameters())

    with pytest.raises(ValueError, match="same dtype"):
        TorchFlatParameterStorage.from_parameters(
            model.parameters(),
            rank=0,
            world_size=2,
        )

    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == before


def test_forward_backward_after_rebinding_preserves_gradient_shapes() -> None:
    model = TinyModel()
    storage = TorchFlatParameterStorage.from_parameters(
        model.parameters(),
        rank=1,
        world_size=2,
    )

    loss = model(torch.ones(4, 3)).square().mean()
    loss.backward()

    assert storage.local_shard.numel() == storage.layout.shard_numel
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.grad.shape == parameter.shape
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    ("rank", "world_size", "group_size"),
    ((True, 2, 64), (0, 0, 64), (2, 2, 64), (0, 2, 0)),
)
def test_invalid_layout_arguments_are_rejected(
    rank: object,
    world_size: object,
    group_size: object,
) -> None:
    model = TinyModel()

    with pytest.raises((TypeError, ValueError)):
        TorchFlatParameterStorage.from_parameters(
            model.parameters(),
            rank=rank,
            world_size=world_size,
            group_size=group_size,
        )


@pytest.mark.parametrize(
    "mode",
    (
        "native_ddp",
        "full_fused",
        "sharded_fp",
        "sharded_compressed",
        "sharded_qwd",
    ),
)
def test_parser_exposes_comparable_modes(mode: str) -> None:
    assert build_parser().parse_args(["--mode", mode]).mode == mode


def test_metrics_report_every_pipeline_stage() -> None:
    metrics = run_fake_step(mode="sharded_compressed")

    assert set(metrics["stage_ms"]) == set(PIPELINE_STAGE_NAMES)
    assert set(metrics["stage_ms"]) == {
        "backward_flatten",
        "compressed_reduce_scatter",
        "local_update",
        "parameter_quantize_gather",
        "parameter_restore_writeback",
    }
    assert metrics["selected_fast_path"] == "compressed_parameter_restore"


def test_qwd_fake_metrics_expose_parameter_communication_contract() -> None:
    metrics = run_fake_step(mode="sharded_qwd")

    assert metrics["selected_fast_path"] == "fused_int8_qwd"
    assert metrics["parameter_communication"] == {
        "algorithm": "qwd",
        "bit": 8,
        "warmup_steps": 0,
        "refresh_interval": 512,
        "relative_error_threshold": 1.0e-2,
        "decision_counts": {"qwd": 1, "fp_refresh": 0},
        "sampled_relative_errors": [],
    }
