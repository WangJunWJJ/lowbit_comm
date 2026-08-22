"""Real CUDA FullTensor plan factory and distributed contract tests."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cuda_extension():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the FullTensor plan contract")
    return importlib.import_module("lowbit_comm._C")


def test_extension_publishes_fulltensor_factory(cuda_extension) -> None:
    assert callable(cuda_extension.create_fulltensor_plan)


def test_extension_publishes_reduced_shard_factory(cuda_extension) -> None:
    assert callable(cuda_extension.create_reduced_shard_plan)


def test_cuda_build_includes_fulltensor_plan_source() -> None:
    setup_source = (ROOT / "setup_cuda.py").read_text(encoding="utf-8")

    assert 'CSRC_DIR / "executor" / "fulltensor_plan.cpp"' in setup_source


def _native_config() -> dict[str, object]:
    return {
        "accumulation_dtype": "fp32",
        "collective": "native",
        "compression": "none",
        "dtype": "fp16",
        "gathered_payload_bytes": 0,
        "group_count": 0,
        "group_size": None,
        "logical_numel": 32,
        "numel": 32,
        "output_bytes": 64,
        "padded_numel": 32,
        "payload_bytes_per_rank": 0,
        "rank": 0,
        "reduction": "sum",
        "workspace_bytes": 0,
        "world_size": 2,
    }


def _int8_gradient_feedback_config(
    *, group_size: int = 64
) -> dict[str, object]:
    numel = 32
    group_count = (numel + group_size - 1) // group_size
    payload_bytes = group_count * (group_size + 2)
    config = _native_config()
    config.update(
        {
            "collective": "compressed_all_gather_reduce",
            "compression": "int8",
            "gathered_payload_bytes": payload_bytes * 2,
            "gradient_error_feedback": True,
            "group_count": group_count,
            "group_size": group_size,
            "padded_numel": group_count * group_size,
            "payload_bytes_per_rank": payload_bytes,
            "workspace_bytes": payload_bytes * 3,
        }
    )
    return config


def test_fulltensor_private_descriptor_contains_gradient_feedback() -> None:
    source = (
        ROOT / "csrc" / "executor" / "fulltensor_plan.cpp"
    ).read_text(encoding="utf-8")

    assert '"gradient_error_feedback"' in source
    assert "group_size != 64" in source


def test_fulltensor_gradient_feedback_uses_one_fused_quant_launch() -> None:
    plan_source = (
        ROOT / "csrc" / "executor" / "fulltensor_plan.cpp"
    ).read_text(encoding="utf-8")
    quant_source = (
        ROOT / "csrc" / "quantization" / "quant_pack_kernel.cu"
    ).read_text(encoding="utf-8")

    assert plan_source.count(
        "inplace_quantize_pack_gradient_error_feedback("
    ) == 1
    assert "candidate_residual" in quant_source


def test_fulltensor_feedback_rejects_alias_before_token_and_side_effects() -> None:
    source = (
        ROOT / "csrc" / "executor" / "fulltensor_plan.cpp"
    ).read_text(encoding="utf-8")
    validation = source.split("void FullTensorPlan::validate_input(", 1)[1]
    validation = validation.split(
        "std::shared_ptr<CudaWork> FullTensorPlan::execute_native", 1
    )[0]
    execute = source.split(
        "std::shared_ptr<CudaWork> FullTensorPlan::execute(", 1
    )[1]

    assert "is_alias_of" in validation
    assert execute.index("validate_input") < execute.index(
        "allocate_cuda_sequence"
    )


def test_fulltensor_feedback_counts_candidate_allocation_after_token() -> None:
    source = (
        ROOT / "csrc" / "executor" / "fulltensor_plan.cpp"
    ).read_text(encoding="utf-8")
    execute = source.split(
        "std::shared_ptr<CudaWork> FullTensorPlan::execute(", 1
    )[1]
    int8 = source.split(
        "std::shared_ptr<CudaWork> FullTensorPlan::execute_int8(", 1
    )[1].split("void FullTensorPlan::exhaust_sequence_for_test", 1)[0]

    assert execute.index("allocate_cuda_sequence") < execute.index(
        "execute_int8"
    )
    assert int8.index("side_effects_.mark_allocation()") < int8.index(
        "torch::empty_like(input)"
    )


def test_fulltensor_factory_attaches_trusted_instance_execute() -> None:
    source = (
        ROOT / "csrc" / "executor" / "fulltensor_plan.cpp"
    ).read_text(encoding="utf-8")
    class_binding, factory_binding = source.split(
        'module.def(\n      "create_fulltensor_plan"', 1
    )

    assert 'module, "FullTensorPlan", py::dynamic_attr())' in class_binding
    assert 'result.attr("execute")' in factory_binding


def test_fulltensor_fused_gradient_feedback_matches_exact_launched_bytes(
    cuda_extension,
) -> None:
    torch = pytest.importorskip("torch")
    del cuda_extension
    group_size = 64
    gradient = (
        torch.arange(67, dtype=torch.float16, device="cuda")
        .remainder(19)
        .sub(9)
        .div(7)
    )
    previous = torch.linspace(
        -0.125,
        0.25,
        gradient.numel(),
        dtype=torch.float16,
        device="cuda",
    )
    groups = (gradient.numel() + group_size - 1) // group_size
    packed = torch.empty(
        groups * (group_size + 2),
        dtype=torch.uint8,
        device="cuda",
    )
    candidate = torch.empty_like(gradient)

    assert torch.ops.lowbit_comm_private.quantize_pack_gradient_error_feedback(
        gradient,
        packed,
        previous,
        candidate,
        group_size,
    )

    chunks = packed.cpu().reshape(groups, group_size + 2)
    raw = (
        chunks[:, :group_size]
        .reshape(groups, -1, 2)
        .flip(2)
        .flatten(1)
        .view(torch.int8)
        .float()
    )
    scales = (
        chunks[:, group_size:]
        .contiguous()
        .view(torch.float16)
        .float()
        .flatten()
    )
    reconstruction = (raw * scales[:, None] / 127.0).flatten()
    prepared = (gradient + previous).cpu()
    expected = (
        prepared.float() - reconstruction[: gradient.numel()]
    ).to(torch.float16)

    assert torch.equal(candidate.cpu(), expected)


def test_fulltensor_factory_accepts_private_group64_gradient_feedback(
    cuda_extension,
) -> None:
    with pytest.raises(ValueError, match="requires a c10d ProcessGroup"):
        cuda_extension.create_fulltensor_plan(
            _int8_gradient_feedback_config(), object()
        )


def test_fulltensor_factory_requires_exact_gradient_feedback_bool(
    cuda_extension,
) -> None:
    config = _int8_gradient_feedback_config()
    config["gradient_error_feedback"] = 1

    with pytest.raises(ValueError, match="must be bool"):
        cuda_extension.create_fulltensor_plan(config, object())


@pytest.mark.parametrize(
    "config",
    (
        {**_native_config(), "gradient_error_feedback": True},
        _int8_gradient_feedback_config(group_size=32),
    ),
)
def test_fulltensor_factory_rejects_gradient_feedback_outside_group64_int8(
    cuda_extension,
    config: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="gradient error feedback"):
        cuda_extension.create_fulltensor_plan(config, object())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("padded_numel", 33),
        ("group_count", 1),
        ("payload_bytes_per_rank", 1),
        ("gathered_payload_bytes", 1),
        ("output_bytes", 63),
    ],
)
def test_factory_rejects_inconsistent_native_layout_before_group_cast(
    cuda_extension,
    field: str,
    value: int,
) -> None:
    config = _native_config()
    config[field] = value

    with pytest.raises(ValueError, match="native layout"):
        cuda_extension.create_fulltensor_plan(config, object())


@pytest.mark.parametrize(
    ("field", "value"),
    [("world_size", 3), ("rank", 2)],
)
def test_factory_rejects_invalid_rank_domain_before_group_cast(
    cuda_extension,
    field: str,
    value: int,
) -> None:
    config = _native_config()
    config[field] = value

    with pytest.raises(ValueError, match="rank|world size"):
        cuda_extension.create_fulltensor_plan(config, object())
