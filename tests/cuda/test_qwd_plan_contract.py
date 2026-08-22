"""Private qWD plan factory, layout, and execution contracts."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG_FIELDS = (
    "accumulation_dtype",
    "collective",
    "compression",
    "dtype",
    "fp32_gathered_bytes",
    "global_numel",
    "group_size",
    "groups_per_shard",
    "output_bytes",
    "payload_bytes_per_rank",
    "qwd_gathered_payload_bytes",
    "rank",
    "shard_numel",
    "start",
    "valid_numel",
    "workspace_bytes",
    "world_size",
)


def qwd_config(
    *, global_numel: int = 67, rank: int = 0, world_size: int = 2
) -> dict[str, object]:
    shard_numel = (global_numel + world_size - 1) // world_size
    start = min(rank * shard_numel, global_numel)
    valid_numel = min(shard_numel, global_numel - start)
    groups = (shard_numel + 63) // 64
    payload = groups * 68
    gathered_payload = payload * world_size
    fp32_gathered = shard_numel * world_size * 4
    return {
        "accumulation_dtype": "fp32",
        "collective": "all_gather",
        "compression": "int8",
        "dtype": "fp16",
        "fp32_gathered_bytes": fp32_gathered,
        "global_numel": global_numel,
        "group_size": 64,
        "groups_per_shard": groups,
        "output_bytes": shard_numel * world_size * 2,
        "payload_bytes_per_rank": payload,
        "qwd_gathered_payload_bytes": gathered_payload,
        "rank": rank,
        "shard_numel": shard_numel,
        "start": start,
        "valid_numel": valid_numel,
        "workspace_bytes": max(payload + gathered_payload, fp32_gathered),
        "world_size": world_size,
    }


@pytest.fixture(scope="module")
def cuda_extension():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the qWD plan contract")
    return importlib.import_module("lowbit_comm._C")


def test_cuda_build_includes_qwd_plan_and_restore_sources() -> None:
    source = (ROOT / "setup_cuda.py").read_text(encoding="utf-8")
    assert 'CSRC_DIR / "executor" / "qwd_plan.cpp"' in source
    assert 'CSRC_DIR / "quantization" / "qwd_restore_kernel.cu"' in source


def test_qwd_plan_uses_one_collective_and_route_kernels() -> None:
    source = (ROOT / "csrc" / "executor" / "qwd_plan.cpp").read_text(
        encoding="utf-8"
    )
    assert source.count("try_inplace_quantize_parameter_delta(") == 1
    assert source.count("process_group_->_allgather_base(") == 2
    assert "process_group_->allgather(" not in source
    assert source.count("try_inplace_dequantize_gathered_add(") == 1
    assert source.count("try_inplace_qwd_refresh_cast(") == 1


def test_qwd_factory_is_private_and_has_no_python_capability_surface() -> None:
    pybind = (ROOT / "csrc" / "pybind.cpp").read_text(encoding="utf-8")
    qwd_source = (ROOT / "csrc" / "executor" / "qwd_plan.cpp").read_text(
        encoding="utf-8"
    )
    assert "bind_qwd_plan(m)" in pybind
    assert '"_create_qwd_plan"' in qwd_source
    assert '"create_qwd_plan"' not in qwd_source.replace(
        '"_create_qwd_plan"', ""
    )
    package_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "lowbit_comm").rglob("*.py")
    )
    assert "_create_qwd_plan" not in package_sources


def test_qwd_reserves_token_before_all_side_effects() -> None:
    source = (ROOT / "csrc" / "executor" / "qwd_plan.cpp").read_text(
        encoding="utf-8"
    )
    execute = source.split("std::shared_ptr<CudaWork> QWDPlan::execute(", 1)[1]
    token = execute.index("allocate_cuda_sequence")
    assert token < execute.index("mark_allocation")
    assert token < execute.index("execute_qwd")
    assert token < execute.index("execute_refresh")
    assert source.count("workspace_pool_->acquire") == 2


def test_extension_publishes_only_private_qwd_factory(cuda_extension) -> None:
    assert callable(cuda_extension._create_qwd_plan)
    assert not hasattr(cuda_extension, "create_qwd_plan")


@pytest.mark.parametrize("field", CONFIG_FIELDS)
def test_factory_requires_every_exact_config_field(
    cuda_extension, field: str
) -> None:
    config = qwd_config()
    del config[field]
    with pytest.raises(ValueError, match="config field|config fields"):
        cuda_extension._create_qwd_plan(config, object())


def test_factory_rejects_extra_config_field(cuda_extension) -> None:
    config = qwd_config()
    config["capability"] = True
    with pytest.raises(ValueError, match="config fields"):
        cuda_extension._create_qwd_plan(config, object())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accumulation_dtype", 1),
        ("collective", 1),
        ("compression", 1),
        ("dtype", 1),
        ("fp32_gathered_bytes", True),
        ("global_numel", True),
        ("group_size", True),
        ("groups_per_shard", 1.0),
        ("output_bytes", True),
        ("payload_bytes_per_rank", 1.0),
        ("qwd_gathered_payload_bytes", True),
        ("rank", True),
        ("shard_numel", 1.0),
        ("start", True),
        ("valid_numel", 1.0),
        ("workspace_bytes", True),
        ("world_size", True),
    ],
)
def test_factory_rejects_bool_as_int_and_other_inexact_scalars(
    cuda_extension, field: str, value: object
) -> None:
    config = qwd_config()
    config[field] = value
    with pytest.raises(ValueError, match=f"field must be (str|int): {field}"):
        cuda_extension._create_qwd_plan(config, object())


@pytest.mark.parametrize("world_size", (2, 4))
@pytest.mark.parametrize("global_numel", (0, 1, 67, 4097))
def test_factory_accepts_world_and_tail_layout_before_group_cast(
    cuda_extension, world_size: int, global_numel: int
) -> None:
    for rank in range(world_size):
        with pytest.raises(ValueError, match="c10d ProcessGroup"):
            cuda_extension._create_qwd_plan(
                qwd_config(
                    global_numel=global_numel,
                    rank=rank,
                    world_size=world_size,
                ),
                object(),
            )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("accumulation_dtype", "fp16", "accumulation"),
        ("collective", "native", "collective"),
        ("compression", "none", "compression"),
        ("dtype", "bf16", "dtype"),
        ("group_size", 32, "group size"),
        ("shard_numel", 35, "shard"),
        ("start", 1, "ownership"),
        ("valid_numel", 33, "ownership"),
        ("groups_per_shard", 2, "layout"),
        ("payload_bytes_per_rank", 67, "layout"),
        ("qwd_gathered_payload_bytes", 137, "layout"),
        ("fp32_gathered_bytes", 271, "layout"),
        ("output_bytes", 135, "output"),
        ("workspace_bytes", 203, "workspace"),
    ],
)
def test_factory_rejects_each_inconsistent_descriptor_before_group_cast(
    cuda_extension, field: str, value: object, message: str
) -> None:
    config = qwd_config()
    config[field] = value
    with pytest.raises(ValueError, match=message):
        cuda_extension._create_qwd_plan(config, object())


@pytest.mark.parametrize(("field", "value"), [("world_size", 3), ("rank", 2)])
def test_factory_rejects_invalid_world_or_rank(
    cuda_extension, field: str, value: int
) -> None:
    config = qwd_config()
    config[field] = value
    with pytest.raises(ValueError, match="world size|rank"):
        cuda_extension._create_qwd_plan(config, object())


def test_factory_rejects_signed_64_payload_overflow(cuda_extension) -> None:
    config = qwd_config()
    config["global_numel"] = (1 << 63) - 1
    config["shard_numel"] = 1 << 62
    config["valid_numel"] = 1 << 62
    with pytest.raises(ValueError, match="overflow"):
        cuda_extension._create_qwd_plan(config, object())


def test_factory_requires_process_group_nccl(cuda_extension) -> None:
    with pytest.raises(ValueError, match="c10d ProcessGroupNCCL"):
        cuda_extension._create_qwd_plan(qwd_config(), object())
