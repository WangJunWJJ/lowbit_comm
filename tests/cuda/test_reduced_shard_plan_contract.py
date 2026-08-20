"""Real CUDA ReducedShard plan factory and execution contracts."""

from __future__ import annotations

import importlib
from pathlib import Path
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cuda_extension():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the ReducedShard plan contract")
    return importlib.import_module("lowbit_comm._C")


@pytest.fixture(scope="module")
def nccl_group():
    torch = pytest.importorskip("torch")
    if torch.distributed.is_initialized():
        pytest.fail("ReducedShard contract requires an isolated ProcessGroup")
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = Path(directory, "rendezvous").resolve().as_uri()
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=rendezvous,
            rank=0,
            world_size=1,
        )
        try:
            yield torch.distributed.group.WORLD
        finally:
            torch.distributed.destroy_process_group()


def _native_config(
    *,
    numel: int = 10,
    rank: int = 0,
    world_size: int = 2,
    dtype: str = "fp16",
    reduction: str = "sum",
) -> dict[str, object]:
    logical_shard_length = (numel + world_size - 1) // world_size
    padded_input_numel = logical_shard_length * world_size
    offset = min(rank * logical_shard_length, numel)
    valid_length = min(logical_shard_length, numel - offset)
    return {
        "accumulation_dtype": "fp32",
        "collective": "native",
        "compression": "none",
        "dtype": dtype,
        "global_numel": numel,
        "group_size": None,
        "groups_per_shard": 0,
        "logical_shard_length": logical_shard_length,
        "numel": numel,
        "offset": offset,
        "output_bytes": logical_shard_length * 2,
        "output_numel": logical_shard_length,
        "payload_bytes_per_destination": 0,
        "rank": rank,
        "receive_payload_bytes": 0,
        "reduction": reduction,
        "send_payload_bytes": 0,
        "transport_shard_length": logical_shard_length,
        "valid_length": valid_length,
        "workspace_bytes": (
            padded_input_numel * 2 if padded_input_numel != numel else 0
        ),
        "world_size": world_size,
    }


def _int8_config() -> dict[str, object]:
    config = _native_config()
    config.update(
        {
            "collective": "compressed_reduce_scatter",
            "compression": "int8",
            "group_size": 16,
            "groups_per_shard": 1,
            "payload_bytes_per_destination": 18,
            "receive_payload_bytes": 36,
            "send_payload_bytes": 36,
            "transport_shard_length": 16,
            "workspace_bytes": 72,
        }
    )
    return config


def test_extension_publishes_reduced_shard_plan(cuda_extension) -> None:
    assert callable(cuda_extension.create_reduced_shard_plan)
    assert cuda_extension.ReducedShardPlan.__name__ == "ReducedShardPlan"


def test_cuda_build_includes_reduced_shard_plan_source() -> None:
    setup_source = (ROOT / "setup_cuda.py").read_text(encoding="utf-8")

    assert 'CSRC_DIR / "executor" / "reduced_shard_plan.cpp"' in setup_source


@pytest.mark.parametrize("field", tuple(_native_config()))
def test_factory_requires_every_exact_config_field(
    cuda_extension,
    field: str,
) -> None:
    config = _native_config()
    del config[field]

    with pytest.raises(ValueError, match="config field|config fields"):
        cuda_extension.create_reduced_shard_plan(config, object())


def test_factory_rejects_extra_config_field(cuda_extension) -> None:
    config = _native_config()
    config["layout"] = object()

    with pytest.raises(ValueError, match="config fields"):
        cuda_extension.create_reduced_shard_plan(config, object())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accumulation_dtype", 1),
        ("collective", 1),
        ("compression", 1),
        ("dtype", 1),
        ("reduction", 1),
        ("global_numel", True),
        ("groups_per_shard", 0.0),
        ("logical_shard_length", True),
        ("numel", 10.0),
        ("offset", True),
        ("output_bytes", 10.0),
        ("output_numel", True),
        ("payload_bytes_per_destination", 0.0),
        ("rank", True),
        ("receive_payload_bytes", 0.0),
        ("send_payload_bytes", 0.0),
        ("transport_shard_length", True),
        ("valid_length", 5.0),
        ("workspace_bytes", 0.0),
        ("world_size", True),
    ],
)
def test_factory_rejects_non_exact_scalar_types(
    cuda_extension,
    field: str,
    value: object,
) -> None:
    config = _native_config()
    config[field] = value

    with pytest.raises(ValueError, match=f"field must be (str|int): {field}"):
        cuda_extension.create_reduced_shard_plan(config, object())


def test_factory_requires_exact_optional_group_size_type(
    cuda_extension,
) -> None:
    config = _int8_config()
    config["group_size"] = True

    with pytest.raises(ValueError, match="group_size"):
        cuda_extension.create_reduced_shard_plan(config, object())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("accumulation_dtype", "fp16", "accumulation"),
        ("collective", "compressed_reduce_scatter", "descriptor"),
        ("compression", "fp8", "compression"),
        ("dtype", "fp32", "dtype"),
        ("reduction", "max", "reduction"),
        ("global_numel", 9, "global numel"),
        ("logical_shard_length", 4, "native layout"),
        ("transport_shard_length", 6, "native layout"),
        ("offset", 1, "ownership"),
        ("valid_length", 4, "ownership"),
        ("groups_per_shard", 1, "native layout"),
        ("payload_bytes_per_destination", 1, "native layout"),
        ("send_payload_bytes", 1, "native layout"),
        ("receive_payload_bytes", 1, "native layout"),
        ("output_numel", 4, "output"),
        ("output_bytes", 8, "output"),
        ("workspace_bytes", 1, "workspace"),
    ],
)
def test_factory_rejects_inconsistent_native_descriptor_before_group_cast(
    cuda_extension,
    field: str,
    value: object,
    message: str,
) -> None:
    config = _native_config()
    config[field] = value

    with pytest.raises(ValueError, match=message):
        cuda_extension.create_reduced_shard_plan(config, object())


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
        cuda_extension.create_reduced_shard_plan(config, object())


def test_factory_rejects_missing_process_group(cuda_extension) -> None:
    with pytest.raises(ValueError, match="requires a c10d ProcessGroup"):
        cuda_extension.create_reduced_shard_plan(_native_config(), None)


def test_factory_rejects_non_nccl_process_group(
    cuda_extension,
    nccl_group,
) -> None:
    torch = pytest.importorskip("torch")
    del nccl_group
    group = torch.distributed.new_group(ranks=[0], backend="gloo")

    with pytest.raises(ValueError, match="requires NCCL"):
        cuda_extension.create_reduced_shard_plan(_native_config(), group)


def test_factory_rejects_process_group_rank_mismatch(
    cuda_extension,
    nccl_group,
) -> None:
    config = _native_config(rank=1)

    with pytest.raises(ValueError, match="rank mismatch"):
        cuda_extension.create_reduced_shard_plan(config, nccl_group)


def test_factory_rejects_process_group_size_mismatch(
    cuda_extension,
    nccl_group,
) -> None:
    config = _native_config(world_size=2)

    with pytest.raises(ValueError, match="world size mismatch"):
        cuda_extension.create_reduced_shard_plan(config, nccl_group)


def test_int8_factory_validates_descriptor_before_group_cast(
    cuda_extension,
) -> None:
    with pytest.raises(ValueError, match="requires a c10d ProcessGroup"):
        cuda_extension.create_reduced_shard_plan(_int8_config(), object())
