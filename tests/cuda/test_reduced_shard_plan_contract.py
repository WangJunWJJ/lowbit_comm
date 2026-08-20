"""Real CUDA ReducedShard plan factory and execution contracts."""

from __future__ import annotations

import importlib
from pathlib import Path
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _layout(
    *,
    numel: int,
    world_size: int,
    group_size: int,
) -> tuple[int, int, int]:
    logical = (numel + world_size - 1) // world_size
    groups = (logical + group_size - 1) // group_size
    transport = groups * group_size
    payload_bytes = world_size * groups * (group_size + 2)
    return logical, transport, payload_bytes


def _reference_payload(
    torch,
    input_tensor,
    *,
    logical: int,
    transport: int,
    world_size: int,
    group_size: int,
):
    source = input_tensor.cpu()
    pieces = []
    groups = transport // group_size
    for destination in range(world_size):
        shard_start = destination * logical
        valid = max(0, min(logical, source.numel() - shard_start))
        shard = torch.zeros(transport, dtype=source.dtype)
        if valid:
            shard[:valid].copy_(source[shard_start : shard_start + valid])
        for group in range(groups):
            values = shard[
                group * group_size : (group + 1) * group_size
            ]
            scale = values.abs().max()
            if scale.item() == 0.0:
                quantized = torch.zeros(group_size, dtype=torch.int8)
            else:
                multiplier = torch.tensor(
                    127.0 / float(scale),
                    dtype=torch.float32,
                )
                quantized = (
                    values.float()
                    .mul(multiplier)
                    .round()
                    .clamp(-127, 127)
                    .to(torch.int8)
                )
            pieces.append(scale.reshape(1).view(torch.uint8))
            pieces.append(quantized.view(torch.uint8))
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.uint8)


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


def test_cuda_build_includes_shard_quantize_pack_source() -> None:
    setup_source = (ROOT / "setup_cuda.py").read_text(encoding="utf-8")

    assert (
        'CSRC_DIR / "quantization" / "shard_quant_pack_kernel.cu"'
        in setup_source
    )


@pytest.mark.parametrize("world_size", (2, 4))
@pytest.mark.parametrize("group_size", (16, 32, 64))
@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
def test_shard_quantize_pack_writes_exact_destination_payload(
    cuda_extension,
    world_size: int,
    group_size: int,
    dtype_name: str,
) -> None:
    torch = pytest.importorskip("torch")
    del cuda_extension
    dtype = getattr(torch, dtype_name)
    numel = world_size * (group_size + 3) - 5
    source = (
        torch.arange(numel, dtype=torch.int64, device="cuda")
        .mul(11)
        .remainder(37)
        .sub(18)
        .to(dtype)
    )
    logical, transport, payload_bytes = _layout(
        numel=numel,
        world_size=world_size,
        group_size=group_size,
    )
    packed = torch.full(
        (payload_bytes,),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )

    supported = torch.ops.lowbit_comm_private.shard_quantize_pack(
        source,
        packed,
        logical,
        transport,
        world_size,
        group_size,
    )

    assert supported is True
    expected = _reference_payload(
        torch,
        source,
        logical=logical,
        transport=transport,
        world_size=world_size,
        group_size=group_size,
    )
    assert torch.equal(packed.cpu(), expected)


def test_shard_quantize_pack_zeros_intra_shard_and_global_tail_padding(
    cuda_extension,
) -> None:
    torch = pytest.importorskip("torch")
    del cuda_extension
    world_size = 4
    group_size = 16
    source = torch.tensor([4.0, -2.0], dtype=torch.float16, device="cuda")
    logical, transport, payload_bytes = _layout(
        numel=source.numel(),
        world_size=world_size,
        group_size=group_size,
    )
    packed = torch.full(
        (payload_bytes,),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )

    torch.ops.lowbit_comm_private.shard_quantize_pack(
        source,
        packed,
        logical,
        transport,
        world_size,
        group_size,
    )

    chunks = packed.cpu().reshape(world_size, group_size + 2)
    assert chunks[0, :2].view(torch.float16).item() == 4.0
    assert chunks[1, :2].view(torch.float16).item() == 2.0
    assert chunks[2:, :].count_nonzero().item() == 0
    assert chunks[:, 3:].count_nonzero().item() == 0


def test_shard_quantize_pack_accepts_zero_numel(cuda_extension) -> None:
    torch = pytest.importorskip("torch")
    del cuda_extension
    source = torch.empty(0, dtype=torch.float16, device="cuda")
    packed = torch.empty(0, dtype=torch.uint8, device="cuda")

    assert torch.ops.lowbit_comm_private.shard_quantize_pack(
        source,
        packed,
        0,
        0,
        4,
        64,
    )


@pytest.mark.parametrize("dtype_name", ("float32", "int8"))
def test_shard_quantize_pack_rejects_unsupported_dtype(
    cuda_extension,
    dtype_name: str,
) -> None:
    torch = pytest.importorskip("torch")
    del cuda_extension
    source = torch.zeros(32, dtype=getattr(torch, dtype_name), device="cuda")
    packed = torch.empty(68, dtype=torch.uint8, device="cuda")

    assert not torch.ops.lowbit_comm_private.shard_quantize_pack(
        source,
        packed,
        16,
        16,
        2,
        16,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("input_layout", "input must be contiguous"),
        ("packed_dtype", "packed must have uint8 dtype"),
        ("packed_layout", "packed must be contiguous"),
        ("packed_shape", "packed must be one-dimensional"),
        ("logical", "logical shard length"),
        ("transport", "transport shard length"),
        ("packed_size", "packed payload size"),
        ("overflow", "overflow"),
    ],
)
def test_shard_quantize_pack_rejects_invalid_layout_before_launch(
    cuda_extension,
    mutation: str,
    message: str,
) -> None:
    torch = pytest.importorskip("torch")
    del cuda_extension
    source = torch.zeros(32, dtype=torch.float16, device="cuda")
    packed = torch.empty(68, dtype=torch.uint8, device="cuda")
    logical = 16
    transport = 16
    world_size = 2
    if mutation == "input_layout":
        source = torch.empty(
            (32, 2), dtype=torch.float16, device="cuda"
        )[:, 0]
    elif mutation == "packed_dtype":
        packed = packed.to(torch.int8)
    elif mutation == "packed_layout":
        packed = torch.empty(
            (68, 2), dtype=torch.uint8, device="cuda"
        )[:, 0]
    elif mutation == "packed_shape":
        packed = packed.reshape(2, 34)
    elif mutation == "logical":
        logical = 15
    elif mutation == "transport":
        transport = 32
    elif mutation == "packed_size":
        packed = packed[:-1]
    else:
        world_size = (1 << 63) - 1
        logical = 1

    with pytest.raises(RuntimeError, match=message):
        torch.ops.lowbit_comm_private.shard_quantize_pack(
            source,
            packed,
            logical,
            transport,
            world_size,
            16,
        )


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("collective", "native", "descriptor"),
        ("group_size", 32, "layout"),
        ("groups_per_shard", 2, "layout"),
        ("transport_shard_length", 32, "layout"),
        ("payload_bytes_per_destination", 19, "layout"),
        ("send_payload_bytes", 37, "layout"),
        ("receive_payload_bytes", 37, "layout"),
        ("workspace_bytes", 73, "layout"),
        ("output_numel", 6, "output"),
        ("output_bytes", 12, "output"),
        ("offset", 1, "ownership"),
        ("valid_length", 4, "ownership"),
    ],
)
def test_int8_factory_rejects_each_key_layout_mutation_before_group_cast(
    cuda_extension,
    field: str,
    value: object,
    message: str,
) -> None:
    config = _int8_config()
    config[field] = value

    with pytest.raises(ValueError, match=message):
        cuda_extension.create_reduced_shard_plan(config, object())


def test_factory_rejects_python_integer_outside_signed_int64(
    cuda_extension,
) -> None:
    config = _native_config()
    config["numel"] = 1 << 63

    with pytest.raises(ValueError, match="signed 64-bit"):
        cuda_extension.create_reduced_shard_plan(config, object())


def test_factory_rejects_native_logical_shard_int64_overflow(
    cuda_extension,
) -> None:
    config = _native_config(numel=(1 << 63) - 1)

    with pytest.raises(ValueError, match="logical shard size overflow"):
        cuda_extension.create_reduced_shard_plan(config, object())
