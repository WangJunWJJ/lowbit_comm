import pytest

from lowbit_comm.backends.cuda.layout import (
    FullTensorLayout,
    build_fulltensor_layout,
)
from lowbit_comm.api.policy import CompressionKind
from lowbit_comm.core.errors import CompileError


@pytest.mark.parametrize(
    "numel,group_size,padded",
    [(1, 16, 16), (16, 16, 16), (17, 16, 32)],
)
def test_layout_pads_only_to_group_boundary(
    numel: int,
    group_size: int,
    padded: int,
) -> None:
    layout = build_fulltensor_layout(
        numel=numel,
        dtype="fp16",
        world_size=4,
        compression=CompressionKind.INT8,
        group_size=group_size,
    )

    assert layout.padded_numel == padded
    assert layout.group_count == padded // group_size
    assert layout.payload_bytes_per_rank == padded + 2 * layout.group_count


def test_layout_accounts_for_gather_output_and_workspace_bytes() -> None:
    layout = build_fulltensor_layout(
        numel=17,
        dtype="bf16",
        world_size=2,
        compression=CompressionKind.INT8,
        group_size=16,
    )

    assert layout == FullTensorLayout(
        logical_numel=17,
        padded_numel=32,
        group_size=16,
        group_count=2,
        payload_bytes_per_rank=36,
        gathered_payload_bytes=72,
        output_bytes=34,
        workspace_bytes=108,
    )


def test_layout_supports_zero_numel_without_a_phantom_group() -> None:
    layout = build_fulltensor_layout(
        numel=0,
        dtype="fp16",
        world_size=2,
        compression=CompressionKind.INT8,
        group_size=64,
    )

    assert layout.padded_numel == 0
    assert layout.group_count == 0
    assert layout.payload_bytes_per_rank == 0
    assert layout.gathered_payload_bytes == 0
    assert layout.output_bytes == 0
    assert layout.workspace_bytes == 0


def test_layout_is_immutable() -> None:
    layout = build_fulltensor_layout(
        numel=16,
        dtype="fp16",
        world_size=4,
        compression=CompressionKind.INT8,
        group_size=16,
    )

    with pytest.raises(AttributeError):
        layout.logical_numel = 32  # type: ignore[misc]


def test_layout_rejects_unregistered_domain_values() -> None:
    with pytest.raises(CompileError, match="world size"):
        build_fulltensor_layout(
            numel=16,
            dtype="fp16",
            world_size=8,
            compression=CompressionKind.INT8,
            group_size=16,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"numel": -1}, "numel"),
        ({"numel": True}, "numel"),
        ({"dtype": "fp32"}, "dtype"),
        ({"world_size": 1}, "world size"),
        ({"group_size": 8}, "group size"),
    ],
)
def test_layout_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "numel": 16,
        "dtype": "fp16",
        "world_size": 2,
        "compression": CompressionKind.INT8,
        "group_size": 16,
    }
    values.update(kwargs)

    with pytest.raises(CompileError, match=message):
        build_fulltensor_layout(**values)  # type: ignore[arg-type]


def test_layout_rejects_byte_count_overflow() -> None:
    with pytest.raises(CompileError, match="overflow"):
        build_fulltensor_layout(
            numel=1 << 63,
            dtype="fp16",
            world_size=4,
            compression=CompressionKind.INT8,
            group_size=16,
        )


def test_fp16_int8_layout_matches_compact_pack_format() -> None:
    layout = build_fulltensor_layout(
        numel=33,
        dtype="fp16",
        world_size=2,
        compression=CompressionKind.INT8,
        group_size=16,
    )

    assert layout.payload_bytes_per_rank == 3 * (16 + 2)
    assert layout.gathered_payload_bytes == 2 * 3 * (16 + 2)


def test_native_layout_has_no_quantized_workspace() -> None:
    layout = build_fulltensor_layout(
        numel=33,
        dtype="fp16",
        world_size=2,
        compression=CompressionKind.NONE,
        group_size=None,
    )

    assert layout.payload_bytes_per_rank == 0
    assert layout.gathered_payload_bytes == 0
    assert layout.workspace_bytes == 0
