import pytest

from lowbit_comm.api.result import (
    FullTensorResult,
    ReducedShardMetadata,
    ReducedShardResult,
)
from lowbit_comm.core.errors import CompileError


class UnhashableInt(int):
    __hash__ = None


class UnhashableTuple(tuple[object, ...]):
    __hash__ = None


class ReducedShardMetadataSubclass(ReducedShardMetadata):
    """A metadata subtype that must not cross the exact result boundary."""


def _metadata(
    metadata_type: type[ReducedShardMetadata] = ReducedShardMetadata,
) -> ReducedShardMetadata:
    return metadata_type(
        global_shape=(10,),
        offset=4,
        valid_length=4,
        padded_length=4,
        owner_rank=1,
    )


def test_full_tensor_result_preserves_value() -> None:
    result = FullTensorResult(value=(1.0, 2.0))

    assert result.value == (1.0, 2.0)


def test_reduced_shard_metadata_defines_exact_ownership() -> None:
    metadata = _metadata()
    result = ReducedShardResult(value=(5.0, 6.0, 7.0, 8.0), metadata=metadata)

    assert result.metadata.stop == 8
    assert hash(metadata)


@pytest.mark.parametrize(
    "metadata",
    [object(), _metadata(ReducedShardMetadataSubclass)],
)
def test_reduced_shard_result_requires_exact_metadata(
    metadata: object,
) -> None:
    with pytest.raises(CompileError, match="metadata"):
        ReducedShardResult(
            value=object(),
            metadata=metadata,  # type: ignore[arg-type]
        )


def test_reduced_shard_result_preserves_arbitrary_value_object() -> None:
    value = object()

    result = ReducedShardResult(value=value, metadata=_metadata())

    assert result.value is value


@pytest.mark.parametrize(
    "offset,valid_length,padded_length,owner_rank",
    [
        (-1, 1, 1, 0),
        (0, -1, 1, 0),
        (0, 2, 1, 0),
        (0, 1, -1, 0),
        (0, 1, 1, -1),
    ],
)
def test_reduced_shard_metadata_rejects_invalid_fields(
    offset: int,
    valid_length: int,
    padded_length: int,
    owner_rank: int,
) -> None:
    with pytest.raises(CompileError):
        ReducedShardMetadata(
            global_shape=(10,),
            offset=offset,
            valid_length=valid_length,
            padded_length=padded_length,
            owner_rank=owner_rank,
        )


@pytest.mark.parametrize("global_shape", [[10], (True,), (1.5,)])
def test_reduced_shard_metadata_rejects_non_tuple_or_non_integer_shape(
    global_shape: object,
) -> None:
    with pytest.raises(CompileError):
        ReducedShardMetadata(
            global_shape=global_shape,  # type: ignore[arg-type]
            offset=0,
            valid_length=1,
            padded_length=1,
            owner_rank=0,
        )


def test_reduced_shard_rejects_range_past_global_numel() -> None:
    with pytest.raises(CompileError):
        ReducedShardMetadata(
            global_shape=(10,),
            offset=8,
            valid_length=4,
            padded_length=4,
            owner_rank=2,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("global_shape", UnhashableTuple((10,))),
        ("global_shape", (UnhashableInt(10),)),
        ("offset", True),
        ("valid_length", UnhashableInt(1)),
        ("padded_length", True),
        ("owner_rank", UnhashableInt(0)),
    ],
)
def test_reduced_shard_metadata_rejects_non_exact_hashable_fields(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "global_shape": (10,),
        "offset": 0,
        "valid_length": 1,
        "padded_length": 1,
        "owner_rank": 0,
    }
    values[field] = value

    with pytest.raises(CompileError):
        ReducedShardMetadata(**values)  # type: ignore[arg-type]
