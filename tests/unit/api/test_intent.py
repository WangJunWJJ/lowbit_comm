import pytest

from lowbit_comm.api.intent import (
    CommunicationIntent,
    CompletionMode,
    OutputSemantics,
    ReductionOp,
    ShapeFamily,
    TensorSpec,
)
from lowbit_comm.core.errors import CompileError


class UnhashableInt(int):
    __hash__ = None


class UnhashableStr(str):
    __hash__ = None


class UnhashableTuple(tuple[object, ...]):
    __hash__ = None


class UnhashableTensorSpec(TensorSpec):
    __hash__ = None


class UnhashableShapeFamily(ShapeFamily):
    __hash__ = None


def test_tensor_spec_reports_numel() -> None:
    tensor = TensorSpec(dtype="float16", shape=(2, 3, 4))

    assert tensor.numel == 24
    assert hash(tensor)


@pytest.mark.parametrize("shape", [(), (-1,), (2, -1)])
def test_tensor_spec_rejects_invalid_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(CompileError):
        TensorSpec(dtype="float16", shape=shape)


@pytest.mark.parametrize("shape", [[16], (True,), (1.5,)])
def test_tensor_spec_rejects_non_tuple_or_non_integer_shape(
    shape: object,
) -> None:
    with pytest.raises(CompileError):
        TensorSpec(dtype="float16", shape=shape)  # type: ignore[arg-type]


def test_shape_family_accepts_aligned_tensor_within_maximum() -> None:
    family = ShapeFamily(max_numel=32, alignment=4)

    assert family.accepts(TensorSpec(dtype="float16", shape=(2, 8)))
    assert not family.accepts(TensorSpec(dtype="float16", shape=(3, 6)))
    assert not family.accepts(TensorSpec(dtype="float16", shape=(8, 8)))


@pytest.mark.parametrize("alignment", [0, -1])
def test_shape_family_rejects_non_positive_alignment(alignment: int) -> None:
    with pytest.raises(CompileError):
        ShapeFamily(max_numel=16, alignment=alignment)


def test_intent_is_hashable_and_immutable() -> None:
    intent = CommunicationIntent(
        tensor=TensorSpec(dtype="float16", shape=(1024,)),
        shape_family=ShapeFamily(max_numel=2048, alignment=2),
        reduction=ReductionOp.MEAN,
        output=OutputSemantics.FULL_TENSOR,
        completion=CompletionMode.ASYNC,
        world_size=4,
        rank=1,
    )
    assert hash(intent)
    with pytest.raises(AttributeError):
        intent.rank = 2


@pytest.mark.parametrize("world_size,rank", [(0, 0), (4, -1), (4, 4)])
def test_intent_rejects_invalid_rank_domain(
    world_size: int,
    rank: int,
) -> None:
    with pytest.raises(CompileError):
        CommunicationIntent(
            tensor=TensorSpec(dtype="float16", shape=(16,)),
            shape_family=ShapeFamily(max_numel=16, alignment=1),
            reduction=ReductionOp.SUM,
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.SYNC,
            world_size=world_size,
            rank=rank,
        )


def test_intent_rejects_tensor_outside_shape_family() -> None:
    with pytest.raises(CompileError):
        CommunicationIntent(
            tensor=TensorSpec(dtype="float16", shape=(10,)),
            shape_family=ShapeFamily(max_numel=16, alignment=4),
            reduction=ReductionOp.SUM,
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.SYNC,
            world_size=1,
            rank=0,
        )


@pytest.mark.parametrize(
    "dtype,shape",
    [
        (1, (16,)),
        (UnhashableStr("float16"), (16,)),
        ("float16", UnhashableTuple((16,))),
        ("float16", (UnhashableInt(16),)),
    ],
)
def test_tensor_spec_rejects_non_exact_hashable_fields(
    dtype: object,
    shape: object,
) -> None:
    with pytest.raises(CompileError):
        TensorSpec(dtype=dtype, shape=shape)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "max_numel,alignment",
    [
        (True, 1),
        (UnhashableInt(16), 1),
        (16, False),
        (16, UnhashableInt(1)),
    ],
)
def test_shape_family_rejects_non_exact_hashable_numbers(
    max_numel: object,
    alignment: object,
) -> None:
    with pytest.raises(CompileError):
        ShapeFamily(
            max_numel=max_numel,  # type: ignore[arg-type]
            alignment=alignment,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "tensor",
            UnhashableTensorSpec(dtype="float16", shape=(16,)),
        ),
        (
            "shape_family",
            UnhashableShapeFamily(max_numel=16, alignment=1),
        ),
        ("reduction", "sum"),
        ("output", "full_tensor"),
        ("completion", "async"),
        ("world_size", True),
        ("rank", UnhashableInt(0)),
    ],
)
def test_intent_rejects_non_exact_contract_fields(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "tensor": TensorSpec(dtype="float16", shape=(16,)),
        "shape_family": ShapeFamily(max_numel=16, alignment=1),
        "reduction": ReductionOp.SUM,
        "output": OutputSemantics.FULL_TENSOR,
        "completion": CompletionMode.SYNC,
        "world_size": 1,
        "rank": 0,
    }
    values[field] = value

    with pytest.raises(CompileError):
        CommunicationIntent(**values)  # type: ignore[arg-type]
