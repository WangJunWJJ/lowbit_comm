import pytest

import lowbit_comm.backends.cuda.loader as loader
from lowbit_comm.api.intent import (
    CommunicationIntent,
    CompletionMode,
    OutputSemantics,
    ReductionOp,
    ShapeFamily,
    TensorSpec,
)
from lowbit_comm.api.policy import (
    AccumulationDType,
    CollectiveKind,
    CompressionKind,
    StrategySpec,
    TopologyKind,
)
from lowbit_comm.backends.cuda.backend import CudaBackend
from lowbit_comm.core.errors import CompileError


def intent(
    *,
    dtype: str = "fp16",
    output: OutputSemantics = OutputSemantics.FULL_TENSOR,
    world_size: int = 2,
    shape: tuple[int, ...] = (32,),
) -> CommunicationIntent:
    return CommunicationIntent(
        tensor=TensorSpec(dtype=dtype, shape=shape),
        shape_family=ShapeFamily(max_numel=1 << 62, alignment=1),
        reduction=ReductionOp.SUM,
        output=output,
        completion=CompletionMode.ASYNC,
        world_size=world_size,
        rank=0,
    )


def strategy(
    *,
    compression: CompressionKind = CompressionKind.INT8,
    collective: CollectiveKind = (
        CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE
    ),
    group_size: int | None = 16,
    **changes: object,
) -> StrategySpec:
    values: dict[str, object] = {
        "compression": compression,
        "collective": collective,
        "topology": TopologyKind.BACKEND_DEFAULT,
        "group_size": group_size,
    }
    values.update(changes)
    return StrategySpec(**values)  # type: ignore[arg-type]


def test_cuda_backend_declares_only_phase2_capabilities() -> None:
    capabilities = CudaBackend().capabilities()

    assert {cap.min_world_size for cap in capabilities} == {2}
    assert {cap.max_world_size for cap in capabilities} == {4}
    assert all(
        cap.output is OutputSemantics.FULL_TENSOR for cap in capabilities
    )
    assert all(
        cap.supported_dtypes == frozenset({"fp16", "bf16"})
        for cap in capabilities
    )
    assert {
        (
            cap.strategy.compression,
            cap.strategy.collective,
            cap.strategy.group_size,
        )
        for cap in capabilities
    } == {
        (CompressionKind.NONE, CollectiveKind.NATIVE, None),
        (
            CompressionKind.INT8,
            CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
            16,
        ),
        (
            CompressionKind.INT8,
            CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
            32,
        ),
        (
            CompressionKind.INT8,
            CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
            64,
        ),
    }


def test_lower_rejects_parameter_feedback_before_loading_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def mark_called() -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(loader, "load_extension", mark_called)

    with pytest.raises(CompileError, match="parameter error feedback"):
        CudaBackend().lower(
            intent(),
            strategy(parameter_error_feedback=True),
        )

    assert called is False


@pytest.mark.parametrize(
    ("requested_intent", "requested_strategy", "message"),
    [
        (
            intent(output=OutputSemantics.REDUCED_SHARD),
            strategy(),
            "full-tensor output",
        ),
        (intent(dtype="fp32"), strategy(), "dtype"),
        (intent(world_size=3), strategy(), "world size"),
        (intent(), strategy(topology=TopologyKind.RING), "topology"),
        (
            intent(),
            strategy(accumulation_dtype=AccumulationDType.FP16),
            "accumulation",
        ),
    ],
)
def test_lower_rejects_unsupported_graph_before_loading_extension(
    monkeypatch: pytest.MonkeyPatch,
    requested_intent: CommunicationIntent,
    requested_strategy: StrategySpec,
    message: str,
) -> None:
    monkeypatch.setattr(
        loader,
        "load_extension",
        lambda: pytest.fail("extension must not be loaded"),
    )

    with pytest.raises(CompileError, match=message):
        CudaBackend().lower(requested_intent, requested_strategy)


def test_lower_rejects_insufficient_workspace_before_loading_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        loader,
        "load_extension",
        lambda: pytest.fail("extension must not be loaded"),
    )

    with pytest.raises(CompileError, match="workspace"):
        CudaBackend().lower(intent(), strategy(workspace_budget_bytes=1))


def test_lower_snapshots_inputs_and_builds_an_immutable_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    class NativePlan:
        def execute(self, value: object) -> object:
            return value

    class Extension:
        def create_fulltensor_plan(self, config: object) -> NativePlan:
            captured.append(config)
            return NativePlan()

    monkeypatch.setattr(loader, "load_extension", Extension)
    requested_intent = intent()
    requested_strategy = strategy()

    plan = CudaBackend().lower(requested_intent, requested_strategy)
    object.__setattr__(requested_intent.tensor, "dtype", "fp32")
    object.__setattr__(requested_strategy, "group_size", 64)

    assert plan.intent.tensor.dtype == "fp16"
    assert plan.strategy.group_size == 16
    assert plan.layout.group_size == 16
    assert len(captured) == 1
    with pytest.raises(AttributeError):
        plan.layout = object()  # type: ignore[misc]


def test_plan_execute_is_limited_to_the_native_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = object()

    class NativePlan:
        def execute(self, value: object) -> object:
            assert value == "input"
            return expected

    class Extension:
        def create_fulltensor_plan(self, config: object) -> NativePlan:
            del config
            return NativePlan()

    monkeypatch.setattr(loader, "load_extension", Extension)

    work = CudaBackend().lower(intent(), strategy()).execute("input")

    assert work is expected


def test_lower_revalidates_a_forged_intent_without_loading_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        loader,
        "load_extension",
        lambda: pytest.fail("extension must not be loaded"),
    )
    request = intent()
    object.__setattr__(request.tensor, "dtype", "fp32")

    with pytest.raises(CompileError, match="unsupported"):
        CudaBackend().lower(request, strategy())
