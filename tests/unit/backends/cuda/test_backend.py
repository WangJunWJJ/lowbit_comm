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
from lowbit_comm.api.result import FullTensorResult
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
    collective: CollectiveKind = (CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE),
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

    assert {(cap.min_world_size, cap.max_world_size) for cap in capabilities} == {
        (2, 2),
        (4, 4),
    }
    assert {cap.output for cap in capabilities} == {
        OutputSemantics.FULL_TENSOR,
        OutputSemantics.REDUCED_SHARD,
    }
    assert all(
        cap.supported_dtypes == frozenset({"fp16", "bf16"}) for cap in capabilities
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


def test_cuda_backend_declares_only_native_reduced_shard_capabilities() -> None:
    capabilities = CudaBackend().capabilities()
    reduced_shard = tuple(
        capability
        for capability in capabilities
        if capability.output is OutputSemantics.REDUCED_SHARD
    )

    assert {
        (capability.min_world_size, capability.max_world_size)
        for capability in reduced_shard
    } == {(2, 2), (4, 4)}
    assert {
        (
            capability.strategy.compression,
            capability.strategy.collective,
            capability.strategy.group_size,
        )
        for capability in reduced_shard
    } == {(CompressionKind.NONE, CollectiveKind.NATIVE, None)}


def test_lower_reduced_shard_int8_passes_exact_layout_to_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    class NativePlan:
        def execute(self, value: object) -> object:
            return value

    class Extension:
        def create_reduced_shard_plan(
            self,
            config: object,
            process_group: object,
        ) -> NativePlan:
            captured.extend((config, process_group))
            return NativePlan()

    group = object()
    monkeypatch.setattr(loader, "load_extension", Extension)

    plan = CudaBackend(group).lower(
        intent(output=OutputSemantics.REDUCED_SHARD, shape=(10,)),
        strategy(
            collective=CollectiveKind.COMPRESSED_REDUCE_SCATTER,
        ),
    )

    assert captured[1] is group
    assert type(captured[0]) is dict
    assert captured[0] == {
        "accumulation_dtype": "fp32",
        "collective": "compressed_reduce_scatter",
        "compression": "int8",
        "dtype": "fp16",
        "global_numel": 10,
        "group_size": 16,
        "groups_per_shard": 1,
        "logical_shard_length": 5,
        "numel": 10,
        "offset": 0,
        "output_bytes": 10,
        "output_numel": 5,
        "payload_bytes_per_destination": 18,
        "rank": 0,
        "receive_payload_bytes": 36,
        "reduction": "sum",
        "send_payload_bytes": 36,
        "transport_shard_length": 16,
        "valid_length": 5,
        "workspace_bytes": 72,
        "world_size": 2,
    }
    assert plan.metadata.global_shape == (10,)
    assert plan.metadata.offset == 0
    assert plan.metadata.valid_length == 5
    assert plan.metadata.padded_length == 5
    assert plan.metadata.owner_rank == 0


@pytest.mark.parametrize(
    "output",
    (OutputSemantics.FULL_TENSOR, OutputSemantics.REDUCED_SHARD),
)
def test_lower_gradient_feedback_passes_private_native_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    output: OutputSemantics,
) -> None:
    captured: list[dict[str, object]] = []

    class NativePlan:
        def execute(self, value: object) -> object:
            return value

    class Extension:
        def create_fulltensor_plan(
            self,
            config: dict[str, object],
            process_group: object,
        ) -> NativePlan:
            del process_group
            captured.append(config)
            return NativePlan()

        create_reduced_shard_plan = create_fulltensor_plan

    monkeypatch.setattr(loader, "load_extension", Extension)
    collective = (
        CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE
        if output is OutputSemantics.FULL_TENSOR
        else CollectiveKind.COMPRESSED_REDUCE_SCATTER
    )

    CudaBackend(object()).lower(
        intent(output=output),
        strategy(
            collective=collective,
            group_size=64,
            error_feedback=True,
        ),
    )

    assert captured[0]["gradient_error_feedback"] is True


@pytest.mark.parametrize(
    "output",
    (OutputSemantics.FULL_TENSOR, OutputSemantics.REDUCED_SHARD),
)
def test_lower_gradient_feedback_adapter_forwards_committed_residual(
    monkeypatch: pytest.MonkeyPatch,
    output: OutputSemantics,
) -> None:
    launches: list[tuple[object, object | None]] = []
    candidate = object()

    class NativeWork:
        def is_completed(self) -> bool:
            return False

        def wait(self) -> object:
            return "native-result"

        def _candidate_gradient_residual(self) -> object:
            return candidate

    class NativePlan:
        def execute(
            self,
            value: object,
            committed_residual: object | None,
        ) -> NativeWork:
            launches.append((value, committed_residual))
            return NativeWork()

    class Extension:
        def create_fulltensor_plan(
            self,
            config: dict[str, object],
            process_group: object,
        ) -> NativePlan:
            del config, process_group
            return NativePlan()

        create_reduced_shard_plan = create_fulltensor_plan

    monkeypatch.setattr(loader, "load_extension", Extension)
    collective = (
        CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE
        if output is OutputSemantics.FULL_TENSOR
        else CollectiveKind.COMPRESSED_REDUCE_SCATTER
    )
    plan = CudaBackend(object()).lower(
        intent(output=output),
        strategy(
            collective=collective,
            group_size=64,
            error_feedback=True,
        ),
    )

    plan.execute("gradient-1").wait()
    plan.execute("gradient-2").wait()

    assert launches == [
        ("gradient-1", None),
        ("gradient-2", candidate),
    ]


def test_cuda_backend_advertises_only_exact_phase2_world_sizes() -> None:
    capabilities = CudaBackend().capabilities()

    def supports_world_size(world_size: int) -> bool:
        return any(
            capability.min_world_size <= world_size
            and (
                capability.max_world_size is None
                or world_size <= capability.max_world_size
            )
            for capability in capabilities
        )

    assert supports_world_size(2)
    assert not supports_world_size(3)
    assert supports_world_size(4)


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
            "reduce-scatter",
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
        def create_fulltensor_plan(
            self,
            config: object,
            process_group: object,
        ) -> NativePlan:
            captured.append(config)
            captured.append(process_group)
            return NativePlan()

    monkeypatch.setattr(loader, "load_extension", Extension)
    requested_intent = intent()
    requested_strategy = strategy()

    process_group = object()
    plan = CudaBackend(process_group).lower(
        requested_intent,
        requested_strategy,
    )
    object.__setattr__(requested_intent.tensor, "dtype", "fp32")
    object.__setattr__(requested_strategy, "group_size", 64)

    assert plan.intent.tensor.dtype == "fp16"
    assert plan.strategy.group_size == 16
    assert plan.layout.group_size == 16
    assert captured[1] is process_group
    config = captured[0]
    assert type(config) is dict
    assert "layout" not in config
    assert config["logical_numel"] == 32
    assert config["payload_bytes_per_rank"] == 36
    assert config["gathered_payload_bytes"] == 72
    assert config["output_bytes"] == 64
    with pytest.raises(AttributeError):
        plan.layout = object()  # type: ignore[misc]


def test_plan_execute_is_limited_to_the_native_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = object()

    class NativeWork:
        def is_completed(self) -> bool:
            return True

        def wait(self) -> object:
            return expected

        def result(self) -> object:
            return expected

    class NativePlan:
        def execute(self, value: object) -> NativeWork:
            assert value == "input"
            return NativeWork()

    class Extension:
        def create_fulltensor_plan(
            self,
            config: object,
            process_group: object,
        ) -> NativePlan:
            del config
            assert process_group is group
            return NativePlan()

    monkeypatch.setattr(loader, "load_extension", Extension)

    group = object()
    work = CudaBackend(group).lower(intent(), strategy()).execute("input")

    result = work.wait()
    assert type(result) is FullTensorResult
    assert result.value is expected


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


def test_lower_requires_process_group_after_graph_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        loader,
        "load_extension",
        lambda: pytest.fail("extension must not be loaded"),
    )

    with pytest.raises(CompileError, match="ProcessGroup"):
        CudaBackend().lower(intent(), strategy())


def test_lower_passes_explicit_group_to_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = object()
    captured: list[object] = []

    class NativePlan:
        def execute(self, value: object) -> object:
            return value

    class Extension:
        def create_fulltensor_plan(
            self,
            config: object,
            process_group: object,
        ) -> NativePlan:
            captured.extend((config, process_group))
            return NativePlan()

    monkeypatch.setattr(loader, "load_extension", Extension)

    CudaBackend(group).lower(intent(), strategy())

    assert captured[1] is group
