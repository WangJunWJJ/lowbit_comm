from __future__ import annotations

import pytest

from ccdl_comm import CommunicationPlan, CompileContext
from ccdl_comm.work import ImmediateWork
from tests.conformance.backend_suite import (
    NATIVE_COLLECTIVES,
    assert_complete_native_protocol,
)


COLLECTIVES = NATIVE_COLLECTIVES


class FakeTensor:
    def __init__(self, value: int = 0) -> None:
        self.value = value
        self.shape = (1,)
        self.dtype = "torch.float16"
        self.device = "cuda:0"

    def new_empty(self, shape):
        assert tuple(shape) == self.shape
        return FakeTensor()


class FakeHandle:
    def __init__(self) -> None:
        self.wait_calls = 0

    def wait(self) -> None:
        self.wait_calls += 1

    def is_completed(self) -> bool:
        return self.wait_calls > 0


class FakeDist:
    class ReduceOp:
        SUM = "sum"
        AVG = "avg"
        PRODUCT = "product"
        MIN = "min"
        MAX = "max"

    def __init__(self) -> None:
        self.calls = []

    def _handle(self, async_op):
        return FakeHandle() if async_op else None

    def all_reduce(self, tensor, *, op, group, async_op):
        self.calls.append(("all_reduce", tensor, op, group, async_op))
        return self._handle(async_op)

    def all_gather(self, output_tensors, tensor, *, group, async_op):
        self.calls.append(
            ("all_gather", output_tensors, tensor, group, async_op)
        )
        return self._handle(async_op)

    def reduce_scatter(
        self,
        output,
        input_tensors,
        *,
        op,
        group,
        async_op,
    ):
        self.calls.append(
            (
                "reduce_scatter",
                output,
                input_tensors,
                op,
                group,
                async_op,
            )
        )
        return self._handle(async_op)

    def all_to_all(self, output_tensors, input_tensors, *, group, async_op):
        self.calls.append(
            ("all_to_all", output_tensors, input_tensors, group, async_op)
        )
        return self._handle(async_op)

    def broadcast(self, tensor, *, src, group, async_op):
        self.calls.append(("broadcast", tensor, src, group, async_op))
        return self._handle(async_op)

    def reduce(self, tensor, *, dst, op, group, async_op):
        self.calls.append(("reduce", tensor, dst, op, group, async_op))
        return self._handle(async_op)

    def gather(self, tensor, *, gather_list, dst, group, async_op):
        self.calls.append(
            ("gather", tensor, gather_list, dst, group, async_op)
        )
        return self._handle(async_op)

    def scatter(self, output, *, scatter_list, src, group, async_op):
        self.calls.append(
            ("scatter", output, scatter_list, src, group, async_op)
        )
        return self._handle(async_op)

    def barrier(self, *, group, async_op, device_ids):
        self.calls.append(("barrier", group, async_op, device_ids))
        return self._handle(async_op)


CONTEXT = CompileContext(
    rank=0,
    world_size=2,
    device="cuda:0",
    shape=(1,),
    dtype="fp16",
    process_group=object(),
)


def test_native_builder_matrix_covers_public_collectives() -> None:
    from ccdl_comm.cuda.native_collectives import NATIVE_BUILDERS

    assert tuple(NATIVE_BUILDERS) == COLLECTIVES


@pytest.mark.parametrize("collective", COLLECTIVES)
@pytest.mark.parametrize("async_op", [False, True])
def test_native_collective_preserves_work_and_execution_info(
    collective: str,
    async_op: bool,
) -> None:
    from ccdl_comm.cuda.native_collectives import (
        NativeCollectiveInput,
        compile_native_collective,
    )

    dist = FakeDist()
    plan = CommunicationPlan(
        collective,
        "native_nccl",
        async_op=async_op,
        root=0,
        reduce_op="sum",
    )
    executor = compile_native_collective(
        plan,
        CONTEXT,
        dist=dist,
    )
    tensor = FakeTensor(1)
    inputs = (FakeTensor(2), FakeTensor(3))
    outputs = (FakeTensor(), FakeTensor())
    invocation = NativeCollectiveInput(
        tensor=tensor,
        input_tensors=inputs,
        output_tensors=outputs,
    )

    work = executor.run(invocation)
    result = work.wait()

    assert dist.calls[0][0] == collective
    assert work.execution_info is executor.execution_info
    assert work.execution_info.executed_strategy == "native_nccl"
    expected_fast_path = (
        "cuda_native_nccl"
        if collective == "all_reduce"
        else f"cuda_native_nccl_{collective}"
    )
    assert work.execution_info.fast_path == expected_fast_path
    assert work.query() is True
    if async_op:
        assert executor.last_handle.wait_calls == 1
    if collective == "barrier":
        assert result is None


def test_native_collective_validates_root_at_compile_time() -> None:
    from ccdl_comm.cuda.native_collectives import compile_native_collective

    with pytest.raises(ValueError, match="root"):
        compile_native_collective(
            CommunicationPlan(
                "broadcast",
                "native_nccl",
                root=2,
            ),
            CONTEXT,
            dist=FakeDist(),
        )


def test_native_collective_rejects_unknown_reduce_operation() -> None:
    from ccdl_comm.cuda.native_collectives import compile_native_collective

    with pytest.raises(ValueError, match="reduce_op"):
        compile_native_collective(
            CommunicationPlan(
                "all_reduce",
                "native_nccl",
                reduce_op="median",
            ),
            CONTEXT,
            dist=FakeDist(),
        )


def test_cuda_backend_capabilities_cover_native_collective_matrix(
    monkeypatch,
) -> None:
    import ccdl_comm.cuda.backend as backend_module
    from ccdl_comm.cuda.backend import register_cuda_backends
    from ccdl_comm.cuda.loader import CudaExtensionStatus
    from ccdl_comm.registry import BackendKey, BackendRegistry

    class InitializedNccl:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_backend(group=None):
            return "nccl"

    monkeypatch.setattr(backend_module, "import_module", lambda name: InitializedNccl)
    registry = BackendRegistry()
    register_cuda_backends(
        registry,
        extension_status=CudaExtensionStatus(False, None, "not required"),
    )

    for collective in COLLECTIVES:
        key = BackendKey(collective, "native_nccl", "cuda", "full")
        assert key in registry
        capabilities = registry.resolve(key).capabilities(CONTEXT)
        assert capabilities.available is True
        assert capabilities.collectives == {collective}


def test_explicit_compressed_unsupported_does_not_silently_run_native(
    monkeypatch,
) -> None:
    import ccdl_comm.cuda.backend as backend_module
    from ccdl_comm import UnsupportedCollective
    from ccdl_comm.compiler import compile as compile_plan
    from ccdl_comm.cuda.backend import register_cuda_backends
    from ccdl_comm.cuda.loader import CudaExtensionStatus
    from ccdl_comm.registry import BackendRegistry

    class InitializedNccl:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_backend(group=None):
            return "nccl"

    monkeypatch.setattr(backend_module, "import_module", lambda name: InitializedNccl)
    registry = BackendRegistry()
    register_cuda_backends(
        registry,
        extension_status=CudaExtensionStatus(False, None, "not required"),
    )

    with pytest.raises(UnsupportedCollective, match="no registered backend"):
        compile_plan(
            CommunicationPlan("broadcast", "compressed"),
            CONTEXT,
            registry=registry,
        )


def test_explicit_fallback_is_reported_before_execution(monkeypatch) -> None:
    import ccdl_comm.cuda.backend as backend_module
    import ccdl_comm.cuda.compiler as compiler_module
    from ccdl_comm.compiler import compile as compile_plan
    from ccdl_comm.cuda.backend import register_cuda_backends
    from ccdl_comm.cuda.loader import CudaExtensionStatus
    from ccdl_comm.registry import BackendRegistry

    dist = FakeDist()
    monkeypatch.setattr(compiler_module, "import_module", lambda name: dist)

    class InitializedNccl:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_backend(group=None):
            return "nccl"

    monkeypatch.setattr(backend_module, "import_module", lambda name: InitializedNccl)
    registry = BackendRegistry()
    register_cuda_backends(
        registry,
        extension_status=CudaExtensionStatus(False, None, "not required"),
    )
    compiled = compile_plan(
        CommunicationPlan(
            "broadcast",
            "compressed",
            fallback=("native_nccl",),
            async_op=False,
        ),
        CONTEXT,
        registry=registry,
    )

    assert compiled.execution_info.fallback_used is True
    assert compiled.execution_info.executed_strategy == "native_nccl"
    assert "explicit fallback" in (compiled.execution_info.fallback_reason or "")


def test_public_collective_api_exports_complete_protocol() -> None:
    import ccdl_comm

    for collective in COLLECTIVES:
        assert callable(getattr(ccdl_comm, collective))
    assert callable(ccdl_comm.compile_collective)
    assert_complete_native_protocol(ccdl_comm.native_collectives())


def test_all_reduce_shortcut_can_reuse_compiled_plan(monkeypatch) -> None:
    import ccdl_comm.collectives.api as api

    class FakeCompiled:
        def __init__(self) -> None:
            self.calls = []

        def run(self, value):
            self.calls.append(value)
            return ImmediateWork(value)

    compiled = FakeCompiled()
    monkeypatch.setattr(
        api,
        "compile_collective",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not compile")
        ),
    )
    tensor = FakeTensor(7)

    work = api.all_reduce(tensor, compiled_plan=compiled)

    assert work.wait() is tensor
    assert compiled.calls == [tensor]


def test_list_collective_shortcuts_build_native_invocations(monkeypatch) -> None:
    import ccdl_comm.collectives.api as api
    from ccdl_comm.cuda.native_collectives import NativeCollectiveInput

    calls = []

    class FakeCompiled:
        def run(self, value):
            calls.append(value)
            return ImmediateWork(value)

    monkeypatch.setattr(
        api,
        "compile_collective",
        lambda *args, **kwargs: FakeCompiled(),
    )
    primary = FakeTensor(1)
    inputs = (FakeTensor(2), FakeTensor(3))
    outputs = (FakeTensor(), FakeTensor())

    all_gather_work = api.all_gather(
        primary,
        output_tensors=outputs,
    )
    api.reduce_scatter(
        primary,
        input_tensors=inputs,
    )
    api.all_to_all(
        output_tensors=outputs,
        input_tensors=inputs,
    )

    assert isinstance(all_gather_work.wait(), NativeCollectiveInput)
    assert calls[0].tensor is primary
    assert calls[0].output_tensors == outputs
    assert calls[1].tensor is primary
    assert calls[1].input_tensors == inputs
    assert calls[2].input_tensors == inputs
    assert calls[2].output_tensors == outputs


def test_root_collective_shortcuts_bind_root_during_compile(monkeypatch) -> None:
    import ccdl_comm.collectives.api as api

    compile_calls = []

    class FakeCompiled:
        def run(self, value):
            return ImmediateWork(value)

    def fake_compile(sample, **kwargs):
        compile_calls.append((sample, kwargs))
        return FakeCompiled()

    monkeypatch.setattr(api, "compile_collective", fake_compile)
    tensor = FakeTensor(1)

    api.broadcast(tensor, src=1)
    api.reduce(tensor, dst=1)
    api.gather(tensor, dst=1)
    api.scatter(tensor, scatter_list=(FakeTensor(), FakeTensor()), src=1)

    assert [item[1]["root"] for item in compile_calls] == [1, 1, 1, 1]


def test_context_from_tensor_binds_process_group_and_runtime_shape() -> None:
    from ccdl_comm.collectives.api import context_from_tensor

    class FakeRuntimeDist:
        @staticmethod
        def get_rank(group=None):
            return 1

        @staticmethod
        def get_world_size(group=None):
            return 4

    group = object()
    context = context_from_tensor(
        FakeTensor(),
        group=group,
        dist=FakeRuntimeDist,
        torch=None,
    )

    assert context.rank == 1
    assert context.world_size == 4
    assert context.process_group is group
    assert context.shape == (1,)
    assert context.dtype == "fp16"
