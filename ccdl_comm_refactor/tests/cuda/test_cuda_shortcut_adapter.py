from ccdl_comm.collectives.work import ImmediateWork
from ccdl_comm.config import CompressionConfig


class FakeTensor:
    shape = (8,)
    dtype = "torch.float16"
    device = "cuda:0"


class FakeCompiledPlan:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, tensor):
        self.calls.append(tensor)
        return ImmediateWork(self.result)


def test_all_reduce_default_shortcut_compiles_and_waits(monkeypatch) -> None:
    import ccdl_comm.collectives.all_reduce as module

    tensor = FakeTensor()
    compiled = FakeCompiledPlan("reduced")
    calls = []

    def fake_compile(tensor_arg, **kwargs):
        calls.append((tensor_arg, kwargs))
        return compiled

    monkeypatch.setattr(module, "_compile_cuda_shortcut", fake_compile)

    result = module.compressed_all_reduce(tensor, config=CompressionConfig(bit=8))

    assert result == "reduced"
    assert compiled.calls == [tensor]
    assert calls[0][1]["collective"] == "all_reduce"
    assert calls[0][1]["strategy"] == "all_gather"


def test_all_reduce_shortcut_returns_work_for_async_call(monkeypatch) -> None:
    import ccdl_comm.collectives.all_reduce as module

    tensor = FakeTensor()
    compiled = FakeCompiledPlan("reduced")
    monkeypatch.setattr(module, "_compile_cuda_shortcut", lambda *args, **kwargs: compiled)

    work = module.compressed_all_reduce(
        tensor,
        config=CompressionConfig(bit=8),
        async_op=True,
    )

    assert work.wait() == "reduced"


def test_all_reduce_injected_path_does_not_compile(monkeypatch) -> None:
    import ccdl_comm.collectives.all_reduce as module

    monkeypatch.setattr(
        module,
        "_compile_cuda_shortcut",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not compile")),
    )
    result = module.compressed_all_reduce(
        FakeTensor(),
        config=CompressionConfig(bit=8),
        quantize=lambda tensor, config: {"buffer": tensor},
        dequantize=lambda payload, shape, config, dtype: payload.buffer,
        all_reduce=lambda payload, op: payload,
        strategy="all_reduce",
        op="sum",
        world_size=1,
    )

    assert isinstance(result, FakeTensor)


def test_reduced_shard_default_shortcut_compiles(monkeypatch) -> None:
    import ccdl_comm.collectives.reduce_scatter as module

    tensor = FakeTensor()
    compiled = FakeCompiledPlan("shard")
    calls = []

    def fake_compile(tensor_arg, **kwargs):
        calls.append((tensor_arg, kwargs))
        return compiled

    monkeypatch.setattr(module, "_compile_cuda_shortcut", fake_compile)

    result = module.compressed_reduce_scatter_shard(
        tensor,
        config=CompressionConfig(bit=8),
    )

    assert result == "shard"
    assert calls[0][1]["collective"] == "reduce_scatter"
    assert calls[0][1]["output_layout"] == "shard"


def test_shortcut_can_reuse_caller_owned_compiled_plan(monkeypatch) -> None:
    import ccdl_comm.collectives.all_reduce as module

    tensor = FakeTensor()
    compiled = FakeCompiledPlan("reduced")
    monkeypatch.setattr(
        module,
        "_compile_cuda_shortcut",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not compile")),
    )

    result = module.compressed_all_reduce(
        tensor,
        config=CompressionConfig(bit=8),
        compiled_plan=compiled,
    )

    assert result == "reduced"
    assert compiled.calls == [tensor]


def test_cuda_shortcut_records_device_architecture_at_compile_time(monkeypatch) -> None:
    import ccdl_comm.cuda.shortcut as module

    captured = {}

    class FakeDist:
        @staticmethod
        def get_rank(group=None):
            return 0

        @staticmethod
        def get_world_size(group=None):
            return 2

    class FakeCuda:
        @staticmethod
        def get_device_name(device):
            return "NVIDIA RTX A6000"

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(
        module,
        "import_module",
        lambda name: FakeDist if name == "torch.distributed" else FakeTorch,
    )
    monkeypatch.setattr(module, "register_cuda_backends", lambda registry, extension_status: None)

    def fake_compile(plan, context, *, registry):
        captured["context"] = context
        return object()

    monkeypatch.setattr(module, "compile", fake_compile)

    module.compile_cuda_shortcut(
        FakeTensor(),
        collective="all_reduce",
        strategy="auto",
        output_layout="full",
        config=CompressionConfig(),
        async_op=False,
        dtype="fp16",
        extension_status=None,
    )

    assert captured["context"].device_architecture == "NVIDIA RTX A6000"
