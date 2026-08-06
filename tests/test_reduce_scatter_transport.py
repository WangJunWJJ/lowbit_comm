import pytest

from ccdl_comm.config import CompressionConfig


class FakeTensor:
    def __init__(self, values, dtype="torch.float32", *, device="cuda:0", contiguous=True, storage_id=None):
        self.values = tuple(values)
        self.shape = (len(self.values),)
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous
        self._storage_id = id(self) if storage_id is None else storage_id

    def reshape(self, shape):
        if shape == (-1,):
            return self
        assert shape == self.shape
        return self

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeTensor(self.values[item], dtype=self.dtype)
        return self.values[item]

    def chunk(self, chunks):
        assert len(self.values) % chunks == 0
        chunk_size = len(self.values) // chunks
        return tuple(
            FakeTensor(self.values[start : start + chunk_size], dtype=self.dtype)
            for start in range(0, len(self.values), chunk_size)
        )

    def new_empty(self, shape):
        return FakeTensor([0.0] * int(shape[0]), dtype=self.dtype, device=self.device)

    def narrow(self, dimension, start, length):
        assert dimension == 0
        return FakeTensorView(self, start, length)

    def new_zeros(self, shape):
        return FakeTensor([0.0] * int(shape[0]), dtype=self.dtype, device=self.device)

    def copy_(self, other):
        self.values = tuple(other.values)
        return self

    def numel(self):
        return len(self.values)

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._storage_id

    def __truediv__(self, divisor):
        return FakeTensor([value / divisor for value in self.values], dtype=self.dtype)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values


class FakeTensorView:
    def __init__(self, parent, start, length):
        self.parent = parent
        self.start = start
        self.length = length
        self.shape = (length,)
        self.dtype = parent.dtype
        self.device = parent.device

    @property
    def values(self):
        return self.parent.values[self.start : self.start + self.length]

    def copy_(self, other):
        values = list(self.parent.values)
        values[self.start : self.start + self.length] = other.values
        self.parent.values = tuple(values)
        return self

    def numel(self):
        return self.length

    def reshape(self, shape):
        assert shape in {(-1,), self.shape}
        return self


class FakeTorch:
    @staticmethod
    def cat(tensors, dim=0):
        assert dim == 0
        values = []
        for tensor in tensors:
            values.extend(tensor.values)
        return FakeTensor(values, dtype=tensors[0].dtype)


def test_reduce_scatter_compressed_restore_gathers_bytes_then_dequantizes() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_all_gather,
    )

    calls = []

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, received, sent):
            received[:] = [FakeTensor([10.0]), FakeTensor([20.0])]

        def all_gather_into_tensor(self, gathered, local):
            calls.append(("restore_all_gather", local.dtype, local.values))
            gathered.values = (11, 22)

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def quantize_chunk(tensor, config, *, extension_status):
        return FakeTensor([sum(tensor.values)])

    def dequantize_reduce(buffers, shape, config, *, dtype, extension_status, reduce):
        return FakeTensor([1.0, 2.0])

    def quantize_restore(tensor, config, *, extension_status):
        calls.append(("restore_quantize", tensor.values))
        return FakeTensor([11], dtype="torch.uint8")

    def dequantize_restore(buffer, shape, config, *, dtype, extension_status, output=None):
        calls.append(("restore_dequantize", buffer.values, shape, dtype))
        decoded = FakeTensor([1.0, 2.0] if buffer.values == (11,) else [3.0, 4.0])
        if output is not None:
            output.copy_(decoded)
            return output
        return decoded

    transport = make_torch_compressed_reduce_scatter_all_gather(
        import_module=import_module,
        quantize=quantize_chunk,
        dequantize_reduce=dequantize_reduce,
        restore_mode="compressed",
        restore_quantize=quantize_restore,
        restore_dequantize=dequantize_restore,
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert result == FakeTensor([1.0, 2.0, 3.0, 4.0])
    assert calls == [
        ("restore_quantize", (1.0, 2.0)),
        ("restore_all_gather", "torch.uint8", (11,)),
        ("restore_dequantize", (11,), (2,), "fp32"),
        ("restore_dequantize", (22,), (2,), "fp32"),
    ]


def test_reduce_scatter_rejects_unknown_restore_mode() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_all_gather,
    )

    with pytest.raises(ValueError, match="restore_mode"):
        make_torch_compressed_reduce_scatter_all_gather(restore_mode="int4")


def test_compressed_restore_validates_caller_workspace_capacity() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_all_gather,
    )

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, received, sent):
            received[:] = [FakeTensor([10.0]), FakeTensor([20.0])]

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    transport = make_torch_compressed_reduce_scatter_all_gather(
        import_module=import_module,
        quantize=lambda tensor, config, extension_status=None: FakeTensor([sum(tensor.values)]),
        dequantize_reduce=lambda buffers, shape, config, **kwargs: FakeTensor([1.0, 2.0]),
        restore_mode="compressed",
        restore_quantize=lambda tensor, config, extension_status=None: FakeTensor(
            [11], dtype="torch.uint8"
        ),
        allocate_compressed_restore_workspace=lambda payload, world_size: FakeTensor(
            [0], dtype="torch.uint8"
        ),
    )

    with pytest.raises(ValueError, match="compressed restore workspace"):
        transport(
            FakeTensor([1.0, 2.0, 3.0, 4.0]),
            config=CompressionConfig(bit=8),
            op="mean",
            async_op=False,
            dtype="fp32",
            extension_status=None,
        )


def test_reduce_scatter_transport_exchanges_compressed_chunks_and_restores_full_bucket() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_all_gather,
    )
    from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan

    calls = []

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, output, input):
            calls.append(("all_to_all", tuple(payload.values for payload in input)))
            output[:] = [FakeTensor([10.0]), FakeTensor([20.0])]

        def all_gather(self, output, input):
            calls.append(("all_gather", input.values))
            output[:] = [input, FakeTensor([50.0, 60.0])]

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def quantize(tensor, config, *, extension_status):
        calls.append(("quantize", tensor.values, config.bit))
        return FakeTensor([sum(tensor.values)])

    def dequantize_reduce(buffers, shape, config, *, dtype, extension_status, reduce):
        calls.append(("dequantize_reduce", tuple(buffer.values for buffer in buffers), shape, dtype, reduce))
        return FakeTensor([1.0, 2.0])

    transport = make_torch_compressed_reduce_scatter_all_gather(
        import_module=import_module,
        quantize=quantize,
        dequantize_reduce=dequantize_reduce,
        chunk_plan=compile_chunk_plan(original_numel=4, world_size=2),
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert result == FakeTensor([1.0, 2.0, 50.0, 60.0])
    assert ("quantize", (1.0, 2.0), 8) in calls
    assert ("quantize", (3.0, 4.0), 8) in calls
    assert ("all_to_all", ((3.0,), (7.0,))) in calls
    assert ("dequantize_reduce", ((10.0,), (20.0,)), (2,), "fp32", "mean") in calls
    assert ("all_gather", (1.0, 2.0)) in calls


def test_full_restore_uses_contiguous_caller_workspace_without_cat() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_all_gather,
    )

    calls = []
    output = FakeTensor([0.0, 0.0, 0.0, 0.0])

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, received, sent):
            received[:] = [FakeTensor([10.0]), FakeTensor([20.0])]

        def all_gather_into_tensor(self, restored, shard):
            calls.append((restored, shard))
            restored.values = (*shard.values, 9.0, 10.0)

        def all_gather(self, *_args):
            raise AssertionError("contiguous fast path must not allocate a tensor list")

    class TorchWithoutCat:
        pass

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return TorchWithoutCat
        raise AssertionError(name)

    transport = make_torch_compressed_reduce_scatter_all_gather(
        import_module=import_module,
        quantize=lambda tensor, config, extension_status=None: FakeTensor(
            [sum(tensor.values)]
        ),
        dequantize_reduce=lambda *args, **kwargs: FakeTensor([5.0, 7.0]),
        allocate_full_output_workspace=lambda shard, world_size: output,
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert result is output
    assert calls == [(output, FakeTensor([5.0, 7.0]))]


def test_async_full_restore_retains_caller_workspace_in_work_resources() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_all_gather,
    )

    output = FakeTensor([0.0, 0.0, 0.0, 0.0])

    class Future:
        def set_result(self, result):
            self.result = result

        def set_exception(self, exception):
            self.exception = exception

    class InnerFuture:
        def then(self, callback):
            callback(self)

    class Handle:
        def __init__(self, received):
            self.received = received

        def get_future(self):
            return InnerFuture()

        def wait(self):
            self.received[:] = [FakeTensor([10.0]), FakeTensor([20.0])]

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, received, sent, async_op=False):
            assert async_op is True
            return Handle(received)

        def all_gather_into_tensor(self, restored, shard):
            restored.values = (*shard.values, 9.0, 10.0)

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    transport = make_torch_compressed_reduce_scatter_all_gather(
        import_module=import_module,
        quantize=lambda tensor, config, extension_status=None: FakeTensor(
            [sum(tensor.values)]
        ),
        dequantize_reduce=lambda *args, **kwargs: FakeTensor([5.0, 7.0]),
        allocate_full_output_workspace=lambda shard, world_size: output,
        future_factory=Future,
    )

    work = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=True,
        dtype="fp32",
        extension_status=None,
    )

    assert output in work.resources
    assert work.wait() is output


def test_full_restore_fallback_copies_into_caller_workspace() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_all_gather,
    )

    output = FakeTensor([0.0, 0.0, 0.0, 0.0])

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, received, sent):
            received[:] = [FakeTensor([10.0]), FakeTensor([20.0])]

        def all_gather(self, shards, local):
            shards[:] = [local, FakeTensor([9.0, 10.0])]

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    transport = make_torch_compressed_reduce_scatter_all_gather(
        import_module=import_module,
        quantize=lambda tensor, config, extension_status=None: FakeTensor(
            [sum(tensor.values)]
        ),
        dequantize_reduce=lambda *args, **kwargs: FakeTensor([5.0, 7.0]),
        allocate_full_output_workspace=lambda tensor, world_size: output,
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert result is output
    assert result.values == (5.0, 7.0, 9.0, 10.0)


def test_reduce_scatter_full_bucket_transport_pads_non_divisible_bucket_and_trims_result() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_all_gather,
    )

    calls = []

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 1

        def all_to_all(self, output, input):
            calls.append(("all_to_all", tuple(payload.values for payload in input)))
            output[:] = [FakeTensor([10.0]), FakeTensor([20.0])]

        def all_gather(self, output, input):
            calls.append(("all_gather", input.values))
            output[:] = [FakeTensor([1.0, 2.0]), input]

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def quantize(tensor, config, *, extension_status):
        calls.append(("quantize", tensor.values))
        return FakeTensor([sum(tensor.values)])

    def dequantize_reduce(buffers, shape, config, *, dtype, extension_status, reduce):
        calls.append(("dequantize_reduce", tuple(buffer.values for buffer in buffers), shape, reduce))
        return FakeTensor([3.0, 0.0])

    transport = make_torch_compressed_reduce_scatter_all_gather(
        import_module=import_module,
        quantize=quantize,
        dequantize_reduce=dequantize_reduce,
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert result == FakeTensor([1.0, 2.0, 3.0])
    assert ("quantize", (1.0, 2.0)) in calls
    assert ("quantize", (3.0, 0.0)) in calls
    assert ("all_gather", (3.0, 0.0)) in calls


def test_reduce_scatter_shard_transport_returns_local_shard_without_full_all_gather() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 1

        def all_to_all(self, output, input):
            calls.append(("all_to_all", tuple(payload.values for payload in input)))
            output[:] = [FakeTensor([30.0]), FakeTensor([40.0])]

        def all_gather(self, output, input):
            raise AssertionError("sharded consumer path must not all_gather restored shards")

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def quantize(tensor, config, *, extension_status):
        calls.append(("quantize", tensor.values, config.bit))
        return FakeTensor([sum(tensor.values)])

    def dequantize_reduce(buffers, shape, config, *, dtype, extension_status, reduce):
        calls.append(("dequantize_reduce", tuple(buffer.values for buffer in buffers), shape, dtype, reduce))
        return FakeTensor([7.0, 11.0])

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=quantize,
        dequantize_reduce=dequantize_reduce,
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert result.shard == FakeTensor([7.0, 11.0])
    assert result.shard_index == 1
    assert result.shard_numel == 2
    assert result.original_shape == (4,)
    assert result.original_numel == 4
    assert result.world_size == 2
    assert result.reduce == "mean"
    assert ("all_to_all", ((3.0,), (7.0,))) in calls
    assert ("dequantize_reduce", ((30.0,), (40.0,)), (2,), "fp32", "mean") in calls


def test_reduce_scatter_shard_transport_reports_padding_metadata_for_uneven_bucket() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 1

        def all_to_all(self, output, input):
            calls.append(("all_to_all", tuple(payload.values for payload in input)))
            output[:] = [FakeTensor([30.0]), FakeTensor([40.0])]

        def all_gather(self, output, input):
            raise AssertionError("sharded consumer path must not all_gather restored shards")

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def quantize(tensor, config, *, extension_status):
        calls.append(("quantize", tensor.values))
        return FakeTensor([sum(tensor.values)])

    def dequantize_reduce(buffers, shape, config, *, dtype, extension_status, reduce):
        calls.append(("dequantize_reduce", shape))
        return FakeTensor([13.0, 0.0])

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=quantize,
        dequantize_reduce=dequantize_reduce,
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert result.shard == FakeTensor([13.0, 0.0])
    assert result.shard_index == 1
    assert result.shard_numel == 2
    assert result.original_numel == 3
    assert result.padded_numel == 4
    assert result.shard_offset == 2
    assert result.shard_end == 3
    assert result.valid_numel == 1
    assert result.padding_numel == 1
    assert result.has_padding is True
    assert result.dtype == "fp32"
    assert result.transport == "compressed_all_to_all"
    assert ("quantize", (3.0, 0.0)) in calls


def test_reduce_scatter_shard_transport_executes_precompiled_chunk_plan_without_full_payload_receives() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )
    from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan

    world_size = 4
    plan = compile_chunk_plan(original_numel=10, world_size=world_size)
    calls = []
    quantized_input_numel = []

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return world_size

        def get_rank(self):
            return 2

        def all_to_all(self, output, input):
            calls.append(("exchange", tuple(payload.numel() for payload in input), len(output)))
            assert len(input) == world_size
            assert len(output) == world_size
            assert all(payload.numel() < plan.original_numel for payload in input)
            output[:] = [FakeTensor([rank + 1.0]) for rank in range(world_size)]

        def all_gather(self, output, input):
            raise AssertionError("ReducedShard path must not restore the full gradient")

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def quantize_chunk(chunk, config, **kwargs):
        quantized_input_numel.append(chunk.numel())
        return FakeTensor([sum(chunk.values)])

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=quantize_chunk,
        dequantize_reduce=lambda payloads, shape, config, **kwargs: FakeTensor([1.0] * shape[0]),
        chunk_plan=plan,
    )
    result = transport(
        FakeTensor(range(10)),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert calls == [("exchange", (1, 1, 1, 1), 4)]
    assert quantized_input_numel == [plan.shard_numel] * world_size
    assert result.shard_index == 2
    assert result.original_numel == 10
    assert result.padded_numel == 12
    assert result.valid_numel + result.padding_numel == result.shard_numel
    assert result.metadata["chunk_plan_precompiled"] is True
    assert result.metadata["received_payload_numel"] < world_size * result.original_numel


@pytest.mark.parametrize("world_size", (1, 2, 3, 4, 5, 8))
def test_reduce_scatter_shard_metadata_is_valid_for_arbitrary_world_size(world_size: int) -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    original_numel = world_size * 5 + (0 if world_size == 1 else 1)
    rank = world_size - 1

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return world_size

        def get_rank(self):
            return rank

        def all_to_all(self, output, input):
            output[:] = [FakeTensor([1.0]) for _ in range(world_size)]

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    result = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=lambda chunk, config, **kwargs: FakeTensor([sum(chunk.values)]),
        dequantize_reduce=lambda payloads, shape, config, **kwargs: FakeTensor([0.0] * shape[0]),
    )(
        FakeTensor(range(original_numel)),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert result.shard_index == rank
    assert result.original_numel == original_numel
    assert result.padded_numel % world_size == 0
    assert result.valid_numel + result.padding_numel == result.shard_numel


@pytest.mark.parametrize("async_op", (False, True))
def test_reduce_scatter_shard_returns_empty_result_without_quantizing_or_communicating(async_op: bool) -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 4

        def get_rank(self):
            return 3

        def all_to_all(self, output, input, async_op=False):
            raise AssertionError("empty shard must not communicate")

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("empty shard must not quantize")
        ),
    )
    result_or_work = transport(
        FakeTensor([]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=async_op,
        dtype="fp32",
        extension_status=None,
    )
    result = result_or_work.wait() if async_op else result_or_work

    assert result.shard == FakeTensor([])
    assert result.shard_index == 3
    assert result.shard_numel == 0
    assert result.original_numel == 0
    assert result.padded_numel == 0
    assert result.valid_numel == 0
    assert result.padding_numel == 0


def test_reduce_scatter_shard_transport_can_use_reduced_shard_workspace() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []
    workspace = FakeTensor([0.0, 0.0])

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, output, input):
            output[:] = [FakeTensor([10.0]), FakeTensor([20.0])]

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def quantize(tensor, config, *, extension_status):
        return FakeTensor([sum(tensor.values)])

    def allocate_workspace(tensor, shape, config):
        calls.append(("allocate_workspace", tensor.values, shape, config.group_size))
        return workspace

    def dequantize_reduce(buffers, shape, config, *, dtype, extension_status, reduce, output=None):
        calls.append(("dequantize_reduce", output, shape, reduce))
        assert output is workspace
        return output

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=quantize,
        dequantize_reduce=dequantize_reduce,
        allocate_reduced_shard_workspace=allocate_workspace,
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert result.shard is workspace
    assert result.metadata["workspace_output"] is True
    assert result.metadata["workspace_shape"] == (2,)
    assert ("allocate_workspace", (1.0, 2.0, 3.0, 4.0), (2,), 64) in calls
    assert ("dequantize_reduce", workspace, (2,), "mean") in calls


def test_reduce_scatter_shard_transport_can_use_compressed_payload_workspaces() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []
    send_workspaces = [FakeTensor([100.0]), FakeTensor([200.0])]
    receive_workspaces = [FakeTensor([0.0]), FakeTensor([0.0])]

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, output, input):
            calls.append(("all_to_all_input_ids", tuple(id(tensor) for tensor in input)))
            calls.append(("all_to_all_output_ids", tuple(id(tensor) for tensor in output)))
            assert input == send_workspaces
            assert output == receive_workspaces
            output[:] = receive_workspaces

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def allocate_send_workspace(tensor, config):
        workspace = send_workspaces[len([call for call in calls if call[0] == "allocate_send"])]
        calls.append(("allocate_send", tensor.values, config.bit, workspace))
        return workspace

    def allocate_receive_workspace(payload, index, world_size, config):
        workspace = receive_workspaces[index]
        calls.append(("allocate_receive", payload.values, index, world_size, config.bit, workspace))
        return workspace

    def quantize(tensor, config, *, extension_status, output=None):
        calls.append(("quantize", tensor.values, output))
        assert output in send_workspaces
        return output

    def dequantize_reduce(buffers, shape, config, *, dtype, extension_status, reduce):
        calls.append(("dequantize_reduce_buffers", tuple(id(buffer) for buffer in buffers)))
        assert buffers == receive_workspaces
        return FakeTensor([5.0, 7.0])

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=quantize,
        dequantize_reduce=dequantize_reduce,
        allocate_quantized_chunk_workspace=allocate_send_workspace,
        allocate_received_payload_workspace=allocate_receive_workspace,
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert result.shard == FakeTensor([5.0, 7.0])
    assert result.metadata["quantized_workspace_output"] is True
    assert result.metadata["received_workspace_output"] is True
    assert ("allocate_send", (1.0, 2.0), 8, send_workspaces[0]) in calls
    assert ("allocate_send", (3.0, 4.0), 8, send_workspaces[1]) in calls
    assert ("allocate_receive", (100.0,), 0, 2, 8, receive_workspaces[0]) in calls
    assert ("allocate_receive", (100.0,), 1, 2, 8, receive_workspaces[1]) in calls


def test_reduce_scatter_shard_transport_can_use_workspace_cache() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )
    from ccdl_comm.communication.workspace import ShardCommunicationWorkspaceCache

    calls = []

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, output, input):
            calls.append(("all_to_all", tuple(id(tensor) for tensor in input), tuple(id(tensor) for tensor in output)))
            output[:] = output

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def quantized_allocator(tensor, config, dtype):
        workspace = FakeTensor([100.0 + len([call for call in calls if call[0] == "send_alloc"])])
        calls.append(("send_alloc", tensor.values, config.bit, config.group_size, dtype, workspace))
        return workspace

    def reduced_allocator(tensor, shape, config):
        workspace = FakeTensor([0.0] * shape[0])
        calls.append(("reduced_alloc", tensor.values, shape, config.group_size, workspace))
        return workspace

    cache = ShardCommunicationWorkspaceCache(
        quantized_allocator=quantized_allocator,
        reduced_allocator=reduced_allocator,
    )

    def quantize(tensor, config, *, extension_status, output=None):
        assert output is not None
        return output

    def dequantize_reduce(buffers, shape, config, *, dtype, extension_status, reduce, output=None):
        assert output is not None
        return output

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=quantize,
        dequantize_reduce=dequantize_reduce,
        workspace_cache=cache,
    )
    bucket = FakeTensor([1.0, 2.0, 3.0, 4.0])
    config = CompressionConfig(bit=8)

    first = transport(bucket, config=config, op="mean", async_op=False, dtype="fp32", extension_status=None)
    second = transport(bucket, config=config, op="mean", async_op=False, dtype="fp32", extension_status=None)

    assert first.shard is second.shard
    assert first.metadata["workspace_cache"] is True
    assert first.metadata["workspace_output"] is True
    assert first.metadata["quantized_workspace_output"] is True
    assert first.metadata["received_workspace_output"] is True
    assert len([call for call in calls if call[0] == "send_alloc"]) == 2
    assert len([call for call in calls if call[0] == "reduced_alloc"]) == 1


def test_reduce_scatter_transport_releases_pooled_session_after_completion() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, output, input):
            calls.append("all_to_all")

    class Completion:
        def wait(self):
            calls.append("completion_wait")

        def query(self):
            return True

    class CompletionManager:
        def record_for(self, tensor):
            calls.append(("record", tensor))
            return Completion()

    class Session:
        leases = ()

        def get_quantized_chunk(self, bucket_key, index, tensor, config, **kwargs):
            calls.append(("send", index))
            return FakeTensor([100.0 + index], dtype="torch.uint8")

        def get_received_payload(self, bucket_key, template, index, **kwargs):
            calls.append(("recv", index))
            return FakeTensor([0.0], dtype="torch.uint8")

        def get_reduced_shard(self, bucket_key, tensor, shape, config, **kwargs):
            calls.append("reduced")
            return FakeTensor([0.0] * shape[0], dtype=tensor.dtype)

        def release(self, *, completion):
            calls.append(("session_release", completion))

    class Provider:
        def begin(self, *, stream):
            calls.append(("begin", stream))
            return Session()

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=lambda tensor, config, **kwargs: kwargs["output"],
        dequantize_reduce=lambda buffers, shape, config, **kwargs: kwargs["output"],
        workspace_cache=Provider(),
        completion_manager=CompletionManager(),
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    completion = calls[-1][1]
    assert result.shard.values == (0.0, 0.0)
    assert calls[-3:] == [("record", result.shard), "completion_wait", ("session_release", completion)]


def test_reduce_scatter_transport_releases_pooled_session_when_quantize_fails() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

    class Completion:
        def wait(self):
            calls.append("completion_wait")

    class CompletionManager:
        def record_for(self, tensor):
            calls.append(("record", tensor))
            return Completion()

    class Session:
        leases = ()

        def get_quantized_chunk(self, bucket_key, index, tensor, config, **kwargs):
            calls.append(("send", index))
            return FakeTensor([100.0], dtype="torch.uint8")

        def release(self, *, completion):
            calls.append(("session_release", completion))

    class Provider:
        def begin(self, *, stream):
            return Session()

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    tensor = FakeTensor([1.0, 2.0, 3.0, 4.0])
    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("quantize failed")),
        workspace_cache=Provider(),
        completion_manager=CompletionManager(),
    )

    with pytest.raises(RuntimeError, match="quantize failed"):
        transport(
            tensor,
            config=CompressionConfig(bit=8),
            op="mean",
            async_op=False,
            dtype="fp32",
            extension_status=None,
        )

    completion = calls[-1][1]
    assert calls[-3:] == [("record", tensor), "completion_wait", ("session_release", completion)]


def test_async_transport_waits_started_collective_before_exception_cleanup() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []

    class Work:
        def wait(self):
            calls.append("work_wait")

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, output, input, async_op=False):
            assert async_op is True
            calls.append("all_to_all_started")
            return Work()

    class Completion:
        def wait(self):
            calls.append("completion_wait")

    class CompletionManager:
        def record_for(self, tensor):
            calls.append(("record", tensor))
            return Completion()

    class Session:
        leases = ()

        def get_quantized_chunk(self, bucket_key, index, tensor, config, **kwargs):
            return FakeTensor([100.0 + index], dtype="torch.uint8")

        def get_received_payload(self, bucket_key, template, index, **kwargs):
            return FakeTensor([0.0], dtype="torch.uint8")

        def get_reduced_shard(self, *args, **kwargs):
            return None

        def release(self, *, completion):
            calls.append(("session_release", completion))

    class Provider:
        def begin(self, *, stream):
            return Session()

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    tensor = FakeTensor([1.0, 2.0, 3.0, 4.0])
    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=lambda tensor, config, **kwargs: kwargs["output"],
        workspace_cache=Provider(),
        completion_manager=CompletionManager(),
        future_factory=lambda: (_ for _ in ()).throw(RuntimeError("future failed")),
    )

    with pytest.raises(RuntimeError, match="future failed"):
        transport(
            tensor,
            config=CompressionConfig(bit=8),
            op="mean",
            async_op=True,
            dtype="fp32",
            extension_status=None,
        )

    assert calls.index("work_wait") < calls.index("completion_wait")
    assert calls.index("completion_wait") < next(
        index for index, call in enumerate(calls) if isinstance(call, tuple) and call[0] == "session_release"
    )


def test_reduce_scatter_shard_transport_uses_fused_dequant_reduce_fastpath() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []
    workspace = FakeTensor([0.0, 0.0])

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, output, input):
            output[:] = [FakeTensor([10.0]), FakeTensor([20.0])]

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def quantize(tensor, config, *, extension_status):
        return FakeTensor([sum(tensor.values)])

    def allocate_workspace(tensor, shape, config):
        return workspace

    def fused_dequant_reduce(buffers, output, *, reduce):
        calls.append(("fused", tuple(buffer.values for buffer in buffers), output, reduce))
        assert output is workspace
        return True

    def dequantize_reduce(*args, **kwargs):
        raise AssertionError("fused fastpath should replace fallback dequantize_reduce")

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=quantize,
        dequantize_reduce=dequantize_reduce,
        allocate_reduced_shard_workspace=allocate_workspace,
        fused_dequantize_reduce=fused_dequant_reduce,
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert result.shard is workspace
    assert result.metadata["fused_dequant_reduce"] is True
    assert calls == [
        ("fused", ((10.0,), (20.0,)), workspace, "mean"),
    ]


def test_reduce_scatter_shard_transport_writes_to_caller_owned_output() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []
    output = FakeTensor([0.0, 0.0])

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, received, compressed):
            calls.append("all_to_all")
            received[:] = [FakeTensor([10.0]), FakeTensor([20.0])]

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def quantize(chunk, config, *, extension_status):
        calls.append("quantize")
        return FakeTensor([sum(chunk.values)])

    def fused_dequant_reduce(buffers, target, *, reduce):
        calls.append("fused")
        assert target is output
        return True

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=quantize,
        fused_dequantize_reduce=fused_dequant_reduce,
        dequantize_reduce=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fallback")),
    )

    reduced = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
        out=output,
    )

    assert reduced.shard is output
    assert reduced.metadata["output_ownership"] == "caller"
    assert reduced.metadata["fused_dequant_reduce"] is True
    assert "fused_dequant_reduce_reason" not in reduced.metadata
    assert calls == ["quantize", "quantize", "all_to_all", "fused"]


@pytest.mark.parametrize(
    ("out", "message"),
    [
        (FakeTensor([0.0]), "numel"),
        (FakeTensor([0.0, 0.0], dtype="torch.float16"), "dtype"),
        (FakeTensor([0.0, 0.0], device="cuda:1"), "device"),
        (FakeTensor([0.0, 0.0], contiguous=False), "contiguous"),
    ],
)
def test_reduce_scatter_shard_transport_rejects_invalid_caller_output_before_communication(out, message) -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, received, compressed):
            calls.append("all_to_all")

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=lambda *args, **kwargs: calls.append("quantize"),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        transport(
            FakeTensor([1.0, 2.0, 3.0, 4.0]),
            config=CompressionConfig(bit=8),
            op="mean",
            async_op=False,
            dtype="fp32",
            extension_status=None,
            out=out,
        )

    assert calls == []


def test_reduce_scatter_shard_transport_rejects_caller_output_alias_before_communication() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []
    source = FakeTensor([1.0, 2.0, 3.0, 4.0], storage_id=17)
    output = FakeTensor([0.0, 0.0], storage_id=17)

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, received, compressed):
            calls.append("all_to_all")

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=lambda *args, **kwargs: calls.append("quantize"),
    )

    with pytest.raises(ValueError, match="alias"):
        transport(
            source,
            config=CompressionConfig(bit=8),
            op="mean",
            async_op=False,
            dtype="fp32",
            extension_status=None,
            out=output,
        )

    assert calls == []


def test_reduce_scatter_shard_transport_rejects_same_numel_different_caller_output_shape() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []
    output = FakeTensor([0.0, 0.0])
    output.shape = (1, 2)

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, received, compressed):
            calls.append("all_to_all")

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=lambda *args, **kwargs: calls.append("quantize"),
    )

    with pytest.raises(ValueError, match="shape"):
        transport(
            FakeTensor([1.0, 2.0, 3.0, 4.0]),
            config=CompressionConfig(bit=8),
            op="mean",
            async_op=False,
            dtype="fp32",
            extension_status=None,
            out=output,
        )

    assert calls == []


def test_reduce_scatter_shard_transport_does_not_treat_zero_data_pointers_as_aliases() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    source = FakeTensor([], storage_id=0)
    output = FakeTensor([], storage_id=0)

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    reduced = make_torch_compressed_reduce_scatter_shard(import_module=import_module)(
        source,
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
        out=output,
    )

    assert reduced.shard is output
    assert reduced.metadata["output_ownership"] == "caller"


def test_reduce_scatter_shard_transport_falls_back_when_fused_fastpath_declines() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []
    workspace = FakeTensor([0.0, 0.0])

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, output, input):
            output[:] = [FakeTensor([10.0]), FakeTensor([20.0])]

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def quantize(tensor, config, *, extension_status):
        return FakeTensor([sum(tensor.values)])

    def allocate_workspace(tensor, shape, config):
        return workspace

    def fused_dequant_reduce(buffers, output, *, reduce):
        calls.append(("fused_declined", output, reduce))
        return False

    def dequantize_reduce(buffers, shape, config, *, dtype, extension_status, reduce, output=None):
        calls.append(("fallback", output, shape, reduce))
        assert output is workspace
        return output

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=quantize,
        dequantize_reduce=dequantize_reduce,
        allocate_reduced_shard_workspace=allocate_workspace,
        fused_dequantize_reduce=fused_dequant_reduce,
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert result.shard is workspace
    assert result.metadata["fused_dequant_reduce"] is False
    assert calls == [
        ("fused_declined", workspace, "mean"),
        ("fallback", workspace, (2,), "mean"),
    ]


def test_reduce_scatter_shard_transport_can_complete_async_all_to_all_work() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_shard,
    )

    calls = []

    class Future:
        def __init__(self):
            self.result = None
            self.exception = None

        def set_result(self, result):
            calls.append(("future_result", result.shard.values))
            self.result = result

        def set_exception(self, exception):
            self.exception = exception

    class InnerFuture:
        def then(self, callback):
            calls.append("then")
            return callback(self)

    class Work:
        def __init__(self, output):
            self._output = output

        def get_future(self):
            calls.append("get_future")
            return InnerFuture()

        def wait(self):
            calls.append("wait")
            self._output[:] = [FakeTensor([30.0]), FakeTensor([40.0])]
            return True

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 1

        def all_to_all(self, output, input, async_op=False):
            calls.append(("all_to_all", async_op, tuple(payload.values for payload in input)))
            assert async_op is True
            return Work(output)

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def quantize(tensor, config, *, extension_status):
        return FakeTensor([sum(tensor.values)])

    def dequantize_reduce(buffers, shape, config, *, dtype, extension_status, reduce):
        calls.append(("dequantize_reduce", tuple(buffer.values for buffer in buffers), shape, reduce))
        return FakeTensor([7.0, 11.0])

    transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=quantize,
        dequantize_reduce=dequantize_reduce,
        future_factory=Future,
    )

    work = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=True,
        dtype="fp32",
        extension_status=None,
    )

    result = work.wait()
    assert work.query() is True
    assert len(work.resources) >= 6
    assert result.shard == FakeTensor([7.0, 11.0])
    assert result.metadata["async_completion"] is True
    assert ("all_to_all", True, ((3.0,), (7.0,))) in calls
    assert "get_future" in calls
    assert "then" in calls
    assert "wait" in calls
    assert ("future_result", (7.0, 11.0)) in calls


def test_reduce_scatter_full_bucket_async_work_gathers_after_shard_completion() -> None:
    from ccdl_comm.communication.reduce_scatter_transport import (
        make_torch_compressed_reduce_scatter_all_gather,
    )

    class Future:
        def __init__(self):
            self.result = None
            self.exception = None

        def set_result(self, result):
            self.result = result

        def set_exception(self, exception):
            self.exception = exception

    class InnerFuture:
        def then(self, callback):
            return callback(self)

    class Work:
        def __init__(self, output):
            self._output = output

        def get_future(self):
            return InnerFuture()

        def wait(self):
            self._output[:] = [FakeTensor([30.0]), FakeTensor([40.0])]

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def get_rank(self):
            return 0

        def all_to_all(self, output, input, async_op=False):
            assert async_op is True
            return Work(output)

        def all_gather(self, output, input):
            output[:] = [input, FakeTensor([9.0, 10.0])]

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    transport = make_torch_compressed_reduce_scatter_all_gather(
        import_module=import_module,
        quantize=lambda tensor, config, extension_status=None: FakeTensor([sum(tensor.values)]),
        dequantize_reduce=lambda buffers, shape, config, dtype, extension_status, reduce: FakeTensor([5.0, 7.0]),
        future_factory=Future,
    )

    work = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=True,
        dtype="fp32",
        extension_status=None,
    )

    assert work.wait() == FakeTensor([5.0, 7.0, 9.0, 10.0])
