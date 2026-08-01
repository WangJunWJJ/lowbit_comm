import pytest

from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import UnsupportedCollective


class FakeTensor:
    def __init__(self, values, dtype="torch.float32"):
        self.values = tuple(values)
        self.shape = (len(self.values),)
        self.dtype = dtype

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
        return FakeTensor([0.0] * int(shape[0]), dtype=self.dtype)

    def new_zeros(self, shape):
        return FakeTensor([0.0] * int(shape[0]), dtype=self.dtype)

    def numel(self):
        return len(self.values)

    def __truediv__(self, divisor):
        return FakeTensor([value / divisor for value in self.values], dtype=self.dtype)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values


class FakeTorch:
    @staticmethod
    def cat(tensors, dim=0):
        assert dim == 0
        values = []
        for tensor in tensors:
            values.extend(tensor.values)
        return FakeTensor(values, dtype=tensors[0].dtype)


def test_reduce_scatter_transport_exchanges_compressed_chunks_and_restores_full_bucket() -> None:
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

    def fused_dequant_reduce(buffers, output, shape, config, *, dtype, extension_status, reduce):
        calls.append(("fused", tuple(buffer.values for buffer in buffers), output, shape, dtype, reduce))
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
        ("fused", ((10.0,), (20.0,)), workspace, (2,), "fp32", "mean"),
    ]


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

    def fused_dequant_reduce(buffers, output, shape, config, *, dtype, extension_status, reduce):
        calls.append(("fused_declined", output, shape, reduce))
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
        ("fused_declined", workspace, (2,), "mean"),
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
