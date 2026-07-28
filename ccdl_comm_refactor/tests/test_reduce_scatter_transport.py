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
