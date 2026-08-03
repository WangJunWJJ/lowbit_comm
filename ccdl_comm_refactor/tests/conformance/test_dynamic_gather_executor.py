from __future__ import annotations

import pytest

from ccdl_comm import CompressionConfig


class FakeTensor:
    def __init__(self, values, *, dtype="torch.float16") -> None:
        self.values = tuple(values)
        self.dtype = dtype
        self.shape = (len(self.values),)
        self.device = "cuda:0"

    def numel(self) -> int:
        return len(self.values)

    def new_zeros(self, shape):
        return FakeTensor([0.0] * int(shape[0]), dtype=self.dtype)

    def new_empty(self, shape, dtype=None):
        return FakeTensor([0.0] * int(shape[0]), dtype=dtype or self.dtype)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeTensor(self.values[item], dtype=self.dtype)
        return self.values[item]


class FakeTorch:
    uint8 = "torch.uint8"

    @staticmethod
    def cat(tensors, dim=0):
        assert dim == 0
        return FakeTensor(
            [value for tensor in tensors for value in tensor.values],
            dtype=tensors[0].dtype,
        )


class FakeDist:
    def __init__(self, *, protocol_version=1) -> None:
        self.protocol_version = protocol_version
        self.metadata_calls = 0
        self.payload_calls = 0

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def get_world_size(self, group=None) -> int:
        return 4

    def all_gather_object(self, output, local, group=None) -> None:
        self.metadata_calls += 1
        output[:] = [
            _metadata(self.protocol_version, 0, 0),
            _metadata(self.protocol_version, 63, 66),
            _metadata(self.protocol_version, 64, 66),
            _metadata(self.protocol_version, 65, 132),
        ]

    def all_gather(self, output, local, group=None) -> None:
        self.payload_calls += 1
        assert local.numel() == 132
        output[:] = [
            FakeTensor([0.0] * 132),
            FakeTensor([1.0] * 66 + [91.0] * 66),
            FakeTensor([2.0] * 66 + [92.0] * 66),
            FakeTensor([3.0] * 132),
        ]


def test_compiled_dynamic_gather_preserves_zero_and_boundary_shapes() -> None:
    from ccdl_comm.cuda.dynamic_gather_executor import compile_dynamic_all_gather

    dist = FakeDist()
    executor = compile_dynamic_all_gather(
        shape_class=(128,),
        config=CompressionConfig(bit=8, group_size=64),
        dtype="fp16",
        import_module_fn=_importer(dist),
        quantize=_quantize,
        dequantize=_dequantize,
    )

    result = executor.run(FakeTensor([4.0] * 65)).wait()

    assert [tensor.shape for tensor in result] == [(0,), (63,), (64,), (65,)]
    assert result[0].values == ()
    assert result[1].values == (1.0,) * 63
    assert result[2].values == (2.0,) * 64
    assert result[3].values == (3.0,) * 65
    assert dist.metadata_calls == 1
    assert dist.payload_calls == 1
    assert executor.metadata_protocol_version == 1
    assert executor.shape_class == (128,)


def test_dynamic_shape_class_cache_reuses_only_equivalent_bounds() -> None:
    from ccdl_comm.cuda.dynamic_gather_executor import (
        DynamicGatherExecutorCache,
        compile_dynamic_all_gather,
    )

    dist = FakeDist()
    cache = DynamicGatherExecutorCache(max_entries=2)
    common = {
        "config": CompressionConfig(bit=8, group_size=64),
        "dtype": "fp16",
        "import_module_fn": _importer(dist),
        "quantize": _quantize,
        "dequantize": _dequantize,
        "cache": cache,
    }

    first = compile_dynamic_all_gather(shape_class=(64,), **common)
    same = compile_dynamic_all_gather(shape_class=(64,), **common)
    larger = compile_dynamic_all_gather(shape_class=(128,), **common)

    assert first is same
    assert larger is not first
    assert len(cache) == 2


def test_dynamic_gather_rejects_newer_metadata_protocol_before_payload() -> None:
    from ccdl_comm.cuda.dynamic_gather_executor import compile_dynamic_all_gather

    dist = FakeDist(protocol_version=2)
    executor = compile_dynamic_all_gather(
        shape_class=(128,),
        config=CompressionConfig(bit=8, group_size=64),
        dtype="fp16",
        import_module_fn=_importer(dist),
        quantize=_quantize,
        dequantize=_dequantize,
    )

    with pytest.raises(RuntimeError, match="metadata protocol version"):
        executor.run(FakeTensor([4.0] * 65))
    assert dist.payload_calls == 0


def test_dynamic_gather_rejects_metadata_payload_size_mismatch() -> None:
    from ccdl_comm.cuda.dynamic_gather_executor import compile_dynamic_all_gather

    class InvalidMetadataDist(FakeDist):
        def all_gather_object(self, output, local, group=None) -> None:
            super().all_gather_object(output, local, group=group)
            output[1]["payload_numel"] = 0

    dist = InvalidMetadataDist()
    executor = compile_dynamic_all_gather(
        shape_class=(128,),
        config=CompressionConfig(bit=8, group_size=64),
        dtype="fp16",
        import_module_fn=_importer(dist),
        quantize=_quantize,
        dequantize=_dequantize,
    )

    with pytest.raises(RuntimeError, match="payload size"):
        executor.run(FakeTensor([4.0] * 65))
    assert dist.payload_calls == 0


def test_dynamic_gather_rejects_tensor_outside_compiled_shape_class() -> None:
    from ccdl_comm.cuda.dynamic_gather_executor import compile_dynamic_all_gather

    executor = compile_dynamic_all_gather(
        shape_class=(64,),
        config=CompressionConfig(bit=8, group_size=64),
        dtype="fp16",
        import_module_fn=_importer(FakeDist()),
        quantize=_quantize,
        dequantize=_dequantize,
    )

    with pytest.raises(ValueError, match="shape class capacity"):
        executor.run(FakeTensor([4.0] * 65))


def test_dynamic_gather_zero_tensor_skips_native_quantize_kernel() -> None:
    from ccdl_comm.cuda.dynamic_gather_executor import compile_dynamic_all_gather

    def reject_quantize(*args, **kwargs):
        raise AssertionError("zero tensors must not launch the quantize kernel")

    executor = compile_dynamic_all_gather(
        shape_class=(128,),
        config=CompressionConfig(bit=8, group_size=64),
        dtype="fp16",
        import_module_fn=_importer(FakeDist()),
        quantize=reject_quantize,
        dequantize=_dequantize,
    )

    result = executor.run(FakeTensor([])).wait()

    assert result[0].shape == (0,)


def test_compiled_dynamic_gather_factory_is_public() -> None:
    from ccdl_comm import compile_dynamic_all_gather

    assert callable(compile_dynamic_all_gather)


def _metadata(version: int, numel: int, payload_numel: int) -> dict[str, object]:
    return {
        "protocol_version": version,
        "shape": (numel,),
        "dtype": "fp16",
        "payload_numel": payload_numel,
    }


def _importer(dist):
    def import_module(name):
        if name == "torch.distributed":
            return dist
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    return import_module


def _quantize(tensor, config, *, extension_status=None):
    del config, extension_status
    payload_numel = 0 if tensor.numel() == 0 else (tensor.numel() + 63) // 64 * 66
    return FakeTensor([4.0] * payload_numel, dtype="torch.uint8")


def _dequantize(
    buffer,
    shape,
    config,
    *,
    dtype,
    extension_status=None,
    output=None,
    reduce_op="none",
):
    del config, dtype, extension_status, output, reduce_op
    return FakeTensor([buffer.values[0]] * shape[0])
