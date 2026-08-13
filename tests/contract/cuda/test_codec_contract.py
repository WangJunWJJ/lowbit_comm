from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lowbit_comm.backends.cuda.codec import (
    ExtensionUnavailable,
    dequantize_into,
    payload_nbytes,
    quantize_into,
    quantize_chunks_into,
    decode_dynamic_metadata_into,
)
from lowbit_comm.backends.cuda.build import create_cuda_extension
from lowbit_comm.backends.cuda.loader import CudaExtensionStatus, load_cuda_extension
from lowbit_comm.core import DataType, QuantizedWire


ROOT = Path(__file__).parents[3]


class _Enum:
    Linear = object()


class _ReduceEnum:
    NONE = object()


class _NativeModule:
    QuantType = _Enum
    ReduceOP = _ReduceEnum

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def inplace_quantize(self, *args: object) -> None:
        self.calls.append(("quantize",) + args)

    def inplace_quantize_chunks(self, *args: object) -> bool:
        self.calls.append(("quantize_chunks",) + args)
        return True

    def inplace_dequantize(self, *args: object) -> None:
        self.calls.append(("dequantize",) + args)

    def inplace_decode_dynamic_metadata(self, *args: object) -> None:
        self.calls.append(("metadata",) + args)


def test_cuda_package_import_is_safe_without_torch_or_extension() -> None:
    module = importlib.import_module("lowbit_comm.backends.cuda")

    assert module is not None


def test_loader_reports_missing_and_broken_extensions_as_data() -> None:
    def missing(name: str) -> object:
        raise ModuleNotFoundError(name=name)

    def broken(name: str) -> object:
        raise ImportError(f"undefined symbol in {name}")

    missing_status = load_cuda_extension(import_module=missing)
    broken_status = load_cuda_extension(import_module=broken)

    assert missing_status.available is False
    assert "not installed" in (missing_status.reason or "")
    assert broken_status.available is False
    assert "undefined symbol" in (broken_status.reason or "")


@pytest.mark.parametrize(
    ("numel", "dtype", "wire", "expected"),
    [
        (64, DataType.FP16, QuantizedWire(8, 64), 66),
        (65, DataType.FP16, QuantizedWire(8, 64), 132),
        (64, DataType.FP32, QuantizedWire(4, 64), 36),
        (0, DataType.FP16, QuantizedWire(8, 64), 0),
    ],
)
def test_payload_sizing_matches_native_linear_layout(
    numel: int,
    dtype: DataType,
    wire: QuantizedWire,
    expected: int,
) -> None:
    assert payload_nbytes(numel, dtype=dtype, wire=wire) == expected


def test_codec_writes_caller_owned_buffers_without_allocation() -> None:
    native = _NativeModule()
    status = CudaExtensionStatus(True, native)
    source = SimpleNamespace(numel=lambda: 64)
    payload = object()
    output = object()
    wire = QuantizedWire(8, 64)

    assert quantize_into(source, payload, wire, extension_status=status) is payload
    assert dequantize_into(
        payload,
        output,
        wire,
        dtype=DataType.FP16,
        extension_status=status,
    ) is output
    assert [call[0] for call in native.calls] == ["quantize", "dequantize"]


def test_codec_quantizes_contiguous_chunks_with_one_native_dispatch() -> None:
    native = _NativeModule()
    status = CudaExtensionStatus(True, native)
    source = object()
    payloads = object()
    wire = QuantizedWire(8, 64, compact=False)

    assert quantize_chunks_into(
        source,
        payloads,
        wire,
        chunk_numel=4096,
        chunks=8,
        payload_stride=4224,
        extension_status=status,
    ) is payloads
    assert [call[0] for call in native.calls] == ["quantize_chunks"]
    assert native.calls[0][3:6] == (4096, 8, 4224)


@pytest.mark.parametrize("compact", (False, True))
def test_chunk_quantization_facade_preserves_wire_layout(compact: bool) -> None:
    native = _NativeModule()
    status = CudaExtensionStatus(True, native)

    quantize_chunks_into(
        object(),
        object(),
        QuantizedWire(8, 64, compact=compact),
        chunk_numel=128,
        chunks=2,
        payload_stride=144,
        extension_status=status,
    )

    assert native.calls[0][-1] is compact


def test_metadata_decoder_dispatches_complete_device_schema() -> None:
    native = _NativeModule()
    metadata = object()
    descriptors = object()

    result = decode_dynamic_metadata_into(
        metadata,
        descriptors,
        world_size=4,
        dtype=DataType.FP16,
        wire=QuantizedWire(8, 64),
        layout_generation=3,
        max_numel=4096,
        payload_stride=4224,
        extension_status=CudaExtensionStatus(True, native),
    )

    assert result is descriptors
    assert native.calls[-1][0] == "metadata"
    assert native.calls[-1][1:5] == (metadata, descriptors, 4, 1)


def test_codec_rejects_missing_extension_and_symbol() -> None:
    unavailable = CudaExtensionStatus(False, None, "not built")
    incomplete = CudaExtensionStatus(True, SimpleNamespace(QuantType=_Enum))

    with pytest.raises(ExtensionUnavailable, match="not built"):
        quantize_into(object(), object(), QuantizedWire(8, 64), extension_status=unavailable)
    with pytest.raises(ExtensionUnavailable, match="inplace_dequantize"):
        dequantize_into(
            object(),
            object(),
            QuantizedWire(8, 64),
            dtype=DataType.FP16,
            extension_status=incomplete,
        )


def test_migrated_native_assets_are_exactly_manifested() -> None:
    csrc = ROOT / "src" / "lowbit_comm" / "backends" / "cuda" / "csrc"
    manifest = json.loads((csrc / "MIGRATED_ASSETS.json").read_text(encoding="utf-8"))
    generated = set(manifest["generated_at_build"])
    actual = {
        path.relative_to(csrc).as_posix()
        for path in csrc.rglob("*")
        if path.is_file()
        and path.name != "MIGRATED_ASSETS.json"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.relative_to(csrc).as_posix() not in generated
    }

    assert set(manifest["files"]) == actual
    assert manifest["legacy_python_imports_allowed"] is False
    assert manifest["native_module"] == "lowbit_comm_cuda_ops"


def test_extension_spec_uses_new_module_and_package_local_sources() -> None:
    captured: dict[str, object] = {}

    def factory(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return kwargs

    extension = create_cuda_extension(
        extension_factory=factory,
        ensure_generated=lambda source_dir: None,
    )

    assert extension["name"] == "lowbit_comm_cuda_ops"
    assert extension["sources"] == sorted(extension["sources"])
    assert all("ccdl_comm" not in source for source in extension["sources"])
    assert any(source.endswith("pybind.cpp") for source in extension["sources"])


def test_pybind_exports_the_build_selected_module_name() -> None:
    pybind = (
        ROOT
        / "src"
        / "lowbit_comm"
        / "backends"
        / "cuda"
        / "csrc"
        / "pybind.cpp"
    ).read_text(encoding="utf-8")

    assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)" in pybind
    assert "PYBIND11_MODULE(ccdl_cuda_ops, m)" not in pybind


def test_metadata_kernel_validates_exact_payload_layout_on_device() -> None:
    kernel = (
        ROOT
        / "src/lowbit_comm/backends/cuda/csrc/quantization/metadata_kernel.cu"
    ).read_text(encoding="utf-8")

    assert "expected_payload" in kernel
    assert "packet[3] != expected_payload" in kernel
