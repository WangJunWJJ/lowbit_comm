"""Compile and validate the migrated CUDA codec on one real GPU."""

from __future__ import annotations

import json
import torch
from torch.utils.cpp_extension import load

from lowbit_comm.backends.cuda.build import CSRC_ROOT, ensure_generated_sources
from lowbit_comm.backends.cuda.codec import dequantize_into, payload_nbytes, quantize_into
from lowbit_comm.backends.cuda.loader import CudaExtensionStatus
from lowbit_comm.core import DataType, QuantizedWire


def main() -> None:
    quantization = CSRC_ROOT / "quantization"
    ensure_generated_sources(quantization)
    sources = [CSRC_ROOT / "pybind.cpp"]
    sources.extend((CSRC_ROOT / "executor").glob("*.cpp"))
    sources.extend(quantization.glob("*.cu"))
    module = load(
        name="lowbit_comm_cuda_ops",
        sources=sorted(str(path) for path in sources),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-U__CUDA_NO_HALF_OPERATORS__"],
        verbose=False,
    )
    status = CudaExtensionStatus(True, module)
    wire = QuantizedWire(bit=8, group_size=64)
    source = torch.linspace(-1.0, 1.0, 4096, device="cuda", dtype=torch.float16)
    payload = torch.empty(
        payload_nbytes(source.numel(), dtype=DataType.FP16, wire=wire),
        device="cuda",
        dtype=torch.uint8,
    )
    restored = torch.empty_like(source)

    quantize_into(source, payload, wire, extension_status=status)
    dequantize_into(
        payload,
        restored,
        wire,
        dtype=DataType.FP16,
        extension_status=status,
    )
    torch.cuda.synchronize()
    max_abs_error = float((source - restored).abs().max().item())
    assert max_abs_error <= 0.02, max_abs_error
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "numel": source.numel(),
                "payload_bytes": payload.numel(),
                "max_abs_error": max_abs_error,
            }
        )
    )


if __name__ == "__main__":
    main()
