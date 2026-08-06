from __future__ import annotations

import os

import torch

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.quantization.codec import (
    dequantize_tensor,
    quantize_tensor,
    update_error_feedback_residual,
)


def main() -> None:
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    current_index = rank * 2
    target_index = current_index + 1
    if torch.cuda.device_count() <= target_index:
        raise RuntimeError(
            f"oracle requires at least {target_index + 1} CUDA devices; "
            f"found {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(current_index)
    status = load_cuda_extension()
    if not status.available:
        raise RuntimeError(status.reason or "CCDL CUDA extension unavailable")

    target = torch.device("cuda", target_index)
    config = CompressionConfig(bit=8, group_size=64, error_feedback=True)
    prepared = torch.linspace(-3.0, 3.0, 257, device=target, dtype=torch.float16)
    payload = quantize_tensor(prepared, config, extension_status=status)
    restored = dequantize_tensor(
        payload,
        tuple(prepared.shape),
        config,
        dtype="fp16",
        extension_status=status,
    )
    residual = torch.empty_like(prepared)
    update_error_feedback_residual(
        prepared,
        restored,
        residual,
        extension_status=status,
    )

    torch.cuda.synchronize(target)
    if prepared.device != target or payload.device != target or restored.device != target:
        raise AssertionError("native operation moved a tensor to the current CUDA device")
    if torch.cuda.current_device() != current_index:
        raise AssertionError("CUDAGuard did not restore the caller's current device")
    torch.testing.assert_close(residual, prepared - restored, rtol=0, atol=0)
    if not torch.isfinite(restored).all():
        raise AssertionError("dequantized tensor contains non-finite values")


if __name__ == "__main__":
    main()
