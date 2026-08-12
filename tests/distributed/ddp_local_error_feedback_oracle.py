from __future__ import annotations

import os

import torch
import torch.distributed as dist

from ccdl_comm.communication.ddp_hook import create_ddp_comm_hook
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.quantization.codec import dequantize_tensor, quantize_tensor
from ccdl_comm.quantization.error_feedback import ErrorFeedbackState


class TensorBucket:
    """Minimal GradBucket-compatible object for the distributed oracle."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def index(self) -> int:
        return 0

    def buffer(self) -> torch.Tensor:
        return self._tensor


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)

    config = CompressionConfig(bit=8, group_size=64, error_feedback=True)
    extension_status = load_cuda_extension()
    if not extension_status.available:
        raise RuntimeError(extension_status.reason or "CCDL CUDA extension unavailable")

    prepared = torch.linspace(-1.0, 1.0, 257, device=device, dtype=torch.float16) * (rank + 1)
    feedback = ErrorFeedbackState()
    hook = create_ddp_comm_hook(
        config,
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        error_feedback=feedback,
        extension_status=extension_status,
    )

    restored_global = hook(None, TensorBucket(prepared)).wait()
    local_payload = quantize_tensor(prepared, config, extension_status=extension_status)
    local_restored = dequantize_tensor(
        local_payload,
        tuple(prepared.shape),
        config,
        dtype="fp16",
        extension_status=extension_status,
    )
    expected_residual = prepared - local_restored
    torch.testing.assert_close(feedback.get(0), expected_residual, rtol=0, atol=0)

    gathered = [torch.empty_like(local_restored) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_restored)
    expected_global = torch.stack(gathered).mean(dim=0)
    torch.testing.assert_close(restored_global, expected_global, rtol=2e-3, atol=2e-3)

    checksum = restored_global.float().sum()
    checksums = [torch.empty_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(checksums, checksum)
    torch.testing.assert_close(torch.stack(checksums), checksums[0].expand_as(torch.stack(checksums)))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
