from __future__ import annotations

import os

import torch
import torch.distributed as dist

from ccdl_comm.communication.cuda_completion import CudaCompletionManager
from ccdl_comm.communication.ddp_hook import create_ddp_comm_hook
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension


class TensorBucket:
    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def index(self) -> int:
        return 0

    def buffer(self) -> torch.Tensor:
        return self._tensor


class CountingCompletion:
    def __init__(self, delegate: object, counters: dict[str, int]) -> None:
        self._delegate = delegate
        self._counters = counters

    def wait_stream(self, stream: object) -> None:
        self._counters["wait_stream"] += 1
        self._delegate.wait_stream(stream)

    def synchronize(self) -> None:
        self._counters["synchronize"] += 1
        self._delegate.synchronize()


class CountingCompletionManager(CudaCompletionManager):
    def __init__(self, counters: dict[str, int], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._counters = counters

    def record_for(self, tensor: object, *, stream: object | None = None) -> CountingCompletion:
        return CountingCompletion(super().record_for(tensor, stream=stream), self._counters)


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    status = load_cuda_extension()
    if not status.available:
        raise RuntimeError(status.reason or "CCDL CUDA extension unavailable")

    counters = {"wait_stream": 0, "synchronize": 0}
    manager = CountingCompletionManager(counters, extension_status=status)
    config = CompressionConfig(bit=8, group_size=64, error_feedback=True)
    hook = create_ddp_comm_hook(
        config,
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        async_gather=True,
        async_error_feedback=True,
        completion_manager=manager,
        extension_status=status,
    )
    gradient = torch.linspace(-1.0, 1.0, 1 << 18, device=device, dtype=torch.float16)
    gradient.mul_(rank + 1)

    result = hook(None, TensorBucket(gradient)).wait()

    if counters != {"wait_stream": 1, "synchronize": 0}:
        raise AssertionError(f"unexpected completion counters: {counters}")
    if not torch.isfinite(result).all():
        raise AssertionError("asynchronous result contains non-finite values")
    checksum = result.float().sum()
    gathered = [torch.empty_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, checksum)
    torch.testing.assert_close(torch.stack(gathered), gathered[0].expand(len(gathered)), rtol=0, atol=0)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
