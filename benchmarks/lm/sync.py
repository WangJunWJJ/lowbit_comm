from dataclasses import dataclass
import time

from .config import MAIN_VARIANTS


@dataclass(frozen=True)
class SyncMetrics:
    elapsed_ms: float
    numel: int
    communicated_bytes: int


class FlatGradientSynchronizer:
    def __init__(self, variant: str, world_size: int, group_size: int = 64):
        if variant not in MAIN_VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.world_size = world_size
        self.group_size = group_size
        self.bit, self.topk = MAIN_VARIANTS[variant]
        self._quantizer = None

    def synchronize(self, model) -> SyncMetrics:
        import torch
        import torch.distributed as dist

        parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
        flat = torch.cat([parameter.grad.detach().reshape(-1).float() for parameter in parameters])
        if flat.is_cuda:
            torch.cuda.synchronize(flat.device)
        started = time.perf_counter()
        if self.world_size > 1:
            if self.variant == "nccl_fp32":
                dist.all_reduce(flat)
            else:
                from ccdl.comm import qall_reduce
                from ccdl.quantization import Quantizer

                if self._quantizer is None:
                    self._quantizer = Quantizer(self.group_size, -1, self.bit, self.topk, False, "fp32")
                qall_reduce(flat, op="sum", quantizer=self._quantizer, method="tree", keep_self=False)
            flat.div_(self.world_size)
        if flat.is_cuda:
            torch.cuda.synchronize(flat.device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.grad.copy_(flat[offset:offset + count].view_as(parameter).to(parameter.grad.dtype))
            offset += count
        if self.bit is None:
            communicated_bytes = flat.numel() * flat.element_size()
        else:
            communicated_bytes = (flat.numel() * self.bit + 7) // 8
        return SyncMetrics(elapsed_ms, flat.numel(), communicated_bytes)
