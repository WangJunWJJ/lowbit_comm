from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class _GradientSlice:
    parameter: torch.nn.Parameter
    offset: int
    numel: int


class FlatGradientSynchronizer:
    def __init__(
        self,
        model: torch.nn.Module,
        mode: str,
        bit: int | None = None,
        topk: int | None = None,
        group_size: int = 64,
    ):
        if mode != "nccl_fp32" and not mode.startswith("ccdl"):
            raise ValueError(f"unsupported synchronization mode: {mode}")
        self.model = model
        self.mode = mode
        self.bit = bit
        self.topk = topk
        self.group_size = group_size
        self._slices: list[_GradientSlice] = []
        self._quantizer = None

    def pack(self) -> torch.Tensor:
        gradients = []
        self._slices.clear()
        offset = 0
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            flat = parameter.grad.detach().reshape(-1)
            gradients.append(flat)
            self._slices.append(_GradientSlice(parameter, offset, flat.numel()))
            offset += flat.numel()
        if not gradients:
            raise RuntimeError("model has no gradients to synchronize")
        flat = torch.cat(gradients).contiguous()
        if self.mode.startswith("ccdl") and flat.numel() % self.group_size:
            padding = self.group_size - flat.numel() % self.group_size
            flat = torch.nn.functional.pad(flat, (0, padding))
        return flat

    def synchronize(self, flat: torch.Tensor) -> torch.Tensor:
        if self.mode == "nccl_fp32":
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(dist.get_world_size())
            return flat
        if flat.numel() % self.group_size:
            raise ValueError("flat gradient length must be divisible by group_size")
        if self._quantizer is None:
            from ccdl.comm import qall_reduce
            from ccdl.quantization import Quantizer

            self._qall_reduce = qall_reduce
            self._quantizer = Quantizer(
                self.group_size, -1, self.bit, self.topk, False, "fp32"
            )
        self._qall_reduce(
            flat,
            op="mean",
            quantizer=self._quantizer,
            method="tree",
            keep_self=False,
            async_op=False,
        )
        return flat

    def unpack(self, flat: torch.Tensor) -> None:
        for gradient_slice in self._slices:
            view = flat.narrow(
                0, gradient_slice.offset, gradient_slice.numel
            ).view_as(gradient_slice.parameter.grad)
            gradient_slice.parameter.grad.copy_(view)
