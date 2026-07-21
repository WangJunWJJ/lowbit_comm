import torch
from copy import deepcopy
from typing import Optional
from ccdl_cuda_ops import quantize, dequantize, inplace_quantize, inplace_dequantize, ReduceOP, QuantType, DType

STR_TO_QUANTTYPE = {
    "linear": QuantType.Linear,
    "normal": QuantType.Normal,
    "uniform": QuantType.Uniform,
    "e3m0": QuantType.E3M0,
    "e2m1": QuantType.E2M1
}

STR_TO_REDUCE_OP = {
    "min": ReduceOP.MIN,
    "max": ReduceOP.MAX,
    "sum": ReduceOP.SUM,
    "none": ReduceOP.NONE
}

STR_TO_DTYPE = {
    "fp16": DType.FP16,
    "bf16": DType.BF16,
    "fp32": DType.FP32
}

def topk_quantize(x: torch.Tensor, group_size, bit, topk, stochastic=False, output: Optional[torch.Tensor] = None, quant_type: Optional[str] = "linear", compact: bool = False) -> torch.Tensor:
    if output is None:
        return quantize(x, group_size, topk, stochastic, bit, STR_TO_QUANTTYPE[quant_type], compact)

    inplace_quantize(x, output, group_size, topk, stochastic, bit, STR_TO_QUANTTYPE[quant_type], compact)
    return output


def topk_dequantize(x: torch.Tensor, group_size, bit, topk, dtype, output: Optional[torch.Tensor] = None, reduce_op: Optional[str] = None, quant_type: Optional[str] = "linear", compact: bool = False) -> torch.Tensor:
    if reduce_op is not None:
        reduce_op = STR_TO_REDUCE_OP[reduce_op.lower()]
    else:
        reduce_op = ReduceOP.NONE

    if output is None:
        assert reduce_op is ReduceOP.NONE, "Output tensor is required when reduce op is not NONE."
        return dequantize(x, group_size, topk, bit, reduce_op, STR_TO_QUANTTYPE[quant_type], STR_TO_DTYPE[dtype], compact)

    
    inplace_dequantize(x, output, group_size, topk, bit, reduce_op, STR_TO_QUANTTYPE[quant_type], compact)
    return output


GROUP_SIZE = [16, 32, 64]
BIT = [4, 8]
TOPK = [0, 1, 2]

def tensor_quantize(x: torch.Tensor, dim: int, group_size: int, bit: int, stochastic=False, topk: int = 0, output_tensor: torch.Tensor = None, quant_type: Optional[str] = "linear", compact=False) -> torch.Tensor:
    assert bit in BIT, f"Only support bit {BIT}."
    assert group_size in GROUP_SIZE, f"Only support group size {GROUP_SIZE}."
    assert topk in TOPK, f"Only support top {TOPK} quantization."
    assert x.dtype in [torch.half, torch.bfloat16, torch.float32], "Only support fp16, bf16 and fp32."
    if dim < 0:
        dim += x.dim()
    
    if dim != x.dim() - 1:
        y = x.transpose(-1, dim).contiguous().view(-1)
    else:
        y = x.contiguous().view(-1)


    return topk_quantize(y, group_size, bit, topk, stochastic, output_tensor, quant_type, compact)


def tensor_dequantize(q: torch.Tensor, dim: int, shape: torch.Size, group_size: int, bit: int, dtype: str, topk: int = 0, output_tensor: torch.Tensor = None, reduce_op = None, quant_type: Optional[str] = "linear", compact=False) -> torch.Tensor:
    assert bit in BIT, f"Only support bit {BIT}."
    assert group_size in GROUP_SIZE, f"Only support group size {GROUP_SIZE}."
    assert topk in TOPK, f"Only support top {TOPK} quantization."
    assert q.dtype in [torch.uint8], "Only support uint8 tensor."

    y = topk_dequantize(q, group_size, bit, topk, dtype, output_tensor, reduce_op, quant_type, compact)
    if output_tensor is None:
        _shape = list(shape)
        if dim < 0:
            dim += len(_shape)

        if dim != len(_shape) - 1:
            _shape[dim], _shape[-1] = _shape[-1], _shape[dim]

        y = y.view(_shape)

        if dim != len(_shape) - 1:
            y = y.transpose(-1, dim).contiguous()

    return y