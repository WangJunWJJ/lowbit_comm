import torch
from typing import Optional
from copy import deepcopy

from .quant import tensor_dequantize, tensor_quantize
from .dummy import fake_tensor_dequantize, fake_tensor_quantize

class Quantizer:
    def __init__(self, group_size, dim, bit, topk, stochastic, dtype, q=None, dummy=False, buffer=False, quant_type="linear", compact=False, **kwargs):
        self.group_size = group_size
        self.dim = dim
        self.bit = bit
        self.topk = topk
        self.stochastic = stochastic
        self.dtype = dtype
        self.q = q
        self.dummy = dummy
        self.buffer = buffer
        self.quant_type = quant_type
        self.compact = compact
        for k, v in kwargs.items():
            setattr(self, k, v)
        if dummy:
            import warnings
            warnings.warn("Using dummy quantizer, this is only for testing purpose.")

    @classmethod 
    def from_dict(cls, dict):
        return cls(
            dict.get("group_size", 32), 
            dict.get("dim", -1), 
            dict.get("bit", 8), 
            dict.get("topk", 0), 
            dict.get("stochastic", False), 
            dict.get("dtype", "fp16"),
            dummy=dict.get("dummy", False),
            quant_type=dict.get("quant_type", "linear"),
            compact=dict("compact", False),
            buffer=False # buffer always False when loading from dict
        )

    def to_dict(self):
        return {
            "group_size": self.group_size,
            "dim": self.dim,
            "bit": self.bit,
            "topk": self.topk,
            "stochastic": self.stochastic,
            "dtype": self.dtype,
            "dummy": self.dummy,
            "quant_type": self.quant_type,
            "compact": self.compact
        }

    def quantize(self, tensor: torch.Tensor, output: Optional[torch.Tensor] = None, **kwargs):
        args = {
            "tensor": tensor,
            "dim": self.dim,
            "group_size": self.group_size,
            "bit": self.bit,
            "stochastic": self.stochastic,
            "topk": self.topk,
            "output": output,
            "dtype": self.dtype,
            "dummy": self.dummy,
            "quant_type": self.quant_type,
            "compact": self.compact
        }
        args.update(kwargs)

        if args["bit"] > 8:
            return tensor

        STR_TO_TORCH_DTYPE = {
            "fp16": torch.half,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }
        assert tensor.dtype == STR_TO_TORCH_DTYPE[args["dtype"]]
        return (fake_tensor_quantize if args["dummy"] else tensor_quantize)(
            args["tensor"], args["dim"], args["group_size"], args["bit"], args["stochastic"], args["topk"], args["output"], args["quant_type"], args["compact"]
        ) 

    def dequantize(self, q: torch.Tensor, shape, output: Optional[torch.Tensor] = None, reduce_op = None, **kwargs):
        args = {
            "q": q,
            "dim": self.dim,
            "shape": shape,
            "group_size": self.group_size,
            "bit": self.bit,
            "topk": self.topk,
            "output": output,
            "dtype": self.dtype,
            "dummy": self.dummy,
            "reduce_op": reduce_op,
            "quant_type": self.quant_type,
            "compact": self.compact
        }
        args.update(kwargs)

        if args["bit"] > 8:
            return q.reshape(shape)

        return (fake_tensor_dequantize if self.dummy else tensor_dequantize)(
            args["q"], args["dim"], args["shape"], args["group_size"], args["bit"], args["dtype"], args["topk"], args["output"], args["reduce_op"], args["quant_type"], args["compact"]
        )

    def get_lenq(self, shape):
        if self.bit > 8:
            return 0
        shape = deepcopy(list(shape))
        dim = self.dim
        if dim < 0:
            dim += len(shape)
        if dim != len(shape) - 1:
            shape[dim], shape[-1] = shape[-1], shape[dim]
        
        numel = 1
        for i in shape:
            numel *= i
        bytes_per_group = self.get_qlen_per_group()

        if self.dummy:
            if self.dtype == "fp32":
                bytes_per_group = self.group_size + (self.topk * 2 + 1) * 4
            else:
                bytes_per_group = self.group_size + (self.topk * 2 + 1) * 2

        num_group = numel // self.group_size

        return num_group * bytes_per_group

    def get_group_size(self):
        return self.group_size

    def get_qlen_per_group(self):
        if self.dtype == "fp32":
            bytes_per_group = self.group_size * self.bit // 8 + 4
            if self.topk == 1:
                bytes_per_group += 8
            elif self.topk == 2:
                bytes_per_group += 12

        else:
            bytes_per_group = self.group_size * self.bit // 8 + 2
            if self.topk == 1:
                bytes_per_group += 4
            elif self.topk == 2:
                bytes_per_group += 6

        return bytes_per_group

        

    def is_quantized(self):
        return self.bit < 16

    def get_q(self, shape, device='cuda'):
        lenq = self.get_lenq(shape)
        if self.q is None or len(self.q) != lenq:
            q = torch.empty((lenq,), device=device, dtype=torch.uint8)
            if self.buffer:
                self.register_q(q)
            return q
        else:
            return self.q

    def register_q(self, q):
        self.q = q
    

DEFAULT_QUANTIZER = Quantizer(None, None, 16, None, None, None)

def update_default_quantizer(quantizer: Quantizer):
    global DEFAULT_QUANTIZER
    DEFAULT_QUANTIZER = quantizer