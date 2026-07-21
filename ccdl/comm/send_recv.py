import torch
import torch.distributed as dist
from .cpu_backend import send_cpu, recv_cpu
from .async_op import Work
from ..quantization import Quantizer, DEFAULT_QUANTIZER

def qsend(tensor: torch.Tensor, dst, group=None, tag=0, quantizer: Quantizer=None):
    if quantizer is None:
        quantizer = DEFAULT_QUANTIZER

    if quantizer.is_quantized():
        q = quantizer.quantize(tensor)
        dist.send(q, dst, group=group, tag=tag)
    else:
        dist.send(tensor, dst, group=group, tag=tag)

def qrecv(tensor: torch.Tensor, src=None, group=None, tag=0, quantizer: Quantizer=None):
    if quantizer is None:
        quantizer = DEFAULT_QUANTIZER

    if quantizer.is_quantized():
        q = quantizer.get_q(tensor.shape)
        dist.recv(q, src, group=group, tag=tag)
        tensor.copy_(quantizer.dequantize(q, tensor.shape))
    else:
        dist.recv(tensor, src, group=group, tag=tag)

def qsend_dyn(tensor: torch.Tensor, dst, group=None, tag=0, quantizer: Quantizer=None):   
    if quantizer is None:
        quantizer = DEFAULT_QUANTIZER

    send_cpu((tensor.shape, quantizer.to_dict()), dst, group=group, tag=tag)
    q = quantizer.quantize(tensor)
    dist.send(q, dst, tag=tag)

def qrecv_dyn(src, group=None, tag=0):
    shape, quantizer_dict = recv_cpu(src, group=group, tag=tag)
    quantizer = Quantizer.from_dict(quantizer_dict)
    q = quantizer.get_q(shape)
    dist.recv(q, src, group=group, tag=tag)
    return quantizer.dequantize(q, shape)

def iqsend(tensor: torch.Tensor, dst, group=None, tag=0, quantizer: Quantizer=None):
    if quantizer is None:
        quantizer = DEFAULT_QUANTIZER

    if quantizer.is_quantized():
        q = quantizer.quantize(tensor)
        return dist.isend(q, dst, group=group, tag=tag)
    else:
        return dist.isend(tensor, dst, group=group, tag=tag)


def iqrecv(tensor: torch.Tensor, src, group= None, tag=0, quantizer: Quantizer=None):
    if quantizer is None:
        quantizer = DEFAULT_QUANTIZER

    if quantizer.is_quantized():
        with Work(True) as work:
            q = quantizer.get_q(tensor.shape)
            handle = dist.irecv(q, src, group=group, tag=tag)
            handle.wait()
            quantizer.dequantize(q, tensor.shape, tensor)
        return work
    else:
        return dist.irecv(tensor, src, group=group, tag=tag)
