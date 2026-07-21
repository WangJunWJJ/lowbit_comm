import torch
import torch.distributed as dist
from typing import List
from .cpu_backend import all_gather_cpu
from .async_op import Work, FakeHandle
from ..quantization import Quantizer, DEFAULT_QUANTIZER

def _qall_gather_base(tensor_list: List[torch.Tensor], tensor: torch.Tensor, group=None, quantizer: List[Quantizer] = None, keep_self = False):
    if quantizer is None:
        quantizer = DEFAULT_QUANTIZER

    if isinstance(quantizer, Quantizer):
        quantizer = [quantizer] * len(tensor_list)

    world_size = dist.get_world_size(group)
    assert len(quantizer) == len(tensor_list)
    assert len(quantizer) == world_size

    if world_size == 1:
        return

    self_rank = dist.get_rank(group)
    q_len_list = [
        qtz.get_lenq(ts.shape) for (qtz, ts) in zip(quantizer, tensor_list)
    ]
    q_buffer = torch.empty(
        (sum(q_len_list),), dtype=torch.uint8, device=tensor.device
    )
    q_list = []
    _st = 0
    for q_len in q_len_list:
        q_list.append(q_buffer[_st: _st+q_len])
        _st += q_len

    quantizer[self_rank].quantize(tensor, q_list[self_rank])
    q = q_list[self_rank]

    handle = dist.all_gather_into_tensor(q_buffer, q, group=group, async_op=True)
    handle.wait()
    for rank, (qtz, ts, _q) in enumerate(zip(quantizer, tensor_list, q_list)):
        if keep_self and rank == self_rank:
            ts.copy_(tensor, non_blocking=True)
        else:
            qtz.dequantize(_q, ts.shape, ts)


def qall_gather(tensor_list: List[torch.Tensor], tensor: torch.Tensor, group=None, async_op = False, quantizer: List[Quantizer] = None, keep_self = False):

    if dist.get_world_size(group) == 1:
        if async_op:
            return FakeHandle()
        return None

    with Work(async_op) as work:
        _qall_gather_base(tensor_list, tensor, group, quantizer, keep_self)

    if async_op:
        return work
    return None


def qall_gather_dyn(tensor: torch.Tensor, group=None, quantizer: Quantizer = None, keep_self = False):
    if quantizer is None:
        quantizer = DEFAULT_QUANTIZER

    q = quantizer.quantize(tensor)
    _tmp_list = all_gather_cpu((tensor.shape, quantizer.to_dict()), group=group)
    shape_list = [t[0] for t in _tmp_list]
    quantizer_list = [Quantizer.from_dict(t[1]) for t in _tmp_list]
    q_list = [qtz.get_q(s) for s, qtz in zip(shape_list, quantizer_list)]
    dist.all_gather(q_list, q, group=group)
    tensor_list = []

    for rank, (shape, qtz, q) in enumerate(zip(shape_list, quantizer_list, q_list)):
        if keep_self and rank == dist.get_rank(group):
            tensor_list.append(tensor)
        else:
            tensor_list.append(qtz.dequantize(q, shape))

    return tensor_list