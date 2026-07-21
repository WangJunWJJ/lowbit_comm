import torch
import torch.distributed as dist
from typing import List
from ..quantization import Quantizer, DEFAULT_QUANTIZER
from .async_op import Work

def _ring_reduce_scatter(output: torch.Tensor, input_list: List[torch.Tensor], op="sum", group=None, quantizer: Quantizer=None, keep_self=False):
    world_size = dist.get_world_size(group)
    assert len(input_list) == world_size
    self_rank = dist.get_rank(group)
    data_rank = list(range(world_size))
    if group is not None:
        index2rank = dist.get_process_group_ranks(group)
    else:
        index2rank = list(range(world_size))

    same_shape = True
    for i in range(1, world_size):
        if input_list[i].shape != input_list[0].shape:
            same_shape = False
            break
    
    if same_shape:
        q = quantizer.get_q(input_list[0].shape)
        recv_q = torch.empty_like(q)

    for _ in range(world_size - 1):
        data_rank = [(i+1) % world_size for i in data_rank]
        send_index = None
        send_target = None
        recv_index = None
        recv_source = None
        for data_index, src in enumerate(data_rank):
            tgt = (src + 1) % world_size
            if src == self_rank:
                send_index = data_index
                send_target = tgt
            if tgt == self_rank:
                recv_index = data_index
                recv_source = src

        ops = []
        assert send_index is not None
        assert recv_index is not None

        if not same_shape:
            q = quantizer.get_q(input_list[send_index].shape)
            recv_q = quantizer.get_q(input_list[recv_index].shape)

        quantizer.quantize(input_list[send_index], q)
        send_target = index2rank[send_target]
        recv_source = index2rank[recv_source]

        if self_rank % 2 == 0:
            ops.append(
                dist.P2POp(dist.isend, q, send_target, group=group)
            )
            ops.append(
                dist.P2POp(dist.irecv, recv_q, recv_source, group=group)
            )
        else:
            ops.append(
                dist.P2POp(dist.irecv, recv_q, recv_source, group=group)
            )
            ops.append(
                dist.P2POp(dist.isend, q, send_target, group=group)
            )
            

        works = dist.batch_isend_irecv(ops)
        for wk in works:
            wk.wait()

        quantizer.dequantize(recv_q, input_list[recv_index].shape, input_list[recv_index], op)

    output.copy_(input_list[self_rank], non_blocking=True)


def _p2p_reduce_scatter(output: torch.Tensor, input_list: List[torch.Tensor], op="sum", group=None, quantizer: Quantizer=None, keep_self=False):
    world_size = dist.get_world_size(group)
    assert world_size > 0 and (world_size & (world_size - 1)) == 0
    self_rank = dist.get_rank(group)
    same_shape = True
    for i in range(1, world_size):
        if input_list[i].shape != input_list[0].shape:
            same_shape = False
            break

    if group is not None:
        index2rank = dist.get_process_group_ranks(group)
    else:
        index2rank = list(range(world_size))

    if same_shape:
        q = quantizer.get_q(input_list[0].shape)

    recv_q = quantizer.get_q(input_list[self_rank].shape)

    for offset in range(1, world_size):
        target = self_rank ^ offset
        ops = []
        if not same_shape:
            q = quantizer.get_q(input_list[target].shape)

        quantizer.quantize(input_list[target], q)

        target_rank = index2rank[target]

        if self_rank < target:
            ops.append(
                dist.P2POp(
                    dist.isend, q, target_rank, group=group
                )
            )
            ops.append(
                dist.P2POp(
                    dist.irecv, recv_q, target_rank, group=group
                )
            )
        else:
            ops.append(
                dist.P2POp(
                    dist.irecv, recv_q, target_rank, group=group
                )
            )
            ops.append(
                dist.P2POp(
                    dist.isend, q, target_rank, group=group
                )
            )

        works = dist.batch_isend_irecv(ops)
        for wk in works:
            wk.wait()

        quantizer.dequantize(recv_q, input_list[self_rank].shape, input_list[self_rank], op)


    output.copy_(input_list[self_rank], non_blocking=True)
       


def qreduce_scatter(output: torch.Tensor, input_list: List[torch.Tensor], op="sum", group=None, async_op=False, quantizer: Quantizer=None, keep_self=False, method=None, target_q = None, target_q_quantizer = None):
    output = output.view(-1)
    input_list = [x.view(-1) for x in input_list]
    if quantizer is None:
        quantizer = DEFAULT_QUANTIZER

    _mean = False
    if op == "mean":
        _mean = True
        op = "sum"

    if hasattr(quantizer, "method"):
        method = quantizer.method

    if method is None:
        method = "p2p"

    with Work(async_op) as work:
        if method == "ring":
            _ring_reduce_scatter(output, input_list, op, group, quantizer, keep_self)
        elif method == "p2p":
            _p2p_reduce_scatter(output, input_list, op, group, quantizer, keep_self)
        else:
            raise ValueError(f"Unsupport method {method} for reduce scatter.")

        if _mean:
            output /= dist.get_world_size(group)

        if target_q is not None:
            assert target_q_quantizer is not None
            target_q_quantizer.quantize(output, target_q)
        
    if async_op:
        return work

    return None
