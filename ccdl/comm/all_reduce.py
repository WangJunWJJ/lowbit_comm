import torch
import torch.distributed as dist
from ..quantization import Quantizer, DEFAULT_QUANTIZER
from .all_gather import _qall_gather_base
from .reduce_scatter import _ring_reduce_scatter, _p2p_reduce_scatter
from .async_op import Work, FakeHandle

def _gather_all_reduce(tensor: torch.Tensor, op="sum", group=None, quantizer: Quantizer = None, keep_self=False):
    assert op == "sum"
    world_size = dist.get_world_size(group)
    self_rank = dist.get_rank(group)
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    tensor_list[self_rank] = tensor
    _qall_gather_base(tensor_list, tensor, group, quantizer, keep_self)
    ret = tensor
    for i in range(0, world_size):
        if i == self_rank:
            continue
        ret += tensor_list[i]
    return ret

def _ring_all_reduce(tensor: torch.Tensor, op="sum", group=None, quantizer: Quantizer = None, keep_self=False):
    world_size = dist.get_world_size(group)
    tensor = tensor.view(-1)
    assert tensor.numel() % world_size == 0
    tensor_list = torch.chunk(tensor, world_size)
    self_rank = dist.get_rank(group)
    _ring_reduce_scatter(tensor_list[self_rank], tensor_list, op, group, quantizer, keep_self)
    _qall_gather_base(tensor_list, tensor_list[self_rank], group, quantizer, keep_self)


def _tree_all_reduce(tensor: torch.Tensor, op="sum", group=None, quantizer: Quantizer = None, keep_self=False):
    world_size = dist.get_world_size(group)
    assert world_size == 2 ** (world_size.bit_length() - 1), "World size must be power of 2"
    offset = 1
    self_rank = dist.get_rank(group)
    q = quantizer.get_q(tensor.shape)
    recv_q = torch.empty_like(q)
    if group is not None:
        index2rank = dist.get_process_group_ranks(group)
    else:
        index2rank = list(range(world_size))

    while offset < world_size:
        quantizer.quantize(tensor, q)

        ops = []
        if (self_rank // offset) % 2 == 0:
            target_rank = index2rank[self_rank + offset]
            ops.append(dist.P2POp(dist.isend, q, target_rank, group=group))
            ops.append(dist.P2POp(dist.irecv, recv_q, target_rank, group=group))

        else:
            target_rank = index2rank[self_rank - offset]
            ops.append(dist.P2POp(dist.irecv, recv_q, target_rank, group=group))
            ops.append(dist.P2POp(dist.isend, q, target_rank, group=group))

        works = dist.batch_isend_irecv(ops)
        if not keep_self:
            quantizer.dequantize(q, tensor.shape, tensor)

        for wk in works:
            wk.wait()
        quantizer.dequantize(recv_q, tensor.shape, tensor, op)
        offset *= 2

def _p2p_all_reduce(tensor, op, group, quantizer, keep_self):
    world_size = dist.get_world_size(group)
    tensor = tensor.view(-1)
    assert tensor.numel() % world_size == 0
    tensor_list = torch.chunk(tensor, world_size)
    self_rank = dist.get_rank(group)
    _p2p_reduce_scatter(tensor_list[self_rank], tensor_list, op, group, quantizer, keep_self)
    _qall_gather_base(tensor_list, tensor_list[self_rank], group, quantizer, keep_self)


def qall_reduce(tensor: torch.Tensor, op="sum", group=None, async_op: bool = False, quantizer: Quantizer=None, keep_self = False, method=None, target_q=None, target_q_quantizer=None):
    if quantizer is None:
        quantizer = DEFAULT_QUANTIZER

    if hasattr(quantizer, "method"):
        method = quantizer.method

    if method is None:
        if dist.get_world_size(group) == 2:
            method = "tree"
        else:
            method = "p2p"

    if op == "mean":
        tensor /= dist.get_world_size(group)
        op = "sum"

    if dist.get_world_size(group) == 1:
        if async_op:
            return FakeHandle()
        return None

    if method == "overlap-gather":
        assert target_q is None
        return overlap_all_gather_qall_reduce(tensor, op, group, async_op, quantizer, keep_self, "gather")
    
    if method == "overlap-p2p":
        assert target_q is None
        return overlap_p2p_qall_reduce(tensor, op, group, async_op, quantizer, keep_self, "gather")

    if method == "overlap-tree":
        assert target_q is None
        return overlap_tree_qall_reduce(tensor, op, group, async_op, quantizer, keep_self, "gather")
    
    if method == "overlap-scale":
        assert target_q is None
        return overlap_scale_qall_reduce(tensor, op, group, async_op, quantizer,)

    with Work(async_op) as handle:
        if method == "tree":
            _tree_all_reduce(tensor, op, group, quantizer, keep_self)
        elif method == "ring":
            _ring_all_reduce(tensor, op, group, quantizer, keep_self)
        elif method == "p2p":
            _p2p_all_reduce(tensor, op, group, quantizer, keep_self)
        elif method == "gather":
            _gather_all_reduce(tensor, op, group, quantizer, keep_self)
        else:
            raise ValueError(f"Unknown method {method}")

        if target_q is not None:
            assert target_q_quantizer.compact
            target_q_quantizer.quantize(tensor, target_q)

    if async_op:
        return handle
    
    return None

def overlap_all_gather_qall_reduce(tensor: torch.Tensor, op="sum", group=None, async_op: bool = True, quantizer: Quantizer=None, keep_self=False, method="gather"):
    assert async_op
    assert method == "gather"
    assert quantizer is not None
    assert not keep_self
    assert op == "sum"
    q = quantizer.quantize(tensor)
    world_size = dist.get_world_size(group)
    q_len = q.size(0)
    q_buff = torch.empty((q_len*world_size,), dtype=q.dtype, device=q.device)
    q_list = [
        q_buff[q_len*i:q_len*(i+1)] for i in range(world_size)
    ]

    all_gather_handle = dist.all_gather_into_tensor(q_buff, q, group, True)
    quantizer.dequantize(q, tensor.shape, tensor)

    class Handle:
        def __init__(self, all_gather_handle, tensor, q_list, quantizer, rank):
            self.all_gather_handle = all_gather_handle
            self.tensor = tensor
            self.q_list = q_list
            self.qtz = quantizer
            self.rank = rank

        def wait(self):
            self.all_gather_handle.wait()
            for rk, q in enumerate(self.q_list):
                if rk != self.rank:
                    self.qtz.dequantize(q, self.tensor.shape, self.tensor, "sum")

    return Handle(all_gather_handle, tensor, q_list, quantizer, dist.get_rank(group))

def overlap_p2p_qall_reduce(tensor: torch.Tensor, op="sum", group=None, async_op: bool = True, quantizer: Quantizer=None, keep_self=False, method="gather"):
    assert async_op
    world_size = dist.get_world_size(group)
    tensor = tensor.view(-1)
    assert tensor.numel() % world_size == 0
    tensor_list = torch.chunk(tensor, world_size)
    self_rank = dist.get_rank(group)
    _ring_reduce_scatter(tensor_list[self_rank], tensor_list, op, group, quantizer, keep_self)
    q = quantizer.quantize(tensor_list[self_rank])
    q_len = q.size(0)
    q_buff = torch.empty((q_len * world_size), dtype=q.dtype, device=q.device)
    q_list = [
        q_buff[q_len*i:q_len*(i+1)] for i in range(world_size)
    ]
    all_gahter_handle = dist.all_gather_into_tensor(q_buff, q, group, True)
    quantizer.dequantize(q, tensor_list[self_rank].shape, tensor_list[self_rank])
    
    class Handle:
        def __init__(self, all_gather_handle, tensor_list, q_list, quantizer, rank):
            self.all_gather_handle = all_gather_handle
            self.tensor_list = tensor_list
            self.q_list = q_list
            self.quantizer = quantizer
            self.rank = rank

        def wait(self):
            self.all_gather_handle.wait()
            for rk, q in enumerate(self.q_list):
                if rk != self.rank:
                    self.quantizer.dequantize(q, self.tensor_list[rk].shape, self.tensor_list[rk])

    return Handle(all_gahter_handle, tensor_list, q_list, quantizer, self_rank)

def overlap_tree_qall_reduce(tensor: torch.Tensor, op="sum", group=None, async_op: bool = True, quantizer: Quantizer=None, keep_self=False, method="gather"):
    assert async_op
    world_size = dist.get_world_size(group)
    assert world_size == 2 ** (world_size.bit_length() - 1), "World size must be power of 2"
    offset = 1
    self_rank = dist.get_rank(group)
    q = quantizer.get_q(tensor.shape)
    recv_q = torch.empty_like(q)
    if group is not None:
        index2rank = dist.get_process_group_ranks(group)
    else:
        index2rank = list(range(world_size))

    while offset < world_size:
        quantizer.quantize(tensor, q)

        ops = []
        if (self_rank // offset) % 2 == 0:
            target_rank = index2rank[self_rank + offset]
            ops.append(dist.P2POp(dist.isend, q, target_rank, group=group))
            ops.append(dist.P2POp(dist.irecv, recv_q, target_rank, group=group))

        else:
            target_rank = index2rank[self_rank - offset]
            ops.append(dist.P2POp(dist.irecv, recv_q, target_rank, group=group))
            ops.append(dist.P2POp(dist.isend, q, target_rank, group=group))

        works = dist.batch_isend_irecv(ops)
        if not keep_self:
            quantizer.dequantize(q, tensor.shape, tensor)

        if offset * 2 >= world_size:
            break

        for wk in works:
            wk.wait()
        quantizer.dequantize(recv_q, tensor.shape, tensor, op)
        offset *= 2

    class Handle:
        def __init__(self, works, recv_q, quantizer, tensor, op):
            self.works = works
            self.recv_q = recv_q
            self.quantizer = quantizer
            self.tensor = tensor
            self.op = op
        
        def wait(self):
            for wk in self.works:
                wk.wait()
            self.quantizer.dequantize(self.recv_q, self.tensor.shape, self.tensor, self.op)

    return Handle(works, recv_q, quantizer, tensor, op)

def overlap_scale_qall_reduce(tensor: torch.Tensor, op="sum", group=None, async_op: bool = True, quantizer: Quantizer = None, keep_self = False, method="all-reduce"):
    assert async_op
    group_size = quantizer.group_size
    abs_tensor = tensor.abs().view(-1, group_size)
    scale = abs_tensor.max(dim=-1, keepdim=True).values / 127
    dist.all_reduce(scale, dist.ReduceOp.MAX, group=group, async_op=False)
    q = (tensor.view(-1, group_size) / scale).to(torch.int8)
    all_reduce_handle = dist.all_reduce(q, op=dist.ReduceOp.SUM, group=group, async_op=True)

    class Handle:
        def __init__(self, all_reduce_handle, q, scale, tensor):
            self.all_reduce_handle = all_reduce_handle
            self.q = q
            self.tensor = tensor
            self.scale = scale
        
        def wait(self):
            self.all_reduce_handle.wait()
            assert self.q.dim() == 2
            torch.mul(q, scale, out=self.tensor)

    return Handle(all_reduce_handle, q, scale, tensor)