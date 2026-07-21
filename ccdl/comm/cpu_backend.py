import pickle
import torch
import torch.distributed as dist
from typing import List
from copy import deepcopy

_CPU_GROUP = None
_CPU_RANK_TO_GPU_RANK = None
_GPU_RANK_TO_CPU_RANK = None
_GPU_GROUP_TO_CPU_GROUP = {}

def init():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    _rank = dist.get_rank()

    global _CPU_GROUP
    _CPU_GROUP = dist.new_group(backend='gloo')
    tmp = {dist.get_rank(_CPU_GROUP): dist.get_rank()}
    tmp_list = [None] * dist.get_world_size()
    dist.all_gather_object(tmp_list, tmp, group = _CPU_GROUP)
    for t in tmp_list:
        tmp.update(t)

    global _CPU_RANK_TO_GPU_RANK
    global _GPU_RANK_TO_CPU_RANK
    _CPU_RANK_TO_GPU_RANK = tmp
    _GPU_RANK_TO_CPU_RANK = {}
    for k, v in tmp.items():
        _GPU_RANK_TO_CPU_RANK[v] = k

    global _GPU_GROUP_TO_CPU_GROUP
    _GPU_GROUP_TO_CPU_GROUP[tuple(range(dist.get_world_size()))] = _CPU_GROUP


    assert dist.get_rank() == _rank


def send_cpu(obj, dst=None, group=None, tag=0, _cpu_dst=None):
    if _CPU_GROUP is None:
        init()

    if _cpu_dst is None:
        _cpu_dst = _GPU_RANK_TO_CPU_RANK[dst]

    bytes_tensor = torch.ByteTensor(list(pickle.dumps(obj)))
    length_tensor = torch.tensor(len(bytes_tensor), dtype=torch.int64)
    dist.send(length_tensor, _cpu_dst, group = _CPU_GROUP, tag = tag)
    dist.send(bytes_tensor, _cpu_dst, group = _CPU_GROUP, tag = tag)


def recv_cpu(src=None, group=None, tag=0, _cpu_src=None):
    if _CPU_GROUP is None:
        init()

    if _cpu_src is None:
        _cpu_src = _GPU_RANK_TO_CPU_RANK[src]

    length_tensor = torch.empty((1,), dtype=torch.int64)
    dist.recv(length_tensor, src=_cpu_src, group=_CPU_GROUP, tag=tag)
    length = length_tensor.item()
    bytes_tensor = torch.ByteTensor(length)
    dist.recv(bytes_tensor, src=_cpu_src, group=_CPU_GROUP, tag=tag)

    obj_bytes = bytes(bytes_tensor.tolist())
    obj = pickle.loads(obj_bytes)
    return obj

def register_cpu_group(rank_list: List[int]):
    global _GPU_GROUP_TO_CPU_GROUP
    rank_list = list(rank_list)
    rank_list.sort()
    if tuple(rank_list) in _GPU_GROUP_TO_CPU_GROUP:
        return
    new_cpu_group = dist.new_group(rank_list, backend="gloo")
    _GPU_GROUP_TO_CPU_GROUP[tuple(rank_list)] = new_cpu_group


def all_gather_cpu(obj, group=None):
    if _CPU_GROUP is None:
        init()
    if group is not None:
        rank_list = dist.get_process_group_ranks(group)
    else:
        rank_list = list(range(dist.get_world_size()))

    obj_list = [None] * len(rank_list)

    origin_rank_list = deepcopy(rank_list)
    rank_list.sort()
    if tuple(rank_list) in _GPU_GROUP_TO_CPU_GROUP:
        cpu_group = _GPU_GROUP_TO_CPU_GROUP[tuple(rank_list)]
        dist.all_gather_object(obj_list, obj, group=cpu_group)
        rank2index = {rank: i for i, rank in enumerate(rank_list)}
        obj_list = [obj_list[rank2index[rank]] for rank in origin_rank_list]

    else:
        rank_list = origin_rank_list
        rank_list = [_GPU_RANK_TO_CPU_RANK[rank] for rank in rank_list]

        self_cpu_rank = _GPU_RANK_TO_CPU_RANK[dist.get_rank()]
        for i, rank in enumerate(rank_list):
            if rank == self_cpu_rank:
                for j in range(len(rank_list)):
                    if j == i:
                        obj_list[j] = obj
                    else:
                        obj_list[j] = recv_cpu(_cpu_src=rank_list[j], tag=rank_list[j])

            else:
                send_cpu(obj, _cpu_dst=rank, tag=self_cpu_rank)

    return obj_list