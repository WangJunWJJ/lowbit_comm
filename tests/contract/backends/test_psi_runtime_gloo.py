"""Real two-process CPU checks; these do not qualify CUDA performance."""

from datetime import timedelta
import socket

import pytest


def _two_rank_training(rank, port):
    import torch
    from tests.benchmarks.psi_training_runtime import (
        FP32MasterWeights,
        RankBatchSampler,
    )

    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}?use_libuv=0",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 1).half())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        master = FP32MasterWeights(optimizer)
        sampler = RankBatchSampler(range(8), rank=rank, world_size=2, batch_size=2)
        seen = list(sampler)
        dataset = torch.arange(16, dtype=torch.float16).view(8, 2) / 16
        loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=2)
        for _ in range(3):
            for batch in loader:
                prediction = model(batch)
                loss = (prediction.float() - 0.25).square().mean()
                loss.backward()
                master.prepare_gradients()
                torch.nn.utils.clip_grad_norm_(master.master_parameters, 1.0)
                optimizer.step()
                master.publish()
                master.zero_grad()
        weights = torch.cat([p.detach().flatten() for p in master.master_parameters])
        all_weights = [torch.empty_like(weights) for _ in range(2)]
        torch.distributed.all_gather(all_weights, weights)
        torch.testing.assert_close(all_weights[0], all_weights[1], rtol=0, atol=0)
        all_indices = [None, None]
        torch.distributed.all_gather_object(all_indices, seen)
        assert set(all_indices[0]).isdisjoint(all_indices[1])
        assert sorted(all_indices[0] + all_indices[1]) == list(range(8))
        for parameter in master.master_parameters:
            assert optimizer.state[parameter]["step"].item() == 6
            assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
            assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_standard_ddp_has_disjoint_data_and_identical_fp32_state():
    torch = pytest.importorskip("torch")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("requires a PyTorch build with Gloo")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(_two_rank_training, args=(port,), nprocs=2, join=True)
