import pytest

torch = pytest.importorskip("torch")

from benchmarks.lm.sync import FlatGradientSynchronizer


def test_flat_gradient_copy_back_without_distributed_sync():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, float(index + 1))
    expected = [parameter.grad.clone() for parameter in model.parameters()]
    synchronizer = FlatGradientSynchronizer("nccl_fp32", world_size=1)
    metrics = synchronizer.synchronize(model)
    assert all(torch.equal(parameter.grad, value) for parameter, value in zip(model.parameters(), expected))
    assert metrics.numel == sum(parameter.numel() for parameter in model.parameters())
    assert metrics.elapsed_ms >= 0


def test_sync_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unknown variant"):
        FlatGradientSynchronizer("unknown", world_size=1)
