import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from benchmarks.cifar10.model import build_model
from benchmarks.cifar10.sync import FlatGradientSynchronizer


def test_resnet18_uses_cifar_stem():
    model = build_model()
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_pack_unpack_preserves_gradients():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    model(torch.randn(5, 4)).sum().backward()
    expected = [parameter.grad.clone() for parameter in model.parameters()]
    synchronizer = FlatGradientSynchronizer(model, mode="nccl_fp32")
    flat = synchronizer.pack()
    for parameter in model.parameters():
        parameter.grad.zero_()
    synchronizer.unpack(flat)
    assert all(
        torch.equal(parameter.grad, reference)
        for parameter, reference in zip(model.parameters(), expected)
    )
