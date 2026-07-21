from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from benchmarks.cifar10 import data


def test_nonzero_rank_waits_for_rank_zero_download(monkeypatch, tmp_path):
    events = []

    class FakeDataset:
        def __init__(self, root, train, download, transform):
            events.append(("dataset", train, download))

        def __len__(self):
            return 16

    monkeypatch.setattr(data.datasets, "CIFAR10", FakeDataset)
    monkeypatch.setattr(data.dist, "barrier", lambda: events.append(("barrier",)))
    monkeypatch.setattr(data, "DataLoader", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(data, "DistributedSampler", lambda *args, **kwargs: object())
    config = SimpleNamespace(seed=1, batch_size_per_rank=2, workers_per_rank=0)

    data.build_loaders(tmp_path, config, rank=1, world_size=2)

    assert events == [
        ("barrier",),
        ("dataset", True, False),
        ("dataset", False, False),
    ]
