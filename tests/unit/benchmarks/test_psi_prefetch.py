"""Real-process contracts for positional stochastic data replay."""

import importlib
import importlib.util
import random

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


class StochasticDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        return torch.tensor(
            [index, random.random(), np.random.random(), torch.rand(()).item()],
            dtype=torch.float64,
        )


class RaisingDataset(StochasticDataset):
    def __getitem__(self, index):
        super().__getitem__(index)
        raise RuntimeError("sample failure")


def tagged_collate(samples):
    return {"samples": torch.stack(samples)}


def factory():
    assert importlib.util.find_spec("tests.benchmarks.psi_prefetch") is not None, (
        "positional deterministic prefetch implementation is missing"
    )
    return importlib.import_module("tests.benchmarks.psi_prefetch").build_prefetch_loader


def loader(workers=0, indices=(3, 1, 3, 7, 1, 4, 5), **kwargs):
    source = torch.utils.data.DataLoader(
        StochasticDataset(), batch_size=2, collate_fn=tagged_collate,
        drop_last=kwargs.pop("drop_last", False),
        pin_memory=kwargs.pop("pin_memory", False),
    )
    config = dict(base_seed=71, epoch=2, start_position=0, num_workers=workers,
                  prefetch_factor=2)
    config.update(kwargs)
    return factory()(source, indices, **config)


def flatten(batches):
    return torch.cat([batch["samples"] for batch in batches])


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0]
    assert np.array_equal(a[1][1], b[1][1])
    assert a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])


def test_real_spawn_workers_match_synchronous_samples_and_order():
    reference = flatten(list(loader()))
    parallel = loader(2)
    assert parallel.multiprocessing_context.get_start_method() == "spawn"
    assert parallel.prefetch_factor == 2
    assert torch.equal(reference, flatten(list(parallel)))
    assert reference[:, 0].tolist() == [3, 1, 3, 7, 1, 4, 5]
    assert not torch.equal(reference[0, 1:], reference[2, 1:])


@pytest.mark.parametrize("workers", [0, 2])
def test_restart_uses_consumed_position_not_prefetch_cursor(workers):
    expected = flatten(list(loader(workers)))
    iterator = iter(loader(workers))
    consumed = [next(iterator), next(iterator)]
    # Multiprocess iterator has dispatched further samples, none are checkpointed.
    del iterator
    resumed = list(loader(workers, start_position=4))
    assert torch.equal(expected, flatten(consumed + resumed))


@pytest.mark.parametrize("workers", [0, 2])
def test_parent_rng_and_augmentation_stream_untouched(workers):
    before = rng_state()
    list(loader(workers))
    assert_rng_equal(before, rng_state())


def test_seed_epoch_changes_and_parent_rng_independence():
    reference = flatten(list(loader()))
    assert not torch.equal(reference, flatten(list(loader(base_seed=72))))
    assert not torch.equal(reference, flatten(list(loader(epoch=3))))
    random.random()
    np.random.random()
    torch.rand(5)
    assert torch.equal(reference, flatten(list(loader())))


@pytest.mark.parametrize("drop_last,expected", [(False, [2, 2, 2, 1]), (True, [2, 2, 2])])
def test_final_batch_and_collate(drop_last, expected):
    result = loader(drop_last=drop_last)
    assert result.collate_fn is tagged_collate
    assert [len(b["samples"]) for b in result] == expected
    assert len(result) == len(expected)
    assert list(loader(start_position=7, drop_last=drop_last)) == []
    assert loader(pin_memory=True).pin_memory is True


@pytest.mark.parametrize("kwargs", [
    {"start_position": -1}, {"start_position": 8}, {"start_position": 1.5},
    {"num_workers": -1}, {"num_workers": True}, {"prefetch_factor": 0},
    {"prefetch_factor": 1.5}, {"epoch": -1}, {"epoch": True},
    {"base_seed": -1}, {"base_seed": 1.5},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        loader(**kwargs)


@pytest.mark.parametrize("indices", [(20,), (-1,), (True,), (1.2,)])
def test_invalid_indices_rejected(indices):
    with pytest.raises(ValueError):
        loader(indices=indices)


def test_exception_restores_cpu_rng():
    source = torch.utils.data.DataLoader(RaisingDataset(), batch_size=2)
    result = factory()(source, (1, 2), base_seed=1, epoch=0)
    before = rng_state()
    with pytest.raises(RuntimeError, match="sample failure"):
        next(iter(result))
    assert_rng_equal(before, rng_state())


def test_unbatched_source_rejected():
    source = torch.utils.data.DataLoader(StochasticDataset(), batch_size=None)
    with pytest.raises(ValueError, match="batch"):
        factory()(source, (1, 2), base_seed=1, epoch=0)
