import importlib.util
import inspect
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[3]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fulltensor = _load(
    "observation_fulltensor", "tests/cuda/distributed_fulltensor_worker.py"
)
reduced = _load("observation_reduced", "tests/cuda/distributed_reduced_shard_worker.py")


def test_fulltensor_metrics_use_float64_without_mutating_inputs():
    actual = torch.tensor([1.0000, 0.3333, -0.7777, 0.1251], dtype=torch.float16)
    expected = torch.tensor([1.0000, 0.3330, -0.7780, 0.1250], dtype=torch.float16)
    actual_before = actual.clone()
    expected_before = expected.clone()
    observed = fulltensor._accuracy_metrics(actual, expected)
    actual64 = actual.to(torch.float64)
    expected64 = expected.to(torch.float64)
    wanted = (
        float(
            ((actual64 - expected64).norm() / expected64.norm().clamp_min(1e-12)).item()
        ),
        float(
            torch.nn.functional.cosine_similarity(
                actual64.flatten(), expected64.flatten(), dim=0
            ).item()
        ),
    )
    assert observed == wanted
    assert torch.equal(actual, actual_before) and torch.equal(expected, expected_before)


def test_fulltensor_metrics_identical_are_exact_and_not_clamped():
    value = torch.linspace(-1, 1, 31, dtype=torch.float16)
    assert fulltensor._accuracy_metrics(value, value) == (0.0, 1.0)
    anti = -value
    _, cosine = fulltensor._accuracy_metrics(anti, value)
    assert cosine == pytest.approx(-1.0) and cosine < 0.0


def test_fulltensor_metrics_preserve_nonfinite_observation():
    actual = torch.tensor([float("nan"), 1.0])
    expected = torch.ones(2)
    relative_l2, cosine = fulltensor._accuracy_metrics(actual, expected)
    assert not torch.isfinite(torch.tensor(relative_l2))
    assert not torch.isfinite(torch.tensor(cosine))


def test_benchmark_calls_metric_helper_after_measured_loop():
    source = inspect.getsource(fulltensor._run_benchmark)
    assert source.index("for _ in range(args.iterations)") < source.index(
        "_accuracy_metrics("
    )
    assert source.count("_accuracy_metrics(") == 1


class _Gather:
    def __init__(self, lines):
        self.lines = lines
        self.calls = []

    def all_gather_object(self, output, value):
        self.calls.append(value)
        output[:] = self.lines


def test_reduced_shard_gathers_once_per_rank_and_only_rank0_prints(capsys, monkeypatch):
    lines = [f"REDUCED_SHARD_METRIC rank={rank} value={rank}" for rank in range(4)]
    fake = _Gather(lines)
    monkeypatch.setattr(reduced, "dist", fake)
    for rank in (1, 2, 3):
        reduced._emit_gathered_metric_lines(lines[rank], rank=rank, world_size=4)
        assert capsys.readouterr().out == ""
    reduced._emit_gathered_metric_lines(lines[0], rank=0, world_size=4)
    assert capsys.readouterr().out == "\n".join(lines) + "\n"
    assert fake.calls == [lines[1], lines[2], lines[3], lines[0]]


def test_reduced_shard_main_consumes_gathered_output_seam():
    source = inspect.getsource(reduced.main)
    assert source.count("_emit_gathered_metric_lines(") == 1
    assert source.index("_emit_gathered_metric_lines(") < source.index(
        "REDUCED_SHARD_OK"
    )
