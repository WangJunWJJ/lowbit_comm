from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.benchmarks.assert_performance_gate import check_metric_maximum


SCRIPT = Path(__file__).resolve().parent / "distributed" / "executor_overhead_perf.py"


def test_executor_overhead_script_exposes_required_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for argument in ("--iterations", "--warmup", "--numel", "--output-json"):
        assert argument in source
    for field in (
        '"direct_executor_us"',
        '"compiled_plan_us"',
        '"overhead_ratio"',
        '"compile_us"',
        '"cache_hit"',
    ):
        assert field in source


def test_steady_state_timer_uses_cuda_events_without_compiling() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    timer = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "measure_cuda_us"
    )
    names = {node.id for node in ast.walk(timer) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(timer) if isinstance(node, ast.Attribute)}

    assert "compile" not in names
    assert "Event" in attributes
    assert "record" in attributes
    assert "elapsed_time" in attributes
    assert "synchronize" in attributes


def test_benchmark_balances_measurement_order_and_requires_cache_hit() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    timer_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "measure_cuda_us"
    ]
    source = SCRIPT.read_text(encoding="utf-8")

    assert len(timer_calls) == 4
    assert '"direct_samples_us"' in source
    assert '"compiled_samples_us"' in source
    assert "if not cache_hit:" in source


def test_benchmark_reports_wall_clock_and_maximum_local_rank_ratio() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"wall_direct_executor_us"' in source
    assert '"wall_compiled_plan_us"' in source
    assert '"wall_overhead_ratio"' in source
    assert "local_overhead_ratio" in source
    assert "maximum_rank_value(local_overhead_ratio" in source


def test_metric_maximum_gate_accepts_boundary_and_rejects_regression() -> None:
    assert check_metric_maximum({"overhead_ratio": 1.01}, metric="overhead_ratio", maximum=1.01) == []
    assert check_metric_maximum(
        {"overhead_ratio": 1.0101},
        metric="overhead_ratio",
        maximum=1.01,
    ) == ["metric overhead_ratio exceeds maximum: 1.010100 > 1.010000"]


def test_metric_maximum_gate_rejects_invalid_measurements() -> None:
    with pytest.raises(ValueError, match="missing metric"):
        check_metric_maximum({}, metric="overhead_ratio", maximum=1.01)
    with pytest.raises(ValueError, match="finite numeric"):
        check_metric_maximum(
            {"overhead_ratio": float("nan")},
            metric="overhead_ratio",
            maximum=1.01,
        )
    with pytest.raises(ValueError, match="finite numeric"):
        check_metric_maximum(
            {"overhead_ratio": 1.0},
            metric="overhead_ratio",
            maximum=float("inf"),
        )


def test_performance_gate_supports_documented_direct_script_invocation(tmp_path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"overhead_ratio": 1.0}), encoding="utf-8")
    gate = Path(__file__).resolve().parent / "benchmarks" / "assert_performance_gate.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(gate),
            "--candidate",
            str(candidate),
            "--metric",
            "overhead_ratio",
            "--max",
            "1.01",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
