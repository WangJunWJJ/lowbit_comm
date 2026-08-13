from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).parents[3]


def test_primitive_matrix_reports_effective_paths_and_latency_quantiles() -> None:
    source = (ROOT / "tests/contract/benchmarks/primitive_matrix.py").read_text(
        encoding="utf-8"
    )

    for mode in (
        '"native_all_reduce"',
        '"compressed_all_gather"',
        '"compressed_reduce_scatter"',
        '"compressed_rs_ag"',
        '"hierarchical_compressed"',
    ):
        assert mode in source
    for field in (
        '"schema_version"',
        '"fingerprint"',
        '"rounds"',
        '"physical_primitive"',
        '"output"',
        '"peak_memory_bytes"',
    ):
        assert field in source
    assert "GroupedTransportRuntime" in source
    assert 'LOWBIT_COMM_TOPOLOGY' in source
    assert 'LOWBIT_COMM_BENCH_ROUNDS' in source
    assert "round_index % 2" in source
    assert "summarize_samples(samples)" in source


def test_benchmark_evidence_schema_preserves_raw_samples() -> None:
    from lowbit_comm.benchmarking import summarize_samples

    summary = summarize_samples([3.0, 1.0, 2.0])

    assert summary["samples_ms"] == [3.0, 1.0, 2.0]
    assert summary["p50_ms"] == 2.0
    assert summary["p95_ms"] == pytest.approx(2.9)
    assert summary["coefficient_of_variation"] == pytest.approx(0.5)


def test_runtime_fingerprint_supports_stable_host_identity(monkeypatch) -> None:
    from lowbit_comm.benchmarking import runtime_fingerprint

    class Properties:
        name = "A6000"
        uuid = "GPU-stable"

    class Cuda:
        nccl = type("Nccl", (), {"version": staticmethod(lambda: (2, 22, 3))})

        @staticmethod
        def get_device_properties(_rank):
            return Properties()

        @staticmethod
        def get_device_capability(_rank):
            return (8, 6)

    class Distributed:
        @staticmethod
        def is_nccl_available():
            return True

    torch = type(
        "Torch",
        (),
        {
            "cuda": Cuda(),
            "distributed": Distributed(),
            "version": type("Version", (), {"cuda": "12.6"})(),
            "__version__": "2.5",
        },
    )()
    monkeypatch.setenv("LOWBIT_COMM_HOST_ID", "a6000-156")

    result = runtime_fingerprint(torch, local_rank=0)

    assert result["host_id"] == "a6000-156"
    assert result["gpu_uuid"] == "GPU-stable"


def test_hierarchical_oracle_is_not_hardcoded_to_four_ranks() -> None:
    source = (
        ROOT / "tests/contract/distributed/hierarchical_compressed_oracle.py"
    ).read_text(encoding="utf-8")

    assert "world_size != 4" not in source
    assert 'LOWBIT_COMM_TOPOLOGY' in source
    assert "(world_size + 1.0) / 2.0" in source
