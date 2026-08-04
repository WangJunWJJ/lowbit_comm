from pathlib import Path


def test_sharded_reduce_scatter_perf_script_compares_true_shard_with_full_restore() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "sharded_reduce_scatter_perf.py").read_text(
        encoding="utf-8"
    )

    assert "compressed_reduce_scatter_shard" in source
    assert "compile_cuda_shortcut" in source
    assert "compiled_plan.run(source).wait()" in source
    assert 'metadata["workspace_cache"] is True' in source
    assert "shard_numel" in source
    assert '"ccdl_shard_ms"' in source
    assert '"compressed_full_restore_ms"' in source
    assert '"speedup_over_full_restore"' in source
    assert "ReducedShard performance gate failed" in source
    assert '"measurement_order": "full-shard-shard-full"' in source
    assert "validate_result" in source
    assert '"results"' in source
    assert '"peak_memory_bytes"' in source
    assert '"effective_gbps"' in source
    assert '"non_finite"' in source
    assert "resolve_benchmark_identity" in source


def test_sharded_reduce_scatter_perf_script_supports_topology_transport() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "sharded_reduce_scatter_perf.py").read_text(
        encoding="utf-8"
    )

    assert "--transport" in source
    assert 'choices=("compressed", "topology")' in source
    assert "--topology-method" in source
    assert "make_native_topology_reduce_scatter_shard" in source
    assert '"transport": args.transport' in source
    assert '"topology_method": args.topology_method if args.transport == "topology" else None' in source
