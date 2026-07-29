from pathlib import Path


def test_sharded_reduce_scatter_perf_script_uses_true_shard_transport() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "sharded_reduce_scatter_perf.py").read_text(
        encoding="utf-8"
    )

    assert "compressed_reduce_scatter_shard" in source
    assert "make_torch_compressed_reduce_scatter_shard" in source
    assert "shard_numel" in source
    assert "all_gather" not in source
    assert '"ccdl_shard_ms"' in source
    assert '"torch_reduce_scatter_ms"' in source


def test_sharded_reduce_scatter_perf_script_supports_topology_transport() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "sharded_reduce_scatter_perf.py").read_text(
        encoding="utf-8"
    )

    assert "--transport" in source
    assert 'choices=("all_to_all", "topology")' in source
    assert "--topology-method" in source
    assert "make_native_topology_reduce_scatter_shard" in source
    assert '"transport": args.transport' in source
    assert '"topology_method": args.topology_method if args.transport == "topology" else None' in source
