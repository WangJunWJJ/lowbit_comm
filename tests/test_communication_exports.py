def test_communication_package_exports_topology_reduce_scatter_factory() -> None:
    from ccdl_comm.communication import make_native_topology_reduce_scatter_shard

    assert callable(make_native_topology_reduce_scatter_shard)
