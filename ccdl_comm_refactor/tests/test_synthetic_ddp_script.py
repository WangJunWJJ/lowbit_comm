from pathlib import Path


def test_synthetic_ddp_compare_exposes_bucket_gate_and_model_size_args() -> None:
    source = (Path(__file__).parent / "distributed" / "synthetic_ddp_compare.py").read_text(encoding="utf-8")

    assert "--min-compress-numel" in source
    assert "--bucket-cap-mb" in source
    assert "--model-dtype" in source
    assert "--width" in source
    assert 'dtype="auto"' in source
    assert "create_ddp_comm_hook" in source
    assert "parameter_count" in source


def test_synthetic_ddp_script_exposes_error_feedback_policy_flags() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "synthetic_ddp_compare.py").read_text(encoding="utf-8")

    assert "--error-feedback" in source
    assert "--error-feedback-policy" in source
    assert "--error-feedback-min-numel" in source
    assert "--error-feedback-warmup-steps" in source
    assert "--error-feedback-period" in source
    assert "error_feedback_policy=args.error_feedback_policy" in source


def test_synthetic_ddp_script_exposes_async_gather_flag() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "synthetic_ddp_compare.py").read_text(encoding="utf-8")

    assert "--async-gather" in source
    assert 'async_gather=(args.async_gather == "true")' in source
    assert '"async_gather": args.async_gather if args.mode == "ccdl" else None' in source


def test_synthetic_ddp_script_exposes_async_error_feedback_flag() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "synthetic_ddp_compare.py").read_text(encoding="utf-8")

    assert "--async-error-feedback" in source
    assert 'async_error_feedback=(args.async_error_feedback == "true")' in source
    assert '"async_error_feedback": args.async_error_feedback if args.mode == "ccdl" else None' in source


def test_synthetic_ddp_script_exposes_fsdp_mode() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "synthetic_ddp_compare.py").read_text(encoding="utf-8")

    assert 'choices=("baseline", "ccdl", "fsdp")' in source
    assert "FullyShardedDataParallel" in source
    assert "fsdp_default" in source
    assert "_ccdl_parameter_count" in source


def test_synthetic_ddp_script_exposes_auto_strategy_metadata() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "synthetic_ddp_compare.py").read_text(encoding="utf-8")

    assert 'choices=("all_gather", "all_reduce", "auto", "hierarchical", "reduce_scatter", "topology")' in source
    assert "_ccdl_strategy_plan" in source
    assert '"selected_strategy": selected_strategy' in source
    assert '"strategy_fallback_reason": strategy_fallback_reason' in source
    assert '"strategy_requires_fallback": strategy_requires_fallback' in source


def test_synthetic_ddp_script_exposes_hierarchical_transport_flags() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "synthetic_ddp_compare.py").read_text(encoding="utf-8")

    assert "--enable-hierarchical-transport" in source
    assert "--hierarchical-local-group-size" in source
    assert "make_torch_hierarchical_all_reduce" in source
    assert "hierarchical_all_reduce=hierarchical_transport" in source


def test_synthetic_ddp_script_exposes_reduce_scatter_transport_flag() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "synthetic_ddp_compare.py").read_text(encoding="utf-8")

    assert "--enable-reduce-scatter-transport" in source
    assert "make_torch_compressed_reduce_scatter_all_gather" in source
    assert "reduce_scatter_all_gather=reduce_scatter_transport" in source
    assert '"enable_reduce_scatter_transport": args.enable_reduce_scatter_transport if args.mode == "ccdl" else None' in source


def test_synthetic_ddp_script_exposes_topology_method_flag() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "synthetic_ddp_compare.py").read_text(encoding="utf-8")

    assert "--topology-method" in source
    assert 'choices=("auto", "tree", "p2p", "ring")' in source
    assert "topology_method=(None if args.topology_method == \"auto\" else args.topology_method)" in source
    assert '"topology_method": args.topology_method if args.mode == "ccdl" else None' in source
