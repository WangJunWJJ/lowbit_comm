from tests.distributed.async_completion_perf import build_parser, make_summary


def test_async_completion_perf_parser_defaults() -> None:
    args = build_parser().parse_args(["--output-json", "result.json"])

    assert args.mode == "all_gather_reduce"
    assert args.numel == 4_194_304
    assert args.bit == 8
    assert args.group_size == 64
    assert args.warmup == 10
    assert args.repeat == 30
    assert args.compute_iters == 8


def test_async_completion_summary_contains_comparable_metrics() -> None:
    summary = make_summary(
        mode="topology",
        world_size=4,
        sync_ms=4.0,
        async_wait_ms=3.8,
        async_overlap_ms=5.0,
        compute_ms=2.0,
        launch_us=40.0,
        relative_l2=0.006,
        max_abs_error=0.02,
    )

    assert summary["mode"] == "topology"
    assert summary["world_size"] == 4
    assert summary["async_speedup_over_sync"] == 4.0 / 3.8
    assert summary["overlap_efficiency"] == (4.0 + 2.0 - 5.0) / 2.0
    assert summary["relative_l2"] == 0.006
    assert summary["max_abs_error"] == 0.02
