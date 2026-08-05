import json

from examples.ddp_training import build_parser, main


def test_parser_supports_three_comparable_modes() -> None:
    parser = build_parser()

    for mode in ("native_ddp", "ccdl_sync", "ccdl_async"):
        args = parser.parse_args(["--mode", mode, "--synthetic"])
        assert args.mode == mode


def test_cli_writes_runner_result_as_json(tmp_path, monkeypatch) -> None:
    output = tmp_path / "result.json"
    expected = {"mode": "native_ddp", "timing": {"measured_steps": 1}}
    monkeypatch.setattr(
        "examples.ddp_training.run_training",
        lambda config: expected,
    )

    exit_code = main(
        [
            "--mode",
            "native_ddp",
            "--synthetic",
            "--steps",
            "2",
            "--warmup-steps",
            "1",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_async_cli_does_not_claim_overlap_without_timeline_evidence() -> None:
    from examples.training.overlap import classify_overlap

    assert classify_overlap(future_returned=True, timeline_intersection_ms=0.0) == (
        "not_overlapped"
    )
