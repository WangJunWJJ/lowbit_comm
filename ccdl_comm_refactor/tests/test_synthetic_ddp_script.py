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
