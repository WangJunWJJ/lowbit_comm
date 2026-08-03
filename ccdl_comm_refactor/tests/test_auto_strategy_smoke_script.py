from __future__ import annotations

import sys

import pytest


def test_parse_args_exposes_task14_gate_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.distributed.auto_strategy_smoke import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "auto_strategy_smoke.py",
            "--numel=8388608",
            "--output-layout=shard",
            "--expect-strategy=compressed",
            "--warmup=10",
            "--repeat=30",
            "--output-json=result.json",
        ],
    )

    args = parse_args()

    assert args.numel == 8_388_608
    assert args.output_layout == "shard"
    assert args.expect_strategy == "compressed"
    assert args.warmup == 10
    assert args.repeat == 30
    assert str(args.output_json) == "result.json"
