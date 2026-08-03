from __future__ import annotations

import sys


def test_parse_args_exposes_task15_gate_contract(monkeypatch) -> None:
    from tests.distributed.hierarchical_stage_smoke import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hierarchical_stage_smoke.py",
            "--numel=8388608",
            "--warmup=5",
            "--repeat=10",
            "--output-json=result.json",
        ],
    )

    args = parse_args()

    assert args.numel == 8_388_608
    assert args.warmup == 5
    assert args.repeat == 10
    assert args.output_json.name == "result.json"
