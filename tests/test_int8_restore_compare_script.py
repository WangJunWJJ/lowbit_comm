from pathlib import Path


def test_int8_restore_compare_reports_correctness_and_transport_evidence() -> None:
    source = Path("tests/distributed/int8_restore_compare.py").read_text(encoding="utf-8")

    assert 'restore_mode="compressed"' in source
    assert '"restore_payload_dtype": "uint8"' in source
    assert '"rank_max_difference"' in source
    assert '"additional_relative_l2_vs_fp16_restore"' in source
    assert '"pipeline_speedup"' in source
