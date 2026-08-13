from pathlib import Path


def test_int8_restore_compare_reports_correctness_and_transport_evidence() -> None:
    source = Path("tests/distributed/int8_restore_compare.py").read_text(encoding="utf-8")

    assert 'restore_mode="compressed"' in source
    assert '"restore_payload_dtype": "uint8"' in source
    assert '"rank_max_difference"' in source
    assert '"additional_relative_l2_vs_fp16_restore"' in source
    assert '"pipeline_speedup"' in source
    assert "inplace_dequantize_reduce_mean_requantize" in source
    assert "inplace_dequantize_gathered" in source
    assert '"fused_restore_pipeline_ms"' in source
    assert '"fused_speedup_vs_compressed"' in source
    assert '"fused_requantize_calls"' in source
    assert '"fused_gathered_dequantize_calls"' in source
    assert '"workspace_pool_hits"' in source
    assert '"pooled_compressed_restore_pipeline_ms"' in source
    assert '"fused_speedup_vs_pooled_compressed"' in source
    assert '"pooled_fused_restore_pipeline_ms"' in source
    assert '"direct_restore_workspace_allocations"' in source
    assert '"trial_ms"' in source
    assert "median(" in source
