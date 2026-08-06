from pathlib import Path


def test_fused_restore_kernel_perf_isolates_both_operator_chains() -> None:
    source = Path("tests/distributed/fused_restore_kernel_perf.py").read_text(encoding="utf-8")

    assert '"legacy_requantize_chain_ms"' in source
    assert '"fused_requantize_ms"' in source
    assert '"requantize_speedup"' in source
    assert '"legacy_gathered_dequantize_ms"' in source
    assert '"fused_gathered_dequantize_ms"' in source
    assert '"gathered_dequantize_speedup"' in source
    assert "torch.cuda.Event" in source
