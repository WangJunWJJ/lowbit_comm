from pathlib import Path


def test_cuda_codec_perf_compares_allocating_and_inplace_codec_paths() -> None:
    source = (Path(__file__).parent / "distributed" / "cuda_codec_perf.py").read_text(encoding="utf-8")

    assert "allocate_quantized_buffer" in source
    assert "allocate_dequantized_buffer" in source
    assert "quant_speedup" in source
    assert "dequant_speedup" in source
    assert "relative_l2" in source
