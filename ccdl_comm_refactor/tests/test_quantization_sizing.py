import pytest

from ccdl_comm.config import CompressionConfig
from ccdl_comm.quantization.sizing import estimate_quantized_size


def test_estimate_quantized_size_matches_8bit_fp16_group_layout():
    estimate = estimate_quantized_size(128, dtype="fp16", config=CompressionConfig())

    assert estimate.numel == 128
    assert estimate.padded_numel == 128
    assert estimate.num_groups == 2
    assert estimate.original_bytes == 256
    assert estimate.quantized_bytes == 132
    assert estimate.compression_ratio == pytest.approx(256 / 132)


def test_estimate_quantized_size_pads_partial_groups():
    estimate = estimate_quantized_size(65, dtype="fp16", config=CompressionConfig())

    assert estimate.padded_numel == 128
    assert estimate.num_groups == 2
    assert estimate.padding_numel == 63
    assert estimate.quantized_bytes == 132


def test_estimate_quantized_size_accounts_for_fp32_scale_bytes():
    estimate = estimate_quantized_size(64, dtype="fp32", config=CompressionConfig())

    assert estimate.original_bytes == 256
    assert estimate.quantized_bytes == 68


def test_estimate_quantized_size_accounts_for_topk_metadata():
    config = CompressionConfig(topk=2)

    estimate = estimate_quantized_size(64, dtype="fp16", config=config)

    assert estimate.quantized_bytes == 72


def test_estimate_quantized_size_supports_experimental_4bit_when_opted_in():
    config = CompressionConfig(bit=4, allow_experimental=True)

    estimate = estimate_quantized_size(64, dtype="fp16", config=config)

    assert estimate.quantized_bytes == 34


def test_estimate_quantized_size_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="dtype"):
        estimate_quantized_size(64, dtype="int8", config=CompressionConfig())
