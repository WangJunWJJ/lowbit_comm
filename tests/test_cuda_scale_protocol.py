from __future__ import annotations

from pathlib import Path


_QUANTIZATION_SOURCE = (
    Path(__file__).parents[1] / "ccdl_comm" / "csrc" / "quantization"
)


def test_cuda_encoders_store_the_effective_scale_used_for_quantization() -> None:
    """Prevent encoders from flooring scale without serializing the floor."""
    quant_pack = (_QUANTIZATION_SOURCE / "quant_pack_kernel.cu").read_text()
    fused_restore = (
        _QUANTIZATION_SOURCE / "dequant_reduce_kernel.cu"
    ).read_text()
    generated_half = (_QUANTIZATION_SOURCE / "quant_kernel.cuh").read_text()
    generated_fp32 = (
        _QUANTIZATION_SOURCE / "quant_kernel_fp32.cuh"
    ).read_text()

    assert "float2half<scalar_t>(fmaxf(max_abs, 1.0e-6f))" in quant_pack
    assert "const float stored_scale = fmaxf(max_abs, 1.0e-6f);" in quant_pack
    assert (
        "float2half<scalar_t>(fmaxf(maxima[0], 1.0e-6f))"
        in fused_restore
    )
    assert "topk_ret.scale = hfmax(" in generated_half
    assert "topk_ret.scale = fmaxf(topk_ret.scale, 1.0e-6f);" in generated_fp32


def test_cuda_encoders_mark_non_finite_groups_in_serialized_scale() -> None:
    """NaN and infinity must survive compression for AMP overflow checks."""
    quant_pack = (_QUANTIZATION_SOURCE / "quant_pack_kernel.cu").read_text()
    fused_restore = (
        _QUANTIZATION_SOURCE / "dequant_reduce_kernel.cu"
    ).read_text()
    generated_half = (_QUANTIZATION_SOURCE / "quant_kernel.cuh").read_text()
    generated_fp32 = (
        _QUANTIZATION_SOURCE / "quant_kernel_fp32.cuh"
    ).read_text()

    assert "non_finite_quant_scale()" in quant_pack
    assert "non_finite_quant_scale()" in fused_restore
    assert "bool has_non_finite = false;" in generated_half
    assert "bool has_non_finite = false;" in generated_fp32
