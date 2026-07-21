from pathlib import Path

from benchmarks.lm.config import MAIN_VARIANTS, RunConfig, expand_main_matrix


def test_main_matrix_has_five_variants_and_three_seeds(tmp_path: Path):
    matrix = expand_main_matrix(tmp_path)
    assert len(matrix) == 15
    assert {item.variant for item in matrix} == set(MAIN_VARIANTS)
    assert {item.seed for item in matrix} == {17, 29, 43}
    assert len({(item.variant, item.seed) for item in matrix}) == 15


def test_quantized_variant_parameters_are_fixed(tmp_path: Path):
    matrix = {item.variant: item for item in expand_main_matrix(tmp_path) if item.seed == 17}
    assert (matrix["nccl_fp32"].bit, matrix["nccl_fp32"].topk) == (None, None)
    assert (matrix["int8_k0"].bit, matrix["int8_k0"].topk) == (8, 0)
    assert (matrix["int8_k2"].bit, matrix["int8_k2"].topk) == (8, 2)
    assert (matrix["int4_k0"].bit, matrix["int4_k0"].topk) == (4, 0)
    assert (matrix["int4_k2"].bit, matrix["int4_k2"].topk) == (4, 2)
    assert all(item.group_size == 64 for item in matrix.values())


def test_run_config_rejects_unknown_variant(tmp_path: Path):
    try:
        RunConfig("bad", 17, tmp_path)
    except ValueError as exc:
        assert "unknown variant" in str(exc)
    else:
        raise AssertionError("unknown variant was accepted")
