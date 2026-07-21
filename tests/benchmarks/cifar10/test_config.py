from pathlib import Path

from benchmarks.cifar10.config import expand_main_matrix


def test_main_matrix_has_fifteen_unique_runs(tmp_path: Path):
    runs = expand_main_matrix(tmp_path)
    assert len(runs) == 15
    assert len({run.run_id for run in runs}) == 15
    assert {run.seed for run in runs} == {1337, 2027, 4099}
    assert {run.variant for run in runs} == {
        "nccl_fp32",
        "ccdl_int8_k0",
        "ccdl_int8_k2",
        "ccdl_int4_k0",
        "ccdl_int4_k2",
    }


def test_quantized_variants_are_group64_and_deterministic(tmp_path: Path):
    for run in expand_main_matrix(tmp_path):
        if run.variant.startswith("ccdl"):
            assert run.group_size == 64
            assert run.stochastic is False
