"""Real CUDA FullTensor plan factory and distributed contract tests."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cuda_extension():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the FullTensor plan contract")
    return importlib.import_module("lowbit_comm._C")


def test_extension_publishes_fulltensor_factory(cuda_extension) -> None:
    assert callable(cuda_extension.create_fulltensor_plan)


def test_cuda_build_includes_fulltensor_plan_source() -> None:
    setup_source = (ROOT / "setup_cuda.py").read_text(encoding="utf-8")

    assert 'CSRC_DIR / "executor" / "fulltensor_plan.cpp"' in setup_source


def _native_config() -> dict[str, object]:
    return {
        "accumulation_dtype": "fp32",
        "collective": "native",
        "compression": "none",
        "dtype": "fp16",
        "gathered_payload_bytes": 0,
        "group_count": 0,
        "group_size": None,
        "logical_numel": 32,
        "numel": 32,
        "output_bytes": 64,
        "padded_numel": 32,
        "payload_bytes_per_rank": 0,
        "rank": 0,
        "reduction": "sum",
        "workspace_bytes": 0,
        "world_size": 2,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("padded_numel", 33),
        ("group_count", 1),
        ("payload_bytes_per_rank", 1),
        ("gathered_payload_bytes", 1),
        ("output_bytes", 63),
    ],
)
def test_factory_rejects_inconsistent_native_layout_before_group_cast(
    cuda_extension,
    field: str,
    value: int,
) -> None:
    config = _native_config()
    config[field] = value

    with pytest.raises(ValueError, match="native layout"):
        cuda_extension.create_fulltensor_plan(config, object())


@pytest.mark.parametrize(
    ("field", "value"),
    [("world_size", 3), ("rank", 2)],
)
def test_factory_rejects_invalid_rank_domain_before_group_cast(
    cuda_extension,
    field: str,
    value: int,
) -> None:
    config = _native_config()
    config[field] = value

    with pytest.raises(ValueError, match="rank|world size"):
        cuda_extension.create_fulltensor_plan(config, object())
