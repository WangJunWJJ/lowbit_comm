from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[3]
CONTRACT = ROOT / "docs" / "architecture" / "architecture_contract.json"


def test_product_version_and_package_name() -> None:
    import lowbit_comm

    assert lowbit_comm.__version__ == "0.3.0"
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["package"] == "lowbit_comm"
    assert contract["source_root"] == "src/lowbit_comm"


def test_contract_separates_program_dimensions() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["program_dimensions"] == [
        "operation",
        "output",
        "wire",
        "algorithm",
    ]
    assert contract["quantized_fulltensor_fp_collectives_allowed"] == 0
    assert contract["reduced_shard_final_all_gather_allowed"] is False
