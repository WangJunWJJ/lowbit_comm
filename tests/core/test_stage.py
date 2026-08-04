from __future__ import annotations

import pytest

from ccdl_comm import CommunicationStage


@pytest.mark.parametrize("field", ["name", "collective", "strategy", "backend", "output_layout"])
def test_stage_rejects_empty_identity_fields(field: str) -> None:
    values = {
        "name": "intra_node",
        "collective": "all_reduce",
        "strategy": "ring",
        "backend": "cuda",
        "output_layout": "full",
    }
    values[field] = "   "

    with pytest.raises(ValueError, match=field):
        CommunicationStage(**values)


def test_stage_is_immutable() -> None:
    stage = CommunicationStage("intra_node", "all_reduce", "ring")

    with pytest.raises(Exception):
        stage.backend = "ascend"  # type: ignore[misc]
