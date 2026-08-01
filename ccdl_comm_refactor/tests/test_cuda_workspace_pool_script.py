import pytest

from tests.distributed.cuda_workspace_pool_smoke import expected_workspace_count


@pytest.mark.parametrize(("world_size", "expected"), [(1, 2), (2, 4), (4, 8), (8, 16)])
def test_workspace_smoke_counts_only_internal_send_and_recv_buffers(world_size, expected) -> None:
    assert expected_workspace_count(world_size) == expected


def test_workspace_smoke_rejects_invalid_world_size() -> None:
    with pytest.raises(ValueError, match="world_size"):
        expected_workspace_count(0)
