from __future__ import annotations

import pytest

from lowbit_comm.backends.cuda import placement
from lowbit_comm.backends.cuda.placement import (
    AppliedCudaProcessPlacement,
    CudaProcessPlacement,
    apply_cuda_process_placement,
    parse_cuda_process_placement,
)


def test_parse_cuda_process_placement_returns_exact_immutable_config() -> None:
    config = parse_cuda_process_placement("0-2,4;3,5", 4)

    assert config == CudaProcessPlacement(
        cpu_affinity_by_local_rank=((0, 1, 2, 4), (3, 5)),
        nccl_channels=4,
    )


def test_empty_cuda_process_placement_is_a_true_noop(monkeypatch) -> None:
    monkeypatch.delattr(placement.os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(placement.os, "sched_setaffinity", raising=False)
    environment = {"UNCHANGED": "yes"}
    monkeypatch.setattr(placement.os, "environ", environment)

    applied = apply_cuda_process_placement(
        CudaProcessPlacement(),
        local_rank=0,
        local_world_size=1,
    )

    assert applied == AppliedCudaProcessPlacement((), None)
    assert environment == {"UNCHANGED": "yes"}


def test_apply_cuda_process_placement_commits_after_full_validation(
    monkeypatch,
) -> None:
    calls: list[tuple[int, set[int]]] = []
    environment: dict[str, str] = {}
    monkeypatch.setattr(placement.os, "environ", environment)
    monkeypatch.setattr(
        placement.os,
        "sched_getaffinity",
        lambda pid: set(range(8)),
        raising=False,
    )
    monkeypatch.setattr(
        placement.os,
        "sched_setaffinity",
        lambda pid, cpus: calls.append((pid, set(cpus))),
        raising=False,
    )

    applied = apply_cuda_process_placement(
        CudaProcessPlacement(((0, 2), (1, 3)), 4),
        local_rank=1,
        local_world_size=2,
    )

    assert applied == AppliedCudaProcessPlacement((1, 3), 4)
    assert calls == [(0, {1, 3})]
    assert environment["NCCL_MIN_NCHANNELS"] == "4"
    assert environment["NCCL_MAX_NCHANNELS"] == "4"


@pytest.mark.parametrize(
    ("mapping", "channels"),
    [
        ([[0]], None),
        (((True,),), None),
        (((0, 0),), None),
        (((0,), (0,)), None),
        ((), True),
        ((), 0),
        ((), 33),
    ],
)
def test_cuda_process_placement_rejects_noncanonical_values(
    mapping: object,
    channels: object,
) -> None:
    with pytest.raises(ValueError):
        CudaProcessPlacement(mapping, channels)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    ("0-;1", "2-1;3", "0,,1;2", "0,0;1", "0;0", "-1;2"),
)
def test_parse_cuda_process_placement_rejects_invalid_maps(value: str) -> None:
    with pytest.raises(ValueError, match="CPU affinity"):
        parse_cuda_process_placement(value, 2)


def test_apply_cuda_process_placement_rejects_before_any_mutation(
    monkeypatch,
) -> None:
    calls: list[object] = []
    environment = {
        "NCCL_MIN_NCHANNELS": "2",
        "NCCL_MAX_NCHANNELS": "2",
    }
    monkeypatch.setattr(placement.os, "environ", environment)
    monkeypatch.setattr(
        placement.os,
        "sched_getaffinity",
        lambda pid: {0, 1},
        raising=False,
    )
    monkeypatch.setattr(
        placement.os,
        "sched_setaffinity",
        lambda pid, cpus: calls.append((pid, cpus)),
        raising=False,
    )

    with pytest.raises(ValueError, match="NCCL channel environment conflicts"):
        apply_cuda_process_placement(
            CudaProcessPlacement(((0,), (1,)), 4),
            local_rank=0,
            local_world_size=2,
        )

    assert calls == []
    assert environment == {
        "NCCL_MIN_NCHANNELS": "2",
        "NCCL_MAX_NCHANNELS": "2",
    }


def test_apply_cuda_process_placement_rejects_unavailable_cpu(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        placement.os,
        "sched_getaffinity",
        lambda pid: {0},
        raising=False,
    )
    monkeypatch.setattr(
        placement.os,
        "sched_setaffinity",
        lambda pid, cpus: pytest.fail("must not set affinity"),
        raising=False,
    )

    with pytest.raises(ValueError, match="unavailable"):
        apply_cuda_process_placement(
            CudaProcessPlacement(((0,), (1,)), None),
            local_rank=1,
            local_world_size=2,
        )


def test_nonempty_affinity_requires_platform_support(monkeypatch) -> None:
    monkeypatch.delattr(placement.os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(placement.os, "sched_setaffinity", raising=False)

    with pytest.raises(ValueError, match="not supported"):
        apply_cuda_process_placement(
            CudaProcessPlacement(((0,),), None),
            local_rank=0,
            local_world_size=1,
        )


def test_channel_only_configuration_does_not_require_affinity(
    monkeypatch,
) -> None:
    monkeypatch.delattr(placement.os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(placement.os, "sched_setaffinity", raising=False)
    environment: dict[str, str] = {}
    monkeypatch.setattr(placement.os, "environ", environment)

    applied = apply_cuda_process_placement(
        CudaProcessPlacement((), 2),
        local_rank=0,
        local_world_size=1,
    )

    assert applied == AppliedCudaProcessPlacement((), 2)
    assert environment == {
        "NCCL_MIN_NCHANNELS": "2",
        "NCCL_MAX_NCHANNELS": "2",
    }
