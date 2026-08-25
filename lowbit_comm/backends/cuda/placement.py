"""Validated process placement applied before CUDA ProcessGroup startup."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re


_CPU_TOKEN = re.compile(r"[0-9]+(?:-[0-9]+)?")
_NCCL_CHANNEL_ENV = ("NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS")
_MAX_CPU_AFFINITY_VALUES = 65_536


def _validate_cpu_affinity_map(value: object) -> None:
    if type(value) is not tuple:
        raise ValueError("CPU affinity map must be an exact tuple")
    claimed: set[int] = set()
    for rank_cpus in value:
        if type(rank_cpus) is not tuple or not rank_cpus:
            raise ValueError(
                "CPU affinity rank entry must be a non-empty tuple"
            )
        previous = -1
        for cpu in rank_cpus:
            if type(cpu) is not int or cpu < 0:
                raise ValueError(
                    "CPU affinity values must be non-negative integers"
                )
            if cpu <= previous:
                raise ValueError(
                    "CPU affinity rank entries must be unique and sorted"
                )
            if cpu in claimed:
                raise ValueError(
                    "CPU affinity map cannot overlap across ranks"
                )
            claimed.add(cpu)
            previous = cpu


def _validate_nccl_channels(value: object) -> None:
    if value is not None and (
        type(value) is not int or not 1 <= value <= 32
    ):
        raise ValueError("NCCL channel count must be between 1 and 32")


@dataclass(frozen=True, slots=True)
class CudaProcessPlacement:
    """One immutable pre-ProcessGroup CUDA process placement request."""

    cpu_affinity_by_local_rank: tuple[tuple[int, ...], ...] = ()
    nccl_channels: int | None = None

    def __post_init__(self) -> None:
        _validate_cpu_affinity_map(self.cpu_affinity_by_local_rank)
        _validate_nccl_channels(self.nccl_channels)


@dataclass(frozen=True, slots=True)
class AppliedCudaProcessPlacement:
    """Exact placement values committed to the current process."""

    selected_cpus: tuple[int, ...]
    nccl_channels: int | None

    def __post_init__(self) -> None:
        if type(self.selected_cpus) is not tuple or any(
            type(cpu) is not int or cpu < 0 for cpu in self.selected_cpus
        ) or self.selected_cpus != tuple(sorted(set(self.selected_cpus))):
            raise ValueError(
                "Applied CPU affinity must be an exact integer tuple"
            )
        _validate_nccl_channels(self.nccl_channels)


def parse_cuda_process_placement(
    cpu_affinity_map: object,
    nccl_channels: object,
) -> CudaProcessPlacement:
    """Parse exact CLI-compatible values into one immutable placement."""
    if type(cpu_affinity_map) is not str:
        raise ValueError("CPU affinity map must be an exact string")
    if type(nccl_channels) is not int or not 0 <= nccl_channels <= 32:
        raise ValueError("NCCL channel count must be between 0 and 32")
    parsed: list[tuple[int, ...]] = []
    parsed_value_count = 0
    if cpu_affinity_map:
        for entry in cpu_affinity_map.split(";"):
            cpus: set[int] = set()
            if not entry:
                raise ValueError("CPU affinity entry is invalid")
            for token in entry.split(","):
                if _CPU_TOKEN.fullmatch(token) is None:
                    raise ValueError("CPU affinity entry is invalid")
                bounds = token.split("-", 1)
                first = int(bounds[0])
                last = int(bounds[-1])
                if last < first:
                    raise ValueError("CPU affinity entry is invalid")
                parsed_value_count += last - first + 1
                if parsed_value_count > _MAX_CPU_AFFINITY_VALUES:
                    raise ValueError("CPU affinity map contains too many CPUs")
                expanded = set(range(first, last + 1))
                if cpus & expanded:
                    raise ValueError(
                        "CPU affinity entry contains duplicate CPUs"
                    )
                cpus.update(expanded)
            parsed.append(tuple(sorted(cpus)))
    return CudaProcessPlacement(
        tuple(parsed),
        None if nccl_channels == 0 else nccl_channels,
    )


def _fresh_placement(value: object) -> CudaProcessPlacement:
    if type(value) is not CudaProcessPlacement:
        raise ValueError("CUDA process placement must be exact")
    try:
        cpu_affinity = value.cpu_affinity_by_local_rank
        nccl_channels = value.nccl_channels
    except AttributeError as error:
        raise ValueError("CUDA process placement graph is invalid") from error
    _validate_cpu_affinity_map(cpu_affinity)
    _validate_nccl_channels(nccl_channels)
    return CudaProcessPlacement(
        tuple(tuple(cpus) for cpus in cpu_affinity),
        nccl_channels,
    )


def apply_cuda_process_placement(
    config: object,
    *,
    local_rank: object,
    local_world_size: object,
) -> AppliedCudaProcessPlacement:
    """Validate completely, then apply one process placement exactly once."""
    placement = _fresh_placement(config)
    if type(local_world_size) is not int or local_world_size <= 0:
        raise ValueError("CUDA placement local world size is invalid")
    if (
        type(local_rank) is not int
        or local_rank < 0
        or local_rank >= local_world_size
    ):
        raise ValueError("CUDA placement local rank is invalid")

    selected_cpus: tuple[int, ...] = ()
    original_cpus: set[int] | None = None
    get_affinity = getattr(os, "sched_getaffinity", None)
    set_affinity = getattr(os, "sched_setaffinity", None)
    if placement.cpu_affinity_by_local_rank:
        if len(placement.cpu_affinity_by_local_rank) != local_world_size:
            raise ValueError(
                "CPU affinity map must contain one entry per rank"
            )
        if not callable(get_affinity) or not callable(set_affinity):
            raise ValueError("CPU affinity is not supported on this platform")
        selected_cpus = placement.cpu_affinity_by_local_rank[local_rank]
        original_cpus = set(get_affinity(0))
        if not set(selected_cpus) <= original_cpus:
            raise ValueError("CPU affinity map contains an unavailable CPU")

    expected_channels = (
        None
        if placement.nccl_channels is None
        else str(placement.nccl_channels)
    )
    if expected_channels is not None:
        for name in _NCCL_CHANNEL_ENV:
            existing = os.environ.get(name)
            if existing is not None and existing != expected_channels:
                raise ValueError(
                    "NCCL channel environment conflicts with config"
                )

    added_environment: list[str] = []
    try:
        if selected_cpus:
            set_affinity(0, set(selected_cpus))
        if expected_channels is not None:
            for name in _NCCL_CHANNEL_ENV:
                if name not in os.environ:
                    os.environ[name] = expected_channels
                    added_environment.append(name)
    except BaseException:
        for name in reversed(added_environment):
            os.environ.pop(name, None)
        if selected_cpus and original_cpus is not None:
            set_affinity(0, original_cpus)
        raise
    return AppliedCudaProcessPlacement(
        selected_cpus,
        placement.nccl_channels,
    )
