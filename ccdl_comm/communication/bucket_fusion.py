from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_MAX_FUSED_BUCKET_NUMEL = 4_000_000
DEFAULT_MAX_FUSED_BUCKET_COUNT = 8


@dataclass(frozen=True)
class BucketDescriptor:
    """Scheduler-facing description of one communication bucket."""

    index: int
    numel: int
    dtype: str = "auto"
    shape: tuple[int, ...] = ()
    requires_full_bucket: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BucketFusionGroup:
    """A consecutive bucket group that can be scheduled together."""

    bucket_indices: tuple[int, ...]
    total_numel: int
    dtype: str
    fused: bool
    reason: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "bucket_indices": self.bucket_indices,
            "total_numel": self.total_numel,
            "dtype": self.dtype,
            "fused": self.fused,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BucketFusionPlan:
    """Explainable bucket-level fusion plan.

    This is intentionally a scheduling contract, not a collective implementation:
    transports can use the fused groups when they can preserve their consumer
    semantics, and safely ignore the plan otherwise.
    """

    enabled: bool
    groups: tuple[BucketFusionGroup, ...]

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def fused_group_count(self) -> int:
        return sum(1 for group in self.groups if group.fused)

    @property
    def fused_bucket_count(self) -> int:
        return sum(len(group.bucket_indices) for group in self.groups if group.fused)

    @property
    def skipped_bucket_count(self) -> int:
        return sum(len(group.bucket_indices) for group in self.groups if not group.fused)

    @property
    def total_fused_numel(self) -> int:
        return sum(group.total_numel for group in self.groups if group.fused)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "group_count": self.group_count,
            "fused_group_count": self.fused_group_count,
            "fused_bucket_count": self.fused_bucket_count,
            "skipped_bucket_count": self.skipped_bucket_count,
            "total_fused_numel": self.total_fused_numel,
            "groups": [group.to_metadata() for group in self.groups],
        }


def plan_bucket_fusion(
    buckets: list[BucketDescriptor] | tuple[BucketDescriptor, ...],
    *,
    enabled: bool = True,
    max_fused_numel: int = DEFAULT_MAX_FUSED_BUCKET_NUMEL,
    max_bucket_count: int = DEFAULT_MAX_FUSED_BUCKET_COUNT,
    require_same_dtype: bool = True,
) -> BucketFusionPlan:
    """Group adjacent compatible buckets for a future fused transport path."""

    if max_fused_numel <= 0:
        raise ValueError("max_fused_numel must be positive")
    if max_bucket_count <= 0:
        raise ValueError("max_bucket_count must be positive")

    groups: list[BucketFusionGroup] = []
    pending: list[BucketDescriptor] = []

    def flush_pending() -> None:
        if not pending:
            return
        groups.append(_make_group(tuple(pending), enabled=enabled))
        pending.clear()

    for bucket in buckets:
        _validate_bucket(bucket)
        if not enabled:
            groups.append(_single(bucket, "fusion disabled"))
            continue
        if bucket.requires_full_bucket:
            flush_pending()
            groups.append(_single(bucket, "requires full-bucket consumer"))
            continue
        if not pending:
            pending.append(bucket)
            continue
        if not _is_compatible(
            pending,
            bucket,
            max_fused_numel=max_fused_numel,
            max_bucket_count=max_bucket_count,
            require_same_dtype=require_same_dtype,
        ):
            flush_pending()
        pending.append(bucket)

    flush_pending()
    return BucketFusionPlan(enabled=enabled, groups=tuple(groups))


def _validate_bucket(bucket: BucketDescriptor) -> None:
    if bucket.numel <= 0:
        raise ValueError("bucket numel must be positive")


def _is_compatible(
    pending: list[BucketDescriptor],
    bucket: BucketDescriptor,
    *,
    max_fused_numel: int,
    max_bucket_count: int,
    require_same_dtype: bool,
) -> bool:
    if len(pending) >= max_bucket_count:
        return False
    if sum(item.numel for item in pending) + bucket.numel > max_fused_numel:
        return False
    if require_same_dtype and bucket.dtype != pending[-1].dtype:
        return False
    return True


def _make_group(buckets: tuple[BucketDescriptor, ...], *, enabled: bool) -> BucketFusionGroup:
    fused = enabled and len(buckets) > 1
    return BucketFusionGroup(
        bucket_indices=tuple(bucket.index for bucket in buckets),
        total_numel=sum(bucket.numel for bucket in buckets),
        dtype=buckets[0].dtype,
        fused=fused,
        reason="compatible adjacent buckets" if fused else "single compatible bucket",
    )


def _single(bucket: BucketDescriptor, reason: str) -> BucketFusionGroup:
    return BucketFusionGroup(
        bucket_indices=(bucket.index,),
        total_numel=bucket.numel,
        dtype=bucket.dtype,
        fused=False,
        reason=reason,
    )
