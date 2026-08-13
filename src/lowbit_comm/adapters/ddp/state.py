"""Transactional gradient error-feedback state for DDP buckets."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True, slots=True)
class CompressionSchema:
    """Immutable identity of the lossy reconstruction used by Gradient EF."""

    bit: int
    group_size: int
    quant_type: str
    compact: bool
    algorithm: str

    def __post_init__(self) -> None:
        if self.bit not in {4, 8}:
            raise ValueError("feedback bit must be 4 or 8")
        if self.group_size not in {16, 32, 64}:
            raise ValueError("feedback group_size must be 16, 32, or 64")
        if not self.quant_type or not self.algorithm:
            raise ValueError("feedback quant_type and algorithm must be non-empty")
        if not isinstance(self.compact, bool):
            raise TypeError("feedback compact must be a bool")


@dataclass(frozen=True, slots=True)
class FeedbackKey:
    bucket: Hashable
    layout_generation: int
    shape: tuple[int, ...]
    dtype: str
    world_size: int
    compression_schema: CompressionSchema


class FeedbackTransaction:
    def __init__(
        self,
        state: GradientFeedbackState,
        key: FeedbackKey,
        prepared: Any,
        *,
        finite: bool,
        state_generation: int,
    ) -> None:
        self._state = state
        self.key = key
        self.prepared = prepared
        self._finite = finite
        self._state_generation = state_generation
        self._closed = False

    def commit(self, local_restored: Any) -> None:
        if self._closed:
            raise RuntimeError("feedback transaction is already closed")
        if not self._finite or not _is_finite(local_restored):
            self._closed = True
            raise RuntimeError("non-finite gradient cannot commit error feedback")
        residual = _detached_clone(self.prepared - local_restored)
        self._state._commit(self.key, residual, self._state_generation)
        self._closed = True

    def abort(self) -> None:
        self._closed = True


class GradientFeedbackState:
    """Bucket-generation-aware Gradient EF owned outside communication Core."""

    def __init__(
        self,
        *,
        layout_generation: int = 0,
        on_commit: Callable[[], None] | None = None,
    ) -> None:
        if layout_generation < 0:
            raise ValueError("layout_generation must be non-negative")
        self._layout_generation = layout_generation
        self._state_generation = 0
        self._residuals: dict[FeedbackKey, Any] = {}
        self._last_invalidation_reason: str | None = None
        self._on_commit = on_commit
        self._lock = RLock()

    def prepare(
        self,
        bucket: Hashable,
        value: Any,
        *,
        world_size: int,
        compression_schema: CompressionSchema,
    ) -> FeedbackTransaction:
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError("world_size must be an integer")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not isinstance(compression_schema, CompressionSchema):
            raise TypeError("compression_schema must be a CompressionSchema")
        key = FeedbackKey(
            bucket=bucket,
            layout_generation=self._layout_generation,
            shape=tuple(getattr(value, "shape", ())),
            dtype=str(getattr(value, "dtype", "unknown")),
            world_size=world_size,
            compression_schema=compression_schema,
        )
        with self._lock:
            residual = self._residuals.get(key)
            state_generation = self._state_generation
        prepared = value if residual is None else value + residual
        return FeedbackTransaction(
            self,
            key,
            prepared,
            finite=_is_finite(value),
            state_generation=state_generation,
        )

    @property
    def last_invalidation_reason(self) -> str | None:
        with self._lock:
            return self._last_invalidation_reason

    def residual(self, bucket: Hashable) -> Any | None:
        with self._lock:
            matches = [
                value
                for key, value in self._residuals.items()
                if key.bucket == bucket
                and key.layout_generation == self._layout_generation
            ]
        return matches[0] if len(matches) == 1 else None

    def rebuild(self, *, layout_generation: int) -> None:
        if layout_generation <= self._layout_generation:
            raise ValueError("layout_generation must increase on rebuild")
        with self._lock:
            self._layout_generation = layout_generation
            self._state_generation += 1
            self._residuals.clear()
            self._last_invalidation_reason = "layout_rebuild"

    def invalidate(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("feedback invalidation reason must be non-empty")
        with self._lock:
            self._state_generation += 1
            self._residuals.clear()
            self._last_invalidation_reason = reason

    def _commit(
        self,
        key: FeedbackKey,
        residual: Any,
        state_generation: int,
    ) -> None:
        with self._lock:
            if state_generation != self._state_generation:
                raise RuntimeError("cannot commit feedback from obsolete feedback state")
            if key.layout_generation != self._layout_generation:
                raise RuntimeError("cannot commit feedback from an obsolete bucket layout")
            self._residuals[key] = residual
        if self._on_commit is not None:
            self._on_commit()


def _detached_clone(value: Any) -> Any:
    detached = value.detach() if hasattr(value, "detach") else value
    return detached.clone() if hasattr(detached, "clone") else detached


def _is_finite(value: Any) -> bool:
    # A CUDA ``isfinite().all().item()`` would synchronize the host on every
    # DDP bucket and destroy compute/communication overlap. Mixed-precision
    # training owns device-side overflow detection; this transactional guard
    # remains strict for CPU values and test doubles.
    if bool(getattr(value, "is_cuda", False)):
        return True
    explicit = getattr(value, "finite", None)
    if explicit is not None:
        return bool(explicit)
    isfinite = getattr(value, "isfinite", None)
    if callable(isfinite):
        result = isfinite()
        all_values = getattr(result, "all", None)
        result = all_values() if callable(all_values) else result
        item = getattr(result, "item", None)
        return bool(item()) if callable(item) else bool(result)
    return True
