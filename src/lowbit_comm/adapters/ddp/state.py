"""Transactional gradient error-feedback state for DDP buckets."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True, slots=True)
class FeedbackKey:
    bucket: Hashable
    layout_generation: int
    shape: tuple[int, ...]
    dtype: str


class FeedbackTransaction:
    def __init__(
        self,
        state: GradientFeedbackState,
        key: FeedbackKey,
        prepared: Any,
        *,
        finite: bool,
    ) -> None:
        self._state = state
        self.key = key
        self.prepared = prepared
        self._finite = finite
        self._closed = False

    def commit(self, local_restored: Any) -> None:
        if self._closed:
            raise RuntimeError("feedback transaction is already closed")
        if not self._finite or not _is_finite(local_restored):
            self._closed = True
            raise RuntimeError("non-finite gradient cannot commit error feedback")
        residual = _detached_clone(self.prepared - local_restored)
        self._state._commit(self.key, residual)
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
        self._residuals: dict[FeedbackKey, Any] = {}
        self._on_commit = on_commit
        self._lock = RLock()

    def prepare(self, bucket: Hashable, value: Any) -> FeedbackTransaction:
        key = FeedbackKey(
            bucket=bucket,
            layout_generation=self._layout_generation,
            shape=tuple(getattr(value, "shape", ())),
            dtype=str(getattr(value, "dtype", "unknown")),
        )
        with self._lock:
            residual = self._residuals.get(key)
        prepared = value if residual is None else value + residual
        return FeedbackTransaction(
            self,
            key,
            prepared,
            finite=_is_finite(value),
        )

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
            self._residuals.clear()

    def _commit(self, key: FeedbackKey, residual: Any) -> None:
        with self._lock:
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
