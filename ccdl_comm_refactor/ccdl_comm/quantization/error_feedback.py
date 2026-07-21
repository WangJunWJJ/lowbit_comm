from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any


def _safe_detached_clone(value: Any) -> Any:
    detached = value.detach() if hasattr(value, "detach") else value
    return detached.clone() if hasattr(detached, "clone") else detached


@dataclass
class ErrorFeedbackState:
    """Track per-bucket compression residuals for error-feedback training."""

    _residuals: dict[Hashable, Any] = field(default_factory=dict)

    def compensate(self, key: Hashable, tensor: Any) -> Any:
        """Add the stored residual to ``tensor`` when one exists."""

        residual = self._residuals.get(key)
        if residual is None:
            return tensor
        return tensor + residual

    def update(self, key: Hashable, *, original: Any, transmitted: Any) -> None:
        """Store the detached residual ``original - transmitted`` for ``key``."""

        self._residuals[key] = _safe_detached_clone(original - transmitted)

    def get(self, key: Hashable) -> Any | None:
        """Return the stored residual for ``key`` when present."""

        return self._residuals.get(key)

    def clear(self, key: Hashable | None = None) -> None:
        """Clear one residual or all residuals."""

        if key is None:
            self._residuals.clear()
            return
        self._residuals.pop(key, None)
