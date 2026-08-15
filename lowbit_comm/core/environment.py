"""Immutable environment identity shared by compilation layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from lowbit_comm.core.errors import CompileError
from lowbit_comm.core.validation import _fresh_validate_exact


Dimensions = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class EnvironmentFingerprint:
    """Immutable environment dimensions supplied by a benchmark."""

    dimensions: Dimensions

    def __post_init__(self) -> None:
        _validate_frozen_dimensions(self.dimensions, "Environment")

    @classmethod
    def from_mapping(
        cls,
        dimensions: Mapping[str, str],
    ) -> EnvironmentFingerprint:
        """Copy and sort environment dimensions without global inspection."""
        return cls(_freeze_dimensions(dimensions, "Environment"))


def _validate_environment_fingerprint_graph(
    environment: object,
) -> EnvironmentFingerprint:
    """Freshly validate an exact environment fingerprint."""
    return _fresh_validate_exact(
        environment,
        EnvironmentFingerprint,
        EnvironmentFingerprint.__post_init__,
        "Environment fingerprint graph is invalid.",
    )


def _freeze_dimensions(
    dimensions: Mapping[str, str],
    owner: str,
) -> Dimensions:
    """Copy exact string dimensions into a deterministic tuple."""
    if not isinstance(dimensions, Mapping):
        raise CompileError(f"{owner} dimensions must be a mapping.")
    entries = tuple(dimensions.items())
    for key, value in entries:
        if type(key) is not str or type(value) is not str:
            raise CompileError(
                f"{owner} dimension names and values must be strings."
            )
    return tuple(sorted(entries))


def _validate_frozen_dimensions(
    dimensions: Dimensions,
    owner: str,
) -> None:
    """Reject direct construction that bypasses immutable mapping copies."""
    if type(dimensions) is not tuple:
        raise CompileError(f"{owner} dimensions must be a tuple.")
    for entry in dimensions:
        if (
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or type(entry[1]) is not str
        ):
            raise CompileError(
                f"{owner} dimensions must contain string pairs."
            )
    if dimensions != tuple(sorted(dimensions)):
        raise CompileError(f"{owner} dimensions must be sorted.")
    if len({key for key, _ in dimensions}) != len(dimensions):
        raise CompileError(f"{owner} dimension names must be unique.")
