"""Reusable assertions for backend collective conformance tests."""

from __future__ import annotations

from collections.abc import Iterable


NATIVE_COLLECTIVES = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast",
    "reduce",
    "gather",
    "scatter",
    "barrier",
)


def assert_complete_native_protocol(collectives: Iterable[str]) -> None:
    """Assert that a backend exposes the complete ordered native protocol."""

    assert tuple(collectives) == NATIVE_COLLECTIVES
