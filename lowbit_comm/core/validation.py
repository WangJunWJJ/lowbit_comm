"""Shared safe invocation for immutable contract invariants."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from lowbit_comm.core.errors import CompileError


_ContractT = TypeVar("_ContractT")


def _fresh_validate_exact(
    value: object,
    expected_type: type[_ContractT],
    validator: Callable[[_ContractT], None] | None,
    message: str,
) -> _ContractT:
    """Validate one exact contract through a trusted unbound function."""
    if type(value) is not expected_type:
        raise CompileError(message)
    exact_value = cast(_ContractT, value)
    if validator is None:
        return exact_value
    try:
        validator(exact_value)
    except CompileError:
        raise
    except Exception as error:
        raise CompileError(message) from error
    return exact_value
