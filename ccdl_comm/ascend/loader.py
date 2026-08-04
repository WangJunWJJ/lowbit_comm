from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module as _import_module
from types import ModuleType
from typing import Callable


@dataclass(frozen=True)
class CannExtensionStatus:
    """Result of attempting to import the optional CANN extension."""

    available: bool
    module: object | None
    reason: str | None = None


def load_cann_extension(
    *,
    module_name: str = "ccdl_cann_ops",
    import_module: Callable[[str], ModuleType | object] = _import_module,
) -> CannExtensionStatus:
    """Load the optional CANN extension without leaking import failures."""

    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name in str(exc):
            return CannExtensionStatus(
                available=False,
                module=None,
                reason=f"{module_name} is not installed",
            )
        return CannExtensionStatus(available=False, module=None, reason=str(exc))
    except ImportError as exc:
        return CannExtensionStatus(available=False, module=None, reason=str(exc))

    return CannExtensionStatus(available=True, module=module)
