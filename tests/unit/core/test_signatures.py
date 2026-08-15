from enum import Enum

import pytest

from lowbit_comm.core.errors import CompileError
from lowbit_comm.core.signatures import _signature_value


class MixedValue(Enum):
    INTEGER_ONE = 1
    STRING_ONE = "1"
    INTEGER_ONE_ALIAS = 1


def test_enum_signature_uses_type_identity_and_canonical_member_name() -> None:
    enum_type = f"enum:{MixedValue.__module__}.{MixedValue.__qualname__}"

    assert _signature_value(MixedValue.INTEGER_ONE) == (
        enum_type,
        "INTEGER_ONE",
    )
    assert _signature_value(MixedValue.STRING_ONE) == (
        enum_type,
        "STRING_ONE",
    )
    assert _signature_value(MixedValue.INTEGER_ONE_ALIAS) == (
        enum_type,
        "INTEGER_ONE",
    )


@pytest.mark.parametrize("value", [object(), "future-field"])
def test_signature_rejects_unsupported_object_types(value: object) -> None:
    with pytest.raises(CompileError, match="unsupported field type"):
        _signature_value(value)
