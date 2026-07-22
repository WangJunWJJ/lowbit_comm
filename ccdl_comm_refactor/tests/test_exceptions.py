import ccdl_comm
from ccdl_comm.exceptions import (
    CCDLError,
    CCDLUnavailableError,
    TorchDistributedUnavailableError,
    UnsupportedCollective,
)


def test_public_exception_hierarchy_is_importable_from_package_root() -> None:
    assert ccdl_comm.CCDLError is CCDLError
    assert ccdl_comm.CCDLUnavailableError is CCDLUnavailableError
    assert ccdl_comm.UnsupportedCollective is UnsupportedCollective
    assert ccdl_comm.TorchDistributedUnavailableError is TorchDistributedUnavailableError


def test_specialized_exceptions_share_ccdl_base_type() -> None:
    assert issubclass(CCDLUnavailableError, CCDLError)
    assert issubclass(UnsupportedCollective, CCDLError)
    assert issubclass(TorchDistributedUnavailableError, CCDLError)


def test_unsupported_collective_message_names_requested_collective() -> None:
    error = UnsupportedCollective("reduce_scatter", reason="cuda backend is unavailable")

    assert "reduce_scatter" in str(error)
    assert "cuda backend is unavailable" in str(error)
