from ccdl_comm.ascend.loader import CannExtensionStatus, load_cann_extension


def test_load_cann_extension_reports_missing_module() -> None:
    def missing_module(name: str) -> object:
        raise ModuleNotFoundError(name)

    status = load_cann_extension(import_module=missing_module)

    assert status.available is False
    assert status.module is None
    assert "ccdl_cann_ops" in status.reason


def test_load_cann_extension_reports_import_error() -> None:
    def broken_module(name: str) -> object:
        raise ImportError("bad cann abi")

    status = load_cann_extension(import_module=broken_module)

    assert status == CannExtensionStatus(available=False, module=None, reason="bad cann abi")


def test_load_cann_extension_returns_module_when_available() -> None:
    module = object()

    status = load_cann_extension(import_module=lambda name: module)

    assert status == CannExtensionStatus(available=True, module=module)
