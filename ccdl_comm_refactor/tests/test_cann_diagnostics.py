from ccdl_comm.ascend.diagnostics import CannCapabilityReport, detect_cann
from ccdl_comm.ascend.loader import CannExtensionStatus


def test_cann_diagnostics_are_exported_from_ascend_package() -> None:
    from ccdl_comm.ascend import CannCapabilityReport as ExportedReport
    from ccdl_comm.ascend import detect_cann as exported_detect_cann

    assert ExportedReport is CannCapabilityReport
    assert exported_detect_cann is detect_cann


def test_cann_capability_report_serializes_to_scheduler_dict(monkeypatch) -> None:
    monkeypatch.setenv("CCDL_COMM_EXPERIMENTAL_ACLNN", "1")
    report = CannCapabilityReport(
        available=True,
        npu=True,
        torch_version="2.7.1",
        torch_npu_version="2.7.1.post4",
        extension_available=True,
        quantize=True,
        dequantize=True,
        compressed_collectives=True,
        ddp_hook=True,
        quantization_path="experimental_aclnn",
        reason=None,
        warnings=("experimental ACLNN path is enabled",),
    )

    assert report.to_dict() == {
        "available": True,
        "backend": "cann",
        "npu": True,
        "torch_version": "2.7.1",
        "torch_npu_version": "2.7.1.post4",
        "extension_available": True,
        "quantization_path": "experimental_aclnn",
        "ops": {
            "quantize": True,
            "dequantize": True,
            "compressed_collectives": True,
            "ddp_hook": True,
        },
        "reason": None,
        "warnings": ["experimental ACLNN path is enabled"],
    }


def test_detect_cann_reports_missing_torch() -> None:
    def missing_torch():
        raise ModuleNotFoundError("No module named 'torch'")

    report = detect_cann(import_torch=missing_torch)

    assert report.available is False
    assert report.npu is False
    assert report.reason == "torch is not installed"


def test_detect_cann_reports_missing_torch_npu() -> None:
    class FakeNpu:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        __version__ = "2.7.1"
        npu = FakeNpu()

    def missing_torch_npu():
        raise ModuleNotFoundError("No module named 'torch_npu'")

    report = detect_cann(import_torch=lambda: FakeTorch, import_torch_npu=missing_torch_npu)

    assert report.available is False
    assert report.npu is True
    assert report.torch_version == "2.7.1"
    assert report.reason == "torch_npu is not installed"


def test_detect_cann_reports_unavailable_when_npu_is_not_available() -> None:
    class FakeNpu:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        __version__ = "2.7.1"
        npu = FakeNpu()

    report = detect_cann(import_torch=lambda: FakeTorch, import_torch_npu=lambda: object())

    assert report.available is False
    assert report.npu is False
    assert report.reason == "Ascend NPU is not available"


def test_detect_cann_reports_extension_import_failure() -> None:
    class FakeNpu:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        __version__ = "2.7.1"
        npu = FakeNpu()

    class FakeTorchNpu:
        __version__ = "2.7.1.post4"

    report = detect_cann(
        import_torch=lambda: FakeTorch,
        import_torch_npu=lambda: FakeTorchNpu,
        load_extension=lambda: CannExtensionStatus(
            available=False,
            module=None,
            reason="bad cann abi",
        ),
    )

    assert report.available is False
    assert report.npu is True
    assert report.extension_available is False
    assert report.reason == "bad cann abi"


def test_detect_cann_reports_safe_default_path(monkeypatch) -> None:
    monkeypatch.delenv("CCDL_COMM_EXPERIMENTAL_ACLNN", raising=False)

    class FakeNpu:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        __version__ = "2.7.1"
        npu = FakeNpu()

    class FakeTorchNpu:
        __version__ = "2.7.1.post4"

    report = detect_cann(
        import_torch=lambda: FakeTorch,
        import_torch_npu=lambda: FakeTorchNpu,
        load_extension=lambda: CannExtensionStatus(available=True, module=object()),
    )

    assert report.available is True
    assert report.extension_available is True
    assert report.quantization_path == "aten_cann"
    assert report.compressed_collectives is True
    assert report.ddp_hook is True
    assert report.warnings == ()


def test_detect_cann_reports_experimental_aclnn_path(monkeypatch) -> None:
    monkeypatch.setenv("CCDL_COMM_EXPERIMENTAL_ACLNN", "1")

    class FakeNpu:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        __version__ = "2.7.1"
        npu = FakeNpu()

    class FakeTorchNpu:
        __version__ = "2.7.1.post4"

    report = detect_cann(
        import_torch=lambda: FakeTorch,
        import_torch_npu=lambda: FakeTorchNpu,
        load_extension=lambda: CannExtensionStatus(available=True, module=object()),
    )

    assert report.available is True
    assert report.quantization_path == "experimental_aclnn"
    assert report.warnings == ("experimental ACLNN path is enabled",)
