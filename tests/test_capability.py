from ccdl_comm.capability import CapabilityReport, detect


def test_capability_report_is_safe_to_construct_without_torch_or_cuda():
    report = CapabilityReport.unavailable("torch is not installed")

    assert report.available is False
    assert report.cuda is False
    assert report.ddp_hook is False
    assert report.reason == "torch is not installed"
    assert report.warnings == ("torch is not installed",)


def test_capability_report_serializes_to_parascale_friendly_dict():
    report = CapabilityReport(
        available=True,
        cuda=True,
        torch_version="2.4.0",
        cuda_arch="8.9",
        quantize=True,
        compressed_collectives=False,
        ddp_hook=True,
        reason=None,
        warnings=("compressed collectives are experimental",),
    )

    assert report.to_dict() == {
        "available": True,
        "cuda": True,
        "torch_version": "2.4.0",
        "cuda_arch": "8.9",
        "ops": {
            "quantize": True,
            "compressed_collectives": False,
            "ddp_hook": True,
        },
        "reason": None,
        "warnings": ["compressed collectives are experimental"],
    }


def test_detect_reports_unavailable_when_torch_cannot_be_imported():
    def missing_torch():
        raise ModuleNotFoundError("No module named 'torch'")

    report = detect(import_torch=missing_torch)

    assert report.available is False
    assert report.reason == "torch is not installed"


def test_detect_reports_unavailable_when_cuda_is_not_available():
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        __version__ = "2.4.0"
        cuda = FakeCuda()

    report = detect(import_torch=lambda: FakeTorch)

    assert report.available is False
    assert report.cuda is False
    assert report.torch_version == "2.4.0"
    assert report.reason == "CUDA is not available"


def test_detect_reports_available_when_torch_cuda_and_extension_are_available():
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability(index=0):
            return (8, 9)

    class FakeTorch:
        __version__ = "2.4.0"
        cuda = FakeCuda()

    report = detect(import_torch=lambda: FakeTorch, import_extension=lambda: object())

    assert report.available is True
    assert report.cuda is True
    assert report.torch_version == "2.4.0"
    assert report.cuda_arch == "8.9"
    assert report.quantize is True
    assert report.compressed_collectives is True
    assert report.ddp_hook is True
    assert "not implemented yet" not in " ".join(report.warnings)
    assert "all_gather" in report.available_strategies
    assert report.verified_strategies == frozenset()


def test_capability_report_adapts_to_immutable_core_capabilities() -> None:
    report = CapabilityReport(
        available=True,
        cuda=True,
        torch_version="2.5.0",
        cuda_arch="8.6",
        quantize=True,
        compressed_collectives=True,
        ddp_hook=True,
        warnings=("experimental",),
    )

    capabilities = report.to_backend_capabilities()

    assert capabilities.backend == "cuda"
    assert capabilities.available is True
    assert capabilities.collectives == frozenset({"all_gather", "all_reduce", "reduce_scatter"})
    assert capabilities.strategies == frozenset({"all_gather", "compressed", "topology"})
    assert capabilities.dtypes == frozenset({"bf16", "fp16", "fp32"})
    assert capabilities.bits == frozenset({8})
    assert capabilities.features == frozenset({"compressed_collectives", "ddp_hook", "quantize"})
    assert capabilities.details == {"cuda_arch": "8.6", "torch_version": "2.5.0"}
    assert capabilities.warnings == ("experimental",)


def test_unavailable_report_preserves_reason_in_core_capabilities() -> None:
    capabilities = CapabilityReport.unavailable("torch is not installed").to_backend_capabilities()

    assert capabilities.available is False
    assert capabilities.reason == "torch is not installed"
    assert capabilities.collectives == frozenset()
