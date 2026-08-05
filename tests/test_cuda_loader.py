from ccdl_comm.cuda.loader import load_cuda_extension


def test_load_cuda_extension_returns_unavailable_when_extension_is_missing():
    def missing_extension(name):
        raise ModuleNotFoundError(f"No module named {name!r}")

    status = load_cuda_extension(import_module=missing_extension)

    assert status.available is False
    assert status.module is None
    assert status.reason == "ccdl_cuda_ops is not installed"


def test_load_cuda_extension_returns_available_with_loaded_module():
    extension = type(
        "Extension",
        (),
        {
            "NATIVE_WORK_ABI_VERSION": 1,
            "TORCH_VERSION": "2.4.0",
            "CUDA_RUNTIME_VERSION": "12.1",
        },
    )()

    status = load_cuda_extension(import_module=lambda name: extension)

    assert status.available is True
    assert status.module is extension
    assert status.reason is None
    assert status.abi_version == 1
    assert status.torch_version == "2.4.0"
    assert status.cuda_runtime_version == "12.1"


def test_load_cuda_extension_does_not_raise_for_linker_or_runtime_import_errors():
    def broken_extension(name):
        raise ImportError("undefined symbol: inplace_quantize")

    status = load_cuda_extension(import_module=broken_extension)

    assert status.available is False
    assert status.module is None
    assert "undefined symbol" in status.reason
