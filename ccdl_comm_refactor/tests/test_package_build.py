from pathlib import Path

from ccdl_comm.build.package import create_package_cuda_extension, package_csrc_root


def test_package_csrc_root_resolves_inside_project_package() -> None:
    root = package_csrc_root()

    assert root.name == "csrc"
    assert root.parent.name == "ccdl_comm"


def test_create_package_cuda_extension_uses_package_csrc_root() -> None:
    created = {}

    def fake_factory(**kwargs):
        created.update(kwargs)
        return created

    def fake_generator(commands):
        quantization_dir = Path(commands[1]).parent
        (quantization_dir / "gen_quant_api.cu").write_text("torch::Tensor quantize(", encoding="utf-8")
        (quantization_dir / "gen_dequant_api.cu").write_text("torch::Tensor dequantize(", encoding="utf-8")

    extension = create_package_cuda_extension(
        run_generator=fake_generator,
        extension_factory=fake_factory,
    )

    assert extension is created
    assert created["name"] == "ccdl_cuda_ops"
    assert all("ccdl_comm" in source for source in created["sources"])
