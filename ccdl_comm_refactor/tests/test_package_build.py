import subprocess

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

    extension = create_package_cuda_extension(
        run_generator=lambda commands: subprocess.run(commands, check=True),
        extension_factory=fake_factory,
    )

    assert extension is created
    assert created["name"] == "ccdl_cuda_ops"
    assert all("ccdl_comm" in source for source in created["sources"])
