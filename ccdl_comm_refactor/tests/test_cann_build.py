from ccdl_comm.build.cann import collect_cann_sources, create_cann_extension


def test_collect_cann_sources_is_deterministic(tmp_path) -> None:
    root = tmp_path / "csrc_ascend"
    root.mkdir()
    (root / "pybind.cpp").write_text("", encoding="utf-8")
    (root / "b.cpp").write_text("", encoding="utf-8")
    (root / "a.cpp").write_text("", encoding="utf-8")

    assert [path.name for path in collect_cann_sources(root)] == ["pybind.cpp", "a.cpp", "b.cpp"]


def test_create_cann_extension_uses_optional_factory(tmp_path) -> None:
    root = tmp_path / "csrc_ascend"
    root.mkdir()
    (root / "pybind.cpp").write_text("", encoding="utf-8")
    (root / "kernel.cpp").write_text("", encoding="utf-8")
    created = {}

    extension = create_cann_extension(
        root,
        extension_factory=lambda **kwargs: created.update(kwargs) or created,
        torch_npu_include_root=lambda: tmp_path / "torch_npu" / "include",
        torch_npu_library_root=lambda: tmp_path / "torch_npu" / "lib",
        cann_include_root=lambda: tmp_path / "Ascend" / "include",
    )

    assert extension is created
    assert created["name"] == "ccdl_cann_ops"
    assert created["sources"] == [str(root / "pybind.cpp"), str(root / "kernel.cpp")]


def test_create_cann_extension_defaults_to_safe_cann_build(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CCDL_COMM_EXPERIMENTAL_ACLNN", raising=False)
    root = tmp_path / "csrc_ascend"
    root.mkdir()
    (root / "pybind.cpp").write_text("", encoding="utf-8")
    created = {}

    create_cann_extension(
        root,
        extension_factory=lambda **kwargs: created.update(kwargs) or created,
        torch_npu_include_root=lambda: tmp_path / "torch_npu" / "include",
        torch_npu_library_root=lambda: tmp_path / "torch_npu" / "lib",
        cann_include_root=lambda: tmp_path / "Ascend" / "include",
    )

    include_dirs = [str(path) for path in created["include_dirs"]]
    assert str(tmp_path / "torch_npu" / "include" / "third_party" / "op-plugin") not in include_dirs
    assert str(tmp_path / "Ascend" / "include") in include_dirs
    assert created["libraries"] == []


def test_create_cann_extension_can_enable_experimental_aclnn(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CCDL_COMM_EXPERIMENTAL_ACLNN", "1")
    root = tmp_path / "csrc_ascend"
    root.mkdir()
    (root / "pybind.cpp").write_text("", encoding="utf-8")
    created = {}
    torch_npu_lib = tmp_path / "torch_npu" / "lib"

    create_cann_extension(
        root,
        extension_factory=lambda **kwargs: created.update(kwargs) or created,
        torch_npu_include_root=lambda: tmp_path / "torch_npu" / "include",
        torch_npu_library_root=lambda: torch_npu_lib,
        cann_include_root=lambda: tmp_path / "Ascend" / "include",
    )

    include_dirs = [str(path) for path in created["include_dirs"]]
    assert str(tmp_path / "torch_npu" / "include" / "third_party" / "op-plugin") in include_dirs
    assert "torch_npu" in created["libraries"]
    assert str(torch_npu_lib) in created["library_dirs"]
    assert str(torch_npu_lib) in created["runtime_library_dirs"]
    assert "-DCCDL_COMM_EXPERIMENTAL_ACLNN" in created["extra_compile_args"]
