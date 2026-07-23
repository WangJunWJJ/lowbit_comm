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

    extension = create_cann_extension(root, extension_factory=lambda **kwargs: created.update(kwargs) or created)

    assert extension is created
    assert created["name"] == "ccdl_cann_ops"
    assert created["sources"] == [str(root / "pybind.cpp"), str(root / "kernel.cpp")]
