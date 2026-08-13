from __future__ import annotations

from pathlib import Path


def test_native_compressed_work_does_not_expose_transport_future_as_pipeline() -> None:
    source = (
        Path(__file__).parents[3]
        / "src"
        / "lowbit_comm"
        / "backends"
        / "cuda"
        / "csrc"
        / "executor"
        / "compressed_work.cpp"
    ).read_text(encoding="utf-8")

    get_future = source.split("py::object CompressedWork::get_future() const", 1)[1]
    get_future = get_future.split("py::object CompressedWork::result()", 1)[0]
    assert 'transport_work_.attr("get_future")' not in get_future
