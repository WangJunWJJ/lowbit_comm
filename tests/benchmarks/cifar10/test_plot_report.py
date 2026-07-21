from benchmarks.cifar10.plot_report import render_bar_svg


def test_render_bar_svg_contains_all_labels_and_values():
    svg = render_bar_svg(
        {"baseline": 10.0, "ccdl": 15.5},
        title="Throughput",
        unit="images/s",
    )
    assert svg.startswith("<svg")
    assert "baseline" in svg
    assert "ccdl" in svg
    assert "15.500" in svg
    assert "images/s" in svg
