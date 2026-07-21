from benchmarks.lm.plot_report import render_bar_svg


def test_svg_escapes_labels():
    svg = render_bar_svg({"A&B": 2.0}, "Speed <ratio>", "x")
    assert "A&amp;B" in svg
    assert "Speed &lt;ratio&gt;" in svg
