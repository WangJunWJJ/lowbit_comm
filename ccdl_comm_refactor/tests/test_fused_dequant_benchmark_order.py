from tests.benchmarks.fused_dequant_executor_gate import measure_balanced


def test_fused_dequant_benchmark_balances_measurement_order() -> None:
    order = []
    values = iter((4.0, 3.0, 2.0, 6.0))

    def measure(operation):
        operation()
        return next(values)

    baseline_ms, fused_ms = measure_balanced(
        measure,
        lambda: order.append("baseline"),
        lambda: order.append("fused"),
    )

    assert order == ["baseline", "fused", "fused", "baseline"]
    assert baseline_ms == 5.0
    assert fused_ms == 2.5
