"""Production timing must not serialize every CUDA stage for instrumentation."""

from types import SimpleNamespace

import pytest

from tests.benchmarks import psi_training_runtime as runtime


def test_production_nested_phases_do_not_create_events_or_synchronize():
    calls = []
    cuda = SimpleNamespace(synchronize=lambda: calls.append("sync"))
    ticks = iter([10.0, 10.25])
    timer = runtime.PhaseTimer("production", lambda: cuda, clock=lambda: next(ticks))

    def update():
        result, elapsed = timer.measure(lambda: calls.append("nested") or "updated")
        assert elapsed == 0.0
        return result

    result = timer.training_step(
        lambda: calls.append("forward") or 3,
        lambda loss: calls.append(("backward", loss)),
        update,
    )
    assert result == (3, "updated", 0.0, 0.0, 0.25)
    assert calls == ["sync", "forward", ("backward", 3), "nested", "sync"]


def test_production_phase_is_cuda_free_and_propagates_errors():
    timer = runtime.PhaseTimer("production", lambda: pytest.fail("CUDA accessed"))
    assert timer.measure(lambda: 7) == (7, 0.0)
    with pytest.raises(ZeroDivisionError):
        timer.measure(lambda: 1 / 0)


def test_diagnostic_retains_cuda_event_timing():
    calls = []

    class Event:
        def record(self):
            calls.append("record")

        def synchronize(self):
            calls.append("sync")

        def elapsed_time(self, end):
            return 2.5

    timer = runtime.PhaseTimer(
        "diagnostic", lambda: SimpleNamespace(Event=lambda **kwargs: Event())
    )
    assert timer.measure(lambda: calls.append("action") or 4) == (4, 0.0025)
    assert calls == ["record", "action", "record", "sync"]


def test_production_protocol_explicitly_marks_phase_timings_unavailable():
    protocol = runtime.training_protocol(
        rank=0, world_size=2, batch_size=16, native_ddp_mode="standard",
        timing_mode="production",
    )
    assert protocol["version"] == 3
    assert protocol["measurement"] == "synchronized_step_wall"
    assert protocol["phase_breakdown_available"] is False
    assert protocol["core_record_field"] == "update_s"
    runtime.validate_training_protocol(protocol)
    protocol["phase_breakdown_available"] = True
    with pytest.raises(ValueError, match="protocol"):
        runtime.validate_training_protocol(protocol)


def test_bad_timing_mode_fails_closed():
    with pytest.raises(ValueError):
        runtime.PhaseTimer("approximate", lambda: None)
    with pytest.raises(ValueError):
        runtime.training_protocol(
            rank=0, world_size=2, batch_size=16, native_ddp_mode="standard",
            timing_mode="approximate",
        )


def test_cli_production_timing_is_explicit_not_an_implicit_old_result_change():
    from tests.benchmarks.psi_v040_training import parse_args

    assert parse_args(["--route", "native"]).timing_mode == "diagnostic"
    assert parse_args(["--route", "native", "--timing-mode", "production"]).timing_mode == "production"
