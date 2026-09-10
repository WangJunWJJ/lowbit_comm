"""Cross-branch contracts; CPU transport stubs are not distributed GPU proof."""
from pathlib import Path

import pytest

from tests.benchmarks import distributed_psi_v040_evaluate as evaluator
from tests.benchmarks import distributed_psi_v040_worker as worker
from tests.benchmarks.psi_training_runtime import PhaseTimer
from tests.benchmarks.psi_v040_training import parse_args, validate_task_result
from tests.unit.benchmarks.test_psi_offline_evaluate import _bound, _plan
from tests.unit.benchmarks.test_psi_rsag_gradient_storage import _engine, _gradients
from tests.unit.benchmarks.test_psi_window_timing import _window_result


@pytest.mark.parametrize("gradient", ["reduced_shard_int8_group64_ef", "reduced_shard_native_fp32"])
@pytest.mark.parametrize("parameter", ["qwd_group64_refresh100", "all_refresh_fp32"])
def test_quality_routes_keep_exact_engine_state_across_observation_modes(monkeypatch, gradient, parameter):
    import torch

    original_timer = worker._cuda_timed
    states = []
    for mode in ("production", "window"):
        args = parse_args(["--route", "rsag_qwd", "--timing-mode", mode,
                           "--rsag-gradient-route", gradient, "--rsag-parameter-route", parameter])
        assert (args.rsag_gradient_route, args.rsag_parameter_route) == (gradient, parameter)
        engine, _, qwd = _engine(monkeypatch, gradient_route=gradient, parameter_route=parameter)

        def forbidden_cuda_provider():
            raise AssertionError("production/window phase timer must not create CUDA events")

        monkeypatch.setattr(worker, "_cuda_timed", original_timer)
        monkeypatch.setattr(worker, "_PHASE_TIMER", PhaseTimer(mode, forbidden_cuda_provider))
        for offset in (0.125, 1.25, -2.0):
            _gradients(engine, offset)
            engine.step(None)
        assert engine.step_count == 3
        assert qwd.modes == (["fp_refresh"]*3 if parameter == "all_refresh_fp32"
                             else ["fp_refresh", "qwd", "qwd"])
        state = engine.state_dict()
        assert state["gradient_route"] == gradient
        assert state["parameter_route"] == parameter
        if gradient == "reduced_shard_native_fp32":
            assert engine._gradient_communication_bytes() == 24
        states.append(state)

    def equal(left, right):
        assert type(left) is type(right)
        if isinstance(left, torch.Tensor):
            assert left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equal(a, b)
        else:
            assert left == right

    equal(*states)


def test_offline_v3_evaluator_rejects_valid_window_v4_without_implicit_migration(tmp_path):
    plan = _plan(tmp_path)
    window = _window_result()
    assert validate_task_result(window) is window
    plan["training_result"] = _bound(Path(plan["training_result"]["path"]), window)
    plan["training_protocol"] = _bound(Path(plan["training_protocol"]["path"]), {
        "training_protocol": window["execution_protocol"],
        "result_sha256": plan["training_result"]["sha256"],
    })
    record = _bound(tmp_path/"window-plan.json", plan)
    with pytest.raises(ValueError, match="original result/protocol binding mismatch"):
        evaluator.load_plan(record["path"], record["sha256"])
