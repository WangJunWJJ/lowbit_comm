from types import SimpleNamespace

import pytest

from tests.benchmarks import psi_training_runtime as runtime
from tests.benchmarks import psi_v040_training as cli
from tests.benchmarks import distributed_psi_v040_worker as worker


@pytest.mark.parametrize("mode,suffix", [("production", "step_wall"), ("diagnostic", "phase_diagnostic")])
def test_fp32_observability_has_separate_scope(mode, suffix):
    info = runtime.measurement_observability(mode, "native", "standard", model_precision="fp32")
    assert info["scope"] == "native_fp32_amp_fp16_" + suffix
    assert not info["gradient_communication_time_available"]


@pytest.mark.parametrize("route", ["cag", "rsag_qwd"])
def test_result_rejects_forged_fp32_non_native(route):
    from tests.unit.benchmarks.test_psi_v040_training import _task_result
    result = _task_result()
    result.update(schema_version=3, route=route)
    result["execution_protocol"] = runtime.training_protocol(
        rank=0, world_size=result["world_size"], batch_size=result["batch_size_per_rank"],
        native_ddp_mode="standard", model_precision="fp32",
    )
    with pytest.raises(ValueError, match="FP32.*native"):
        cli.validate_task_result(result)


@pytest.mark.parametrize("field,value", [("phase_breakdown_available", 0), ("model_warmup_backwards", 2.0)])
def test_protocol_rejects_equivalent_wrong_types(field, value):
    protocol = runtime.training_protocol(rank=0, world_size=2, batch_size=4,
        native_ddp_mode="standard", timing_mode="production")
    protocol[field] = value
    with pytest.raises(ValueError, match="protocol"):
        runtime.validate_training_protocol(protocol)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp32_master_aliases_parameters_and_gradients_and_matches_adamw(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p = torch.nn.Parameter(torch.tensor([1.1, -0.7], device=device))
    ref = torch.nn.Parameter(p.detach().clone())
    optimizer = torch.optim.AdamW([p], lr=0.013)
    reference = torch.optim.AdamW([ref], lr=0.013)
    bridge = runtime.FP32MasterWeights(optimizer)
    assert bridge.master_parameters[0] is p
    assert optimizer.param_groups[0]["params"][0] is p
    for step in range(6):
        p.grad = torch.tensor([0.2 + step / 10, -0.3], device=device)
        ref.grad = p.grad.clone()
        grad = p.grad
        bridge.prepare_gradients()
        assert p.grad is grad
        optimizer.step()
        reference.step()
        version = p._version
        bridge.publish()
        assert p._version == version
        torch.testing.assert_close(p, ref, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq"):
            assert optimizer.state[p][key].dtype == torch.float32
            torch.testing.assert_close(optimizer.state[p][key], reference.state[ref][key], rtol=0, atol=0)
        bridge.zero_grad()
        assert p.grad is None
    snapshot = bridge.state_dict()
    with torch.no_grad():
        p.add_(1)
    assert not torch.equal(snapshot["parameters"][0], p)
    bridge.load_state_dict(snapshot)
    torch.testing.assert_close(p, ref, rtol=0, atol=0)


@pytest.mark.parametrize("route,mode", [("cag", "standard"), ("rsag_qwd", "standard"), ("native", "diagnostic")])
def test_fp32_invalid_route_rejected_before_torch_or_workspace(monkeypatch, route, mode):
    def forbidden():
        raise AssertionError("must reject before importing torch")
    monkeypatch.setattr(worker, "_torch", forbidden)
    with pytest.raises(ValueError, match="FP32.*native.*standard"):
        worker._run(SimpleNamespace(model_precision="fp32", route=route, native_ddp_mode=mode, timing_mode="production"))


def test_cli_precision_and_model_conversion():
    torch = pytest.importorskip("torch")
    assert cli.parse_args(["--route", "native"]).model_precision == "fp16"
    assert cli.parse_args(["--route", "native", "--model-precision", "fp32"]).model_precision == "fp32"
    model = torch.nn.Linear(2, 1).half()
    parameters = tuple(model.parameters())
    worker._convert_model_precision(model, "fp32")
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(a is b for a, b in zip(parameters, model.parameters()))
    worker._convert_model_precision(model, "fp16")
    assert all(p.dtype == torch.float16 for p in model.parameters())


def test_precision_protocol_and_historical_read():
    args = dict(rank=0, world_size=2, batch_size=4, native_ddp_mode="standard")
    fp16 = runtime.training_protocol(**args)
    fp32 = runtime.training_protocol(**args, model_precision="fp32")
    assert fp32["model_precision"] == "fp32"
    assert fp32 != fp16
    runtime.validate_training_protocol(fp32)
    old = {k: v for k, v in fp16.items() if k not in {
        "timing_mode", "phase_breakdown_available", "core_record_field",
        "model_warmup_backwards", "oracle_policy",
    }}
    old["version"] = 2
    runtime.validate_training_protocol(old)
    old["model_precision"] = "fp32"
    with pytest.raises(ValueError, match="protocol"):
        runtime.validate_training_protocol(old)


def test_checkpoint_precision_drift_rejected_before_mutation(monkeypatch):
    args = dict(rank=0, world_size=2, batch_size=4, native_ddp_mode="standard")
    fp16 = runtime.training_protocol(**args)
    fp32 = runtime.training_protocol(**args, model_precision="fp32")
    monkeypatch.setattr(worker, "_torch", lambda: SimpleNamespace())
    monkeypatch.setattr(worker, "_validate_checkpoint_payload", lambda *a, **k: None)
    class Untouched:
        def load_state_dict(self, state):
            raise AssertionError("mutated before precision validation")
    with pytest.raises(ValueError, match="protocol"):
        worker._load_checkpoint(None, route="native", model=Untouched(),
            engine=SimpleNamespace(training_protocol=fp32), scheduler=None,
            scaler=None, amp_scale=None, train_loader=None,
            payload={"training_protocol": fp16, "model": {}})
