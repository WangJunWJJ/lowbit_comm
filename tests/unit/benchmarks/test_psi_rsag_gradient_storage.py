"""CPU/CUDA storage contracts; fake transport isolates the real worker update."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from lowbit_comm.experimental.rsag import ShardLayout
from tests.benchmarks import distributed_psi_v040_worker as worker


@dataclass
class _WireLayout:
    send_payload_bytes: int = 7
    receive_payload_bytes: int = 7


def _engine(
    monkeypatch,
    *,
    dtype=torch.float16,
    world_size=3,
    device="cpu",
    parameter_route="qwd_group64_refresh100",
    gradient_route="reduced_shard_int8_group64_ef",
):
    model = torch.nn.ParameterList([
        torch.nn.Parameter(torch.arange(6, device=device, dtype=dtype).view(2, 3)),
        torch.nn.Parameter(torch.ones(1, device=device, dtype=dtype)),
    ])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    layout = ShardLayout.build(7, world_size, world_size - 1)
    observed = []
    plan = SimpleNamespace(layout=_WireLayout(), _committed_residual=None)

    def reduce(value):
        observed.append(value)
        plan._committed_residual = value.clone()
        shard = torch.zeros(layout.padded_numel, device=device, dtype=torch.float16)
        shard[:layout.valid_numel].copy_(value[layout.start:layout.start + layout.valid_numel])
        return SimpleNamespace(wait=lambda: SimpleNamespace(value=shard))

    plan.execute = reduce
    plan._restore_committed_residual = lambda value: setattr(plan, "_committed_residual", value)
    qwd = SimpleNamespace(fail=False, modes=[])

    def gather(master, flat, mode):
        qwd.modes.append(mode)
        def wait():
            if qwd.fail:
                raise RuntimeError("injected qWD failure")
            result = flat.clone()
            result[layout.start:layout.start + layout.valid_numel].copy_(master[:layout.valid_numel])
            return result
        return SimpleNamespace(wait=wait)

    qwd.execute = gather
    monkeypatch.setattr(worker, "_create_rsag_qwd_plans", lambda *a, **k: SimpleNamespace(
        layout=layout, gradient_plan=plan, qwd_plan=qwd,
        qwd_gathered_payload_bytes=14, fp32_gathered_bytes=28,
    ))
    monkeypatch.setattr(worker, "_cuda_timed", lambda action: (action(), 0.0))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_unscale_and_detect_overflow", lambda *a: (False, None, 0.0, 0))
    monkeypatch.setattr(worker, "_advance_amp_scaler", lambda *a, **k: 1.0)
    engine = worker.RSAGQWDUpdateEngine(
        model=model, optimizer=optimizer, grad_clip=1.0, rank=world_size - 1,
        world_size=world_size, process_group=object(), amp_scale=worker._AmpScaleState(1.0),
        parameter_route=parameter_route,
        gradient_route=gradient_route,
    )
    return engine, observed, qwd


def _gradients(engine, offset):
    # Include a noncontiguous input and fresh grad objects on every iteration.
    first = engine.parameters[0]
    first.grad = (torch.arange(6, device=first.device, dtype=first.dtype) + offset).view(3, 2).t()
    engine.parameters[1].grad = torch.full_like(engine.parameters[1], offset + 0.25)
    return torch.cat([p.grad.detach().reshape(-1) for p in engine.parameters]).half()


def test_all_refresh_route_publishes_fp32_every_successful_update(monkeypatch):
    engine, _, qwd = _engine(monkeypatch, parameter_route="all_refresh_fp32")

    for offset in (0.125, 1.25, -2.0):
        _gradients(engine, offset)
        engine.step(None)

    assert engine.parameter_route == "all_refresh_fp32"
    assert engine.step_count == 3
    assert qwd.modes == ["fp_refresh", "fp_refresh", "fp_refresh"]


def test_all_refresh_checkpoint_rejects_default_route(monkeypatch):
    default, _, _ = _engine(monkeypatch)
    all_refresh, _, _ = _engine(monkeypatch, parameter_route="all_refresh_fp32")

    checkpoint = all_refresh.state_dict()
    with pytest.raises(ValueError, match="parameter route"):
        default.load_state_dict(checkpoint)


def test_native_gradient_route_is_explicit_and_checkpoint_bound(monkeypatch):
    engine, _, _ = _engine(
        monkeypatch,
        gradient_route="reduced_shard_native_fp32",
    )

    assert engine.gradient_route == "reduced_shard_native_fp32"
    checkpoint = engine.state_dict()
    assert checkpoint["gradient_route"] == "reduced_shard_native_fp32"

    default, _, _ = _engine(monkeypatch)
    with pytest.raises(ValueError, match="gradient route"):
        default.load_state_dict(checkpoint)


def test_native_gradient_route_reports_nonzero_wire_estimate(monkeypatch):
    engine, _, _ = _engine(
        monkeypatch,
        world_size=3,
        gradient_route="reduced_shard_native_fp32",
    )

    # Global 7-element FP16 reduce-scatter, counting send+receive traffic.
    assert engine._gradient_communication_bytes() == 18


@pytest.mark.parametrize("world_size", [1, 3, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_step_reuses_gradient_storage_with_exact_values(monkeypatch, world_size, dtype):
    engine, observed, _ = _engine(monkeypatch, dtype=dtype, world_size=world_size)
    for offset in (0.125, -2.5, 7.0):
        expected = _gradients(engine, offset)
        engine.step(None)
        torch.testing.assert_close(observed[-1], expected, rtol=0, atol=0)
        assert observed[-1].is_contiguous()
        assert observed[-1].dtype == torch.float16
    assert len({value.data_ptr() for value in observed}) == 1
    assert engine.step_count == 3
    assert torch.count_nonzero(engine.master[engine.layout.valid_numel:]) == 0


def test_reused_storage_does_not_alias_checkpoint_or_failed_update(monkeypatch):
    engine, observed, qwd = _engine(monkeypatch)
    _gradients(engine, 0.125)
    engine.step(None)
    saved = engine.state_dict()
    frozen = {key: saved["optimizer"][key].clone() for key in ("master", "exp_avg", "exp_avg_sq")}
    feedback = saved["gradient_feedback"]["residual"].clone()
    model = [p.detach().clone() for p in engine.parameters]
    qwd.fail = True
    _gradients(engine, 2.0)
    with pytest.raises(RuntimeError, match="injected qWD failure"):
        engine.step(None)
    assert engine.step_count == 1
    for key, value in frozen.items():
        torch.testing.assert_close(getattr(engine, key), value, rtol=0, atol=0)
        torch.testing.assert_close(saved["optimizer"][key], value, rtol=0, atol=0)
    for parameter, value in zip(engine.parameters, model):
        torch.testing.assert_close(parameter, value, rtol=0, atol=0)
    torch.testing.assert_close(engine.gradient_plan._committed_residual, feedback, rtol=0, atol=0)
    qwd.fail = False
    _gradients(engine, -3.0)
    engine.step(None)
    for key, value in frozen.items():
        torch.testing.assert_close(saved["optimizer"][key], value, rtol=0, atol=0)
    torch.testing.assert_close(saved["gradient_feedback"]["residual"], feedback, rtol=0, atol=0)
    assert len({value.data_ptr() for value in observed}) == 1


def test_overflow_and_missing_gradient_do_not_touch_reused_storage(monkeypatch):
    engine, observed, _ = _engine(monkeypatch)
    _gradients(engine, 1.0)
    engine.step(None)
    packed = observed[-1].clone()
    _gradients(engine, 2.0)
    monkeypatch.setattr(worker, "_unscale_and_detect_overflow", lambda *a: (True, None, 0.0, 0))
    assert engine.step(None)["skipped"]
    assert len(observed) == 1
    assert engine.step_count == 1
    torch.testing.assert_close(observed[-1], packed, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="every RSAG/qWD parameter requires a gradient"):
        engine.step(None)
    assert len(observed) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA device required")
def test_cuda_step_reuses_gradient_storage(monkeypatch):
    engine, observed, _ = _engine(monkeypatch, device="cuda")
    for offset in (0.125, -2.0):
        expected = _gradients(engine, offset)
        engine.step(None)
        torch.cuda.synchronize()
        torch.testing.assert_close(observed[-1], expected, rtol=0, atol=0)
    assert observed[0].data_ptr() == observed[1].data_ptr()
