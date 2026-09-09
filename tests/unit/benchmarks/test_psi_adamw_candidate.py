"""Exact copy-then-step oracle for alternate AdamW storage."""

import pytest

torch = pytest.importorskip("torch")

from lowbit_comm.experimental.rsag import ShardLayout, ShardedAdamW
from tests.benchmarks import distributed_psi_v040_worker as worker
from tests.unit.benchmarks.test_psi_rsag_gradient_storage import _engine, _gradients


def _optimizer(layout, device):
    return ShardedAdamW(layout, torch.linspace(-2, 2, layout.padded_numel, device=device),
                        learning_rate=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def _old_step(source, gradient, lr, decay):
    target = _optimizer(source.layout, source.master.device)
    for name in ("master", "exp_avg", "exp_avg_sq"):
        getattr(target, name).copy_(getattr(source, name))
    target.step_count = source.step_count + 1
    valid = source.layout.valid_numel
    if valid:
        master, first, second = (getattr(target, name)[:valid]
                                 for name in ("master", "exp_avg", "exp_avg_sq"))
        gradient = gradient[:valid]
        beta1, beta2 = target.betas
        master.mul_(1.0 - lr * decay[:valid])
        master.mul_(1.0 - lr * target.weight_decay)
        first.lerp_(gradient, 1.0 - beta1)
        second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        correction1 = 1.0 - beta1**target.step_count
        correction2 = (1.0 - beta2**target.step_count) ** 0.5
        denominator = second.sqrt().div_(correction2).add_(target.eps)
        master.addcdiv_(first, denominator, value=-(lr / correction1))
    for name in ("master", "exp_avg", "exp_avg_sq"):
        getattr(target, name)[valid:].zero_()
    return target


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
@pytest.mark.parametrize("world_size,rank", [(1, 0), (3, 2), (8, 7)])
@pytest.mark.parametrize("global_numel", [7, 4099])
def test_alternate_update_matches_copy_step_bitwise(device, world_size, rank, global_numel):
    layout = ShardLayout.build(global_numel, world_size, rank)
    source, candidate = _optimizer(layout, device), _optimizer(layout, device)
    pointers = {source.master.data_ptr(), candidate.master.data_ptr()}
    decay = torch.linspace(0, 0.3, layout.padded_numel, device=device)
    for step in range(12):
        lr = (0.01, 0.0, 0.003)[step % 3]
        gradient = torch.sin(torch.arange(layout.padded_numel, device=device) + step)
        frozen = source.state_dict()
        expected = _old_step(source, gradient, lr, decay)
        candidate._step_from_prevalidated(source, gradient, learning_rate=lr, weight_decay=decay)
        for name in ("master", "exp_avg", "exp_avg_sq"):
            assert torch.equal(getattr(candidate, name).view(torch.int32), getattr(expected, name).view(torch.int32))
            assert torch.equal(getattr(source, name), frozen[name])
        assert candidate.step_count == source.step_count + 1
        assert candidate.master.data_ptr() in pointers
        source, candidate = candidate, source


@pytest.mark.parametrize("alias", ["master", "cross", "gradient", "decay"])
def test_candidate_rejects_alias_before_mutation(alias):
    layout = ShardLayout.build(7, 1, 0)
    source, candidate = _optimizer(layout, "cpu"), _optimizer(layout, "cpu")
    gradient, decay = torch.ones(7), torch.zeros(7)
    if alias == "master":
        candidate.master = source.master
    elif alias == "cross":
        candidate.exp_avg = candidate.master
    elif alias == "gradient":
        gradient = candidate.exp_avg
    else:
        decay = candidate.master
    frozen = source.state_dict()
    with pytest.raises(ValueError, match="alias"):
        candidate._step_from_prevalidated(source, gradient, learning_rate=0.01, weight_decay=decay)
    for name in ("master", "exp_avg", "exp_avg_sq"):
        assert torch.equal(getattr(source, name), frozen[name])
    assert candidate.step_count == 0


def test_worker_candidate_uses_direct_staging(monkeypatch):
    engine, _, _ = _engine(monkeypatch, world_size=1)
    called = []
    original = getattr(ShardedAdamW, "_step_from_prevalidated", None)
    assert original is not None, "direct staging entrypoint missing"
    def record(self, *args, **kwargs):
        called.append(args[0])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(ShardedAdamW, "_step_from_prevalidated", record)
    source = engine.sharded_optimizer
    _gradients(engine, 0.1)
    engine.step(None)
    assert called == [source]


def test_candidate_partial_computation_failure_preserves_committed_state(monkeypatch):
    layout = ShardLayout.build(7, 1, 0)
    source, candidate = _optimizer(layout, "cpu"), _optimizer(layout, "cpu")
    gradient, decay = torch.ones(7), torch.full((7,), 0.1)
    frozen = source.state_dict()
    original = torch.lerp
    def fail(*args, **kwargs):
        raise RuntimeError("injected moment failure")
    monkeypatch.setattr(torch, "lerp", fail)
    with pytest.raises(RuntimeError, match="injected moment failure"):
        candidate._step_from_prevalidated(source, gradient, learning_rate=0.01, weight_decay=decay)
    for name in ("master", "exp_avg", "exp_avg_sq"):
        assert torch.equal(getattr(source, name), frozen[name])
    assert source.step_count == 0
    monkeypatch.setattr(torch, "lerp", original)
    expected = _old_step(source, gradient, 0.01, decay)
    candidate._step_from_prevalidated(source, gradient, learning_rate=0.01, weight_decay=decay)
    for name in ("master", "exp_avg", "exp_avg_sq"):
        assert torch.equal(getattr(candidate, name).view(torch.int32), getattr(expected, name).view(torch.int32))
        assert torch.equal(frozen[name], getattr(source, name))
    assert candidate.step_count == 1


def test_direct_staging_does_not_copy_state_tensors(monkeypatch):
    layout = ShardLayout.build(7, 1, 0)
    source, candidate = _optimizer(layout, "cpu"), _optimizer(layout, "cpu")
    gradient, decay = torch.ones(7), torch.zeros(7)
    def forbidden(*args, **kwargs):
        raise AssertionError("copy-then-transform is forbidden")
    monkeypatch.setattr(torch.Tensor, "copy_", forbidden)
    candidate._step_from_prevalidated(source, gradient, learning_rate=0.01, weight_decay=decay)


def test_worker_zero_lr_keeps_checkpoint_restorable(monkeypatch):
    engine, _, _ = _engine(monkeypatch, world_size=1)
    engine.optimizer.param_groups[0]["lr"] = 0.0
    before = engine.master.clone()
    _gradients(engine, 0.1)
    engine.step(None)
    assert torch.equal(engine.master, before)
    assert engine.step_count == 1
    assert torch.count_nonzero(engine.exp_avg) > 0
    saved = engine.state_dict()
    restored = _optimizer(engine.layout, "cpu")
    restored.load_state_dict(saved["optimizer"])
    assert restored.step_count == 1
    assert restored.learning_rate == engine.base_learning_rate


@pytest.mark.parametrize("failure_stage", ["norm", "candidate", "model_staging"])
def test_worker_post_gradient_failure_rolls_back_feedback_and_retries_exactly(monkeypatch, failure_stage):
    engine, _, qwd = _engine(monkeypatch, world_size=1)
    baseline, _, _ = _engine(monkeypatch, world_size=1)
    for current in (engine, baseline):
        _gradients(current, 0.1)
        current.step(None)
    saved = engine.state_dict()
    frozen_model = [parameter.detach().clone() for parameter in engine.parameters]
    _gradients(baseline, 2.0)
    baseline.step(None)

    calls = []
    original_qwd = qwd.execute
    def record_qwd(*args, **kwargs):
        calls.append(True)
        return original_qwd(*args, **kwargs)
    monkeypatch.setattr(qwd, "execute", record_qwd)
    if failure_stage == "norm":
        target, name = worker, "norm_clip_coefficient"
    elif failure_stage == "candidate":
        target, name = engine, "_adamw_candidate"
    else:
        target, name = engine, "_copy_model_to_padded_flat"
    original = getattr(target, name)
    def fail_after_work(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected post-gradient failure")
    monkeypatch.setattr(target, name, fail_after_work)
    _gradients(engine, 2.0)
    with pytest.raises(RuntimeError, match="injected post-gradient failure"):
        engine.step(None)
    assert calls == []  # No parameter Work exists on these failure paths.
    assert engine.step_count == 1
    for name in ("master", "exp_avg", "exp_avg_sq"):
        assert torch.equal(getattr(engine, name), saved["optimizer"][name])
    for actual, expected in zip(engine.parameters, frozen_model):
        assert torch.equal(actual, expected)
    assert torch.equal(engine.gradient_plan._committed_residual,
                       saved["gradient_feedback"]["residual"])
    restore_name = {"norm": "norm_clip_coefficient", "candidate": "_adamw_candidate",
                    "model_staging": "_copy_model_to_padded_flat"}[failure_stage]
    monkeypatch.setattr(target, restore_name, original)
    _gradients(engine, 2.0)
    engine.step(None)
    assert calls == [True]
    assert engine.step_count == baseline.step_count
    for name in ("master", "exp_avg", "exp_avg_sq"):
        assert torch.equal(getattr(engine, name).view(torch.int32),
                           getattr(baseline, name).view(torch.int32))
    for actual, expected in zip(engine.parameters, baseline.parameters):
        assert torch.equal(actual, expected)
    assert torch.equal(engine.gradient_plan._committed_residual,
                       baseline.gradient_plan._committed_residual)
