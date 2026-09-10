"""Read-only model evaluation and strict checkpoint identity contracts."""
from collections import OrderedDict
from copy import deepcopy
import random

import numpy as np
import pytest

from tests.benchmarks import psi_checkpoint_evaluation as evaluation

torch = pytest.importorskip("torch")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.register_buffer("normalizer", torch.tensor([2.0, 3.0]))
        self.register_buffer("empty", torch.zeros(0))
        self.child = torch.nn.Dropout()


def _state(model, route):
    state = deepcopy(model.state_dict())
    if route in ("native", "cag"):
        wrapped = OrderedDict(("module."+key, value) for key, value in state.items())
        wrapped._metadata = OrderedDict(("module"+("."+key if key else ""), value)
                                        for key, value in state._metadata.items())
        return wrapped
    return state


@pytest.mark.parametrize("route", ["native", "cag", "rsag_qwd"])
def test_model_restore_preserves_source_state_and_loads_normalizer(route):
    source, target = _Model(), _Model()
    source.normalizer.add_(7)
    state = _state(source, route)
    keys, metadata = list(state), deepcopy(state._metadata)
    evaluation.restore_model_weights(target, state, route=route)
    assert list(state) == keys and state._metadata == metadata
    assert all(torch.equal(a, target.state_dict()[key]) for key, a in source.state_dict().items())


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype", "nan", "prefix"])
def test_invalid_model_state_is_rejected_before_any_weight_changes(mutation):
    model = _Model()
    before = deepcopy(model.state_dict())
    state = _state(_Model(), "native")
    key = "module.linear.weight"
    if mutation == "missing":
        del state[key]
    elif mutation == "extra":
        state["module.unexpected"] = torch.ones(1)
    elif mutation == "shape":
        state[key] = torch.zeros(4)
    elif mutation == "dtype":
        state[key] = state[key].half()
    elif mutation == "nan":
        state[key].fill_(float("nan"))
    else:
        state["linear.weight"] = state.pop(key)
    with pytest.raises(ValueError):
        evaluation.restore_model_weights(model, state, route="native")
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_rsag_does_not_silently_strip_ddp_prefix():
    with pytest.raises(ValueError):
        evaluation.restore_model_weights(_Model(), _state(_Model(), "native"), route="rsag_qwd")


def _rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_rng_equal(before):
    after = _rng_snapshot()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


@pytest.mark.parametrize("throw", [False, True])
def test_evaluation_context_preserves_rng_modes_weights_and_gradients(throw):
    model = _Model().train()
    model.child.eval()
    model.linear.weight.grad = torch.ones_like(model.linear.weight)
    state = deepcopy(model.state_dict())
    rng = _rng_snapshot()
    try:
        with evaluation.preserved_evaluation(model, seed=20260821):
            assert not any(module.training for module in model.modules())
            assert not torch.is_grad_enabled()
            random.random(), np.random.rand(), torch.rand(2)
            if throw:
                raise RuntimeError("evaluation failure")
    except RuntimeError as error:
        assert throw and str(error) == "evaluation failure"
    _assert_rng_equal(rng)
    assert model.training and not model.child.training
    assert torch.equal(model.linear.weight.grad, torch.ones_like(model.linear.weight))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())


def test_mutating_evaluation_is_rejected_and_state_is_restored():
    model = _Model()
    before = deepcopy(model.state_dict())
    rng = _rng_snapshot()
    with pytest.raises(ValueError, match="mutated"):
        with evaluation.preserved_evaluation(model, seed=3):
            model.normalizer.add_(1)
            model.linear.weight.add_(4)
            model.linear.weight.grad = torch.ones_like(model.linear.weight)
    _assert_rng_equal(rng)
    assert model.linear.weight.grad is None
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_checkpoint_identity_mismatch_is_rejected():
    from tests.benchmarks.psi_training_runtime import training_protocol
    protocol = training_protocol(rank=0, world_size=2, batch_size=16, native_ddp_mode="standard",
                                 timing_mode="production", data_mode="deterministic", seed=21)
    payload = {"route": "native", "epoch": 3, "step": 4212, "training_protocol": protocol}
    evaluation.validate_evaluation_identity(payload, route="native", epoch=3, step=4212,
                                            training_protocol=protocol)
    for field, wrong in [("route", "cag"), ("epoch", 2), ("step", True)]:
        broken = dict(payload, **{field: wrong})
        with pytest.raises(ValueError):
            evaluation.validate_evaluation_identity(broken, route="native", epoch=3, step=4212,
                                                    training_protocol=protocol)
    broken = deepcopy(payload)
    broken["training_protocol"]["rank"] = 1
    with pytest.raises(ValueError):
        evaluation.validate_evaluation_identity(broken, route="native", epoch=3, step=4212,
                                                training_protocol=protocol)


@pytest.mark.parametrize("mutation", ["dtype", "shape", "replace", "nonpersistent"])
def test_tensor_storage_contract_is_restored_after_mutation(mutation):
    model = _Model()
    model.register_buffer("transient", torch.tensor(5.0), persistent=False)
    original = model.normalizer
    pointer = original.data_ptr()
    before = original.clone()
    with pytest.raises(ValueError, match="mutated"):
        with evaluation.preserved_evaluation(model, seed=3):
            if mutation == "dtype":
                model.normalizer.data = model.normalizer.double()
            elif mutation == "shape":
                model.normalizer.data = torch.zeros(8)
            elif mutation == "replace":
                model.normalizer = torch.zeros(2)
            else:
                model.transient.add_(1)
    assert model.normalizer is original and original.data_ptr() == pointer
    assert model.normalizer.dtype == before.dtype and torch.equal(model.normalizer, before)
    assert model.transient.item() == 5.0


@pytest.mark.parametrize("engine", [None, [], {}, {"layout": None},
    {"gradient_route": "grad", "parameter_route": "param", "layout": {"rank": True, "world_size": 2}},
    {"gradient_route": "grad", "parameter_route": "param", "layout": {"rank": 1, "world_size": 3}},
])
def test_malformed_rsag_engine_identity_has_controlled_validation_error(engine):
    with pytest.raises(ValueError):
        evaluation.validate_rsag_engine_identity(engine, gradient_route="grad", parameter_route="param",
                                                 rank=1, world_size=2)


def test_rsag_engine_identity_binds_both_routes_and_world():
    engine = {"gradient_route": "grad", "parameter_route": "param", "layout": {"rank": 1, "world_size": 2}}
    evaluation.validate_rsag_engine_identity(engine, gradient_route="grad", parameter_route="param",
                                             rank=1, world_size=2)
    with pytest.raises(ValueError):
        evaluation.validate_rsag_engine_identity(engine, gradient_route="other", parameter_route="param",
                                                 rank=1, world_size=2)
