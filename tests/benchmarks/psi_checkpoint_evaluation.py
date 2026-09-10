"""Strict model-only restoration and non-mutating offline evaluation context."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import random


def restore_model_weights(model: object, state: object, *, route: str) -> None:
    """Preflight every tensor before loading; never mutate checkpoint mappings."""
    import torch
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

    if route not in ("native", "cag", "rsag_qwd") or not isinstance(state, dict):
        raise ValueError("invalid evaluation model state or route")
    candidate = OrderedDict(state)
    if hasattr(state, "_metadata"):
        candidate._metadata = deepcopy(state._metadata)
    if route in ("native", "cag"):
        if not candidate or any(not isinstance(key, str) or not key.startswith("module.") for key in candidate):
            raise ValueError("DDP checkpoint model keys must have the module prefix")
        consume_prefix_in_state_dict_if_present(candidate, "module.")
    destination = model.state_dict()
    if set(candidate) != set(destination):
        raise ValueError("checkpoint model key mismatch")
    for key, tensor in candidate.items():
        target = destination[key]
        if (not isinstance(tensor, torch.Tensor) or not isinstance(target, torch.Tensor)
                or tensor.shape != target.shape or tensor.dtype != target.dtype):
            raise ValueError("checkpoint model shape/dtype mismatch: " + key)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("checkpoint model contains nonfinite values: " + key)
    model.load_state_dict(candidate, strict=True)


def validate_evaluation_identity(
    payload: dict, *, route: str, epoch: int, step: int, training_protocol: dict,
) -> None:
    """Additional binding after the training worker's full checkpoint validation."""
    from tests.benchmarks.psi_training_runtime import validate_training_protocol

    validate_training_protocol(training_protocol)
    if (route not in ("native", "cag", "rsag_qwd") or type(epoch) is not int or epoch < 1
            or type(step) is not int or step < 1):
        raise ValueError("invalid expected checkpoint evaluation identity")
    if (payload.get("route") != route or type(payload.get("epoch")) is not int
            or payload["epoch"] != epoch or type(payload.get("step")) is not int
            or payload["step"] != step or payload.get("training_protocol") != training_protocol):
        raise ValueError("checkpoint evaluation identity mismatch")


def validate_rsag_engine_identity(
    engine: object, *, gradient_route: str, parameter_route: str, rank: int, world_size: int,
) -> None:
    if type(engine) is not dict or type(engine.get("layout")) is not dict:
        raise ValueError("RSAG checkpoint engine/layout is invalid")
    layout = engine["layout"]
    if (engine.get("gradient_route") != gradient_route or engine.get("parameter_route") != parameter_route
            or type(layout.get("rank")) is not int or layout["rank"] != rank
            or type(layout.get("world_size")) is not int or layout["world_size"] != world_size):
        raise ValueError("RSAG checkpoint route/layout mismatch")


@contextmanager
def preserved_evaluation(model: object, *, seed: int):
    """Restore RNG, module modes, tensor state and gradients, including on error.

    Tensor/gradient mutations are restored and rejected. This is not a sandbox
    for arbitrary model code or replacement of the model's module structure.
    CUDA must be initialized before entry if evaluation will use it; otherwise
    only Python/NumPy/Torch CPU RNG state is captured. The CLI initializes CUDA
    before calling this context.
    """
    import numpy as np
    import torch

    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("evaluation seed must be a nonnegative 63-bit integer")
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.modules()]
    registrations = [(module, name, dict(getattr(module, name))) for module, _ in modes
                     for name in ("_parameters", "_buffers")]
    gradients = [(parameter, parameter.grad) for parameter in model.parameters()]
    unique_tensors = {id(tensor): tensor for _, _, values in registrations
                      for tensor in values.values() if tensor is not None}
    unique_tensors.update({id(grad): grad for _, grad in gradients if grad is not None})
    tensors = [(tensor, tensor.detach(), tensor.detach().clone(), tensor.requires_grad)
               for tensor in unique_tensors.values()]
    failed, mutated = False, False
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.random.default_generator.manual_seed(seed)
        if cuda_rng is not None:
            torch.cuda.manual_seed_all(seed)
        model.eval()
        with torch.no_grad():
            yield
    except BaseException:
        failed = True
        raise
    finally:
        try:
            for module, name, values in registrations:
                current = getattr(module, name)
                mutated = mutated or set(current) != set(values) or any(
                    current[key] is not value for key, value in values.items() if key in current
                )
            mutated = mutated or any(
                tensor.dtype != value.dtype or tensor.device != value.device
                or tensor.shape != value.shape or tensor.requires_grad != requires_grad
                or not torch.equal(tensor, value)
                for tensor, _, value, requires_grad in tensors
            )
            for parameter, gradient in gradients:
                mutated = mutated or parameter.grad is not gradient
            if mutated:
                with torch.no_grad():
                    for tensor, storage, value, requires_grad in tensors:
                        storage.copy_(value)
                        tensor.data = storage
                        tensor.requires_grad_(requires_grad)
                for module, name, values in registrations:
                    current = getattr(module, name)
                    current.clear()
                    current.update(values)
                for parameter, gradient in gradients:
                    parameter.grad = gradient
        finally:
            for module, mode in modes:
                module.training = mode
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        if mutated and not failed:
            raise ValueError("evaluation mutated model state or gradients")
