"""Deterministic configurable MLP used by the training example."""

from __future__ import annotations

from typing import Any


def build_mlp(config: Any, *, torch: Any) -> Any:
    layers = []
    input_features = config.input_dim
    for _ in range(config.depth):
        layers.append(torch.nn.Linear(input_features, config.hidden_dim))
        layers.append(torch.nn.GELU())
        input_features = config.hidden_dim
    layers.append(torch.nn.Linear(config.hidden_dim, config.num_classes))
    return torch.nn.Sequential(*layers)


def count_parameters(model: Any) -> int:
    wrapped = getattr(model, "module", model)
    return sum(int(parameter.numel()) for parameter in wrapped.parameters())
