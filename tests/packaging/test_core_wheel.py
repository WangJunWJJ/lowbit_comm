"""Wheel packaging contracts for the experimental RSAG/qWD namespace."""

from __future__ import annotations

import builtins
import importlib
import sys


def test_experimental_namespace_import_is_torch_lazy(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("experimental import eagerly loaded torch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for name in tuple(sys.modules):
        if name == "lowbit_comm.experimental" or name.startswith(
            "lowbit_comm.experimental."
        ):
            sys.modules.pop(name)

    module = importlib.import_module("lowbit_comm.experimental")

    assert module.RSAG_EVIDENCE_SCHEMA_VERSION == 1
    assert module.ShardLayout.__module__ == "lowbit_comm.experimental.rsag"
