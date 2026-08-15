"""Contracts for the narrow, import-safe public communication facade."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest

import lowbit_comm
from lowbit_comm.api.intent import (
    CommunicationIntent,
    CompletionMode,
    OutputSemantics,
    ReductionOp,
    ShapeFamily,
    TensorSpec,
)
from lowbit_comm.api.policy import (
    AutoPolicy,
    CollectiveKind,
    CompressionKind,
    ExplicitPolicy,
    NativePolicy,
    StrategySpec,
    TopologyKind,
)
from lowbit_comm.backends.protocols import BackendPlan
from lowbit_comm.compiler.evidence import EnvironmentFingerprint
from lowbit_comm.core.errors import CompileError, ExecutionError
from lowbit_comm.core.plan import (
    CompilationContext,
    ExecutionPlan,
    PlanOrigin,
)
from lowbit_comm.runtime.work import CompletedWork, FailedWork


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_PUBLIC_NAMES = {
    "AccumulationDType",
    "AutoConstraints",
    "AutoPolicy",
    "CollectiveKind",
    "CommunicationIntent",
    "CommunicationWork",
    "CompilationContext",
    "CompiledCommunicator",
    "CompletionMode",
    "CompressionKind",
    "ExplicitPolicy",
    "FullTensorResult",
    "NativePolicy",
    "OutputSemantics",
    "ReducedShardMetadata",
    "ReducedShardResult",
    "ReductionOp",
    "ShapeFamily",
    "StrategySpec",
    "TensorSpec",
    "TopologyKind",
    "compile_communicator",
}


class CommunicationIntentSubclass(CommunicationIntent):
    """An intent subtype rejected by the exact facade boundary."""


class NativePolicySubclass(NativePolicy):
    """A native-policy subtype rejected by the exact facade boundary."""


class AutoPolicySubclass(AutoPolicy):
    """An auto-policy subtype rejected by the exact facade boundary."""


class ExplicitPolicySubclass(ExplicitPolicy):
    """An explicit-policy subtype rejected by the exact facade boundary."""


class CompilationContextSubclass(CompilationContext):
    """A context subtype rejected by the exact facade boundary."""


class ExecutionPlanSubclass(ExecutionPlan):
    """A plan subtype rejected by the exact facade boundary."""


class EchoBackendPlan:
    """Backend plan that records direct execution calls."""

    def __init__(self) -> None:
        self.execute_calls = 0

    def execute(self, value: object) -> CompletedWork[object]:
        self.execute_calls += 1
        return CompletedWork(value)


class ReturningBackendPlan:
    """Backend plan that returns one pre-created work unchanged."""

    def __init__(self, work: FailedWork[object]) -> None:
        self.work = work

    def execute(self, value: object) -> FailedWork[object]:
        del value
        return self.work


@dataclass
class CountingCompiler:
    """Structural compiler double returning one formal plan."""

    plan: ExecutionPlan
    compile_call_count: int = 0

    def compile(
        self,
        intent: CommunicationIntent,
        policy: NativePolicy,
        context: CompilationContext,
    ) -> ExecutionPlan:
        del intent, policy, context
        self.compile_call_count += 1
        return self.plan


def _intent(
    intent_type: type[CommunicationIntent] = CommunicationIntent,
) -> CommunicationIntent:
    return intent_type(
        tensor=TensorSpec(dtype="float32", shape=(1,)),
        shape_family=ShapeFamily(max_numel=1, alignment=1),
        reduction=ReductionOp.SUM,
        output=OutputSemantics.FULL_TENSOR,
        completion=CompletionMode.SYNC,
        world_size=1,
        rank=0,
    )


def _context(
    context_type: type[CompilationContext] = CompilationContext,
) -> CompilationContext:
    return context_type(
        environment=EnvironmentFingerprint.from_mapping(
            {"hardware": "contract-test"}
        ),
        workspace_budget_bytes=0,
        node_count=1,
        workload_class="contract-test",
        bucket_min_bytes=4,
        bucket_max_bytes=4,
    )


def _strategy() -> StrategySpec:
    return StrategySpec(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=TopologyKind.BACKEND_DEFAULT,
    )


def _plan(
    backend_plan: object,
    plan_type: type[ExecutionPlan] = ExecutionPlan,
) -> ExecutionPlan:
    return plan_type(
        intent=_intent(),
        strategy=_strategy(),
        backend_id="contract-test",
        backend_plan=cast(BackendPlan, backend_plan),
        origin=PlanOrigin.NATIVE,
        signature="contract-test-signature",
        evidence_fingerprint=None,
    )


def _corrupted_plan_without_execute() -> ExecutionPlan:
    """Bypass frozen state to exercise the facade's defensive check."""
    plan = _plan(EchoBackendPlan())
    object.__setattr__(plan, "backend_plan", object())
    return plan


def test_public_api_is_exactly_the_stable_semantic_surface() -> None:
    assert set(lowbit_comm.__all__) == EXPECTED_PUBLIC_NAMES
    assert set(lowbit_comm.api.__all__) == EXPECTED_PUBLIC_NAMES
    assert len(lowbit_comm.__all__) == len(set(lowbit_comm.__all__))


def test_public_api_excludes_internal_and_legacy_symbols() -> None:
    excluded = {
        "BackendRegistry",
        "Compiler",
        "EvidenceStore",
        "ExecutionPlan",
        "ReferenceBackend",
        "Registry",
        "_C",
        "full_fused",
        "qall_gather_dyn",
        "sharded_compressed",
    }

    assert excluded.isdisjoint(lowbit_comm.__all__)


def test_isolated_import_does_not_load_torch_or_cuda_extension() -> None:
    script = """
import importlib.abc
import sys

class ForbiddenImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError(f"forbidden import: {fullname}")
        if fullname == "lowbit_comm._C":
            raise AssertionError(f"forbidden import: {fullname}")
        return None

sys.meta_path.insert(0, ForbiddenImport())
sys.path.insert(0, sys.argv[1])
import lowbit_comm
assert "torch" not in sys.modules
assert "lowbit_comm._C" not in sys.modules
print(",".join(sorted(lowbit_comm.__all__)))
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert set(result.stdout.strip().split(",")) == EXPECTED_PUBLIC_NAMES


def test_compiled_communicator_is_frozen_slotted_and_compile_once() -> None:
    backend_plan = EchoBackendPlan()
    compiler = CountingCompiler(_plan(backend_plan))

    communicator = lowbit_comm.compile_communicator(
        _intent(),
        NativePolicy(),
        context=_context(),
        compiler=compiler,
    )

    assert communicator.plan is compiler.plan
    assert not hasattr(communicator, "__dict__")
    with pytest.raises(FrozenInstanceError):
        communicator.plan = compiler.plan
    assert communicator.execute(17).wait() == 17
    assert communicator.execute(23).wait() == 23
    assert compiler.compile_call_count == 1
    assert backend_plan.execute_calls == 2


def test_execute_returns_backend_failure_work_unchanged() -> None:
    failure = ExecutionError("backend failed")
    work = FailedWork[object](failure)
    compiler = CountingCompiler(_plan(ReturningBackendPlan(work)))
    communicator = lowbit_comm.compile_communicator(
        _intent(),
        NativePolicy(),
        context=_context(),
        compiler=compiler,
    )

    returned = communicator.execute(object())

    assert returned is work
    with pytest.raises(ExecutionError) as caught:
        returned.wait()
    assert caught.value is failure
    assert compiler.compile_call_count == 1


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("intent", object(), "intent"),
        ("policy", object(), "policy"),
        ("context", object(), "context"),
    ],
)
def test_compile_boundary_rejects_invalid_input_types_before_compiling(
    argument: str,
    value: object,
    message: str,
) -> None:
    compiler = CountingCompiler(_plan(EchoBackendPlan()))
    arguments = {
        "intent": _intent(),
        "policy": NativePolicy(),
        "context": _context(),
        "compiler": compiler,
    }
    arguments[argument] = value

    with pytest.raises(CompileError, match=message):
        lowbit_comm.compile_communicator(
            cast(CommunicationIntent, arguments["intent"]),
            cast(NativePolicy, arguments["policy"]),
            context=cast(CompilationContext, arguments["context"]),
            compiler=cast(CountingCompiler, arguments["compiler"]),
        )

    assert compiler.compile_call_count == 0


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        (
            "intent",
            _intent(CommunicationIntentSubclass),
            "intent",
        ),
        ("policy", NativePolicySubclass(), "policy"),
        ("policy", AutoPolicySubclass(), "policy"),
        (
            "policy",
            ExplicitPolicySubclass(_strategy()),
            "policy",
        ),
        (
            "context",
            _context(CompilationContextSubclass),
            "context",
        ),
    ],
)
def test_compile_boundary_rejects_contract_subclasses_before_compiling(
    argument: str,
    value: object,
    message: str,
) -> None:
    compiler = CountingCompiler(_plan(EchoBackendPlan()))
    arguments = {
        "intent": _intent(),
        "policy": NativePolicy(),
        "context": _context(),
    }
    arguments[argument] = value

    with pytest.raises(CompileError, match=message):
        lowbit_comm.compile_communicator(
            cast(CommunicationIntent, arguments["intent"]),
            cast(NativePolicy, arguments["policy"]),
            context=cast(CompilationContext, arguments["context"]),
            compiler=compiler,
        )

    assert compiler.compile_call_count == 0


def test_compile_boundary_rejects_non_callable_compiler() -> None:
    with pytest.raises(CompileError, match="compiler"):
        lowbit_comm.compile_communicator(
            _intent(),
            NativePolicy(),
            context=_context(),
            compiler=cast(CountingCompiler, object()),
        )


@pytest.mark.parametrize(
    ("compiled", "message"),
    [
        (object(), "ExecutionPlan"),
        (_corrupted_plan_without_execute(), "backend plan"),
    ],
)
def test_compile_boundary_rejects_invalid_compiler_results(
    compiled: object,
    message: str,
) -> None:
    compiler = CountingCompiler(cast(ExecutionPlan, compiled))

    with pytest.raises(CompileError, match=message):
        lowbit_comm.compile_communicator(
            _intent(),
            NativePolicy(),
            context=_context(),
            compiler=compiler,
        )

    assert compiler.compile_call_count == 1


def test_direct_construction_requires_an_exact_execution_plan() -> None:
    for plan in (
        cast(ExecutionPlan, object()),
        _plan(EchoBackendPlan(), ExecutionPlanSubclass),
    ):
        with pytest.raises(CompileError, match="ExecutionPlan"):
            lowbit_comm.CompiledCommunicator(plan)


def test_compile_boundary_rejects_execution_plan_subclass() -> None:
    compiler = CountingCompiler(
        _plan(EchoBackendPlan(), ExecutionPlanSubclass)
    )

    with pytest.raises(CompileError, match="ExecutionPlan"):
        lowbit_comm.compile_communicator(
            _intent(),
            NativePolicy(),
            context=_context(),
            compiler=compiler,
        )

    assert compiler.compile_call_count == 1
