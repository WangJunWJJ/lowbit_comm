"""Contracts for the narrow, import-safe public communication facade."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, dataclass
import inspect
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
from typing import Callable, cast

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
    AutoConstraints,
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
from lowbit_comm.core.errors import (
    CapabilityError,
    CompileError,
    ExecutionError,
    LowbitCommError,
)
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
    "CapabilityError",
    "CollectiveKind",
    "CommunicationIntent",
    "CommunicationWork",
    "CompilationContext",
    "CompileError",
    "CompiledCommunicator",
    "CompletionMode",
    "CompressionKind",
    "ExplicitPolicy",
    "ExecutionError",
    "FullTensorResult",
    "LowbitCommError",
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


class StringSubclass(str):
    pass


class IntSubclass(int):
    pass


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


class ExplodingCompileProperty:
    """Compiler-shaped object whose descriptor must not be evaluated."""

    compile_property_accesses = 0

    @property
    def compile(self) -> object:
        type(self).compile_property_accesses += 1
        raise AssertionError("compile property evaluated")


@dataclass
class GuardedCompilerLookup:
    """Compiler whose method must be invoked without dynamic lookup."""

    plan: ExecutionPlan
    compile_call_count: int = 0
    compile_attribute_accesses: int = 0

    def __getattribute__(self, name: str) -> object:
        if name == "compile":
            accesses = object.__getattribute__(
                self,
                "compile_attribute_accesses",
            )
            object.__setattr__(
                self,
                "compile_attribute_accesses",
                accesses + 1,
            )
            raise AssertionError("compile attribute accessed dynamically")
        return object.__getattribute__(self, name)

    def compile(
        self,
        intent: CommunicationIntent,
        policy: NativePolicy,
        context: CompilationContext,
    ) -> ExecutionPlan:
        del intent, policy, context
        self.compile_call_count += 1
        return self.plan


class CallableCompiler:
    """Callable object suitable for instance and slot compiler fields."""

    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan
        self.compile_call_count = 0

    def __call__(
        self,
        intent: CommunicationIntent,
        policy: NativePolicy,
        context: CompilationContext,
    ) -> ExecutionPlan:
        del intent, policy, context
        self.compile_call_count += 1
        return self.plan


class StaticMethodCompiler:
    """Structural compiler exposing an exact staticmethod descriptor."""

    plan: ExecutionPlan
    compile_call_count = 0

    @staticmethod
    def compile(
        intent: CommunicationIntent,
        policy: NativePolicy,
        context: CompilationContext,
    ) -> ExecutionPlan:
        del intent, policy, context
        StaticMethodCompiler.compile_call_count += 1
        return StaticMethodCompiler.plan


class ClassMethodCompiler:
    """Structural compiler exposing an exact classmethod descriptor."""

    plan: ExecutionPlan
    compile_call_count = 0

    @classmethod
    def compile(
        cls,
        intent: CommunicationIntent,
        policy: NativePolicy,
        context: CompilationContext,
    ) -> ExecutionPlan:
        del intent, policy, context
        cls.compile_call_count += 1
        return cls.plan


class InstanceCallableCompiler:
    """Structural compiler storing its callable in the instance dict."""

    def __init__(self, compile_method: CallableCompiler) -> None:
        self.compile = compile_method


class SlottedCallableCompiler:
    """Structural compiler storing its callable in an instance slot."""

    __slots__ = ("compile",)

    def __init__(self, compile_method: CallableCompiler) -> None:
        self.compile = compile_method


class NonCallableCompiler:
    """Compiler-shaped object with a non-callable member."""

    compile = object()


class UninitializedSlottedCompiler:
    """Compiler-shaped object with an uninitialized callable slot."""

    __slots__ = ("compile",)


class DynamicCompileCompiler:
    """Compiler-shaped object that only pretends to expose compile."""

    def __init__(self) -> None:
        self.getattr_calls = 0

    def __getattr__(self, name: str) -> object:
        self.getattr_calls += 1
        if name == "compile":
            return CallableCompiler(_plan(EchoBackendPlan()))
        raise AttributeError(name)


class MaliciousStaticMethod(staticmethod):
    """Staticmethod subclass whose internals must never be accessed."""

    func_accesses = 0

    def __getattribute__(self, name: str) -> object:
        if name == "__func__":
            type(self).func_accesses += 1
            raise ValueError("malicious staticmethod accessed")
        return staticmethod.__getattribute__(self, name)


class MaliciousClassMethod(classmethod):
    """Classmethod subclass whose internals must never be accessed."""

    func_accesses = 0

    def __getattribute__(self, name: str) -> object:
        if name == "__func__":
            type(self).func_accesses += 1
            raise ValueError("malicious classmethod accessed")
        return classmethod.__getattribute__(self, name)


def _unreachable_compile(
    intent: CommunicationIntent,
    policy: NativePolicy,
    context: CompilationContext,
) -> ExecutionPlan:
    del intent, policy, context
    raise AssertionError("malicious descriptor invoked")


class MaliciousStaticCompiler:
    compile = MaliciousStaticMethod(_unreachable_compile)


class MaliciousClassCompiler:
    compile = MaliciousClassMethod(_unreachable_compile)


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
    *,
    dtype: str = "float32",
) -> CommunicationIntent:
    return intent_type(
        tensor=TensorSpec(dtype=dtype, shape=(1,)),
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
        environment=EnvironmentFingerprint.from_mapping({"hardware": "contract-test"}),
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


def _compressed_strategy() -> StrategySpec:
    return StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.RING,
        group_size=32,
    )


def _plan(
    backend_plan: object,
    plan_type: type[ExecutionPlan] = ExecutionPlan,
    *,
    intent: CommunicationIntent | None = None,
    strategy: StrategySpec | None = None,
    origin: PlanOrigin = PlanOrigin.NATIVE,
    evidence_fingerprint: str | None = None,
) -> ExecutionPlan:
    return plan_type(
        intent=_intent() if intent is None else intent,
        strategy=_strategy() if strategy is None else strategy,
        backend_id="contract-test",
        backend_plan=cast(BackendPlan, backend_plan),
        origin=origin,
        signature="contract-test-signature",
        evidence_fingerprint=evidence_fingerprint,
    )


def _corrupted_plan_without_execute() -> ExecutionPlan:
    """Bypass frozen state to exercise the facade's defensive check."""
    plan = _plan(EchoBackendPlan())
    object.__setattr__(plan, "backend_plan", object())
    return plan


def _corrupted_plan_field(field_name: str) -> ExecutionPlan:
    """Forge one invalid plan field after successful construction."""
    plan = _plan(EchoBackendPlan())
    object.__setattr__(plan, field_name, "")
    return plan


def _normal_method_compiler(
    plan: ExecutionPlan,
) -> tuple[object, Callable[[], int]]:
    compiler = CountingCompiler(plan)
    return compiler, lambda: compiler.compile_call_count


def _static_method_compiler(
    plan: ExecutionPlan,
) -> tuple[object, Callable[[], int]]:
    StaticMethodCompiler.plan = plan
    StaticMethodCompiler.compile_call_count = 0
    return (
        StaticMethodCompiler(),
        lambda: StaticMethodCompiler.compile_call_count,
    )


def _class_method_compiler(
    plan: ExecutionPlan,
) -> tuple[object, Callable[[], int]]:
    ClassMethodCompiler.plan = plan
    ClassMethodCompiler.compile_call_count = 0
    return (
        ClassMethodCompiler(),
        lambda: ClassMethodCompiler.compile_call_count,
    )


def _instance_callable_compiler(
    plan: ExecutionPlan,
) -> tuple[object, Callable[[], int]]:
    compile_method = CallableCompiler(plan)
    return (
        InstanceCallableCompiler(compile_method),
        lambda: compile_method.compile_call_count,
    )


def _slotted_callable_compiler(
    plan: ExecutionPlan,
) -> tuple[object, Callable[[], int]]:
    compile_method = CallableCompiler(plan)
    return (
        SlottedCallableCompiler(compile_method),
        lambda: compile_method.compile_call_count,
    )


def _property_compiler() -> tuple[object, Callable[[], int]]:
    ExplodingCompileProperty.compile_property_accesses = 0
    return (
        ExplodingCompileProperty(),
        lambda: ExplodingCompileProperty.compile_property_accesses,
    )


def _dynamic_compiler() -> tuple[object, Callable[[], int]]:
    compiler = DynamicCompileCompiler()
    return compiler, lambda: compiler.getattr_calls


def _malicious_static_compiler() -> tuple[object, Callable[[], int]]:
    MaliciousStaticMethod.func_accesses = 0
    return (
        MaliciousStaticCompiler(),
        lambda: MaliciousStaticMethod.func_accesses,
    )


def _malicious_class_compiler() -> tuple[object, Callable[[], int]]:
    MaliciousClassMethod.func_accesses = 0
    return (
        MaliciousClassCompiler(),
        lambda: MaliciousClassMethod.func_accesses,
    )


def test_public_api_is_exactly_the_stable_semantic_surface() -> None:
    assert set(lowbit_comm.__all__) == EXPECTED_PUBLIC_NAMES
    assert set(lowbit_comm.api.__all__) == EXPECTED_PUBLIC_NAMES
    assert len(lowbit_comm.__all__) == len(set(lowbit_comm.__all__))


def test_public_errors_preserve_core_identity() -> None:
    core_errors = {
        "LowbitCommError": LowbitCommError,
        "CompileError": CompileError,
        "CapabilityError": CapabilityError,
        "ExecutionError": ExecutionError,
    }

    for name, core_error in core_errors.items():
        assert getattr(lowbit_comm, name) is core_error
        assert getattr(lowbit_comm.api, name) is core_error


def test_public_error_hierarchy_is_stable() -> None:
    assert LowbitCommError.__bases__ == (Exception,)
    for error_type in (CompileError, CapabilityError, ExecutionError):
        assert error_type.__bases__ == (LowbitCommError,)


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


@pytest.mark.parametrize(
    "module_name",
    ["lowbit_comm.api.communicator", "lowbit_comm"],
)
def test_public_facade_import_does_not_load_compiler_implementations(
    module_name: str,
) -> None:
    script = """
import importlib
import importlib.abc
import sys

FORBIDDEN = frozenset(
    {
        "lowbit_comm.compiler.compiler",
        "lowbit_comm.compiler.evidence",
        "lowbit_comm.compiler.registry",
    }
)

class ForbiddenImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in FORBIDDEN:
            raise AssertionError(f"forbidden import: {fullname}")
        return None

sys.meta_path.insert(0, ForbiddenImport())
sys.path.insert(0, sys.argv[1])
importlib.import_module(sys.argv[2])
assert FORBIDDEN.isdisjoint(sys.modules)
import lowbit_comm
assert FORBIDDEN.isdisjoint(sys.modules)
print(",".join(sorted(lowbit_comm.__all__)))
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT), module_name],
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


def test_execute_source_is_only_direct_backend_delegation() -> None:
    source = dedent(inspect.getsource(lowbit_comm.CompiledCommunicator.execute))
    function = cast(ast.FunctionDef, ast.parse(source).body[0])

    assert len(function.body) == 2
    assert isinstance(function.body[0], ast.Expr)
    assert isinstance(function.body[1], ast.Return)
    assert not any(
        isinstance(node, (ast.If, ast.IfExp, ast.Match)) for node in ast.walk(function)
    )
    assert lowbit_comm.CompiledCommunicator.execute.__code__.co_names == (
        "plan",
        "backend_plan",
        "execute",
    )


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


@pytest.mark.parametrize(
    "scenario",
    [
        "intent-shape",
        "intent-dtype-subclass",
        "explicit-strategy",
        "auto-constraints",
        "environment",
        "context-node-count",
    ],
)
def test_compile_boundary_revalidates_nested_caller_graph_before_compiling(
    scenario: str,
) -> None:
    intent = _intent()
    policy: NativePolicy | AutoPolicy | ExplicitPolicy = NativePolicy()
    context = _context()
    if scenario == "intent-shape":
        object.__setattr__(intent.tensor, "shape", ())
    elif scenario == "intent-dtype-subclass":
        object.__setattr__(
            intent.tensor,
            "dtype",
            StringSubclass("float32"),
        )
    elif scenario == "explicit-strategy":
        policy = ExplicitPolicy(_compressed_strategy())
        object.__setattr__(policy.strategy, "group_size", -1)
    elif scenario == "auto-constraints":
        policy = AutoPolicy(AutoConstraints())
        object.__setattr__(policy.constraints, "denied_topologies", set())
    elif scenario == "environment":
        object.__setattr__(context.environment, "dimensions", "invalid")
    else:
        object.__setattr__(context, "node_count", 0)
    compiler = CountingCompiler(_plan(EchoBackendPlan()))

    with pytest.raises(CompileError):
        lowbit_comm.compile_communicator(
            intent,
            policy,
            context=context,
            compiler=compiler,
        )

    assert compiler.compile_call_count == 0


def test_compile_boundary_rejects_non_callable_compiler() -> None:
    with pytest.raises(lowbit_comm.CompileError, match="compiler"):
        lowbit_comm.compile_communicator(
            _intent(),
            NativePolicy(),
            context=_context(),
            compiler=cast(CountingCompiler, object()),
        )


def test_compile_boundary_rejects_compile_property_without_accessing_it() -> None:
    compiler = ExplodingCompileProperty()
    ExplodingCompileProperty.compile_property_accesses = 0

    with pytest.raises(CompileError, match="compiler"):
        lowbit_comm.compile_communicator(
            _intent(),
            NativePolicy(),
            context=_context(),
            compiler=cast(CountingCompiler, compiler),
        )

    assert compiler.compile_property_accesses == 0


def test_compile_boundary_invokes_method_without_dynamic_attribute_lookup() -> None:
    compiler = GuardedCompilerLookup(_plan(EchoBackendPlan()))

    communicator = lowbit_comm.compile_communicator(
        _intent(),
        NativePolicy(),
        context=_context(),
        compiler=cast(CountingCompiler, compiler),
    )

    assert communicator.plan is compiler.plan
    assert compiler.compile_attribute_accesses == 0
    assert compiler.compile_call_count == 1


@pytest.mark.parametrize(
    "compiler_factory",
    [
        _normal_method_compiler,
        _static_method_compiler,
        _class_method_compiler,
        _instance_callable_compiler,
        _slotted_callable_compiler,
    ],
    ids=[
        "normal-instance-method",
        "exact-staticmethod",
        "exact-classmethod",
        "instance-dict-callable",
        "slot-callable",
    ],
)
def test_compile_boundary_accepts_safe_structural_compiler_callables(
    compiler_factory: Callable[
        [ExecutionPlan],
        tuple[object, Callable[[], int]],
    ],
) -> None:
    plan = _plan(EchoBackendPlan())
    compiler, compile_calls = compiler_factory(plan)

    communicator = lowbit_comm.compile_communicator(
        _intent(),
        NativePolicy(),
        context=_context(),
        compiler=cast(CountingCompiler, compiler),
    )

    assert communicator.plan is plan
    assert compile_calls() == 1


@pytest.mark.parametrize(
    "compiler_factory",
    [
        lambda: (NonCallableCompiler(), lambda: 0),
        lambda: (UninitializedSlottedCompiler(), lambda: 0),
        _property_compiler,
        _dynamic_compiler,
        _malicious_static_compiler,
        _malicious_class_compiler,
    ],
    ids=[
        "noncallable",
        "uninitialized-slot",
        "property",
        "dynamic",
        "malicious-staticmethod-subclass",
        "malicious-classmethod-subclass",
    ],
)
def test_compile_boundary_rejects_unsafe_structural_compiler_callables(
    compiler_factory: Callable[[], tuple[object, Callable[[], int]]],
) -> None:
    compiler, unsafe_accesses = compiler_factory()

    with pytest.raises(CompileError, match="compiler"):
        lowbit_comm.compile_communicator(
            _intent(),
            NativePolicy(),
            context=_context(),
            compiler=cast(CountingCompiler, compiler),
        )

    assert unsafe_accesses() == 0


@pytest.mark.parametrize(
    ("request_intent", "policy", "compiled"),
    [
        (
            _intent(),
            NativePolicy(),
            _plan(EchoBackendPlan(), intent=_intent(dtype="float16")),
        ),
        (
            _intent(),
            NativePolicy(),
            _plan(EchoBackendPlan(), origin=PlanOrigin.EXPLICIT),
        ),
        (
            _intent(),
            NativePolicy(),
            _plan(EchoBackendPlan(), strategy=_compressed_strategy()),
        ),
        (
            _intent(),
            NativePolicy(),
            _plan(EchoBackendPlan(), evidence_fingerprint="unexpected"),
        ),
        (
            _intent(),
            ExplicitPolicy(_compressed_strategy()),
            _plan(
                EchoBackendPlan(),
                strategy=_compressed_strategy(),
                origin=PlanOrigin.NATIVE,
            ),
        ),
        (
            _intent(),
            ExplicitPolicy(_compressed_strategy()),
            _plan(EchoBackendPlan(), origin=PlanOrigin.EXPLICIT),
        ),
        (
            _intent(),
            ExplicitPolicy(_compressed_strategy()),
            _plan(
                EchoBackendPlan(),
                strategy=_compressed_strategy(),
                origin=PlanOrigin.EXPLICIT,
                evidence_fingerprint="unexpected",
            ),
        ),
        (
            _intent(),
            AutoPolicy(),
            _plan(EchoBackendPlan()),
        ),
        (
            _intent(),
            AutoPolicy(
                AutoConstraints(denied_compressions=frozenset({CompressionKind.INT8}))
            ),
            _plan(
                EchoBackendPlan(),
                strategy=_compressed_strategy(),
                origin=PlanOrigin.AUTO,
                evidence_fingerprint="evidence",
            ),
        ),
        (
            _intent(),
            AutoPolicy(),
            _plan(
                EchoBackendPlan(),
                strategy=_compressed_strategy(),
                origin=PlanOrigin.NATIVE_FALLBACK,
            ),
        ),
        (
            _intent(),
            AutoPolicy(),
            _plan(
                EchoBackendPlan(),
                strategy=_compressed_strategy(),
                origin=PlanOrigin.AUTO,
            ),
        ),
        (
            _intent(),
            AutoPolicy(),
            _plan(
                EchoBackendPlan(),
                origin=PlanOrigin.NATIVE_FALLBACK,
                evidence_fingerprint="unexpected",
            ),
        ),
    ],
    ids=[
        "wrong-intent",
        "native-wrong-origin",
        "native-wrong-strategy",
        "native-wrong-evidence",
        "explicit-wrong-origin",
        "explicit-wrong-strategy",
        "explicit-wrong-evidence",
        "auto-wrong-origin",
        "auto-constraint-denied",
        "auto-fallback-nonnative",
        "auto-missing-evidence",
        "auto-fallback-evidence",
    ],
)
def test_compile_boundary_rejects_semantically_invalid_execution_plans(
    request_intent: CommunicationIntent,
    policy: NativePolicy | AutoPolicy | ExplicitPolicy,
    compiled: ExecutionPlan,
) -> None:
    compiler = CountingCompiler(compiled)

    with pytest.raises(CompileError):
        lowbit_comm.compile_communicator(
            request_intent,
            policy,
            context=_context(),
            compiler=compiler,
        )

    assert compiler.compile_call_count == 1
    assert cast(EchoBackendPlan, compiled.backend_plan).execute_calls == 0


@pytest.mark.parametrize(
    ("policy", "compiled"),
    [
        (NativePolicy(), _plan(EchoBackendPlan())),
        (
            ExplicitPolicy(_compressed_strategy()),
            _plan(
                EchoBackendPlan(),
                strategy=_compressed_strategy(),
                origin=PlanOrigin.EXPLICIT,
            ),
        ),
        (
            AutoPolicy(),
            _plan(
                EchoBackendPlan(),
                strategy=_compressed_strategy(),
                origin=PlanOrigin.AUTO,
                evidence_fingerprint="evidence",
            ),
        ),
        (
            AutoPolicy(),
            _plan(EchoBackendPlan(), origin=PlanOrigin.NATIVE_FALLBACK),
        ),
    ],
    ids=["native", "explicit", "auto", "auto-fallback"],
)
def test_compile_boundary_accepts_semantically_valid_execution_plans(
    policy: NativePolicy | AutoPolicy | ExplicitPolicy,
    compiled: ExecutionPlan,
) -> None:
    compiler = CountingCompiler(compiled)

    communicator = lowbit_comm.compile_communicator(
        _intent(),
        policy,
        context=_context(),
        compiler=compiler,
    )

    assert communicator.plan is compiled
    assert compiler.compile_call_count == 1


@pytest.mark.parametrize(
    ("compiled", "message"),
    [
        (object(), "ExecutionPlan"),
        (_corrupted_plan_without_execute(), "backend plan"),
        (_corrupted_plan_field("backend_id"), "identifier"),
        (_corrupted_plan_field("signature"), "signature"),
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
    if (
        type(compiled) is ExecutionPlan
        and type(compiled.backend_plan) is EchoBackendPlan
    ):
        assert compiled.backend_plan.execute_calls == 0


@pytest.mark.parametrize(
    "scenario",
    ["intent-empty-shape", "intent-dtype-subclass", "strategy-int-subclass"],
)
def test_compiled_communicator_revalidates_nested_plan_graph(
    scenario: str,
) -> None:
    plan = _plan(EchoBackendPlan())
    if scenario == "intent-empty-shape":
        object.__setattr__(plan.intent.tensor, "shape", ())
    elif scenario == "intent-dtype-subclass":
        object.__setattr__(
            plan.intent.tensor,
            "dtype",
            StringSubclass("float32"),
        )
    else:
        strategy = _compressed_strategy()
        plan = _plan(
            EchoBackendPlan(),
            strategy=strategy,
            origin=PlanOrigin.EXPLICIT,
        )
        object.__setattr__(
            plan.strategy,
            "group_size",
            IntSubclass(32),
        )

    with pytest.raises(CompileError):
        lowbit_comm.CompiledCommunicator(plan)

    assert cast(EchoBackendPlan, plan.backend_plan).execute_calls == 0


@pytest.mark.parametrize(
    "scenario",
    ["intent-dtype-subclass", "strategy-int-subclass"],
)
def test_compile_boundary_rejects_value_equal_forged_plan_graph(
    scenario: str,
) -> None:
    request_intent = _intent()
    if scenario == "intent-dtype-subclass":
        plan = _plan(EchoBackendPlan(), intent=_intent())
        policy: NativePolicy | ExplicitPolicy = NativePolicy()
        object.__setattr__(
            plan.intent.tensor,
            "dtype",
            StringSubclass("float32"),
        )
    else:
        request_strategy = _compressed_strategy()
        plan_strategy = _compressed_strategy()
        plan = _plan(
            EchoBackendPlan(),
            strategy=plan_strategy,
            origin=PlanOrigin.EXPLICIT,
        )
        policy = ExplicitPolicy(request_strategy)
        object.__setattr__(
            plan.strategy,
            "group_size",
            IntSubclass(32),
        )
    compiler = CountingCompiler(plan)

    with pytest.raises(CompileError):
        lowbit_comm.compile_communicator(
            request_intent,
            policy,
            context=_context(),
            compiler=compiler,
        )

    assert compiler.compile_call_count == 1
    assert cast(EchoBackendPlan, plan.backend_plan).execute_calls == 0


def test_direct_construction_requires_an_exact_execution_plan() -> None:
    for plan in (
        cast(ExecutionPlan, object()),
        _plan(EchoBackendPlan(), ExecutionPlanSubclass),
    ):
        with pytest.raises(CompileError, match="ExecutionPlan"):
            lowbit_comm.CompiledCommunicator(plan)


def test_compile_boundary_rejects_execution_plan_subclass() -> None:
    compiler = CountingCompiler(_plan(EchoBackendPlan(), ExecutionPlanSubclass))

    with pytest.raises(CompileError, match="ExecutionPlan"):
        lowbit_comm.compile_communicator(
            _intent(),
            NativePolicy(),
            context=_context(),
            compiler=compiler,
        )

    assert compiler.compile_call_count == 1
