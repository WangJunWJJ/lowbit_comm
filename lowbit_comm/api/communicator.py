"""Compile-once communication facade."""

from dataclasses import dataclass
from typing import Callable

from lowbit_comm.api.intent import CommunicationIntent
from lowbit_comm.api.policy import (
    AutoPolicy,
    ExplicitPolicy,
    NativePolicy,
    StrategySpec,
    _auto_constraints_allow,
    _canonical_native_strategy,
)
from lowbit_comm.compiler.compiler import Compiler
from lowbit_comm.core.errors import CompileError
from lowbit_comm.core.plan import (
    CompilationContext,
    ExecutionPlan,
    PlanOrigin,
    _resolve_static_callable_member,
)
from lowbit_comm.runtime.work import CommunicationWork


Policy = NativePolicy | AutoPolicy | ExplicitPolicy


@dataclass(frozen=True, slots=True)
class CompiledCommunicator:
    """Execute one immutable plan selected entirely at compile time."""

    plan: ExecutionPlan

    def __post_init__(self) -> None:
        _validate_plan(self.plan)

    def execute(self, value: object) -> CommunicationWork[object]:
        """Execute the pre-bound backend plan without policy lookup."""
        return self.plan.backend_plan.execute(value)


def compile_communicator(
    intent: CommunicationIntent,
    policy: Policy,
    *,
    context: CompilationContext,
    compiler: Compiler,
) -> CompiledCommunicator:
    """Compile exactly once and bind the resulting execution plan."""
    compile_method = _validate_compile_inputs(
        intent,
        policy,
        context,
        compiler,
    )
    plan = compile_method(intent, policy, context)
    _validate_plan_semantics(plan, intent, policy)
    return CompiledCommunicator(plan)


def _validate_compile_inputs(
    intent: CommunicationIntent,
    policy: Policy,
    context: CompilationContext,
    compiler: object,
) -> Callable[..., object]:
    """Reject malformed values before invoking the compiler boundary."""
    if type(intent) is not CommunicationIntent:
        raise CompileError("Facade intent must be CommunicationIntent.")
    if type(policy) not in (NativePolicy, AutoPolicy, ExplicitPolicy):
        raise CompileError("Facade policy has an unsupported type.")
    if type(context) is not CompilationContext:
        raise CompileError("Facade context must be CompilationContext.")
    return _resolve_static_callable_member(
        compiler,
        "compile",
        "Facade compiler must provide compile().",
    )


def _validate_plan(plan: object) -> None:
    """Validate compiler output before exposing an executable facade."""
    if type(plan) is not ExecutionPlan:
        raise CompileError("Compiler must return an ExecutionPlan.")
    ExecutionPlan.__post_init__(plan)


def _validate_plan_semantics(
    plan: object,
    intent: CommunicationIntent,
    policy: Policy,
) -> None:
    """Reject a structurally valid plan inconsistent with its request."""
    _validate_plan(plan)
    if type(plan.intent) is not CommunicationIntent or plan.intent != intent:
        raise CompileError("ExecutionPlan intent does not match the request.")
    if type(plan.strategy) is not StrategySpec:
        raise CompileError("ExecutionPlan strategy must be StrategySpec.")
    if type(plan.origin) is not PlanOrigin:
        raise CompileError("ExecutionPlan origin must be PlanOrigin.")
    if plan.origin is PlanOrigin.AUTO:
        if (
            type(plan.evidence_fingerprint) is not str
            or not plan.evidence_fingerprint
        ):
            raise CompileError(
                "Auto ExecutionPlan requires an evidence fingerprint."
            )
    elif plan.evidence_fingerprint is not None:
        raise CompileError(
            "Non-Auto ExecutionPlan cannot carry evidence."
        )

    if type(policy) is NativePolicy:
        if (
            plan.origin is not PlanOrigin.NATIVE
            or plan.strategy != _canonical_native_strategy()
        ):
            raise CompileError("Native policy requires a native plan.")
        return
    if type(policy) is ExplicitPolicy:
        if (
            plan.origin is not PlanOrigin.EXPLICIT
            or plan.strategy != policy.strategy
        ):
            raise CompileError("Explicit policy requires its exact strategy.")
        return
    if plan.origin is PlanOrigin.AUTO:
        if not _auto_constraints_allow(policy.constraints, plan.strategy):
            raise CompileError("Auto plan violates policy constraints.")
        return
    if (
        plan.origin is not PlanOrigin.NATIVE_FALLBACK
        or plan.strategy != _canonical_native_strategy()
    ):
        raise CompileError("Auto fallback requires a native plan.")
