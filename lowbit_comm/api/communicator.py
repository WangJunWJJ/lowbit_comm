"""Compile-once communication facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from lowbit_comm.api.intent import (
    CommunicationIntent,
    _validate_communication_intent_graph,
)
from lowbit_comm.api.policy import (
    AutoPolicy,
    ExplicitPolicy,
    NativePolicy,
    StrategySpec,
    _auto_constraints_allow,
    _canonical_native_strategy,
    _validate_policy_graph,
)
from lowbit_comm.core.errors import CompileError
from lowbit_comm.core.plan import (
    CompilationContext,
    ExecutionPlan,
    PlanOrigin,
    _resolve_static_callable_member,
    _validate_compilation_context_graph,
    _validate_execution_plan_graph,
)
from lowbit_comm.runtime.work import CommunicationWork

if TYPE_CHECKING:
    from lowbit_comm.compiler.compiler import Compiler


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
    _validate_communication_intent_graph(intent)
    _validate_policy_graph(policy)
    _validate_compilation_context_graph(context)
    return _resolve_static_callable_member(
        compiler,
        "compile",
        "Facade compiler must provide compile().",
    )


def _validate_plan(plan: object) -> None:
    """Validate compiler output before exposing an executable facade."""
    _validate_execution_plan_graph(plan)


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
