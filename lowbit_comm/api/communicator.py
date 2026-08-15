"""Compile-once communication facade."""

from dataclasses import dataclass

from lowbit_comm.api.intent import CommunicationIntent
from lowbit_comm.api.policy import AutoPolicy, ExplicitPolicy, NativePolicy
from lowbit_comm.compiler.compiler import Compiler
from lowbit_comm.core.errors import CompileError
from lowbit_comm.core.plan import CompilationContext, ExecutionPlan
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
    _validate_compile_inputs(intent, policy, context, compiler)
    plan = compiler.compile(intent, policy, context)
    return CompiledCommunicator(plan)


def _validate_compile_inputs(
    intent: CommunicationIntent,
    policy: Policy,
    context: CompilationContext,
    compiler: object,
) -> None:
    """Reject malformed values before invoking the compiler boundary."""
    if type(intent) is not CommunicationIntent:
        raise CompileError("Facade intent must be CommunicationIntent.")
    if type(policy) not in (NativePolicy, AutoPolicy, ExplicitPolicy):
        raise CompileError("Facade policy has an unsupported type.")
    if type(context) is not CompilationContext:
        raise CompileError("Facade context must be CompilationContext.")
    if not callable(getattr(compiler, "compile", None)):
        raise CompileError("Facade compiler must provide compile().")


def _validate_plan(plan: object) -> None:
    """Validate compiler output before exposing an executable facade."""
    if type(plan) is not ExecutionPlan:
        raise CompileError("Compiler must return an ExecutionPlan.")
    if not callable(getattr(plan.backend_plan, "execute", None)):
        raise CompileError(
            "ExecutionPlan backend plan must provide execute()."
        )
