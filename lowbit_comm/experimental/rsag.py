"""Installable state and fail-closed routing for experimental RSAG/qWD."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from math import isfinite
import os
import socket

from lowbit_comm.core.errors import CapabilityError
from lowbit_comm.experimental.compatibility import (
    RSAG_CUDA_EXTENSION_ABI,
    RSAG_LOWBIT_COMM_VERSION,
    RSAGRuntimeABI,
    _nccl_version_string,
    compute_rsag_build_fingerprint,
    is_verified_rsag_runtime,
)


_MAX_SIGNED_64 = (1 << 63) - 1
RSAG_EVIDENCE_SCHEMA_VERSION = 2
RSAG_CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class RSAGEnvironment:
    """Exact runtime identity used to qualify one RSAG/qWD route."""

    world_size: int
    node_count: int
    logical_bytes: int
    topology_class: str
    transport: str
    gpu_model: str
    torch_version: str
    cuda_version: str
    nccl_version: str
    lowbit_comm_version: str
    cuda_extension_abi: int
    checkpoint_schema_version: int
    build_fingerprint: str

    def __post_init__(self) -> None:
        _require_positive_int(self.world_size, "world_size")
        _require_positive_int(self.node_count, "node_count")
        if self.node_count > self.world_size:
            raise ValueError("node_count cannot exceed world_size")
        _require_nonnegative_int(self.logical_bytes, "logical_bytes")
        _require_nonnegative_int(
            self.cuda_extension_abi,
            "cuda_extension_abi",
        )
        _require_nonnegative_int(
            self.checkpoint_schema_version,
            "checkpoint_schema_version",
        )
        for field_name in _ENVIRONMENT_STRING_FIELDS:
            _require_exact_string(getattr(self, field_name), field_name)
        _require_sha256(self.build_fingerprint, "build_fingerprint")


@dataclass(frozen=True, slots=True)
class RSAGEvidence:
    """One topology-scoped, multi-seed RSAG/qWD qualification record."""

    schema_version: int
    world_size: int
    node_count: int
    min_logical_bytes: int
    max_logical_bytes: int
    topology_class: str
    transport: str
    gpu_model: str
    torch_version: str
    cuda_version: str
    nccl_version: str
    lowbit_comm_version: str
    cuda_extension_abi: int
    checkpoint_schema_version: int
    build_fingerprint: str
    seed_speedups_percent: tuple[float, ...]
    quality_passed: bool

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.schema_version, "schema_version")
        _require_positive_int(self.world_size, "world_size")
        _require_positive_int(self.node_count, "node_count")
        if self.node_count > self.world_size:
            raise ValueError("node_count cannot exceed world_size")
        _require_nonnegative_int(
            self.min_logical_bytes,
            "min_logical_bytes",
        )
        _require_nonnegative_int(
            self.max_logical_bytes,
            "max_logical_bytes",
        )
        if self.min_logical_bytes != self.max_logical_bytes:
            raise ValueError("evidence must bind one exact logical_bytes value")
        _require_nonnegative_int(
            self.cuda_extension_abi,
            "cuda_extension_abi",
        )
        _require_nonnegative_int(
            self.checkpoint_schema_version,
            "checkpoint_schema_version",
        )
        for field_name in _ENVIRONMENT_STRING_FIELDS:
            _require_exact_string(getattr(self, field_name), field_name)
        _require_sha256(self.build_fingerprint, "build_fingerprint")
        if (
            type(self.seed_speedups_percent) is not tuple
            or not self.seed_speedups_percent
        ):
            raise ValueError("seed_speedups_percent must be a non-empty tuple")
        if any(
            type(value) is not float or not isfinite(value)
            for value in self.seed_speedups_percent
        ):
            raise ValueError(
                "seed_speedups_percent must contain finite exact floats"
            )
        if type(self.quality_passed) is not bool:
            raise ValueError("quality_passed must be an exact bool")

    def matches(self, environment: RSAGEnvironment) -> bool:
        """Return whether every evidence-key field matches the runtime."""
        return (
            self.world_size == environment.world_size
            and self.node_count == environment.node_count
            and self.cuda_extension_abi == environment.cuda_extension_abi
            and self.checkpoint_schema_version
            == environment.checkpoint_schema_version
            and self.min_logical_bytes == environment.logical_bytes
            and self.max_logical_bytes == environment.logical_bytes
            and all(
                getattr(self, field_name) == getattr(environment, field_name)
                for field_name in _ENVIRONMENT_STRING_FIELDS
            )
        )


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Auditable route decision with no implicit compression fallback."""

    route: str
    reason: str
    evidence_schema_version: int | None

    def __post_init__(self) -> None:
        if self.route not in {"native", "rsag_qwd"}:
            raise ValueError("route is invalid")
        _require_exact_string(self.reason, "reason")
        if self.evidence_schema_version is not None:
            _require_nonnegative_int(
                self.evidence_schema_version,
                "evidence_schema_version",
            )

    @property
    def uses_rsag(self) -> bool:
        """Return whether the qualified low-bit route is selected."""
        return self.route == "rsag_qwd"


_ENVIRONMENT_STRING_FIELDS = (
    "topology_class",
    "transport",
    "gpu_model",
    "torch_version",
    "cuda_version",
    "nccl_version",
    "lowbit_comm_version",
    "build_fingerprint",
)
_UNKNOWN_IDENTITIES = frozenset(
    {"unknown", "unavailable", "n/a", "none", "not_available"}
)
_ATTESTED_TOPOLOGY_ENV = "LOWBIT_COMM_RSAG_ATTESTED_TOPOLOGY"
_ATTESTED_TRANSPORT_ENV = "LOWBIT_COMM_RSAG_ATTESTED_TRANSPORT"
_RANK_IDENTITY_FIELDS = frozenset(
    {
        "hostname",
        "logical_bytes",
        "topology_class",
        "transport",
        "gpu_model",
        "torch_version",
        "cuda_version",
        "nccl_version",
        "lowbit_comm_version",
        "cuda_extension_abi",
        "checkpoint_schema_version",
        "build_fingerprint",
        "error",
    }
)
_RANK_CONSENSUS_FIELDS = tuple(
    sorted(_RANK_IDENTITY_FIELDS - {"hostname", "error"})
)


@dataclass(frozen=True, slots=True)
class ShardLayout:
    """One contiguous padded shard and its valid global ownership."""

    global_numel: int
    world_size: int
    rank: int
    start: int
    valid_numel: int
    padded_numel: int

    def __post_init__(self) -> None:
        _validate_shard_layout(self, "ShardLayout")

    @classmethod
    def build(
        cls,
        global_numel: int,
        world_size: int,
        rank: int,
    ) -> ShardLayout:
        """Create a validated, contiguous ownership layout."""
        _require_nonnegative_int(global_numel, "global_numel")
        _require_positive_int(world_size, "world_size")
        if type(rank) is not int or rank < 0 or rank >= world_size:
            raise ValueError("rank must be an exact integer within world_size")

        padded_numel = _checked_ceil_div(global_numel, world_size)
        start = min(_checked_mul(rank, padded_numel), global_numel)
        valid_numel = min(padded_numel, global_numel - start)
        return cls(
            global_numel=global_numel,
            world_size=world_size,
            rank=rank,
            start=start,
            valid_numel=valid_numel,
            padded_numel=padded_numel,
        )


@dataclass(frozen=True, slots=True)
class QWDSchedule:
    """Deterministic full-precision refresh cadence for qWD steps."""

    refresh_interval: int

    def __post_init__(self) -> None:
        _require_positive_int(self.refresh_interval, "refresh_interval")
        if self.refresh_interval != 100:
            raise ValueError("refresh_interval must be exactly 100")

    def mode(self, step: int, force_refresh: bool = False) -> str:
        """Return the communication route required for one optimizer step."""
        _require_nonnegative_int(step, "step")
        if type(force_refresh) is not bool:
            raise ValueError("force_refresh must be an exact bool")
        if force_refresh or step % self.refresh_interval == 0:
            return "fp_refresh"
        return "qwd"


def select_rsag_route(
    environment: RSAGEnvironment,
    evidence: tuple[RSAGEvidence, ...],
    *,
    requested: str = "auto",
) -> RouteDecision:
    """Select RSAG/qWD only for one exact, positive evidence record."""
    if type(environment) is not RSAGEnvironment:
        raise ValueError("environment must be an exact RSAGEnvironment")
    environment.__post_init__()
    if type(evidence) is not tuple or any(
        type(record) is not RSAGEvidence for record in evidence
    ):
        raise ValueError("evidence must be an exact tuple of RSAGEvidence")
    for record in evidence:
        record.__post_init__()
    if type(requested) is not str or requested not in {
        "auto",
        "native",
        "rsag_qwd",
    }:
        raise ValueError("requested route is invalid")
    if requested == "native":
        return RouteDecision("native", "requested_native", None)

    decision = _select_automatic_route(environment, evidence)
    if requested == "rsag_qwd" and not decision.uses_rsag:
        raise CapabilityError(
            f"RSAG/qWD route is not eligible: {decision.reason}."
        )
    return decision


def detect_rsag_environment(
    process_group: object,
    *,
    logical_bytes: int,
    topology_class: str,
    transport: str,
) -> RSAGEnvironment:
    """Gather the live binary, GPU, rank, and node evidence identity."""
    _require_nonnegative_int(logical_bytes, "logical_bytes")
    _require_exact_string(topology_class, "topology_class")
    _require_exact_string(transport, "transport")
    torch = _torch()
    distributed = torch.distributed
    if not distributed.is_initialized():
        raise CapabilityError(
            "RSAG/qWD environment detection requires distributed init."
        )
    world_size = distributed.get_world_size(process_group)
    _require_positive_int(world_size, "world_size")
    local_identity = _local_rank_identity(torch, logical_bytes)
    rank_identities: list[object | None] = [None for _ in range(world_size)]
    distributed.all_gather_object(
        rank_identities,
        local_identity,
        group=process_group,
    )
    identities = _validated_rank_identities(rank_identities)
    baseline = identities[0]
    if any(
        identity[field_name] != baseline[field_name]
        for identity in identities[1:]
        for field_name in _RANK_CONSENSUS_FIELDS
    ):
        raise CapabilityError(
            "RSAG/qWD rank identity is not globally consistent."
        )
    if (
        baseline["topology_class"] != topology_class
        or baseline["transport"] != transport
    ):
        raise CapabilityError(
            "RSAG/qWD launcher attestation differs from the requested "
            "environment."
        )
    hostnames = tuple(identity["hostname"] for identity in identities)
    node_count = len(set(hostnames))
    _validate_attested_node_count(
        baseline["topology_class"],
        node_count,
    )
    return RSAGEnvironment(
        world_size=world_size,
        node_count=node_count,
        logical_bytes=logical_bytes,
        topology_class=baseline["topology_class"],
        transport=baseline["transport"],
        gpu_model=baseline["gpu_model"],
        torch_version=baseline["torch_version"],
        cuda_version=baseline["cuda_version"],
        nccl_version=baseline["nccl_version"],
        lowbit_comm_version=baseline["lowbit_comm_version"],
        cuda_extension_abi=baseline["cuda_extension_abi"],
        checkpoint_schema_version=baseline["checkpoint_schema_version"],
        build_fingerprint=baseline["build_fingerprint"],
    )


def _local_rank_identity(torch: object, logical_bytes: int) -> dict[str, object]:
    identity = dict.fromkeys(_RANK_IDENTITY_FIELDS)
    try:
        if not torch.cuda.is_available():
            raise CapabilityError(
                "RSAG/qWD environment detection requires CUDA."
            )
        topology_class = os.environ.get(_ATTESTED_TOPOLOGY_ENV)
        transport = os.environ.get(_ATTESTED_TRANSPORT_ENV)
        _validate_runtime_attestation(topology_class, transport)
        cuda_version = torch.version.cuda
        if type(cuda_version) is not str or not cuda_version:
            raise CapabilityError("RSAG/qWD CUDA version is unavailable.")
        try:
            nccl_version = _nccl_version_string(torch.cuda.nccl.version())
        except (TypeError, ValueError) as error:
            raise CapabilityError(
                "RSAG/qWD NCCL version is unavailable."
            ) from error
        identity.update(
            {
                "hostname": socket.gethostname(),
                "logical_bytes": logical_bytes,
                "topology_class": topology_class,
                "transport": transport,
                "gpu_model": torch.cuda.get_device_name(),
                "torch_version": str(torch.__version__),
                "cuda_version": cuda_version,
                "nccl_version": nccl_version,
                "lowbit_comm_version": RSAG_LOWBIT_COMM_VERSION,
                "cuda_extension_abi": RSAG_CUDA_EXTENSION_ABI,
                "checkpoint_schema_version": RSAG_CHECKPOINT_SCHEMA_VERSION,
                "build_fingerprint": compute_rsag_build_fingerprint(),
                "error": None,
            }
        )
    except Exception as error:
        identity["error"] = f"{type(error).__name__}: {error}"
    return identity


def _validate_runtime_attestation(
    topology_class: object,
    transport: object,
) -> None:
    try:
        _require_exact_string(topology_class, _ATTESTED_TOPOLOGY_ENV)
        _require_exact_string(transport, _ATTESTED_TRANSPORT_ENV)
    except ValueError as error:
        raise CapabilityError(
            "RSAG/qWD launcher attestation is unavailable."
        ) from error
    if topology_class in {"single_node_pcie", "single_node_nvlink"}:
        if transport != "nccl_p2p":
            raise CapabilityError(
                "RSAG/qWD launcher attestation is not normalized."
            )
        if os.environ.get("NCCL_P2P_DISABLE") == "1":
            raise CapabilityError(
                "RSAG/qWD launcher attestation conflicts with NCCL P2P."
            )
        return
    if topology_class == "cross_node_socket":
        prefix = "nccl_socket_"
        if not transport.startswith(prefix):
            raise CapabilityError(
                "RSAG/qWD launcher attestation is not normalized."
            )
        interface = transport[len(prefix) :]
        if not interface or any(
            not (character.isalnum() or character in "_.-")
            for character in interface
        ):
            raise CapabilityError(
                "RSAG/qWD launcher attestation has an invalid interface."
            )
        configured_interface = os.environ.get("NCCL_SOCKET_IFNAME", "")
        if configured_interface.startswith("="):
            configured_interface = configured_interface[1:]
        if (
            os.environ.get("NCCL_IB_DISABLE") != "1"
            or configured_interface != interface
        ):
            raise CapabilityError(
                "RSAG/qWD launcher attestation conflicts with NCCL Socket."
            )
        return
    raise CapabilityError(
        "RSAG/qWD launcher attestation has an unsupported topology."
    )


def _validated_rank_identities(
    values: list[object | None],
) -> tuple[dict[str, object], ...]:
    identities: list[dict[str, object]] = []
    errors: list[str] = []
    for rank, value in enumerate(values):
        if type(value) is not dict or set(value) != _RANK_IDENTITY_FIELDS:
            raise CapabilityError(
                "RSAG/qWD gathered rank identity is incomplete."
            )
        error = value["error"]
        if error is not None:
            if type(error) is not str or not error:
                raise CapabilityError(
                    "RSAG/qWD gathered rank error is invalid."
                )
            errors.append(f"rank {rank}: {error}")
        identities.append(value)
    if errors:
        raise CapabilityError(
            "RSAG/qWD rank identity or launcher attestation failed: "
            + "; ".join(errors)
        )
    try:
        for rank, identity in enumerate(identities):
            _require_exact_string(identity["hostname"], f"rank {rank} hostname")
            _require_nonnegative_int(
                identity["logical_bytes"],
                f"rank {rank} logical_bytes",
            )
            for field_name in _ENVIRONMENT_STRING_FIELDS:
                _require_exact_string(
                    identity[field_name],
                    f"rank {rank} {field_name}",
                )
            _require_nonnegative_int(
                identity["cuda_extension_abi"],
                f"rank {rank} cuda_extension_abi",
            )
            _require_nonnegative_int(
                identity["checkpoint_schema_version"],
                f"rank {rank} checkpoint_schema_version",
            )
            _require_sha256(
                identity["build_fingerprint"],
                f"rank {rank} build_fingerprint",
            )
    except (OverflowError, ValueError) as error:
        raise CapabilityError(
            "RSAG/qWD gathered rank identity is invalid."
        ) from error
    return tuple(identities)


def _validate_attested_node_count(
    topology_class: object,
    node_count: int,
) -> None:
    if (
        node_count == 1
        and topology_class not in {"single_node_pcie", "single_node_nvlink"}
    ) or (node_count > 1 and topology_class != "cross_node_socket"):
        raise CapabilityError(
            "RSAG/qWD launcher topology attestation conflicts with hostnames."
        )


def _select_automatic_route(
    environment: RSAGEnvironment,
    evidence: tuple[RSAGEvidence, ...],
) -> RouteDecision:
    if any(
        _is_unknown_identity(getattr(environment, field_name))
        for field_name in _ENVIRONMENT_STRING_FIELDS
    ):
        return RouteDecision("native", "unknown_environment", None)
    if (
        environment.lowbit_comm_version != RSAG_LOWBIT_COMM_VERSION
        or environment.cuda_extension_abi != RSAG_CUDA_EXTENSION_ABI
        or environment.checkpoint_schema_version
        != RSAG_CHECKPOINT_SCHEMA_VERSION
    ):
        return RouteDecision(
            "native",
            "unsupported_binary_identity",
            None,
        )
    runtime = RSAGRuntimeABI(
        torch_version=environment.torch_version,
        cuda_version=environment.cuda_version,
        nccl_version=environment.nccl_version,
        cuda_extension_abi=environment.cuda_extension_abi,
    )
    if not is_verified_rsag_runtime(runtime):
        return RouteDecision(
            "native",
            "unsupported_runtime_matrix",
            None,
        )
    matching = tuple(
        record for record in evidence if record.matches(environment)
    )
    if not matching:
        return RouteDecision("native", "no_exact_evidence", None)
    if len(matching) != 1:
        return RouteDecision("native", "ambiguous_evidence", None)
    record = matching[0]
    if record.schema_version != RSAG_EVIDENCE_SCHEMA_VERSION:
        return RouteDecision(
            "native",
            "unsupported_evidence_schema",
            record.schema_version,
        )
    if not record.quality_passed:
        return RouteDecision(
            "native",
            "quality_failed",
            record.schema_version,
        )
    if any(value <= 0.0 for value in record.seed_speedups_percent):
        return RouteDecision(
            "native",
            "nonpositive_seed",
            record.schema_version,
        )
    return RouteDecision(
        "rsag_qwd",
        "qualified_positive_evidence",
        record.schema_version,
    )


@dataclass(frozen=True, slots=True)
class RSAGQWDPlans:
    """Compiled gradient and weight-delta plans for one qualified rank."""

    layout: ShardLayout
    gradient_plan: object
    qwd_plan: object
    qwd_gathered_payload_bytes: int
    fp32_gathered_bytes: int


class RSAGQWDAdapter:
    """Fail-closed training-framework seam for experimental RSAG/qWD."""

    __slots__ = ("_environment", "_evidence", "_requested", "_schedule")

    def __init__(
        self,
        environment: RSAGEnvironment,
        evidence: tuple[RSAGEvidence, ...],
        *,
        requested: str = "auto",
    ) -> None:
        decision = select_rsag_route(
            environment,
            evidence,
            requested=requested,
        )
        self._environment = environment
        self._evidence = evidence
        self._requested = requested
        self._schedule = QWDSchedule(refresh_interval=100)
        if decision != self.decision:
            raise CapabilityError("RSAG/qWD decision is not reproducible.")

    @property
    def environment(self) -> RSAGEnvironment:
        """Return the immutable environment bound to this adapter."""
        return self._environment

    @property
    def decision(self) -> RouteDecision:
        """Re-evaluate the immutable evidence before every use."""
        return select_rsag_route(
            self._environment,
            self._evidence,
            requested=self._requested,
        )

    @property
    def schedule(self) -> QWDSchedule:
        """Return the fixed qWD refresh schedule."""
        return self._schedule

    def mode(self, step: int, force_refresh: bool = False) -> str:
        """Return Native or the deterministic qualified qWD mode."""
        if not self.decision.uses_rsag:
            return "native"
        return self.schedule.mode(step, force_refresh=force_refresh)

    def create_plans(
        self,
        process_group: object,
        *,
        global_numel: int,
        rank: int,
    ) -> RSAGQWDPlans:
        """Compile CUDA plans only after exact evidence qualification."""
        if not self.decision.uses_rsag:
            raise CapabilityError(
                "RSAG/qWD plans require a qualified route decision."
            )
        _require_nonnegative_int(global_numel, "global_numel")
        if type(rank) is not int or rank < 0 or rank >= self.environment.world_size:
            raise ValueError("rank must be within the qualified world_size")
        if _checked_mul(global_numel, 2) != self.environment.logical_bytes:
            raise CapabilityError(
                "RSAG/qWD plan size does not match qualified evidence."
            )
        runtime_environment = detect_rsag_environment(
            process_group,
            logical_bytes=self.environment.logical_bytes,
            topology_class=self.environment.topology_class,
            transport=self.environment.transport,
        )
        if runtime_environment != self.environment:
            raise CapabilityError(
                "RSAG/qWD live runtime identity differs from evidence."
            )
        return _create_rsag_qwd_plans(
            process_group,
            global_numel=global_numel,
            world_size=self.environment.world_size,
            rank=rank,
        )


@dataclass(frozen=True, slots=True)
class _ResidualCandidate:
    """Prepared residual data owned by one transaction."""

    owner: int
    value: object


class CommittedResidual:
    """Error-feedback residual with explicit publish-or-abort semantics."""

    __slots__ = ("_committed", "_pending", "_owner")

    def __init__(self, value: object) -> None:
        self._committed = deepcopy(value)
        self._pending: _ResidualCandidate | None = None
        self._owner = id(self)

    @property
    def value(self) -> object:
        """Return a copy of the last committed residual."""
        return deepcopy(self._committed)

    def prepare(self, value: object) -> _ResidualCandidate:
        """Stage a candidate without changing the committed residual."""
        candidate = _ResidualCandidate(self._owner, deepcopy(value))
        self._pending = candidate
        return candidate

    def commit(self, candidate: _ResidualCandidate) -> None:
        """Publish the one candidate currently staged by this residual."""
        if (
            type(candidate) is not _ResidualCandidate
            or candidate is not self._pending
        ):
            raise ValueError("candidate was not prepared by this transaction")
        self._committed = deepcopy(candidate.value)
        self._pending = None

    def abort(self) -> None:
        """Discard staged state while retaining the last committed residual."""
        self._pending = None


class ShardedAdamW:
    """FP32 AdamW state for the valid portion of one padded shard."""

    __slots__ = (
        "layout",
        "master",
        "exp_avg",
        "exp_avg_sq",
        "step_count",
        "learning_rate",
        "betas",
        "eps",
        "weight_decay",
        "amp_state",
        "rng_state",
        "force_refresh",
    )

    def __init__(
        self,
        layout: ShardLayout,
        master: object,
        *,
        learning_rate: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> None:
        self.layout = _trusted_shard_layout(layout)
        self.learning_rate = _require_finite_float(
            learning_rate,
            "learning_rate",
            positive=True,
        )
        self.betas = _require_betas(betas)
        self.eps = _require_finite_float(eps, "eps", positive=True)
        self.weight_decay = _require_finite_float(
            weight_decay,
            "weight_decay",
            nonnegative=True,
        )
        self.master = _validated_shard_tensor(master, layout, "master").clone()
        self._zero_padding(self.master)
        self.exp_avg = self.master.new_zeros(layout.padded_numel)
        self.exp_avg_sq = self.master.new_zeros(layout.padded_numel)
        self.step_count = 0
        self.amp_state: dict[str, object] = {}
        self.rng_state: dict[str, object] = {}
        self.force_refresh = False

    def step(self, gradient_shard: object) -> None:
        """Apply one FP32 AdamW update to valid elements only."""
        gradient = _validated_shard_tensor(
            gradient_shard,
            self.layout,
            "gradient_shard",
            device=self.master.device,
        )
        self._step_validated(gradient)

    def step_prevalidated(self, gradient_shard: object) -> None:
        (
            "Update from a same-device shard whose finiteness was already "
            "proven."
        )
        gradient = _validated_shard_tensor(
            gradient_shard,
            self.layout,
            "gradient_shard",
            device=self.master.device,
            require_finite=False,
        )
        self._step_validated(gradient)

    def _step_validated(self, gradient: object) -> None:
        valid = self.layout.valid_numel
        if valid == 0:
            self.step_count += 1
            self._zero_padding(self.master)
            return

        torch = _torch()
        with torch.no_grad():
            master = self.master[:valid]
            exp_avg = self.exp_avg[:valid]
            exp_avg_sq = self.exp_avg_sq[:valid]
            gradient = gradient[:valid]
            beta1, beta2 = self.betas
            self.step_count += 1
            master.mul_(1.0 - self.learning_rate * self.weight_decay)
            exp_avg.lerp_(gradient, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                gradient,
                gradient,
                value=1.0 - beta2,
            )
            bias_correction1 = 1.0 - beta1**self.step_count
            bias_correction2_sqrt = (1.0 - beta2**self.step_count) ** 0.5
            denominator = (
                exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
            )
            master.addcdiv_(
                exp_avg,
                denominator,
                value=-(self.learning_rate / bias_correction1),
            )
            self._zero_padding(self.master)
            self._zero_padding(self.exp_avg)
            self._zero_padding(self.exp_avg_sq)

    def state_dict(self) -> dict[str, object]:
        """Return a detached checkpoint with exact builtin state containers."""
        if type(self.amp_state) is not dict:
            raise ValueError("amp_state must be an exact dict")
        if type(self.rng_state) is not dict:
            raise ValueError("rng_state must be an exact dict")
        return {
            "schema_version": RSAG_CHECKPOINT_SCHEMA_VERSION,
            "layout": _layout_state(self.layout),
            "master": self.master.clone(),
            "exp_avg": self.exp_avg.clone(),
            "exp_avg_sq": self.exp_avg_sq.clone(),
            "step_count": self.step_count,
            "learning_rate": self.learning_rate,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "amp_state": deepcopy(self.amp_state),
            "rng_state": deepcopy(self.rng_state),
            "force_refresh": self.force_refresh,
        }

    def _audit_state(self) -> dict[str, object]:
        """Borrow current tensors for immediate synchronous state hashing."""
        if type(self.amp_state) is not dict:
            raise ValueError("amp_state must be an exact dict")
        if type(self.rng_state) is not dict:
            raise ValueError("rng_state must be an exact dict")
        return {
            "schema_version": RSAG_CHECKPOINT_SCHEMA_VERSION,
            "layout": _layout_state(self.layout),
            "master": self.master,
            "exp_avg": self.exp_avg,
            "exp_avg_sq": self.exp_avg_sq,
            "step_count": self.step_count,
            "learning_rate": self.learning_rate,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "amp_state": self.amp_state,
            "rng_state": self.rng_state,
            "force_refresh": self.force_refresh,
        }

    def load_state_dict(self, state: object) -> None:
        """Restore one fully validated checkpoint without cadence drift."""
        if type(state) is not dict:
            raise ValueError("state must be an exact dict")
        expected_fields = {
            "schema_version",
            "layout",
            "master",
            "exp_avg",
            "exp_avg_sq",
            "step_count",
            "learning_rate",
            "betas",
            "eps",
            "weight_decay",
            "amp_state",
            "rng_state",
            "force_refresh",
        }
        if set(state) != expected_fields:
            raise ValueError("state fields are invalid")
        schema_version = state["schema_version"]
        if (
            type(schema_version) is not int
            or schema_version != RSAG_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError("schema_version is incompatible")
        layout_state = state["layout"]
        _validate_checkpoint_layout_state(layout_state, self.layout)
        master = _validated_shard_tensor(
            state["master"],
            self.layout,
            "master",
            device=self.master.device,
        )
        exp_avg = _validated_shard_tensor(
            state["exp_avg"],
            self.layout,
            "exp_avg",
            device=self.master.device,
        )
        exp_avg_sq = _validated_shard_tensor(
            state["exp_avg_sq"],
            self.layout,
            "exp_avg_sq",
            device=self.master.device,
        )
        _require_zero_padding(master, self.layout, "master")
        _require_zero_padding(exp_avg, self.layout, "exp_avg")
        _require_zero_padding(exp_avg_sq, self.layout, "exp_avg_sq")
        step_count = state["step_count"]
        if type(step_count) is not int or step_count < 0:
            raise ValueError("step_count must be a non-negative exact integer")
        learning_rate = _require_finite_float(
            state["learning_rate"],
            "learning_rate",
            positive=True,
        )
        betas = _require_betas(state["betas"])
        eps = _require_finite_float(state["eps"], "eps", positive=True)
        weight_decay = _require_finite_float(
            state["weight_decay"],
            "weight_decay",
            nonnegative=True,
        )
        amp_state = state["amp_state"]
        if type(amp_state) is not dict:
            raise ValueError("amp_state must be an exact dict")
        rng_state = state["rng_state"]
        if type(rng_state) is not dict:
            raise ValueError("rng_state must be an exact dict")
        force_refresh = state["force_refresh"]
        if type(force_refresh) is not bool:
            raise ValueError("force_refresh must be an exact bool")

        prepared_master = master.clone()
        prepared_exp_avg = exp_avg.clone()
        prepared_exp_avg_sq = exp_avg_sq.clone()
        prepared_amp_state = deepcopy(amp_state)
        prepared_rng_state = deepcopy(rng_state)

        self.master = prepared_master
        self.exp_avg = prepared_exp_avg
        self.exp_avg_sq = prepared_exp_avg_sq
        self.step_count = step_count
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amp_state = prepared_amp_state
        self.rng_state = prepared_rng_state
        self.force_refresh = force_refresh

    def _zero_padding(self, value: object) -> None:
        if self.layout.valid_numel < self.layout.padded_numel:
            value[self.layout.valid_numel :].zero_()


def flatten_parameter_copy(parameters: object) -> object:
    """Flatten CPU FP32 parameters into detached contiguous storage."""
    torch = _torch()
    if type(parameters) is not tuple:
        raise ValueError("parameters must be an exact tuple")
    flattened = [
        _validated_parameter(parameter, index)
        for index, parameter in enumerate(parameters)
    ]
    if not flattened:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(
        [parameter.detach().reshape(-1).clone() for parameter in flattened]
    )


def copy_flat_to_parameters(flat: object, parameters: object) -> None:
    """Copy a flat CPU FP32 vector back into same-sized parameters."""
    torch = _torch()
    if type(parameters) is not tuple:
        raise ValueError("parameters must be an exact tuple")
    if type(flat) is not torch.Tensor or flat.device.type != "cpu":
        raise ValueError("flat must be a CPU tensor")
    if flat.dtype is not torch.float32 or flat.ndim != 1:
        raise ValueError("flat must be a one-dimensional FP32 tensor")
    validated = [
        _validated_parameter(parameter, index)
        for index, parameter in enumerate(parameters)
    ]
    total_numel = sum(parameter.numel() for parameter in validated)
    if flat.numel() != total_numel:
        raise ValueError("flat has an invalid numel")
    offset = 0
    with torch.no_grad():
        for parameter in validated:
            next_offset = offset + parameter.numel()
            parameter.copy_(flat[offset:next_offset].reshape_as(parameter))
            offset = next_offset


def _create_rsag_qwd_plans(
    process_group: object,
    *,
    global_numel: int,
    world_size: int,
    rank: int,
) -> RSAGQWDPlans:
    lowbit = import_module("lowbit_comm")
    backend_module = import_module("lowbit_comm.backends.cuda.backend")
    loader = import_module("lowbit_comm.backends.cuda.loader")
    layout = ShardLayout.build(global_numel, world_size, rank)
    intent = lowbit.CommunicationIntent(
        tensor=lowbit.TensorSpec(dtype="fp16", shape=(global_numel,)),
        shape_family=lowbit.ShapeFamily(max_numel=global_numel, alignment=1),
        reduction=lowbit.ReductionOp.MEAN,
        output=lowbit.OutputSemantics.REDUCED_SHARD,
        completion=lowbit.CompletionMode.ASYNC,
        world_size=world_size,
        rank=rank,
    )
    strategy = lowbit.StrategySpec(
        compression=lowbit.CompressionKind.INT8,
        collective=lowbit.CollectiveKind.COMPRESSED_REDUCE_SCATTER,
        topology=lowbit.TopologyKind.BACKEND_DEFAULT,
        group_size=64,
        error_feedback=True,
    )
    gradient_plan = backend_module.CudaBackend(process_group).lower(
        intent,
        strategy,
    )
    qwd_config = _qwd_config(global_numel, rank, world_size)
    extension = loader.load_extension()
    create_qwd_plan = getattr(extension, "_create_qwd_plan", None)
    if not callable(create_qwd_plan):
        raise CapabilityError(
            "CUDA extension does not provide the experimental qWD ABI."
        )
    qwd_plan = create_qwd_plan(qwd_config, process_group)
    return RSAGQWDPlans(
        layout=layout,
        gradient_plan=gradient_plan,
        qwd_plan=qwd_plan,
        qwd_gathered_payload_bytes=int(
            qwd_config["qwd_gathered_payload_bytes"]
        ),
        fp32_gathered_bytes=int(qwd_config["fp32_gathered_bytes"]),
    )


def _qwd_config(
    global_numel: int,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    layout = ShardLayout.build(global_numel, world_size, rank)
    groups = _checked_ceil_div(layout.padded_numel, 64)
    payload = _checked_mul(groups, 68)
    gathered_payload = _checked_mul(payload, world_size)
    fp32_gathered = _checked_mul(
        _checked_mul(layout.padded_numel, world_size),
        4,
    )
    output_bytes = _checked_mul(
        _checked_mul(layout.padded_numel, world_size),
        2,
    )
    workspace_bytes = max(
        payload + gathered_payload,
        fp32_gathered,
    )
    if workspace_bytes > _MAX_SIGNED_64:
        raise OverflowError("qWD workspace size overflow")
    return {
        "accumulation_dtype": "fp32",
        "collective": "all_gather",
        "compression": "int8",
        "dtype": "fp16",
        "fp32_gathered_bytes": fp32_gathered,
        "global_numel": global_numel,
        "group_size": 64,
        "groups_per_shard": groups,
        "output_bytes": output_bytes,
        "payload_bytes_per_rank": payload,
        "qwd_gathered_payload_bytes": gathered_payload,
        "rank": rank,
        "shard_numel": layout.padded_numel,
        "start": layout.start,
        "valid_numel": layout.valid_numel,
        "workspace_bytes": workspace_bytes,
        "world_size": world_size,
    }


def _torch() -> object:
    try:
        return import_module("torch")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ShardedAdamW requires the optional torch package"
        ) from error


def _trusted_shard_layout(value: object) -> ShardLayout:
    if type(value) is not ShardLayout:
        raise ValueError("layout must be an exact ShardLayout")
    _validate_shard_layout(value, "layout")
    return value


def _layout_state(layout: ShardLayout) -> dict[str, int]:
    """Return the exact immutable identity of one validated shard."""
    _validate_shard_layout(layout, "layout")
    return {
        "global_numel": layout.global_numel,
        "world_size": layout.world_size,
        "rank": layout.rank,
        "start": layout.start,
        "valid_numel": layout.valid_numel,
        "padded_numel": layout.padded_numel,
    }


def _validate_checkpoint_layout_state(
    value: object,
    expected_layout: ShardLayout,
) -> None:
    expected = _layout_state(expected_layout)
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("layout is incompatible")
    if any(type(value[field_name]) is not int for field_name in expected):
        raise ValueError("layout fields must be exact integers")
    if value != expected:
        raise ValueError("layout is incompatible")


def _validate_shard_layout(layout: ShardLayout, name: str) -> None:
    _require_nonnegative_int(layout.global_numel, f"{name}.global_numel")
    _require_positive_int(layout.world_size, f"{name}.world_size")
    if (
        type(layout.rank) is not int
        or layout.rank < 0
        or layout.rank >= layout.world_size
    ):
        raise ValueError(f"{name}.rank is outside world_size")
    _require_nonnegative_int(layout.start, f"{name}.start")
    _require_nonnegative_int(layout.valid_numel, f"{name}.valid_numel")
    _require_nonnegative_int(layout.padded_numel, f"{name}.padded_numel")
    expected_padded_numel = _checked_ceil_div(
        layout.global_numel,
        layout.world_size,
    )
    if layout.padded_numel != expected_padded_numel:
        raise ValueError(f"{name}.padded_numel is inconsistent")
    expected_start = min(
        _checked_mul(layout.rank, layout.padded_numel),
        layout.global_numel,
    )
    if layout.start != expected_start:
        raise ValueError(f"{name}.start is inconsistent")
    expected_valid_numel = min(
        layout.padded_numel,
        layout.global_numel - layout.start,
    )
    if layout.valid_numel != expected_valid_numel:
        raise ValueError(f"{name}.valid_numel is inconsistent")


def _validated_shard_tensor(
    value: object,
    layout: ShardLayout,
    name: str,
    *,
    device: object | None = None,
    require_finite: bool = True,
) -> object:
    torch = _torch()
    if type(value) is not torch.Tensor:
        raise ValueError(f"{name} must be an exact tensor")
    if value.device.type not in {"cpu", "cuda"}:
        raise ValueError(f"{name} must be a CPU or CUDA tensor")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on the optimizer state device")
    if value.dtype is not torch.float32 or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional FP32 tensor")
    if value.numel() != layout.padded_numel:
        raise ValueError(f"{name} has an invalid numel")
    if require_finite and not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")
    return value


def _validated_parameter(value: object, index: int) -> object:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"parameter {index} must be a tensor")
    if value.device.type != "cpu" or value.dtype is not torch.float32:
        raise ValueError(f"parameter {index} must be a CPU FP32 tensor")
    return value


def _require_zero_padding(
    value: object, layout: ShardLayout, name: str
) -> None:
    if layout.valid_numel == layout.padded_numel:
        return
    if value[layout.valid_numel :].count_nonzero().item() != 0:
        raise ValueError(f"{name} padding must be zero")


def _require_finite_float(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite exact float")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_betas(value: object) -> tuple[float, float]:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("betas must be an exact two-item tuple")
    beta1 = _require_finite_float(value[0], "betas")
    beta2 = _require_finite_float(value[1], "betas")
    if beta1 < 0.0 or beta1 >= 1.0 or beta2 < 0.0 or beta2 >= 1.0:
        raise ValueError("betas must be in [0, 1)")
    return beta1, beta2


def _require_exact_string(value: object, name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty exact string")


def _require_sha256(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 string")


def _is_unknown_identity(value: str) -> bool:
    return value.casefold() in _UNKNOWN_IDENTITIES




def _require_nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative exact integer")
    if value > _MAX_SIGNED_64:
        raise OverflowError(f"{name} exceeds the signed 64-bit domain")


def _require_positive_int(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact integer")
    if value > _MAX_SIGNED_64:
        raise OverflowError(f"{name} exceeds the signed 64-bit domain")


def _checked_ceil_div(value: int, divisor: int) -> int:
    quotient, remainder = divmod(value, divisor)
    return quotient + (remainder != 0)


def _checked_mul(left: int, right: int) -> int:
    result = left * right
    if result > _MAX_SIGNED_64:
        raise OverflowError("shard layout size overflow")
    return result
