"""Installable state and fail-closed routing for experimental RSAG/qWD."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
from importlib import import_module
import json
from math import isfinite
import os
from pathlib import Path
import socket
from threading import Lock

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
class RSAGLaunchAttestation:
    """NCCL topology settings captured before process-group creation."""

    topology_class: str
    transport: str
    nccl_ib_disable: str | None
    nccl_net: str | None
    nccl_socket_ifname: str | None
    nccl_p2p_disable: str | None

    def __post_init__(self) -> None:
        _validate_launch_attestation_values(self)


_CAPTURED_LAUNCH_ATTESTATIONS: dict[
    int,
    tuple[
        RSAGLaunchAttestation,
        tuple[object, ...],
        object,
        str,
        str,
    ],
] = {}
_RSAG_LAUNCH_ATTEMPTED = False
_RSAG_LAUNCH_LOCK = Lock()


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
            raise ValueError("seed_speedups_percent must contain finite exact floats")
        if type(self.quality_passed) is not bool:
            raise ValueError("quality_passed must be an exact bool")

    def matches(self, environment: RSAGEnvironment) -> bool:
        """Return whether every evidence-key field matches the runtime."""
        return (
            self.world_size == environment.world_size
            and self.node_count == environment.node_count
            and self.cuda_extension_abi == environment.cuda_extension_abi
            and self.checkpoint_schema_version == environment.checkpoint_schema_version
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


@dataclass(frozen=True, slots=True)
class _RSAGPlanPreflight:
    """Pickle-safe rank-local qualification request for one collective gate."""

    qualification_fingerprint: str | None
    global_numel: int | None
    rank: int | None
    error: str | None


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
_CONTROL_NAMESPACE = "lowbit_comm/rsag/launch/v1"
_CONTROL_TIMEOUT = timedelta(minutes=10)
_DMI_UUID_PATH = Path("/sys/class/dmi/id/product_uuid")
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_NVIDIA_GPU_INFO_ROOT = Path("/proc/driver/nvidia/gpus")
_PRODUCT_TOPOLOGY_CLASS = "cross_node_socket"
_RANK_IDENTITY_FIELDS = frozenset(
    {
        "hostname",
        "physical_node_id",
        "gpu_inventory_fingerprint",
        "logical_bytes",
        "requested_topology_class",
        "requested_transport",
        "topology_class",
        "transport",
        "nccl_ib_disable",
        "nccl_net",
        "nccl_socket_ifname",
        "nccl_p2p_disable",
        "gpu_model",
        "torch_version",
        "cuda_version",
        "nccl_version",
        "lowbit_comm_version",
        "cuda_extension_abi",
        "checkpoint_schema_version",
        "build_fingerprint",
        "plan_preflight",
        "error",
    }
)
_RANK_CONSENSUS_FIELDS = tuple(
    sorted(
        _RANK_IDENTITY_FIELDS
        - {
            "hostname",
            "physical_node_id",
            "gpu_inventory_fingerprint",
            "plan_preflight",
            "error",
        }
    )
)
_PLAN_PREFLIGHT_FIELDS = frozenset(
    {"qualification_fingerprint", "global_numel", "rank", "error"}
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
    policy: str = "interval100"

    def __post_init__(self) -> None:
        _require_positive_int(self.refresh_interval, "refresh_interval")
        if self.refresh_interval != 100:
            raise ValueError("refresh_interval must be exactly 100")
        if type(self.policy) is not str or self.policy not in {
            "interval100",
            "all_refresh",
        }:
            raise ValueError("policy must be interval100 or all_refresh")

    def mode(self, step: int, force_refresh: bool = False) -> str:
        """Return the communication route required for one optimizer step."""
        _require_nonnegative_int(step, "step")
        if type(force_refresh) is not bool:
            raise ValueError("force_refresh must be an exact bool")
        if force_refresh or self.policy == "all_refresh" or step % self.refresh_interval == 0:
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
        raise CapabilityError(f"RSAG/qWD route is not eligible: {decision.reason}.")
    return decision


def detect_rsag_environment(
    process_group: object,
    *,
    logical_bytes: object,
    topology_class: object,
    transport: object,
    launch_attestation: object,
    plan_preflight: object | None = None,
) -> RSAGEnvironment:
    """Gather the live binary, GPU, rank, and node evidence identity."""
    torch = _torch()
    distributed = torch.distributed
    if not distributed.is_initialized():
        raise CapabilityError(
            "RSAG/qWD environment detection requires distributed init."
        )
    world_size = distributed.get_world_size(process_group)
    _require_positive_int(world_size, "world_size")
    local_identity = _local_rank_identity(
        torch,
        process_group=process_group,
        logical_bytes=logical_bytes,
        requested_topology_class=topology_class,
        requested_transport=transport,
        launch_attestation=launch_attestation,
        plan_preflight=plan_preflight,
    )
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
        raise CapabilityError("RSAG/qWD rank identity is not globally consistent.")
    _validate_plan_preflights(identities)
    if (
        baseline["topology_class"] != baseline["requested_topology_class"]
        or baseline["transport"] != baseline["requested_transport"]
    ):
        raise CapabilityError(
            "RSAG/qWD launcher attestation differs from the requested environment."
        )
    node_count = _physical_node_count(identities)
    _validate_attested_node_count(
        baseline["topology_class"],
        node_count,
    )
    return RSAGEnvironment(
        world_size=world_size,
        node_count=node_count,
        logical_bytes=baseline["logical_bytes"],
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


def _local_rank_identity(
    torch: object,
    *,
    process_group: object,
    logical_bytes: object,
    requested_topology_class: object,
    requested_transport: object,
    launch_attestation: object,
    plan_preflight: object | None,
) -> dict[str, object]:
    identity = dict.fromkeys(_RANK_IDENTITY_FIELDS)
    safe_plan_preflight = _plan_preflight_state(plan_preflight)
    identity.update(
        {
            "logical_bytes": logical_bytes if type(logical_bytes) is int else None,
            "requested_topology_class": (
                requested_topology_class
                if type(requested_topology_class) is str
                else None
            ),
            "requested_transport": (
                requested_transport if type(requested_transport) is str else None
            ),
            "plan_preflight": safe_plan_preflight,
        }
    )
    try:
        _require_nonnegative_int(logical_bytes, "logical_bytes")
        _require_exact_string(
            requested_topology_class,
            "topology_class",
        )
        _require_exact_string(requested_transport, "transport")
        if (
            plan_preflight is not None
            and type(plan_preflight) is not _RSAGPlanPreflight
        ):
            raise CapabilityError("RSAG/qWD plan preflight is invalid.")
        if not torch.cuda.is_available():
            raise CapabilityError("RSAG/qWD environment detection requires CUDA.")
        if type(launch_attestation) is not RSAGLaunchAttestation:
            raise CapabilityError(
                "RSAG/qWD requires a pre-process-group launch attestation."
            )
        captured = _CAPTURED_LAUNCH_ATTESTATIONS.get(id(launch_attestation))
        if (
            captured is None
            or captured[0] is not launch_attestation
            or captured[2] is not process_group
        ):
            raise CapabilityError(
                "RSAG/qWD requires a registered process-group launch."
            )
        if captured[1] != _launch_attestation_state(launch_attestation):
            raise CapabilityError(
                "RSAG/qWD launcher attestation changed since capture."
            )
        launch_attestation.__post_init__()
        if _read_current_launch_attestation() != launch_attestation:
            raise CapabilityError(
                "RSAG/qWD launcher attestation changed after capture."
            )
        cuda_version = torch.version.cuda
        if type(cuda_version) is not str or not cuda_version:
            raise CapabilityError("RSAG/qWD CUDA version is unavailable.")
        try:
            nccl_version = _nccl_version_string(torch.cuda.nccl.version())
        except (TypeError, ValueError) as error:
            raise CapabilityError("RSAG/qWD NCCL version is unavailable.") from error
        identity.update(
            {
                "hostname": socket.gethostname(),
                "physical_node_id": captured[3],
                "gpu_inventory_fingerprint": captured[4],
                "topology_class": launch_attestation.topology_class,
                "transport": launch_attestation.transport,
                "nccl_ib_disable": launch_attestation.nccl_ib_disable,
                "nccl_net": launch_attestation.nccl_net,
                "nccl_socket_ifname": launch_attestation.nccl_socket_ifname,
                "nccl_p2p_disable": launch_attestation.nccl_p2p_disable,
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


def _claim_launch_attempt() -> None:
    global _RSAG_LAUNCH_ATTEMPTED
    with _RSAG_LAUNCH_LOCK:
        if _RSAG_LAUNCH_ATTEMPTED:
            raise CapabilityError(
                "RSAG/qWD launcher-fatal: process-group launch is one-shot; "
                "restart every worker."
            )
        _RSAG_LAUNCH_ATTEMPTED = True


def _timedelta_microseconds(value: timedelta) -> int:
    return (value.days * 86_400 + value.seconds) * 1_000_000 + value.microseconds


def initialize_rsag_process_group(
    *,
    backend: str = "nccl",
    **kwargs: object,
) -> RSAGLaunchAttestation:
    """Collectively attest NCCL settings and initialize the bound group."""
    _claim_launch_attempt()
    torch = _torch()
    distributed = torch.distributed
    if distributed.is_initialized():
        raise CapabilityError(
            "RSAG/qWD launcher-fatal: a process group already exists; "
            "restart every worker."
        )
    launch_errors: list[str] = []
    if type(backend) is not str or backend != "nccl":
        launch_errors.append("RSAG/qWD requires the NCCL backend.")
    unsupported = set(kwargs) - {"timeout"}
    if unsupported:
        launch_errors.append(
            "RSAG/qWD launcher does not accept process-group arguments: "
            + ", ".join(sorted(unsupported))
            + "."
        )
    control_timeout = kwargs.get("timeout", _CONTROL_TIMEOUT)
    if type(control_timeout) is not timedelta or control_timeout.total_seconds() <= 0:
        launch_errors.append("RSAG/qWD requires a positive timedelta timeout.")
        process_group_timeout_us = None
    else:
        process_group_timeout_us = _timedelta_microseconds(control_timeout)
    (
        attestation,
        store,
        rank,
        world_size,
        generation,
        physical_node_id,
        gpu_inventory_fingerprint,
    ) = _rendezvous_launch_control(
        distributed,
        timeout=_CONTROL_TIMEOUT,
        backend=backend if type(backend) is str else None,
        process_group_timeout_us=process_group_timeout_us,
        launch_error="; ".join(launch_errors) if launch_errors else None,
    )
    init_error: str | None = None
    try:
        distributed.init_process_group(
            backend=backend,
            store=store,
            rank=rank,
            world_size=world_size,
            timeout=control_timeout,
        )
    except Exception as error:
        init_error = f"{type(error).__name__}: {error}"
    try:
        init_records = _exchange_control_records(
            store,
            "init",
            {"rank": rank, "error": init_error},
            rank,
            world_size,
            generation,
        )
        _validate_init_control_records(init_records, generation)
        try:
            current_attestation = _read_current_launch_attestation()
        except Exception as error:
            post_record = {
                "rank": rank,
                "attestation": None,
                "error": f"{type(error).__name__}: {error}",
            }
        else:
            post_record = {
                "rank": rank,
                "attestation": list(_launch_attestation_state(current_attestation)),
                "error": None,
            }
        post_records = _exchange_control_records(
            store,
            "post",
            post_record,
            rank,
            world_size,
            generation,
        )
        _validate_post_init_control_records(
            post_records,
            attestation,
            generation,
        )
        if not distributed.is_initialized():
            raise CapabilityError(
                "RSAG/qWD default process group disappeared before binding."
            )
        process_group = distributed.group.WORLD
        if process_group is None:
            raise CapabilityError(
                "RSAG/qWD default process group is unavailable for binding."
            )
        _CAPTURED_LAUNCH_ATTESTATIONS[id(attestation)] = (
            attestation,
            _launch_attestation_state(attestation),
            process_group,
            physical_node_id,
            gpu_inventory_fingerprint,
        )
    except Exception as error:
        _CAPTURED_LAUNCH_ATTESTATIONS.pop(id(attestation), None)
        _safe_destroy_process_group(distributed)
        if type(error) is CapabilityError:
            raise
        raise CapabilityError("RSAG/qWD process-group binding failed.") from error
    return attestation


def _rendezvous_launch_control(
    distributed: object,
    *,
    timeout: timedelta,
    backend: object,
    process_group_timeout_us: object,
    launch_error: str | None,
) -> tuple[RSAGLaunchAttestation, object, int, int, str, str, str]:
    """Join the torchrun store before reading any rank-local attestation."""
    try:
        rendezvous = distributed.rendezvous("env://", timeout=timeout)
        store, rank, world_size = next(rendezvous)
        _require_nonnegative_int(rank, "rank")
        _require_positive_int(world_size, "world_size")
        if rank >= world_size:
            raise ValueError("rank must be smaller than world_size")
    except Exception as error:
        raise CapabilityError(
            "RSAG/qWD could not join the torchrun control rendezvous."
        ) from error
    generation = _control_generation()
    errors: list[str] = [launch_error] if launch_error is not None else []
    try:
        hostname = socket.gethostname()
        _require_exact_string(hostname, "hostname")
    except Exception as error:
        hostname = "unavailable"
        errors.append(f"{type(error).__name__}: {error}")
    try:
        (
            physical_node_id,
            gpu_inventory_fingerprint,
        ) = _hardware_node_identity()
    except Exception as error:
        physical_node_id = None
        gpu_inventory_fingerprint = None
        errors.append(f"{type(error).__name__}: {error}")
    try:
        attestation = _read_current_launch_attestation()
        attestation_state: list[object] | None = list(
            _launch_attestation_state(attestation)
        )
    except Exception as error:
        attestation_state = None
        errors.append(f"{type(error).__name__}: {error}")
    local_record = {
        "rank": rank,
        "backend": backend,
        "process_group_timeout_us": process_group_timeout_us,
        "hostname": hostname,
        "physical_node_id": physical_node_id,
        "gpu_inventory_fingerprint": gpu_inventory_fingerprint,
        "attestation": attestation_state,
        "error": "; ".join(errors) if errors else None,
    }
    records = _exchange_control_records(
        store,
        "pre",
        local_record,
        rank,
        world_size,
        generation,
    )
    return (
        _validated_launch_control_attestation(records, generation),
        store,
        rank,
        world_size,
        generation,
        physical_node_id,
        gpu_inventory_fingerprint,
    )


def _exchange_control_records(
    store: object,
    phase: str,
    local_record: dict[str, object],
    rank: int,
    world_size: int,
    generation: str,
) -> tuple[object, ...]:
    """Exchange JSON-only control records through the rendezvous store."""
    _require_exact_string(phase, "control phase")
    _require_nonnegative_int(rank, "rank")
    _require_positive_int(world_size, "world_size")
    if rank >= world_size:
        raise CapabilityError("RSAG/qWD control rank is out of range.")
    try:
        _require_sha256(generation, "control generation")
    except ValueError as error:
        raise CapabilityError("RSAG/qWD control generation is invalid.") from error
    if "generation" in local_record:
        raise CapabilityError("RSAG/qWD control generation is launcher-owned.")
    envelope = {**local_record, "generation": generation}
    prefix = f"{_CONTROL_NAMESPACE}/{generation}/{phase}"
    keys = [f"{prefix}/{candidate}" for candidate in range(world_size)]
    try:
        payload = json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        store.set(keys[rank], payload)
        store.wait(keys)
        return tuple(json.loads(bytes(store.get(key)).decode("utf-8")) for key in keys)
    except Exception as error:
        raise CapabilityError(f"RSAG/qWD {phase} control exchange failed.") from error


def _validated_launch_control_attestation(
    records: tuple[object, ...],
    generation: str,
) -> RSAGLaunchAttestation:
    """Return one unanimous, cross-node Socket launch attestation."""
    validated = _validated_control_records(
        records,
        fields={
            "rank",
            "backend",
            "process_group_timeout_us",
            "hostname",
            "physical_node_id",
            "gpu_inventory_fingerprint",
            "attestation",
            "error",
        },
        phase="pre-init attestation",
        generation=generation,
    )
    errors = _control_errors(validated)
    if errors:
        raise CapabilityError(
            "RSAG/qWD pre-init attestation failed: " + "; ".join(errors)
        )
    try:
        for rank, record in enumerate(validated):
            if record["backend"] != "nccl":
                raise ValueError(f"rank {rank} backend is invalid")
            _require_positive_int(
                record["process_group_timeout_us"],
                f"rank {rank} process_group_timeout_us",
            )
    except ValueError as error:
        raise CapabilityError(
            "RSAG/qWD pre-init process-group request is invalid."
        ) from error
    baseline_request = (
        validated[0]["backend"],
        validated[0]["process_group_timeout_us"],
    )
    if any(
        (record["backend"], record["process_group_timeout_us"]) != baseline_request
        for record in validated[1:]
    ):
        raise CapabilityError(
            "RSAG/qWD pre-init process-group request is not globally consistent."
        )
    attestations = tuple(
        _attestation_from_control_state(record["attestation"]) for record in validated
    )
    baseline = attestations[0]
    if any(value != baseline for value in attestations[1:]):
        raise CapabilityError(
            "RSAG/qWD pre-init attestation is not globally consistent."
        )
    hostnames = tuple(record["hostname"] for record in validated)
    physical_node_ids = tuple(record["physical_node_id"] for record in validated)
    gpu_inventory_fingerprints = tuple(
        record["gpu_inventory_fingerprint"] for record in validated
    )
    try:
        for rank, hostname in enumerate(hostnames):
            _require_exact_string(hostname, f"rank {rank} hostname")
        for rank, node_id in enumerate(physical_node_ids):
            _require_sha256(
                node_id,
                f"rank {rank} physical node ID",
            )
        for rank, fingerprint in enumerate(gpu_inventory_fingerprints):
            _require_sha256(
                fingerprint,
                f"rank {rank} GPU inventory fingerprint",
            )
    except ValueError as error:
        raise CapabilityError(
            "RSAG/qWD pre-init hardware node identity is invalid."
        ) from error
    inventory_by_node: dict[str, str] = {}
    for node_id, fingerprint in zip(
        physical_node_ids,
        gpu_inventory_fingerprints,
        strict=True,
    ):
        prior = inventory_by_node.setdefault(node_id, fingerprint)
        if prior != fingerprint:
            raise CapabilityError(
                "RSAG/qWD GPU inventory differs within one physical node."
            )
    if (
        baseline.topology_class != _PRODUCT_TOPOLOGY_CLASS
        or len(set(physical_node_ids)) <= 1
    ):
        raise CapabilityError(
            "RSAG/qWD product qualification requires a derived cross-node "
            "Socket topology."
        )
    return baseline


def _validate_init_control_records(
    records: tuple[object, ...],
    generation: str,
) -> None:
    validated = _validated_control_records(
        records,
        fields={"rank", "error"},
        phase="process-group initialization",
        generation=generation,
    )
    errors = _control_errors(validated)
    if errors:
        raise CapabilityError(
            "RSAG/qWD process-group initialization failed: " + "; ".join(errors)
        )


def _validate_post_init_control_records(
    records: tuple[object, ...],
    expected: RSAGLaunchAttestation,
    generation: str,
) -> None:
    validated = _validated_control_records(
        records,
        fields={"rank", "attestation", "error"},
        phase="post-init attestation",
        generation=generation,
    )
    errors = _control_errors(validated)
    if errors:
        raise CapabilityError(
            "RSAG/qWD post-init attestation failed: " + "; ".join(errors)
        )
    attestations = tuple(
        _attestation_from_control_state(record["attestation"]) for record in validated
    )
    if any(attestation != expected for attestation in attestations):
        raise CapabilityError(
            "RSAG/qWD launcher environment changed during process-group init."
        )


def _validated_control_records(
    records: tuple[object, ...],
    *,
    fields: set[str],
    phase: str,
    generation: str,
) -> tuple[dict[str, object], ...]:
    if type(records) is not tuple or not records:
        raise CapabilityError(f"RSAG/qWD {phase} records are unavailable.")
    validated: list[dict[str, object]] = []
    for rank, record in enumerate(records):
        if type(record) is not dict or set(record) != fields | {"generation"}:
            raise CapabilityError(f"RSAG/qWD {phase} record is malformed.")
        if record["generation"] != generation:
            raise CapabilityError(f"RSAG/qWD {phase} launch generation is invalid.")
        if type(record["rank"]) is not int or record["rank"] != rank:
            raise CapabilityError(f"RSAG/qWD {phase} rank identity is invalid.")
        error = record["error"]
        if error is not None and (type(error) is not str or not error):
            raise CapabilityError(f"RSAG/qWD {phase} error is malformed.")
        validated.append(record)
    return tuple(validated)


def _control_errors(records: tuple[dict[str, object], ...]) -> tuple[str, ...]:
    return tuple(
        f"rank {rank}: {record['error']}"
        for rank, record in enumerate(records)
        if record["error"] is not None
    )


def _attestation_from_control_state(value: object) -> RSAGLaunchAttestation:
    if type(value) is not list or len(value) != 6:
        raise CapabilityError("RSAG/qWD control attestation is malformed.")
    try:
        return RSAGLaunchAttestation(*value)
    except (CapabilityError, TypeError, ValueError) as error:
        raise CapabilityError("RSAG/qWD control attestation is invalid.") from error


def _safe_destroy_process_group(distributed: object) -> None:
    try:
        if distributed.is_initialized():
            distributed.destroy_process_group()
    except Exception:
        pass


def _control_generation() -> str:
    """Derive one launcher-owned generation for elastic Store isolation."""
    values: dict[str, str] = {}
    for name in (
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
    ):
        value = os.environ.get(name)
        try:
            _require_exact_string(value, name)
        except ValueError as error:
            raise CapabilityError(
                "RSAG/qWD requires complete torchrun generation metadata."
            ) from error
        values[name] = value
    for name in (
        "TORCHELASTIC_RESTART_COUNT",
        "MASTER_PORT",
        "WORLD_SIZE",
    ):
        value = values[name]
        if not value.isdecimal() or str(int(value)) != value:
            raise CapabilityError(
                "RSAG/qWD torchrun generation metadata is not canonical."
            )
    if int(values["WORLD_SIZE"]) <= 0 or not (1 <= int(values["MASTER_PORT"]) <= 65535):
        raise CapabilityError("RSAG/qWD torchrun generation metadata is out of range.")
    payload = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _hardware_node_identity() -> tuple[str, str]:
    """Return separate physical-node and NVIDIA inventory identities."""
    try:
        dmi_uuid = _read_hardware_identity(_DMI_UUID_PATH)
        boot_id = _read_hardware_identity(_BOOT_ID_PATH)
        information_paths = tuple(sorted(_NVIDIA_GPU_INFO_ROOT.glob("*/information")))
        if not information_paths:
            raise ValueError("NVIDIA GPU inventory is unavailable")
        inventory: list[tuple[str, str, str]] = []
        for path in information_paths:
            fields: dict[str, str] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                name, value = line.split(":", 1)
                normalized_name = name.strip()
                if normalized_name in {"Model", "GPU UUID", "Bus Location"}:
                    normalized_value = value.strip()
                    _require_exact_string(
                        normalized_value,
                        f"NVIDIA {normalized_name}",
                    )
                    if normalized_name in fields:
                        raise ValueError("duplicate NVIDIA inventory field")
                    fields[normalized_name] = normalized_value
            if set(fields) != {"Model", "GPU UUID", "Bus Location"}:
                raise ValueError("NVIDIA GPU inventory is incomplete")
            inventory.append(
                (
                    fields["Model"],
                    fields["GPU UUID"],
                    fields["Bus Location"],
                )
            )
        return (
            _compute_physical_node_id(
                dmi_uuid=dmi_uuid,
                boot_id=boot_id,
            ),
            _compute_gpu_inventory_fingerprint(
                tuple(sorted(inventory)),
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CapabilityError(
            "RSAG/qWD trusted hardware node identity is unavailable."
        ) from error


def _read_hardware_identity(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip().lower()
    _require_exact_string(value, str(path))
    if len(value) > 256 or any(character.isspace() for character in value):
        raise ValueError(f"{path} contains a malformed identity")
    return value


def _compute_physical_node_id(
    *,
    dmi_uuid: str,
    boot_id: str,
) -> str:
    _require_exact_string(dmi_uuid, "dmi_uuid")
    _require_exact_string(boot_id, "boot_id")
    payload = json.dumps(
        {
            "boot_id": boot_id,
            "dmi_uuid": dmi_uuid,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _compute_gpu_inventory_fingerprint(
    gpu_inventory: tuple[tuple[str, str, str], ...],
) -> str:
    if type(gpu_inventory) is not tuple or not gpu_inventory:
        raise ValueError("gpu_inventory must be a non-empty exact tuple")
    for index, device in enumerate(gpu_inventory):
        if type(device) is not tuple or len(device) != 3:
            raise ValueError(f"gpu_inventory[{index}] is invalid")
        for field_index, value in enumerate(device):
            _require_exact_string(
                value,
                f"gpu_inventory[{index}][{field_index}]",
            )
    payload = json.dumps(
        gpu_inventory,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _read_current_launch_attestation() -> RSAGLaunchAttestation:
    return RSAGLaunchAttestation(
        topology_class=os.environ.get(_ATTESTED_TOPOLOGY_ENV),
        transport=os.environ.get(_ATTESTED_TRANSPORT_ENV),
        nccl_ib_disable=os.environ.get("NCCL_IB_DISABLE"),
        nccl_net=os.environ.get("NCCL_NET"),
        nccl_socket_ifname=os.environ.get("NCCL_SOCKET_IFNAME"),
        nccl_p2p_disable=os.environ.get("NCCL_P2P_DISABLE"),
    )


def _launch_attestation_state(
    attestation: RSAGLaunchAttestation,
) -> tuple[object, ...]:
    return (
        attestation.topology_class,
        attestation.transport,
        attestation.nccl_ib_disable,
        attestation.nccl_net,
        attestation.nccl_socket_ifname,
        attestation.nccl_p2p_disable,
    )


def _validate_launch_attestation_values(
    attestation: RSAGLaunchAttestation,
) -> None:
    try:
        _require_exact_string(
            attestation.topology_class,
            _ATTESTED_TOPOLOGY_ENV,
        )
        _require_exact_string(
            attestation.transport,
            _ATTESTED_TRANSPORT_ENV,
        )
        for field_name in (
            "nccl_ib_disable",
            "nccl_net",
            "nccl_socket_ifname",
            "nccl_p2p_disable",
        ):
            value = getattr(attestation, field_name)
            if value is not None:
                _require_exact_string(value, field_name)
    except ValueError as error:
        raise CapabilityError(
            "RSAG/qWD launcher attestation is unavailable."
        ) from error
    if attestation.topology_class in {
        "single_node_pcie",
        "single_node_nvlink",
    }:
        if attestation.transport != "nccl_p2p":
            raise CapabilityError("RSAG/qWD launcher attestation is not normalized.")
        if attestation.nccl_p2p_disable not in {None, "0"}:
            raise CapabilityError(
                "RSAG/qWD launcher attestation conflicts with NCCL P2P."
            )
        return
    if attestation.topology_class == "cross_node_socket":
        prefix = "nccl_socket_"
        if not attestation.transport.startswith(prefix):
            raise CapabilityError("RSAG/qWD launcher attestation is not normalized.")
        interface = attestation.transport[len(prefix) :]
        if not interface or any(
            not (character.isalnum() or character in "_.-") for character in interface
        ):
            raise CapabilityError(
                "RSAG/qWD launcher attestation has an invalid interface."
            )
        if (
            attestation.nccl_ib_disable != "1"
            or attestation.nccl_net != "Socket"
            or attestation.nccl_socket_ifname != f"={interface}"
        ):
            raise CapabilityError(
                "RSAG/qWD attestation requires NCCL_NET=Socket and one exact "
                "NCCL Socket interface."
            )
        if attestation.nccl_p2p_disable not in {None, "0"}:
            raise CapabilityError(
                "RSAG/qWD cross-node qualification requires NCCL P2P."
            )
        return
    raise CapabilityError("RSAG/qWD launcher attestation has an unsupported topology.")


def _validated_rank_identities(
    values: list[object | None],
) -> tuple[dict[str, object], ...]:
    identities: list[dict[str, object]] = []
    errors: list[str] = []
    for rank, value in enumerate(values):
        if type(value) is not dict or set(value) != _RANK_IDENTITY_FIELDS:
            raise CapabilityError("RSAG/qWD gathered rank identity is incomplete.")
        error = value["error"]
        if error is not None:
            if type(error) is not str or not error:
                raise CapabilityError("RSAG/qWD gathered rank error is invalid.")
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
            _require_sha256(
                identity["physical_node_id"],
                f"rank {rank} physical_node_id",
            )
            _require_sha256(
                identity["gpu_inventory_fingerprint"],
                f"rank {rank} gpu_inventory_fingerprint",
            )
            _require_nonnegative_int(
                identity["logical_bytes"],
                f"rank {rank} logical_bytes",
            )
            _require_exact_string(
                identity["requested_topology_class"],
                f"rank {rank} requested_topology_class",
            )
            _require_exact_string(
                identity["requested_transport"],
                f"rank {rank} requested_transport",
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
            for field_name in (
                "nccl_ib_disable",
                "nccl_net",
                "nccl_socket_ifname",
                "nccl_p2p_disable",
            ):
                value = identity[field_name]
                if value is not None:
                    _require_exact_string(
                        value,
                        f"rank {rank} {field_name}",
                    )
    except (OverflowError, ValueError) as error:
        raise CapabilityError("RSAG/qWD gathered rank identity is invalid.") from error
    return tuple(identities)


def _validate_hardware_rank_identities(
    identities: tuple[dict[str, object], ...],
) -> None:
    inventory_by_node: dict[str, str] = {}
    for identity in identities:
        node_id = identity["physical_node_id"]
        fingerprint = identity["gpu_inventory_fingerprint"]
        prior = inventory_by_node.setdefault(node_id, fingerprint)
        if prior != fingerprint:
            raise CapabilityError(
                "RSAG/qWD GPU inventory differs within one physical node."
            )


def _physical_node_count(
    identities: tuple[dict[str, object], ...],
) -> int:
    _validate_hardware_rank_identities(identities)
    return len({identity["physical_node_id"] for identity in identities})


def _validate_plan_preflights(
    identities: tuple[dict[str, object], ...],
) -> None:
    values = tuple(identity["plan_preflight"] for identity in identities)
    if all(value is None for value in values):
        return
    if any(
        type(value) is not dict or set(value) != _PLAN_PREFLIGHT_FIELDS
        for value in values
    ):
        raise CapabilityError("RSAG/qWD plan preflight is incomplete.")
    preflights = values
    errors = tuple(
        f"rank {rank}: {preflight['error']}"
        for rank, preflight in enumerate(preflights)
        if preflight["error"] is not None
    )
    if errors:
        raise CapabilityError("RSAG/qWD plan preflight failed: " + "; ".join(errors))
    reference = preflights[0]
    try:
        _require_sha256(
            reference["qualification_fingerprint"],
            "qualification_fingerprint",
        )
        _require_nonnegative_int(reference["global_numel"], "global_numel")
        for rank, preflight in enumerate(preflights):
            _require_sha256(
                preflight["qualification_fingerprint"],
                "qualification_fingerprint",
            )
            _require_nonnegative_int(preflight["global_numel"], "global_numel")
            if type(preflight["rank"]) is not int or preflight["rank"] != rank:
                raise ValueError("rank does not match collective order")
    except (OverflowError, ValueError) as error:
        raise CapabilityError("RSAG/qWD plan preflight is invalid.") from error
    if any(
        preflight["qualification_fingerprint"] != reference["qualification_fingerprint"]
        or preflight["global_numel"] != reference["global_numel"]
        for preflight in preflights[1:]
    ):
        raise CapabilityError("RSAG/qWD plan preflight is not globally consistent.")


def _plan_preflight_state(value: object | None) -> dict[str, object] | None:
    if value is None:
        return None
    if type(value) is not _RSAGPlanPreflight:
        return {
            "qualification_fingerprint": None,
            "global_numel": None,
            "rank": None,
            "error": "plan_preflight must be exact",
        }
    return {
        "qualification_fingerprint": value.qualification_fingerprint,
        "global_numel": value.global_numel,
        "rank": value.rank,
        "error": value.error,
    }


def _validate_attested_node_count(
    topology_class: object,
    node_count: int,
) -> None:
    if (
        node_count == 1
        and topology_class not in {"single_node_pcie", "single_node_nvlink"}
    ) or (node_count > 1 and topology_class != "cross_node_socket"):
        raise CapabilityError(
            "RSAG/qWD launcher topology attestation conflicts with physical "
            "hardware node identities."
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
    if environment.topology_class != _PRODUCT_TOPOLOGY_CLASS:
        return RouteDecision(
            "native",
            "unsupported_product_topology",
            None,
        )
    if (
        environment.lowbit_comm_version != RSAG_LOWBIT_COMM_VERSION
        or environment.cuda_extension_abi != RSAG_CUDA_EXTENSION_ABI
        or environment.checkpoint_schema_version != RSAG_CHECKPOINT_SCHEMA_VERSION
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
    matching = tuple(record for record in evidence if record.matches(environment))
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

    __slots__ = (
        "_collective_decision",
        "_environment",
        "_evidence",
        "_launch_attestation",
        "_requested",
        "_schedule",
    )

    def __init__(
        self,
        environment: RSAGEnvironment,
        evidence: tuple[RSAGEvidence, ...],
        *,
        requested: str = "auto",
        launch_attestation: RSAGLaunchAttestation | None = None,
    ) -> None:
        validation_request = "native" if requested == "native" else "auto"
        decision = select_rsag_route(
            environment,
            evidence,
            requested=validation_request,
        )
        if requested not in {"auto", "native", "rsag_qwd"}:
            raise ValueError("requested route is invalid")
        self._collective_decision: RouteDecision | None = None
        self._environment = environment
        self._evidence = evidence
        if launch_attestation is not None:
            if type(launch_attestation) is not RSAGLaunchAttestation:
                raise ValueError(
                    "launch_attestation must be an exact RSAGLaunchAttestation"
                )
            launch_attestation.__post_init__()
        self._launch_attestation = launch_attestation
        self._requested = requested
        self._schedule = QWDSchedule(refresh_interval=100)
        if decision != self._local_decision():
            raise CapabilityError("RSAG/qWD decision is not reproducible.")

    @property
    def environment(self) -> RSAGEnvironment:
        """Return the immutable environment bound to this adapter."""
        return self._environment

    @property
    def decision(self) -> RouteDecision:
        """Return only a route admitted by collective qualification."""
        if self._collective_decision is not None:
            return self._collective_decision
        local = self._local_decision()
        if local.uses_rsag:
            return RouteDecision(
                "native",
                "collective_qualification_required",
                local.evidence_schema_version,
            )
        return local

    def _local_decision(self) -> RouteDecision:
        validation_request = "native" if self._requested == "native" else "auto"
        return select_rsag_route(
            self._environment,
            self._evidence,
            requested=validation_request,
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

    def qualify_collectively(
        self,
        process_group: object,
        *,
        rank: int,
    ) -> RouteDecision:
        """Make every rank publish one identical route before branching."""
        if self._collective_decision is not None:
            return self._collective_decision
        logical_bytes = self.environment.logical_bytes
        global_numel = logical_bytes // 2 if type(logical_bytes) is int else None
        preflight = _prepare_plan_preflight(
            self,
            global_numel,
            rank,
            require_collective=False,
        )
        try:
            runtime_environment = detect_rsag_environment(
                process_group,
                logical_bytes=self.environment.logical_bytes,
                topology_class=self.environment.topology_class,
                transport=self.environment.transport,
                launch_attestation=self._launch_attestation,
                plan_preflight=preflight,
            )
        except CapabilityError:
            decision = RouteDecision(
                "native",
                "collective_qualification_failed",
                None,
            )
        else:
            local = self._local_decision()
            decision = (
                local
                if runtime_environment == self.environment and preflight.error is None
                else RouteDecision(
                    "native",
                    "collective_qualification_failed",
                    None,
                )
            )
        self._collective_decision = decision
        return decision

    def create_plans(
        self,
        process_group: object,
        *,
        global_numel: int,
        rank: int,
    ) -> RSAGQWDPlans:
        """Compile CUDA plans only after exact evidence qualification."""
        if self._collective_decision is None or not (
            self._collective_decision.uses_rsag
        ):
            raise CapabilityError("RSAG/qWD plans require collective qualification.")
        preflight = _prepare_plan_preflight(
            self,
            global_numel,
            rank,
            require_collective=True,
        )
        runtime_environment = detect_rsag_environment(
            process_group,
            logical_bytes=self.environment.logical_bytes,
            topology_class=self.environment.topology_class,
            transport=self.environment.transport,
            launch_attestation=self._launch_attestation,
            plan_preflight=preflight,
        )
        if runtime_environment != self.environment:
            raise CapabilityError(
                "RSAG/qWD live runtime identity differs from evidence."
            )
        if preflight.error is not None:
            raise CapabilityError(
                "RSAG/qWD plan preflight failed after collective validation."
            )
        return _create_rsag_qwd_plans(
            process_group,
            global_numel=global_numel,
            world_size=self.environment.world_size,
            rank=rank,
        )


def _prepare_plan_preflight(
    adapter: RSAGQWDAdapter,
    global_numel: object,
    rank: object,
    *,
    require_collective: bool,
) -> _RSAGPlanPreflight:
    safe_global_numel = global_numel if type(global_numel) is int else None
    safe_rank = rank if type(rank) is int else None
    fingerprint: str | None = None
    try:
        decision = adapter._local_decision()
        if not decision.uses_rsag:
            raise CapabilityError("RSAG/qWD plans require a qualified route decision.")
        if require_collective and (
            adapter._collective_decision is None
            or not adapter._collective_decision.uses_rsag
        ):
            raise CapabilityError("RSAG/qWD plans require collective qualification.")
        if adapter._launch_attestation is None:
            raise CapabilityError("RSAG/qWD plans require launch attestation.")
        adapter._launch_attestation.__post_init__()
        _require_nonnegative_int(global_numel, "global_numel")
        if type(rank) is not int or rank < 0 or rank >= adapter.environment.world_size:
            raise ValueError("rank must be within the qualified world_size")
        if _checked_mul(global_numel, 2) != adapter.environment.logical_bytes:
            raise CapabilityError(
                "RSAG/qWD plan size does not match qualified evidence."
            )
        qualification = (
            adapter.environment,
            adapter._evidence,
            adapter._requested,
            decision,
            adapter._launch_attestation,
        )
        fingerprint = sha256(repr(qualification).encode("utf-8")).hexdigest()
        error = None
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    return _RSAGPlanPreflight(
        qualification_fingerprint=fingerprint,
        global_numel=safe_global_numel,
        rank=safe_rank,
        error=error,
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
        if type(candidate) is not _ResidualCandidate or candidate is not self._pending:
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
        "Update from a same-device shard whose finiteness was already proven."
        gradient = _validated_shard_tensor(
            gradient_shard,
            self.layout,
            "gradient_shard",
            device=self.master.device,
            require_finite=False,
        )
        self._step_validated(gradient)

    def _step_from_prevalidated(
        self, source: "ShardedAdamW", gradient: object, *,
        learning_rate: float, weight_decay: object,
    ) -> None:
        """Stage a worker update without copying or mutating committed state.

        The worker has already collectively checked gradient finiteness. Only
        structural checks run here; none reads device values or synchronizes.
        """
        if type(source) is not ShardedAdamW or self.layout != source.layout:
            raise ValueError("candidate source layout mismatch")
        if (self.betas, self.eps, self.weight_decay) != (
            source.betas, source.eps, source.weight_decay
        ):
            raise ValueError("candidate optimizer configuration mismatch")
        learning_rate = _require_finite_float(
            learning_rate, "learning_rate", nonnegative=True,
        )
        tensors = [self.master, self.exp_avg, self.exp_avg_sq,
                   source.master, source.exp_avg, source.exp_avg_sq,
                   gradient, weight_decay]
        for tensor in tensors:
            _validated_shard_tensor(tensor, self.layout, "candidate tensor",
                                    device=source.master.device, require_finite=False)
        # Conservatively reject shared storage, including disjoint views. This
        # catches destination/source, destination/gradient, and cross-state
        # aliases before any mutation, without a device synchronization.
        storage = [(tensor.device, tensor.untyped_storage().data_ptr())
                   if tensor.numel() else None for tensor in tensors]
        for index in range(3):
            if storage[index] is not None and storage[index] in storage[index + 1:]:
                raise ValueError("candidate buffers must not alias state or inputs")
        self.learning_rate = learning_rate
        self._step_validated(gradient, source=source, weight_decay=weight_decay)

    def _step_validated(
        self, gradient: object, *, source: "ShardedAdamW | None" = None,
        weight_decay: object = None,
    ) -> None:
        source = self if source is None else source
        valid = self.layout.valid_numel
        if valid == 0:
            self.step_count = source.step_count + 1
            self._zero_padding(self.master)
            self._zero_padding(self.exp_avg)
            self._zero_padding(self.exp_avg_sq)
            return

        torch = _torch()
        with torch.no_grad():
            master = self.master[:valid]
            exp_avg = self.exp_avg[:valid]
            exp_avg_sq = self.exp_avg_sq[:valid]
            gradient = gradient[:valid]
            beta1, beta2 = self.betas
            self.step_count = source.step_count + 1
            if weight_decay is not None:
                torch.mul(source.master[:valid],
                          1.0 - self.learning_rate * weight_decay[:valid], out=master)
                # Preserve the worker's original per-element decay followed by
                # the optimizer's scalar decay, including its multiply by one.
                master.mul_(1.0 - self.learning_rate * self.weight_decay)
            else:
                torch.mul(source.master[:valid],
                          1.0 - self.learning_rate * self.weight_decay, out=master)
            torch.lerp(source.exp_avg[:valid], gradient, 1.0 - beta1, out=exp_avg)
            torch.mul(source.exp_avg_sq[:valid], beta2, out=exp_avg_sq).addcmul_(
                gradient,
                gradient,
                value=1.0 - beta2,
            )
            bias_correction1 = 1.0 - beta1**self.step_count
            bias_correction2_sqrt = (1.0 - beta2**self.step_count) ** 0.5
            denominator = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
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
        qwd_gathered_payload_bytes=int(qwd_config["qwd_gathered_payload_bytes"]),
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


def _require_zero_padding(value: object, layout: ShardLayout, name: str) -> None:
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
