"""Immutable evidence records and deterministic promotion gates."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite

from lowbit_comm.api.intent import CommunicationIntent
from lowbit_comm.api.policy import StrategySpec
from lowbit_comm.core.environment import (
    Dimensions,
    EnvironmentFingerprint,
    _freeze_dimensions,
    _validate_frozen_dimensions,
)
from lowbit_comm.core.errors import CompileError
from lowbit_comm.core.signatures import (
    compression_bit_width,
    dtype_bit_width,
    logical_size_bytes,
    strategy_signature,
    wire_size_bytes,
)


EVIDENCE_SCHEMA_VERSION = 1
COMMUNICATION_REJECT_BELOW_PERCENT = -2.0
COMMUNICATION_LONG_TEST_AT_PERCENT = 5.0
RECOMMENDED_E2E_AT_PERCENT = 5.0
PRODUCTION_AUTO_E2E_AT_PERCENT = 10.0
MAX_QUALITY_LOSS_PERCENT = 1.0
MAX_CONVERGENCE_STEP_INCREASE_PERCENT = 5.0
MAX_WORST_RUN_REGRESSION_PERCENT = 2.0
MIN_SEEDS = 3

_REQUIRED_DIMENSIONS = frozenset(
    {
        "hardware",
        "interconnect",
        "software",
        "nodes",
        "world_size",
        "strategy",
        "topology",
        "output",
        "dtype",
        "logical_bytes",
        "wire_bytes",
        "bucket_min_bytes",
        "bucket_max_bytes",
        "bit_width",
        "group_size",
        "error_feedback",
        "overlap",
        "workload",
    }
)
_REQUEST_DIMENSIONS = _REQUIRED_DIMENSIONS - {
    "hardware",
    "interconnect",
    "software",
}


class EvidenceStatus(str, Enum):
    """Promotion state assigned to one exact evidence unit."""

    REJECTED = "rejected"
    EXPERIMENTAL = "experimental"
    LONG_TEST = "long_test"
    RECOMMENDED = "recommended"
    PRODUCTION_AUTO = "production_auto"


@dataclass(frozen=True, slots=True)
class EvidenceKey:
    """Versioned, exact-match key for one benchmark evidence unit."""

    schema_version: int
    dimensions: Dimensions

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise CompileError("Evidence schema version must be an integer.")
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise CompileError("Evidence schema version must be 1.")
        _validate_frozen_dimensions(self.dimensions, "Evidence")
        missing = _REQUIRED_DIMENSIONS - dict(self.dimensions).keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise CompileError(
                f"Evidence key is missing required dimensions: {names}."
            )

    @classmethod
    def from_mapping(
        cls,
        schema_version: int,
        dimensions: Mapping[str, str],
    ) -> "EvidenceKey":
        """Build a versioned key from an immutable sorted copy."""
        return cls(
            schema_version=schema_version,
            dimensions=_freeze_dimensions(dimensions, "Evidence"),
        )

    @classmethod
    def from_request(
        cls,
        *,
        environment: EnvironmentFingerprint,
        intent: CommunicationIntent,
        strategy: StrategySpec,
        node_count: int,
        workload_class: str,
        bucket_min_bytes: int,
        bucket_max_bytes: int,
    ) -> "EvidenceKey":
        """Derive an exact key solely from immutable request fields.

        ``world_size`` is the collective rank count. The local rank is
        deliberately excluded so every participant derives one selection
        key. Raw benchmark records may carry per-rank measurements later.
        """
        _validate_request_fields(
            environment=environment,
            intent=intent,
            strategy=strategy,
            node_count=node_count,
            workload_class=workload_class,
            bucket_min_bytes=bucket_min_bytes,
            bucket_max_bytes=bucket_max_bytes,
        )
        dimensions = dict(environment.dimensions)
        collisions = dimensions.keys() & _REQUEST_DIMENSIONS
        if collisions:
            names = ", ".join(sorted(collisions))
            raise CompileError(
                f"Environment contains request dimensions: {names}."
            )
        bit_width = compression_bit_width(strategy)
        if bit_width == 0:
            bit_width = dtype_bit_width(intent.tensor.dtype)
        group_size = strategy.group_size
        dimensions.update(
            {
                "nodes": str(node_count),
                # Rank count is keyed; the caller's local rank is not.
                "world_size": str(intent.world_size),
                "strategy": strategy_signature(strategy),
                "topology": strategy.topology.value,
                "output": intent.output.value,
                "dtype": intent.tensor.dtype,
                "logical_bytes": str(logical_size_bytes(intent.tensor)),
                "wire_bytes": str(
                    wire_size_bytes(intent.tensor, strategy)
                ),
                "bucket_min_bytes": str(bucket_min_bytes),
                "bucket_max_bytes": str(bucket_max_bytes),
                "bit_width": str(bit_width),
                "group_size": (
                    "none" if group_size is None else str(group_size)
                ),
                "error_feedback": _bool_dimension(
                    strategy.error_feedback
                ),
                "overlap": _bool_dimension(strategy.overlap),
                "workload": workload_class,
            }
        )
        return cls.from_mapping(
            schema_version=EVIDENCE_SCHEMA_VERSION,
            dimensions=dimensions,
        )


@dataclass(frozen=True, slots=True)
class EvidenceMetrics:
    """Finite measurements used by end-to-end promotion gates."""

    communication_gain_percent: float
    end_to_end_gain_percent: float
    quality_loss_percent: float
    convergence_step_increase_percent: float
    worst_run_gain_percent: float
    seeds: int
    cross_workload_reproduced: bool

    def __post_init__(self) -> None:
        for name in (
            "communication_gain_percent",
            "end_to_end_gain_percent",
            "quality_loss_percent",
            "convergence_step_increase_percent",
            "worst_run_gain_percent",
        ):
            _validate_percentage(getattr(self, name), name)
        if type(self.seeds) is not int or self.seeds <= 0:
            raise CompileError("Evidence seeds must be a positive integer.")
        if type(self.cross_workload_reproduced) is not bool:
            raise CompileError(
                "Cross-workload reproduction must be a boolean."
            )


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Immutable status and metrics for one exact evidence key."""

    key: EvidenceKey
    strategy: StrategySpec
    status: EvidenceStatus
    metrics: EvidenceMetrics

    def __post_init__(self) -> None:
        if type(self.key) is not EvidenceKey:
            raise CompileError("Evidence record key must be an EvidenceKey.")
        if type(self.strategy) is not StrategySpec:
            raise CompileError(
                "Evidence record strategy must be a StrategySpec."
            )
        if type(self.status) is not EvidenceStatus:
            raise CompileError(
                "Evidence record status must be an EvidenceStatus."
            )
        if type(self.metrics) is not EvidenceMetrics:
            raise CompileError(
                "Evidence record metrics must be EvidenceMetrics."
            )
        _validate_record_strategy_key(self.key, self.strategy)


@dataclass(frozen=True, slots=True, init=False)
class EvidenceStore:
    """Immutable exact-key evidence collection used by Production-Auto."""

    records: tuple[EvidenceRecord, ...]

    def __init__(
        self,
        records: Iterable[EvidenceRecord] = (),
    ) -> None:
        try:
            frozen_records = tuple(records)
        except TypeError as error:
            raise CompileError("Evidence records must be iterable.") from error
        seen: set[EvidenceKey] = set()
        for record in frozen_records:
            if type(record) is not EvidenceRecord:
                raise CompileError(
                    "Evidence store entries must be EvidenceRecord values."
                )
            if record.key in seen:
                raise CompileError("Evidence store keys must be unique.")
            seen.add(record.key)
        object.__setattr__(self, "records", frozen_records)

    def production_auto_match(
        self,
        key: EvidenceKey,
    ) -> EvidenceRecord | None:
        """Return only an exactly equal Production-Auto record."""
        if type(key) is not EvidenceKey:
            raise CompileError("Evidence lookup key must be an EvidenceKey.")
        for record in self.records:
            if (
                record.key == key
                and record.status is EvidenceStatus.PRODUCTION_AUTO
            ):
                return record
        return None


def classify_communication_gate(
    *,
    communication_gain_percent: float,
    exposed_communication_gain_percent: float,
) -> EvidenceStatus:
    """Classify whether communication evidence can enter a long test."""
    _validate_percentage(
        communication_gain_percent,
        "communication_gain_percent",
    )
    _validate_percentage(
        exposed_communication_gain_percent,
        "exposed_communication_gain_percent",
    )
    if communication_gain_percent < COMMUNICATION_REJECT_BELOW_PERCENT:
        return EvidenceStatus.REJECTED
    if (
        communication_gain_percent
        >= COMMUNICATION_LONG_TEST_AT_PERCENT
        or exposed_communication_gain_percent > 0.0
    ):
        return EvidenceStatus.LONG_TEST
    return EvidenceStatus.EXPERIMENTAL


def classify_end_to_end_gate(
    metrics: EvidenceMetrics,
) -> EvidenceStatus:
    """Classify exact benchmark metrics using approved promotion gates."""
    if type(metrics) is not EvidenceMetrics:
        raise CompileError("End-to-end gate requires EvidenceMetrics.")
    if not _passes_recommended_requirements(metrics):
        return EvidenceStatus.EXPERIMENTAL
    if (
        metrics.end_to_end_gain_percent
        >= PRODUCTION_AUTO_E2E_AT_PERCENT
        and metrics.cross_workload_reproduced
        and metrics.worst_run_gain_percent
        >= -MAX_WORST_RUN_REGRESSION_PERCENT
    ):
        return EvidenceStatus.PRODUCTION_AUTO
    return EvidenceStatus.RECOMMENDED


def _passes_recommended_requirements(metrics: EvidenceMetrics) -> bool:
    """Return whether all Recommended thresholds are met."""
    return (
        metrics.end_to_end_gain_percent >= RECOMMENDED_E2E_AT_PERCENT
        and metrics.quality_loss_percent <= MAX_QUALITY_LOSS_PERCENT
        and metrics.convergence_step_increase_percent
        <= MAX_CONVERGENCE_STEP_INCREASE_PERCENT
        and metrics.seeds >= MIN_SEEDS
    )


def _validate_record_strategy_key(
    key: EvidenceKey,
    strategy: StrategySpec,
) -> None:
    """Require canonical strategy dimensions to agree with the record."""
    dimensions = dict(key.dimensions)
    bit_width = compression_bit_width(strategy)
    if bit_width == 0:
        bit_width = dtype_bit_width(dimensions["dtype"])
    group_size = strategy.group_size
    expected = {
        "bit_width": str(bit_width),
        "error_feedback": _bool_dimension(strategy.error_feedback),
        "group_size": (
            "none" if group_size is None else str(group_size)
        ),
        "overlap": _bool_dimension(strategy.overlap),
        "strategy": strategy_signature(strategy),
        "topology": strategy.topology.value,
    }
    if any(dimensions[name] != value for name, value in expected.items()):
        raise CompileError(
            "Evidence strategy must agree with its canonical key dimensions."
        )


def _validate_percentage(value: float, name: str) -> None:
    """Require a finite built-in float for a percentage measurement."""
    if type(value) is not float or not isfinite(value):
        raise CompileError(f"Evidence {name} must be a finite float.")


def _validate_request_fields(
    *,
    environment: EnvironmentFingerprint,
    intent: CommunicationIntent,
    strategy: StrategySpec,
    node_count: int,
    workload_class: str,
    bucket_min_bytes: int,
    bucket_max_bytes: int,
) -> None:
    """Validate exact immutable inputs to request key construction."""
    if type(environment) is not EnvironmentFingerprint:
        raise CompileError(
            "Evidence environment must be an EnvironmentFingerprint."
        )
    if type(intent) is not CommunicationIntent:
        raise CompileError("Evidence intent must be a CommunicationIntent.")
    if type(strategy) is not StrategySpec:
        raise CompileError("Evidence strategy must be a StrategySpec.")
    if type(node_count) is not int or node_count <= 0:
        raise CompileError("Evidence node count must be a positive integer.")
    if type(workload_class) is not str or not workload_class:
        raise CompileError("Evidence workload class must be a string.")
    if type(bucket_min_bytes) is not int or bucket_min_bytes < 0:
        raise CompileError(
            "Evidence bucket minimum must be a non-negative integer."
        )
    if type(bucket_max_bytes) is not int or bucket_max_bytes < 0:
        raise CompileError(
            "Evidence bucket maximum must be a non-negative integer."
        )
    if bucket_min_bytes > bucket_max_bytes:
        raise CompileError(
            "Evidence bucket minimum cannot exceed its maximum."
        )


def _bool_dimension(value: bool) -> str:
    """Return the canonical lowercase dimension for an exact boolean."""
    return "true" if value else "false"
