"""Opt-in experimental adapters that remain outside the stable API."""

from lowbit_comm.experimental.rsag import (
    RSAG_CHECKPOINT_SCHEMA_VERSION,
    RSAG_CUDA_EXTENSION_ABI,
    RSAG_EVIDENCE_SCHEMA_VERSION,
    RSAG_LOWBIT_COMM_VERSION,
    CommittedResidual,
    QWDSchedule,
    RouteDecision,
    RSAGEnvironment,
    RSAGEvidence,
    RSAGQWDAdapter,
    RSAGQWDPlans,
    ShardLayout,
    ShardedAdamW,
    copy_flat_to_parameters,
    detect_rsag_environment,
    flatten_parameter_copy,
    select_rsag_route,
)


__all__ = (
    "RSAG_CHECKPOINT_SCHEMA_VERSION",
    "RSAG_CUDA_EXTENSION_ABI",
    "RSAG_EVIDENCE_SCHEMA_VERSION",
    "RSAG_LOWBIT_COMM_VERSION",
    "CommittedResidual",
    "QWDSchedule",
    "RouteDecision",
    "RSAGEnvironment",
    "RSAGEvidence",
    "RSAGQWDAdapter",
    "RSAGQWDPlans",
    "ShardLayout",
    "ShardedAdamW",
    "copy_flat_to_parameters",
    "detect_rsag_environment",
    "flatten_parameter_copy",
    "select_rsag_route",
)
