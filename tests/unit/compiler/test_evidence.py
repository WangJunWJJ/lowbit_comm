from dataclasses import FrozenInstanceError, replace
from math import inf, nan

import pytest

from lowbit_comm.api.intent import (
    CommunicationIntent,
    CompletionMode,
    OutputSemantics,
    ReductionOp,
    ShapeFamily,
    TensorSpec,
)
from lowbit_comm.api.policy import (
    AccumulationDType,
    CollectiveKind,
    CompressionKind,
    StrategySpec,
    TopologyKind,
)
from lowbit_comm.compiler.evidence import (
    EnvironmentFingerprint,
    EvidenceKey,
    EvidenceMetrics,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceStore,
    classify_communication_gate,
    classify_end_to_end_gate,
)
from lowbit_comm.core.errors import CompileError


REQUIRED_DIMENSIONS = {
    "hardware": "a6000",
    "interconnect": "pcie4",
    "software": "cu121-torch24-nccl220",
    "nodes": "1",
    "world_size": "4",
    "strategy": "int8-cag-ring",
    "topology": "ring",
    "output": "full_tensor",
    "dtype": "float16",
    "logical_bytes": "16777216",
    "wire_bytes": "4194304",
    "bucket_min_bytes": "4194304",
    "bucket_max_bytes": "16777216",
    "bit_width": "8",
    "group_size": "128",
    "error_feedback": "true",
    "overlap": "true",
    "workload": "communication_bound",
}


def metrics(**overrides: object) -> EvidenceMetrics:
    values: dict[str, object] = {
        "communication_gain_percent": 12.0,
        "end_to_end_gain_percent": 10.0,
        "quality_loss_percent": 0.5,
        "convergence_step_increase_percent": 2.0,
        "worst_run_gain_percent": -2.0,
        "seeds": 3,
        "cross_workload_reproduced": True,
    }
    values.update(overrides)
    return EvidenceMetrics(**values)  # type: ignore[arg-type]


def evidence_strategy() -> StrategySpec:
    return StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.RING,
        group_size=128,
        error_feedback=True,
        overlap=True,
    )


def key_for(bucket_max_bytes: int) -> EvidenceKey:
    dimensions = dict(REQUIRED_DIMENSIONS)
    dimensions["bucket_max_bytes"] = str(bucket_max_bytes)
    return EvidenceKey.from_mapping(
        schema_version=1,
        dimensions=dimensions,
    )


def record_for(
    bucket_max_bytes: int,
    status: EvidenceStatus = EvidenceStatus.PRODUCTION_AUTO,
) -> EvidenceRecord:
    return EvidenceRecord(
        key=key_for(bucket_max_bytes),
        strategy=evidence_strategy(),
        status=status,
        metrics=metrics(),
    )


def compressed_request() -> tuple[
    EnvironmentFingerprint,
    CommunicationIntent,
    StrategySpec,
]:
    environment = EnvironmentFingerprint.from_mapping(
        {
            "software": "test-stack",
            "hardware": "a6000",
            "interconnect": "pcie4",
        }
    )
    intent = CommunicationIntent(
        tensor=TensorSpec(dtype="float16", shape=(256,)),
        shape_family=ShapeFamily(max_numel=256, alignment=1),
        reduction=ReductionOp.MEAN,
        output=OutputSemantics.FULL_TENSOR,
        completion=CompletionMode.ASYNC,
        world_size=4,
        rank=0,
    )
    strategy = StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.RING,
        group_size=128,
        error_feedback=True,
        overlap=True,
    )
    return environment, intent, strategy


def test_communication_regression_over_two_percent_is_rejected() -> None:
    status = classify_communication_gate(
        communication_gain_percent=-2.01,
        exposed_communication_gain_percent=0.0,
    )
    assert status is EvidenceStatus.REJECTED


@pytest.mark.parametrize(
    ("communication_gain", "exposed_gain", "expected"),
    [
        (-2.0, 0.0, EvidenceStatus.EXPERIMENTAL),
        (4.99, 0.0, EvidenceStatus.EXPERIMENTAL),
        (5.0, 0.0, EvidenceStatus.LONG_TEST),
        (-2.01, 0.01, EvidenceStatus.REJECTED),
    ],
)
def test_communication_gate_has_exact_boundaries(
    communication_gain: float,
    exposed_gain: float,
    expected: EvidenceStatus,
) -> None:
    assert classify_communication_gate(
        communication_gain_percent=communication_gain,
        exposed_communication_gain_percent=exposed_gain,
    ) is expected


def test_five_percent_e2e_gain_is_recommended() -> None:
    candidate = metrics(
        communication_gain_percent=8.0,
        end_to_end_gain_percent=5.0,
        quality_loss_percent=1.0,
        convergence_step_increase_percent=5.0,
        worst_run_gain_percent=1.0,
        cross_workload_reproduced=False,
    )
    assert classify_end_to_end_gate(
        candidate
    ) is EvidenceStatus.RECOMMENDED


def test_auto_requires_ten_percent_and_cross_workload_reproduction() -> None:
    assert classify_end_to_end_gate(
        metrics()
    ) is EvidenceStatus.PRODUCTION_AUTO


@pytest.mark.parametrize(
    "overrides",
    [
        {"end_to_end_gain_percent": 4.99},
        {"quality_loss_percent": 1.01},
        {"convergence_step_increase_percent": 5.01},
        {"seeds": 2},
    ],
)
def test_recommended_gate_rejects_values_outside_boundaries(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "end_to_end_gain_percent": 5.0,
        "cross_workload_reproduced": False,
    }
    values.update(overrides)
    candidate = metrics(**values)
    assert classify_end_to_end_gate(
        candidate
    ) is EvidenceStatus.EXPERIMENTAL


def test_e2e_gate_does_not_repeat_the_communication_gate() -> None:
    candidate = metrics(
        communication_gain_percent=4.99,
        end_to_end_gain_percent=5.0,
        cross_workload_reproduced=False,
    )
    assert classify_end_to_end_gate(
        candidate
    ) is EvidenceStatus.RECOMMENDED


def test_production_auto_worst_run_boundary_is_inclusive() -> None:
    assert classify_end_to_end_gate(
        metrics(worst_run_gain_percent=-2.0)
    ) is EvidenceStatus.PRODUCTION_AUTO
    assert classify_end_to_end_gate(
        metrics(worst_run_gain_percent=-2.01)
    ) is EvidenceStatus.RECOMMENDED


def test_metrics_require_finite_exact_percentage_values() -> None:
    for value in (nan, inf, -inf):
        with pytest.raises(CompileError):
            metrics(quality_loss_percent=value)
    with pytest.raises(CompileError):
        metrics(quality_loss_percent=1)


def test_communication_gate_rejects_non_finite_percentages() -> None:
    with pytest.raises(CompileError):
        classify_communication_gate(
            communication_gain_percent=nan,
            exposed_communication_gain_percent=0.0,
        )


def test_environment_fingerprint_is_sorted_immutable_and_hashable() -> None:
    fingerprint = EnvironmentFingerprint.from_mapping(
        {"software": "test", "hardware": "a6000"}
    )
    assert fingerprint.dimensions == (
        ("hardware", "a6000"),
        ("software", "test"),
    )
    assert hash(fingerprint)
    with pytest.raises(FrozenInstanceError):
        fingerprint.dimensions = ()


def test_dimension_mappings_require_exact_strings() -> None:
    with pytest.raises(CompileError):
        EnvironmentFingerprint.from_mapping({"hardware": 1})
    with pytest.raises(CompileError):
        EnvironmentFingerprint.from_mapping({1: "a6000"})


@pytest.mark.parametrize("missing_name", ["workload", "topology"])
def test_evidence_key_requires_schema_one_and_all_dimensions(
    missing_name: str,
) -> None:
    with pytest.raises(CompileError):
        EvidenceKey.from_mapping(
            schema_version=2,
            dimensions=REQUIRED_DIMENSIONS,
        )
    missing = dict(REQUIRED_DIMENSIONS)
    del missing[missing_name]
    with pytest.raises(CompileError):
        EvidenceKey.from_mapping(schema_version=1, dimensions=missing)


def test_topology_is_an_independent_exact_key_dimension() -> None:
    ring_key = key_for(16 * 1024 * 1024)
    tree_dimensions = dict(REQUIRED_DIMENSIONS)
    tree_dimensions["topology"] = "tree"
    tree_key = EvidenceKey.from_mapping(
        schema_version=1,
        dimensions=tree_dimensions,
    )
    assert tree_key != ring_key
    assert dict(tree_key.dimensions)["strategy"] == "int8-cag-ring"


def test_from_request_derives_deterministic_exact_dimensions() -> None:
    environment, intent, strategy = compressed_request()
    key = EvidenceKey.from_request(
        environment=environment,
        intent=intent,
        strategy=strategy,
        node_count=1,
        workload_class="communication_bound",
        bucket_min_bytes=512,
        bucket_max_bytes=1024,
    )
    assert dict(key.dimensions) == {
        "hardware": "a6000",
        "interconnect": "pcie4",
        "software": "test-stack",
        "nodes": "1",
        "world_size": "4",
        "strategy": "int8-cag-ring",
        "topology": "ring",
        "output": "full_tensor",
        "dtype": "float16",
        "logical_bytes": "512",
        "wire_bytes": "264",
        "bucket_min_bytes": "512",
        "bucket_max_bytes": "1024",
        "bit_width": "8",
        "group_size": "128",
        "error_feedback": "true",
        "overlap": "true",
        "workload": "communication_bound",
    }
    assert key == EvidenceKey.from_request(
        environment=environment,
        intent=intent,
        strategy=strategy,
        node_count=1,
        workload_class="communication_bound",
        bucket_min_bytes=512,
        bucket_max_bytes=1024,
    )
    assert hash(key)


def test_from_request_rejects_invalid_exact_primitive_types() -> None:
    environment, intent, strategy = compressed_request()
    with pytest.raises(CompileError):
        EvidenceKey.from_request(
            environment=environment,
            intent=intent,
            strategy=strategy,
            node_count=True,
            workload_class="communication_bound",
            bucket_min_bytes=512,
            bucket_max_bytes=1024,
        )


def test_from_request_rejects_environment_topology_collision() -> None:
    environment, intent, strategy = compressed_request()
    collided = EnvironmentFingerprint.from_mapping(
        dict(environment.dimensions, topology="tree")
    )
    with pytest.raises(CompileError):
        EvidenceKey.from_request(
            environment=collided,
            intent=intent,
            strategy=strategy,
            node_count=1,
            workload_class="communication_bound",
            bucket_min_bytes=512,
            bucket_max_bytes=1024,
        )


def test_from_request_populates_topology_from_strategy() -> None:
    environment, intent, strategy = compressed_request()
    tree_key = EvidenceKey.from_request(
        environment=environment,
        intent=intent,
        strategy=replace(strategy, topology=TopologyKind.TREE),
        node_count=1,
        workload_class="communication_bound",
        bucket_min_bytes=512,
        bucket_max_bytes=1024,
    )
    assert dict(tree_key.dimensions)["topology"] == "tree"


def test_world_size_is_rank_count_but_local_rank_is_not_keyed() -> None:
    environment, intent, strategy = compressed_request()

    def derive(candidate: CommunicationIntent) -> EvidenceKey:
        return EvidenceKey.from_request(
            environment=environment,
            intent=candidate,
            strategy=strategy,
            node_count=1,
            workload_class="communication_bound",
            bucket_min_bytes=512,
            bucket_max_bytes=1024,
        )

    baseline = derive(intent)
    assert derive(replace(intent, world_size=8, rank=0)) != baseline
    assert derive(replace(intent, rank=1)) == baseline


def test_from_request_distinguishes_all_strategy_configuration() -> None:
    environment, intent, strategy = compressed_request()

    def derive(candidate: StrategySpec) -> EvidenceKey:
        return EvidenceKey.from_request(
            environment=environment,
            intent=intent,
            strategy=candidate,
            node_count=1,
            workload_class="communication_bound",
            bucket_min_bytes=512,
            bucket_max_bytes=1024,
        )

    baseline = derive(strategy)
    alternatives = (
        replace(
            strategy,
            accumulation_dtype=AccumulationDType.FP16,
        ),
        replace(strategy, parameter_error_feedback=True),
        replace(strategy, workspace_budget_bytes=1024),
    )
    assert all(derive(candidate) != baseline for candidate in alternatives)


def test_evidence_store_never_uses_nearest_neighbor_for_auto() -> None:
    store = EvidenceStore([record_for(16 * 1024 * 1024)])
    requested = key_for(16 * 1024 * 1024 + 1)
    assert store.production_auto_match(requested) is None


def test_evidence_store_only_returns_production_auto_records() -> None:
    rejected = record_for(
        16 * 1024 * 1024,
        status=EvidenceStatus.RECOMMENDED,
    )
    store = EvidenceStore([rejected])
    assert store.production_auto_match(rejected.key) is None


def test_evidence_record_is_immutable_and_hashable() -> None:
    record = record_for(16 * 1024 * 1024)
    assert hash(record)
    with pytest.raises(FrozenInstanceError):
        record.status = EvidenceStatus.REJECTED


def test_evidence_record_requires_an_exact_strategy_contract() -> None:
    with pytest.raises(CompileError):
        EvidenceRecord(
            key=key_for(16 * 1024 * 1024),
            strategy=object(),  # type: ignore[arg-type]
            status=EvidenceStatus.PRODUCTION_AUTO,
            metrics=metrics(),
        )


@pytest.mark.parametrize(
    "strategy",
    [
        StrategySpec(
            compression=CompressionKind.NONE,
            collective=CollectiveKind.NATIVE,
            topology=TopologyKind.BACKEND_DEFAULT,
        ),
        replace(evidence_strategy(), topology=TopologyKind.TREE),
        replace(evidence_strategy(), group_size=64),
        replace(
            evidence_strategy(),
            accumulation_dtype=AccumulationDType.FP16,
        ),
        replace(evidence_strategy(), error_feedback=False),
        replace(evidence_strategy(), parameter_error_feedback=True),
        replace(evidence_strategy(), overlap=False),
        replace(evidence_strategy(), workspace_budget_bytes=1024),
    ],
)
def test_evidence_record_rejects_strategy_key_disagreement(
    strategy: StrategySpec,
) -> None:
    with pytest.raises(CompileError):
        EvidenceRecord(
            key=key_for(16 * 1024 * 1024),
            strategy=strategy,
            status=EvidenceStatus.PRODUCTION_AUTO,
            metrics=metrics(),
        )


def test_evidence_record_rejects_bit_width_disagreement() -> None:
    dimensions = dict(REQUIRED_DIMENSIONS)
    dimensions["bit_width"] = "4"
    key = EvidenceKey.from_mapping(
        schema_version=1,
        dimensions=dimensions,
    )

    with pytest.raises(CompileError):
        EvidenceRecord(
            key=key,
            strategy=evidence_strategy(),
            status=EvidenceStatus.PRODUCTION_AUTO,
            metrics=metrics(),
        )
