from dataclasses import FrozenInstanceError, fields, replace
from math import inf, nan

import pytest

import lowbit_comm.compiler.evidence as evidence_module
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
    LegacyEvidenceMetrics,
    LegacyEvidenceRecord,
    classify_communication_gate,
    classify_end_to_end_gate,
    derive_evidence_status,
)
from lowbit_comm.core.errors import CompileError
from lowbit_comm.core.signatures import (
    _COLLECTIVE_INTENT_FIELDS,
    _LOCAL_INTENT_FIELDS,
    intent_signature,
)


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

CURRENT_INTENT_DIMENSIONS = {
    "completion",
    "intent_signature",
    "reduction",
    "shape_family_alignment",
    "shape_family_max_numel",
    "tensor_shape",
}


def metrics(**overrides: object) -> EvidenceMetrics:
    values: dict[str, object] = {
        "communication_gain_percent": 12.0,
        "exposed_communication_gain_percent": 1.0,
        "end_to_end_gain_percent": 10.0,
        "quality_loss_percent": 0.5,
        "convergence_step_increase_percent": 2.0,
        "worst_run_gain_percent": -2.0,
        "seeds": 3,
        "cross_workload_reproduced": True,
    }
    values.update(overrides)
    return EvidenceMetrics(**values)  # type: ignore[arg-type]


def legacy_metrics() -> LegacyEvidenceMetrics:
    return LegacyEvidenceMetrics(
        communication_gain_percent=12.0,
        end_to_end_gain_percent=10.0,
        quality_loss_percent=0.5,
        convergence_step_increase_percent=2.0,
        worst_run_gain_percent=-2.0,
        seeds=3,
        cross_workload_reproduced=True,
    )


def evidence_strategy() -> StrategySpec:
    return StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.RING,
        group_size=128,
        error_feedback=True,
        overlap=True,
    )


def key_for(
    bucket_max_bytes: int,
    *,
    schema_version: int = 2,
) -> EvidenceKey:
    dimensions = dict(REQUIRED_DIMENSIONS)
    if schema_version == 2:
        _, intent, _ = compressed_request()
        dimensions.update(
            {
                "completion": intent.completion.value,
                "intent_signature": intent_signature(intent),
                "reduction": intent.reduction.value,
                "shape_family_alignment": str(
                    intent.shape_family.alignment
                ),
                "shape_family_max_numel": str(
                    intent.shape_family.max_numel
                ),
                "tensor_shape": "[256]",
            }
        )
    dimensions["bucket_max_bytes"] = str(bucket_max_bytes)
    return EvidenceKey.from_mapping(
        schema_version=schema_version,
        dimensions=dimensions,
    )


def record_for(
    bucket_max_bytes: int,
    status: EvidenceStatus = EvidenceStatus.PRODUCTION_AUTO,
    record_metrics: EvidenceMetrics | None = None,
    *,
    schema_version: int = 2,
) -> EvidenceRecord:
    return EvidenceRecord(
        key=key_for(
            bucket_max_bytes,
            schema_version=schema_version,
        ),
        strategy=evidence_strategy(),
        status=status,
        metrics=metrics() if record_metrics is None else record_metrics,
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
    ("overrides", "expected"),
    [
        (
            {
                "communication_gain_percent": -2.01,
                "exposed_communication_gain_percent": 10.0,
            },
            EvidenceStatus.REJECTED,
        ),
        (
            {
                "communication_gain_percent": 4.99,
                "exposed_communication_gain_percent": 0.0,
            },
            EvidenceStatus.EXPERIMENTAL,
        ),
        (
            {
                "communication_gain_percent": 4.99,
                "exposed_communication_gain_percent": 0.01,
                "end_to_end_gain_percent": 4.99,
            },
            EvidenceStatus.LONG_TEST,
        ),
        (
            {
                "communication_gain_percent": 5.0,
                "exposed_communication_gain_percent": 0.0,
                "end_to_end_gain_percent": 5.0,
                "cross_workload_reproduced": False,
            },
            EvidenceStatus.RECOMMENDED,
        ),
    ],
)
def test_derived_status_composes_communication_and_e2e_gates(
    overrides: dict[str, object],
    expected: EvidenceStatus,
) -> None:
    assert derive_evidence_status(metrics(**overrides)) is expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"end_to_end_gain_percent": 9.99},
        {"quality_loss_percent": 1.01},
        {"convergence_step_increase_percent": 5.01},
        {"seeds": 2},
        {"cross_workload_reproduced": False},
        {"worst_run_gain_percent": -2.01},
    ],
)
def test_derived_production_status_requires_every_e2e_gate(
    overrides: dict[str, object],
) -> None:
    assert (
        derive_evidence_status(metrics(**overrides))
        is not EvidenceStatus.PRODUCTION_AUTO
    )


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


class FloatSubclass(float):
    pass


@pytest.mark.parametrize(
    "value",
    [0, True, FloatSubclass(0.0), nan, inf, -inf],
)
def test_exposed_gain_requires_an_exact_finite_float(
    value: object,
) -> None:
    with pytest.raises(CompileError):
        metrics(exposed_communication_gain_percent=value)


@pytest.mark.parametrize("value", [True, 1.0])
def test_seeds_require_an_exact_integer(value: object) -> None:
    with pytest.raises(CompileError):
        metrics(seeds=value)


@pytest.mark.parametrize("value", [0, 1])
def test_cross_workload_requires_an_exact_boolean(value: object) -> None:
    with pytest.raises(CompileError):
        metrics(cross_workload_reproduced=value)


def test_derived_status_requires_exact_metrics() -> None:
    class MetricsSubclass(EvidenceMetrics):
        pass

    with pytest.raises(CompileError):
        derive_evidence_status(
            MetricsSubclass(
                communication_gain_percent=12.0,
                exposed_communication_gain_percent=1.0,
                end_to_end_gain_percent=10.0,
                quality_loss_percent=0.5,
                convergence_step_increase_percent=2.0,
                worst_run_gain_percent=-2.0,
                seeds=3,
                cross_workload_reproduced=True,
            )
        )


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
def test_evidence_key_requires_a_supported_schema_and_all_dimensions(
    missing_name: str,
) -> None:
    with pytest.raises(CompileError):
        EvidenceKey.from_mapping(
            schema_version=3,
            dimensions=REQUIRED_DIMENSIONS,
        )
    missing = dict(REQUIRED_DIMENSIONS)
    del missing[missing_name]
    with pytest.raises(CompileError):
        EvidenceKey.from_mapping(schema_version=1, dimensions=missing)
    assert EvidenceKey.from_mapping(
        schema_version=1,
        dimensions=REQUIRED_DIMENSIONS,
    ).schema_version == 1


def test_schema_two_rejects_a_literal_schema_one_key() -> None:
    with pytest.raises(CompileError, match="intent_signature"):
        EvidenceKey.from_mapping(
            schema_version=2,
            dimensions=REQUIRED_DIMENSIONS,
        )


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
    dimensions = dict(key.dimensions)
    assert dimensions == {
        "hardware": "a6000",
        "interconnect": "pcie4",
        "software": "test-stack",
        "nodes": "1",
        "world_size": "4",
        "intent_signature": intent_signature(intent),
        "strategy": "int8-cag-ring",
        "topology": "ring",
        "output": "full_tensor",
        "dtype": "float16",
        "tensor_shape": "[256]",
        "shape_family_max_numel": "256",
        "shape_family_alignment": "1",
        "reduction": "mean",
        "completion": "async",
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
    assert key.schema_version == 2
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


@pytest.mark.parametrize("missing_name", sorted(CURRENT_INTENT_DIMENSIONS))
def test_schema_two_requires_every_current_intent_dimension(
    missing_name: str,
) -> None:
    environment, intent, strategy = compressed_request()
    dimensions = dict(
        EvidenceKey.from_request(
            environment=environment,
            intent=intent,
            strategy=strategy,
            node_count=1,
            workload_class="communication_bound",
            bucket_min_bytes=512,
            bucket_max_bytes=1024,
        ).dimensions
    )
    dimensions.pop(missing_name, None)

    with pytest.raises(CompileError, match=missing_name):
        EvidenceKey.from_mapping(
            schema_version=2,
            dimensions=dimensions,
        )


def test_intent_signature_field_policy_excludes_only_local_rank() -> None:
    intent_fields = {field.name for field in fields(CommunicationIntent)}

    assert _LOCAL_INTENT_FIELDS == frozenset({"rank"})
    assert _COLLECTIVE_INTENT_FIELDS | _LOCAL_INTENT_FIELDS == intent_fields
    assert not _COLLECTIVE_INTENT_FIELDS & _LOCAL_INTENT_FIELDS


def test_schema_two_classifies_every_strategy_field() -> None:
    assert evidence_module._STRATEGY_DIMENSIONS_BY_FIELD == {
        "compression": frozenset(
            {"bit_width", "strategy", "wire_bytes"}
        ),
        "collective": frozenset({"strategy"}),
        "topology": frozenset({"strategy", "topology"}),
        "group_size": frozenset({"group_size", "wire_bytes"}),
        "accumulation_dtype": frozenset({"strategy"}),
        "error_feedback": frozenset({"error_feedback"}),
        "parameter_error_feedback": frozenset({"strategy"}),
        "overlap": frozenset({"overlap"}),
        "workspace_budget_bytes": frozenset({"strategy"}),
    }
    assert set(evidence_module._STRATEGY_DIMENSIONS_BY_FIELD) == {
        field.name for field in fields(StrategySpec)
    }


@pytest.mark.parametrize(
    "drift",
    [
        "missing_field",
        "unknown_field",
        "empty_dimensions",
        "unknown_dimension",
    ],
)
def test_schema_two_strategy_classification_fails_closed(
    drift: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classifier = dict(evidence_module._STRATEGY_DIMENSIONS_BY_FIELD)
    if drift == "missing_field":
        classifier.pop("overlap")
    elif drift == "unknown_field":
        classifier["future_field"] = frozenset({"strategy"})
    elif drift == "empty_dimensions":
        classifier["group_size"] = frozenset()
    else:
        classifier["group_size"] = frozenset({"future_dimension"})
    monkeypatch.setattr(
        evidence_module,
        "_STRATEGY_DIMENSIONS_BY_FIELD",
        classifier,
    )
    environment, request_intent, request_strategy = compressed_request()

    with pytest.raises(CompileError, match="strategy.*classification"):
        EvidenceKey.from_request(
            environment=environment,
            intent=request_intent,
            strategy=request_strategy,
            node_count=1,
            workload_class="communication_bound",
            bucket_min_bytes=512,
            bucket_max_bytes=1024,
        )


@pytest.mark.parametrize(
    "alternative",
    [
        CommunicationIntent(
            tensor=TensorSpec(dtype="float32", shape=(256,)),
            shape_family=ShapeFamily(max_numel=256, alignment=1),
            reduction=ReductionOp.MEAN,
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.ASYNC,
            world_size=4,
            rank=0,
        ),
        CommunicationIntent(
            tensor=TensorSpec(dtype="float16", shape=(16, 16)),
            shape_family=ShapeFamily(max_numel=256, alignment=1),
            reduction=ReductionOp.MEAN,
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.ASYNC,
            world_size=4,
            rank=0,
        ),
        CommunicationIntent(
            tensor=TensorSpec(dtype="float16", shape=(256,)),
            shape_family=ShapeFamily(max_numel=512, alignment=1),
            reduction=ReductionOp.MEAN,
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.ASYNC,
            world_size=4,
            rank=0,
        ),
        CommunicationIntent(
            tensor=TensorSpec(dtype="float16", shape=(256,)),
            shape_family=ShapeFamily(max_numel=256, alignment=2),
            reduction=ReductionOp.MEAN,
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.ASYNC,
            world_size=4,
            rank=0,
        ),
        CommunicationIntent(
            tensor=TensorSpec(dtype="float16", shape=(256,)),
            shape_family=ShapeFamily(max_numel=256, alignment=1),
            reduction=ReductionOp.SUM,
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.ASYNC,
            world_size=4,
            rank=0,
        ),
        CommunicationIntent(
            tensor=TensorSpec(dtype="float16", shape=(256,)),
            shape_family=ShapeFamily(max_numel=256, alignment=1),
            reduction=ReductionOp.MEAN,
            output=OutputSemantics.REDUCED_SHARD,
            completion=CompletionMode.ASYNC,
            world_size=4,
            rank=0,
        ),
        CommunicationIntent(
            tensor=TensorSpec(dtype="float16", shape=(256,)),
            shape_family=ShapeFamily(max_numel=256, alignment=1),
            reduction=ReductionOp.MEAN,
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.SYNC,
            world_size=4,
            rank=0,
        ),
        CommunicationIntent(
            tensor=TensorSpec(dtype="float16", shape=(256,)),
            shape_family=ShapeFamily(max_numel=256, alignment=1),
            reduction=ReductionOp.MEAN,
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.ASYNC,
            world_size=8,
            rank=0,
        ),
    ],
)
def test_collective_intent_signature_binds_every_shared_field(
    alternative: CommunicationIntent,
) -> None:
    _, intent, _ = compressed_request()

    assert intent_signature(alternative) != intent_signature(intent)


def test_collective_intent_signature_excludes_local_rank() -> None:
    _, intent, _ = compressed_request()

    assert intent_signature(replace(intent, rank=1)) == (
        intent_signature(intent)
    )


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


@pytest.mark.parametrize("dimension", sorted(CURRENT_INTENT_DIMENSIONS))
def test_from_request_rejects_environment_intent_dimension_collisions(
    dimension: str,
) -> None:
    environment, intent, strategy = compressed_request()
    collided = EnvironmentFingerprint.from_mapping(
        dict(environment.dimensions, **{dimension: "forged"})
    )
    with pytest.raises(CompileError, match=dimension):
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
        record_metrics=metrics(
            end_to_end_gain_percent=5.0,
            cross_workload_reproduced=False,
        ),
    )
    store = EvidenceStore([rejected])
    assert store.production_auto_match(rejected.key) is None


def test_lookup_rejects_every_fresh_valid_duplicate_key_in_both_orders(
) -> None:
    first = record_for(16 * 1024 * 1024)
    second = record_for(16 * 1024 * 1024 + 1)
    forward = EvidenceStore([first, second])
    reverse = EvidenceStore([second, first])
    object.__setattr__(second, "key", first.key)

    assert forward.production_auto_match(first.key) is None
    assert reverse.production_auto_match(first.key) is None


def test_duplicate_key_group_does_not_hide_another_unique_record() -> None:
    first = record_for(16 * 1024 * 1024)
    second = record_for(16 * 1024 * 1024 + 1)
    unique = record_for(16 * 1024 * 1024 + 2)
    store = EvidenceStore([first, second, unique])
    object.__setattr__(second, "key", first.key)

    assert store.production_auto_match(first.key) is None
    assert store.production_auto_match(unique.key) is unique


def test_invalid_current_record_does_not_poison_a_unique_valid_key() -> None:
    valid = record_for(16 * 1024 * 1024)
    invalid = record_for(16 * 1024 * 1024 + 1)
    store = EvidenceStore([invalid, valid])
    object.__setattr__(invalid, "key", valid.key)
    object.__setattr__(invalid.metrics, "quality_loss_percent", 5.0)

    assert store.production_auto_match(valid.key) is valid


def test_evidence_record_is_immutable_and_hashable() -> None:
    record = record_for(16 * 1024 * 1024)
    assert hash(record)
    with pytest.raises(FrozenInstanceError):
        record.status = EvidenceStatus.REJECTED


def test_record_rejects_a_supplied_status_that_is_not_derived() -> None:
    with pytest.raises(CompileError, match="derived"):
        record_for(
            16 * 1024 * 1024,
            status=EvidenceStatus.PRODUCTION_AUTO,
            record_metrics=metrics(end_to_end_gain_percent=0.0),
        )


@pytest.mark.parametrize(
    "status",
    ["production_auto", True, 1],
)
def test_record_status_requires_the_exact_enum(status: object) -> None:
    with pytest.raises(CompileError, match="EvidenceStatus"):
        EvidenceRecord(
            key=key_for(16 * 1024 * 1024),
            strategy=evidence_strategy(),
            status=status,  # type: ignore[arg-type]
            metrics=metrics(),
        )


def test_current_record_subclass_fails_at_construction() -> None:
    class EvidenceRecordSubclass(EvidenceRecord):
        pass

    with pytest.raises(CompileError, match="EvidenceRecord"):
        EvidenceRecordSubclass(
            key=key_for(16 * 1024 * 1024),
            strategy=evidence_strategy(),
            status=EvidenceStatus.PRODUCTION_AUTO,
            metrics=metrics(),
        )


def test_legacy_record_subclass_fails_at_construction() -> None:
    class LegacyEvidenceRecordSubclass(LegacyEvidenceRecord):
        pass

    with pytest.raises(CompileError, match="LegacyEvidenceRecord"):
        LegacyEvidenceRecordSubclass(
            key=key_for(16 * 1024 * 1024, schema_version=1),
            strategy=evidence_strategy(),
            status=EvidenceStatus.PRODUCTION_AUTO,
            metrics=legacy_metrics(),
        )


def test_legacy_schema_one_record_is_diagnostic_only() -> None:
    legacy = LegacyEvidenceRecord(
        key=key_for(
            16 * 1024 * 1024,
            schema_version=1,
        ),
        strategy=evidence_strategy(),
        status=EvidenceStatus.PRODUCTION_AUTO,
        metrics=legacy_metrics(),
    )
    store = EvidenceStore([legacy])

    assert store.records == (legacy,)
    assert store.production_auto_match(legacy.key) is None


def test_current_record_rejects_a_legacy_key() -> None:
    with pytest.raises(CompileError, match="schema version 2"):
        record_for(
            16 * 1024 * 1024,
            schema_version=1,
        )


def test_legacy_record_rejects_a_current_key() -> None:
    with pytest.raises(CompileError, match="schema version 1"):
        LegacyEvidenceRecord(
            key=key_for(16 * 1024 * 1024),
            strategy=evidence_strategy(),
            status=EvidenceStatus.PRODUCTION_AUTO,
            metrics=legacy_metrics(),
        )


def test_store_revalidates_a_forged_frozen_record() -> None:
    forged = record_for(16 * 1024 * 1024)
    object.__setattr__(forged, "status", EvidenceStatus.REJECTED)

    with pytest.raises(CompileError, match="derived"):
        EvidenceStore([forged])


def test_lookup_revalidates_a_record_forged_after_store_creation() -> None:
    record = record_for(16 * 1024 * 1024)
    store = EvidenceStore([record])
    object.__setattr__(record.metrics, "quality_loss_percent", 5.0)

    assert store.production_auto_match(record.key) is None


def test_lookup_rejects_a_forged_metric_type() -> None:
    record = record_for(16 * 1024 * 1024)
    store = EvidenceStore([record])
    object.__setattr__(record.metrics, "seeds", True)

    assert store.production_auto_match(record.key) is None


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
    dimensions = dict(key_for(16 * 1024 * 1024).dimensions)
    dimensions["bit_width"] = "4"
    key = EvidenceKey.from_mapping(
        schema_version=2,
        dimensions=dimensions,
    )

    with pytest.raises(CompileError):
        EvidenceRecord(
            key=key,
            strategy=evidence_strategy(),
            status=EvidenceStatus.PRODUCTION_AUTO,
            metrics=metrics(),
        )
