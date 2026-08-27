"""Contracts for the installable, fail-closed RSAG/qWD adapter."""

from __future__ import annotations

from dataclasses import replace
from inspect import getsource
from pathlib import Path
from types import SimpleNamespace

import pytest

from lowbit_comm import CapabilityError
from lowbit_comm.experimental import rsag as rsag_module
from lowbit_comm.experimental.rsag import (
    RSAG_CHECKPOINT_SCHEMA_VERSION,
    RSAG_CUDA_EXTENSION_ABI,
    RSAG_EVIDENCE_SCHEMA_VERSION,
    RSAG_LOWBIT_COMM_VERSION,
    QWDSchedule,
    RSAGEnvironment,
    RSAGEvidence,
    RSAGQWDAdapter,
    ShardLayout,
    ShardedAdamW,
    select_rsag_route,
)


def _environment(**changes: object) -> RSAGEnvironment:
    values: dict[str, object] = {
        "world_size": 2,
        "node_count": 1,
        "logical_bytes": 16 * 1024 * 1024,
        "topology_class": "single_node_nvlink",
        "transport": "nccl_p2p",
        "gpu_model": "NVIDIA RTX A6000",
        "torch_version": "2.5.0a0+872d972e41.nv24.08",
        "cuda_version": "12.6",
        "nccl_version": "2.22.3",
        "lowbit_comm_version": "0.4.0.dev0",
        "cuda_extension_abi": 1,
    }
    values.update(changes)
    return RSAGEnvironment(**values)  # type: ignore[arg-type]


def _evidence(**changes: object) -> RSAGEvidence:
    values: dict[str, object] = {
        "schema_version": RSAG_EVIDENCE_SCHEMA_VERSION,
        "world_size": 2,
        "node_count": 1,
        "min_logical_bytes": 8 * 1024 * 1024,
        "max_logical_bytes": 32 * 1024 * 1024,
        "topology_class": "single_node_nvlink",
        "transport": "nccl_p2p",
        "gpu_model": "NVIDIA RTX A6000",
        "torch_version": "2.5.0a0+872d972e41.nv24.08",
        "cuda_version": "12.6",
        "nccl_version": "2.22.3",
        "lowbit_comm_version": "0.4.0.dev0",
        "cuda_extension_abi": 1,
        "seed_speedups_percent": (1.0, 0.2, 0.01),
        "quality_passed": True,
    }
    values.update(changes)
    return RSAGEvidence(**values)  # type: ignore[arg-type]


def test_checkpoint_v2_publishes_layout_and_exact_refresh_state() -> None:
    save_source = getsource(ShardedAdamW.state_dict)
    load_source = getsource(ShardedAdamW.load_state_dict)

    assert RSAG_CHECKPOINT_SCHEMA_VERSION == 2
    assert '"layout": _layout_state(self.layout)' in save_source
    assert '"force_refresh": self.force_refresh' in save_source
    assert 'force_refresh = state["force_refresh"]' in load_source
    assert "self.force_refresh = force_refresh" in load_source
    assert "self.force_refresh = True" not in load_source


def test_exact_positive_evidence_enables_rsag_qwd() -> None:
    decision = select_rsag_route(_environment(), (_evidence(),))

    assert decision.route == "rsag_qwd"
    assert decision.reason == "qualified_positive_evidence"
    assert decision.evidence_schema_version == RSAG_EVIDENCE_SCHEMA_VERSION
    assert decision.uses_rsag is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("world_size", 4),
        ("node_count", 2),
        ("logical_bytes", 7 * 1024 * 1024),
        ("topology_class", "cross_node_socket"),
        ("transport", "nccl_socket"),
        ("gpu_model", "NVIDIA GeForce RTX 4090"),
    ],
)
def test_any_environment_mismatch_falls_back_to_native(
    field: str,
    value: object,
) -> None:
    decision = select_rsag_route(
        replace(_environment(), **{field: value}),
        (_evidence(),),
    )

    assert decision.route == "native"
    assert decision.reason == "no_exact_evidence"
    assert decision.uses_rsag is False


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        (_evidence(seed_speedups_percent=(1.0, 0.0, 0.2)), "nonpositive_seed"),
        (_evidence(seed_speedups_percent=(1.0, -0.1, 0.2)), "nonpositive_seed"),
        (_evidence(quality_passed=False), "quality_failed"),
        (_evidence(schema_version=0), "unsupported_evidence_schema"),
    ],
)
def test_unqualified_evidence_falls_back_to_native(
    evidence: RSAGEvidence,
    reason: str,
) -> None:
    decision = select_rsag_route(_environment(), (evidence,))

    assert decision.route == "native"
    assert decision.reason == reason


@pytest.mark.parametrize(
    "changes",
    [
        {"cuda_extension_abi": 2},
        {"lowbit_comm_version": "0.4.1"},
    ],
)
def test_matching_evidence_for_another_binary_cannot_enable_rsag(
    changes: dict[str, object],
) -> None:
    decision = select_rsag_route(
        replace(_environment(), **changes),
        (replace(_evidence(), **changes),),
    )

    assert decision.route == "native"
    assert decision.reason == "unsupported_binary_identity"


@pytest.mark.parametrize(
    "changes",
    [
        {"torch_version": "2.6.0"},
        {"cuda_version": "12.8"},
        {"nccl_version": "2.23.0"},
    ],
)
def test_matching_evidence_for_an_unverified_runtime_cannot_enable_rsag(
    changes: dict[str, object],
) -> None:
    decision = select_rsag_route(
        replace(_environment(), **changes),
        (replace(_evidence(), **changes),),
    )

    assert decision.route == "native"
    assert decision.reason == "unsupported_runtime_matrix"


@pytest.mark.parametrize(
    "field",
    [
        "topology_class",
        "transport",
        "gpu_model",
        "torch_version",
        "cuda_version",
        "nccl_version",
        "lowbit_comm_version",
    ],
)
def test_unknown_environment_fails_closed(field: str) -> None:
    decision = select_rsag_route(
        replace(_environment(), **{field: "unknown"}),
        (_evidence(),),
    )

    assert decision.route == "native"
    assert decision.reason == "unknown_environment"


def test_duplicate_exact_evidence_is_ambiguous_and_fails_closed() -> None:
    evidence = _evidence()

    decision = select_rsag_route(_environment(), (evidence, evidence))

    assert decision.route == "native"
    assert decision.reason == "ambiguous_evidence"


def test_explicit_rsag_rejects_an_unqualified_environment() -> None:
    with pytest.raises(CapabilityError, match="no_exact_evidence"):
        select_rsag_route(
            _environment(gpu_model="NVIDIA GeForce RTX 4090"),
            (_evidence(),),
            requested="rsag_qwd",
        )


def test_explicit_native_never_selects_compression() -> None:
    decision = select_rsag_route(
        _environment(),
        (_evidence(),),
        requested="native",
    )

    assert decision.route == "native"
    assert decision.reason == "requested_native"


def test_adapter_owns_the_decision_and_refresh_schedule() -> None:
    adapter = RSAGQWDAdapter(_environment(), (_evidence(),))

    assert adapter.decision.route == "rsag_qwd"
    assert adapter.schedule == QWDSchedule(refresh_interval=100)
    assert adapter.mode(0) == "fp_refresh"
    assert adapter.mode(1) == "qwd"


def test_adapter_route_identity_is_read_only() -> None:
    adapter = RSAGQWDAdapter(_environment(), (_evidence(),))

    with pytest.raises(AttributeError):
        adapter.decision = select_rsag_route(_environment(), ())
    with pytest.raises(AttributeError):
        adapter.environment = _environment(gpu_model="forged")


def test_native_adapter_cannot_construct_lowbit_plans() -> None:
    adapter = RSAGQWDAdapter(_environment(), ())

    with pytest.raises(CapabilityError, match="qualified route"):
        adapter.create_plans(object(), global_numel=1, rank=0)


def test_qualified_adapter_rejects_a_size_outside_its_evidence_key() -> None:
    adapter = RSAGQWDAdapter(_environment(), (_evidence(),))

    with pytest.raises(CapabilityError, match="size"):
        adapter.create_plans(object(), global_numel=1, rank=0)


def test_qualified_adapter_rechecks_the_live_runtime_identity(
    monkeypatch,
) -> None:
    environment = _environment()
    adapter = RSAGQWDAdapter(environment, (_evidence(),))
    monkeypatch.setattr(
        rsag_module,
        "detect_rsag_environment",
        lambda *args, **kwargs: replace(
            environment,
            gpu_model="forged runtime",
        ),
    )

    with pytest.raises(CapabilityError, match="runtime identity"):
        adapter.create_plans(
            object(),
            global_numel=environment.logical_bytes // 2,
            rank=0,
        )


def test_environment_detection_normalizes_missing_nccl_to_capability_error(
    monkeypatch,
) -> None:
    distributed = SimpleNamespace(
        is_initialized=lambda: True,
        get_world_size=lambda group: 1,
        all_gather_object=lambda values, value, group: values.__setitem__(
            0,
            value,
        ),
    )
    fake_torch = SimpleNamespace(
        __version__="2.5.0a0+872d972e41.nv24.08",
        version=SimpleNamespace(cuda="12.6"),
        distributed=distributed,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda: "NVIDIA RTX A6000",
            nccl=SimpleNamespace(version=lambda: None),
        ),
    )
    monkeypatch.setattr(rsag_module, "_torch", lambda: fake_torch)

    with pytest.raises(CapabilityError, match="NCCL"):
        rsag_module.detect_rsag_environment(
            object(),
            logical_bytes=1024,
            topology_class="single_node_pcie",
            transport="nccl_p2p",
        )


def test_checkpoint_has_a_versioned_schema_and_rejects_mismatch() -> None:
    torch = pytest.importorskip("torch")
    layout = ShardLayout.build(3, 2, 0)
    optimizer = ShardedAdamW(
        layout,
        torch.tensor([1.0, -2.0], dtype=torch.float32),
        learning_rate=0.1,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    checkpoint = optimizer.state_dict()

    assert RSAG_CHECKPOINT_SCHEMA_VERSION == 2
    assert checkpoint["schema_version"] == RSAG_CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["layout"] == {
        "global_numel": 3,
        "world_size": 2,
        "rank": 0,
        "start": 0,
        "valid_numel": 2,
        "padded_numel": 2,
    }
    assert checkpoint["force_refresh"] is False
    checkpoint["schema_version"] = 0
    with pytest.raises(ValueError, match="schema_version"):
        optimizer.load_state_dict(checkpoint)


@pytest.mark.parametrize(
    "value",
    [True, 0, 1.0, ""],
)
def test_evidence_collection_requires_an_exact_tuple(value: object) -> None:
    with pytest.raises(ValueError, match="evidence"):
        select_rsag_route(_environment(), value)  # type: ignore[arg-type]


def test_empty_evidence_collection_falls_back_to_native() -> None:
    decision = select_rsag_route(_environment(), ())

    assert decision.route == "native"
    assert decision.reason == "no_exact_evidence"


def test_qualification_worker_reuses_packaged_plan_construction() -> None:
    worker = (
        Path(__file__).parents[2]
        / "benchmarks"
        / "distributed_psi_v040_worker.py"
    )
    source = worker.read_text(encoding="utf-8")

    assert "_create_rsag_qwd_plans(" in source
    assert "def _qwd_config(" not in source


def test_binary_identity_constants_match_packaging_and_loader() -> None:
    from lowbit_comm.backends.cuda.loader import CUDA_ABI_VERSION
    from lowbit_comm._version import __version__

    assert RSAG_LOWBIT_COMM_VERSION == __version__
    assert RSAG_CUDA_EXTENSION_ABI == CUDA_ABI_VERSION
