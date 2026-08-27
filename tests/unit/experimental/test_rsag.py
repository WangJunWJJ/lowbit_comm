"""Contracts for the installable, fail-closed RSAG/qWD adapter."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from inspect import getsource
import json
import os
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


class _CloneValue:
    def __init__(self, value: int) -> None:
        self.value = value
        self.device = None

    def clone(self) -> _CloneValue:
        return _CloneValue(self.value)


class _ExplodingDeepcopy:
    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise RuntimeError("injected deepcopy failure")


_TEST_PROCESS_GROUP = object()


@pytest.fixture(autouse=True)
def _reset_rsag_launch_state(monkeypatch):
    rsag_module._CAPTURED_LAUNCH_ATTESTATIONS.clear()
    monkeypatch.setattr(rsag_module, "_RSAG_LAUNCH_ATTEMPTED", False)
    yield
    rsag_module._CAPTURED_LAUNCH_ATTESTATIONS.clear()


def _launch_attestation(
    *,
    topology_class: str = "cross_node_socket",
    transport: str = "nccl_socket_eno2",
    nccl_ib_disable: str | None = "1",
    nccl_net: str | None = None,
    nccl_socket_ifname: str | None = "=eno2",
) -> object:
    attestation_type = getattr(rsag_module, "RSAGLaunchAttestation", None)
    assert attestation_type is not None
    if topology_class == "cross_node_socket" and nccl_net is None:
        nccl_net = "Socket"
    elif topology_class != "cross_node_socket":
        if transport == "nccl_socket_eno2":
            transport = "nccl_p2p"
        if nccl_ib_disable == "1":
            nccl_ib_disable = None
        if nccl_socket_ifname == "=eno2":
            nccl_socket_ifname = None
    return attestation_type(
        topology_class=topology_class,
        transport=transport,
        nccl_ib_disable=nccl_ib_disable,
        nccl_net=nccl_net,
        nccl_socket_ifname=nccl_socket_ifname,
        nccl_p2p_disable=None,
    )


def _captured_launch_attestation(
    monkeypatch,
    *,
    topology_class: str = "cross_node_socket",
    transport: str = "nccl_socket_eno2",
    nccl_ib_disable: str | None = "1",
    nccl_net: str | None = None,
    nccl_socket_ifname: str | None = "=eno2",
) -> object:
    if topology_class == "cross_node_socket" and nccl_net is None:
        nccl_net = "Socket"
    elif topology_class != "cross_node_socket":
        if transport == "nccl_socket_eno2":
            transport = "nccl_p2p"
        if nccl_ib_disable == "1":
            nccl_ib_disable = None
        if nccl_socket_ifname == "=eno2":
            nccl_socket_ifname = None
    values = {
        "LOWBIT_COMM_RSAG_ATTESTED_TOPOLOGY": topology_class,
        "LOWBIT_COMM_RSAG_ATTESTED_TRANSPORT": transport,
        "NCCL_IB_DISABLE": nccl_ib_disable,
        "NCCL_NET": nccl_net,
        "NCCL_SOCKET_IFNAME": nccl_socket_ifname,
        "NCCL_P2P_DISABLE": None,
    }
    for name, value in values.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with monkeypatch.context() as capture_patch:
        initialized = False

        def init_process_group(**kwargs: object) -> None:
            nonlocal initialized
            assert kwargs["backend"] == "nccl"
            initialized = True

        distributed = SimpleNamespace(
            group=SimpleNamespace(WORLD=_TEST_PROCESS_GROUP),
            is_initialized=lambda: initialized,
            init_process_group=init_process_group,
            destroy_process_group=lambda: None,
        )
        capture_patch.setattr(
            rsag_module,
            "_torch",
            lambda: SimpleNamespace(distributed=distributed),
        )
        capture_patch.setattr(
            rsag_module,
            "_rendezvous_launch_control",
            lambda *args, **kwargs: (
                _launch_attestation(
                    topology_class=topology_class,
                    transport=transport,
                    nccl_ib_disable=nccl_ib_disable,
                    nccl_net=nccl_net,
                    nccl_socket_ifname=nccl_socket_ifname,
                ),
                object(),
                0,
                2,
                "a" * 64,
                "b" * 64,
                "c" * 64,
            ),
        )
        capture_patch.setattr(
            rsag_module,
            "_exchange_control_records",
            lambda store, phase, local, rank, world_size, generation: (
                {**local, "generation": generation},
                {**local, "rank": 1, "generation": generation},
            ),
        )
        return rsag_module.initialize_rsag_process_group(backend="nccl")


def _environment(**changes: object) -> RSAGEnvironment:
    values: dict[str, object] = {
        "world_size": 2,
        "node_count": 2,
        "logical_bytes": 16 * 1024 * 1024,
        "topology_class": "cross_node_socket",
        "transport": "nccl_socket_eno2",
        "gpu_model": "NVIDIA RTX A6000",
        "torch_version": "2.5.0a0+872d972e41.nv24.08",
        "cuda_version": "12.6",
        "nccl_version": "2.22.3",
        "lowbit_comm_version": "0.4.0.dev0",
        "cuda_extension_abi": 1,
        "checkpoint_schema_version": RSAG_CHECKPOINT_SCHEMA_VERSION,
        "build_fingerprint": "a" * 64,
    }
    values.update(changes)
    return RSAGEnvironment(**values)  # type: ignore[arg-type]


def _evidence(**changes: object) -> RSAGEvidence:
    values: dict[str, object] = {
        "schema_version": RSAG_EVIDENCE_SCHEMA_VERSION,
        "world_size": 2,
        "node_count": 2,
        "min_logical_bytes": 16 * 1024 * 1024,
        "max_logical_bytes": 16 * 1024 * 1024,
        "topology_class": "cross_node_socket",
        "transport": "nccl_socket_eno2",
        "gpu_model": "NVIDIA RTX A6000",
        "torch_version": "2.5.0a0+872d972e41.nv24.08",
        "cuda_version": "12.6",
        "nccl_version": "2.22.3",
        "lowbit_comm_version": "0.4.0.dev0",
        "cuda_extension_abi": 1,
        "checkpoint_schema_version": RSAG_CHECKPOINT_SCHEMA_VERSION,
        "build_fingerprint": "a" * 64,
        "seed_speedups_percent": (1.0, 0.2, 0.01),
        "quality_passed": True,
    }
    values.update(changes)
    return RSAGEvidence(**values)  # type: ignore[arg-type]


def _launch_control_record(
    rank: int,
    hostname: str,
    *,
    attestation: object | None = None,
    error: str | None = None,
    generation: str = "a" * 64,
    physical_node_id: str | None = None,
    gpu_inventory_fingerprint: str = "f" * 64,
    backend: str = "nccl",
    process_group_timeout_us: int = 600_000_000,
) -> dict[str, object]:
    captured = _launch_attestation() if attestation is None else attestation
    return {
        "rank": rank,
        "backend": backend,
        "process_group_timeout_us": process_group_timeout_us,
        "generation": generation,
        "hostname": hostname,
        "physical_node_id": physical_node_id
        or sha256(hostname.encode("utf-8")).hexdigest(),
        "gpu_inventory_fingerprint": gpu_inventory_fingerprint,
        "attestation": list(rsag_module._launch_attestation_state(captured)),
        "error": error,
    }


def test_checkpoint_v2_publishes_layout_and_exact_refresh_state() -> None:
    save_source = getsource(ShardedAdamW.state_dict)
    load_source = getsource(ShardedAdamW.load_state_dict)

    assert RSAG_CHECKPOINT_SCHEMA_VERSION == 2
    assert '"layout": _layout_state(self.layout)' in save_source
    assert '"force_refresh": self.force_refresh' in save_source
    assert 'force_refresh = state["force_refresh"]' in load_source
    assert "self.force_refresh = force_refresh" in load_source
    assert "self.force_refresh = True" not in load_source


def test_rsag_evidence_schema_v2_binds_checkpoint_and_build_identity() -> None:
    assert RSAG_EVIDENCE_SCHEMA_VERSION == 2
    assert "checkpoint_schema_version" in RSAGEnvironment.__slots__
    assert "build_fingerprint" in RSAGEnvironment.__slots__
    assert "checkpoint_schema_version" in RSAGEvidence.__slots__
    assert "build_fingerprint" in RSAGEvidence.__slots__


def test_launch_attempt_claim_is_lock_protected() -> None:
    assert "with _RSAG_LAUNCH_LOCK" in getsource(rsag_module._claim_launch_attempt)


def test_exact_positive_evidence_enables_rsag_qwd() -> None:
    decision = select_rsag_route(_environment(), (_evidence(),))

    assert decision.route == "rsag_qwd"
    assert decision.reason == "qualified_positive_evidence"
    assert decision.evidence_schema_version == RSAG_EVIDENCE_SCHEMA_VERSION
    assert decision.uses_rsag is True


@pytest.mark.parametrize(
    "topology_class",
    ["single_node_pcie", "single_node_nvlink"],
)
def test_single_node_topology_is_not_product_qualified(
    topology_class: str,
) -> None:
    environment = _environment(
        node_count=1,
        topology_class=topology_class,
        transport="nccl_p2p",
    )
    evidence = _evidence(
        node_count=1,
        topology_class=topology_class,
        transport="nccl_p2p",
    )

    decision = select_rsag_route(environment, (evidence,))

    assert decision.route == "native"
    assert decision.reason == "unsupported_product_topology"


def test_evidence_rejects_a_logical_byte_range() -> None:
    with pytest.raises(ValueError, match="exact logical_bytes"):
        _evidence(
            min_logical_bytes=8 * 1024 * 1024,
            max_logical_bytes=32 * 1024 * 1024,
        )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("world_size", 4, "no_exact_evidence"),
        ("node_count", 1, "no_exact_evidence"),
        ("logical_bytes", 7 * 1024 * 1024, "no_exact_evidence"),
        (
            "topology_class",
            "single_node_pcie",
            "unsupported_product_topology",
        ),
        ("transport", "nccl_socket", "no_exact_evidence"),
        ("gpu_model", "NVIDIA GeForce RTX 4090", "no_exact_evidence"),
        ("checkpoint_schema_version", 1, "unsupported_binary_identity"),
        ("build_fingerprint", "b" * 64, "no_exact_evidence"),
    ],
)
def test_any_environment_mismatch_falls_back_to_native(
    field: str,
    value: object,
    reason: str,
) -> None:
    decision = select_rsag_route(
        replace(_environment(), **{field: value}),
        (_evidence(),),
    )

    assert decision.route == "native"
    assert decision.reason == reason
    assert decision.uses_rsag is False


def test_checkpoint_schema_mismatch_in_evidence_falls_back_to_native() -> None:
    decision = select_rsag_route(
        _environment(),
        (_evidence(checkpoint_schema_version=1),),
    )

    assert decision.route == "native"
    assert decision.reason == "no_exact_evidence"
    assert decision.uses_rsag is False


@pytest.mark.parametrize(
    "value",
    ["", "a" * 63, "A" * 64, "g" * 64, True],
)
def test_build_fingerprint_requires_exact_lowercase_sha256(value: object) -> None:
    with pytest.raises(ValueError, match="build_fingerprint"):
        _environment(build_fingerprint=value)
    with pytest.raises(ValueError, match="build_fingerprint"):
        _evidence(build_fingerprint=value)


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

    assert adapter.decision.route == "native"
    assert adapter.decision.reason == "collective_qualification_required"
    assert adapter.schedule == QWDSchedule(refresh_interval=100)
    assert adapter.mode(0) == "native"
    assert adapter.mode(1) == "native"


def test_adapter_route_identity_is_read_only() -> None:
    adapter = RSAGQWDAdapter(_environment(), (_evidence(),))

    with pytest.raises(AttributeError):
        adapter.decision = select_rsag_route(_environment(), ())
    with pytest.raises(AttributeError):
        adapter.environment = _environment(gpu_model="forged")


def test_native_adapter_cannot_construct_lowbit_plans(monkeypatch) -> None:
    adapter = RSAGQWDAdapter(
        _environment(),
        (),
        launch_attestation=_captured_launch_attestation(monkeypatch),
    )
    monkeypatch.setattr(
        rsag_module,
        "detect_rsag_environment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CapabilityError(kwargs["plan_preflight"].error)
        ),
    )

    with pytest.raises(CapabilityError, match="collective qualification"):
        adapter.create_plans(object(), global_numel=1, rank=0)


def test_qualified_adapter_rejects_a_size_outside_its_evidence_key(
    monkeypatch,
) -> None:
    adapter = RSAGQWDAdapter(
        _environment(),
        (_evidence(),),
        launch_attestation=_captured_launch_attestation(monkeypatch),
    )
    monkeypatch.setattr(
        rsag_module,
        "detect_rsag_environment",
        lambda *args, **kwargs: _environment(),
    )
    assert adapter.qualify_collectively(object(), rank=0).uses_rsag
    monkeypatch.setattr(
        rsag_module,
        "detect_rsag_environment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CapabilityError(kwargs["plan_preflight"].error)
        ),
    )

    with pytest.raises(CapabilityError, match="size"):
        adapter.create_plans(object(), global_numel=1, rank=0)


def test_qualified_adapter_rechecks_the_live_runtime_identity(
    monkeypatch,
) -> None:
    environment = _environment()
    adapter = RSAGQWDAdapter(
        environment,
        (_evidence(),),
        launch_attestation=_captured_launch_attestation(monkeypatch),
    )
    monkeypatch.setattr(
        rsag_module,
        "detect_rsag_environment",
        lambda *args, **kwargs: environment,
    )
    assert adapter.qualify_collectively(object(), rank=0).uses_rsag
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
            _TEST_PROCESS_GROUP,
            global_numel=environment.logical_bytes // 2,
            rank=0,
        )


def test_environment_detection_normalizes_missing_nccl_to_capability_error(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TOPOLOGY",
        "cross_node_socket",
    )
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TRANSPORT",
        "nccl_socket_eno2",
    )
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    monkeypatch.setenv("NCCL_NET", "Socket")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "=eno2")
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
    monkeypatch.setattr(
        rsag_module,
        "compute_rsag_build_fingerprint",
        lambda: "a" * 64,
    )

    with pytest.raises(CapabilityError, match="NCCL"):
        rsag_module.detect_rsag_environment(
            _TEST_PROCESS_GROUP,
            logical_bytes=1024,
            topology_class="single_node_pcie",
            transport="nccl_p2p",
            launch_attestation=_captured_launch_attestation(
                monkeypatch,
                topology_class="single_node_pcie",
            ),
        )


def test_environment_detection_rejects_attested_transport_drift(
    monkeypatch,
) -> None:
    attestation = _captured_launch_attestation(
        monkeypatch,
        topology_class="single_node_pcie",
    )
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TRANSPORT",
        "nccl_socket_eno2",
    )
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
            nccl=SimpleNamespace(version=lambda: (2, 22, 3)),
        ),
    )
    monkeypatch.setattr(rsag_module, "_torch", lambda: fake_torch)
    monkeypatch.setattr(
        rsag_module,
        "compute_rsag_build_fingerprint",
        lambda: "a" * 64,
    )

    with pytest.raises(CapabilityError, match="attestation"):
        rsag_module.detect_rsag_environment(
            _TEST_PROCESS_GROUP,
            logical_bytes=1024,
            topology_class="single_node_pcie",
            transport="nccl_p2p",
            launch_attestation=attestation,
        )


def test_environment_detection_rejects_heterogeneous_rank_identity(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TOPOLOGY",
        "cross_node_socket",
    )
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TRANSPORT",
        "nccl_socket_eno2",
    )
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "=eno2")

    def gather(values: list[object], value: object, group: object) -> None:
        del group
        if type(value) is dict:
            values[0] = value
            stale = value.copy()
            stale["hostname"] = "node-b"
            stale["build_fingerprint"] = "b" * 64
            values[1] = stale
            return
        values[0] = "node-a"
        values[1] = "node-b"

    distributed = SimpleNamespace(
        is_initialized=lambda: True,
        get_world_size=lambda group: 2,
        all_gather_object=gather,
    )
    fake_torch = SimpleNamespace(
        __version__="2.5.0a0+872d972e41.nv24.08",
        version=SimpleNamespace(cuda="12.6"),
        distributed=distributed,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda: "NVIDIA RTX A6000",
            nccl=SimpleNamespace(version=lambda: (2, 22, 3)),
        ),
    )
    monkeypatch.setattr(rsag_module, "_torch", lambda: fake_torch)
    monkeypatch.setattr(
        rsag_module,
        "compute_rsag_build_fingerprint",
        lambda: "a" * 64,
    )

    with pytest.raises(CapabilityError, match="globally consistent"):
        rsag_module.detect_rsag_environment(
            _TEST_PROCESS_GROUP,
            logical_bytes=1024,
            topology_class="cross_node_socket",
            transport="nccl_socket_eno2",
            launch_attestation=_captured_launch_attestation(
                monkeypatch,
                topology_class="cross_node_socket",
                transport="nccl_socket_eno2",
                nccl_ib_disable="1",
                nccl_socket_ifname="=eno2",
            ),
        )


def test_launch_attestation_is_bound_during_process_group_init(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TOPOLOGY",
        "cross_node_socket",
    )
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TRANSPORT",
        "nccl_socket_eno2",
    )
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    monkeypatch.setenv("NCCL_NET", "Socket")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "=eno2")
    initialize = getattr(rsag_module, "initialize_rsag_process_group", None)
    assert callable(initialize)
    initialized = False
    observed_transport = None

    def init_process_group(**kwargs: object) -> None:
        nonlocal initialized, observed_transport
        assert kwargs["backend"] == "nccl"
        observed_transport = os.environ["LOWBIT_COMM_RSAG_ATTESTED_TRANSPORT"]
        initialized = True

    distributed = SimpleNamespace(
        group=SimpleNamespace(WORLD=_TEST_PROCESS_GROUP),
        is_initialized=lambda: initialized,
        init_process_group=init_process_group,
        destroy_process_group=lambda: None,
    )
    fake_torch = SimpleNamespace(distributed=distributed)
    monkeypatch.setattr(rsag_module, "_torch", lambda: fake_torch)
    attested = _launch_attestation()
    monkeypatch.setattr(
        rsag_module,
        "_rendezvous_launch_control",
        lambda *args, **kwargs: (
            attested,
            object(),
            0,
            2,
            "a" * 64,
            "b" * 64,
            "c" * 64,
        ),
    )
    monkeypatch.setattr(
        rsag_module,
        "_exchange_control_records",
        lambda store, phase, local, rank, world_size, generation: (
            {**local, "generation": generation},
            {**local, "rank": 1, "generation": generation},
        ),
    )

    attestation = initialize(backend="nccl")

    assert attestation == attested
    assert observed_transport == "nccl_socket_eno2"
    with pytest.raises(CapabilityError, match="one-shot"):
        initialize(backend="nccl")


def test_launch_control_collectively_rejects_one_invalid_rank() -> None:
    records = (
        _launch_control_record(0, "node-a"),
        _launch_control_record(
            1,
            "node-b",
            error="CapabilityError: invalid NCCL Socket interface",
        ),
    )

    with pytest.raises(CapabilityError, match="rank 1"):
        rsag_module._validated_launch_control_attestation(records, "a" * 64)


def test_launch_control_collectively_rejects_asymmetric_backend() -> None:
    records = (
        _launch_control_record(0, "node-a"),
        _launch_control_record(
            1,
            "node-b",
            backend="gloo",
            error="RSAG/qWD requires the NCCL backend.",
        ),
    )

    with pytest.raises(CapabilityError, match="rank 1"):
        rsag_module._validated_launch_control_attestation(records, "a" * 64)


def test_launch_control_collectively_rejects_asymmetric_timeout() -> None:
    records = (
        _launch_control_record(0, "node-a"),
        _launch_control_record(
            1,
            "node-b",
            process_group_timeout_us=300_000_000,
        ),
    )

    with pytest.raises(CapabilityError, match="globally consistent"):
        rsag_module._validated_launch_control_attestation(records, "a" * 64)


def test_launch_control_joins_store_before_reporting_invalid_local_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "test-run")
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "0")
    monkeypatch.setenv("MASTER_ADDR", "192.168.8.156")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TOPOLOGY",
        raising=False,
    )
    monkeypatch.delenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TRANSPORT",
        raising=False,
    )

    class FakeStore:
        def __init__(self) -> None:
            self.values: dict[str, bytes] = {}

        def set(self, key: str, value: bytes) -> None:
            self.values[key] = value

        def wait(self, keys: list[str]) -> None:
            local = json.loads(self.values[keys[0]].decode("utf-8"))
            self.values[keys[1]] = json.dumps(
                _launch_control_record(
                    1,
                    "node-b",
                    generation=local["generation"],
                ),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")

        def get(self, key: str) -> bytes:
            return self.values[key]

    store = FakeStore()
    distributed = SimpleNamespace(
        rendezvous=lambda *args, **kwargs: iter(((store, 0, 2),)),
    )
    monkeypatch.setattr(
        rsag_module,
        "_hardware_node_identity",
        lambda: ("0" * 64, "1" * 64),
    )

    with pytest.raises(CapabilityError, match="rank 0"):
        rsag_module._rendezvous_launch_control(
            distributed,
            timeout=rsag_module._CONTROL_TIMEOUT,
            backend="nccl",
            process_group_timeout_us=600_000_000,
            launch_error=None,
        )

    assert any(key.endswith("/pre/0") for key in store.values)


def test_control_generation_changes_on_elastic_restart(monkeypatch) -> None:
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "production-run")
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "0")
    monkeypatch.setenv("MASTER_ADDR", "192.168.8.156")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("WORLD_SIZE", "4")
    first = rsag_module._control_generation()
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "1")

    second = rsag_module._control_generation()

    assert first != second


def test_control_exchange_does_not_consume_stale_generation() -> None:
    stale_generation = "a" * 64
    generation = "b" * 64
    local = {
        "rank": 0,
        "error": None,
    }

    class FakeStore:
        def __init__(self) -> None:
            stale_prefix = f"{rsag_module._CONTROL_NAMESPACE}/{stale_generation}/init"
            self.values = {
                f"{stale_prefix}/0": b'{"error":null,"generation":"'
                + stale_generation.encode("ascii")
                + b'","rank":0}',
                f"{stale_prefix}/1": b'{"error":null,"generation":"'
                + stale_generation.encode("ascii")
                + b'","rank":1}',
            }
            self.waited: list[str] = []

        def set(self, key: str, value: bytes) -> None:
            self.values[key] = value

        def wait(self, keys: list[str]) -> None:
            self.waited = keys
            peer = {**local, "rank": 1, "generation": generation}
            self.values[keys[1]] = json.dumps(
                peer,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")

        def get(self, key: str) -> bytes:
            return self.values[key]

    store = FakeStore()

    records = rsag_module._exchange_control_records(
        store,
        "init",
        local,
        0,
        2,
        generation,
    )

    assert all(f"/{generation}/init/" in key for key in store.waited)
    assert records[1]["generation"] == generation


def test_gpu_inventory_fingerprint_binds_pci_inventory() -> None:
    baseline = rsag_module._compute_gpu_inventory_fingerprint(
        (
            (
                "NVIDIA RTX A6000",
                "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "0000:01:00.0",
            ),
        ),
    )
    changed = rsag_module._compute_gpu_inventory_fingerprint(
        (
            (
                "NVIDIA RTX A6000",
                "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "0000:02:00.0",
            ),
        ),
    )

    assert baseline != changed


def test_distinct_container_hostnames_do_not_forge_distinct_nodes() -> None:
    physical_node = "c" * 64
    records = (
        _launch_control_record(
            0,
            "container-a",
            physical_node_id=physical_node,
        ),
        _launch_control_record(
            1,
            "container-b",
            physical_node_id=physical_node,
        ),
    )

    with pytest.raises(CapabilityError, match="cross-node Socket"):
        rsag_module._validated_launch_control_attestation(records, "a" * 64)


def test_same_physical_node_rejects_gpu_inventory_drift() -> None:
    physical_node = "c" * 64
    records = (
        _launch_control_record(
            0,
            "container-a",
            physical_node_id=physical_node,
            gpu_inventory_fingerprint="d" * 64,
        ),
        _launch_control_record(
            1,
            "container-b",
            physical_node_id=physical_node,
            gpu_inventory_fingerprint="e" * 64,
        ),
    )

    with pytest.raises(CapabilityError, match="inventory differs"):
        rsag_module._validated_launch_control_attestation(records, "a" * 64)


def test_node_count_uses_physical_id_not_container_hostname() -> None:
    identities = tuple(
        {
            "hostname": "shared-container-name" if rank < 2 else "other",
            "physical_node_id": character * 64,
            "gpu_inventory_fingerprint": "f" * 64,
        }
        for rank, character in enumerate(("a", "b", "c"))
    )

    assert rsag_module._physical_node_count(identities) == 3


@pytest.mark.parametrize(
    "topology_class",
    ["single_node_pcie", "single_node_nvlink"],
)
def test_launch_control_rejects_untrusted_single_node_topology(
    topology_class: str,
) -> None:
    attestation = _launch_attestation(
        topology_class=topology_class,
        transport="nccl_p2p",
        nccl_ib_disable=None,
        nccl_net=None,
        nccl_socket_ifname=None,
    )
    records = (
        _launch_control_record(0, "node-a", attestation=attestation),
        _launch_control_record(1, "node-a", attestation=attestation),
    )

    with pytest.raises(CapabilityError, match="cross-node Socket"):
        rsag_module._validated_launch_control_attestation(records, "a" * 64)


def test_post_init_control_collectively_rejects_one_rank_drift() -> None:
    attestation = _launch_attestation()
    drifted = _launch_attestation(
        transport="nccl_socket_eno3",
        nccl_socket_ifname="=eno3",
    )
    records = (
        {
            "rank": 0,
            "generation": "a" * 64,
            "attestation": list(rsag_module._launch_attestation_state(attestation)),
            "error": None,
        },
        {
            "rank": 1,
            "generation": "a" * 64,
            "attestation": list(rsag_module._launch_attestation_state(drifted)),
            "error": None,
        },
    )

    with pytest.raises(CapabilityError, match="changed during"):
        rsag_module._validate_post_init_control_records(
            records,
            attestation,
            "a" * 64,
        )


def test_post_init_attestation_read_failure_destroys_process_group(
    monkeypatch,
) -> None:
    attestation = _launch_attestation()
    initialized = False
    destroyed = False

    def init_process_group(**kwargs: object) -> None:
        nonlocal initialized
        assert kwargs["backend"] == "nccl"
        initialized = True

    def destroy_process_group() -> None:
        nonlocal initialized, destroyed
        destroyed = True
        initialized = False

    distributed = SimpleNamespace(
        group=SimpleNamespace(WORLD=_TEST_PROCESS_GROUP),
        is_initialized=lambda: initialized,
        init_process_group=init_process_group,
        destroy_process_group=destroy_process_group,
    )
    monkeypatch.setattr(
        rsag_module,
        "_torch",
        lambda: SimpleNamespace(distributed=distributed),
    )
    monkeypatch.setattr(
        rsag_module,
        "_rendezvous_launch_control",
        lambda *args, **kwargs: (
            attestation,
            object(),
            0,
            2,
            "a" * 64,
            "b" * 64,
            "c" * 64,
        ),
    )

    def exchange(store, phase, local, rank, world_size, generation):
        del store, rank, world_size
        if phase == "init":
            return (
                {**local, "generation": generation},
                {**local, "rank": 1, "generation": generation},
            )
        return (
            {**local, "generation": generation},
            {
                "rank": 1,
                "generation": generation,
                "attestation": None,
                "error": None,
            },
        )

    monkeypatch.setattr(rsag_module, "_exchange_control_records", exchange)
    monkeypatch.setattr(
        rsag_module,
        "_read_current_launch_attestation",
        lambda: (_ for _ in ()).throw(RuntimeError("injected post-read")),
    )

    with pytest.raises(CapabilityError, match="post-init attestation"):
        rsag_module.initialize_rsag_process_group()

    assert destroyed is True
    assert id(attestation) not in rsag_module._CAPTURED_LAUNCH_ATTESTATIONS


def test_post_group_control_exchange_failure_destroys_process_group(
    monkeypatch,
) -> None:
    attestation = _launch_attestation()
    initialized = False
    destroyed = False

    def init_process_group(**kwargs: object) -> None:
        nonlocal initialized
        initialized = True

    def destroy_process_group() -> None:
        nonlocal initialized, destroyed
        destroyed = True
        initialized = False

    distributed = SimpleNamespace(
        group=SimpleNamespace(WORLD=_TEST_PROCESS_GROUP),
        is_initialized=lambda: initialized,
        init_process_group=init_process_group,
        destroy_process_group=destroy_process_group,
    )
    monkeypatch.setattr(
        rsag_module,
        "_torch",
        lambda: SimpleNamespace(distributed=distributed),
    )
    monkeypatch.setattr(
        rsag_module,
        "_rendezvous_launch_control",
        lambda *args, **kwargs: (
            attestation,
            object(),
            0,
            2,
            "a" * 64,
            "b" * 64,
            "c" * 64,
        ),
    )
    monkeypatch.setattr(
        rsag_module,
        "_read_current_launch_attestation",
        lambda: attestation,
    )
    monkeypatch.setattr(
        rsag_module,
        "_exchange_control_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CapabilityError("injected control exchange failure")
        ),
    )

    with pytest.raises(CapabilityError, match="control exchange"):
        rsag_module.initialize_rsag_process_group()

    assert destroyed is True
    with pytest.raises(CapabilityError, match="one-shot"):
        rsag_module.initialize_rsag_process_group()


def test_missing_default_group_during_binding_is_cleaned_up(monkeypatch) -> None:
    attestation = _launch_attestation()
    initialized = False
    destroyed = False

    def init_process_group(**kwargs: object) -> None:
        nonlocal initialized
        initialized = True

    def destroy_process_group() -> None:
        nonlocal initialized, destroyed
        destroyed = True
        initialized = False

    distributed = SimpleNamespace(
        group=SimpleNamespace(WORLD=None),
        is_initialized=lambda: initialized,
        init_process_group=init_process_group,
        destroy_process_group=destroy_process_group,
    )
    monkeypatch.setattr(
        rsag_module,
        "_torch",
        lambda: SimpleNamespace(distributed=distributed),
    )
    monkeypatch.setattr(
        rsag_module,
        "_rendezvous_launch_control",
        lambda *args, **kwargs: (
            attestation,
            object(),
            0,
            2,
            "a" * 64,
            "b" * 64,
            "c" * 64,
        ),
    )
    monkeypatch.setattr(
        rsag_module,
        "_read_current_launch_attestation",
        lambda: attestation,
    )
    monkeypatch.setattr(
        rsag_module,
        "_exchange_control_records",
        lambda store, phase, local, rank, world_size, generation: (
            {**local, "generation": generation},
            {**local, "rank": 1, "generation": generation},
        ),
    )

    with pytest.raises(CapabilityError, match="unavailable for binding"):
        rsag_module.initialize_rsag_process_group()

    assert destroyed is True
    assert id(attestation) not in rsag_module._CAPTURED_LAUNCH_ATTESTATIONS


def test_environment_detection_rejects_post_group_forged_attestation(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TOPOLOGY",
        "single_node_pcie",
    )
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TRANSPORT",
        "nccl_p2p",
    )
    forged = _launch_attestation(topology_class="single_node_pcie")
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
            nccl=SimpleNamespace(version=lambda: (2, 22, 3)),
        ),
    )
    monkeypatch.setattr(rsag_module, "_torch", lambda: fake_torch)
    monkeypatch.setattr(
        rsag_module,
        "compute_rsag_build_fingerprint",
        lambda: "a" * 64,
    )

    with pytest.raises(CapabilityError, match="registered process-group"):
        rsag_module.detect_rsag_environment(
            _TEST_PROCESS_GROUP,
            logical_bytes=1024,
            topology_class="single_node_pcie",
            transport="nccl_p2p",
            launch_attestation=forged,
        )


def test_environment_detection_rejects_mutated_captured_attestation(
    monkeypatch,
) -> None:
    attestation = _captured_launch_attestation(
        monkeypatch,
        topology_class="single_node_pcie",
    )
    object.__setattr__(attestation, "topology_class", "single_node_nvlink")
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TOPOLOGY",
        "single_node_nvlink",
    )
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
            nccl=SimpleNamespace(version=lambda: (2, 22, 3)),
        ),
    )
    monkeypatch.setattr(rsag_module, "_torch", lambda: fake_torch)
    monkeypatch.setattr(
        rsag_module,
        "compute_rsag_build_fingerprint",
        lambda: "a" * 64,
    )

    with pytest.raises(CapabilityError, match="changed since capture"):
        rsag_module.detect_rsag_environment(
            _TEST_PROCESS_GROUP,
            logical_bytes=1024,
            topology_class="single_node_nvlink",
            transport="nccl_p2p",
            launch_attestation=attestation,
        )


def test_cross_node_attestation_requires_intra_node_p2p() -> None:
    attestation_type = rsag_module.RSAGLaunchAttestation

    with pytest.raises(CapabilityError, match="NCCL P2P"):
        attestation_type(
            topology_class="cross_node_socket",
            transport="nccl_socket_eno2",
            nccl_ib_disable="1",
            nccl_net="Socket",
            nccl_socket_ifname="=eno2",
            nccl_p2p_disable="1",
        )


@pytest.mark.parametrize("nccl_net", [None, "IB", "socket"])
def test_cross_node_attestation_requires_socket_net(
    nccl_net: str | None,
) -> None:
    with pytest.raises(CapabilityError, match="NCCL_NET=Socket"):
        rsag_module.RSAGLaunchAttestation(
            topology_class="cross_node_socket",
            transport="nccl_socket_eno2",
            nccl_ib_disable="1",
            nccl_net=nccl_net,
            nccl_socket_ifname="=eno2",
            nccl_p2p_disable=None,
        )


@pytest.mark.parametrize(
    "socket_ifname",
    ["eno2", "=eno2,eno3", "^docker0"],
)
def test_socket_attestation_requires_one_exact_interface(
    socket_ifname: str,
) -> None:
    with pytest.raises(CapabilityError, match="exact NCCL Socket interface"):
        _launch_attestation(
            topology_class="cross_node_socket",
            transport="nccl_socket_eno2",
            nccl_ib_disable="1",
            nccl_socket_ifname=socket_ifname,
        )


def test_environment_detection_collectively_rejects_requested_rank_drift(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TOPOLOGY",
        "cross_node_socket",
    )
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TRANSPORT",
        "nccl_socket_eno2",
    )

    def gather(values: list[object], value: object, group: object) -> None:
        del group
        assert type(value) is dict
        values[0] = value
        drifted = value.copy()
        drifted["requested_transport"] = "nccl_socket_eno2"
        values[1] = drifted

    distributed = SimpleNamespace(
        is_initialized=lambda: True,
        get_world_size=lambda group: 2,
        all_gather_object=gather,
    )
    fake_torch = SimpleNamespace(
        __version__="2.5.0a0+872d972e41.nv24.08",
        version=SimpleNamespace(cuda="12.6"),
        distributed=distributed,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda: "NVIDIA RTX A6000",
            nccl=SimpleNamespace(version=lambda: (2, 22, 3)),
        ),
    )
    monkeypatch.setattr(rsag_module, "_torch", lambda: fake_torch)
    monkeypatch.setattr(
        rsag_module,
        "compute_rsag_build_fingerprint",
        lambda: "a" * 64,
    )

    with pytest.raises(CapabilityError, match="globally consistent"):
        rsag_module.detect_rsag_environment(
            _TEST_PROCESS_GROUP,
            logical_bytes=1024,
            topology_class="single_node_nvlink",
            transport="nccl_p2p",
            launch_attestation=_captured_launch_attestation(monkeypatch),
        )


def test_plan_preflight_collectively_rejects_one_rank_size_drift(
    monkeypatch,
) -> None:
    environment = _environment()
    attestation = _captured_launch_attestation(monkeypatch)
    adapter = RSAGQWDAdapter(
        environment,
        (_evidence(),),
        launch_attestation=attestation,
    )
    with monkeypatch.context() as qualify_patch:
        qualify_patch.setattr(
            rsag_module,
            "detect_rsag_environment",
            lambda *args, **kwargs: environment,
        )
        assert adapter.qualify_collectively(object(), rank=0).uses_rsag
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TOPOLOGY",
        "cross_node_socket",
    )
    monkeypatch.setenv(
        "LOWBIT_COMM_RSAG_ATTESTED_TRANSPORT",
        "nccl_socket_eno2",
    )

    def gather(values: list[object], value: object, group: object) -> None:
        del group
        assert type(value) is dict
        values[0] = value
        drifted = value.copy()
        preflight = value["plan_preflight"]
        drifted["plan_preflight"] = {
            **preflight,
            "global_numel": preflight["global_numel"] + 1,
            "rank": 1,
        }
        values[1] = drifted

    distributed = SimpleNamespace(
        is_initialized=lambda: True,
        get_world_size=lambda group: 2,
        all_gather_object=gather,
    )
    fake_torch = SimpleNamespace(
        __version__="2.5.0a0+872d972e41.nv24.08",
        version=SimpleNamespace(cuda="12.6"),
        distributed=distributed,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda: "NVIDIA RTX A6000",
            nccl=SimpleNamespace(version=lambda: (2, 22, 3)),
        ),
    )
    monkeypatch.setattr(rsag_module, "_torch", lambda: fake_torch)
    monkeypatch.setattr(
        rsag_module,
        "compute_rsag_build_fingerprint",
        lambda: "a" * 64,
    )

    with pytest.raises(CapabilityError, match="plan preflight"):
        adapter.create_plans(
            _TEST_PROCESS_GROUP,
            global_numel=environment.logical_bytes // 2,
            rank=0,
        )


def test_collective_qualification_falls_back_on_rank_evidence_drift(
    monkeypatch,
) -> None:
    environment = _environment()
    attestation = _captured_launch_attestation(monkeypatch)
    adapter = RSAGQWDAdapter(
        environment,
        (_evidence(),),
        launch_attestation=attestation,
    )

    def gather(values: list[object], value: object, group: object) -> None:
        del group
        assert type(value) is dict
        values[0] = value
        drifted = value.copy()
        preflight = value["plan_preflight"]
        drifted["plan_preflight"] = {
            **preflight,
            "qualification_fingerprint": "b" * 64,
            "rank": 1,
        }
        values[1] = drifted

    distributed = SimpleNamespace(
        is_initialized=lambda: True,
        get_world_size=lambda group: 2,
        all_gather_object=gather,
    )
    fake_torch = SimpleNamespace(
        __version__="2.5.0a0+872d972e41.nv24.08",
        version=SimpleNamespace(cuda="12.6"),
        distributed=distributed,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda: "NVIDIA RTX A6000",
            nccl=SimpleNamespace(version=lambda: (2, 22, 3)),
        ),
    )
    monkeypatch.setattr(rsag_module, "_torch", lambda: fake_torch)
    monkeypatch.setattr(
        rsag_module,
        "compute_rsag_build_fingerprint",
        lambda: "a" * 64,
    )
    qualify = getattr(adapter, "qualify_collectively", None)
    assert callable(qualify)

    decision = qualify(_TEST_PROCESS_GROUP, rank=0)

    assert decision.route == "native"
    assert decision.reason == "collective_qualification_failed"
    assert adapter.decision == decision
    assert adapter.mode(1) == "native"


def test_collective_qualification_wraps_malformed_logical_bytes(
    monkeypatch,
) -> None:
    environment = _environment()
    attestation = _captured_launch_attestation(monkeypatch)
    adapter = RSAGQWDAdapter(
        environment,
        (_evidence(),),
        launch_attestation=attestation,
    )
    object.__setattr__(environment, "logical_bytes", "forged")

    def reject_preflight(*args, **kwargs) -> object:
        del args
        preflight = kwargs["plan_preflight"]
        assert preflight.error is not None
        raise CapabilityError("collective preflight rejected")

    monkeypatch.setattr(
        rsag_module,
        "detect_rsag_environment",
        reject_preflight,
    )

    decision = adapter.qualify_collectively(_TEST_PROCESS_GROUP, rank=0)

    assert decision.route == "native"
    assert decision.reason == "collective_qualification_failed"


def test_collective_qualification_enables_identical_ranks(monkeypatch) -> None:
    environment = _environment()
    attestation = _captured_launch_attestation(monkeypatch)
    adapter = RSAGQWDAdapter(
        environment,
        (_evidence(),),
        launch_attestation=attestation,
    )

    def gather(values: list[object], value: object, group: object) -> None:
        del group
        assert type(value) is dict
        values[0] = value
        peer = value.copy()
        peer["hostname"] = "node-b"
        peer["physical_node_id"] = "d" * 64
        peer["plan_preflight"] = {
            **value["plan_preflight"],
            "rank": 1,
        }
        values[1] = peer

    distributed = SimpleNamespace(
        is_initialized=lambda: True,
        get_world_size=lambda group: 2,
        all_gather_object=gather,
    )
    fake_torch = SimpleNamespace(
        __version__="2.5.0a0+872d972e41.nv24.08",
        version=SimpleNamespace(cuda="12.6"),
        distributed=distributed,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda: "NVIDIA RTX A6000",
            nccl=SimpleNamespace(version=lambda: (2, 22, 3)),
        ),
    )
    monkeypatch.setattr(rsag_module, "_torch", lambda: fake_torch)
    monkeypatch.setattr(
        rsag_module,
        "compute_rsag_build_fingerprint",
        lambda: "a" * 64,
    )

    decision = adapter.qualify_collectively(_TEST_PROCESS_GROUP, rank=0)

    assert decision.route == "rsag_qwd"
    assert adapter.decision == decision
    assert adapter.mode(1) == "qwd"


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
    checkpoint["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        optimizer.load_state_dict(checkpoint)


def test_checkpoint_copy_failure_does_not_partially_commit(monkeypatch) -> None:
    layout = ShardLayout.build(1, 1, 0)
    optimizer = object.__new__(ShardedAdamW)
    optimizer.layout = layout
    optimizer.master = _CloneValue(1)
    optimizer.exp_avg = _CloneValue(2)
    optimizer.exp_avg_sq = _CloneValue(3)
    optimizer.step_count = 4
    optimizer.learning_rate = 0.1
    optimizer.betas = (0.9, 0.999)
    optimizer.eps = 1.0e-8
    optimizer.weight_decay = 0.01
    optimizer.amp_state = {"scale": 1024.0}
    optimizer.rng_state = {"seed": 7}
    optimizer.force_refresh = False
    monkeypatch.setattr(
        rsag_module,
        "_validated_shard_tensor",
        lambda value, layout, name, device=None: value,
    )
    monkeypatch.setattr(
        rsag_module,
        "_require_zero_padding",
        lambda value, layout, name: None,
    )
    state = {
        "schema_version": RSAG_CHECKPOINT_SCHEMA_VERSION,
        "layout": {
            "global_numel": 1,
            "world_size": 1,
            "rank": 0,
            "start": 0,
            "valid_numel": 1,
            "padded_numel": 1,
        },
        "master": _CloneValue(11),
        "exp_avg": _CloneValue(12),
        "exp_avg_sq": _CloneValue(13),
        "step_count": 14,
        "learning_rate": 0.2,
        "betas": (0.8, 0.9),
        "eps": 1.0e-7,
        "weight_decay": 0.02,
        "amp_state": {"failure": _ExplodingDeepcopy()},
        "rng_state": {"seed": 17},
        "force_refresh": True,
    }

    with pytest.raises(RuntimeError, match="injected deepcopy failure"):
        optimizer.load_state_dict(state)

    assert optimizer.master.value == 1
    assert optimizer.exp_avg.value == 2
    assert optimizer.exp_avg_sq.value == 3
    assert optimizer.step_count == 4
    assert optimizer.learning_rate == 0.1
    assert optimizer.betas == (0.9, 0.999)
    assert optimizer.eps == 1.0e-8
    assert optimizer.weight_decay == 0.01
    assert optimizer.amp_state == {"scale": 1024.0}
    assert optimizer.rng_state == {"seed": 7}
    assert optimizer.force_refresh is False


def test_checkpoint_layout_rejects_bool_fields_before_mutation() -> None:
    layout = ShardLayout.build(1, 1, 0)
    optimizer = object.__new__(ShardedAdamW)
    optimizer.layout = layout
    optimizer.master = _CloneValue(1)
    optimizer.exp_avg = _CloneValue(2)
    optimizer.exp_avg_sq = _CloneValue(3)
    optimizer.step_count = 4
    optimizer.learning_rate = 0.1
    optimizer.betas = (0.9, 0.999)
    optimizer.eps = 1.0e-8
    optimizer.weight_decay = 0.01
    optimizer.amp_state = {"scale": 1024.0}
    optimizer.rng_state = {"seed": 7}
    optimizer.force_refresh = False
    state = {
        "schema_version": RSAG_CHECKPOINT_SCHEMA_VERSION,
        "layout": {
            "global_numel": 1,
            "world_size": 1,
            "rank": False,
            "start": False,
            "valid_numel": 1,
            "padded_numel": 1,
        },
        "master": _CloneValue(11),
        "exp_avg": _CloneValue(12),
        "exp_avg_sq": _CloneValue(13),
        "step_count": 14,
        "learning_rate": 0.2,
        "betas": (0.8, 0.9),
        "eps": 1.0e-7,
        "weight_decay": 0.02,
        "amp_state": {"scale": 2048.0},
        "rng_state": {"seed": 17},
        "force_refresh": True,
    }

    with pytest.raises(ValueError, match="layout"):
        optimizer.load_state_dict(state)

    assert optimizer.master.value == 1
    assert optimizer.exp_avg.value == 2
    assert optimizer.exp_avg_sq.value == 3
    assert optimizer.step_count == 4


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
    worker = Path(__file__).parents[2] / "benchmarks" / "distributed_psi_v040_worker.py"
    source = worker.read_text(encoding="utf-8")

    assert "_create_rsag_qwd_plans(" in source
    assert "def _qwd_config(" not in source


def test_binary_identity_constants_match_packaging_and_loader() -> None:
    from lowbit_comm.backends.cuda.loader import CUDA_ABI_VERSION
    from lowbit_comm._version import __version__

    assert RSAG_LOWBIT_COMM_VERSION == __version__
    assert RSAG_CUDA_EXTENSION_ABI == CUDA_ABI_VERSION
