"""Strict loading tests for externally distributed RSAG evidence."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lowbit_comm.experimental import (
    RSAGEvidence,
    RSAGEvidenceManifest,
    load_rsag_evidence_manifest,
)


_BUILD_FINGERPRINT = "a" * 64
_SUMMARY_SHA256 = "b" * 64
_SOURCE_COMMIT = "c" * 40
_LOGICAL_BYTES = 89_912_620


def _record(*, world_size: int = 2) -> dict[str, object]:
    return {
        "schema_version": 2,
        "world_size": world_size,
        "node_count": 2,
        "min_logical_bytes": _LOGICAL_BYTES,
        "max_logical_bytes": _LOGICAL_BYTES,
        "topology_class": "cross_node_socket",
        "transport": "nccl_socket_eno2",
        "gpu_model": "NVIDIA RTX A6000",
        "torch_version": "2.5.0a0+872d972e41.nv24.08",
        "cuda_version": "12.6",
        "nccl_version": "2.22.3",
        "lowbit_comm_version": "0.4.0.dev0",
        "cuda_extension_abi": 1,
        "checkpoint_schema_version": 2,
        "build_fingerprint": _BUILD_FINGERPRINT,
        "seed_speedups_percent": [3.75, 3.94, 3.98],
        "quality_passed": True,
    }


def _manifest() -> dict[str, object]:
    return {
        "manifest_schema_version": 1,
        "qualification_metric": "external_runner_wall_gain_percent",
        "source_commit": _SOURCE_COMMIT,
        "build_fingerprint": _BUILD_FINGERPRINT,
        "logical_bytes": _LOGICAL_BYTES,
        "production_summary_sha256": _SUMMARY_SHA256,
        "records": [_record()],
    }


def _write_manifest(path: Path, value: object) -> str:
    payload = json.dumps(value, sort_keys=True).encode("utf-8")
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def _evidence(**changes: object) -> RSAGEvidence:
    fields = _record()
    fields.update(changes)
    speedups = fields["seed_speedups_percent"]
    assert type(speedups) is list
    fields["seed_speedups_percent"] = tuple(speedups)
    return RSAGEvidence(**fields)


def test_loads_a_checksum_pinned_manifest_into_exact_frozen_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.json"
    digest = _write_manifest(path, _manifest())

    loaded = load_rsag_evidence_manifest(path, expected_sha256=digest)

    assert type(loaded) is RSAGEvidenceManifest
    assert loaded.manifest_schema_version == 1
    assert loaded.source_commit == _SOURCE_COMMIT
    assert loaded.build_fingerprint == _BUILD_FINGERPRINT
    assert loaded.logical_bytes == _LOGICAL_BYTES
    assert loaded.production_summary_sha256 == _SUMMARY_SHA256
    assert loaded.manifest_sha256 == digest
    assert type(loaded.records) is tuple
    assert len(loaded.records) == 1
    assert type(loaded.records[0]) is RSAGEvidence
    assert loaded.records[0].seed_speedups_percent == (3.75, 3.94, 3.98)


def test_rejects_manifest_when_pinned_checksum_differs(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    _write_manifest(path, _manifest())

    with pytest.raises(ValueError, match="checksum"):
        load_rsag_evidence_manifest(path, expected_sha256="d" * 64)


def test_rejects_symlink_even_when_target_checksum_matches(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    link = tmp_path / "evidence.json"
    digest = _write_manifest(target, _manifest())
    try:
        os.symlink(target, link)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="regular file|symbolic link"):
        load_rsag_evidence_manifest(link, expected_sha256=digest)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    payload = b'{"manifest_schema_version":1,"manifest_schema_version":1}'
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_rsag_evidence_manifest(
            path,
            expected_sha256=sha256(payload).hexdigest(),
        )


def test_rejects_nonfinite_json_numbers(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    payload = json.dumps(_manifest()).replace("3.75", "NaN").encode("utf-8")
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="finite|constant"):
        load_rsag_evidence_manifest(
            path,
            expected_sha256=sha256(payload).hexdigest(),
        )


@pytest.mark.parametrize("scope", ["manifest", "record"])
def test_rejects_unknown_fields(tmp_path: Path, scope: str) -> None:
    value = _manifest()
    if scope == "manifest":
        value["allow_unsafe"] = True
    else:
        records = value["records"]
        assert type(records) is list
        record = records[0]
        assert type(record) is dict
        record["allow_unsafe"] = True
    path = tmp_path / "evidence.json"
    digest = _write_manifest(path, value)

    with pytest.raises(ValueError, match="fields"):
        load_rsag_evidence_manifest(path, expected_sha256=digest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("build_fingerprint", "d" * 64),
        ("min_logical_bytes", _LOGICAL_BYTES - 1),
        ("max_logical_bytes", _LOGICAL_BYTES + 1),
    ],
)
def test_rejects_record_identity_that_drifts_from_manifest(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    manifest = _manifest()
    records = manifest["records"]
    assert type(records) is list
    record = records[0]
    assert type(record) is dict
    record[field] = value
    path = tmp_path / "evidence.json"
    digest = _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="manifest identity"):
        load_rsag_evidence_manifest(path, expected_sha256=digest)


def test_rejects_duplicate_evidence_keys(tmp_path: Path) -> None:
    manifest = _manifest()
    records = manifest["records"]
    assert type(records) is list
    duplicate = deepcopy(records[0])
    assert type(duplicate) is dict
    duplicate["seed_speedups_percent"] = [4.0, 4.1, 4.2]
    records.append(duplicate)
    path = tmp_path / "evidence.json"
    digest = _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="duplicate evidence key"):
        load_rsag_evidence_manifest(path, expected_sha256=digest)


def test_rejects_manifest_larger_than_the_bounded_input(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    payload = b" " * (1_048_576 + 1)
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="size limit"):
        load_rsag_evidence_manifest(
            path,
            expected_sha256=sha256(payload).hexdigest(),
        )


def test_rejects_file_identity_drift_while_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "evidence.json"
    digest = _write_manifest(path, _manifest())
    real_fstat = os.fstat
    calls = 0

    def drifting_fstat(descriptor: int) -> object:
        nonlocal calls
        facts = real_fstat(descriptor)
        calls += 1
        if calls == 1:
            return facts
        return SimpleNamespace(
            st_mode=facts.st_mode,
            st_dev=facts.st_dev,
            st_ino=facts.st_ino,
            st_size=facts.st_size,
            st_mtime_ns=facts.st_mtime_ns + 1,
            st_ctime_ns=facts.st_ctime_ns,
        )

    monkeypatch.setattr(os, "fstat", drifting_fstat)

    with pytest.raises(ValueError, match="changed while reading"):
        load_rsag_evidence_manifest(path, expected_sha256=digest)
    assert calls == 2


def test_direct_manifest_construction_rejects_record_identity_drift() -> None:
    with pytest.raises(ValueError, match="manifest identity"):
        RSAGEvidenceManifest(
            manifest_schema_version=1,
            qualification_metric="external_runner_wall_gain_percent",
            source_commit=_SOURCE_COMMIT,
            build_fingerprint=_BUILD_FINGERPRINT,
            logical_bytes=_LOGICAL_BYTES,
            production_summary_sha256=_SUMMARY_SHA256,
            manifest_sha256="d" * 64,
            records=(_evidence(build_fingerprint="e" * 64),),
        )


def test_direct_manifest_construction_rejects_duplicate_evidence_keys() -> None:
    evidence = _evidence()
    with pytest.raises(ValueError, match="duplicate evidence key"):
        RSAGEvidenceManifest(
            manifest_schema_version=1,
            qualification_metric="external_runner_wall_gain_percent",
            source_commit=_SOURCE_COMMIT,
            build_fingerprint=_BUILD_FINGERPRINT,
            logical_bytes=_LOGICAL_BYTES,
            production_summary_sha256=_SUMMARY_SHA256,
            manifest_sha256="d" * 64,
            records=(evidence, evidence),
        )
