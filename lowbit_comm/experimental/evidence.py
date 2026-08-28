"""Checksum-pinned loading for externally distributed RSAG evidence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Final

from lowbit_comm.experimental.rsag import RSAGEvidence


_MANIFEST_SCHEMA_VERSION: Final = 1
_QUALIFICATION_METRIC: Final = "external_runner_wall_gain_percent"
_MAX_MANIFEST_BYTES: Final = 1_048_576
_MANIFEST_FIELDS: Final = frozenset(
    {
        "manifest_schema_version",
        "qualification_metric",
        "source_commit",
        "build_fingerprint",
        "logical_bytes",
        "production_summary_sha256",
        "records",
    }
)
_RECORD_FIELDS: Final = frozenset(
    {
        "schema_version",
        "world_size",
        "node_count",
        "min_logical_bytes",
        "max_logical_bytes",
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
        "seed_speedups_percent",
        "quality_passed",
    }
)
_EVIDENCE_KEY_FIELDS: Final = (
    "world_size",
    "node_count",
    "min_logical_bytes",
    "max_logical_bytes",
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
)


@dataclass(frozen=True, slots=True)
class RSAGEvidenceManifest:
    """Immutable metadata and records loaded from one pinned manifest."""

    manifest_schema_version: int
    qualification_metric: str
    source_commit: str
    build_fingerprint: str
    logical_bytes: int
    production_summary_sha256: str
    manifest_sha256: str
    records: tuple[RSAGEvidence, ...]

    def __post_init__(self) -> None:
        if (
            type(self.manifest_schema_version) is not int
            or self.manifest_schema_version != _MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError("RSAG evidence manifest schema is unsupported")
        if (
            type(self.qualification_metric) is not str
            or self.qualification_metric != _QUALIFICATION_METRIC
        ):
            raise ValueError("RSAG evidence qualification metric is unsupported")
        _require_lower_hex(self.source_commit, 40, "source_commit")
        _require_sha256(self.build_fingerprint, "build_fingerprint")
        _require_sha256(
            self.production_summary_sha256,
            "production_summary_sha256",
        )
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        if type(self.logical_bytes) is not int or self.logical_bytes < 0:
            raise ValueError("RSAG evidence logical_bytes must be a non-negative int")
        if (
            type(self.records) is not tuple
            or not self.records
            or any(type(record) is not RSAGEvidence for record in self.records)
        ):
            raise ValueError("RSAG evidence records must be a non-empty exact tuple")
        _validate_record_identities(self)


def load_rsag_evidence_manifest(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> RSAGEvidenceManifest:
    """Load one explicit regular file only when its pinned SHA256 matches."""
    _require_sha256(expected_sha256, "expected_sha256")
    payload = _read_bounded_regular_file(path)
    manifest_sha256 = sha256(payload).hexdigest()
    if manifest_sha256 != expected_sha256:
        raise ValueError("RSAG evidence manifest checksum does not match the pin")
    document = _decode_json_document(payload)
    if type(document) is not dict or set(document) != _MANIFEST_FIELDS:
        raise ValueError("RSAG evidence manifest fields are invalid")
    records_value = document["records"]
    if type(records_value) is not list or not records_value:
        raise ValueError("RSAG evidence manifest records must be a non-empty list")
    records = tuple(
        _load_record(
            value,
            build_fingerprint=document["build_fingerprint"],
            logical_bytes=document["logical_bytes"],
        )
        for value in records_value
    )
    manifest = RSAGEvidenceManifest(
        manifest_schema_version=document["manifest_schema_version"],
        qualification_metric=document["qualification_metric"],
        source_commit=document["source_commit"],
        build_fingerprint=document["build_fingerprint"],
        logical_bytes=document["logical_bytes"],
        production_summary_sha256=document["production_summary_sha256"],
        manifest_sha256=manifest_sha256,
        records=records,
    )
    return manifest


def _read_bounded_regular_file(path: str | os.PathLike[str]) -> bytes:
    try:
        manifest_path = Path(path)
    except (TypeError, ValueError) as error:
        raise ValueError("RSAG evidence manifest path is invalid") from error
    try:
        path_facts = manifest_path.lstat()
    except OSError as error:
        raise ValueError("RSAG evidence manifest cannot be inspected") from error
    if stat.S_ISLNK(path_facts.st_mode):
        raise ValueError("RSAG evidence manifest must not be a symbolic link")
    if not stat.S_ISREG(path_facts.st_mode):
        raise ValueError("RSAG evidence manifest must be a regular file")
    if path_facts.st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("RSAG evidence manifest exceeds the size limit")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(manifest_path, flags)
    except OSError as error:
        raise ValueError("RSAG evidence manifest cannot be opened safely") from error
    try:
        opened_facts = os.fstat(descriptor)
        opened_identity = _file_identity(opened_facts)
        if not stat.S_ISREG(opened_facts.st_mode):
            raise ValueError("RSAG evidence manifest must be a regular file")
        if (
            path_facts.st_dev != opened_facts.st_dev
            or path_facts.st_ino != opened_facts.st_ino
        ):
            raise ValueError("RSAG evidence manifest changed while opening")
        if opened_facts.st_size > _MAX_MANIFEST_BYTES:
            raise ValueError("RSAG evidence manifest exceeds the size limit")
        chunks: list[bytes] = []
        remaining = _MAX_MANIFEST_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > _MAX_MANIFEST_BYTES:
            raise ValueError("RSAG evidence manifest exceeds the size limit")
        if opened_facts.st_size != len(payload):
            raise ValueError("RSAG evidence manifest changed while reading")
        if _file_identity(os.fstat(descriptor)) != opened_identity:
            raise ValueError("RSAG evidence manifest changed while reading")
        return payload
    except OSError as error:
        raise ValueError("RSAG evidence manifest cannot be read safely") from error
    finally:
        os.close(descriptor)


def _file_identity(facts: object) -> tuple[object, ...]:
    return (
        getattr(facts, "st_dev"),
        getattr(facts, "st_ino"),
        getattr(facts, "st_mode"),
        getattr(facts, "st_size"),
        getattr(facts, "st_mtime_ns"),
        getattr(facts, "st_ctime_ns"),
    )


def _decode_json_document(payload: bytes) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("RSAG evidence manifest must be UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("RSAG evidence manifest is not valid JSON") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"RSAG evidence manifest has duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"RSAG evidence manifest JSON constant is not finite: {value}")


def _load_record(
    value: object,
    *,
    build_fingerprint: object,
    logical_bytes: object,
) -> RSAGEvidence:
    if type(value) is not dict or set(value) != _RECORD_FIELDS:
        raise ValueError("RSAG evidence record fields are invalid")
    fields = dict(value)
    if (
        fields["build_fingerprint"] != build_fingerprint
        or fields["min_logical_bytes"] != logical_bytes
        or fields["max_logical_bytes"] != logical_bytes
    ):
        raise ValueError("RSAG evidence record differs from manifest identity")
    speedups = fields["seed_speedups_percent"]
    if type(speedups) is not list:
        raise ValueError("RSAG evidence seed speedups must be a list")
    fields["seed_speedups_percent"] = tuple(speedups)
    try:
        return RSAGEvidence(**fields)
    except (TypeError, ValueError) as error:
        raise ValueError("RSAG evidence record is invalid") from error


def _validate_record_identities(manifest: RSAGEvidenceManifest) -> None:
    keys: set[tuple[object, ...]] = set()
    for record in manifest.records:
        if (
            record.build_fingerprint != manifest.build_fingerprint
            or record.min_logical_bytes != manifest.logical_bytes
            or record.max_logical_bytes != manifest.logical_bytes
        ):
            raise ValueError("RSAG evidence record differs from manifest identity")
        key = tuple(getattr(record, field_name) for field_name in _EVIDENCE_KEY_FIELDS)
        if key in keys:
            raise ValueError("RSAG evidence manifest has a duplicate evidence key")
        keys.add(key)


def _require_sha256(value: object, name: str) -> None:
    _require_lower_hex(value, 64, name)


def _require_lower_hex(value: object, length: int, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"RSAG evidence {name} must be lowercase hexadecimal")


__all__ = ("RSAGEvidenceManifest", "load_rsag_evidence_manifest")
