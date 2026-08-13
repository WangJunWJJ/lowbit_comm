"""Lossless evidence helpers shared by runnable benchmark entry points."""

from __future__ import annotations

import os
import platform
import statistics
from typing import Any


SCHEMA_VERSION = 1


def percentile(samples: list[float], value: float) -> float:
    if not samples:
        raise ValueError("samples must not be empty")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * value / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(samples: list[float]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must not be empty")
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "samples_ms": list(samples),
        "p50_ms": percentile(samples, 50),
        "p95_ms": percentile(samples, 95),
        "mean_ms": mean,
        "coefficient_of_variation": deviation / mean if mean else 0.0,
    }


def runtime_fingerprint(torch: Any, *, local_rank: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(local_rank)
    nccl_version = torch.cuda.nccl.version() if torch.distributed.is_nccl_available() else None
    return {
        "host_id": os.environ.get("LOWBIT_COMM_HOST_ID", platform.node()),
        "gpu_name": properties.name,
        "gpu_uuid": str(getattr(properties, "uuid", "unknown")),
        "gpu_capability": list(torch.cuda.get_device_capability(local_rank)),
        "gpu_visible_index": local_rank,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_version": nccl_version,
        "extension_path": os.environ.get("LOWBIT_COMM_CUDA_EXTENSION_PATH"),
        "commit": os.environ.get("LOWBIT_COMM_COMMIT", "unknown"),
    }
