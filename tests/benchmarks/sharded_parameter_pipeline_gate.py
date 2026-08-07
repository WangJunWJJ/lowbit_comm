"""Correctness and median-throughput gate for the sharded parameter pipeline."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Any, Sequence


MODES = ("native_ddp", "full_fused", "sharded_fp", "sharded_compressed")


class GateFailure(RuntimeError):
    """Raised when benchmark evidence cannot authorize the fast path."""


def evaluate(
    results_dir: Path | str,
    *,
    world_size: int,
    min_speedup_vs_native: float = 1.05,
) -> dict[str, Any]:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if not isfinite(min_speedup_vs_native) or min_speedup_vs_native <= 0:
        raise ValueError("min_speedup_vs_native must be finite and > 0")
    root = Path(results_dir)
    grouped: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mode = payload.get("mode")
        if mode in grouped and int(payload.get("world_size", -1)) == world_size:
            grouped[mode].append(payload)
    for mode, trials in grouped.items():
        if len(trials) < 3:
            raise GateFailure(f"{mode} must provide at least three trials")

    expected_environment = None
    throughput: dict[str, list[float]] = {mode: [] for mode in MODES}
    for mode in MODES:
        for payload in grouped[mode]:
            environment = payload.get("benchmark_environment")
            if not isinstance(environment, dict):
                raise GateFailure("every trial must provide benchmark environment")
            signature = (
                tuple(environment.get("physical_gpu_ids", ())),
                environment.get("image_id"),
                environment.get("source_hash"),
                environment.get("extension_sha256"),
            )
            if expected_environment is None:
                expected_environment = signature
            elif signature != expected_environment:
                raise GateFailure("benchmark environment differs across trials")

            correctness = payload.get("correctness", {})
            if not correctness.get("finite_loss", False):
                raise GateFailure(f"{mode} produced a non-finite loss")
            difference = float(correctness.get("max_parameter_difference", float("nan")))
            if not isfinite(difference) or difference != 0.0:
                raise GateFailure(f"{mode} rank parameter difference must be exactly zero")
            final_loss = float(payload.get("loss", {}).get("final", float("nan")))
            if not isfinite(final_loss):
                raise GateFailure(f"{mode} final loss must be finite")
            value = float(
                payload.get("timing", {}).get(
                    "throughput_samples_per_second",
                    float("nan"),
                )
            )
            if not isfinite(value) or value <= 0:
                raise GateFailure(f"{mode} throughput must be finite and > 0")
            throughput[mode].append(value)

            if mode == "sharded_compressed":
                if payload.get("fallback_reason") is not None:
                    raise GateFailure("sharded_compressed trial used fallback")
                if payload.get("selected_fast_path") != "compressed_parameter_restore":
                    raise GateFailure("sharded_compressed trial did not select its fast path")

    medians = {mode: median(values) for mode, values in throughput.items()}
    speedup = medians["sharded_compressed"] / medians["native_ddp"]
    if speedup < min_speedup_vs_native:
        raise GateFailure(
            f"compressed speedup {speedup:.6f} is below required "
            f"{min_speedup_vs_native:.6f}"
        )
    return {
        "passed": True,
        "world_size": world_size,
        "trial_count": {mode: len(grouped[mode]) for mode in MODES},
        "native_median": medians["native_ddp"],
        "full_fused_median": medians["full_fused"],
        "sharded_fp_median": medians["sharded_fp"],
        "compressed_median": medians["sharded_compressed"],
        "speedup_vs_native": speedup,
        "speedup_vs_full_fused": (
            medians["sharded_compressed"] / medians["full_fused"]
        ),
        "benchmark_environment": {
            "physical_gpu_ids": list(expected_environment[0]),
            "image_id": expected_environment[1],
            "source_hash": expected_environment[2],
            "extension_sha256": expected_environment[3],
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--min-speedup-vs-native", type=float, default=1.05)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate(
        args.results_dir,
        world_size=args.world_size,
        min_speedup_vs_native=args.min_speedup_vs_native,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
