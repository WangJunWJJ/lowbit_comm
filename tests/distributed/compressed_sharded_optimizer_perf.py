"""Distributed synthetic runner for the four CCDL parameter pipeline modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.training.compressed_sharded_optimizer import (
    MODES,
    CompressedShardedRunConfig,
    run_training,
)
from examples.training.config import TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--model-numel", type=int, default=44_960_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _hidden_dim_for(target: int, *, input_dim: int, depth: int, classes: int) -> int:
    if target < 1:
        raise ValueError("model_numel must be >= 1")
    quadratic = depth - 1
    linear = input_dim + classes + depth
    estimate = max(
        1,
        int((-linear + math.sqrt(linear * linear + 4 * quadratic * target)) / (2 * quadratic)),
    )

    def count(hidden: int) -> int:
        return (
            input_dim * hidden
            + hidden
            + (depth - 1) * (hidden * hidden + hidden)
            + hidden * classes
            + classes
        )

    return min(range(max(1, estimate - 4), estimate + 5), key=lambda value: abs(count(value) - target))


def _source_hash(repository: Path) -> str:
    override = os.environ.get("CCDL_SOURCE_HASH")
    if override:
        return override
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _extension_hash(repository: Path) -> str:
    candidates = sorted(repository.glob("ccdl_cuda_ops*.so"))
    if not candidates:
        return "missing"
    digest = hashlib.sha256()
    with candidates[0].open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if actual_world_size != args.world_size:
        raise RuntimeError(
            f"WORLD_SIZE={actual_world_size} does not match --world-size={args.world_size}"
        )
    repository = Path(__file__).resolve().parents[2]
    hidden_dim = _hidden_dim_for(
        args.model_numel,
        input_dim=1024,
        depth=3,
        classes=1760,
    )
    config = CompressedShardedRunConfig(
        mode=args.mode,
        training=TrainingConfig(
            mode="ccdl_sync" if args.mode == "full_fused" else "native_ddp",
            synthetic=True,
            steps=args.steps,
            warmup_steps=args.warmup,
            batch_size_per_rank=args.batch_size,
            input_dim=1024,
            hidden_dim=hidden_dim,
            depth=3,
            num_classes=1760,
            device="cuda",
            dtype=args.dtype,
            output=args.output_json,
        ),
    )
    payload = run_training(config)
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return 0
    if payload is None:
        raise RuntimeError("rank 0 did not receive benchmark metrics")
    payload["requested_model_numel"] = args.model_numel
    payload["benchmark_environment"] = {
        "physical_gpu_ids": [
            int(value)
            for value in os.environ.get(
                "CCDL_PHYSICAL_GPU_IDS",
                os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            ).split(",")
            if value.strip()
        ],
        "image_id": os.environ.get("CCDL_IMAGE_ID", "unknown"),
        "source_hash": _source_hash(repository),
        "extension_sha256": _extension_hash(repository),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
