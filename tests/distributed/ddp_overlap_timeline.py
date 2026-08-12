"""A6000 oracle proving that async CCDL has a non-zero CUDA timeline overlap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from examples.ddp_training import run_training  # noqa: E402 - path bootstrap above
from examples.training.config import TrainingConfig  # noqa: E402 - path bootstrap above


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--batch-size-per-rank", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1760)
    args = parser.parse_args()

    result = run_training(
        TrainingConfig(
            mode="ccdl_async",
            synthetic=True,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            batch_size_per_rank=args.batch_size_per_rank,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            num_classes=args.num_classes,
            device="cuda",
            dtype="fp16",
            bucket_cap_mb=1,
            output=Path("dist/ddp-overlap-timeline.json"),
        )
    )
    if result is None:
        return
    timing = result["timing"]
    if timing["communication_ms"] <= 0:
        raise AssertionError("timeline did not record bucket communication")
    if timing["compute_ms"] <= 0:
        raise AssertionError("timeline did not record backward computation")
    intersection = (
        timing["communication_ms"]
        + timing["compute_ms"]
        - timing["overlapped_ms"]
    )
    if intersection <= 0:
        raise AssertionError("async communication did not intersect backward timeline")
    if timing["overlap_classification"] != "timeline_overlapped":
        raise AssertionError("async label was emitted without matching timeline evidence")
    if not result["correctness"]["rank_parameters_consistent"]:
        raise AssertionError("optimizer consumed inconsistent reduced gradients")
    if not result["correctness"]["finite_loss"]:
        raise AssertionError("training produced non-finite loss")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
