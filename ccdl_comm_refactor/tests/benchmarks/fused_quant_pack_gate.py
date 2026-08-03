from __future__ import annotations

import argparse
import math


def evaluate(candidate_ms: float, baseline_ms: float) -> list[str]:
    """Return performance-gate failures for residual quant-pack latency."""

    if not math.isfinite(candidate_ms) or candidate_ms <= 0:
        return [f"candidate latency must be finite and > 0: {candidate_ms}"]
    if not math.isfinite(baseline_ms) or baseline_ms <= 0:
        return [f"baseline latency must be finite and > 0: {baseline_ms}"]
    if candidate_ms > baseline_ms:
        return [f"residual quant-pack regression: {candidate_ms:.6f} > {baseline_ms:.6f} ms"]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate fused residual quant-pack latency")
    parser.add_argument("--baseline-ms", type=float, required=True)
    parser.add_argument("--candidate-ms", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = evaluate(candidate_ms=args.candidate_ms, baseline_ms=args.baseline_ms)
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        "residual quant-pack gate passed: "
        f"{args.candidate_ms:.6f} <= {args.baseline_ms:.6f} ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
