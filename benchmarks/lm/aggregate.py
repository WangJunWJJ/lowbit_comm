import statistics
import argparse
import csv
import json
from pathlib import Path


def fp32_perplexity_target(final_perplexities: list[float]) -> float:
    return statistics.mean(final_perplexities) * 1.01


def convergence_step(evaluations: list[dict], target: float, persistence: int = 3):
    for index in range(len(evaluations) - persistence + 1):
        window = evaluations[index:index + persistence]
        if all(row["perplexity"] <= target for row in window):
            first = window[0]
            return first["step"], first["wall_time_sec"]
    return None


def summarize_values(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate(root: Path) -> dict:
    runs = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "COMPLETED.json").is_file():
            continue
        complete = json.loads((directory / "COMPLETED.json").read_text(encoding="utf-8"))
        rows = _read_jsonl(directory / "metrics.jsonl")
        steps = [row for row in rows if row["event"] == "step"]
        evaluations = [row for row in rows if row["event"] == "eval"]
        stable_steps = steps[20:] if len(steps) > 20 else steps
        runs.append({
            "variant": complete["variant"], "seed": complete["seed"],
            "steps": complete["steps"], "wall_time_sec": complete["wall_time_sec"],
            "final_perplexity": evaluations[-1]["perplexity"],
            "final_val_loss": evaluations[-1]["val_loss"],
            "final_token_accuracy": evaluations[-1]["token_accuracy"],
            "mean_sync_ms": statistics.mean(row["sync_ms"] for row in stable_steps),
            "mean_tokens_per_sec": statistics.mean(row["tokens_per_sec"] for row in stable_steps),
            "mean_communicated_bytes": statistics.mean(row["communicated_bytes"] for row in stable_steps),
            "peak_memory_mb": max(row["peak_memory_mb"] for row in steps),
            "evaluations": evaluations,
        })
    fp32 = [run["final_perplexity"] for run in runs if run["variant"] == "nccl_fp32"]
    if len(fp32) != 3:
        raise ValueError(f"expected 3 completed FP32 runs, found {len(fp32)}")
    target = fp32_perplexity_target(fp32)
    for run in runs:
        result = convergence_step(run.pop("evaluations"), target)
        run["convergence_step"] = result[0] if result else None
        run["convergence_wall_time_sec"] = result[1] if result else None
    variants = {}
    metric_names = ["wall_time_sec", "final_perplexity", "final_val_loss", "final_token_accuracy",
                    "mean_sync_ms", "mean_tokens_per_sec", "mean_communicated_bytes", "peak_memory_mb"]
    for variant in sorted({run["variant"] for run in runs}):
        selected = [run for run in runs if run["variant"] == variant]
        variants[variant] = {name: summarize_values([run[name] for run in selected]) for name in metric_names}
        converged = [run for run in selected if run["convergence_step"] is not None]
        variants[variant]["converged_runs"] = len(converged)
        if converged:
            variants[variant]["convergence_step"] = summarize_values([run["convergence_step"] for run in converged])
            variants[variant]["convergence_wall_time_sec"] = summarize_values([run["convergence_wall_time_sec"] for run in converged])
    baseline = variants["nccl_fp32"]
    for values in variants.values():
        values["sync_speedup"] = baseline["mean_sync_ms"]["mean"] / values["mean_sync_ms"]["mean"]
        values["throughput_speedup"] = values["mean_tokens_per_sec"]["mean"] / baseline["mean_tokens_per_sec"]["mean"]
    return {"perplexity_target": target, "run_count": len(runs), "runs": runs, "variants": variants}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = aggregate(args.input)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "runs.csv").open("w", newline="", encoding="utf-8") as output:
        fields = list(result["runs"][0])
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader(); writer.writerows(result["runs"])
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        fields = ["variant", "final_perplexity", "token_accuracy", "sync_ms", "sync_speedup", "tokens_per_sec", "throughput_speedup", "wall_time_sec", "convergence_step", "convergence_wall_time_sec", "converged_runs"]
        writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader()
        for variant, row in result["variants"].items():
            writer.writerow({"variant": variant, "final_perplexity": row["final_perplexity"]["mean"], "token_accuracy": row["final_token_accuracy"]["mean"], "sync_ms": row["mean_sync_ms"]["mean"], "sync_speedup": row["sync_speedup"], "tokens_per_sec": row["mean_tokens_per_sec"]["mean"], "throughput_speedup": row["throughput_speedup"], "wall_time_sec": row["wall_time_sec"]["mean"], "convergence_step": row.get("convergence_step", {}).get("mean"), "convergence_wall_time_sec": row.get("convergence_wall_time_sec", {}).get("mean"), "converged_runs": row["converged_runs"]})


if __name__ == "__main__":
    main()
