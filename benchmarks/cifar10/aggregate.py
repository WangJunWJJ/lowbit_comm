import argparse
import csv
import json
import statistics
from pathlib import Path


def find_convergence(rows: list[dict], threshold: float, patience: int = 5):
    for start in range(len(rows) - patience + 1):
        window = rows[start : start + patience]
        if all(row["val_top1"] >= threshold for row in window):
            row = rows[start]
            return {
                "epoch": row["epoch"],
                "optimizer_steps": row["optimizer_steps"],
                "wall_s": row["wall_s"],
            }
    return None


def _read_epochs(run_dir: Path) -> list[dict]:
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if json.loads(line)["kind"] == "epoch"]


def aggregate(root: Path) -> dict:
    runs = {}
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows = _read_epochs(run_dir)
        if len(rows) != config["epochs"]:
            raise ValueError(f"incomplete run {run_dir}: {len(rows)} epochs")
        runs[(config["variant"], config["seed"])] = (config, rows)
    details = []
    for (variant, seed), (_, rows) in sorted(runs.items()):
        baseline = runs[("nccl_fp32", seed)][1]
        threshold = 0.99 * baseline[-1]["val_top1"]
        details.append(
            {
                "variant": variant,
                "seed": seed,
                "threshold": threshold,
                "final_top1": rows[-1]["val_top1"],
                "best_top1": max(row["val_top1"] for row in rows),
                "last5_top1": statistics.mean(row["val_top1"] for row in rows[-5:]),
                "convergence": find_convergence(rows, threshold),
                "images_per_s": statistics.mean(row["images_per_s"] for row in rows),
                "sync_ms": statistics.mean(row["sync_ms"] for row in rows),
                "peak_memory_mb": max(row["peak_memory_mb"] for row in rows),
            }
        )
    summary = {"runs": details, "variants": {}}
    for variant in sorted({row["variant"] for row in details}):
        selected = [row for row in details if row["variant"] == variant]
        summary["variants"][variant] = {
            key: {
                "mean": statistics.mean(row[key] for row in selected),
                "std": statistics.stdev(row[key] for row in selected),
            }
            for key in ("final_top1", "best_top1", "last5_top1", "images_per_s", "sync_ms")
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = result["runs"]
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "convergence"])
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "convergence"} for row in rows)


if __name__ == "__main__":
    main()
