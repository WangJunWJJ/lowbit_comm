from html import escape
import argparse
import json
from pathlib import Path


def render_bar_svg(values: dict[str, float], title: str, unit: str) -> str:
    width, height = 720, 100 + 55 * len(values)
    maximum = max(values.values(), default=1.0) or 1.0
    rows = []
    for index, (label, value) in enumerate(values.items()):
        y = 70 + index * 55
        bar_width = 480 * value / maximum
        rows.append(f'<text x="10" y="{y + 18}">{escape(label)}</text>')
        rows.append(f'<rect x="150" y="{y}" width="{bar_width:.1f}" height="24" fill="#4878CF"/>')
        rows.append(f'<text x="{160 + bar_width:.1f}" y="{y + 18}">{value:.3f} {escape(unit)}</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="10" y="30" font-size="20">{escape(title)}</text>{"".join(rows)}</svg>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    result = json.loads(args.summary.read_text(encoding="utf-8")); variants = result["variants"]
    charts = {
        "sync_speedup.svg": ({key: value["sync_speedup"] for key, value in variants.items()}, "梯度同步加速比", "x"),
        "throughput.svg": ({key: value["mean_tokens_per_sec"]["mean"] for key, value in variants.items()}, "训练吞吐", "tokens/s"),
        "perplexity.svg": ({key: value["final_perplexity"]["mean"] for key, value in variants.items()}, "最终验证集 Perplexity", "ppl"),
        "wall_time.svg": ({key: value["wall_time_sec"]["mean"] for key, value in variants.items()}, "端到端墙钟时间", "s"),
    }
    for filename, (values, title, unit) in charts.items():
        (args.output / filename).write_text(render_bar_svg(values, title, unit), encoding="utf-8")
    lines = ["# CCDL Qwen2-0.5B Alpaca 双卡训练报告", "", f"正式运行数：{result['run_count']}；收敛目标 perplexity：{result['perplexity_target']:.6f}。", "", "| 配置 | 最终 PPL | Token准确率 | 同步(ms) | 同步加速 | tokens/s | 吞吐加速 | 墙钟(s) | 收敛步数 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, value in variants.items():
        convergence = value.get("convergence_step", {}).get("mean")
        lines.append(f"| {key} | {value['final_perplexity']['mean']:.4f}±{value['final_perplexity']['std']:.4f} | {100*value['final_token_accuracy']['mean']:.2f}% | {value['mean_sync_ms']['mean']:.2f} | {value['sync_speedup']:.2f}x | {value['mean_tokens_per_sec']['mean']:.0f} | {value['throughput_speedup']:.2f}x | {value['wall_time_sec']['mean']:.1f} | {convergence if convergence is not None else '未收敛'} |")
    lines += ["", "说明：同步和吞吐均排除前 20 个优化步；收敛要求连续三个评估点达到目标。Pilot 与 smoke 不进入本表。"]
    (args.output / "CCDL_QWEN2_ALPACA_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
