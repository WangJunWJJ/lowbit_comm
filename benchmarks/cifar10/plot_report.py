import argparse
import html
import json
from pathlib import Path


def render_bar_svg(values: dict[str, float], title: str, unit: str) -> str:
    width, height = 960, 520
    margin_left, margin_top, chart_height = 110, 70, 350
    bar_width = 110
    gap = (width - margin_left - 50 - bar_width * len(values)) / max(1, len(values))
    maximum = max(values.values()) * 1.1 or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-size="24" font-family="sans-serif">{html.escape(title)}</text>',
        f'<text x="25" y="250" transform="rotate(-90 25 250)" text-anchor="middle" font-size="15" font-family="sans-serif">{html.escape(unit)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top+chart_height}" x2="{width-35}" y2="{margin_top+chart_height}" stroke="#333"/>',
    ]
    for index, (label, value) in enumerate(values.items()):
        x = margin_left + gap / 2 + index * (bar_width + gap)
        bar_height = chart_height * value / maximum
        y = margin_top + chart_height - bar_height
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="#3978b5"/>',
                f'<text x="{x+bar_width/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="14" font-family="sans-serif">{value:.3f}</text>',
                f'<text x="{x+bar_width/2:.1f}" y="{margin_top+chart_height+24}" text-anchor="middle" font-size="12" font-family="sans-serif">{html.escape(label)}</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    variants = list(summary["variants"])
    accuracy = {name: summary["variants"][name]["final_top1"]["mean"] for name in variants}
    throughput = {name: summary["variants"][name]["images_per_s"]["mean"] for name in variants}
    sync = {name: summary["variants"][name]["sync_ms"]["mean"] for name in variants}
    (args.output / "final_accuracy.svg").write_text(render_bar_svg(accuracy, "Final validation Top-1", "%"), encoding="utf-8")
    (args.output / "throughput.svg").write_text(render_bar_svg(throughput, "Training throughput", "images/s"), encoding="utf-8")
    (args.output / "gradient_sync.svg").write_text(render_bar_svg(sync, "Gradient synchronization latency", "ms"), encoding="utf-8")
    lines = [
        "# CCDL CIFAR-10 双卡训练报告",
        "",
        "## 三种子汇总",
        "",
        "| 配置 | 最终 Top-1 | 吞吐 images/s | 同步 ms |",
        "|---|---:|---:|---:|",
    ]
    for name in variants:
        row = summary["variants"][name]
        lines.append(
            f"| {name} | {row['final_top1']['mean']:.3f} ± {row['final_top1']['std']:.3f} | "
            f"{row['images_per_s']['mean']:.1f} ± {row['images_per_s']['std']:.1f} | "
            f"{row['sync_ms']['mean']:.3f} ± {row['sync_ms']['std']:.3f} |"
        )
    lines += [
        "",
        "## 限制",
        "",
        "本实验采用受控的扁平梯度同步，不测量 DDP bucket 的计算通信重叠；结果来自单机双 GPU，尚不能外推到多机或语言模型训练。",
    ]
    (args.output / "CCDL_CIFAR10_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
