from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


VALID = {"A>B", "B>A"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def setup_font() -> None:
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def physical_label(row: dict[str, str]) -> str:
    contact = row.get("contact_stage_label_proxy", "")
    stereo = row.get("stereo_geometry_label_proxy", "")
    contact_clear = contact in VALID
    stereo_clear = stereo in VALID
    if contact_clear and stereo_clear:
        return contact if contact == stereo else "unsure"
    if contact_clear:
        return contact
    if stereo_clear:
        return stereo
    return "unsure"


def branch_metrics(rows: list[dict[str, str]], name: str, labels: list[str]) -> dict[str, Any]:
    total = len(rows)
    clear = [idx for idx, label in enumerate(labels) if label in VALID]
    correct = sum(1 for idx in clear if labels[idx] == rows[idx].get("candidate_label", ""))
    wrong = len(clear) - correct
    unsure = total - len(clear)
    return {
        "branch": name,
        "total": total,
        "clear": len(clear),
        "correct": correct,
        "wrong": wrong,
        "unsure": unsure,
        "coverage": len(clear) / total if total else 0.0,
        "clear_precision": correct / len(clear) if clear else 0.0,
        "all_pair_agreement": correct / total if total else 0.0,
    }


def annotate_bars(ax, bars, fmt="{:.1f}%") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height * 100),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1F2937",
        )


def plot(metrics: list[dict[str, Any]], out_png: Path, out_svg: Path) -> None:
    setup_font()
    names = [item["branch"] for item in metrics]
    coverage = [item["coverage"] for item in metrics]
    precision = [item["clear_precision"] for item in metrics]
    correct = [item["correct"] for item in metrics]
    wrong = [item["wrong"] for item in metrics]
    unsure = [item["unsure"] for item in metrics]
    colors = ["#2F6CE5", "#0E8C78"]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=180)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.38, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    x = np.arange(len(names))
    bars1 = ax1.bar(x, coverage, color=colors, width=0.56)
    ax1.set_title("清晰偏好覆盖率", fontsize=14, fontweight="bold")
    ax1.set_xticks(x, names)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Clear / All pairs")
    ax1.grid(axis="y", alpha=0.25)
    annotate_bars(ax1, bars1)

    bars2 = ax2.bar(x, precision, color=colors, width=0.56)
    ax2.set_title("清晰标签准确率", fontsize=14, fontweight="bold")
    ax2.set_xticks(x, names)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Correct / Clear")
    ax2.grid(axis="y", alpha=0.25)
    annotate_bars(ax2, bars2)

    width = 0.26
    ax3.bar(x - width, correct, width, label="与 weak reference 一致", color="#17A673")
    ax3.bar(x, wrong, width, label="与 weak reference 冲突", color="#D94B5A")
    ax3.bar(x + width, unsure, width, label="unsure / 无清晰偏好", color="#9AA4B2")
    ax3.set_title("样本数分布", fontsize=14, fontweight="bold")
    ax3.set_xticks(x, names)
    ax3.set_ylabel("Pair count")
    ax3.grid(axis="y", alpha=0.25)
    ax3.legend(loc="upper center", ncol=3, frameon=False)
    for xpos, values in zip((x - width, x, x + width), (correct, wrong, unsure)):
        for xi, yi in zip(xpos, values):
            ax3.text(xi, yi + 18, str(int(yi)), ha="center", va="bottom", fontsize=10, color="#1F2937")

    fig.suptitle("LLM 语义判断 vs 物理规则判断", fontsize=20, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.025,
        "注：这里的正确/错误相对 candidate_label（由成功/失败/near-miss/时间顺序构造的 weak reference），不是人工真值。",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, metrics: list[dict[str, Any]]) -> None:
    lines = [
        "# LLM vs Physical Branch Comparison",
        "",
        "| branch | total | clear | coverage | correct | wrong | unsure | clear precision | all-pair agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in metrics:
        lines.append(
            f"| {item['branch']} | {item['total']} | {item['clear']} | "
            f"{item['coverage']:.3f} | {item['correct']} | {item['wrong']} | {item['unsure']} | "
            f"{item['clear_precision']:.3f} | {item['all_pair_agreement']:.3f} |"
        )
    lines += [
        "",
        "解释：",
        "",
        "- LLM/MiMo 分支表示商业 VLM 对 A/B 视频片段的语义偏好判断。",
        "- 物理分支表示 contact/stage 与 stereo/depth 的保守合并判断。",
        "- `clear precision` 只在分支给出 A>B 或 B>A 时计算。",
        "- `all-pair agreement` 把 unsure 也计入分母，更接近整体可用率。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LLM/MiMo branch vs physical branch diagnostics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"D:\Users\User\Desktop\reward_model_dataset\fusion_labels\bootstrap_v1_fusion_stereo_v1\final_pair_labels_v0.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"D:\Users\User\Desktop\reward_model_dataset\figures\llm_vs_physical"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_csv(args.input)
    mimo_labels = [row.get("mimo_preference", "") for row in rows]
    physical_labels = [physical_label(row) for row in rows]
    metrics = [
        branch_metrics(rows, "LLM / MiMo", mimo_labels),
        branch_metrics(rows, "Physical", physical_labels),
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "llm_vs_physical_metrics.csv", metrics)
    (args.out_dir / "llm_vs_physical_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.out_dir / "llm_vs_physical_report.md", metrics)
    plot(
        metrics,
        args.out_dir / "llm_vs_physical_comparison.png",
        args.out_dir / "llm_vs_physical_comparison.svg",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(args.out_dir / "llm_vs_physical_comparison.png")


if __name__ == "__main__":
    main()
