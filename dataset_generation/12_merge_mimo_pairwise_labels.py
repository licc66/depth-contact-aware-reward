from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PREF_ORDER = ("A>B", "B>A", "unsure")
AGREEMENT_ORDER = ("true", "false", "unsure")


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def counter_table(rows: list[dict[str, str]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row.get(group_key, "")][row.get("mimo_preference", "")] += 1
    table: list[dict[str, Any]] = []
    for group in sorted(grouped):
        counter = grouped[group]
        total = sum(counter.values())
        item: dict[str, Any] = {group_key: group, "total": total}
        for pref in PREF_ORDER:
            item[pref] = counter.get(pref, 0)
        item["candidate_agreement_rate_excluding_unsure"] = agreement_rate(
            [row for row in rows if row.get(group_key, "") == group]
        )
        table.append(item)
    return table


def agreement_rate(rows: list[dict[str, str]]) -> float | None:
    clear = [row for row in rows if row.get("mimo_preference") != "unsure"]
    if not clear:
        return None
    agree = sum(1 for row in clear if row.get("agrees_with_candidate") == "true")
    return round(agree / len(clear), 4)


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_pref = Counter(row.get("mimo_preference", "") for row in rows)
    by_agreement = Counter(row.get("agrees_with_candidate", "") for row in rows)
    by_need = Counter(row.get("needs_vlm_preference_label", "") for row in rows)
    pair_ids = [row.get("pair_id", "") for row in rows]
    duplicates = sorted([pair_id for pair_id, count in Counter(pair_ids).items() if pair_id and count > 1])
    return {
        "num_labels": len(rows),
        "num_unique_pair_ids": len(set(pair_ids)),
        "duplicate_pair_ids": duplicates,
        "needs_vlm_preference_label": dict(by_need),
        "preference_distribution": {key: by_pref.get(key, 0) for key in PREF_ORDER},
        "candidate_agreement": {key: by_agreement.get(key, 0) for key in AGREEMENT_ORDER},
        "clear_preference_candidate_agreement_rate": agreement_rate(rows),
        "by_task": counter_table(rows, "task_id"),
        "by_pair_type": counter_table(rows, "pair_type"),
        "by_label_source": counter_table(rows, "needs_vlm_preference_label"),
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            if value is None:
                value = "-"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(path: Path, summary: dict[str, Any], core_count: int, optional_count: int) -> None:
    pref = summary["preference_distribution"]
    agree = summary["candidate_agreement"]
    lines = [
        "# MiMo bootstrap pairwise labels 合并报告",
        "",
        "## 总览",
        "",
        f"- core/true pair: {core_count}",
        f"- optional intra-success temporal pair: {optional_count}",
        f"- 合并后 pair: {summary['num_labels']}",
        f"- 唯一 pair_id: {summary['num_unique_pair_ids']}",
        f"- 重复 pair_id 数: {len(summary['duplicate_pair_ids'])}",
        f"- 清晰偏好中与候选标签一致率: {summary['clear_preference_candidate_agreement_rate']}",
        "",
        "## MiMo 偏好分布",
        "",
        "| preference | count |",
        "| --- | --- |",
        f"| A>B | {pref.get('A>B', 0)} |",
        f"| B>A | {pref.get('B>A', 0)} |",
        f"| unsure | {pref.get('unsure', 0)} |",
        "",
        "## 与候选标签一致性",
        "",
        "| agrees_with_candidate | count |",
        "| --- | --- |",
        f"| true | {agree.get('true', 0)} |",
        f"| false | {agree.get('false', 0)} |",
        f"| unsure | {agree.get('unsure', 0)} |",
        "",
        "## 按任务统计",
        "",
        *markdown_table(
            summary["by_task"],
            ["task_id", "total", "A>B", "B>A", "unsure", "candidate_agreement_rate_excluding_unsure"],
        ),
        "",
        "## 按 pair 类型统计",
        "",
        *markdown_table(
            summary["by_pair_type"],
            ["pair_type", "total", "A>B", "B>A", "unsure", "candidate_agreement_rate_excluding_unsure"],
        ),
        "",
        "## 使用建议",
        "",
        "- `needs_vlm_preference_label=true` 是主训练 preference 数据，覆盖成功、near-miss、失败/截断之间的语义偏好。",
        "- `needs_vlm_preference_label=optional` 是成功轨迹内部时间顺序数据，建议作为 progress/order 辅助监督，权重低于 true pair。",
        "- MiMo 输出是视觉语义偏好，不是最终 reward 真值；后续应与 stereo/contact/stage 特征融合后再训练 reward/progress model。",
        "- `unsure` 不建议直接当负样本，可丢弃、降权，或作为不确定样本保留给后续主动筛查。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge core and optional MiMo pairwise labels.")
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--optional", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    core_rows = load_csv(args.core)
    optional_rows = load_csv(args.optional)
    rows = core_rows + optional_rows
    rows.sort(key=lambda item: (item.get("task_id", ""), item.get("needs_vlm_preference_label", ""), item.get("pair_id", "")))

    summary = summarize(rows)
    write_csv(args.out / "mimo_pairwise_labels_all.csv", rows)
    write_json(args.out / "mimo_pairwise_labels_all.json", rows)
    write_json(args.out / "mimo_pairwise_labels_all_summary.json", summary)
    write_report(args.out / "MiMo_pairwise_labels_all_report.md", summary, len(core_rows), len(optional_rows))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
