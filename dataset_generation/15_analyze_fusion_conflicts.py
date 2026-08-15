from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONFLICT_BUCKETS = {
    "contact_mimo_conflict_review": "contact_stage_label_proxy",
    "stereo_mimo_conflict_review": "stereo_geometry_label_proxy",
}


REVIEW_COLUMNS = [
    "pair_id",
    "task_id",
    "pair_type",
    "needs_vlm_preference_label",
    "candidate_label",
    "mimo_preference",
    "contact_stage_label_proxy",
    "stereo_geometry_label_proxy",
    "fusion_bucket_v0",
    "fusion_reason_v0",
    "mimo_reason",
    "mimo_visible_failure_or_uncertainty",
    "contact_clip_a_end_stage_name",
    "contact_clip_b_end_stage_name",
    "contact_stage_score_diff_a_minus_b",
    "stereo_score_diff_a_minus_b",
    "stereo_clip_a_end_dist_m",
    "stereo_clip_b_end_dist_m",
    "clip_a_video_path_local",
    "clip_b_video_path_local",
    "clip_a_start_frame",
    "clip_a_end_frame_exclusive",
    "clip_b_start_frame",
    "clip_b_end_frame_exclusive",
]


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


def pct(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def counter_table(rows: list[dict[str, str]], group_key: str) -> list[dict[str, Any]]:
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_group[row.get(group_key, "")][row.get("fusion_bucket_v0", "")] += 1
    out: list[dict[str, Any]] = []
    for group, counter in sorted(by_group.items()):
        item: dict[str, Any] = {group_key: group, "total": sum(counter.values())}
        item.update(dict(counter))
        out.append(item)
    return out


def conflict_accuracy(rows: list[dict[str, str]], bucket: str, heuristic_field: str) -> dict[str, Any]:
    items = [row for row in rows if row.get("fusion_bucket_v0") == bucket]
    total = len(items)
    heuristic_agree = sum(1 for row in items if row.get(heuristic_field) == row.get("candidate_label"))
    mimo_agree = sum(1 for row in items if row.get("mimo_preference") == row.get("candidate_label"))
    both_disagree = sum(
        1
        for row in items
        if row.get(heuristic_field) != row.get("candidate_label")
        and row.get("mimo_preference") != row.get("candidate_label")
    )
    return {
        "bucket": bucket,
        "heuristic_field": heuristic_field,
        "num_rows": total,
        "heuristic_agrees_candidate": heuristic_agree,
        "heuristic_agrees_candidate_rate": pct(heuristic_agree, total),
        "mimo_agrees_candidate": mimo_agree,
        "mimo_agrees_candidate_rate": pct(mimo_agree, total),
        "both_disagree_candidate": both_disagree,
        "by_task": dict(Counter(row.get("task_id", "") for row in items)),
        "by_pair_type": dict(Counter(row.get("pair_type", "") for row in items)),
    }


def combined_physical_conflict_accuracy(rows: list[dict[str, str]]) -> dict[str, Any]:
    items: list[tuple[dict[str, str], str]] = []
    for bucket, field in CONFLICT_BUCKETS.items():
        items.extend((row, field) for row in rows if row.get("fusion_bucket_v0") == bucket)
    total = len(items)
    heuristic_agree = sum(1 for row, field in items if row.get(field) == row.get("candidate_label"))
    mimo_agree = sum(1 for row, _ in items if row.get("mimo_preference") == row.get("candidate_label"))
    return {
        "num_rows": total,
        "heuristic_agrees_candidate": heuristic_agree,
        "heuristic_agrees_candidate_rate": pct(heuristic_agree, total),
        "mimo_agrees_candidate": mimo_agree,
        "mimo_agrees_candidate_rate": pct(mimo_agree, total),
    }


def select_review_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("needs_manual_review_v0") != "true":
            continue
        selected.append({col: row.get(col, "") for col in REVIEW_COLUMNS})
    return selected


def markdown_counter(counter: dict[str, int], title: str) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| item | count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {key} | {value} |")
    return lines


def markdown_accuracy(item: dict[str, Any]) -> list[str]:
    return [
        f"### {item['bucket']}",
        "",
        f"- rows: {item['num_rows']}",
        f"- heuristic field: `{item['heuristic_field']}`",
        f"- heuristic agrees with candidate: {item['heuristic_agrees_candidate']} ({item['heuristic_agrees_candidate_rate']})",
        f"- MiMo agrees with candidate: {item['mimo_agrees_candidate']} ({item['mimo_agrees_candidate_rate']})",
        f"- both disagree with candidate: {item['both_disagree_candidate']}",
        "",
        *markdown_counter(item["by_task"], "by task"),
        "",
        *markdown_counter(item["by_pair_type"], "by pair type"),
    ]


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Fusion Conflict Diagnostics v0",
        "",
        "Important caveat: `candidate_label` is a constructed weak reference from success/failure/near-miss or temporal-order rules. It is useful for diagnosing whether a proxy matches the dataset design, but it is not human-verified ground truth.",
        "",
        f"- total rows: {summary['total_rows']}",
        f"- manual review rows: {summary['manual_review_rows']}",
        f"- MiMo-vs-physical clear conflicts: {summary['combined_physical_conflicts']['num_rows']}",
        f"- physical side agrees with candidate in clear conflicts: {summary['combined_physical_conflicts']['heuristic_agrees_candidate']} ({summary['combined_physical_conflicts']['heuristic_agrees_candidate_rate']})",
        f"- MiMo agrees with candidate in clear conflicts: {summary['combined_physical_conflicts']['mimo_agrees_candidate']} ({summary['combined_physical_conflicts']['mimo_agrees_candidate_rate']})",
        "",
        "## Conflict Bucket Counts",
        "",
        *markdown_counter(summary["review_bucket_counts"], "review buckets"),
        "",
        "## Candidate-Agreement Diagnostics",
        "",
    ]
    for item in summary["conflict_accuracy"]:
        lines.extend(markdown_accuracy(item))
        lines.append("")
    lines.extend(
        [
            "## Initial Interpretation",
            "",
            "- contact/stage conflicts are the strongest signal in v0: they agree with the constructed candidate labels much more often than MiMo in the conflict subset.",
            "- stereo/depth conflicts are weaker: many stereo-only conflicts disagree with the candidate label, so stereo thresholds and task-specific geometry should be improved before treating them as hard constraints.",
            "- The next iteration should improve heuristic correctness by separating contact/stage hard gates from stereo/depth soft progress, and by manually auditing a small stratified sample of review rows.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze conflicts between fusion heuristics and MiMo labels.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_csv(args.input)
    review_rows = [row for row in rows if row.get("needs_manual_review_v0") == "true"]
    summary = {
        "total_rows": len(rows),
        "manual_review_rows": len(review_rows),
        "review_bucket_counts": dict(Counter(row.get("fusion_bucket_v0", "") for row in review_rows)),
        "review_by_task": counter_table(review_rows, "task_id"),
        "review_by_pair_type": counter_table(review_rows, "pair_type"),
        "conflict_accuracy": [
            conflict_accuracy(rows, bucket, field) for bucket, field in CONFLICT_BUCKETS.items()
        ],
        "combined_physical_conflicts": combined_physical_conflict_accuracy(rows),
    }
    write_json(args.out / "conflict_diagnostics_summary_v0.json", summary)
    write_csv(args.out / "manual_review_pairs_diagnostic_v0.csv", select_review_rows(rows))
    write_report(args.out / "conflict_diagnostics_report_v0.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
