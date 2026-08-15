from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_LABELS = {"A>B", "B>A"}
SPLITS = ("train", "val", "test")


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


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clip_trajectory_id(clip_id: str) -> str:
    return re.sub(r"-C\d+$", "", clip_id or "")


def base_success_id(clip_id: str) -> str:
    traj = clip_trajectory_id(clip_id)
    match = re.match(r"^(SC|SP|PEG)-SUCC-\d+", traj)
    if match:
        return match.group(0)
    return traj


def pair_group_id(row: dict[str, str]) -> str:
    groups = sorted({base_success_id(row["clip_a_id"]), base_success_id(row["clip_b_id"])})
    return f"{row['task_id']}::" + "+".join(groups)


def clean_reason(row: dict[str, str]) -> str:
    label = row.get("final_preference_label_v0", "")
    if label not in VALID_LABELS:
        return "final_label_not_clear"
    if row.get("needs_manual_review_v0", "").lower() == "true":
        return "manual_review"
    if label != row.get("candidate_label", ""):
        return "final_label_candidate_mismatch"
    if as_float(row.get("preference_loss_weight_v0", "0")) <= 0:
        return "zero_weight"
    if row.get("clip_a_video_path_local") and not Path(row["clip_a_video_path_local"]).exists():
        return "clip_a_path_missing"
    if row.get("clip_b_video_path_local") and not Path(row["clip_b_video_path_local"]).exists():
        return "clip_b_path_missing"
    return "clean"


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def assign_groups(rows: list[dict[str, Any]], ratios: dict[str, float]) -> dict[str, str]:
    by_task_group: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_task_group[row["task_id"]][row["source_group_id"]].append(row)

    group_split: dict[str, str] = {}
    for task_id, groups in sorted(by_task_group.items()):
        total = sum(len(items) for items in groups.values())
        target = {split: ratios[split] * total for split in SPLITS}
        counts = {split: 0 for split in SPLITS}
        ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), stable_hash(item[0])))
        for group_id, items in ordered:
            # Greedy assignment to the split furthest below its target, with a small
            # deterministic hash tie-breaker.
            deficits = {
                split: target[split] - counts[split] + (stable_hash(group_id + split) % 1000) * 1e-9
                for split in SPLITS
            }
            split = max(SPLITS, key=lambda name: deficits[name])
            counts[split] += len(items)
            group_split[group_id] = split
    return group_split


def attach_split(rows: list[dict[str, str]], group_split: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["source_group_id"] = pair_group_id(row)
        item["split_v1"] = group_split[item["source_group_id"]]
        item["winner_clip_id_v1"] = row["clip_a_id"] if row["final_preference_label_v0"] == "A>B" else row["clip_b_id"]
        item["loser_clip_id_v1"] = row["clip_b_id"] if row["final_preference_label_v0"] == "A>B" else row["clip_a_id"]
        item["label_source_note_v1"] = "clean_fusion_stereo_v1_candidate_consistent"
        out.append(item)
    return out


def summarize(rows: list[dict[str, Any]], rejected: list[dict[str, Any]], ratios: dict[str, float]) -> dict[str, Any]:
    by_split = defaultdict(Counter)
    by_task = defaultdict(Counter)
    by_pair_type = defaultdict(Counter)
    by_label = defaultdict(Counter)
    by_source = defaultdict(Counter)
    group_by_split = defaultdict(set)
    preference_rows = Counter()
    order_rows = Counter()

    for row in rows:
        split = row["split_v1"]
        by_split[split]["rows"] += 1
        by_task[split][row["task_id"]] += 1
        by_pair_type[split][row["pair_type"]] += 1
        by_label[split][row["final_preference_label_v0"]] += 1
        by_source[split][row["fusion_label_source_v0"]] += 1
        group_by_split[split].add(row["source_group_id"])
        if row.get("use_for_preference_loss_v0") == "true":
            preference_rows[split] += 1
        if row.get("use_for_order_loss_v0") == "true":
            order_rows[split] += 1

    rejected_reasons = Counter(row["reject_reason_v1"] for row in rejected)
    return {
        "ratios_requested": ratios,
        "num_clean_rows": len(rows),
        "num_rejected_rows": len(rejected),
        "rejected_reasons": dict(rejected_reasons),
        "by_split": {split: dict(by_split[split]) for split in SPLITS},
        "source_groups_by_split": {split: len(group_by_split[split]) for split in SPLITS},
        "preference_loss_rows_by_split": dict(preference_rows),
        "order_loss_rows_by_split": dict(order_rows),
        "by_task": {split: dict(by_task[split]) for split in SPLITS},
        "by_pair_type": {split: dict(by_pair_type[split]) for split in SPLITS},
        "by_final_label": {split: dict(by_label[split]) for split in SPLITS},
        "by_fusion_source": {split: dict(by_source[split]) for split in SPLITS},
        "leakage_check": leakage_check(rows),
    }


def leakage_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_to_splits: dict[str, set[str]] = defaultdict(set)
    base_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_to_splits[row["source_group_id"]].add(row["split_v1"])
        for clip_key in ("clip_a_id", "clip_b_id"):
            base_to_splits[f"{row['task_id']}::{base_success_id(row[clip_key])}"].add(row["split_v1"])
    leaking_groups = {group: sorted(splits) for group, splits in group_to_splits.items() if len(splits) > 1}
    leaking_bases = {base: sorted(splits) for base, splits in base_to_splits.items() if len(splits) > 1}
    return {
        "source_group_leakage_count": len(leaking_groups),
        "base_success_id_leakage_count": len(leaking_bases),
        "source_group_leakage": leaking_groups,
        "base_success_id_leakage": leaking_bases,
    }


def markdown_counter_table(title: str, data: dict[str, dict[str, int]]) -> list[str]:
    keys = sorted({key for counter in data.values() for key in counter})
    lines = [f"## {title}", "", "| split | " + " | ".join(keys) + " |", "| --- | " + " | ".join("---:" for _ in keys) + " |"]
    for split in SPLITS:
        counter = data.get(split, {})
        lines.append("| " + split + " | " + " | ".join(str(counter.get(key, 0)) for key in keys) + " |")
    return lines


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dataset Split v1",
        "",
        f"- clean rows: {summary['num_clean_rows']}",
        f"- rejected rows: {summary['num_rejected_rows']}",
        f"- source group leakage count: {summary['leakage_check']['source_group_leakage_count']}",
        f"- base success id leakage count: {summary['leakage_check']['base_success_id_leakage_count']}",
        "",
        "## Split Counts",
        "",
        "| split | rows | source groups | preference rows | order rows |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        lines.append(
            f"| {split} | {summary['by_split'].get(split, {}).get('rows', 0)} | "
            f"{summary['source_groups_by_split'].get(split, 0)} | "
            f"{summary['preference_loss_rows_by_split'].get(split, 0)} | "
            f"{summary['order_loss_rows_by_split'].get(split, 0)} |"
        )
    lines += ["", "## Rejected Reasons", "", "| reason | count |", "| --- | ---: |"]
    for reason, count in sorted(summary["rejected_reasons"].items()):
        lines.append(f"| {reason} | {count} |")
    lines += [""] + markdown_counter_table("By Task", summary["by_task"])
    lines += [""] + markdown_counter_table("By Pair Type", summary["by_pair_type"])
    lines += [
        "",
        "## Filter",
        "",
        "Kept rows satisfy all conditions:",
        "",
        "- `final_preference_label_v0` is `A>B` or `B>A`.",
        "- `needs_manual_review_v0=false`.",
        "- `final_preference_label_v0 == candidate_label`.",
        "- `preference_loss_weight_v0 > 0`.",
        "- local A/B video paths exist.",
        "",
        "The split is grouped by source success trajectory id parsed from clip ids, so variants from the same success rollout do not cross train/val/test.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean train/val/test splits for fusion reward model data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    total_ratio = sum(ratios.values())
    ratios = {key: value / total_ratio for key, value in ratios.items()}

    rows = load_csv(args.input)
    clean: list[dict[str, str]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        reason = clean_reason(row)
        if reason == "clean":
            clean.append(row)
        else:
            item = dict(row)
            item["reject_reason_v1"] = reason
            rejected.append(item)

    with_group: list[dict[str, Any]] = []
    for row in clean:
        item = dict(row)
        item["source_group_id"] = pair_group_id(row)
        with_group.append(item)
    group_split = assign_groups(with_group, ratios)
    split_rows = attach_split(clean, group_split)

    summary = summarize(split_rows, rejected, ratios)
    write_csv(args.out / "clean_pairs_all.csv", split_rows)
    write_csv(args.out / "rejected_pairs_v1.csv", rejected)
    for split in SPLITS:
        split_items = [row for row in split_rows if row["split_v1"] == split]
        write_csv(args.out / f"{split}_pairs.csv", split_items)
        write_csv(args.out / f"{split}_preference_pairs.csv", [row for row in split_items if row["use_for_preference_loss_v0"] == "true"])
        write_csv(args.out / f"{split}_order_pairs.csv", [row for row in split_items if row["use_for_order_loss_v0"] == "true"])
    write_json(args.out / "split_summary.json", summary)
    write_report(args.out / "split_report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
