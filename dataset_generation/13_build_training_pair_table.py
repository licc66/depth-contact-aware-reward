from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_PREFS = {"A>B", "B>A"}

BASE_COLUMNS = [
    "pair_id",
    "task_id",
    "split",
    "task_goal_text",
    "pair_type",
    "candidate_label",
    "candidate_confidence",
    "needs_vlm_preference_label",
    "clip_a_id",
    "clip_b_id",
    "clip_a_video_path_windows",
    "clip_a_start_frame",
    "clip_a_end_frame_exclusive",
    "clip_a_sample_frame_indices",
    "clip_b_video_path_windows",
    "clip_b_start_frame",
    "clip_b_end_frame_exclusive",
    "clip_b_sample_frame_indices",
    "rationale",
]

MIMO_COLUMNS = [
    "mimo_preference",
    "mimo_confidence",
    "agrees_with_candidate",
    "clip_a_stage",
    "clip_b_stage",
    "reason",
    "visible_failure_or_uncertainty",
    "mimo_model",
    "raw_response_path",
]

STEREO_COLUMNS = [
    "stereo_geometry_label_proxy",
    "stereo_geometry_label_agrees_with_pair_label",
    "clip_a_end_stage_proxy",
    "clip_b_end_stage_proxy",
    "clip_a_end_score_proxy",
    "clip_b_end_score_proxy",
    "score_diff_a_minus_b",
    "clip_a_end_dist_m",
    "clip_b_end_dist_m",
    "clip_a_end_depth_error_m",
    "clip_b_end_depth_error_m",
    "feature_source",
]

CONTACT_COLUMNS = [
    "contact_stage_label_proxy",
    "contact_stage_label_agrees_with_pair_label",
    "clip_a_end_stage_id",
    "clip_b_end_stage_id",
    "clip_a_end_stage_name",
    "clip_b_end_stage_name",
    "clip_a_end_stage_score",
    "clip_b_end_stage_score",
    "stage_score_diff_a_minus_b",
    "clip_a_support_contact_ratio",
    "clip_b_support_contact_ratio",
    "clip_a_grasp_ratio",
    "clip_b_grasp_ratio",
    "contact_feature_source",
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


def index_by_pair_id(paths: list[Path]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in load_csv(path):
            pair_id = row["pair_id"]
            if pair_id in index:
                raise ValueError(f"Duplicate pair_id across inputs: {pair_id}")
            index[pair_id] = row
    return index


def pref_relation(a: str, b: str, a_name: str, b_name: str) -> str:
    if a not in VALID_PREFS:
        return f"{a_name}_unsure"
    if b not in VALID_PREFS:
        return f"{b_name}_unsure"
    return "agree" if a == b else "conflict"


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def derive_use_hints(row: dict[str, Any]) -> None:
    need = str(row.get("needs_vlm_preference_label", "")).lower()
    mimo = row.get("mimo_preference", "")
    contact = row.get("contact_stage_label_proxy", "")
    stereo = row.get("stereo_geometry_label_proxy", "")
    candidate_agreement = row.get("mimo_agrees_with_candidate", "")

    contact_relation = pref_relation(mimo, contact, "mimo", "contact")
    stereo_relation = pref_relation(mimo, stereo, "mimo", "stereo")
    physical_conflict = contact_relation == "conflict" or stereo_relation == "conflict"

    row["mimo_clear_preference"] = bool_text(mimo in VALID_PREFS)
    row["contact_clear_preference"] = bool_text(contact in VALID_PREFS)
    row["stereo_clear_preference"] = bool_text(stereo in VALID_PREFS)
    row["mimo_vs_contact_stage_proxy"] = contact_relation
    row["mimo_vs_stereo_geometry_proxy"] = stereo_relation
    row["physical_proxy_conflict_with_mimo"] = bool_text(physical_conflict)

    weight = 0.0
    bucket = "no_preference_unsure"
    label = ""
    if mimo in VALID_PREFS and physical_conflict:
        bucket = "hard_conflict_review"
    elif mimo in VALID_PREFS and need == "true":
        label = mimo
        bucket = "main_preference"
        weight = 1.0 if candidate_agreement == "true" else 0.5
    elif mimo in VALID_PREFS and need == "optional":
        label = mimo
        if candidate_agreement == "true":
            bucket = "optional_order"
            weight = 0.4
        else:
            bucket = "low_weight_optional_or_candidate_conflict"
            weight = 0.1

    row["preference_label_hint_v0"] = label
    row["preference_loss_weight_hint_v0"] = weight
    row["supervision_bucket_v0"] = bucket


def add_prefixed(row: dict[str, Any], source: dict[str, str] | None, columns: list[str], prefix: str, rename: dict[str, str] | None = None) -> None:
    rename = rename or {}
    for col in columns:
        target = rename.get(col, f"{prefix}{col}")
        row[target] = "" if source is None else source.get(col, "")


def summarize(rows: list[dict[str, Any]], missing: dict[str, int]) -> dict[str, Any]:
    by_bucket = Counter(row["supervision_bucket_v0"] for row in rows)
    by_task_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    by_need_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    by_pref = Counter(row.get("mimo_preference", "") for row in rows)
    by_pair_type: dict[str, Counter[str]] = defaultdict(Counter)
    total_weight = 0.0
    nonzero_weight = 0
    for row in rows:
        by_task_bucket[row["task_id"]][row["supervision_bucket_v0"]] += 1
        by_need_bucket[row["needs_vlm_preference_label"]][row["supervision_bucket_v0"]] += 1
        by_pair_type[row["pair_type"]][row["supervision_bucket_v0"]] += 1
        weight = float(row["preference_loss_weight_hint_v0"])
        total_weight += weight
        nonzero_weight += int(weight > 0)
    return {
        "num_rows": len(rows),
        "missing_sources": missing,
        "mimo_preference_distribution": dict(by_pref),
        "supervision_bucket_v0": dict(by_bucket),
        "nonzero_preference_weight_rows": nonzero_weight,
        "sum_preference_weight_hint_v0": round(total_weight, 4),
        "by_task_bucket": {key: dict(value) for key, value in by_task_bucket.items()},
        "by_label_need_bucket": {key: dict(value) for key, value in by_need_bucket.items()},
        "by_pair_type_bucket": {key: dict(value) for key, value in by_pair_type.items()},
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Training pair table report",
        "",
        f"- rows: {summary['num_rows']}",
        f"- rows with nonzero preference weight hint: {summary['nonzero_preference_weight_rows']}",
        f"- sum preference weight hint: {summary['sum_preference_weight_hint_v0']}",
        f"- missing sources: {summary['missing_sources']}",
        "",
        "## MiMo preference distribution",
        "",
    ]
    for key, value in sorted(summary["mimo_preference_distribution"].items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Supervision bucket v0", ""]
    for key, value in sorted(summary["supervision_bucket_v0"].items()):
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "Notes:",
        "- This table is a joined data manifest, not a final reward label set.",
        "- `preference_label_hint_v0` and `preference_loss_weight_hint_v0` are conservative hints for an initial preference-loss experiment.",
        "- `hard_conflict_review` rows should be reviewed or handled by a later fusion rule before being used as strong labels.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join pair queues, MiMo labels, stereo labels, and contact/stage labels.")
    parser.add_argument("--queues", nargs="+", type=Path, required=True)
    parser.add_argument("--mimo-labels", type=Path, required=True)
    parser.add_argument("--stereo-pair-labels", nargs="+", type=Path, required=True)
    parser.add_argument("--contact-pair-labels", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue_rows = []
    for queue_path in args.queues:
        queue_rows.extend(load_csv(queue_path))

    mimo_index = index_by_pair_id([args.mimo_labels])
    stereo_index = index_by_pair_id(args.stereo_pair_labels)
    contact_index = index_by_pair_id(args.contact_pair_labels)

    rows: list[dict[str, Any]] = []
    missing = {"mimo": 0, "stereo": 0, "contact": 0}
    for queue_row in queue_rows:
        pair_id = queue_row["pair_id"]
        mimo = mimo_index.get(pair_id)
        stereo = stereo_index.get(pair_id)
        contact = contact_index.get(pair_id)
        missing["mimo"] += int(mimo is None)
        missing["stereo"] += int(stereo is None)
        missing["contact"] += int(contact is None)

        row: dict[str, Any] = {col: queue_row.get(col, "") for col in BASE_COLUMNS}
        add_prefixed(
            row,
            mimo,
            MIMO_COLUMNS,
            "mimo_",
            {
                "mimo_preference": "mimo_preference",
                "mimo_confidence": "mimo_confidence",
                "agrees_with_candidate": "mimo_agrees_with_candidate",
                "clip_a_stage": "mimo_clip_a_stage_text",
                "clip_b_stage": "mimo_clip_b_stage_text",
                "reason": "mimo_reason",
                "visible_failure_or_uncertainty": "mimo_visible_failure_or_uncertainty",
                "mimo_model": "mimo_model",
                "raw_response_path": "mimo_raw_response_path",
            },
        )
        add_prefixed(
            row,
            stereo,
            STEREO_COLUMNS,
            "stereo_",
            {
                "stereo_geometry_label_proxy": "stereo_geometry_label_proxy",
                "stereo_geometry_label_agrees_with_pair_label": "stereo_geometry_label_agrees_with_pair_label",
                "feature_source": "stereo_feature_source",
            },
        )
        add_prefixed(
            row,
            contact,
            CONTACT_COLUMNS,
            "contact_",
            {
                "contact_stage_label_proxy": "contact_stage_label_proxy",
                "contact_stage_label_agrees_with_pair_label": "contact_stage_label_agrees_with_pair_label",
                "contact_feature_source": "contact_feature_source",
            },
        )
        derive_use_hints(row)
        rows.append(row)

    rows.sort(key=lambda item: (item["task_id"], item["pair_id"]))
    summary = summarize(rows, missing)
    write_csv(args.out / "training_pairs_joined.csv", rows)
    write_json(args.out / "training_pairs_joined.json", rows)
    write_json(args.out / "training_pairs_joined_summary.json", summary)
    write_report(args.out / "training_pairs_joined_report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
