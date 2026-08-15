from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_PREFS = {"A>B", "B>A"}


KEEP_COLUMNS = [
    "pair_id",
    "task_id",
    "split",
    "task_goal_text",
    "pair_type",
    "needs_vlm_preference_label",
    "candidate_label",
    "candidate_confidence",
    "clip_a_id",
    "clip_b_id",
    "clip_a_video_path_windows",
    "clip_b_video_path_windows",
    "clip_a_start_frame",
    "clip_a_end_frame_exclusive",
    "clip_a_sample_frame_indices",
    "clip_b_start_frame",
    "clip_b_end_frame_exclusive",
    "clip_b_sample_frame_indices",
    "mimo_preference",
    "mimo_confidence",
    "mimo_agrees_with_candidate",
    "mimo_clip_a_stage_text",
    "mimo_clip_b_stage_text",
    "mimo_reason",
    "mimo_visible_failure_or_uncertainty",
    "stereo_geometry_label_proxy",
    "stereo_score_diff_a_minus_b",
    "stereo_clip_a_end_score_proxy",
    "stereo_clip_b_end_score_proxy",
    "stereo_clip_a_end_dist_m",
    "stereo_clip_b_end_dist_m",
    "stereo_clip_a_end_depth_error_m",
    "stereo_clip_b_end_depth_error_m",
    "contact_stage_label_proxy",
    "contact_clip_a_end_stage_id",
    "contact_clip_b_end_stage_id",
    "contact_clip_a_end_stage_name",
    "contact_clip_b_end_stage_name",
    "contact_stage_score_diff_a_minus_b",
    "contact_clip_a_grasp_ratio",
    "contact_clip_b_grasp_ratio",
    "contact_clip_a_support_contact_ratio",
    "contact_clip_b_support_contact_ratio",
    "preference_label_hint_v0",
    "preference_loss_weight_hint_v0",
    "supervision_bucket_v0",
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


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clear_pref(value: str) -> bool:
    return value in VALID_PREFS


def opposite(label: str) -> str:
    if label == "A>B":
        return "B>A"
    if label == "B>A":
        return "A>B"
    return "unsure"


def relation(lhs: str, rhs: str, rhs_name: str) -> str:
    if not clear_pref(lhs):
        return "mimo_unsure"
    if not clear_pref(rhs):
        return f"{rhs_name}_unsure"
    return "agree" if lhs == rhs else "conflict"


def remap_path(value: str, old_root: str, new_root: str) -> str:
    if not value:
        return ""
    old = old_root.rstrip("\\/")
    new = new_root.rstrip("\\/")
    if value.lower().startswith(old.lower()):
        return new + value[len(old) :]
    return value


def choose_from_physics(row: dict[str, str], need: str) -> tuple[str, float, str, str]:
    contact = row["contact_stage_label_proxy"]
    stereo = row["stereo_geometry_label_proxy"]
    if clear_pref(contact) and clear_pref(stereo):
        if contact == stereo:
            weight = 0.60 if need == "true" else 0.25
            return contact, weight, "physical_consensus_without_mimo", "MiMo unsure; contact/stage and stereo/depth agree."
        return "", 0.0, "physical_proxy_conflict_review", "MiMo unsure; contact/stage and stereo/depth conflict."
    if clear_pref(contact):
        weight = 0.50 if need == "true" else 0.20
        return contact, weight, "contact_stage_only_without_mimo", "MiMo unsure; using contact/stage as conservative stage order."
    if clear_pref(stereo):
        weight = 0.35 if need == "true" else 0.15
        return stereo, weight, "stereo_geometry_only_without_mimo", "MiMo unsure; using stereo/depth as weak stage-internal progress."
    return "", 0.0, "no_preference_unsure", "No clear MiMo, contact/stage, or stereo/depth preference."


def fuse_row(row: dict[str, str], stereo_conflict_margin: float) -> dict[str, Any]:
    need = row.get("needs_vlm_preference_label", "").lower()
    mimo = row.get("mimo_preference", "")
    contact = row.get("contact_stage_label_proxy", "")
    stereo = row.get("stereo_geometry_label_proxy", "")
    candidate = row.get("candidate_label", "")
    mimo_conf = as_float(row.get("mimo_confidence", ""), 0.0)
    stereo_margin = abs(as_float(row.get("stereo_score_diff_a_minus_b", ""), 0.0))

    contact_rel = relation(mimo, contact, "contact")
    stereo_rel = relation(mimo, stereo, "stereo")
    physical_support_count = int(contact_rel == "agree") + int(stereo_rel == "agree")
    physics_clear_count = int(clear_pref(contact)) + int(clear_pref(stereo))

    final_label = ""
    weight = 0.0
    bucket = "no_preference_unsure"
    source = "none"
    reason = ""
    use_for_preference_loss = "false"
    use_for_order_loss = "false"
    needs_manual_review = "false"

    if clear_pref(mimo):
        if contact_rel == "conflict":
            bucket = "contact_mimo_conflict_review"
            source = "review"
            reason = "Contact/stage proxy conflicts with MiMo; contact is treated as a hard stage gate in v0."
            needs_manual_review = "true"
        elif stereo_rel == "conflict" and contact_rel != "agree" and stereo_margin > stereo_conflict_margin:
            bucket = "stereo_mimo_conflict_review"
            source = "review"
            reason = "Stereo/depth proxy strongly conflicts with MiMo and contact/stage does not support MiMo."
            needs_manual_review = "true"
        else:
            final_label = mimo
            source = "mimo_semantic"
            if need == "optional":
                use_for_order_loss = "true"
                if candidate == mimo:
                    bucket = "optional_order_semantic"
                    weight = 0.35
                    reason = "Optional temporal pair; MiMo agrees with temporal candidate."
                else:
                    bucket = "optional_order_low_confidence"
                    weight = 0.10
                    reason = "Optional temporal pair; MiMo is clear but disagrees with temporal candidate."
            else:
                use_for_preference_loss = "true"
                bucket = "main_semantic_preference"
                weight = 1.0
                reason = "Core pair; MiMo gives clear semantic preference without hard contact/stage conflict."
                if candidate and candidate != mimo:
                    weight *= 0.55
                    bucket = "main_semantic_candidate_conflict"
                    reason += " Candidate rule disagrees, so weight is reduced."
                if physics_clear_count == 0:
                    weight *= 0.85
                    reason += " Physical proxies are unsure, so weight is slightly reduced."
                elif physical_support_count == 0:
                    weight *= 0.70
                    bucket = "main_semantic_soft_physics_conflict"
                    reason += " Stereo/depth conflict is kept only as a soft warning in v0."
                elif physical_support_count >= 1:
                    reason += " At least one physical proxy supports MiMo."

            if stereo_rel == "conflict" and bucket not in {"stereo_mimo_conflict_review", "contact_mimo_conflict_review"}:
                weight *= 0.80
                reason += " Stereo/depth conflicts, so weight is softened."

    else:
        final_label, weight, bucket, reason = choose_from_physics(row, need)
        if final_label:
            source = "physical_proxy"
            if need == "optional":
                use_for_order_loss = "true"
            else:
                use_for_preference_loss = "true"
        else:
            source = "none"
            if bucket.endswith("_review"):
                needs_manual_review = "true"

    weight = round(max(0.0, min(1.0, weight)), 4)
    return {
        "final_preference_label_v0": final_label,
        "final_preference_winner_clip_v0": "A" if final_label == "A>B" else ("B" if final_label == "B>A" else ""),
        "final_preference_loser_clip_v0": "B" if final_label == "A>B" else ("A" if final_label == "B>A" else ""),
        "final_opposite_label_v0": opposite(final_label),
        "preference_loss_weight_v0": weight,
        "use_for_preference_loss_v0": use_for_preference_loss if weight > 0 else "false",
        "use_for_order_loss_v0": use_for_order_loss if weight > 0 else "false",
        "fusion_bucket_v0": bucket,
        "fusion_label_source_v0": source,
        "fusion_reason_v0": reason,
        "needs_manual_review_v0": needs_manual_review,
        "mimo_vs_contact_stage_v0": contact_rel,
        "mimo_vs_stereo_geometry_v0": stereo_rel,
        "physical_support_count_v0": physical_support_count,
        "physical_clear_count_v0": physics_clear_count,
        "stereo_conflict_margin_v0": round(stereo_margin, 6),
        "mimo_confidence_float": round(mimo_conf, 4),
    }


def build_rows(rows: list[dict[str, str]], old_root: str, new_root: str, stereo_conflict_margin: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {col: row.get(col, "") for col in KEEP_COLUMNS}
        item["clip_a_video_path_local"] = remap_path(row.get("clip_a_video_path_windows", ""), old_root, new_root)
        item["clip_b_video_path_local"] = remap_path(row.get("clip_b_video_path_windows", ""), old_root, new_root)
        item.update(fuse_row(row, stereo_conflict_margin))
        output.append(item)
    output.sort(key=lambda x: (x["task_id"], x["pair_id"]))
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket = Counter(row["fusion_bucket_v0"] for row in rows)
    by_label = Counter(row["final_preference_label_v0"] or "unsure" for row in rows)
    by_source = Counter(row["fusion_label_source_v0"] for row in rows)
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    by_pair_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_need: dict[str, Counter[str]] = defaultdict(Counter)
    total_weight = 0.0
    preference_rows = 0
    order_rows = 0
    review_rows = 0
    for row in rows:
        bucket = row["fusion_bucket_v0"]
        by_task[row["task_id"]][bucket] += 1
        by_pair_type[row["pair_type"]][bucket] += 1
        by_need[row["needs_vlm_preference_label"]][bucket] += 1
        weight = float(row["preference_loss_weight_v0"])
        total_weight += weight
        preference_rows += int(row["use_for_preference_loss_v0"] == "true")
        order_rows += int(row["use_for_order_loss_v0"] == "true")
        review_rows += int(row["needs_manual_review_v0"] == "true")
    return {
        "num_rows": len(rows),
        "final_preference_label_v0": dict(by_label),
        "fusion_bucket_v0": dict(by_bucket),
        "fusion_label_source_v0": dict(by_source),
        "use_for_preference_loss_rows": preference_rows,
        "use_for_order_loss_rows": order_rows,
        "manual_review_rows": review_rows,
        "sum_preference_loss_weight_v0": round(total_weight, 4),
        "by_task_bucket": {key: dict(value) for key, value in by_task.items()},
        "by_pair_type_bucket": {key: dict(value) for key, value in by_pair_type.items()},
        "by_label_need_bucket": {key: dict(value) for key, value in by_need.items()},
    }


def markdown_counter(counter: dict[str, int], key_name: str, count_name: str = "count") -> list[str]:
    lines = [
        f"| {key_name} | {count_name} |",
        "| --- | ---: |",
    ]
    for key, value in sorted(counter.items()):
        lines.append(f"| {key} | {value} |")
    return lines


def write_report(path: Path, summary: dict[str, Any], input_path: Path, stereo_conflict_margin: float) -> None:
    lines = [
        "# Stage 2 fusion labels v0",
        "",
        f"- input: `{input_path}`",
        f"- rows: {summary['num_rows']}",
        f"- stereo conflict margin threshold: {stereo_conflict_margin}",
        f"- preference-loss rows: {summary['use_for_preference_loss_rows']}",
        f"- order-loss rows: {summary['use_for_order_loss_rows']}",
        f"- manual-review rows: {summary['manual_review_rows']}",
        f"- sum preference loss weight: {summary['sum_preference_loss_weight_v0']}",
        "",
        "## Final Label Distribution",
        "",
        *markdown_counter(summary["final_preference_label_v0"], "final_preference_label_v0"),
        "",
        "## Fusion Buckets",
        "",
        *markdown_counter(summary["fusion_bucket_v0"], "fusion_bucket_v0"),
        "",
        "## Method Notes",
        "",
        "- v0 treats MiMo as an offline semantic preference signal, not as a ground-truth reward.",
        "- contact/stage is used as a hard stage gate: clear contact-stage conflict sends the pair to review.",
        "- stereo/depth is used as a softer stage-internal progress signal: strong conflict without contact support sends the pair to review; weak conflict only lowers weight.",
        "- optional temporal pairs are kept for order/progress supervision with lower weights than core true pairs.",
        "- unsure rows are kept in the manifest but receive zero preference-loss weight.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative stage-2 fusion labels from MiMo, stereo, and contact/stage proxies.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--old-root", default=r"E:\reward_model_dataset")
    parser.add_argument("--new-root", default=r"D:\Users\User\Desktop\reward_model_dataset")
    parser.add_argument("--stereo-conflict-margin", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_csv(args.input)
    fused = build_rows(rows, args.old_root, args.new_root, args.stereo_conflict_margin)
    summary = summarize(fused)

    trainable = [row for row in fused if float(row["preference_loss_weight_v0"]) > 0]
    preference_trainable = [row for row in fused if row["use_for_preference_loss_v0"] == "true"]
    order_trainable = [row for row in fused if row["use_for_order_loss_v0"] == "true"]
    review = [row for row in fused if row["needs_manual_review_v0"] == "true"]
    unsure = [row for row in fused if not row["final_preference_label_v0"] and row["needs_manual_review_v0"] != "true"]

    write_csv(args.out / "final_pair_labels_v0.csv", fused)
    write_json(args.out / "final_pair_labels_v0.json", fused)
    write_csv(args.out / "trainable_pairs_v0.csv", trainable)
    write_csv(args.out / "preference_loss_pairs_v0.csv", preference_trainable)
    write_csv(args.out / "order_loss_pairs_v0.csv", order_trainable)
    write_csv(args.out / "manual_review_pairs_v0.csv", review)
    write_csv(args.out / "unsure_zero_weight_pairs_v0.csv", unsure)
    write_json(args.out / "fusion_summary_v0.json", summary)
    write_report(args.out / "fusion_report_v0.md", summary, args.input, args.stereo_conflict_margin)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
