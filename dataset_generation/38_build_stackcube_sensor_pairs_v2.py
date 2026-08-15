"""Rebuild StackCube pair windows on the sensor-consistent v2 collection.

The legacy table is used only to preserve which temporal windows are compared.
Video paths, pair/clip ids, simulator reference labels, and semantic-label
provenance are rebuilt so old frozen-snapshot labels cannot silently supervise
the new settled trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_PREFERENCES = {"A>B", "B>A"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def sample_id_from_video(path: str) -> str:
    return Path(path).stem


def physical_target(supervision: dict[str, str]) -> tuple[int, float, float]:
    """Match the offline target construction used by trainer 35."""

    stage = max(1, min(4, integer(supervision.get("gt_stage_candidate"), 1))) - 1
    if stage == 0:
        distance = number(supervision.get("gt_tcp_object_dist_m"), 0.35)
        local = max(0.0, min(1.0, (0.35 - distance) / 0.325))
    elif stage == 1:
        distance = number(supervision.get("gt_object_goal_3d_dist_m"), 0.30)
        local = max(0.0, min(1.0, 1.0 - distance / 0.30))
    elif stage == 2:
        distance = number(supervision.get("gt_object_goal_3d_dist_m"), 0.08)
        spatial = max(0.0, min(1.0, 1.0 - distance / 0.08))
        on_support = number(supervision.get("gt_is_cubeA_on_cubeB"), 0.0)
        static = number(supervision.get("gt_is_cubeA_static"), 0.0)
        released = 1.0 - number(supervision.get("gt_is_cubeA_grasped"), 0.0)
        local = max(
            0.0,
            min(1.0, 0.45 * spatial + 0.30 * on_support + 0.15 * static + 0.10 * released),
        )
    else:
        local = 1.0
    return stage + 1, local, (stage + local) / 4.0


def preference(
    target_a: tuple[int, float, float],
    target_b: tuple[int, float, float],
    potential_epsilon: float,
) -> tuple[str, str]:
    stage_delta = target_a[0] - target_b[0]
    if stage_delta:
        return ("A>B" if stage_delta > 0 else "B>A"), "stage"
    delta = target_a[2] - target_b[2]
    if abs(delta) >= potential_epsilon:
        return ("A>B" if delta > 0 else "B>A"), "progress"
    return "unsure", "none"


def grouped_rows(
    rows: list[dict[str, str]], key: str
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    for values in grouped.values():
        values.sort(key=lambda item: integer(item["meta_saved_frame_index"]))
    return dict(grouped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--legacy-pairs", type=Path, required=True)
    parser.add_argument("--sensor-dataset", type=Path, required=True)
    parser.add_argument("--conflicts-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--potential-epsilon", type=float, default=0.02)
    parser.add_argument(
        "--require-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    legacy_rows = [
        row for row in read_csv(args.legacy_pairs) if row.get("task_id") == "stackcube"
    ]
    sensor_by_sample = grouped_rows(
        read_csv(args.sensor_dataset / "sensor_features.csv"), "meta_sample_id"
    )
    supervision_by_sample = grouped_rows(
        read_csv(args.sensor_dataset / "offline_supervision.csv"), "meta_sample_id"
    )
    conflict_ids = {
        row["sample_id"] for row in read_csv(args.conflicts_csv) if row.get("sample_id")
    }

    output: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for legacy in legacy_rows:
        sample_a = sample_id_from_video(legacy["clip_a_video_path_windows"])
        sample_b = sample_id_from_video(legacy["clip_b_video_path_windows"])
        conflict_sides = [sample for sample in (sample_a, sample_b) if sample in conflict_ids]
        if conflict_sides:
            excluded.append(
                {
                    "legacy_pair_id": legacy["pair_id"],
                    "reason": "contains_label_conflict_trajectory",
                    "conflict_sample_ids": ";".join(sorted(set(conflict_sides))),
                }
            )
            continue
        if sample_a not in sensor_by_sample or sample_b not in sensor_by_sample:
            raise KeyError(f"missing sensor trajectory for {sample_a!r} or {sample_b!r}")
        split_a = sensor_by_sample[sample_a][0]["meta_split"]
        split_b = sensor_by_sample[sample_b][0]["meta_split"]
        group_a = sensor_by_sample[sample_a][0]["meta_source_group_id"]
        group_b = sensor_by_sample[sample_b][0]["meta_source_group_id"]
        if split_a != split_b or group_a != group_b:
            raise RuntimeError(
                f"pair {legacy['pair_id']} crosses split/group: "
                f"{split_a}/{group_a} vs {split_b}/{group_b}"
            )

        end_a = integer(legacy["clip_a_end_frame_exclusive"])
        end_b = integer(legacy["clip_b_end_frame_exclusive"])
        if not (0 < end_a <= len(supervision_by_sample[sample_a])):
            raise IndexError(f"invalid A end frame for {legacy['pair_id']}: {end_a}")
        if not (0 < end_b <= len(supervision_by_sample[sample_b])):
            raise IndexError(f"invalid B end frame for {legacy['pair_id']}: {end_b}")
        target_a = physical_target(supervision_by_sample[sample_a][end_a - 1])
        target_b = physical_target(supervision_by_sample[sample_b][end_b - 1])
        reference, basis = preference(target_a, target_b, args.potential_epsilon)

        video_a = (
            args.sensor_dataset
            / "samples"
            / split_a
            / sample_a
            / "left_stereo.mp4"
        )
        video_b = (
            args.sensor_dataset
            / "samples"
            / split_b
            / sample_b
            / "left_stereo.mp4"
        )
        if args.require_videos and (not video_a.exists() or not video_b.exists()):
            raise FileNotFoundError(f"missing sensor-aligned video for {legacy['pair_id']}")

        pair_index = len(output)
        row: dict[str, Any] = dict(legacy)
        row.update(
            {
                "legacy_pair_id": legacy["pair_id"],
                "pair_id": f"SCB2-P{pair_index:06d}",
                "split": split_a,
                "split_v2": split_a,
                "source_group_id": group_a,
                "legacy_clip_a_id": legacy["clip_a_id"],
                "legacy_clip_b_id": legacy["clip_b_id"],
                "clip_a_id": f"{legacy['clip_a_id']}-SENSORV2",
                "clip_b_id": f"{legacy['clip_b_id']}-SENSORV2",
                "clip_a_sample_id": sample_a,
                "clip_b_sample_id": sample_b,
                "legacy_clip_a_video_path_windows": legacy["clip_a_video_path_windows"],
                "legacy_clip_b_video_path_windows": legacy["clip_b_video_path_windows"],
                "clip_a_video_path_windows": str(video_a),
                "clip_b_video_path_windows": str(video_b),
                "legacy_mimo_preference": legacy.get("mimo_preference", ""),
                "legacy_mimo_confidence": legacy.get("mimo_confidence", ""),
                "legacy_mimo_model": legacy.get("mimo_model", ""),
                "mimo_preference": "",
                "mimo_confidence": "",
                "mimo_model": "",
                "mimo_raw_response_path": "",
                "needs_vlm_preference_label": "true",
                "reference_label_v2": reference,
                "reference_basis_v2": basis,
                "reference_a_stage_v2": target_a[0],
                "reference_b_stage_v2": target_b[0],
                "reference_a_local_progress_v2": round(target_a[1], 6),
                "reference_b_local_progress_v2": round(target_b[1], 6),
                "reference_a_potential_v2": round(target_a[2], 6),
                "reference_b_potential_v2": round(target_b[2], 6),
                "semantic_label_provenance_v2": "unlabeled_sensor_aligned_video",
            }
        )
        output.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "stackcube_pairs_v2.csv", output)
    write_csv(args.out_dir / "excluded_pairs_v2.csv", excluded)

    reference_counts = Counter(row["reference_label_v2"] for row in output)
    pair_type_counts = Counter(row["pair_type"] for row in output)
    split_counts = Counter(row["split_v2"] for row in output)
    stated = [row for row in output if row["reference_label_v2"] in VALID_PREFERENCES]
    candidate_matches = sum(
        row.get("candidate_label") == row["reference_label_v2"] for row in stated
    )
    legacy_semantic_stated = [
        row for row in output if row["legacy_mimo_preference"] in VALID_PREFERENCES
    ]
    legacy_semantic_matches = sum(
        row["legacy_mimo_preference"] == row["reference_label_v2"]
        for row in legacy_semantic_stated
    )
    summary = {
        "schema_version": "stackcube_sensor_pairs_v2.0",
        "legacy_pair_table": str(args.legacy_pairs),
        "sensor_dataset": str(args.sensor_dataset),
        "output_pairs": len(output),
        "excluded_pairs": len(excluded),
        "excluded_conflict_trajectories": len(conflict_ids),
        "split_counts": dict(split_counts),
        "pair_type_counts": dict(pair_type_counts),
        "reference_label_counts": dict(reference_counts),
        "candidate_vs_v2_reference_accuracy": (
            candidate_matches / len(stated) if stated else 0.0
        ),
        "legacy_semantic_vs_v2_reference_accuracy": (
            legacy_semantic_matches / len(legacy_semantic_stated)
            if legacy_semantic_stated
            else 0.0
        ),
        "legacy_semantic_stated_rows": len(legacy_semantic_stated),
        "semantic_labels_ready_for_primary_fusion": False,
        "notes": [
            "Legacy pairs provide temporal windows only.",
            "Legacy MiMo labels are retained for audit and are not primary v2 supervision.",
            "reference_label_v2 is offline simulator supervision and is never a model input.",
        ],
    }
    write_json(args.out_dir / "pair_build_summary_v2.json", summary)
    report = [
        "# StackCube Sensor Pair Table v2",
        "",
        f"- retained pairs: {len(output)}",
        f"- excluded conflict pairs: {len(excluded)}",
        f"- split counts: {dict(split_counts)}",
        f"- reference labels: {dict(reference_counts)}",
        f"- candidate/reference accuracy: {summary['candidate_vs_v2_reference_accuracy']:.3f}",
        f"- legacy semantic/reference accuracy: {summary['legacy_semantic_vs_v2_reference_accuracy']:.3f}",
        "",
        "Primary v2 fusion remains blocked until the sensor-aligned videos are relabeled.",
        "The retained legacy semantic labels are audit-only because they came from different videos.",
    ]
    (args.out_dir / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
