from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from physical_progress_branch_v1 import PhysicalProgressRuntime
from reward_common_v1 import trajectory_split_map


TASK_IDS = ("stackcube", "stackpyramid", "peginsertion")

CONTACT_STRESS_FEATURES = (
    "is_grasping_object",
    "finger_object_contact_force_n",
    "finger_object_contact",
    "object_support_contact_force_n",
    "object_support_contacts",
    "released_object",
    "top_cube_cubeA_contact_force_n",
    "top_cube_cubeB_contact_force_n",
    "top_cube_contacts_both_base_cubes",
    "peg_box_contact_force_n",
    "left_finger_touch_object",
    "right_finger_touch_object",
    "both_fingers_touch_object",
    "left_finger_object_contact_force_n",
    "right_finger_object_contact_force_n",
    "gripper_width",
    "grasp_confirmed",
)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor
        while end + 1 < len(order) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end) / 2.0
        for index in range(cursor, end + 1):
            output[order[index]] = rank
        cursor = end + 1
    return output


def spearman(values_a: list[float], values_b: list[float]) -> float:
    rank_a, rank_b = ranks(values_a), ranks(values_b)
    mean_a, mean_b = statistics.mean(rank_a), statistics.mean(rank_b)
    numerator = sum(
        (value_a - mean_a) * (value_b - mean_b)
        for value_a, value_b in zip(rank_a, rank_b)
    )
    denominator = math.sqrt(
        sum((value - mean_a) ** 2 for value in rank_a)
        * sum((value - mean_b) ** 2 for value in rank_b)
    )
    return numerator / denominator if denominator else 0.0


def is_clean_success(row: dict[str, str]) -> bool:
    sample_id = row["sample_id"].upper()
    return (
        "success" in row.get("source_type", "").lower()
        and not row.get("near_miss_type", "").strip()
        and "OFFSET" not in sample_id
        and "TRUNC" not in sample_id
    )


def source_success_id(sample_id: str) -> str:
    for separator in ("-OFFSET-", "-TRUNC-"):
        if separator in sample_id:
            return sample_id.split(separator, 1)[0]
    return sample_id


def terminal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trajectories": 0,
            "stage4_rate": 0.0,
            "potential_ge_075_rate": 0.0,
            "mean_potential": 0.0,
            "max_potential": 0.0,
        }
    potentials = [float(row["terminal_potential"]) for row in rows]
    return {
        "trajectories": len(rows),
        "stage4_rate": statistics.mean(row["terminal_stage"] == 4 for row in rows),
        "potential_ge_075_rate": statistics.mean(value >= 0.75 for value in potentials),
        "mean_potential": statistics.mean(potentials),
        "max_potential": max(potentials),
    }


def terminal_summary_by_split(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        split: terminal_summary(
            [row for row in rows if row.get("split") == split]
        )
        for split in ("train", "val", "test")
    }


def score_terminal(
    runtime: PhysicalProgressRuntime,
    task_id: str,
    sample_id: str,
    frames: list[dict[str, Any]],
    split: str | None = None,
) -> dict[str, Any]:
    terminal = runtime.score(task_id, frames)
    return {
        "sample_id": sample_id,
        "split": split if split is not None else frames[-1].get("split", ""),
        "source_type": frames[-1].get("source_type", ""),
        "near_miss_type": frames[-1].get("near_miss_type", ""),
        "terminal_stage": terminal["stage"],
        "terminal_potential": terminal["potential"],
        "terminal_confidence": terminal["confidence"],
    }


def evaluate_task(
    runtime: PhysicalProgressRuntime,
    dataset_root: Path,
    task_id: str,
    canonical_splits: dict[tuple[str, str], str] | None = None,
) -> dict[str, Any]:
    folder = f"{task_id}_bootstrap_v1"
    stereo_rows = load_csv(
        dataset_root / "stereo_features" / folder / "frame_stereo_geometry_features.csv"
    )
    contact_rows = load_csv(
        dataset_root / "contact_stage_features" / folder / "frame_contact_stage_features.csv"
    )
    contact_by_key = {
        (row["sample_id"], row["frame_idx"]): row for row in contact_rows
    }
    frames_by_sample: dict[str, list[dict[str, Any]]] = {}
    for stereo_row in stereo_rows:
        key = (stereo_row["sample_id"], stereo_row["frame_idx"])
        frames_by_sample.setdefault(stereo_row["sample_id"], []).append(
            {**stereo_row, **contact_by_key[key]}
        )
    for frames in frames_by_sample.values():
        frames.sort(key=lambda row: int(float(row["frame_idx"])))

    sample_ids = sorted(
        sample_id
        for sample_id, frames in frames_by_sample.items()
        if is_clean_success(frames[-1])
    )
    trajectory_rows: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        frames = frames_by_sample[sample_id]
        end_indices = list(range(0, len(frames), runtime.sequence_length))
        if end_indices[-1] != len(frames) - 1:
            end_indices.append(len(frames) - 1)
        potentials = [
            runtime.score(task_id, frames[: end_index + 1])["potential"]
            for end_index in end_indices
        ]
        terminal = runtime.score(task_id, frames)
        preterminal_end = max(0, len(frames) - runtime.sequence_length - 1)
        preterminal = runtime.score(task_id, frames[: preterminal_end + 1])
        trajectory_rows.append(
            {
                "sample_id": sample_id,
                "split": (
                    canonical_splits.get((task_id, sample_id), "unassigned")
                    if canonical_splits is not None
                    else frames[-1].get("split", "")
                ),
                "frames": len(frames),
                "spearman": spearman(
                    [float(index) for index in end_indices],
                    potentials,
                ),
                "initial_potential": potentials[0],
                "preterminal_potential": preterminal["potential"],
                "terminal_potential": terminal["potential"],
                "terminal_stage": terminal["stage"],
                "terminal_drop": max(
                    0.0,
                    preterminal["potential"] - terminal["potential"],
                ),
            }
        )

    failure_rows: list[dict[str, Any]] = []
    near_miss_rows: list[dict[str, Any]] = []
    truncated_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    clean_terminal_contact = {
        sample_id: frames_by_sample[sample_id][-1] for sample_id in sample_ids
    }
    for sample_id, frames in sorted(frames_by_sample.items()):
        if sample_id in clean_terminal_contact:
            continue
        split = (
            canonical_splits.get((task_id, sample_id), "unassigned")
            if canonical_splits is not None
            else None
        )
        result = score_terminal(runtime, task_id, sample_id, frames, split=split)
        failure_rows.append(result)
        if frames[-1].get("source_type", "") == "perturbed_success_final_state":
            near_miss_rows.append(result)
            reference = clean_terminal_contact.get(source_success_id(sample_id))
            if reference is not None:
                stressed_frames: list[dict[str, Any]] = []
                for frame in frames:
                    stressed = dict(frame)
                    for feature_name in CONTACT_STRESS_FEATURES:
                        if feature_name in reference:
                            stressed[feature_name] = reference[feature_name]
                    stressed_frames.append(stressed)
                stress_rows.append(
                    score_terminal(
                        runtime,
                        task_id,
                        sample_id,
                        stressed_frames,
                        split=split,
                    )
                )
        elif frames[-1].get("source_type", "") == "truncated_success_trajectory":
            truncated_rows.append(result)

    return {
        "trajectories": len(trajectory_rows),
        "mean_spearman": statistics.mean(row["spearman"] for row in trajectory_rows),
        "terminal_stage4_rate": statistics.mean(
            row["terminal_stage"] == 4 for row in trajectory_rows
        ),
        "terminal_saturated_rate": statistics.mean(
            math.isclose(row["terminal_potential"], 1.0, abs_tol=1e-6)
            for row in trajectory_rows
        ),
        "mean_terminal_drop": statistics.mean(
            row["terminal_drop"] for row in trajectory_rows
        ),
        "failure_rejection": terminal_summary(failure_rows),
        "failure_rejection_by_split": terminal_summary_by_split(failure_rows),
        "near_miss_rejection": terminal_summary(near_miss_rows),
        "near_miss_rejection_by_split": terminal_summary_by_split(near_miss_rows),
        "truncated_rejection": terminal_summary(truncated_rows),
        "near_miss_success_contact_stress": terminal_summary(stress_rows),
        "near_miss_success_contact_stress_by_split": terminal_summary_by_split(
            stress_rows
        ),
        "trajectories_detail": trajectory_rows,
        "failure_detail": failure_rows,
        "near_miss_success_contact_stress_detail": stress_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\Users\User\Desktop\reward_model_dataset"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            r"D:\Users\User\Desktop\reward_model_dataset\reward_model_runs"
            r"\physical_progress_branch_v1\best_model.pt"
        ),
    )
    parser.add_argument(
        "--split-map-dir",
        type=Path,
        default=None,
        help="Canonical script-17 split directory. If omitted, by-split "
        "summaries reproduce the contaminated legacy frame split.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    runtime = PhysicalProgressRuntime.from_checkpoint(args.checkpoint, args.device)
    canonical_splits = (
        trajectory_split_map(args.split_map_dir)
        if args.split_map_dir is not None
        else None
    )
    if canonical_splits is None:
        print(
            "WARNING: trajectory by-split summaries use the legacy per-frame split; "
            "pass --split-map-dir for leakage-free summaries.",
            file=sys.stderr,
        )
    results = {
        task_id: evaluate_task(
            runtime,
            args.dataset_root,
            task_id,
            canonical_splits,
        )
        for task_id in TASK_IDS
    }
    results["split_provenance"] = {
        "source": (
            "canonical_source_group_v1" if canonical_splits is not None else "legacy_frame_split"
        ),
        "split_map_dir": str(args.split_map_dir) if args.split_map_dir else None,
    }
    output_path = args.checkpoint.parent / "trajectory_sanity.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
