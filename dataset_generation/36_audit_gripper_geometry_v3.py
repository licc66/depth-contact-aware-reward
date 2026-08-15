"""Audit image-derived gripper geometry against offline simulator truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


COMPARISONS = {
    "gripper_object_3d_dist_m": "gt_tcp_object_dist_m",
    "object_goal_3d_dist_m": "gt_object_goal_3d_dist_m",
    "object_goal_xy_error_m": "gt_object_goal_xy_error_m",
    "object_goal_height_error_m": "gt_object_goal_height_error_m",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def number(value: Any) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def integer(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float:
    if len(left) < 2:
        return float("nan")
    if rank:
        left, right = average_ranks(left), average_ranks(right)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def aligned_tables(
    sensor_path: Path, supervision_path: Path
) -> list[tuple[dict[str, str], dict[str, str]]]:
    sensors = read_csv(sensor_path)
    supervision = read_csv(supervision_path)
    key_names = ("meta_sample_id", "meta_saved_frame_index")
    sensor_map = {tuple(row[key] for key in key_names): row for row in sensors}
    supervision_map = {
        tuple(row[key] for key in key_names): row for row in supervision
    }
    if len(sensor_map) != len(sensors) or len(supervision_map) != len(supervision):
        raise RuntimeError("duplicate frame keys")
    if set(sensor_map) != set(supervision_map):
        raise RuntimeError("sensor/supervision frame keys differ")
    keys = sorted(sensor_map, key=lambda key: (key[0], integer(key[1])))
    return [(sensor_map[key], supervision_map[key]) for key in keys]


def metrics(
    aligned: list[tuple[dict[str, str], dict[str, str]]],
    indices: Iterable[int],
    sensor_name: str,
    target_name: str,
) -> dict[str, Any]:
    indices = list(indices)
    pairs = [
        (number(aligned[index][0].get(sensor_name)), number(aligned[index][1].get(target_name)))
        for index in indices
    ]
    pairs = [pair for pair in pairs if all(math.isfinite(value) for value in pair)]
    if not pairs:
        return {
            "rows": len(indices),
            "valid_count": 0,
            "valid_ratio": 0.0,
            "mae_m": float("nan"),
            "p90_m": float("nan"),
            "bias_m": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }
    predicted = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    target = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    error = predicted - target
    return {
        "rows": len(indices),
        "valid_count": len(pairs),
        "valid_ratio": len(pairs) / max(1, len(indices)),
        "mae_m": float(np.mean(np.abs(error))),
        "p90_m": float(np.quantile(np.abs(error), 0.90)),
        "bias_m": float(np.mean(error)),
        "pearson": correlation(predicted, target),
        "spearman": correlation(predicted, target, rank=True),
    }


def group_indices(
    aligned: list[tuple[dict[str, str], dict[str, str]]]
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"all": list(range(len(aligned)))}
    definitions = (
        ("split", "meta_split", False),
        ("source", "meta_source_type", False),
        ("source_group", "meta_source_group_id", False),
        ("stage", "gt_stage_candidate", True),
    )
    for prefix, name, from_supervision in definitions:
        values = sorted(
            {
                pair[1 if from_supervision else 0].get(name, "")
                for pair in aligned
            }
        )
        for value in values:
            groups[f"{prefix}:{value}"] = [
                index
                for index, pair in enumerate(aligned)
                if pair[1 if from_supervision else 0].get(name, "") == value
            ]
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aligned = aligned_tables(
        args.dataset_dir / "sensor_features.csv",
        args.dataset_dir / "offline_supervision.csv",
    )
    rows = []
    for group, indices in group_indices(aligned).items():
        for sensor_name, target_name in COMPARISONS.items():
            rows.append(
                {
                    "group": group,
                    "sensor_feature": sensor_name,
                    "target_feature": target_name,
                    **metrics(aligned, indices, sensor_name, target_name),
                }
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "geometry_errors.csv", rows)
    all_rows = {row["sensor_feature"]: row for row in rows if row["group"] == "all"}
    sensors = [pair[0] for pair in aligned]
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "frames": len(aligned),
        "trajectories": len({row["meta_sample_id"] for row in sensors}),
        "gripper_status_counts": dict(
            Counter(row.get("sensor_gripper_status", "") for row in sensors)
        ),
        "gripper_object": all_rows["gripper_object_3d_dist_m"],
        "object_goal": all_rows["object_goal_3d_dist_m"],
    }
    write_json(args.out_dir / "audit_summary.json", summary)
    gripper = summary["gripper_object"]
    object_goal = summary["object_goal"]
    lines = [
        "# Gripper-Augmented Stereo Audit v3",
        "",
        f"- frames: {summary['frames']}",
        f"- trajectories: {summary['trajectories']}",
        f"- gripper-object valid ratio: {gripper['valid_ratio']:.3f}",
        f"- gripper-object MAE: {gripper['mae_m'] * 1000:.2f} mm",
        f"- gripper-object P90: {gripper['p90_m'] * 1000:.2f} mm",
        f"- gripper-object Spearman: {gripper['spearman']:.3f}",
        f"- object-goal valid ratio: {object_goal['valid_ratio']:.3f}",
        f"- object-goal MAE: {object_goal['mae_m'] * 1000:.2f} mm",
        "",
        "The TCP pose is used only as offline audit truth. The online feature is",
        "triangulated from stereo RGB, disparity, and visible robot-link masks.",
        "The masks currently come from ManiSkill renderer segmentation and must be",
        "replaced by a detector/segmenter before making a real-world claim.",
    ]
    (args.out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.out_dir / "RESULTS.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
