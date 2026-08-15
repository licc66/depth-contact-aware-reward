"""Audit StackCube sensor-consistent v2 data and the frozen v1 physical model."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_wrapper_module():
    path = SCRIPT_DIR / "30_maniskill_reward_wrapper_v1.py"
    spec = importlib.util.spec_from_file_location("reward_wrapper_for_sensor_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wrapper = load_wrapper_module()


GEOMETRY_COMPARISON_FEATURES = [
    "object_cam_x",
    "object_cam_y",
    "object_cam_depth_m",
    "goal_cam_x",
    "goal_cam_y",
    "goal_cam_depth_m",
    "support_cam_depth_m",
    "object_goal_3d_dist_m",
    "object_goal_xy_error_m",
    "object_goal_height_error_m",
    "delta_object_goal_dist_m",
    "object_moved_from_start_m",
    "object_goal_cam_lateral_error_m",
    "object_goal_cam_depth_error_m",
    "left_object_depth_m",
    "right_object_depth_m",
    "left_goal_depth_m",
    "object_pseudo_disparity_px",
    "goal_pseudo_disparity_px",
    "object_speed_proxy_m_per_step",
    "object_static_proxy",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def number(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def mean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return float(np.mean(array)) if array.size else float("nan")


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def correlation(x: Iterable[float], y: Iterable[float], rank: bool = False) -> float:
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 2:
        return float("nan")
    left = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    right = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if rank:
        left = average_ranks(left)
        right = average_ranks(right)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def error_metrics(predicted: list[float], target: list[float]) -> dict[str, Any]:
    pairs = [
        (a, b)
        for a, b in zip(predicted, target)
        if math.isfinite(a) and math.isfinite(b)
    ]
    if not pairs:
        return {
            "valid_count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "p50_abs_error": float("nan"),
            "p90_abs_error": float("nan"),
            "p95_abs_error": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }
    pred = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    gt = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    error = pred - gt
    absolute = np.abs(error)
    return {
        "valid_count": len(pairs),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "p50_abs_error": float(np.quantile(absolute, 0.50)),
        "p90_abs_error": float(np.quantile(absolute, 0.90)),
        "p95_abs_error": float(np.quantile(absolute, 0.95)),
        "pearson": correlation(pred, gt),
        "spearman": correlation(pred, gt, rank=True),
    }


def aligned_tables(
    sensor_path: Path, supervision_path: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sensors = read_csv(sensor_path)
    supervision = read_csv(supervision_path)
    key_names = ("meta_sample_id", "meta_saved_frame_index")
    sensor_map = {tuple(row[key] for key in key_names): row for row in sensors}
    supervision_map = {
        tuple(row[key] for key in key_names): row for row in supervision
    }
    if len(sensor_map) != len(sensors) or len(supervision_map) != len(supervision):
        raise RuntimeError("duplicate sample/frame keys in v2 tables")
    if set(sensor_map) != set(supervision_map):
        missing_sensor = set(supervision_map) - set(sensor_map)
        missing_supervision = set(sensor_map) - set(supervision_map)
        raise RuntimeError(
            f"sensor/supervision keys differ: missing_sensor={len(missing_sensor)} "
            f"missing_supervision={len(missing_supervision)}"
        )
    keys = sorted(sensor_map, key=lambda key: (key[0], integer(key[1])))
    return [sensor_map[key] for key in keys], [supervision_map[key] for key in keys]


def sensor_frame(row: dict[str, str], feature_names: list[str]) -> dict[str, float]:
    frame = {}
    for name in feature_names:
        value = number(row.get(name))
        if math.isfinite(value):
            frame[name] = value
    return frame


def gt_progress_target(supervision: dict[str, str], sensor: dict[str, str]) -> float:
    stage = integer(supervision.get("gt_stage_candidate"), 1)
    if stage >= 4:
        local = 1.0
    elif stage == 3:
        on_support = number(supervision.get("gt_is_cubeA_on_cubeB"), 0.0)
        is_static = number(supervision.get("gt_is_cubeA_static"), 0.0)
        released = number(sensor.get("released_object"), 0.0)
        local = float(np.clip(0.4 * on_support + 0.3 * released + 0.3 * is_static, 0.0, 1.0))
    elif stage == 2:
        distance = number(supervision.get("gt_object_goal_3d_dist_m"), 0.30)
        local = float(np.clip(1.0 - distance / 0.30, 0.0, 1.0))
    else:
        tcp_distance = number(supervision.get("gt_tcp_object_dist_m"), 0.50)
        local = float(np.clip(1.0 - tcp_distance / 0.50, 0.0, 1.0))
    return float(((stage - 1) + local) / 4.0)


def geometry_error_rows(
    sensors: list[dict[str, str]], supervision: list[dict[str, str]]
) -> list[dict[str, Any]]:
    groups = {
        "all": list(range(len(sensors))),
    }
    for source in sorted({row["meta_source_type"] for row in sensors}):
        groups[f"source:{source}"] = [
            index for index, row in enumerate(sensors) if row["meta_source_type"] == source
        ]
    for split in sorted({row["meta_split"] for row in sensors}):
        groups[f"split:{split}"] = [
            index for index, row in enumerate(sensors) if row["meta_split"] == split
        ]

    rows = []
    for group_name, indices in groups.items():
        for feature in GEOMETRY_COMPARISON_FEATURES:
            predicted = [number(sensors[index].get(feature)) for index in indices]
            target = [number(supervision[index].get(f"gt_{feature}")) for index in indices]
            metrics = error_metrics(predicted, target)
            rows.append(
                {
                    "group": group_name,
                    "feature": feature,
                    "row_count": len(indices),
                    "prediction_valid_ratio": metrics["valid_count"] / max(1, len(indices)),
                    **metrics,
                }
            )
    return rows


def score_v1_model(
    sensors: list[dict[str, str]],
    supervision: list[dict[str, str]],
    checkpoint: Path,
) -> list[dict[str, Any]]:
    scorer = wrapper.FrozenPhysicalScorer(checkpoint, task_id="stackcube", device="cpu")
    feature_names = list(scorer.runtime.feature_names)
    grouped: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for sensor, gt in zip(sensors, supervision):
        grouped[sensor["meta_sample_id"]].append((sensor, gt))
    predictions = []
    for sample_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda pair: integer(pair[0]["meta_saved_frame_index"]))
        history: list[dict[str, float]] = []
        for sensor, gt in rows:
            history.append(sensor_frame(sensor, feature_names))
            result = scorer(history[-scorer.runtime.history_window :])
            predictions.append(
                {
                    "meta_sample_id": sample_id,
                    "meta_saved_frame_index": sensor["meta_saved_frame_index"],
                    "meta_replay_step": sensor["meta_replay_step"],
                    "meta_split": sensor["meta_split"],
                    "meta_source_type": sensor["meta_source_type"],
                    "meta_near_miss_type": sensor["meta_near_miss_type"],
                    "meta_source_group_id": sensor["meta_source_group_id"],
                    "model_potential": result["potential"],
                    "model_stage": result["stage"],
                    "model_local_progress": result["local_progress"],
                    "model_confidence": result["confidence"],
                    "model_depth_validity_ratio": result["depth_validity_ratio"],
                    "model_contact_validity_ratio": result["contact_validity_ratio"],
                    "gt_progress_target_diagnostic": gt_progress_target(gt, sensor),
                    "gt_stage_candidate": gt["gt_stage_candidate"],
                    "gt_success": gt["gt_success"],
                    "gt_object_goal_3d_dist_m": gt["gt_object_goal_3d_dist_m"],
                    "sensor_object_goal_3d_dist_m": sensor.get(
                        "object_goal_3d_dist_m", ""
                    ),
                    "label_expected_success_from_manifest": gt.get(
                        "label_expected_success_from_manifest", ""
                    ),
                    "label_terminal_rank_from_manifest": gt.get(
                        "label_terminal_rank_from_manifest", ""
                    ),
                }
            )
    return predictions


def trajectory_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["meta_sample_id"]].append(row)
    output = []
    for sample_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: integer(row["meta_saved_frame_index"]))
        potentials = [number(row["model_potential"]) for row in rows]
        gt_progress = [number(row["gt_progress_target_diagnostic"]) for row in rows]
        replay_steps = [number(row["meta_replay_step"]) for row in rows]
        sensor_distances = [number(row["sensor_object_goal_3d_dist_m"]) for row in rows]
        gt_distances = [number(row["gt_object_goal_3d_dist_m"]) for row in rows]
        terminal = rows[-1]
        expected_success = truthy(terminal["label_expected_success_from_manifest"])
        replay_complete = not expected_success or truthy(terminal["gt_success"])
        output.append(
            {
                "sample_id": sample_id,
                "split": terminal["meta_split"],
                "source_type": terminal["meta_source_type"],
                "near_miss_type": terminal["meta_near_miss_type"],
                "source_group_id": terminal["meta_source_group_id"],
                "frames": len(rows),
                "expected_success": expected_success,
                "terminal_gt_success": truthy(terminal["gt_success"]),
                "replay_complete_for_terminal_audit": replay_complete,
                "terminal_rank": integer(terminal["label_terminal_rank_from_manifest"], -1),
                "terminal_model_potential": number(terminal["model_potential"]),
                "terminal_model_stage": integer(terminal["model_stage"]),
                "terminal_model_confidence": number(terminal["model_confidence"]),
                "terminal_sensor_distance_m": number(
                    terminal["sensor_object_goal_3d_dist_m"]
                ),
                "terminal_gt_distance_m": number(terminal["gt_object_goal_3d_dist_m"]),
                "potential_vs_gt_progress_spearman": correlation(
                    potentials, gt_progress, rank=True
                ),
                "potential_vs_replay_step_spearman": correlation(
                    potentials, replay_steps, rank=True
                ),
                "sensor_vs_gt_distance_spearman": correlation(
                    sensor_distances, gt_distances, rank=True
                ),
                "sensor_distance_mae_m": error_metrics(
                    sensor_distances, gt_distances
                )["mae"],
            }
        )
    return output


def pair_order_rows(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        if row["replay_complete_for_terminal_audit"] and row["terminal_rank"] >= 0:
            grouped[row["source_group_id"]].append(row)
    pairs = []
    for group_id, rows in sorted(grouped.items()):
        for index_a in range(len(rows)):
            for index_b in range(index_a + 1, len(rows)):
                a, b = rows[index_a], rows[index_b]
                if a["terminal_rank"] == b["terminal_rank"]:
                    continue
                high, low = (a, b) if a["terminal_rank"] > b["terminal_rank"] else (b, a)
                model_margin = high["terminal_model_potential"] - low["terminal_model_potential"]
                sensor_distance_margin = (
                    low["terminal_sensor_distance_m"] - high["terminal_sensor_distance_m"]
                )
                pairs.append(
                    {
                        "source_group_id": group_id,
                        "high_sample_id": high["sample_id"],
                        "low_sample_id": low["sample_id"],
                        "high_rank": high["terminal_rank"],
                        "low_rank": low["terminal_rank"],
                        "model_margin": model_margin,
                        "model_order_correct": int(model_margin > 0.0),
                        "sensor_distance_margin": sensor_distance_margin,
                        "sensor_distance_order_correct": int(sensor_distance_margin > 0.0),
                    }
                )
    return pairs


def summarize(
    sensors: list[dict[str, str]],
    supervision: list[dict[str, str]],
    errors: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    all_errors = {
        row["feature"]: row for row in errors if row["group"] == "all"
    }
    complete_terminal = [
        row for row in trajectories if row["replay_complete_for_terminal_audit"]
    ]
    success_terminal = [row for row in complete_terminal if row["terminal_gt_success"]]
    failure_terminal = [row for row in complete_terminal if not row["terminal_gt_success"]]
    offset_terminal = [
        row
        for row in failure_terminal
        if row["source_type"] == "perturbed_success_final_state"
    ]
    stage4_threshold = [row for row in complete_terminal if row["terminal_model_stage"] == 4]
    completion_predicted = [
        row for row in complete_terminal if row["terminal_model_potential"] >= 0.8
    ]
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for row in sensors:
        split_by_group[row["meta_source_group_id"]].add(row["meta_split"])
    leaking = [group for group, splits in split_by_group.items() if len(splits) > 1]

    return {
        "dataset": {
            "frames": len(sensors),
            "trajectories": len(trajectories),
            "source_groups": len(split_by_group),
            "source_group_leakage_count": len(leaking),
            "source_counts": dict(Counter(row["meta_source_type"] for row in sensors)),
            "stereo_status_counts": dict(
                Counter(row.get("sensor_stereo_status", "") for row in sensors)
            ),
            "mean_sensor_adapter_time_ms": mean_or_nan(
                number(row.get("sensor_adapter_time_ms")) for row in sensors
            ),
        },
        "sensor_error": {
            "object_depth_mae_m": all_errors["object_cam_depth_m"]["mae"],
            "goal_depth_mae_m": all_errors["goal_cam_depth_m"]["mae"],
            "object_goal_distance_mae_m": all_errors["object_goal_3d_dist_m"]["mae"],
            "object_goal_distance_p90_m": all_errors["object_goal_3d_dist_m"]["p90_abs_error"],
            "object_goal_xy_mae_m": all_errors["object_goal_xy_error_m"]["mae"],
            "object_goal_height_mae_m": all_errors["object_goal_height_error_m"]["mae"],
        },
        "v1_model_on_sensor_domain": {
            "frame_potential_vs_gt_progress_spearman": correlation(
                [number(row["model_potential"]) for row in predictions],
                [number(row["gt_progress_target_diagnostic"]) for row in predictions],
                rank=True,
            ),
            "mean_success_trajectory_progress_spearman": mean_or_nan(
                row["potential_vs_gt_progress_spearman"]
                for row in trajectories
                if row["source_type"] == "official_motionplanning_success"
                and row["replay_complete_for_terminal_audit"]
            ),
            "complete_terminal_trajectories": len(complete_terminal),
            "terminal_success_count": len(success_terminal),
            "terminal_failure_count": len(failure_terminal),
            "success_stage4_rate": (
                sum(row["terminal_model_stage"] == 4 for row in success_terminal)
                / max(1, len(success_terminal))
            ),
            "failure_stage4_false_positive_rate": (
                sum(row["terminal_model_stage"] == 4 for row in failure_terminal)
                / max(1, len(failure_terminal))
            ),
            "offset_stage4_false_positive_rate": (
                sum(row["terminal_model_stage"] == 4 for row in offset_terminal)
                / max(1, len(offset_terminal))
            ),
            "potential_completion_precision_at_0.8": (
                sum(row["terminal_gt_success"] for row in completion_predicted)
                / max(1, len(completion_predicted))
            ),
            "potential_completion_recall_at_0.8": (
                sum(row["terminal_model_potential"] >= 0.8 for row in success_terminal)
                / max(1, len(success_terminal))
            ),
            "stage4_prediction_count": len(stage4_threshold),
        },
        "terminal_pair_order": {
            "pair_count": len(pairs),
            "v1_model_accuracy": mean_or_nan(
                float(row["model_order_correct"]) for row in pairs
            ),
            "raw_sensor_distance_accuracy": mean_or_nan(
                float(row["sensor_distance_order_correct"]) for row in pairs
            ),
        },
        "interpretation_limits": [
            "gt_progress_target_diagnostic is a diagnostic event heuristic, not an online model input.",
            "Terminal audits exclude expected-success trajectories whose replay was intentionally clipped.",
            "The bootstrap trajectories are derived from a small number of source-success groups.",
        ],
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    data = summary["dataset"]
    sensor = summary["sensor_error"]
    model = summary["v1_model_on_sensor_domain"]
    pairs = summary["terminal_pair_order"]
    lines = [
        "# StackCube Sensor-Domain Audit v2",
        "",
        "## Dataset",
        "",
        f"- frames: {data['frames']}",
        f"- trajectories: {data['trajectories']}",
        f"- independent source groups: {data['source_groups']}",
        f"- source-group leakage: {data['source_group_leakage_count']}",
        f"- mean sensor adapter time: {data['mean_sensor_adapter_time_ms']:.2f} ms/frame",
        "",
        "## Stereo Geometry Error",
        "",
        f"- object depth MAE: {sensor['object_depth_mae_m'] * 1000:.2f} mm",
        f"- goal depth MAE: {sensor['goal_depth_mae_m'] * 1000:.2f} mm",
        f"- object-goal 3D distance MAE: {sensor['object_goal_distance_mae_m'] * 1000:.2f} mm",
        f"- object-goal 3D distance P90: {sensor['object_goal_distance_p90_m'] * 1000:.2f} mm",
        f"- XY error MAE: {sensor['object_goal_xy_mae_m'] * 1000:.2f} mm",
        f"- height error MAE: {sensor['object_goal_height_mae_m'] * 1000:.2f} mm",
        "",
        "## Frozen v1 on Sensor Inputs",
        "",
        f"- frame potential vs diagnostic GT progress Spearman: {model['frame_potential_vs_gt_progress_spearman']:.4f}",
        f"- mean success-trajectory Spearman: {model['mean_success_trajectory_progress_spearman']:.4f}",
        f"- success stage-4 rate: {model['success_stage4_rate']:.4f}",
        f"- failure stage-4 false-positive rate: {model['failure_stage4_false_positive_rate']:.4f}",
        f"- offset stage-4 false-positive rate: {model['offset_stage4_false_positive_rate']:.4f}",
        f"- completion precision at potential >= 0.8: {model['potential_completion_precision_at_0.8']:.4f}",
        f"- completion recall at potential >= 0.8: {model['potential_completion_recall_at_0.8']:.4f}",
        "",
        "## Terminal Pair Ordering",
        "",
        f"- pairs: {pairs['pair_count']}",
        f"- frozen v1 accuracy: {pairs['v1_model_accuracy']:.4f}",
        f"- raw SGBM distance accuracy: {pairs['raw_sensor_distance_accuracy']:.4f}",
        "",
        "## Boundary",
        "",
        "These results measure the sensor-domain transition. They do not establish RL performance.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--physical-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sensors, supervision = aligned_tables(
        args.dataset_dir / "sensor_features.csv",
        args.dataset_dir / "offline_supervision.csv",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    errors = geometry_error_rows(sensors, supervision)
    predictions = score_v1_model(
        sensors, supervision, checkpoint=args.physical_checkpoint
    )
    trajectories = trajectory_rows(predictions)
    pairs = pair_order_rows(trajectories)
    summary = summarize(
        sensors, supervision, errors, predictions, trajectories, pairs
    )
    write_csv(args.out_dir / "geometry_feature_errors.csv", errors)
    write_csv(args.out_dir / "frame_v1_predictions.csv", predictions)
    write_csv(args.out_dir / "trajectory_v1_metrics.csv", trajectories)
    write_csv(args.out_dir / "terminal_pair_order.csv", pairs)
    write_json(args.out_dir / "sensor_domain_audit.json", summary)
    write_markdown(args.out_dir / "RESULTS.md", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
