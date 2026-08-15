"""Train sensor-domain StackCube physical progress models and ablations.

The model input is restricted to an explicit observable whitelist from
``sensor_features.csv``.  Simulator truth from ``offline_supervision.csv`` is
used only to construct offline stage/progress targets and losses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from physical_progress_branch_v2 import (
    PhysicalProgressBranchV2,
    pairwise_preference_loss,
    temporal_order_loss,
)


SPLITS = ("train", "val", "test")
NUM_STAGES = 4

DEPTH_FEATURE_WHITELIST = (
    "gripper_object_3d_dist_m",
    "gripper_goal_3d_dist_m",
    "gripper_object_cam_lateral_error_m",
    "gripper_object_cam_depth_error_m",
    "object_goal_3d_dist_m",
    "object_goal_xy_error_m",
    "object_goal_height_error_m",
    "delta_object_goal_dist_m",
    "object_moved_from_start_m",
    "object_speed_proxy_m_per_step",
    "object_static_proxy",
)

CONTACT_FEATURE_WHITELIST = (
    "is_grasping_object",
    "finger_object_contact_force_n",
    "finger_object_contact",
    "object_support_contact_force_n",
    "object_support_contacts",
    "released_object",
    "left_finger_touch_object",
    "right_finger_touch_object",
    "both_fingers_touch_object",
    "left_finger_object_contact_force_n",
    "right_finger_object_contact_force_n",
    "gripper_width",
    "grasp_confirmed",
)

FORBIDDEN_FEATURE_TOKENS = (
    "meta_",
    "gt_",
    "label_",
    "success",
    "stage",
    "rank",
    "source_type",
    "near_miss",
    "replay_step",
    "saved_frame",
    "pose",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def number(value: Any, default: float = float("nan")) -> float:
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1.0
    if text in {"false", "no"}:
        return 0.0
    try:
        parsed = float(text)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 2:
        return float("nan")
    a = average_ranks(np.asarray([pair[0] for pair in pairs]))
    b = average_ranks(np.asarray([pair[1] for pair in pairs]))
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def macro_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for stage in range(NUM_STAGES):
        tp = int(np.sum((target == stage) & (prediction == stage)))
        fp = int(np.sum((target != stage) & (prediction == stage)))
        fn = int(np.sum((target == stage) & (prediction != stage)))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def validate_feature_names(names: Iterable[str]) -> None:
    forbidden = [
        name
        for name in names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"forbidden online model features: {forbidden}")


def aligned_tables(
    sensor_path: Path, supervision_path: Path
) -> list[tuple[dict[str, str], dict[str, str]]]:
    sensors = read_csv(sensor_path)
    supervision = read_csv(supervision_path)
    keys = ("meta_sample_id", "meta_saved_frame_index")
    sensor_map = {tuple(row[key] for key in keys): row for row in sensors}
    supervision_map = {tuple(row[key] for key in keys): row for row in supervision}
    if len(sensor_map) != len(sensors) or len(supervision_map) != len(supervision):
        raise RuntimeError("duplicate sample/frame key")
    if set(sensor_map) != set(supervision_map):
        raise RuntimeError("sensor and supervision tables do not align")
    ordered = sorted(sensor_map, key=lambda key: (key[0], integer(key[1])))
    return [(sensor_map[key], supervision_map[key]) for key in ordered]


def select_features(
    rows: list[dict[str, str]], candidates: Iterable[str]
) -> list[str]:
    header = set(rows[0]) if rows else set()
    selected = [name for name in candidates if name in header]
    validate_feature_names(selected)
    return selected


def raw_matrix(
    rows: list[dict[str, str]], names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(rows), len(names)), dtype=np.float32)
    valid = np.zeros_like(values, dtype=bool)
    for row_index, row in enumerate(rows):
        for feature_index, name in enumerate(names):
            parsed = number(row.get(name))
            if math.isfinite(parsed):
                values[row_index, feature_index] = parsed
                valid[row_index, feature_index] = True
    return values, valid


def robust_stats(
    matrices: Iterable[tuple[np.ndarray, np.ndarray]], feature_count: int
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.zeros(feature_count, dtype=np.float32)
    scales = np.ones(feature_count, dtype=np.float32)
    matrix_list = list(matrices)
    for index in range(feature_count):
        values = np.concatenate(
            [raw[:, index][valid[:, index]] for raw, valid in matrix_list]
        )
        if values.size == 0:
            continue
        centers[index] = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
        if scale < 1e-5:
            scale = float(np.std(values))
        scales[index] = max(scale, 1e-3)
    return centers, scales


def prepare_numpy(
    raw: np.ndarray,
    valid: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    clip_value: float,
) -> np.ndarray:
    safe = np.where(valid, raw, center.reshape(1, -1))
    normalized = np.clip(
        (safe - center.reshape(1, -1)) / scale.reshape(1, -1),
        -clip_value,
        clip_value,
    )
    return np.concatenate([normalized, valid.astype(np.float32)], axis=-1).astype(
        np.float32
    )


def physical_targets(
    sensor: dict[str, str], supervision: dict[str, str]
) -> tuple[int, float, float]:
    stage = int(np.clip(integer(supervision.get("gt_stage_candidate"), 1), 1, 4)) - 1
    if stage == 0:
        distance = number(supervision.get("gt_tcp_object_dist_m"), 0.35)
        local = float(np.clip((0.35 - distance) / 0.325, 0.0, 1.0))
    elif stage == 1:
        distance = number(supervision.get("gt_object_goal_3d_dist_m"), 0.30)
        local = float(np.clip(1.0 - distance / 0.30, 0.0, 1.0))
    elif stage == 2:
        distance = number(supervision.get("gt_object_goal_3d_dist_m"), 0.08)
        spatial = float(np.clip(1.0 - distance / 0.08, 0.0, 1.0))
        on_support = number(supervision.get("gt_is_cubeA_on_cubeB"), 0.0)
        static = number(supervision.get("gt_is_cubeA_static"), 0.0)
        released = 1.0 - number(supervision.get("gt_is_cubeA_grasped"), 0.0)
        local = float(
            np.clip(0.45 * spatial + 0.30 * on_support + 0.15 * static + 0.10 * released, 0.0, 1.0)
        )
    else:
        local = 1.0
    potential = float((stage + local) / NUM_STAGES)
    return stage, local, potential


@dataclass
class Trajectory:
    sample_id: str
    split: str
    source_type: str
    source_group_id: str
    terminal_rank: int
    sensor_rows: list[dict[str, str]]
    supervision_rows: list[dict[str, str]]
    depth_raw: np.ndarray
    depth_valid: np.ndarray
    contact_raw: np.ndarray
    contact_valid: np.ndarray
    stage_targets: np.ndarray
    local_targets: np.ndarray
    potential_targets: np.ndarray
    depth_features: np.ndarray | None = None
    contact_features: np.ndarray | None = None


def build_trajectories(
    aligned: list[tuple[dict[str, str], dict[str, str]]],
    depth_names: list[str],
    contact_names: list[str],
) -> list[Trajectory]:
    grouped: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for sensor, supervision in aligned:
        grouped[sensor["meta_sample_id"]].append((sensor, supervision))
    trajectories = []
    for sample_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda pair: integer(pair[0]["meta_saved_frame_index"]))
        sensors = [pair[0] for pair in rows]
        supervision = [pair[1] for pair in rows]
        depth_raw, depth_valid = raw_matrix(sensors, depth_names)
        contact_raw, contact_valid = raw_matrix(sensors, contact_names)
        targets = [physical_targets(sensor, gt) for sensor, gt in rows]
        trajectories.append(
            Trajectory(
                sample_id=sample_id,
                split=sensors[-1]["meta_split"],
                source_type=sensors[-1]["meta_source_type"],
                source_group_id=sensors[-1]["meta_source_group_id"],
                terminal_rank=integer(
                    supervision[-1].get("label_terminal_rank_from_manifest"), -1
                ),
                sensor_rows=sensors,
                supervision_rows=supervision,
                depth_raw=depth_raw,
                depth_valid=depth_valid,
                contact_raw=contact_raw,
                contact_valid=contact_valid,
                stage_targets=np.asarray([target[0] for target in targets], dtype=np.int64),
                local_targets=np.asarray([target[1] for target in targets], dtype=np.float32),
                potential_targets=np.asarray([target[2] for target in targets], dtype=np.float32),
            )
        )
    return trajectories


def label_conflict_reason(trajectory: Trajectory) -> str | None:
    expected_success = truthy(
        trajectory.supervision_rows[-1].get("label_expected_success_from_manifest")
    )
    success_frames = sum(
        truthy(row.get("gt_success")) for row in trajectory.supervision_rows
    )
    terminal_success = truthy(trajectory.supervision_rows[-1].get("gt_success"))
    if not expected_success and success_frames > 0:
        return "manifest_failure_contains_gt_success"
    if expected_success and not terminal_success:
        return "manifest_success_terminal_not_success"
    return None


@dataclass
class Window:
    sample_id: str
    end_index: int
    split: str
    source_type: str
    source_group_id: str
    depth: np.ndarray
    contact: np.ndarray
    frame_mask: np.ndarray
    stages: np.ndarray
    local: np.ndarray
    potential: np.ndarray


def sample_history_indices(
    end_index: int, history_window: int, sequence_length: int
) -> tuple[list[int], np.ndarray]:
    start = max(0, end_index - history_window + 1)
    indices = list(range(start, end_index + 1))
    if len(indices) > sequence_length:
        positions = np.linspace(0, len(indices) - 1, sequence_length).round().astype(int)
        indices = [indices[position] for position in positions]
    mask = np.ones(len(indices), dtype=bool)
    if len(indices) < sequence_length:
        padding = sequence_length - len(indices)
        indices += [indices[-1]] * padding
        mask = np.concatenate([mask, np.zeros(padding, dtype=bool)])
    return indices, mask


def make_window(
    trajectory: Trajectory,
    end_index: int,
    history_window: int,
    sequence_length: int,
) -> Window:
    if trajectory.depth_features is None or trajectory.contact_features is None:
        raise RuntimeError("trajectory features have not been normalized")
    indices, mask = sample_history_indices(end_index, history_window, sequence_length)
    return Window(
        sample_id=trajectory.sample_id,
        end_index=end_index,
        split=trajectory.split,
        source_type=trajectory.source_type,
        source_group_id=trajectory.source_group_id,
        depth=trajectory.depth_features[indices],
        contact=trajectory.contact_features[indices],
        frame_mask=mask,
        stages=trajectory.stage_targets[indices],
        local=trajectory.local_targets[indices],
        potential=trajectory.potential_targets[indices],
    )


class WindowDataset(Dataset):
    def __init__(self, windows: list[Window]) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        return {
            "depth": torch.from_numpy(window.depth),
            "contact": torch.from_numpy(window.contact),
            "frame_mask": torch.from_numpy(window.frame_mask),
            "stages": torch.from_numpy(window.stages),
            "local": torch.from_numpy(window.local),
            "potential": torch.from_numpy(window.potential),
            "task_id": torch.tensor(0, dtype=torch.long),
        }


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[Window, Window]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _window(prefix: str, window: Window, output: dict[str, torch.Tensor]) -> None:
        output[f"{prefix}_depth"] = torch.from_numpy(window.depth)
        output[f"{prefix}_contact"] = torch.from_numpy(window.contact)
        output[f"{prefix}_mask"] = torch.from_numpy(window.frame_mask)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        high, low = self.pairs[index]
        output: dict[str, torch.Tensor] = {
            "task_id": torch.tensor(0, dtype=torch.long)
        }
        self._window("high", high, output)
        self._window("low", low, output)
        return output


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def terminal_pairs(
    trajectories: list[Trajectory], history_window: int, sequence_length: int
) -> list[tuple[Window, Window]]:
    grouped: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        if trajectory.terminal_rank >= 0:
            grouped[trajectory.source_group_id].append(trajectory)
    output = []
    for group in grouped.values():
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                left, right = group[left_index], group[right_index]
                if left.terminal_rank == right.terminal_rank:
                    continue
                high, low = (
                    (left, right)
                    if left.terminal_rank > right.terminal_rank
                    else (right, left)
                )
                output.append(
                    (
                        make_window(
                            high, len(high.sensor_rows) - 1, history_window, sequence_length
                        ),
                        make_window(
                            low, len(low.sensor_rows) - 1, history_window, sequence_length
                        ),
                    )
                )
    return output


def compute_class_weights(trajectories: list[Trajectory], device: str) -> torch.Tensor:
    targets = np.concatenate([trajectory.stage_targets for trajectory in trajectories])
    counts = np.asarray([(targets == stage).sum() for stage in range(NUM_STAGES)])
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def frame_losses(
    output: Any,
    batch: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    stage3_completion_weight: float,
    interval_margin: float,
) -> dict[str, torch.Tensor]:
    valid = batch["frame_mask"].bool()
    stage = F.cross_entropy(
        output.stage_logits[valid], batch["stages"][valid], weight=class_weights
    )
    local = F.smooth_l1_loss(output.local_progress[valid], batch["local"][valid])
    potential = F.l1_loss(
        output.potential[valid], batch["potential"][valid]
    )
    temporal = temporal_order_loss(
        output.potential, batch["potential"], batch["frame_mask"]
    )
    completion_logits = output.stage_logits[..., 3] - torch.logsumexp(
        output.stage_logits[..., :3], dim=-1
    )
    completion_targets = batch["stages"].eq(3).to(output.potential.dtype)
    completion_weights = torch.ones_like(completion_targets)
    completion_weights = torch.where(
        batch["stages"].eq(2),
        torch.full_like(completion_weights, stage3_completion_weight),
        completion_weights,
    )
    completion = F.binary_cross_entropy_with_logits(
        completion_logits[valid],
        completion_targets[valid],
        weight=completion_weights[valid],
    )
    lower = batch["stages"].to(output.potential.dtype) / NUM_STAGES
    upper = (batch["stages"].to(output.potential.dtype) + 1.0) / NUM_STAGES
    interval_error = F.relu(lower - interval_margin - output.potential) + F.relu(
        output.potential - upper - interval_margin
    )
    interval = interval_error[valid].mean()
    return {
        "stage": stage,
        "local": local,
        "potential": potential,
        "temporal": temporal,
        "completion": completion,
        "interval": interval,
    }


@torch.inference_mode()
def predict_windows(
    model: PhysicalProgressBranchV2,
    windows: list[Window],
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    loader = DataLoader(WindowDataset(windows), batch_size=batch_size, shuffle=False)
    predictions = []
    cursor = 0
    model.eval()
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        output = model(
            batch["depth"], batch["contact"], batch["task_id"], batch["frame_mask"]
        )
        count = batch["task_id"].shape[0]
        for index in range(count):
            window = windows[cursor + index]
            last = int(batch["frame_mask"][index].long().sum().item()) - 1
            depth_valid_ratio = (
                float(
                    batch["depth"][
                        index, last, model.depth_feature_count :
                    ].mean().item()
                )
                if model.depth_feature_count > 0
                else 0.0
            )
            predictions.append(
                {
                    "sample_id": window.sample_id,
                    "end_index": window.end_index,
                    "source_type": window.source_type,
                    "source_group_id": window.source_group_id,
                    "target_stage": int(batch["stages"][index, last].item()),
                    "target_local": float(batch["local"][index, last].item()),
                    "target_potential": float(batch["potential"][index, last].item()),
                    "predicted_stage": int(output.clip_stage[index].item()),
                    "predicted_local": float(output.clip_local_progress[index].item()),
                    "predicted_potential": float(output.clip_potential[index].item()),
                    "depth_gate": float(output.clip_modality_gates[index, 0].item()),
                    "contact_gate": float(output.clip_modality_gates[index, 1].item()),
                    "confidence": float(output.clip_confidence[index].item()),
                    "depth_valid_ratio": depth_valid_ratio,
                }
            )
        cursor += count
    return predictions


def pair_accuracy(
    predictions: list[dict[str, Any]], trajectories: list[Trajectory]
) -> tuple[float, int]:
    terminal_index = {
        trajectory.sample_id: len(trajectory.sensor_rows) - 1
        for trajectory in trajectories
    }
    potential = {
        row["sample_id"]: row["predicted_potential"]
        for row in predictions
        if row["end_index"] == terminal_index[row["sample_id"]]
    }
    grouped: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        if trajectory.terminal_rank >= 0 and trajectory.sample_id in potential:
            grouped[trajectory.source_group_id].append(trajectory)
    correct = []
    for group in grouped.values():
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                left, right = group[left_index], group[right_index]
                if left.terminal_rank == right.terminal_rank:
                    continue
                high, low = (
                    (left, right)
                    if left.terminal_rank > right.terminal_rank
                    else (right, left)
                )
                correct.append(potential[high.sample_id] > potential[low.sample_id])
    return (float(np.mean(correct)) if correct else float("nan"), len(correct))


def summarize_predictions(
    predictions: list[dict[str, Any]], trajectories: list[Trajectory]
) -> dict[str, Any]:
    target_stage = np.asarray([row["target_stage"] for row in predictions])
    predicted_stage = np.asarray([row["predicted_stage"] for row in predictions])
    target_local = np.asarray([row["target_local"] for row in predictions])
    predicted_local = np.asarray([row["predicted_local"] for row in predictions])
    target_potential = np.asarray([row["target_potential"] for row in predictions])
    predicted_potential = np.asarray(
        [row["predicted_potential"] for row in predictions]
    )
    pair_acc, pair_count = pair_accuracy(predictions, trajectories)

    trajectory_rhos = []
    for trajectory in trajectories:
        if trajectory.source_type != "official_motionplanning_success":
            continue
        rows = [row for row in predictions if row["sample_id"] == trajectory.sample_id]
        trajectory_rhos.append(
            spearman(
                [row["predicted_potential"] for row in rows],
                [row["target_potential"] for row in rows],
            )
        )
    terminal = []
    by_sample = defaultdict(list)
    for row in predictions:
        by_sample[row["sample_id"]].append(row)
    trajectory_map = {trajectory.sample_id: trajectory for trajectory in trajectories}
    for sample_id, rows in by_sample.items():
        final = max(rows, key=lambda row: row["end_index"])
        gt_success = truthy(
            trajectory_map[sample_id].supervision_rows[-1].get("gt_success")
        )
        terminal.append((final, gt_success))
    successes = [item for item in terminal if item[1]]
    failures = [item for item in terminal if not item[1]]
    finite_depth_gates = [
        row["depth_gate"]
        for row in predictions
        if math.isfinite(float(row["depth_gate"]))
    ]
    finite_contact_gates = [
        row["contact_gate"]
        for row in predictions
        if math.isfinite(float(row["contact_gate"]))
    ]
    depth_missing = [
        row for row in predictions if float(row.get("depth_valid_ratio", 0.0)) <= 0.0
    ]
    depth_present = [
        row for row in predictions if float(row.get("depth_valid_ratio", 0.0)) > 0.0
    ]
    noncomplete_frames = [row for row in predictions if row["target_stage"] != 3]
    stage3_frames = [row for row in predictions if row["target_stage"] == 2]
    near_miss_frames = [
        row
        for row in predictions
        if row.get("source_type")
        in {"perturbed_success_final_state", "truncated_success_trajectory"}
        and row["target_stage"] != 3
    ]
    group_rhos = []
    for group_id in sorted({row["source_group_id"] for row in predictions}):
        rows = [row for row in predictions if row["source_group_id"] == group_id]
        group_rhos.append(
            spearman(
                [row["predicted_potential"] for row in rows],
                [row["target_potential"] for row in rows],
            )
        )
    return {
        "frames": len(predictions),
        "trajectories": len(trajectories),
        "stage_accuracy": float(np.mean(target_stage == predicted_stage)),
        "stage_macro_f1": macro_f1(target_stage, predicted_stage),
        "local_progress_mae": float(np.mean(np.abs(target_local - predicted_local))),
        "potential_mae": float(np.mean(np.abs(target_potential - predicted_potential))),
        "potential_spearman": spearman(predicted_potential, target_potential),
        "terminal_pair_accuracy": pair_acc,
        "terminal_pair_count": pair_count,
        "mean_success_trajectory_spearman": float(
            np.nanmean(trajectory_rhos) if trajectory_rhos else float("nan")
        ),
        "success_stage4_recall": float(
            np.mean([row[0]["predicted_stage"] == 3 for row in successes])
            if successes
            else float("nan")
        ),
        "failure_stage4_false_positive_rate": float(
            np.mean([row[0]["predicted_stage"] == 3 for row in failures])
            if failures
            else float("nan")
        ),
        "frame_stage4_false_positive_rate": float(
            np.mean([row["predicted_stage"] == 3 for row in noncomplete_frames])
            if noncomplete_frames
            else float("nan")
        ),
        "stage3_to_stage4_frame_rate": float(
            np.mean([row["predicted_stage"] == 3 for row in stage3_frames])
            if stage3_frames
            else float("nan")
        ),
        "near_miss_frame_potential_ge_075_rate": float(
            np.mean(
                [row["predicted_potential"] >= 0.75 for row in near_miss_frames]
            )
            if near_miss_frames
            else float("nan")
        ),
        "mean_depth_gate": float(
            np.mean(finite_depth_gates) if finite_depth_gates else float("nan")
        ),
        "mean_contact_gate": float(
            np.mean(finite_contact_gates) if finite_contact_gates else float("nan")
        ),
        "depth_missing_frames": len(depth_missing),
        "potential_mae_depth_missing": float(
            np.mean(
                [
                    abs(row["predicted_potential"] - row["target_potential"])
                    for row in depth_missing
                ]
            )
            if depth_missing
            else float("nan")
        ),
        "potential_mae_depth_present": float(
            np.mean(
                [
                    abs(row["predicted_potential"] - row["target_potential"])
                    for row in depth_present
                ]
            )
            if depth_present
            else float("nan")
        ),
        "worst_source_group_potential_spearman": float(
            np.nanmin(group_rhos) if group_rhos else float("nan")
        ),
    }


def evaluate_split(
    model: PhysicalProgressBranchV2,
    trajectories: list[Trajectory],
    device: str,
    batch_size: int,
    history_window: int,
    sequence_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    windows = [
        make_window(trajectory, end, history_window, sequence_length)
        for trajectory in trajectories
        for end in range(len(trajectory.sensor_rows))
    ]
    predictions = predict_windows(model, windows, device, batch_size)
    return summarize_predictions(predictions, trajectories), predictions


def observable_rule_output(sensor: dict[str, str]) -> tuple[int, float, float]:
    gripper_distance = number(sensor.get("gripper_object_3d_dist_m"))
    goal_distance = number(sensor.get("object_goal_3d_dist_m"))
    moved = number(sensor.get("object_moved_from_start_m"), 0.0)
    grasped = max(
        number(sensor.get("is_grasping_object"), 0.0),
        number(sensor.get("grasp_confirmed"), 0.0),
    )
    touched = number(sensor.get("both_fingers_touch_object"), 0.0)
    support = number(sensor.get("object_support_contacts"), 0.0)
    released = number(sensor.get("released_object"), 0.0)
    static = number(sensor.get("object_static_proxy"), 0.0)
    spatial_success = math.isfinite(goal_distance) and goal_distance <= 0.035
    if spatial_success and support >= 0.5 and released >= 0.5 and static >= 0.5:
        stage = 3
    elif support >= 0.5 or (
        math.isfinite(goal_distance)
        and goal_distance <= 0.050
        and released >= 0.5
        and grasped < 0.5
    ):
        stage = 2
    elif grasped >= 0.5 or touched >= 0.5 or moved > 0.02:
        stage = 1
    else:
        stage = 0

    if stage == 0:
        local = (
            float(np.clip((0.35 - gripper_distance) / 0.325, 0.0, 1.0))
            if math.isfinite(gripper_distance)
            else 0.0
        )
    elif stage == 1:
        local = (
            float(np.clip(1.0 - goal_distance / 0.30, 0.0, 1.0))
            if math.isfinite(goal_distance)
            else 0.0
        )
    elif stage == 2:
        spatial = (
            float(np.clip(1.0 - goal_distance / 0.08, 0.0, 1.0))
            if math.isfinite(goal_distance)
            else 0.0
        )
        local = float(
            np.clip(
                0.45 * spatial
                + 0.30 * support
                + 0.15 * static
                + 0.10 * released,
                0.0,
                1.0,
            )
        )
    else:
        local = 1.0
    return stage, local, float((stage + local) / NUM_STAGES)


def evaluate_observable_rule(
    trajectories: list[Trajectory],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = []
    for trajectory in trajectories:
        for end_index, sensor in enumerate(trajectory.sensor_rows):
            predicted_stage, predicted_local, predicted_potential = (
                observable_rule_output(sensor)
            )
            predictions.append(
                {
                    "sample_id": trajectory.sample_id,
                    "end_index": end_index,
                    "source_type": trajectory.source_type,
                    "source_group_id": trajectory.source_group_id,
                    "target_stage": int(trajectory.stage_targets[end_index]),
                    "target_local": float(trajectory.local_targets[end_index]),
                    "target_potential": float(
                        trajectory.potential_targets[end_index]
                    ),
                    "predicted_stage": predicted_stage,
                    "predicted_local": predicted_local,
                    "predicted_potential": predicted_potential,
                    "depth_gate": float("nan"),
                    "contact_gate": float("nan"),
                    "confidence": 1.0,
                    "depth_valid_ratio": float(
                        trajectory.depth_valid[end_index].mean()
                    ),
                }
            )
    return summarize_predictions(predictions, trajectories), predictions


def selection_score(metrics: dict[str, Any]) -> float:
    rho = max(0.0, float(metrics["potential_spearman"]))
    pair = float(metrics["terminal_pair_accuracy"])
    pair = pair if math.isfinite(pair) else 0.0
    return (
        0.25 * float(metrics["stage_macro_f1"])
        + 0.25 * rho
        + 0.35 * pair
        + 0.15 * max(0.0, 1.0 - float(metrics["potential_mae"]))
    )


def train_variant(
    args: argparse.Namespace,
    variant: str,
    trajectories: dict[str, list[Trajectory]],
    depth_names: list[str],
    contact_names: list[str],
    depth_center: np.ndarray,
    depth_scale: np.ndarray,
    contact_center: np.ndarray,
    contact_scale: np.ndarray,
    device: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    use_depth = variant != "contact_only"
    use_contact = variant != "depth_only"
    model = PhysicalProgressBranchV2(
        depth_feature_count=len(depth_names),
        contact_feature_count=len(contact_names),
        num_tasks=1,
        num_stages=NUM_STAGES,
        modality_hidden_dim=args.modality_hidden_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        task_embedding_dim=args.task_embedding_dim,
        dropout=args.dropout,
        use_depth=use_depth,
        use_contact=use_contact,
        direct_potential_head=args.direct_potential_head,
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters >= 1_000_000:
        raise RuntimeError(f"model exceeds 1M parameter budget: {parameters}")

    train_windows = [
        make_window(trajectory, end, args.history_window, args.sequence_length)
        for trajectory in trajectories["train"]
        for end in range(len(trajectory.sensor_rows))
    ]
    train_pairs = terminal_pairs(
        trajectories["train"], args.history_window, args.sequence_length
    )
    source_counts = Counter(window.source_type for window in train_windows)
    sample_weights = torch.tensor(
        [1.0 / source_counts[window.source_type] for window in train_windows],
        dtype=torch.double,
    )
    balanced_sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_windows),
        replacement=True,
    )
    train_loader = DataLoader(
        WindowDataset(train_windows),
        batch_size=args.batch_size,
        sampler=balanced_sampler,
        num_workers=0,
    )
    pair_loader = DataLoader(
        PairDataset(train_pairs),
        batch_size=min(args.pair_batch_size, max(1, len(train_pairs))),
        shuffle=True,
        num_workers=0,
    )
    class_weights = compute_class_weights(trajectories["train"], device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        pair_iterator = iter(pair_loader)
        totals = defaultdict(float)
        examples = 0
        for raw_batch in train_loader:
            batch = to_device(raw_batch, device)
            try:
                raw_pair = next(pair_iterator)
            except StopIteration:
                pair_iterator = iter(pair_loader)
                raw_pair = next(pair_iterator)
            pair = to_device(raw_pair, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["depth"],
                batch["contact"],
                batch["task_id"],
                batch["frame_mask"],
                modality_dropout=args.modality_dropout,
            )
            components = frame_losses(
                output,
                batch,
                class_weights,
                args.stage3_completion_weight,
                args.interval_margin,
            )
            high = model(
                pair["high_depth"],
                pair["high_contact"],
                pair["task_id"],
                pair["high_mask"],
                modality_dropout=args.modality_dropout,
            )
            low = model(
                pair["low_depth"],
                pair["low_contact"],
                pair["task_id"],
                pair["low_mask"],
                modality_dropout=args.modality_dropout,
            )
            components["pair"] = pairwise_preference_loss(
                high.clip_potential, low.clip_potential, args.preference_temperature
            )
            loss = (
                args.stage_loss_weight * components["stage"]
                + args.local_loss_weight * components["local"]
                + args.potential_loss_weight * components["potential"]
                + args.temporal_loss_weight * components["temporal"]
                + args.pair_loss_weight * components["pair"]
                + args.completion_loss_weight * components["completion"]
                + args.interval_loss_weight * components["interval"]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            count = int(batch["task_id"].numel())
            examples += count
            totals["total"] += float(loss.item()) * count
            for name, value in components.items():
                totals[name] += float(value.item()) * count

        val_metrics, _ = evaluate_split(
            model,
            trajectories["val"],
            device,
            args.eval_batch_size,
            args.history_window,
            args.sequence_length,
        )
        score = selection_score(val_metrics)
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / max(1, examples) for name, value in totals.items()},
            **{f"val_{name}": value for name, value in val_metrics.items()},
            "val_selection_score": score,
        }
        history.append(row)
        if score > best_score + args.min_improvement:
            best_score = score
            best_epoch = epoch
            patience = 0
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        else:
            patience += 1
        print(
            f"{variant} epoch={epoch:03d} loss={row['train_total']:.4f} "
            f"val_f1={val_metrics['stage_macro_f1']:.3f} "
            f"val_rho={val_metrics['potential_spearman']:.3f} "
            f"val_pair={val_metrics['terminal_pair_accuracy']:.3f} "
            f"score={score:.3f} best={best_score:.3f}",
            flush=True,
        )
        if patience >= args.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError(f"{variant} training produced no checkpoint")
    model.load_state_dict(best_state)
    variant_dir = args.out_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 7 if args.direct_potential_head else 6,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "variant": variant,
        "model_config": model.config(),
        "state_dict": model.state_dict(),
        "task_ids": ["stackcube"],
        "depth_feature_names": depth_names,
        "contact_feature_names": contact_names,
        "depth_center": depth_center.tolist(),
        "depth_scale": depth_scale.tolist(),
        "contact_center": contact_center.tolist(),
        "contact_scale": contact_scale.tolist(),
        "feature_clip_value": args.feature_clip_value,
        "sequence_length": args.sequence_length,
        "history_window": args.history_window,
        "target_definition": "privileged_gt_stage_and_stage_local_geometry_v2",
    }
    torch.save(checkpoint, variant_dir / "best_model.pt")
    write_csv(variant_dir / "train_history.csv", history)
    metrics_by_split = {}
    for split in SPLITS:
        metrics, predictions = evaluate_split(
            model,
            trajectories[split],
            device,
            args.eval_batch_size,
            args.history_window,
            args.sequence_length,
        )
        metrics_by_split[split] = metrics
        write_csv(variant_dir / f"{split}_frame_predictions.csv", predictions)
    summary = {
        "variant": variant,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "best_val_selection_score": best_score,
        "checkpoint": str(variant_dir / "best_model.pt"),
        "metrics": metrics_by_split,
    }
    write_json(variant_dir / "metrics.json", metrics_by_split)
    write_json(variant_dir / "run_summary.json", summary)
    return summary, metrics_by_split


def verify_group_splits(trajectories: list[Trajectory]) -> dict[str, str]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for trajectory in trajectories:
        group_splits[trajectory.source_group_id].add(trajectory.split)
    leaking = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaking:
        raise RuntimeError(f"source-group leakage: {leaking}")
    return {group: next(iter(splits)) for group, splits in group_splits.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("depth_only", "contact_only", "fusion"),
        default=("depth_only", "contact_only", "fusion"),
    )
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--history-window", type=int, default=24)
    parser.add_argument("--feature-clip-value", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pair-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--modality-hidden-dim", type=int, default=96)
    parser.add_argument("--temporal-hidden-dim", type=int, default=128)
    parser.add_argument("--task-embedding-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--modality-dropout", type=float, default=0.10)
    parser.add_argument(
        "--direct-potential-head",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--stage-loss-weight", type=float, default=1.0)
    parser.add_argument("--local-loss-weight", type=float, default=0.40)
    parser.add_argument("--potential-loss-weight", type=float, default=1.20)
    parser.add_argument("--temporal-loss-weight", type=float, default=0.10)
    parser.add_argument("--pair-loss-weight", type=float, default=0.50)
    parser.add_argument("--completion-loss-weight", type=float, default=0.50)
    parser.add_argument("--interval-loss-weight", type=float, default=0.30)
    parser.add_argument("--stage3-completion-weight", type=float, default=4.0)
    parser.add_argument("--interval-margin", type=float, default=0.01)
    parser.add_argument("--preference-temperature", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--group-fold-index",
        type=int,
        default=-1,
        help="Rotate source groups: selected index=test, next index=val, others=train.",
    )
    parser.add_argument(
        "--exclude-label-conflicts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude manifest-failure trajectories that trigger simulator success.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"
    sensor_path = args.dataset_dir / "sensor_features.csv"
    supervision_path = args.dataset_dir / "offline_supervision.csv"
    aligned = aligned_tables(sensor_path, supervision_path)
    sensor_rows = [pair[0] for pair in aligned]
    depth_names = select_features(
        sensor_rows, DEPTH_FEATURE_WHITELIST
    )
    contact_names = select_features(
        sensor_rows, CONTACT_FEATURE_WHITELIST
    )
    if "gripper_object_3d_dist_m" not in depth_names:
        raise RuntimeError(
            "v2 requires observable gripper-object geometry; recollect with "
            "--include-gripper-geometry"
        )
    trajectories_all = build_trajectories(
        aligned, depth_names, contact_names
    )
    conflict_rows = []
    clean_trajectories = []
    for trajectory in trajectories_all:
        reason = label_conflict_reason(trajectory)
        if reason is None:
            clean_trajectories.append(trajectory)
            continue
        conflict_rows.append(
            {
                "sample_id": trajectory.sample_id,
                "original_split": trajectory.split,
                "source_type": trajectory.source_type,
                "source_group_id": trajectory.source_group_id,
                "reason": reason,
                "frames": len(trajectory.sensor_rows),
                "gt_success_frames": sum(
                    truthy(row.get("gt_success"))
                    for row in trajectory.supervision_rows
                ),
                "terminal_gt_success": int(
                    truthy(trajectory.supervision_rows[-1].get("gt_success"))
                ),
                "terminal_rank": trajectory.terminal_rank,
            }
        )
    if args.exclude_label_conflicts:
        trajectories_all = clean_trajectories
    fold_definition = None
    if args.group_fold_index >= 0:
        groups = sorted({trajectory.source_group_id for trajectory in trajectories_all})
        if args.group_fold_index >= len(groups):
            raise ValueError(
                f"--group-fold-index must be in [0, {len(groups) - 1}]"
            )
        test_group = groups[args.group_fold_index]
        val_group = groups[(args.group_fold_index + 1) % len(groups)]
        for trajectory in trajectories_all:
            if trajectory.source_group_id == test_group:
                trajectory.split = "test"
            elif trajectory.source_group_id == val_group:
                trajectory.split = "val"
            else:
                trajectory.split = "train"
        fold_definition = {
            "fold_index": args.group_fold_index,
            "test_group": test_group,
            "val_group": val_group,
            "train_groups": [
                group for group in groups if group not in {test_group, val_group}
            ],
        }
    group_splits = verify_group_splits(trajectories_all)
    trajectories = {
        split: [trajectory for trajectory in trajectories_all if trajectory.split == split]
        for split in SPLITS
    }
    if any(not trajectories[split] for split in SPLITS):
        raise RuntimeError("train/val/test must all contain trajectories")

    train_depth = [
        (trajectory.depth_raw, trajectory.depth_valid)
        for trajectory in trajectories["train"]
    ]
    train_contact = [
        (trajectory.contact_raw, trajectory.contact_valid)
        for trajectory in trajectories["train"]
    ]
    depth_center, depth_scale = robust_stats(train_depth, len(depth_names))
    contact_center, contact_scale = robust_stats(train_contact, len(contact_names))
    for trajectory in trajectories_all:
        trajectory.depth_features = prepare_numpy(
            trajectory.depth_raw,
            trajectory.depth_valid,
            depth_center,
            depth_scale,
            args.feature_clip_value,
        )
        trajectory.contact_features = prepare_numpy(
            trajectory.contact_raw,
            trajectory.contact_valid,
            contact_center,
            contact_scale,
            args.feature_clip_value,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "excluded_label_conflicts.csv", conflict_rows)
    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "device": device,
        "seed": args.seed,
        "variants": args.variants,
        "depth_feature_names": depth_names,
        "contact_feature_names": contact_names,
        "depth_center": depth_center.tolist(),
        "depth_scale": depth_scale.tolist(),
        "contact_center": contact_center.tolist(),
        "contact_scale": contact_scale.tolist(),
        "source_group_splits": group_splits,
        "group_fold_definition": fold_definition,
        "exclude_label_conflicts": bool(args.exclude_label_conflicts),
        "excluded_label_conflict_count": (
            len(conflict_rows) if args.exclude_label_conflicts else 0
        ),
        "trajectory_counts": {
            split: len(trajectories[split]) for split in SPLITS
        },
        "frame_counts": {
            split: sum(len(item.sensor_rows) for item in trajectories[split])
            for split in SPLITS
        },
        "stage_counts_train": dict(
            Counter(
                int(stage)
                for trajectory in trajectories["train"]
                for stage in trajectory.stage_targets
            )
        ),
        "target_boundary": {
            "model_inputs": "sensor_features.csv explicit whitelist only",
            "training_only_supervision": "offline_supervision.csv gt_* fields",
            "online_privileged_inputs": False,
        },
        "training_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    write_json(args.out_dir / "run_config.json", run_config)
    summaries = []
    metrics: dict[str, dict[str, dict[str, Any]]] = {"observable_rule": {}}
    rule_dir = args.out_dir / "observable_rule"
    rule_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        split_metrics, predictions = evaluate_observable_rule(trajectories[split])
        metrics["observable_rule"][split] = split_metrics
        write_csv(rule_dir / f"{split}_frame_predictions.csv", predictions)
    write_json(rule_dir / "metrics.json", metrics["observable_rule"])
    for variant in args.variants:
        summary, split_metrics = train_variant(
            args,
            variant,
            trajectories,
            depth_names,
            contact_names,
            depth_center,
            depth_scale,
            contact_center,
            contact_scale,
            device,
        )
        summaries.append(summary)
        metrics[variant] = split_metrics

    comparison_rows = []
    for variant in ["observable_rule", *args.variants]:
        for split in SPLITS:
            comparison_rows.append(
                {"variant": variant, "split": split, **metrics[variant][split]}
            )
    write_csv(args.out_dir / "variant_comparison.csv", comparison_rows)
    write_json(args.out_dir / "all_metrics.json", metrics)
    lines = [
        "# Physical Progress Branch v2",
        "",
        f"- sensor dataset: `{args.dataset_dir}`",
        f"- independent source groups: {len(group_splits)}",
        f"- depth features: {len(depth_names)}",
        f"- contact features: {len(contact_names)}",
        "- simulator truth is training/evaluation supervision only",
        "",
        "| variant | split | stage macro-F1 | potential rho | pair accuracy | potential MAE | success rho | terminal FPR | frame FPR | near-miss >=0.75 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['variant']} | {row['split']} | {row['stage_macro_f1']:.3f} | "
            f"{row['potential_spearman']:.3f} | {row['terminal_pair_accuracy']:.3f} | "
            f"{row['potential_mae']:.3f} | {row['mean_success_trajectory_spearman']:.3f} | "
            f"{row['failure_stage4_false_positive_rate']:.3f} | "
            f"{row['frame_stage4_false_positive_rate']:.3f} | "
            f"{row['near_miss_frame_potential_ge_075_rate']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation Boundary",
        "",
        "These are supervised sensor-domain results on eight source-success groups.",
        "They do not establish policy-learning improvement or broad task generalization.",
    ]
    (args.out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.out_dir / "RESULTS.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
