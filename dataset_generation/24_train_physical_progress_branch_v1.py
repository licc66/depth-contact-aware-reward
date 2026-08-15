from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from physical_progress_branch_v1 import (
    PhysicalProgressBranch,
    monotonic_progress_loss,
    pair_preference_probability,
    prepare_physical_features,
    progress_regression_loss,
    stage_classification_loss,
    weighted_pairwise_loss,
)
from reward_common_v1 import trajectory_split_map


TASK_IDS = ("peginsertion", "stackcube", "stackpyramid")
TASK_INDEX = {task_id: idx for idx, task_id in enumerate(TASK_IDS)}
SPLITS = ("train", "val", "test")
VALID_LABELS = {"A>B", "B>A"}
NUM_STAGES = 4

# These are observable or replaceable by a real stereo/contact backend. Rule
# outputs, simulator success, time progress, and stage IDs are deliberately
# excluded from the model inputs.
DEPTH_FEATURE_CANDIDATES = (
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
    "depth_confidence",
    "depth_valid_ratio",
    "disparity_confidence",
)

CONTACT_FEATURE_CANDIDATES = (
    "object_speed_proxy_m_per_step",
    "is_grasping_object",
    "finger_object_contact_force_n",
    "finger_object_contact",
    "object_support_contact_force_n",
    "object_support_contacts",
    "released_object",
    "object_static_proxy",
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

NEAR_MISS_CONTACT_AUGMENT_FEATURES = (
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

FEATURE_CANDIDATES = DEPTH_FEATURE_CANDIDATES + CONTACT_FEATURE_CANDIDATES


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


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
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_numeric(value: Any) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return 1.0
    if lowered in {"false", "no"}:
        return 0.0
    try:
        parsed = float(text)
        return parsed if math.isfinite(parsed) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def parse_indices(value: str) -> list[int]:
    return [int(float(item)) for item in value.replace(",", ";").split(";") if item.strip()]


def as_float(value: Any, default: float = 0.0) -> float:
    parsed = parse_numeric(value)
    return parsed if math.isfinite(parsed) else default


def supervision_weight(row: dict[str, str], column: str) -> float:
    if column == "uniform":
        return 1.0
    text = str(row.get(column, "")).strip().lower()
    confidence_weights = {"high": 1.0, "medium": 0.7, "low": 0.4}
    if text in confidence_weights:
        return confidence_weights[text]
    return max(as_float(text, 1.0), 1e-4)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resample_indices(indices: list[int], length: int) -> tuple[list[int], list[bool]]:
    if not indices:
        return [0] * length, [False] * length
    if len(indices) >= length:
        positions = np.linspace(0, len(indices) - 1, length).round().astype(int)
        return [indices[position] for position in positions], [True] * length
    padded = list(indices) + [indices[-1]] * (length - len(indices))
    mask = [True] * len(indices) + [False] * (length - len(indices))
    return padded, mask


@dataclass
class FrameRecord:
    values: np.ndarray
    valid: np.ndarray
    stage_target: int
    local_progress_target: float
    split: str


@dataclass
class ClipSequence:
    raw_features: np.ndarray
    valid_features: np.ndarray
    frame_mask: np.ndarray
    stage_targets: np.ndarray
    progress_targets: np.ndarray
    monotonic_eligible: bool


@dataclass
class RawPairSplit:
    rows: list[dict[str, str]]
    raw_a: np.ndarray
    raw_b: np.ndarray
    valid_a: np.ndarray
    valid_b: np.ndarray
    mask_a: np.ndarray
    mask_b: np.ndarray
    stage_a: np.ndarray
    stage_b: np.ndarray
    progress_a: np.ndarray
    progress_b: np.ndarray
    task_ids: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    monotonic_a: np.ndarray
    monotonic_b: np.ndarray


@dataclass
class PreparedPairSplit:
    rows: list[dict[str, str]]
    features_a: np.ndarray
    features_b: np.ndarray
    mask_a: np.ndarray
    mask_b: np.ndarray
    stage_a: np.ndarray
    stage_b: np.ndarray
    progress_a: np.ndarray
    progress_b: np.ndarray
    task_ids: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    monotonic_a: np.ndarray
    monotonic_b: np.ndarray


@dataclass
class RawAuxiliarySplit:
    sample_ids: list[str]
    raw_features: np.ndarray
    valid_features: np.ndarray
    frame_mask: np.ndarray
    stage_targets: np.ndarray
    progress_targets: np.ndarray
    task_ids: np.ndarray


@dataclass
class PreparedAuxiliarySplit:
    sample_ids: list[str]
    features: np.ndarray
    frame_mask: np.ndarray
    stage_targets: np.ndarray
    progress_targets: np.ndarray
    task_ids: np.ndarray


class PairSequenceDataset(Dataset):
    def __init__(self, split: PreparedPairSplit):
        self.split = split

    def __len__(self) -> int:
        return len(self.split.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features_a": torch.from_numpy(self.split.features_a[index]),
            "features_b": torch.from_numpy(self.split.features_b[index]),
            "mask_a": torch.from_numpy(self.split.mask_a[index]),
            "mask_b": torch.from_numpy(self.split.mask_b[index]),
            "stage_a": torch.from_numpy(self.split.stage_a[index]),
            "stage_b": torch.from_numpy(self.split.stage_b[index]),
            "progress_a": torch.from_numpy(self.split.progress_a[index]),
            "progress_b": torch.from_numpy(self.split.progress_b[index]),
            "task_id": torch.tensor(self.split.task_ids[index], dtype=torch.long),
            "label": torch.tensor(self.split.labels[index], dtype=torch.float32),
            "weight": torch.tensor(self.split.weights[index], dtype=torch.float32),
            "monotonic_a": torch.tensor(bool(self.split.monotonic_a[index]), dtype=torch.bool),
            "monotonic_b": torch.tensor(bool(self.split.monotonic_b[index]), dtype=torch.bool),
        }


class AuxiliarySequenceDataset(Dataset):
    def __init__(self, split: PreparedAuxiliarySplit):
        self.split = split

    def __len__(self) -> int:
        return len(self.split.sample_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.split.features[index]),
            "frame_mask": torch.from_numpy(self.split.frame_mask[index]),
            "stage_targets": torch.from_numpy(self.split.stage_targets[index]),
            "progress_targets": torch.from_numpy(
                self.split.progress_targets[index]
            ),
            "task_id": torch.tensor(self.split.task_ids[index], dtype=torch.long),
        }


def task_folder(task_id: str) -> str:
    return f"{task_id}_bootstrap_v1"


def is_clean_success(row: dict[str, str]) -> bool:
    sample_id = row.get("sample_id", row.get("trajectory_id", "")).upper()
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


def build_stage_local_progress_targets(
    contact_rows: list[dict[str, str]],
) -> dict[tuple[str, int], float]:
    stage_frames: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in contact_rows:
        if not is_clean_success(row):
            continue
        stage_id = int(as_float(row.get("stage_id", "0"), 0.0))
        if not 1 <= stage_id <= NUM_STAGES:
            continue
        sample_id = row["sample_id"]
        frame_idx = int(float(row["frame_idx"]))
        stage_frames[(sample_id, stage_id)].append(frame_idx)

    targets: dict[tuple[str, int], float] = {}
    for (sample_id, stage_id), frame_indices in stage_frames.items():
        ordered = sorted(set(frame_indices))
        if stage_id == NUM_STAGES:
            for frame_idx in ordered:
                targets[(sample_id, frame_idx)] = 1.0
            continue
        if len(ordered) == 1:
            targets[(sample_id, ordered[0])] = 1.0
            continue
        first, last = ordered[0], ordered[-1]
        denominator = max(last - first, 1)
        for frame_idx in ordered:
            targets[(sample_id, frame_idx)] = (frame_idx - first) / denominator
    return targets


def load_physical_frame_store(
    dataset_root: Path,
) -> tuple[
    dict[tuple[str, str, int], FrameRecord],
    dict[str, dict[str, str]],
    dict[str, Any],
]:
    frame_store: dict[tuple[str, str, int], FrameRecord] = {}
    clip_map: dict[str, dict[str, str]] = {}
    augmented_frames = 0
    augmented_trajectories: set[str] = set()

    for task_id in TASK_IDS:
        folder = task_folder(task_id)
        stereo_rows = load_csv(
            dataset_root / "stereo_features" / folder / "frame_stereo_geometry_features.csv"
        )
        contact_rows = load_csv(
            dataset_root / "contact_stage_features" / folder / "frame_contact_stage_features.csv"
        )
        contact_by_key = {
            (row["sample_id"], int(float(row["frame_idx"]))): row for row in contact_rows
        }
        contact_rows_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in contact_rows:
            contact_rows_by_sample[row["sample_id"]].append(row)
        clean_success_terminal = {
            sample_id: max(
                rows,
                key=lambda row: int(float(row["frame_idx"])),
            )
            for sample_id, rows in contact_rows_by_sample.items()
            if rows and is_clean_success(rows[-1])
        }
        local_progress_targets = build_stage_local_progress_targets(contact_rows)

        for stereo_row in stereo_rows:
            frame_idx = int(float(stereo_row["frame_idx"]))
            sample_id = stereo_row["sample_id"]
            contact_row = contact_by_key.get((sample_id, frame_idx), {})
            merged = {**stereo_row, **contact_row}
            if (
                contact_row.get("source_type", "") == "perturbed_success_final_state"
                and parse_numeric(contact_row.get("contact_observation_valid", "")) == 0.0
            ):
                reference = clean_success_terminal.get(source_success_id(sample_id))
                if reference is not None:
                    copied_any = False
                    for name in NEAR_MISS_CONTACT_AUGMENT_FEATURES:
                        value = parse_numeric(reference.get(name, ""))
                        if math.isfinite(value):
                            merged[name] = value
                            copied_any = True
                    if copied_any:
                        augmented_frames += 1
                        augmented_trajectories.add(sample_id)
            values = np.asarray([parse_numeric(merged.get(name, "")) for name in FEATURE_CANDIDATES], dtype=np.float32)
            valid = np.isfinite(values)
            stage_id = int(as_float(contact_row.get("stage_id", "0"), 0.0))
            stage_target = stage_id - 1 if 1 <= stage_id <= NUM_STAGES else -1
            frame_store[(task_id, sample_id, frame_idx)] = FrameRecord(
                values=values,
                valid=valid,
                stage_target=stage_target,
                local_progress_target=local_progress_targets.get(
                    (sample_id, frame_idx),
                    float("nan"),
                ),
                split=contact_row.get("split", stereo_row.get("split", "train")),
            )

        clip_rows = load_csv(
            dataset_root / "contact_stage_features" / folder / "clip_contact_stage_features.csv"
        )
        for row in clip_rows:
            clip_map[row["clip_id"]] = row

    augmentation_summary = {
        "strategy": (
            "For frozen teleport near-misses, copy observable contact/gripper "
            "features from the corresponding successful source trajectory. "
            "This is adversarial training augmentation so depth must reject "
            "spatial failure despite success-like contact."
        ),
        "augmented_frames": augmented_frames,
        "augmented_trajectories": len(augmented_trajectories),
    }
    return frame_store, clip_map, augmentation_summary


def build_auxiliary_splits(
    frame_store: dict[tuple[str, str, int], FrameRecord],
    sequence_length: int,
    history_window: int,
    canonical_splits: dict[tuple[str, str], str] | None = None,
) -> dict[str, RawAuxiliarySplit]:
    trajectories: dict[tuple[str, str], list[tuple[int, FrameRecord]]] = defaultdict(list)
    for (task_id, sample_id, frame_idx), record in frame_store.items():
        trajectories[(task_id, sample_id)].append((frame_idx, record))

    if canonical_splits is not None:
        missing = sorted(set(canonical_splits) - set(trajectories))
        if missing:
            preview = ", ".join(f"{task}/{sample}" for task, sample in missing[:5])
            raise RuntimeError(
                f"canonical split map references {len(missing)} trajectories with no "
                f"frame data (first: {preview})"
            )

    buckets: dict[str, dict[str, list[Any]]] = {
        split: defaultdict(list) for split in SPLITS
    }
    for (task_id, sample_id), indexed_records in sorted(trajectories.items()):
        ordered = [
            record for _, record in sorted(indexed_records, key=lambda item: item[0])
        ]
        if not ordered:
            continue
        if canonical_splits is not None:
            split = canonical_splits.get((task_id, sample_id), "")
        else:
            split = ordered[-1].split
        if split not in buckets:
            continue
        history = ordered[-history_window:]
        positions, frame_mask = resample_indices(
            list(range(len(history))),
            sequence_length,
        )
        records = [history[position] for position in positions]
        bucket = buckets[split]
        bucket["sample_ids"].append(sample_id)
        bucket["raw_features"].append(
            np.stack([record.values for record in records]).astype(np.float32)
        )
        bucket["valid_features"].append(
            np.stack([record.valid for record in records]).astype(bool)
        )
        bucket["frame_mask"].append(np.asarray(frame_mask, dtype=bool))
        bucket["stage_targets"].append(
            np.asarray([record.stage_target for record in records], dtype=np.int64)
        )
        bucket["progress_targets"].append(
            np.asarray(
                [record.local_progress_target for record in records],
                dtype=np.float32,
            )
        )
        bucket["task_ids"].append(TASK_INDEX[task_id])

    output: dict[str, RawAuxiliarySplit] = {}
    for split, bucket in buckets.items():
        if not bucket["sample_ids"]:
            raise RuntimeError(f"No auxiliary trajectories found for split={split}")
        output[split] = RawAuxiliarySplit(
            sample_ids=list(bucket["sample_ids"]),
            raw_features=np.stack(bucket["raw_features"]),
            valid_features=np.stack(bucket["valid_features"]),
            frame_mask=np.stack(bucket["frame_mask"]),
            stage_targets=np.stack(bucket["stage_targets"]),
            progress_targets=np.stack(bucket["progress_targets"]),
            task_ids=np.asarray(bucket["task_ids"], dtype=np.int64),
        )
    return output


def nearest_frame_record(
    frame_store: dict[tuple[str, str, int], FrameRecord],
    task_id: str,
    trajectory_id: str,
    frame_idx: int,
) -> FrameRecord:
    exact = frame_store.get((task_id, trajectory_id, frame_idx))
    if exact is not None:
        return exact
    candidates = [
        (abs(key[2] - frame_idx), record)
        for key, record in frame_store.items()
        if key[0] == task_id and key[1] == trajectory_id
    ]
    if not candidates:
        raise KeyError(f"No physical frames for {task_id}/{trajectory_id}")
    return min(candidates, key=lambda item: item[0])[1]


def build_clip_sequence(
    row: dict[str, str],
    side: str,
    frame_store: dict[tuple[str, str, int], FrameRecord],
    clip_map: dict[str, dict[str, str]],
    sequence_length: int,
) -> ClipSequence:
    clip_id = row[f"clip_{side}_id"]
    clip_info = clip_map.get(clip_id)
    if clip_info is None:
        raise KeyError(f"Missing clip metadata for {clip_id}")
    trajectory_id = clip_info["trajectory_id"]
    indices, frame_mask = resample_indices(
        parse_indices(row[f"clip_{side}_sample_frame_indices"]),
        sequence_length,
    )
    records = [
        nearest_frame_record(frame_store, row["task_id"], trajectory_id, frame_idx)
        for frame_idx in indices
    ]
    raw = np.stack([record.values for record in records]).astype(np.float32)
    valid = np.stack([record.valid for record in records]).astype(bool)
    stages = np.asarray([record.stage_target for record in records], dtype=np.int64)
    progress = np.asarray(
        [record.local_progress_target for record in records],
        dtype=np.float32,
    )

    near_miss_type = clip_info.get("near_miss_type", "").strip()
    source_type = clip_info.get("source_type", "").lower()
    trajectory_upper = trajectory_id.upper()
    monotonic_eligible = (
        "success" in source_type
        and not near_miss_type
        and "OFFSET" not in trajectory_upper
        and "TRUNC" not in trajectory_upper
    )
    return ClipSequence(
        raw_features=raw,
        valid_features=valid,
        frame_mask=np.asarray(frame_mask, dtype=bool),
        stage_targets=stages,
        progress_targets=progress,
        monotonic_eligible=monotonic_eligible,
    )


def build_raw_split(
    rows: list[dict[str, str]],
    frame_store: dict[tuple[str, str, int], FrameRecord],
    clip_map: dict[str, dict[str, str]],
    sequence_length: int,
    label_column: str,
    weight_column: str,
) -> RawPairSplit:
    kept_rows: list[dict[str, str]] = []
    sequences_a: list[ClipSequence] = []
    sequences_b: list[ClipSequence] = []
    labels: list[float] = []
    weights: list[float] = []
    task_ids: list[int] = []
    cache: dict[tuple[str, tuple[int, ...]], ClipSequence] = {}

    for row in rows:
        label = row.get(label_column, "")
        if label not in VALID_LABELS:
            continue
        side_sequences: dict[str, ClipSequence] = {}
        for side in ("a", "b"):
            cache_key = (
                row[f"clip_{side}_id"],
                tuple(parse_indices(row[f"clip_{side}_sample_frame_indices"])),
            )
            if cache_key not in cache:
                cache[cache_key] = build_clip_sequence(
                    row,
                    side,
                    frame_store,
                    clip_map,
                    sequence_length,
                )
            side_sequences[side] = cache[cache_key]
        kept_rows.append(row)
        sequences_a.append(side_sequences["a"])
        sequences_b.append(side_sequences["b"])
        labels.append(1.0 if label == "A>B" else 0.0)
        weights.append(supervision_weight(row, weight_column))
        task_ids.append(TASK_INDEX[row["task_id"]])

    def stack(attribute: str, sequences: list[ClipSequence]) -> np.ndarray:
        return np.stack([getattr(sequence, attribute) for sequence in sequences])

    return RawPairSplit(
        rows=kept_rows,
        raw_a=stack("raw_features", sequences_a),
        raw_b=stack("raw_features", sequences_b),
        valid_a=stack("valid_features", sequences_a),
        valid_b=stack("valid_features", sequences_b),
        mask_a=stack("frame_mask", sequences_a),
        mask_b=stack("frame_mask", sequences_b),
        stage_a=stack("stage_targets", sequences_a),
        stage_b=stack("stage_targets", sequences_b),
        progress_a=stack("progress_targets", sequences_a),
        progress_b=stack("progress_targets", sequences_b),
        task_ids=np.asarray(task_ids, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
        monotonic_a=np.asarray([sequence.monotonic_eligible for sequence in sequences_a], dtype=bool),
        monotonic_b=np.asarray([sequence.monotonic_eligible for sequence in sequences_b], dtype=bool),
    )


def select_and_fit_features(
    train: RawPairSplit,
    min_valid_ratio: float,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, dict[str, float]]:
    raw = np.concatenate([train.raw_a, train.raw_b], axis=0)
    valid = np.concatenate([train.valid_a, train.valid_b], axis=0)
    valid_ratio = valid.mean(axis=(0, 1))
    selected: list[int] = []
    centers: list[float] = []
    scales: list[float] = []
    ratios: dict[str, float] = {}

    for index, name in enumerate(FEATURE_CANDIDATES):
        ratios[name] = float(valid_ratio[index])
        observed = raw[..., index][valid[..., index]]
        if valid_ratio[index] < min_valid_ratio or observed.size == 0:
            continue
        center = float(np.median(observed))
        q25, q75 = np.percentile(observed, [25.0, 75.0])
        scale = float((q75 - q25) / 1.349)
        if scale < 1e-6:
            scale = float(np.std(observed))
        if scale < 1e-6:
            scale = 1.0
        selected.append(index)
        centers.append(center)
        scales.append(scale)

    if not selected:
        raise RuntimeError("No usable physical features were found")
    return (
        np.asarray(selected, dtype=np.int64),
        [FEATURE_CANDIDATES[index] for index in selected],
        np.asarray(centers, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
        ratios,
    )


def transform_feature_array(
    values: np.ndarray,
    valid: np.ndarray,
    selected: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    clip_value: float,
) -> np.ndarray:
    values = values[..., selected]
    valid = valid[..., selected]
    safe = np.where(valid, values, center)
    normalized = np.clip(
        (safe - center) / np.maximum(scale, 1e-6),
        -clip_value,
        clip_value,
    )
    return np.concatenate(
        [normalized, valid.astype(np.float32)],
        axis=-1,
    ).astype(np.float32)


def prepare_split(
    raw: RawPairSplit,
    selected: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    clip_value: float,
) -> PreparedPairSplit:
    return PreparedPairSplit(
        rows=raw.rows,
        features_a=transform_feature_array(
            raw.raw_a,
            raw.valid_a,
            selected,
            center,
            scale,
            clip_value,
        ),
        features_b=transform_feature_array(
            raw.raw_b,
            raw.valid_b,
            selected,
            center,
            scale,
            clip_value,
        ),
        mask_a=raw.mask_a,
        mask_b=raw.mask_b,
        stage_a=raw.stage_a,
        stage_b=raw.stage_b,
        progress_a=raw.progress_a,
        progress_b=raw.progress_b,
        task_ids=raw.task_ids,
        labels=raw.labels,
        weights=raw.weights,
        monotonic_a=raw.monotonic_a,
        monotonic_b=raw.monotonic_b,
    )


def prepare_auxiliary_split(
    raw: RawAuxiliarySplit,
    selected: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    clip_value: float,
) -> PreparedAuxiliarySplit:
    return PreparedAuxiliarySplit(
        sample_ids=raw.sample_ids,
        features=transform_feature_array(
            raw.raw_features,
            raw.valid_features,
            selected,
            center,
            scale,
            clip_value,
        ),
        frame_mask=raw.frame_mask,
        stage_targets=raw.stage_targets,
        progress_targets=raw.progress_targets,
        task_ids=raw.task_ids,
    )


def compute_stage_class_weights(
    train: PreparedPairSplit,
    auxiliary: PreparedAuxiliarySplit,
    device: str,
) -> torch.Tensor:
    targets = np.concatenate(
        [
            train.stage_a[train.mask_a],
            train.stage_b[train.mask_b],
            auxiliary.stage_targets[auxiliary.frame_mask],
        ]
    )
    counts = np.asarray([(targets == stage).sum() for stage in range(NUM_STAGES)], dtype=np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def batch_losses(
    model: PhysicalProgressBranch,
    batch: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output_a = model(batch["features_a"], batch["task_id"], batch["mask_a"])
    output_b = model(batch["features_b"], batch["task_id"], batch["mask_b"])
    pair_loss = weighted_pairwise_loss(
        output_a.clip_potential,
        output_b.clip_potential,
        batch["label"],
        batch["weight"],
        temperature=args.preference_temperature,
    )
    stage_loss_a = stage_classification_loss(
        output_a.stage_logits,
        batch["stage_a"],
        batch["mask_a"],
        class_weights,
    )
    stage_loss_b = stage_classification_loss(
        output_b.stage_logits,
        batch["stage_b"],
        batch["mask_b"],
        class_weights,
    )
    stage_loss = 0.5 * (stage_loss_a + stage_loss_b)
    progress_loss_a = progress_regression_loss(
        output_a.local_progress,
        batch["progress_a"],
        batch["mask_a"],
    )
    progress_loss_b = progress_regression_loss(
        output_b.local_progress,
        batch["progress_b"],
        batch["mask_b"],
    )
    progress_loss = 0.5 * (progress_loss_a + progress_loss_b)
    monotonic_loss_a = monotonic_progress_loss(
        output_a.potential,
        batch["mask_a"],
        batch["monotonic_a"],
    )
    monotonic_loss_b = monotonic_progress_loss(
        output_b.potential,
        batch["mask_b"],
        batch["monotonic_b"],
    )
    monotonic_loss = 0.5 * (monotonic_loss_a + monotonic_loss_b)
    total = (
        pair_loss
        + args.stage_loss_weight * stage_loss
        + args.progress_loss_weight * progress_loss
        + args.monotonic_loss_weight * monotonic_loss
    )
    return total, {
        "pair": pair_loss.detach(),
        "stage": stage_loss.detach(),
        "progress": progress_loss.detach(),
        "monotonic": monotonic_loss.detach(),
    }


def auxiliary_batch_losses(
    model: PhysicalProgressBranch,
    batch: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(
        batch["features"],
        batch["task_id"],
        batch["frame_mask"],
    )
    stage_loss = stage_classification_loss(
        output.stage_logits,
        batch["stage_targets"],
        batch["frame_mask"],
        class_weights,
    )
    progress_loss = progress_regression_loss(
        output.local_progress,
        batch["progress_targets"],
        batch["frame_mask"],
    )
    return stage_loss, progress_loss


@torch.no_grad()
def evaluate_auxiliary(
    model: PhysicalProgressBranch,
    prepared: PreparedAuxiliarySplit,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(
        AuxiliarySequenceDataset(prepared),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    stage_correct = 0
    stage_total = 0
    progress_errors: list[np.ndarray] = []
    terminal_correct = 0
    terminal_total = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(
            batch["features"],
            batch["task_id"],
            batch["frame_mask"],
        )
        valid = batch["frame_mask"] & batch["stage_targets"].ge(0)
        predictions = output.stage_probs.argmax(dim=-1)
        stage_correct += int(
            (predictions[valid] == batch["stage_targets"][valid]).sum().item()
        )
        stage_total += int(valid.sum().item())

        terminal_index = batch["frame_mask"].long().sum(dim=1).sub(1).clamp_min(0)
        batch_index = torch.arange(
            terminal_index.shape[0],
            device=terminal_index.device,
        )
        terminal_target = batch["stage_targets"][batch_index, terminal_index]
        terminal_prediction = output.clip_stage
        terminal_valid = terminal_target.ge(0)
        terminal_correct += int(
            (terminal_prediction[terminal_valid] == terminal_target[terminal_valid])
            .sum()
            .item()
        )
        terminal_total += int(terminal_valid.sum().item())

        progress_valid = batch["frame_mask"] & torch.isfinite(
            batch["progress_targets"]
        )
        if progress_valid.any():
            progress_errors.append(
                torch.abs(
                    output.local_progress[progress_valid]
                    - batch["progress_targets"][progress_valid]
                )
                .cpu()
                .numpy()
            )
    errors = (
        np.concatenate(progress_errors)
        if progress_errors
        else np.asarray([], dtype=np.float32)
    )
    return {
        "trajectories": len(prepared.sample_ids),
        "stage_frame_accuracy": stage_correct / max(stage_total, 1),
        "stage_frame_rows": stage_total,
        "terminal_stage_accuracy": terminal_correct / max(terminal_total, 1),
        "terminal_rows": terminal_total,
        "local_progress_mae": float(errors.mean()) if errors.size else None,
        "local_progress_supervised_frames": int(errors.size),
    }


@torch.no_grad()
def evaluate(
    model: PhysicalProgressBranch,
    prepared: PreparedPairSplit,
    device: str,
    batch_size: int,
    temperature: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = DataLoader(PairSequenceDataset(prepared), batch_size=batch_size, shuffle=False)
    model.eval()
    probability_parts: list[np.ndarray] = []
    potential_a_parts: list[np.ndarray] = []
    potential_b_parts: list[np.ndarray] = []
    confidence_a_parts: list[np.ndarray] = []
    confidence_b_parts: list[np.ndarray] = []
    stage_pred_a_parts: list[np.ndarray] = []
    stage_pred_b_parts: list[np.ndarray] = []
    local_progress_a_parts: list[np.ndarray] = []
    local_progress_b_parts: list[np.ndarray] = []

    stage_correct = 0
    stage_total = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output_a = model(batch["features_a"], batch["task_id"], batch["mask_a"])
        output_b = model(batch["features_b"], batch["task_id"], batch["mask_b"])
        probabilities = pair_preference_probability(
            output_a.clip_potential,
            output_b.clip_potential,
            temperature,
        )
        probability_parts.append(probabilities.cpu().numpy())
        potential_a_parts.append(output_a.clip_potential.cpu().numpy())
        potential_b_parts.append(output_b.clip_potential.cpu().numpy())
        confidence_a_parts.append(output_a.clip_confidence.cpu().numpy())
        confidence_b_parts.append(output_b.clip_confidence.cpu().numpy())
        stage_pred_a_parts.append(output_a.clip_stage.cpu().numpy())
        stage_pred_b_parts.append(output_b.clip_stage.cpu().numpy())
        local_progress_a_parts.append(output_a.local_progress.cpu().numpy())
        local_progress_b_parts.append(output_b.local_progress.cpu().numpy())

        for output, target, mask in (
            (output_a, batch["stage_a"], batch["mask_a"]),
            (output_b, batch["stage_b"], batch["mask_b"]),
        ):
            valid = mask & target.ge(0)
            stage_correct += int((output.stage_probs.argmax(dim=-1)[valid] == target[valid]).sum().item())
            stage_total += int(valid.sum().item())

    probabilities = np.concatenate(probability_parts)
    potentials_a = np.concatenate(potential_a_parts)
    potentials_b = np.concatenate(potential_b_parts)
    confidence_a = np.concatenate(confidence_a_parts)
    confidence_b = np.concatenate(confidence_b_parts)
    stage_pred_a = np.concatenate(stage_pred_a_parts)
    stage_pred_b = np.concatenate(stage_pred_b_parts)
    local_progress_a = np.concatenate(local_progress_a_parts)
    local_progress_b = np.concatenate(local_progress_b_parts)
    predicted = (probabilities >= 0.5).astype(np.float32)
    correct = predicted == prepared.labels
    progress_valid_a = prepared.mask_a & np.isfinite(prepared.progress_a)
    progress_valid_b = prepared.mask_b & np.isfinite(prepared.progress_b)
    progress_errors = np.concatenate(
        [
            np.abs(local_progress_a[progress_valid_a] - prepared.progress_a[progress_valid_a]),
            np.abs(local_progress_b[progress_valid_b] - prepared.progress_b[progress_valid_b]),
        ]
    )

    metrics: dict[str, Any] = {
        "rows": int(len(prepared.labels)),
        "pair_accuracy": float(correct.mean()),
        "stage_frame_accuracy": float(stage_correct / max(stage_total, 1)),
        "stage_frame_rows": int(stage_total),
        "local_progress_mae": (
            float(progress_errors.mean()) if progress_errors.size else None
        ),
        "local_progress_supervised_frames": int(progress_errors.size),
        "mean_pair_margin": float(np.abs(potentials_a - potentials_b).mean()),
        "mean_pair_probability_confidence": float((2.0 * np.abs(probabilities - 0.5)).mean()),
    }
    for key in ("task_id", "pair_type"):
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(prepared.rows):
            grouped[row.get(key, "")].append(index)
        metrics[f"by_{key}"] = {
            group: {
                "rows": len(indices),
                "pair_accuracy": float(correct[np.asarray(indices)].mean()),
            }
            for group, indices in sorted(grouped.items())
        }

    hard_mask = np.asarray(
        [
            any(token in row.get("pair_type", "") for token in ("near_miss", "truncated", "offset"))
            for row in prepared.rows
        ],
        dtype=bool,
    )
    if hard_mask.any():
        metrics["hard_negative_rows"] = int(hard_mask.sum())
        metrics["hard_negative_accuracy"] = float(correct[hard_mask].mean())

    prediction_rows: list[dict[str, Any]] = []
    for index, row in enumerate(prepared.rows):
        pair_confidence = (
            2.0
            * abs(float(probabilities[index]) - 0.5)
            * math.sqrt(max(float(confidence_a[index] * confidence_b[index]), 0.0))
        )
        prediction_rows.append(
            {
                "pair_id": row["pair_id"],
                "task_id": row["task_id"],
                "pair_type": row["pair_type"],
                "target_label": "A>B" if prepared.labels[index] == 1.0 else "B>A",
                "predicted_label": "A>B" if predicted[index] == 1.0 else "B>A",
                "correct": bool(correct[index]),
                "probability_a_better": float(probabilities[index]),
                "potential_a": float(potentials_a[index]),
                "potential_b": float(potentials_b[index]),
                "potential_margin_a_minus_b": float(potentials_a[index] - potentials_b[index]),
                "stage_a": int(stage_pred_a[index] + 1),
                "stage_b": int(stage_pred_b[index] + 1),
                "stage_confidence_a": float(confidence_a[index]),
                "stage_confidence_b": float(confidence_b[index]),
                "pair_confidence": pair_confidence,
            }
        )
    return metrics, prediction_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a causal depth/contact stage-conditioned physical progress branch."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\Users\User\Desktop\reward_model_dataset"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path(
            r"D:\Users\User\Desktop\reward_model_dataset\dataset_splits"
            r"\bootstrap_v1_fusion_stereo_v1_clean"
        ),
    )
    parser.add_argument(
        "--split-map-dir",
        type=Path,
        default=None,
        help="Canonical script-17 split directory for auxiliary trajectory "
        "supervision. If omitted, the legacy per-frame split is used only for "
        "backward reproduction and validation metrics are contaminated.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            r"D:\Users\User\Desktop\reward_model_dataset\reward_model_runs"
            r"\physical_progress_branch_v1"
        ),
    )
    parser.add_argument("--label-column", default="candidate_label")
    parser.add_argument("--weight-column", default="candidate_confidence")
    parser.add_argument("--sequence-length", type=int, default=6)
    parser.add_argument("--history-window", type=int, default=16)
    parser.add_argument("--min-feature-valid-ratio", type=float, default=0.02)
    parser.add_argument("--feature-clip-value", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--frame-hidden-dim", type=int, default=128)
    parser.add_argument("--temporal-hidden-dim", type=int, default=128)
    parser.add_argument("--task-embedding-dim", type=int, default=16)
    parser.add_argument("--num-gru-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--preference-temperature", type=float, default=0.10)
    parser.add_argument("--completion-threshold", type=float, default=0.80)
    parser.add_argument("--stage-loss-weight", type=float, default=0.30)
    parser.add_argument("--progress-loss-weight", type=float, default=0.50)
    parser.add_argument("--monotonic-loss-weight", type=float, default=0.05)
    parser.add_argument("--aux-stage-loss-weight", type=float, default=0.50)
    parser.add_argument("--aux-progress-loss-weight", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.history_window < args.sequence_length:
        raise ValueError("history-window must be at least sequence-length")
    set_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"dataset_root={args.dataset_root}")
    print("loading per-frame stereo/contact features...")
    frame_store, clip_map, contact_augmentation = load_physical_frame_store(
        args.dataset_root
    )
    print(
        "near_miss_contact_augmentation="
        f"{contact_augmentation['augmented_trajectories']} trajectories / "
        f"{contact_augmentation['augmented_frames']} frames"
    )
    canonical_splits = (
        trajectory_split_map(args.split_map_dir)
        if args.split_map_dir is not None
        else None
    )
    if canonical_splits is None:
        print(
            "WARNING: auxiliary supervision is using the legacy per-frame split. "
            "Pass --split-map-dir equal to --split-dir for leakage-free metrics.",
            file=sys.stderr,
        )
    else:
        print(
            f"auxiliary_split_source=canonical ({len(canonical_splits)} trajectories)"
        )
    raw_auxiliary = build_auxiliary_splits(
        frame_store,
        args.sequence_length,
        args.history_window,
        canonical_splits,
    )
    split_rows = {split: load_csv(args.split_dir / f"{split}_pairs.csv") for split in SPLITS}
    raw_splits = {
        split: build_raw_split(
            split_rows[split],
            frame_store,
            clip_map,
            args.sequence_length,
            args.label_column,
            args.weight_column,
        )
        for split in SPLITS
    }
    selected, feature_names, center, scale, valid_ratios = select_and_fit_features(
        raw_splits["train"],
        args.min_feature_valid_ratio,
    )
    prepared = {
        split: prepare_split(
            raw_splits[split],
            selected,
            center,
            scale,
            args.feature_clip_value,
        )
        for split in SPLITS
    }
    prepared_auxiliary = {
        split: prepare_auxiliary_split(
            raw_auxiliary[split],
            selected,
            center,
            scale,
            args.feature_clip_value,
        )
        for split in SPLITS
    }

    model = PhysicalProgressBranch(
        input_dim=2 * len(feature_names),
        num_tasks=len(TASK_IDS),
        num_stages=NUM_STAGES,
        frame_hidden_dim=args.frame_hidden_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        task_embedding_dim=args.task_embedding_dim,
        num_gru_layers=args.num_gru_layers,
        dropout=args.dropout,
        completion_threshold=args.completion_threshold,
    ).to(device)
    class_weights = compute_stage_class_weights(
        prepared["train"],
        prepared_auxiliary["train"],
        device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        PairSequenceDataset(prepared["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    auxiliary_loader = DataLoader(
        AuxiliarySequenceDataset(prepared_auxiliary["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    print(f"selected_features={len(feature_names)}")
    print(f"model_parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(
        "auxiliary_terminal_windows="
        + ", ".join(
            f"{split}:{len(prepared_auxiliary[split].sample_ids)}"
            for split in SPLITS
        )
    )
    history: list[dict[str, Any]] = []
    best_selection_score = -1.0
    best_val_accuracy = -1.0
    best_val_stage_accuracy = -1.0
    best_val_progress_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = Counter()
        seen = 0
        auxiliary_iterator = iter(auxiliary_loader)
        progress = tqdm(train_loader, desc=f"physical-v1 epoch {epoch:03d}/{args.epochs}", leave=False)
        for raw_batch in progress:
            batch = move_batch(raw_batch, device)
            try:
                raw_auxiliary_batch = next(auxiliary_iterator)
            except StopIteration:
                auxiliary_iterator = iter(auxiliary_loader)
                raw_auxiliary_batch = next(auxiliary_iterator)
            auxiliary_batch = move_batch(raw_auxiliary_batch, device)
            optimizer.zero_grad(set_to_none=True)
            total_loss, components = batch_losses(model, batch, class_weights, args)
            auxiliary_stage_loss, auxiliary_progress_loss = auxiliary_batch_losses(
                model,
                auxiliary_batch,
                class_weights,
            )
            total_loss = (
                total_loss
                + args.aux_stage_loss_weight * auxiliary_stage_loss
                + args.aux_progress_loss_weight * auxiliary_progress_loss
            )
            components["aux_stage"] = auxiliary_stage_loss.detach()
            components["aux_progress"] = auxiliary_progress_loss.detach()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_size = int(batch["label"].numel())
            seen += batch_size
            sums["total"] += float(total_loss.item()) * batch_size
            for name, value in components.items():
                sums[name] += float(value.item()) * batch_size
            progress.set_postfix(
                loss=f"{sums['total'] / max(seen, 1):.4f}",
                pair=f"{sums['pair'] / max(seen, 1):.4f}",
            )

        val_metrics, _ = evaluate(
            model,
            prepared["val"],
            device,
            args.eval_batch_size,
            args.preference_temperature,
        )
        val_auxiliary_metrics = evaluate_auxiliary(
            model,
            prepared_auxiliary["val"],
            device,
            args.eval_batch_size,
        )
        val_accuracy = float(val_metrics["pair_accuracy"])
        val_stage_accuracy = float(
            val_auxiliary_metrics["terminal_stage_accuracy"]
        )
        val_progress_mae = float(
            val_auxiliary_metrics["local_progress_mae"] or float("inf")
        )
        selection_score = (
            0.50 * val_accuracy
            + 0.40 * val_stage_accuracy
            + 0.10 * max(0.0, 1.0 - val_progress_mae)
        )
        better = selection_score > best_selection_score
        if better:
            best_selection_score = selection_score
            best_val_accuracy = val_accuracy
            best_val_stage_accuracy = val_stage_accuracy
            best_val_progress_mae = val_progress_mae
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        epoch_row = {
            "epoch": epoch,
            "train_total_loss": sums["total"] / max(seen, 1),
            "train_pair_loss": sums["pair"] / max(seen, 1),
            "train_stage_loss": sums["stage"] / max(seen, 1),
            "train_progress_loss": sums["progress"] / max(seen, 1),
            "train_monotonic_loss": sums["monotonic"] / max(seen, 1),
            "train_aux_stage_loss": sums["aux_stage"] / max(seen, 1),
            "train_aux_progress_loss": sums["aux_progress"] / max(seen, 1),
            "val_pair_accuracy": val_accuracy,
            "val_aux_terminal_stage_accuracy": val_stage_accuracy,
            "val_aux_stage_frame_accuracy": val_auxiliary_metrics[
                "stage_frame_accuracy"
            ],
            "val_aux_local_progress_mae": val_progress_mae,
            "val_selection_score": selection_score,
            "best_val_pair_accuracy": best_val_accuracy,
            "best_val_selection_score": best_selection_score,
        }
        history.append(epoch_row)
        tqdm.write(
            f"epoch {epoch:03d}: loss={epoch_row['train_total_loss']:.4f} "
            f"val_pair={val_accuracy:.3f} val_stage={val_stage_accuracy:.3f} "
            f"val_progress_mae={val_progress_mae:.3f} "
            f"selection={selection_score:.3f} best={best_selection_score:.3f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint = {
        "format_version": 5,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_config": model.config(),
        "state_dict": model.state_dict(),
        "task_ids": list(TASK_IDS),
        "num_stages": NUM_STAGES,
        "feature_names": feature_names,
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
        "feature_clip_value": args.feature_clip_value,
        "preference_temperature": args.preference_temperature,
        "completion_threshold": args.completion_threshold,
        "sequence_length": args.sequence_length,
        "history_window": args.history_window,
        "label_column": args.label_column,
        "contact_augmentation": contact_augmentation,
        "auxiliary_split_source": (
            "canonical_source_group_v1" if canonical_splits is not None else "legacy_frame_split"
        ),
        "split_map_dir": str(args.split_map_dir) if args.split_map_dir else None,
    }
    torch.save(checkpoint, args.out_dir / "best_model.pt")
    write_csv(args.out_dir / "train_history.csv", history)

    all_metrics: dict[str, Any] = {}
    for split in SPLITS:
        metrics, predictions = evaluate(
            model,
            prepared[split],
            device,
            args.eval_batch_size,
            args.preference_temperature,
        )
        metrics["auxiliary_trajectory"] = evaluate_auxiliary(
            model,
            prepared_auxiliary[split],
            device,
            args.eval_batch_size,
        )
        all_metrics[split] = metrics
        write_csv(args.out_dir / f"{split}_predictions.csv", predictions)

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "dataset_root": str(args.dataset_root),
        "split_dir": str(args.split_dir),
        "split_map_dir": str(args.split_map_dir) if args.split_map_dir else None,
        "auxiliary_split_source": (
            "canonical_source_group_v1" if canonical_splits is not None else "legacy_frame_split"
        ),
        "out_dir": str(args.out_dir),
        "model_config": model.config(),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "feature_names": feature_names,
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
        "feature_valid_ratios": valid_ratios,
        "stage_class_weights": class_weights.detach().cpu().tolist(),
        "contact_augmentation": contact_augmentation,
        "split_rows": {split: len(prepared[split].labels) for split in SPLITS},
        "auxiliary_terminal_windows": {
            split: len(prepared_auxiliary[split].sample_ids) for split in SPLITS
        },
        "label_distribution": {
            split: dict(Counter(row[args.label_column] for row in raw_splits[split].rows))
            for split in SPLITS
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "stage_loss_weight": args.stage_loss_weight,
            "progress_loss_weight": args.progress_loss_weight,
            "monotonic_loss_weight": args.monotonic_loss_weight,
            "aux_stage_loss_weight": args.aux_stage_loss_weight,
            "aux_progress_loss_weight": args.aux_progress_loss_weight,
            "preference_temperature": args.preference_temperature,
            "completion_threshold": args.completion_threshold,
            "sequence_length": args.sequence_length,
            "history_window": args.history_window,
            "checkpoint_selection": (
                "0.50*pair_accuracy + 0.40*aux_terminal_stage_accuracy "
                "+ 0.10*(1-aux_local_progress_mae)"
            ),
            "seed": args.seed,
        },
    }
    write_json(args.out_dir / "run_config.json", run_config)
    write_json(args.out_dir / "metrics.json", all_metrics)

    summary_lines = [
        "# Physical Progress Branch v1",
        "",
        f"- checkpoint: `{args.out_dir / 'best_model.pt'}`",
        f"- parameters: {run_config['model_parameters']:,}",
        f"- selected physical features: {len(feature_names)}",
        f"- label column: `{args.label_column}`",
        (
            "- adversarial near-miss contact augmentation: "
            f"{contact_augmentation['augmented_trajectories']} trajectories / "
            f"{contact_augmentation['augmented_frames']} frames"
        ),
        "",
        "| split | pairs | pair accuracy | terminal stage accuracy | progress MAE | hard-negative accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        metrics = all_metrics[split]
        auxiliary_metrics = metrics["auxiliary_trajectory"]
        summary_lines.append(
            f"| {split} | {metrics['rows']} | {metrics['pair_accuracy']:.3f} | "
            f"{auxiliary_metrics['terminal_stage_accuracy']:.3f} | "
            f"{auxiliary_metrics['local_progress_mae']:.3f} | "
            f"{metrics.get('hard_negative_accuracy', 0.0):.3f} |"
        )
    summary_lines += [
        "",
        "## Input Boundary",
        "",
        "The model does not consume simulator success, task-rule scores, time progress,",
        "candidate/fusion labels, or stage IDs. Contact-derived stage IDs are training-only",
        "auxiliary targets. The current stereo geometry tables are env-state projections and",
        "must be replaced by real stereo estimates before claiming observation-only results.",
        "",
        "## Selected Features",
        "",
        *[f"- `{name}`" for name in feature_names],
    ]
    (args.out_dir / "RESULTS.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print((args.out_dir / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
