"""Export learned physical-branch scores for every A/B pair (Phase 2).

For each pair row in the split tables (or a full pre-filter pair table), this
script reconstructs both clips' observable depth/contact frame histories from
the existing stereo/contact frame tables, runs the frozen
``PhysicalProgressRuntime``, and writes per-side stage distributions, local
progress, potential, confidence, sensor-validity ratios, and pair-level
preference probability / margin / confidence / unsure state.

Guarantees enforced in code (not by convention):

- Teacher/rule label columns (``candidate_label``, ``mimo_*``, ``final_*``,
  ``*_proxy``) are never read while computing physical outputs; they are only
  copied through as metadata. ``tests/test_physical_pair_export_v1.py``
  poisons them and asserts score invariance.
- Every feature name consumed from the frame tables passes the deny list.
- Clip windows must satisfy the runtime history contract
  (window <= history_window, runtime 16->6 resampling reproduces the pair
  table's sampled indices); violations abort with counts instead of silently
  proceeding (audit F8).
- Nearest-frame substitution is bounded and counted (audit F9).
- Splits/groups come from the pair tables only, never the legacy per-frame
  ``split`` column (audit F2).

Requires torch. On machines without torch this script exits with a precise
dependency report and status code 3 (nothing is fabricated).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reward_common_v1 import (  # noqa: E402
    SPLITS,
    TASK_IDS,
    assert_features_allowed,
    as_float,
    base_success_id,
    clip_trajectory_id,
    default_dataset_root,
    default_split_dir,
    linspace_resample,
    load_csv,
    pair_group_id,
    parse_indices,
    parse_numeric,
    sha256_file,
    write_csv,
    write_json,
)

EXPORT_VERSION = "physical_pair_scores_v1.0"

# Observable feature families duplicated from 24_train (kept in sync by
# tests/test_feature_deny_list_v1.py which parses both sources).
DEPTH_FEATURES = (
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
CONTACT_FEATURES = (
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
NEAR_MISS_CONTACT_AUGMENT_FEATURES = tuple(
    name
    for name in CONTACT_FEATURES
    if name not in ("object_speed_proxy_m_per_step", "object_static_proxy")
)

# Columns copied through untouched. The compute path never reads them.
PASSTHROUGH_COLUMNS = (
    "pair_id",
    "task_id",
    "pair_type",
    "source_group_id",
    "clip_a_id",
    "clip_b_id",
)

NUM_STAGES = 4


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def clip_uid(clip_id: str, video_path: str, indices: list[int]) -> str:
    return short_hash(f"{clip_id}|{video_path}|{';'.join(map(str, indices))}")


def is_clean_success_row(row: dict[str, str]) -> bool:
    sample_id = row.get("sample_id", "").upper()
    return (
        "success" in row.get("source_type", "").lower()
        and not row.get("near_miss_type", "").strip()
        and "OFFSET" not in sample_id
        and "TRUNC" not in sample_id
    )


def load_frame_tables(
    dataset_root: Path,
    augment_near_miss_contact: bool,
) -> tuple[dict[tuple[str, str, int], dict[str, float]], dict[str, dict[str, str]], dict[str, Any]]:
    """(task, sample, frame) -> observable feature dict; clip_id -> clip meta.

    Mirrors 24_train's ``load_physical_frame_store`` including (optionally)
    the adversarial success-like contact augmentation, but never stores the
    legacy split column or any stage/label field in the feature dict.
    """
    assert_features_allowed(DEPTH_FEATURES + CONTACT_FEATURES, "26_export frame features")
    store: dict[tuple[str, str, int], dict[str, float]] = {}
    clip_map: dict[str, dict[str, str]] = {}
    augmented_frames = 0
    augmented_trajectories: set[str] = set()

    for task_id in TASK_IDS:
        folder = f"{task_id}_bootstrap_v1"
        stereo_rows = load_csv(dataset_root / "stereo_features" / folder / "frame_stereo_geometry_features.csv")
        contact_rows = load_csv(dataset_root / "contact_stage_features" / folder / "frame_contact_stage_features.csv")
        contact_by_key = {(r["sample_id"], int(float(r["frame_idx"]))): r for r in contact_rows}
        contact_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in contact_rows:
            contact_by_sample[row["sample_id"]].append(row)
        clean_success_terminal: dict[str, dict[str, str]] = {}
        for sample_id, rows in contact_by_sample.items():
            if not rows:
                continue
            terminal = max(rows, key=lambda r: int(float(r["frame_idx"])))
            if is_clean_success_row(terminal):
                clean_success_terminal[sample_id] = terminal
        for stereo_row in stereo_rows:
            sample_id = stereo_row["sample_id"]
            frame_idx = int(float(stereo_row["frame_idx"]))
            contact_row = contact_by_key.get((sample_id, frame_idx), {})
            merged: dict[str, str] = {**stereo_row, **contact_row}
            if (
                augment_near_miss_contact
                and contact_row.get("source_type", "") == "perturbed_success_final_state"
                and parse_numeric(contact_row.get("contact_observation_valid", "")) == 0.0
            ):
                reference = clean_success_terminal.get(base_success_id(sample_id))
                if reference is not None:
                    copied = False
                    for name in NEAR_MISS_CONTACT_AUGMENT_FEATURES:
                        value = parse_numeric(reference.get(name, ""))
                        if math.isfinite(value):
                            merged[name] = str(value)
                            copied = True
                    if copied:
                        augmented_frames += 1
                        augmented_trajectories.add(sample_id)
            frame: dict[str, float] = {}
            for name in DEPTH_FEATURES + CONTACT_FEATURES:
                value = parse_numeric(merged.get(name, ""))
                if math.isfinite(value):
                    frame[name] = value
            store[(task_id, sample_id, frame_idx)] = frame

        for row in load_csv(dataset_root / "contact_stage_features" / folder / "clip_contact_stage_features.csv"):
            clip_map[row["clip_id"]] = row

    summary = {
        "augment_near_miss_contact": augment_near_miss_contact,
        "augmented_frames": augmented_frames,
        "augmented_trajectories": len(augmented_trajectories),
    }
    return store, clip_map, summary


class FrameLookup:
    """Bounded nearest-frame lookup with substitution accounting (audit F9)."""

    def __init__(self, store: dict[tuple[str, str, int], dict[str, float]], max_gap: int) -> None:
        self.max_gap = max_gap
        self.by_traj: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.store = store
        for (task_id, sample_id, frame_idx) in store:
            self.by_traj[(task_id, sample_id)].append(frame_idx)
        for key in self.by_traj:
            self.by_traj[key].sort()
        self.substitutions = 0
        self.max_observed_gap = 0

    def get(self, task_id: str, trajectory_id: str, frame_idx: int) -> dict[str, float]:
        exact = self.store.get((task_id, trajectory_id, frame_idx))
        if exact is not None:
            return exact
        frames = self.by_traj.get((task_id, trajectory_id))
        if not frames:
            raise KeyError(f"No physical frames for {task_id}/{trajectory_id}")
        nearest = min(frames, key=lambda idx: abs(idx - frame_idx))
        gap = abs(nearest - frame_idx)
        if gap > self.max_gap:
            raise KeyError(
                f"nearest frame gap {gap} > max_gap {self.max_gap} for "
                f"{task_id}/{trajectory_id}@{frame_idx}"
            )
        self.substitutions += 1
        self.max_observed_gap = max(self.max_observed_gap, gap)
        return self.store[(task_id, trajectory_id, nearest)]


def clip_window_frames(
    row: dict[str, str],
    side: str,
    clip_map: dict[str, dict[str, str]],
    lookup: FrameLookup,
    history_window: int,
    sequence_length: int,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    clip_id = row[f"clip_{side}_id"]
    clip_info = clip_map.get(clip_id)
    if clip_info is None:
        raise KeyError(f"Missing clip metadata for {clip_id}")
    trajectory_id = clip_info["trajectory_id"]
    start = int(float(row[f"clip_{side}_start_frame"]))
    end = int(float(row[f"clip_{side}_end_frame_exclusive"]))
    window = list(range(start, end))
    if not window:
        raise ValueError(f"empty clip window for {clip_id}")
    if len(window) > history_window:
        raise ValueError(
            f"clip window {len(window)} frames exceeds runtime history_window "
            f"{history_window} for {clip_id}; refusing silent truncation (audit F8)"
        )
    sampled = parse_indices(row[f"clip_{side}_sample_frame_indices"])
    runtime_sampled = linspace_resample(window, sequence_length)
    consistent = sampled == runtime_sampled if sampled else True
    if not consistent:
        raise ValueError(
            f"pair-table samples {sampled} do not match runtime resampling "
            f"{runtime_sampled} for {clip_id}"
        )
    frames = [lookup.get(row["task_id"], trajectory_id, idx) for idx in window]
    info = {
        "clip_id": clip_id,
        "trajectory_id": trajectory_id,
        "window_start": start,
        "window_end_exclusive": end,
        "window_frames": len(window),
        "pair_table_indices": sampled,
        "runtime_resample_indices": runtime_sampled,
        "sampling_consistent_with_runtime": consistent,
    }
    return frames, info


def validity_ratios(frames: list[dict[str, float]]) -> dict[str, float]:
    def ratio(names: tuple[str, ...]) -> float:
        total = len(frames) * len(names)
        if total == 0:
            return 0.0
        present = sum(1 for frame in frames for name in names if name in frame)
        return present / total

    return {
        "depth_validity_ratio": ratio(DEPTH_FEATURES),
        "contact_validity_ratio": ratio(CONTACT_FEATURES),
    }


def load_pair_rows(pairs_arg: Path | None, split_dir: Path) -> list[dict[str, str]]:
    if pairs_arg is not None:
        rows = load_csv(pairs_arg)
        for row in rows:
            row.setdefault("split_v1", row.get("split_v1", row.get("split", "")))
        return rows
    rows = []
    for split in SPLITS:
        for row in load_csv(split_dir / f"{split}_pairs.csv"):
            row["split_v1"] = split
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--split-dir", type=Path, default=default_split_dir())
    parser.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="Optional full (pre-filter) pair table; overrides --split-dir row source.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Physical branch checkpoint; default <dataset-root>/reward_model_runs/physical_progress_branch_v1/best_model.pt",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--frame-source",
        choices=("clip_window", "pair_indices"),
        default="clip_window",
        help="clip_window feeds the full 16-frame window through the runtime 16->6 path; "
        "pair_indices feeds exactly the pair table's 6 sampled frames.",
    )
    parser.add_argument(
        "--near-miss-contact-augment",
        choices=("train_like", "off"),
        default="train_like",
        help="train_like mirrors 24_train's adversarial success-like contact copy; "
        "off leaves frozen near-miss contact missing (validity 0).",
    )
    parser.add_argument("--max-frame-gap", type=int, default=3)
    parser.add_argument(
        "--unsure-epsilon",
        type=float,
        default=0.02,
        help="|P(A>B)-0.5| below this exports physical_preference=unsure. "
        "Raw probabilities are exported too; calibrated thresholds are fit later in 27.",
    )
    parser.add_argument("--export-embeddings", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import numpy as np
        import torch  # noqa: F401
    except ImportError as exc:
        print(
            "MISSING DEPENDENCY: this exporter requires torch (and numpy). "
            f"Import failed with: {exc}. Install torch in the Windows PyTorch "
            "environment described in FABLE5_OPERATION_GUIDE.md section 6 and rerun. "
            "No output was fabricated.",
            file=sys.stderr,
        )
        return 3

    from physical_progress_branch_v1 import PhysicalProgressRuntime

    checkpoint_path = args.checkpoint or (
        args.dataset_root / "reward_model_runs" / "physical_progress_branch_v1" / "best_model.pt"
    )
    out_dir = args.out_dir or (
        args.dataset_root / "physical_pair_scores" / "physical_pair_scores_v1"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = PhysicalProgressRuntime.from_checkpoint(checkpoint_path, device=args.device)
    assert_features_allowed(runtime.feature_names, "checkpoint feature_names")

    store, clip_map, augment_summary = load_frame_tables(
        args.dataset_root,
        augment_near_miss_contact=(args.near_miss_contact_augment == "train_like"),
    )
    lookup = FrameLookup(store, max_gap=args.max_frame_gap)
    pair_rows = load_pair_rows(args.pairs, args.split_dir)
    print(f"pairs={len(pair_rows)} checkpoint={checkpoint_path}")

    output_rows: list[dict[str, Any]] = []
    embeddings: dict[str, Any] = {}
    inconsistent_sampling = 0
    score_cache: dict[str, dict[str, Any]] = {}

    for row in pair_rows:
        record: dict[str, Any] = {key: row.get(key, "") for key in PASSTHROUGH_COLUMNS}
        record["source_group_id"] = row.get("source_group_id") or pair_group_id(row)
        record["split_v1"] = row.get("split_v1", "")
        side_results: dict[str, dict[str, Any]] = {}
        for side in ("a", "b"):
            frames, info = clip_window_frames(
                row, side, clip_map, lookup, runtime.history_window, runtime.sequence_length
            )
            if not info["sampling_consistent_with_runtime"]:
                inconsistent_sampling += 1
            if args.frame_source == "pair_indices" and info["pair_table_indices"]:
                frames = [
                    lookup.get(row["task_id"], info["trajectory_id"], idx)
                    for idx in info["pair_table_indices"]
                ]
            cache_key = (
                f"{row['task_id']}|{info['clip_id']}|{info['window_start']}|"
                f"{info['window_end_exclusive']}|{args.frame_source}"
            )
            if cache_key not in score_cache:
                score_cache[cache_key] = runtime.score(
                    row["task_id"], frames, return_embedding=args.export_embeddings
                )
            result = score_cache[cache_key]
            side_results[side] = result
            ratios = validity_ratios(frames)
            record[f"clip_{side}_trajectory_id"] = info["trajectory_id"]
            record[f"clip_{side}_base_success_id"] = base_success_id(info["trajectory_id"])
            record[f"clip_{side}_window_frames"] = info["window_frames"]
            record[f"clip_{side}_sampling_consistent"] = info["sampling_consistent_with_runtime"]
            record[f"phys_{side}_stage"] = result["stage"]
            for stage_index in range(NUM_STAGES):
                record[f"phys_{side}_stage_p{stage_index + 1}"] = round(
                    float(result["stage_probabilities"][stage_index]), 6
                )
            record[f"phys_{side}_local_progress"] = round(float(result["local_progress"]), 6)
            record[f"phys_{side}_potential"] = round(float(result["potential"]), 6)
            record[f"phys_{side}_confidence"] = round(float(result["confidence"]), 6)
            record[f"phys_{side}_depth_validity_ratio"] = round(ratios["depth_validity_ratio"], 6)
            record[f"phys_{side}_contact_validity_ratio"] = round(ratios["contact_validity_ratio"], 6)
            if args.export_embeddings and "embedding" in result:
                uid = clip_uid(
                    row[f"clip_{side}_id"],
                    row.get(f"clip_{side}_video_path_local", row.get(f"clip_{side}_video_path_windows", "")),
                    parse_indices(row.get(f"clip_{side}_sample_frame_indices", "")),
                )
                embeddings[uid] = np.asarray(result["embedding"], dtype=np.float32)
                record[f"clip_{side}_embedding_uid"] = uid

        potential_a = float(side_results["a"]["potential"])
        potential_b = float(side_results["b"]["potential"])
        probability = 1.0 / (
            1.0 + math.exp(-(potential_a - potential_b) / runtime.preference_temperature)
        )
        margin = potential_a - potential_b
        confidence = (
            2.0
            * abs(probability - 0.5)
            * math.sqrt(
                max(float(side_results["a"]["confidence"]) * float(side_results["b"]["confidence"]), 0.0)
            )
        )
        if abs(probability - 0.5) < args.unsure_epsilon:
            preference = "unsure"
        elif probability > 0.5:
            preference = "A>B"
        else:
            preference = "B>A"
        record["phys_probability_a_better"] = round(probability, 6)
        record["phys_margin_a_minus_b"] = round(margin, 6)
        record["phys_pair_confidence"] = round(confidence, 6)
        record["phys_preference"] = preference
        output_rows.append(record)

    scores_path = out_dir / "physical_pair_scores_v1.csv"
    write_csv(scores_path, output_rows)
    if args.export_embeddings:
        np.savez(
            out_dir / "physical_clip_embeddings_v1.npz",
            **{uid: vector for uid, vector in embeddings.items()},
        )

    manifest = {
        "export_version": EXPORT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_history_window": runtime.history_window,
        "checkpoint_sequence_length": runtime.sequence_length,
        "preference_temperature": runtime.preference_temperature,
        "frame_source": args.frame_source,
        "near_miss_contact_augment": augment_summary,
        "unsure_epsilon": args.unsure_epsilon,
        "pairs_input": str(args.pairs) if args.pairs else str(args.split_dir),
        "rows": len(output_rows),
        "unique_clips_scored": len(score_cache),
        "nearest_frame_substitutions": lookup.substitutions,
        "max_nearest_frame_gap": lookup.max_observed_gap,
        "sampling_inconsistent_clips": inconsistent_sampling,
        "embeddings_exported": len(embeddings),
        "label_columns_consumed_for_scoring": [],
        "note": (
            "Physical outputs are computed only from observable depth/contact "
            "frame features; teacher/candidate/fusion label columns are "
            "pass-through metadata and were not read for scoring."
        ),
    }
    write_json(out_dir / "physical_pair_scores_v1_manifest.json", manifest)
    print(f"wrote {scores_path} rows={len(output_rows)}")
    print(
        f"substitutions={lookup.substitutions} max_gap={lookup.max_observed_gap} "
        f"sampling_inconsistent={inconsistent_sampling}"
    )
    if inconsistent_sampling:
        print(
            "WARNING: some clip windows resample differently from the pair table "
            "indices; inspect the manifest before building fusion labels.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
