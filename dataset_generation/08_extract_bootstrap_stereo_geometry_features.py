from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_OUT_ROOT = Path(r"E:\reward_model_dataset\stereo_features")

TASK_DISTANCE_SCALE = {
    "stackcube": 0.24,
    "stackpyramid": 0.28,
    "peginsertion": 0.18,
}

PEG_HEAD_LOCAL_OFFSET = np.array([0.1024, 0.0, 0.0], dtype=np.float64)
PEG_HOLE_LOCAL_OFFSET = np.array([0.0, 0.0043, -0.0056], dtype=np.float64)


@dataclass(frozen=True)
class StereoRig:
    center_eye: np.ndarray
    look_at: np.ndarray
    up: np.ndarray
    baseline_m: float
    focal_px: float


def make_stereo_rig() -> StereoRig:
    return StereoRig(
        center_eye=np.array([0.55, -0.65, 0.42], dtype=np.float64),
        look_at=np.array([0.0, 0.0, 0.08], dtype=np.float64),
        up=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        baseline_m=0.08,
        focal_px=520.0,
    )


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def wsl_to_windows(path: str) -> Path:
    path = (path or "").strip()
    if path.startswith("/mnt/") and len(path) > 6:
        drive = path[5].upper()
        rest = path[7:].replace("/", "\\")
        return Path(f"{drive}:\\{rest}")
    return Path(path)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(-1)[:4]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 1e-8:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_axes(rig: StereoRig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = rig.look_at - rig.center_eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, rig.up)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(x_axis, z_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return x_axis, y_axis, z_axis


def camera_centers(rig: StereoRig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis, _, _ = camera_axes(rig)
    left = rig.center_eye - x_axis * (rig.baseline_m / 2.0)
    right = rig.center_eye + x_axis * (rig.baseline_m / 2.0)
    return left, right, rig.center_eye


def world_to_cam(point: np.ndarray, eye: np.ndarray, rig: StereoRig) -> np.ndarray:
    x_axis, y_axis, z_axis = camera_axes(rig)
    delta = np.asarray(point, dtype=np.float64) - eye
    return np.array(
        [
            float(np.dot(delta, x_axis)),
            float(np.dot(delta, y_axis)),
            float(np.dot(delta, z_axis)),
        ],
        dtype=np.float64,
    )


def pseudo_disparity(depth_m: float, rig: StereoRig) -> float:
    if not math.isfinite(depth_m) or depth_m <= 1e-6:
        return float("nan")
    return float(rig.focal_px * rig.baseline_m / depth_m)


def resolve_h5_path(traj: dict[str, str]) -> Path:
    for key in ("h5_path_windows", "h5_path", "source_h5_path_windows", "source_h5_path"):
        value = (traj.get(key) or "").strip()
        if value:
            return wsl_to_windows(value)
    raise ValueError(f"No h5/source_h5 path for {traj.get('sample_id')}")


def read_actor_arrays(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        group = f["traj_0/env_states/actors"]
        return {name: np.asarray(group[name], dtype=np.float64) for name in group.keys()}


def trim_arrays_for_source(traj: dict[str, str], arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if traj.get("source_type") != "truncated_success_trajectory":
        return arrays
    stop_step = safe_int(traj.get("stop_step"), max(len(next(iter(arrays.values()))) - 1, 0))
    keep = max(1, min(len(next(iter(arrays.values()))), stop_step + 1))
    return {name: arr[:keep] for name, arr in arrays.items()}


def repeat_positions(value: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(np.asarray(value, dtype=np.float64).reshape(1, 3), max(1, n), axis=0)


def positions_from_manifest(traj: dict[str, str], task_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    n = safe_int(traj.get("num_frames"), 1)
    if task_id == "stackcube":
        obj = np.array([safe_float(traj.get("cubeA_x")), safe_float(traj.get("cubeA_y")), safe_float(traj.get("cubeA_z"))])
        goal = np.array([safe_float(traj.get("goal_x")), safe_float(traj.get("goal_y")), safe_float(traj.get("goal_z"))])
        support = goal - np.array([0.0, 0.0, 0.04], dtype=np.float64)
        return repeat_positions(obj, n), repeat_positions(goal, n), repeat_positions(support, n), "manifest_exact_stackcube_snapshot"

    if task_id == "stackpyramid":
        obj = np.array([safe_float(traj.get("cubeC_x")), safe_float(traj.get("cubeC_y")), safe_float(traj.get("cubeC_z"))])
        goal = np.array([safe_float(traj.get("goal_x")), safe_float(traj.get("goal_y")), safe_float(traj.get("goal_z"))])
        support = goal - np.array([0.0, 0.0, 0.04], dtype=np.float64)
        return repeat_positions(obj, n), repeat_positions(goal, n), repeat_positions(support, n), "manifest_exact_stackpyramid_snapshot"

    if task_id == "peginsertion":
        obj = np.array([safe_float(traj.get("peg_head_x")), safe_float(traj.get("peg_head_y")), safe_float(traj.get("peg_head_z"))])
        goal = np.array([safe_float(traj.get("hole_x")), safe_float(traj.get("hole_y")), safe_float(traj.get("hole_z"))])
        support = goal.copy()
        return repeat_positions(obj, n), repeat_positions(goal, n), repeat_positions(support, n), "manifest_exact_peginsertion_snapshot"

    raise ValueError(task_id)


def positions_from_h5(traj: dict[str, str], task_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if traj.get("source_type") == "perturbed_success_final_state":
        return positions_from_manifest(traj, task_id)

    arrays = trim_arrays_for_source(traj, read_actor_arrays(resolve_h5_path(traj)))
    if task_id == "stackcube":
        cube_a = arrays["cubeA"][:, :3]
        cube_b = arrays["cubeB"][:, :3]
        half = float(np.median(np.minimum(cube_a[: min(8, len(cube_a)), 2], cube_b[: min(8, len(cube_b)), 2])))
        if not math.isfinite(half) or half <= 0:
            half = 0.02
        goal = cube_b + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64)
        return cube_a, goal, cube_b, "h5_env_state_stackcube"

    if task_id == "stackpyramid":
        cube_a = arrays["cubeA"][:, :3]
        cube_b = arrays["cubeB"][:, :3]
        cube_c = arrays["cubeC"][:, :3]
        half = float(np.median(np.minimum(cube_a[: min(8, len(cube_a)), 2], cube_b[: min(8, len(cube_b)), 2])))
        if not math.isfinite(half) or half <= 0:
            half = 0.02
        support = 0.5 * (cube_a + cube_b)
        goal = support + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64)
        return cube_c, goal, support, "h5_env_state_stackpyramid"

    if task_id == "peginsertion":
        peg = arrays["peg"]
        box = arrays["box_with_hole"]
        peg_p = peg[:, :3]
        peg_q = peg[:, 3:7]
        box_p = box[:, :3]
        box_q = box[:, 3:7]
        peg_head = np.zeros_like(peg_p)
        hole = np.zeros_like(box_p)
        for i in range(len(peg_p)):
            peg_head[i] = peg_p[i] + quat_to_matrix(peg_q[i]) @ PEG_HEAD_LOCAL_OFFSET
            hole[i] = box_p[i] + quat_to_matrix(box_q[i]) @ PEG_HOLE_LOCAL_OFFSET
        return peg_head, hole, hole, "h5_env_state_peginsertion_with_constant_hole_offsets"

    raise ValueError(task_id)


def frame_to_state_idx(frame_idx: int, video_frames: int, state_frames: int) -> int:
    if state_frames <= 1 or video_frames <= 1:
        return max(0, min(state_frames - 1, frame_idx))
    ratio = max(0.0, min(1.0, frame_idx / float(video_frames - 1)))
    return int(round(ratio * (state_frames - 1)))


def stage_proxy(task_id: str, moved_m: float, dist_m: float, xy_error_m: float, height_error_m: float) -> int:
    if task_id in {"stackcube", "stackpyramid"}:
        if xy_error_m <= 0.030 and height_error_m <= 0.018:
            return 3
        if xy_error_m <= 0.060 and height_error_m <= 0.045:
            return 2
        if moved_m >= 0.025:
            return 1
        return 0
    if task_id == "peginsertion":
        if dist_m <= 0.018:
            return 3
        if dist_m <= 0.045:
            return 2
        if moved_m >= 0.030:
            return 1
        return 0
    raise ValueError(task_id)


def progress_from_distance(task_id: str, dist_m: float) -> float:
    scale = TASK_DISTANCE_SCALE.get(task_id, 0.25)
    return float(np.clip(1.0 - dist_m / scale, 0.0, 1.0))


def build_frame_rows(traj: dict[str, str], task_id: str, rig: StereoRig) -> list[dict[str, Any]]:
    obj_states, goal_states, support_states, source = positions_from_h5(traj, task_id)
    state_frames = len(obj_states)
    video_frames = safe_int(traj.get("num_frames")) or state_frames
    start_obj = obj_states[0]
    left_eye, right_eye, center_eye = camera_centers(rig)
    rows: list[dict[str, Any]] = []
    prev_dist = float("nan")
    for frame_idx in range(video_frames):
        state_idx = frame_to_state_idx(frame_idx, video_frames, state_frames)
        obj = obj_states[state_idx]
        goal = goal_states[state_idx]
        support = support_states[state_idx]
        delta = obj - goal
        dist = float(np.linalg.norm(delta))
        xy_error = float(np.linalg.norm(delta[:2]))
        height_error = abs(float(delta[2]))
        moved = float(np.linalg.norm(obj - start_obj))
        stage = stage_proxy(task_id, moved, dist, xy_error, height_error)
        progress = progress_from_distance(task_id, dist)
        physical_score = stage + progress

        obj_cam = world_to_cam(obj, center_eye, rig)
        goal_cam = world_to_cam(goal, center_eye, rig)
        support_cam = world_to_cam(support, center_eye, rig)
        obj_left = world_to_cam(obj, left_eye, rig)
        obj_right = world_to_cam(obj, right_eye, rig)
        goal_left = world_to_cam(goal, left_eye, rig)
        obj_goal_cam = obj_cam - goal_cam

        rows.append(
            {
                "sample_id": traj["sample_id"],
                "task_id": task_id,
                "split": traj.get("split", "train"),
                "source_type": traj.get("source_type", ""),
                "near_miss_type": traj.get("near_miss_type", ""),
                "frame_idx": frame_idx,
                "state_idx": state_idx,
                "video_frames": video_frames,
                "state_frames": state_frames,
                "time_progress": frame_idx / max(1, video_frames - 1),
                "object_x": float(obj[0]),
                "object_y": float(obj[1]),
                "object_z": float(obj[2]),
                "goal_x": float(goal[0]),
                "goal_y": float(goal[1]),
                "goal_z": float(goal[2]),
                "support_x": float(support[0]),
                "support_y": float(support[1]),
                "support_z": float(support[2]),
                "object_goal_3d_dist_m": dist,
                "object_goal_xy_error_m": xy_error,
                "object_goal_height_error_m": height_error,
                "delta_object_goal_dist_m": 0.0 if not math.isfinite(prev_dist) else prev_dist - dist,
                "object_moved_from_start_m": moved,
                "object_cam_x": float(obj_cam[0]),
                "object_cam_y": float(obj_cam[1]),
                "object_cam_depth_m": float(obj_cam[2]),
                "goal_cam_x": float(goal_cam[0]),
                "goal_cam_y": float(goal_cam[1]),
                "goal_cam_depth_m": float(goal_cam[2]),
                "support_cam_depth_m": float(support_cam[2]),
                "object_goal_cam_lateral_error_m": float(np.linalg.norm(obj_goal_cam[:2])),
                "object_goal_cam_depth_error_m": abs(float(obj_goal_cam[2])),
                "left_object_depth_m": float(obj_left[2]),
                "right_object_depth_m": float(obj_right[2]),
                "left_goal_depth_m": float(goal_left[2]),
                "object_pseudo_disparity_px": pseudo_disparity(float(obj_cam[2]), rig),
                "goal_pseudo_disparity_px": pseudo_disparity(float(goal_cam[2]), rig),
                "stage_proxy": stage,
                "geometry_progress_proxy": progress,
                "physical_score_proxy": physical_score,
                "feature_source": source,
            }
        )
        prev_dist = dist
    return rows


def aggregate_clip(clip: dict[str, str], frame_rows_by_sample: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    sample_id = clip["trajectory_id"]
    start = safe_int(clip["start_frame"])
    end = safe_int(clip["end_frame_exclusive"])
    source_rows = frame_rows_by_sample.get(sample_id, [])
    rows = source_rows[start:end]
    if not rows:
        return {"clip_id": clip["clip_id"], "has_stereo_geometry_features": False}

    end_row = rows[-1]
    distances = np.array([safe_float(r["object_goal_3d_dist_m"]) for r in rows], dtype=np.float64)
    stages = np.array([safe_int(r["stage_proxy"]) for r in rows], dtype=np.int64)
    scores = np.array([safe_float(r["physical_score_proxy"]) for r in rows], dtype=np.float64)
    depth_errors = np.array([safe_float(r["object_goal_cam_depth_error_m"]) for r in rows], dtype=np.float64)
    lateral_errors = np.array([safe_float(r["object_goal_cam_lateral_error_m"]) for r in rows], dtype=np.float64)

    return {
        "clip_id": clip["clip_id"],
        "trajectory_id": sample_id,
        "task_id": clip["task_id"],
        "split": clip["split"],
        "source_type": clip["source_type"],
        "near_miss_type": clip.get("near_miss_type", ""),
        "start_frame": start,
        "end_frame_exclusive": end,
        "num_frames": len(rows),
        "has_stereo_geometry_features": True,
        "end_stage_proxy": int(end_row["stage_proxy"]),
        "max_stage_proxy": int(np.max(stages)),
        "end_object_goal_3d_dist_m": float(end_row["object_goal_3d_dist_m"]),
        "mean_object_goal_3d_dist_m": float(np.mean(distances)),
        "min_object_goal_3d_dist_m": float(np.min(distances)),
        "end_object_goal_cam_depth_error_m": float(end_row["object_goal_cam_depth_error_m"]),
        "mean_object_goal_cam_depth_error_m": float(np.mean(depth_errors)),
        "end_object_goal_cam_lateral_error_m": float(end_row["object_goal_cam_lateral_error_m"]),
        "mean_object_goal_cam_lateral_error_m": float(np.mean(lateral_errors)),
        "end_object_pseudo_disparity_px": float(end_row["object_pseudo_disparity_px"]),
        "end_goal_pseudo_disparity_px": float(end_row["goal_pseudo_disparity_px"]),
        "end_geometry_progress_proxy": float(end_row["geometry_progress_proxy"]),
        "mean_geometry_progress_proxy": float(np.mean([safe_float(r["geometry_progress_proxy"]) for r in rows])),
        "end_physical_score_proxy": float(end_row["physical_score_proxy"]),
        "mean_physical_score_proxy": float(np.mean(scores)),
        "delta_object_goal_dist_m": float(distances[0] - distances[-1]) if len(distances) else 0.0,
        "feature_source": "fixed_stereo_projection_geometry",
    }


def build_pair_rows(pairs: list[dict[str, str]], clip_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_clip = {row["clip_id"]: row for row in clip_rows if row.get("has_stereo_geometry_features")}
    out: list[dict[str, Any]] = []
    for pair in pairs:
        a = by_clip[pair["clip_a_id"]]
        b = by_clip[pair["clip_b_id"]]
        score_a = safe_float(a["end_physical_score_proxy"])
        score_b = safe_float(b["end_physical_score_proxy"])
        diff = score_a - score_b
        if diff > 0.10:
            stereo_label = "A>B"
        elif diff < -0.10:
            stereo_label = "B>A"
        else:
            stereo_label = "unsure"
        out.append(
            {
                **pair,
                "stereo_geometry_label_proxy": stereo_label,
                "stereo_geometry_label_agrees_with_pair_label": stereo_label == pair["label"],
                "clip_a_end_stage_proxy": a["end_stage_proxy"],
                "clip_b_end_stage_proxy": b["end_stage_proxy"],
                "clip_a_end_score_proxy": score_a,
                "clip_b_end_score_proxy": score_b,
                "score_diff_a_minus_b": diff,
                "clip_a_end_dist_m": a["end_object_goal_3d_dist_m"],
                "clip_b_end_dist_m": b["end_object_goal_3d_dist_m"],
                "clip_a_end_depth_error_m": a["end_object_goal_cam_depth_error_m"],
                "clip_b_end_depth_error_m": b["end_object_goal_cam_depth_error_m"],
                "feature_source": "fixed_stereo_projection_geometry",
            }
        )
    return out


def summarize(
    task_id: str,
    trajectories: list[dict[str, str]],
    frame_rows: list[dict[str, Any]],
    clip_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    stereo_label_counts: dict[str, int] = {}
    for row in trajectories:
        source_counts[row["source_type"]] = source_counts.get(row["source_type"], 0) + 1
    for row in pair_rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
        stereo_label_counts[row["stereo_geometry_label_proxy"]] = stereo_label_counts.get(row["stereo_geometry_label_proxy"], 0) + 1
    return {
        "task_id": task_id,
        "num_trajectories": len(trajectories),
        "num_frame_rows": len(frame_rows),
        "num_clip_rows": len(clip_rows),
        "num_pair_rows": len(pair_rows),
        "source_counts": source_counts,
        "pair_label_counts": label_counts,
        "stereo_geometry_label_proxy_counts": stereo_label_counts,
        "pair_label_agreement_rate": (
            sum(1 for r in pair_rows if r["stereo_geometry_label_agrees_with_pair_label"]) / len(pair_rows) if pair_rows else 0.0
        ),
        "feature_source": "fixed_stereo_projection_geometry",
        "note": "Full-dataset stereo geometry proxy. It projects object/goal states into a fixed stereo rig and computes pseudo-disparity/depth features; it is not dense SGBM output.",
        "peg_note": "PegInsertion h5 rows use constant peg-head and box-hole local offsets inferred from the ManiSkill task; perturbed final states use manifest snapshot coordinates.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract fixed-stereo geometry features for a bootstrap dataset.")
    parser.add_argument("--task-id", required=True, choices=["stackcube", "stackpyramid", "peginsertion"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out or (DEFAULT_OUT_ROOT / f"{args.task_id}_bootstrap_v1")
    trajectories = load_csv(args.dataset / "trajectory_manifest.csv")
    clips = load_csv(args.indices / "clip_manifest.csv")
    pairs = load_csv(args.indices / "pair_manifest.csv")
    rig = make_stereo_rig()

    frame_rows: list[dict[str, Any]] = []
    frame_rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for traj in trajectories:
        rows = build_frame_rows(traj, args.task_id, rig)
        frame_rows.extend(rows)
        frame_rows_by_sample[traj["sample_id"]] = rows

    clip_rows = [aggregate_clip(clip, frame_rows_by_sample) for clip in clips]
    pair_rows = build_pair_rows(pairs, clip_rows)
    summary = summarize(args.task_id, trajectories, frame_rows, clip_rows, pair_rows)
    rig_config = {
        "center_eye": rig.center_eye.tolist(),
        "look_at": rig.look_at.tolist(),
        "up": rig.up.tolist(),
        "baseline_m": rig.baseline_m,
        "focal_px": rig.focal_px,
    }

    write_csv(out_dir / "frame_stereo_geometry_features.csv", frame_rows)
    write_json(out_dir / "frame_stereo_geometry_features.json", frame_rows)
    write_csv(out_dir / "clip_stereo_geometry_features.csv", clip_rows)
    write_json(out_dir / "clip_stereo_geometry_features.json", clip_rows)
    write_csv(out_dir / "pair_stereo_geometry_labels.csv", pair_rows)
    write_json(out_dir / "pair_stereo_geometry_labels.json", pair_rows)
    write_json(out_dir / "stereo_geometry_summary.json", summary)
    write_json(out_dir / "stereo_rig_config.json", rig_config)
    (out_dir / "README.txt").write_text(
        "\n".join(
            [
                f"{args.task_id} bootstrap fixed-stereo geometry features",
                "",
                "输出：frame / clip / pair 三级 CSV+JSON。",
                "含义：把 ManiSkill env_state 或 manifest 末态投影到固定双目相机坐标系，得到 depth / pseudo disparity / object-goal stereo alignment 特征。",
                "",
                "注意：",
                "- 这是全量低成本双目几何 proxy，不是 dense SGBM 视差图。",
                "- SGBM/RAFT-Stereo/IGEV 等像素级后端可以在同一 rig 上作为后续替换或审计。",
                "- PegInsertion h5 的 peg_head/hole 使用本任务固定 local offset 近似；perturbed final state 使用 manifest 记录的真实末态坐标。",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
