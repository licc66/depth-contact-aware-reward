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


DEFAULT_DATASET = Path(r"E:\reward_model_dataset\raw_rollouts\stackcube_bootstrap_v1")
DEFAULT_INDICES = Path(r"E:\reward_model_dataset\pair_indices\stackcube_bootstrap_v1")
DEFAULT_OUT = Path(r"E:\reward_model_dataset\depth_contact_features\stackcube_bootstrap_v1")


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


def camera_axes(rig: StereoRig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = rig.look_at - rig.center_eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, rig.up)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(x_axis, z_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return x_axis, y_axis, z_axis


def world_to_cam(point: np.ndarray, eye: np.ndarray, rig: StereoRig) -> np.ndarray:
    x_axis, y_axis, z_axis = camera_axes(rig)
    delta = point - eye
    return np.array(
        [
            float(np.dot(delta, x_axis)),
            float(np.dot(delta, y_axis)),
            float(np.dot(delta, z_axis)),
        ],
        dtype=np.float64,
    )


def pseudo_disparity(depth_m: float, rig: StereoRig) -> float:
    if depth_m <= 1e-6:
        return float("nan")
    return float(rig.focal_px * rig.baseline_m / depth_m)


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


def frame_to_state_idx(frame_idx: int, video_frames: int, state_frames: int) -> int:
    if state_frames <= 1 or video_frames <= 1:
        return max(0, min(state_frames - 1, frame_idx))
    ratio = max(0.0, min(1.0, frame_idx / float(video_frames - 1)))
    return int(round(ratio * (state_frames - 1)))


def progress_from_distance(dist_m: float) -> float:
    # StackCube starts are usually within about 0.20-0.24 m of the stacking goal.
    return float(np.clip(1.0 - dist_m / 0.24, 0.0, 1.0))


def stage_from_geometry(
    moved_m: float,
    cube_a_z: float,
    cube_b_z: float,
    half_size: float,
    xy_error_m: float,
    height_error_m: float,
    static_proxy: bool,
) -> int:
    on_target = xy_error_m <= 0.025 and height_error_m <= 0.015 and cube_a_z > cube_b_z + half_size
    if on_target and static_proxy:
        return 3
    if xy_error_m <= 0.045 and height_error_m <= 0.035 and cube_a_z > cube_b_z + half_size:
        return 2
    if moved_m >= 0.025 or cube_a_z > cube_b_z + 1.25 * half_size:
        return 1
    return 0


def read_trajectory_arrays(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        group = f["traj_0"]
        return {
            "cubeA": np.asarray(group["env_states/actors/cubeA"], dtype=np.float64),
            "cubeB": np.asarray(group["env_states/actors/cubeB"], dtype=np.float64),
            "success": np.asarray(group["success"], dtype=bool) if "success" in group else np.array([], dtype=bool),
        }


def resolve_h5_path(traj: dict[str, str]) -> Path:
    for key in ("h5_path", "source_h5_path_windows", "source_h5_path"):
        value = (traj.get(key) or "").strip()
        if value:
            return wsl_to_windows(value)
    raise ValueError(f"No h5/source_h5 path for {traj.get('sample_id')}")


def arrays_for_trajectory(traj: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    source_type = traj.get("source_type", "")
    h5_path = resolve_h5_path(traj)
    arrays = read_trajectory_arrays(h5_path)
    cube_a = arrays["cubeA"][:, :3]
    cube_b = arrays["cubeB"][:, :3]

    if source_type == "truncated_success_trajectory":
        stop_step = safe_int(traj.get("stop_step"), len(cube_a) - 1)
        keep = max(1, min(len(cube_a), stop_step + 1))
        return cube_a[:keep], cube_b[:keep]

    if source_type == "perturbed_success_final_state":
        video_frames = safe_int(traj.get("num_frames"), 1)
        base_b = cube_b[-1]
        half_size = float(np.median(np.minimum(cube_a[: min(8, len(cube_a)), 2], cube_b[: min(8, len(cube_b)), 2])))
        if not math.isfinite(half_size) or half_size <= 0:
            half_size = 0.02
        goal = np.array(
            [
                safe_float(traj.get("goal_x"), float(base_b[0])),
                safe_float(traj.get("goal_y"), float(base_b[1])),
                safe_float(traj.get("goal_z"), float(base_b[2] + 2.0 * half_size)),
            ],
            dtype=np.float64,
        )
        a_pos = np.array(
            [
                safe_float(traj.get("cubeA_x"), float(goal[0])),
                safe_float(traj.get("cubeA_y"), float(goal[1])),
                safe_float(traj.get("cubeA_z"), float(goal[2])),
            ],
            dtype=np.float64,
        )
        b_pos = np.array([goal[0], goal[1], goal[2] - 2.0 * half_size], dtype=np.float64)
        return np.repeat(a_pos[None, :], max(1, video_frames), axis=0), np.repeat(b_pos[None, :], max(1, video_frames), axis=0)

    return cube_a, cube_b


def extract_frame_rows(traj: dict[str, str], rig: StereoRig) -> list[dict[str, Any]]:
    cube_a, cube_b = arrays_for_trajectory(traj)
    state_frames = len(cube_a)
    if state_frames == 0:
        return []

    video_frames = safe_int(traj.get("num_frames")) or state_frames
    half_size = float(np.median(np.minimum(cube_a[: min(8, state_frames), 2], cube_b[: min(8, state_frames), 2])))
    if not math.isfinite(half_size) or half_size <= 0:
        half_size = 0.02

    start_cube_a = cube_a[0]
    rows: list[dict[str, Any]] = []
    prev_dist = float("nan")

    for frame_idx in range(video_frames):
        state_idx = frame_to_state_idx(frame_idx, video_frames, state_frames)
        a_pos = cube_a[state_idx]
        b_pos = cube_b[state_idx]
        goal = b_pos + np.array([0.0, 0.0, 2.0 * half_size], dtype=np.float64)
        object_goal = a_pos - goal
        dist = float(np.linalg.norm(object_goal))
        xy_error = float(np.linalg.norm(object_goal[:2]))
        height_error = abs(float(object_goal[2]))
        moved = float(np.linalg.norm(a_pos - start_cube_a))

        if state_idx > 0:
            speed = float(np.linalg.norm(cube_a[state_idx, :3] - cube_a[state_idx - 1, :3]))
        else:
            speed = 0.0
        static_proxy = speed <= 0.002
        on_target_proxy = xy_error <= 0.025 and height_error <= 0.015 and a_pos[2] > b_pos[2] + half_size
        success_proxy = bool(on_target_proxy and static_proxy)
        stage = stage_from_geometry(moved, float(a_pos[2]), float(b_pos[2]), half_size, xy_error, height_error, static_proxy)

        object_cam = world_to_cam(a_pos, rig.center_eye, rig)
        goal_cam = world_to_cam(goal, rig.center_eye, rig)
        object_goal_cam = object_cam - goal_cam
        progress = progress_from_distance(dist)
        physical_score = stage + progress

        rows.append(
            {
                "sample_id": traj["sample_id"],
                "task_id": "stackcube",
                "split": traj["split"],
                "source_type": traj["source_type"],
                "near_miss_type": traj.get("near_miss_type", ""),
                "frame_idx": frame_idx,
                "state_idx": state_idx,
                "video_frames": video_frames,
                "state_frames": state_frames,
                "time_progress": frame_idx / max(1, video_frames - 1),
                "cubeA_x": float(a_pos[0]),
                "cubeA_y": float(a_pos[1]),
                "cubeA_z": float(a_pos[2]),
                "cubeB_x": float(b_pos[0]),
                "cubeB_y": float(b_pos[1]),
                "cubeB_z": float(b_pos[2]),
                "goal_x": float(goal[0]),
                "goal_y": float(goal[1]),
                "goal_z": float(goal[2]),
                "cube_half_size_m": half_size,
                "object_goal_3d_dist_m": dist,
                "object_goal_xy_error_m": xy_error,
                "object_goal_height_error_m": height_error,
                "delta_object_goal_dist_m": 0.0 if not math.isfinite(prev_dist) else prev_dist - dist,
                "object_moved_from_start_m": moved,
                "object_frame_speed_proxy_m": speed,
                "object_cam_x": float(object_cam[0]),
                "object_cam_y": float(object_cam[1]),
                "object_cam_depth_m": float(object_cam[2]),
                "goal_cam_x": float(goal_cam[0]),
                "goal_cam_y": float(goal_cam[1]),
                "goal_cam_depth_m": float(goal_cam[2]),
                "object_goal_cam_lateral_error_m": float(np.linalg.norm(object_goal_cam[:2])),
                "object_goal_cam_depth_error_m": abs(float(object_goal_cam[2])),
                "object_pseudo_disparity_px": pseudo_disparity(float(object_cam[2]), rig),
                "goal_pseudo_disparity_px": pseudo_disparity(float(goal_cam[2]), rig),
                "stack_contact_proxy": bool(on_target_proxy),
                "static_proxy": bool(static_proxy),
                "success_proxy": bool(success_proxy),
                "stage_proxy": stage,
                "geometry_progress_proxy": progress,
                "physical_score_proxy": physical_score,
                "feature_source": "env_state_geometry_proxy_no_contact_sensor",
            }
        )
        prev_dist = dist
    return rows


def aggregate_clip(clip: dict[str, str], frame_rows_by_sample: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    sample_id = clip["trajectory_id"]
    start = safe_int(clip["start_frame"])
    end = safe_int(clip["end_frame_exclusive"])
    rows = frame_rows_by_sample[sample_id][start:end]
    if not rows:
        return {"clip_id": clip["clip_id"], "has_physics_features": False}

    end_row = rows[-1]
    distances = np.array([safe_float(r["object_goal_3d_dist_m"]) for r in rows], dtype=np.float64)
    xy_errors = np.array([safe_float(r["object_goal_xy_error_m"]) for r in rows], dtype=np.float64)
    stages = np.array([safe_int(r["stage_proxy"]) for r in rows], dtype=np.int64)
    scores = np.array([safe_float(r["physical_score_proxy"]) for r in rows], dtype=np.float64)

    return {
        "clip_id": clip["clip_id"],
        "trajectory_id": sample_id,
        "task_id": "stackcube",
        "split": clip["split"],
        "source_type": clip["source_type"],
        "near_miss_type": clip.get("near_miss_type", ""),
        "start_frame": start,
        "end_frame_exclusive": end,
        "num_frames": len(rows),
        "has_physics_features": True,
        "end_stage_proxy": int(end_row["stage_proxy"]),
        "max_stage_proxy": int(np.max(stages)),
        "end_success_proxy": bool(end_row["success_proxy"]),
        "any_stack_contact_proxy": bool(any(bool(r["stack_contact_proxy"]) for r in rows)),
        "end_object_goal_3d_dist_m": float(end_row["object_goal_3d_dist_m"]),
        "mean_object_goal_3d_dist_m": float(np.mean(distances)),
        "min_object_goal_3d_dist_m": float(np.min(distances)),
        "end_object_goal_xy_error_m": float(end_row["object_goal_xy_error_m"]),
        "mean_object_goal_xy_error_m": float(np.mean(xy_errors)),
        "end_object_goal_height_error_m": float(end_row["object_goal_height_error_m"]),
        "end_object_goal_cam_depth_error_m": float(end_row["object_goal_cam_depth_error_m"]),
        "end_geometry_progress_proxy": float(end_row["geometry_progress_proxy"]),
        "mean_geometry_progress_proxy": float(np.mean([safe_float(r["geometry_progress_proxy"]) for r in rows])),
        "end_physical_score_proxy": float(end_row["physical_score_proxy"]),
        "mean_physical_score_proxy": float(np.mean(scores)),
        "delta_object_goal_dist_m": float(distances[0] - distances[-1]) if len(distances) else 0.0,
        "feature_source": "env_state_geometry_proxy_no_contact_sensor",
    }


def build_pair_rows(pairs: list[dict[str, str]], clip_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_clip = {row["clip_id"]: row for row in clip_rows if row.get("has_physics_features")}
    out: list[dict[str, Any]] = []
    for pair in pairs:
        a = by_clip[pair["clip_a_id"]]
        b = by_clip[pair["clip_b_id"]]
        score_a = safe_float(a["end_physical_score_proxy"])
        score_b = safe_float(b["end_physical_score_proxy"])
        diff = score_a - score_b
        if diff > 0.10:
            physical_label = "A>B"
        elif diff < -0.10:
            physical_label = "B>A"
        else:
            physical_label = "unsure"
        out.append(
            {
                **pair,
                "physical_label_proxy": physical_label,
                "physical_label_agrees_with_pair_label": physical_label == pair["label"],
                "clip_a_end_stage_proxy": a["end_stage_proxy"],
                "clip_b_end_stage_proxy": b["end_stage_proxy"],
                "clip_a_end_score_proxy": score_a,
                "clip_b_end_score_proxy": score_b,
                "score_diff_a_minus_b": diff,
                "clip_a_end_dist_m": a["end_object_goal_3d_dist_m"],
                "clip_b_end_dist_m": b["end_object_goal_3d_dist_m"],
                "feature_source": "env_state_geometry_proxy_no_contact_sensor",
            }
        )
    return out


def summarize(
    trajectories: list[dict[str, str]],
    frame_rows: list[dict[str, Any]],
    clip_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    pair_label_counts: dict[str, int] = {}
    physical_label_counts: dict[str, int] = {}
    for row in trajectories:
        source_counts[row["source_type"]] = source_counts.get(row["source_type"], 0) + 1
    for row in pair_rows:
        pair_label_counts[row["label"]] = pair_label_counts.get(row["label"], 0) + 1
        physical_label_counts[row["physical_label_proxy"]] = physical_label_counts.get(row["physical_label_proxy"], 0) + 1

    terminal = [r for r in clip_rows if r.get("has_physics_features")]
    return {
        "num_trajectories": len(trajectories),
        "num_frame_rows": len(frame_rows),
        "num_clip_rows": len(clip_rows),
        "num_pair_rows": len(pair_rows),
        "source_counts": source_counts,
        "pair_label_counts": pair_label_counts,
        "physical_label_proxy_counts": physical_label_counts,
        "pair_label_agreement_rate": (
            sum(1 for r in pair_rows if r["physical_label_agrees_with_pair_label"]) / len(pair_rows) if pair_rows else 0.0
        ),
        "clip_end_success_proxy_count": sum(1 for r in terminal if r.get("end_success_proxy")),
        "feature_source": "env_state_geometry_proxy_no_contact_sensor",
        "note": "These are low-cost geometry/depth/contact proxies from ManiSkill env_states, not real rendered stereo depth or simulator contact-force logs.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract StackCube bootstrap physics proxy features from saved h5 states.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--indices", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = load_csv(args.dataset / "trajectory_manifest.csv")
    clips = load_csv(args.indices / "clip_manifest.csv")
    pairs = load_csv(args.indices / "pair_manifest.csv")
    rig = make_stereo_rig()

    frame_rows: list[dict[str, Any]] = []
    frame_rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for traj in trajectories:
        rows = extract_frame_rows(traj, rig)
        frame_rows.extend(rows)
        frame_rows_by_sample[traj["sample_id"]] = rows

    clip_rows = [aggregate_clip(clip, frame_rows_by_sample) for clip in clips]
    pair_rows = build_pair_rows(pairs, clip_rows)
    summary = summarize(trajectories, frame_rows, clip_rows, pair_rows)

    write_csv(args.out / "frame_physics_proxy_features.csv", frame_rows)
    write_json(args.out / "frame_physics_proxy_features.json", frame_rows)
    write_csv(args.out / "clip_physics_proxy_features.csv", clip_rows)
    write_json(args.out / "clip_physics_proxy_features.json", clip_rows)
    write_csv(args.out / "pair_physics_proxy_labels.csv", pair_rows)
    write_json(args.out / "pair_physics_proxy_labels.json", pair_rows)
    write_json(args.out / "physics_proxy_summary.json", summary)
    (args.out / "README.txt").write_text(
        "\n".join(
            [
                "StackCube bootstrap physics proxy features",
                "",
                "输入：E:\\reward_model_dataset\\raw_rollouts\\stackcube_bootstrap_v1 的 h5/env_states 与 pair 索引。",
                "输出：frame / clip / pair 三级 CSV+JSON。",
                "",
                "注意：",
                "- 当前是 env_state 几何 proxy，不是真实双目图像估计结果。",
                "- stack_contact_proxy 是由 cubeA/cubeB 空间关系估计的接触/堆叠 proxy，不是真实 contact-force log。",
                "- object/goal camera depth 使用固定双目相机几何投影，与 stereo_depth_truth 中的 prototype 设定一致。",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
