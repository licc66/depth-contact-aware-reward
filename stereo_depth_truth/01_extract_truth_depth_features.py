from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    env_id: str
    h5_path: Path
    summary_path: Path


@dataclass(frozen=True)
class StereoRig:
    center_eye: np.ndarray
    look_at: np.ndarray
    up: np.ndarray
    baseline_m: float
    focal_px: float


def default_tasks(root: Path) -> list[TaskSpec]:
    base = root / "paper_style_tasks" / "outputs" / "wsl_motionplanning"
    return [
        TaskSpec(
            task_id="stackcube",
            env_id="StackCube-v1",
            h5_path=base / "StackCube-v1" / "motionplanning" / "stackcube_wsl_mplib.h5",
            summary_path=base / "StackCube-v1" / "motionplanning" / "stackcube_wsl_mplib_run_summary.json",
        ),
        TaskSpec(
            task_id="stackpyramid",
            env_id="StackPyramid-v1",
            h5_path=base / "StackPyramid-v1" / "motionplanning" / "stackpyramid_wsl_mplib.h5",
            summary_path=base / "StackPyramid-v1" / "motionplanning" / "stackpyramid_wsl_mplib_run_summary.json",
        ),
        TaskSpec(
            task_id="peginsertion",
            env_id="PegInsertionSide-v1",
            h5_path=base / "PegInsertionSide-v1" / "motionplanning" / "peg_insertion_wsl_mplib.h5",
            summary_path=base / "PegInsertionSide-v1" / "motionplanning" / "peg_insertion_wsl_mplib_run_summary.json",
        ),
    ]


def to_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def vec3(value: Any) -> np.ndarray:
    return to_np(value).reshape(-1)[:3].astype(np.float64)


def bool_scalar(value: Any) -> bool:
    arr = to_np(value).reshape(-1)
    return bool(arr[0]) if arr.size else bool(value)


def scalar(value: Any) -> float:
    arr = to_np(value).reshape(-1)
    return float(arr[0]) if arr.size else float(value)


def norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(value))


def make_stereo_rig() -> StereoRig:
    # A fixed side-front, slightly top-down rig. We use this as the coordinate
    # system that a future stereo matcher would recover points in.
    return StereoRig(
        center_eye=np.array([0.55, -0.65, 0.42], dtype=np.float64),
        look_at=np.array([0.00, 0.00, 0.08], dtype=np.float64),
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


def camera_centers(rig: StereoRig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis, _, _ = camera_axes(rig)
    left = rig.center_eye - x_axis * (rig.baseline_m / 2.0)
    right = rig.center_eye + x_axis * (rig.baseline_m / 2.0)
    return left, right, rig.center_eye


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


def read_actions(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f["traj_0/actions"], dtype=np.float32)


def read_success_seed(path: Path, fallback: int = 0) -> int:
    if not path.exists():
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    for record in records:
        if record.get("success"):
            return int(record.get("seed", fallback))
    return fallback


def get_entities(env, task_id: str) -> dict[str, np.ndarray | float | bool]:
    ue = env.unwrapped
    tcp = vec3(ue.agent.tcp.pose.p)

    if task_id == "stackcube":
        half = float(scalar(ue.cube_half_size))
        object_pos = vec3(ue.cubeA.pose.p)
        support_pos = vec3(ue.cubeB.pose.p)
        goal_pos = support_pos + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64)
        info = ue.evaluate()
        return {
            "object_name": "cubeA",
            "goal_name": "cubeB_top_center",
            "tcp": tcp,
            "object": object_pos,
            "goal": goal_pos,
            "support": support_pos,
            "success": bool_scalar(info["success"]),
            "task_truth_1": float(bool_scalar(info["is_cubeA_on_cubeB"])),
            "task_truth_2": float(bool_scalar(info["is_cubeA_static"])),
        }

    if task_id == "stackpyramid":
        half = float(scalar(ue.cube_half_size))
        object_pos = vec3(ue.cubeC.pose.p)
        base_a = vec3(ue.cubeA.pose.p)
        base_b = vec3(ue.cubeB.pose.p)
        base_center = 0.5 * (base_a + base_b)
        goal_pos = base_center + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64)
        info = ue.evaluate()
        return {
            "object_name": "cubeC",
            "goal_name": "base_cubes_top_center",
            "tcp": tcp,
            "object": object_pos,
            "goal": goal_pos,
            "support": base_center,
            "success": bool_scalar(info["success"]),
            "task_truth_1": norm(base_a - base_b),
            "task_truth_2": float("nan"),
        }

    if task_id == "peginsertion":
        info = ue.evaluate()
        peg_head = vec3(ue.peg_head_pose.p)
        hole = vec3(ue.box_hole_pose.p)
        peg_head_at_hole = vec3(info["peg_head_pos_at_hole"])
        return {
            "object_name": "peg_head",
            "goal_name": "hole_center",
            "tcp": tcp,
            "object": peg_head,
            "goal": hole,
            "support": vec3(ue.peg.pose.p),
            "success": bool_scalar(info["success"]),
            "task_truth_1": float(peg_head_at_hole[0]),
            "task_truth_2": float(np.linalg.norm(peg_head_at_hole[1:3])),
        }

    raise ValueError(f"Unknown task_id: {task_id}")


def append_geometry(row: dict[str, Any], entities: dict[str, Any], rig: StereoRig) -> None:
    left_eye, right_eye, center_eye = camera_centers(rig)
    object_pos = np.asarray(entities["object"], dtype=np.float64)
    goal_pos = np.asarray(entities["goal"], dtype=np.float64)
    tcp_pos = np.asarray(entities["tcp"], dtype=np.float64)

    object_center_cam = world_to_cam(object_pos, center_eye, rig)
    goal_center_cam = world_to_cam(goal_pos, center_eye, rig)
    tcp_center_cam = world_to_cam(tcp_pos, center_eye, rig)
    object_left_cam = world_to_cam(object_pos, left_eye, rig)
    goal_left_cam = world_to_cam(goal_pos, left_eye, rig)
    tcp_left_cam = world_to_cam(tcp_pos, left_eye, rig)
    object_right_cam = world_to_cam(object_pos, right_eye, rig)
    goal_right_cam = world_to_cam(goal_pos, right_eye, rig)

    object_goal_world = object_pos - goal_pos
    object_goal_cam = object_center_cam - goal_center_cam
    tcp_object_world = tcp_pos - object_pos
    tcp_goal_world = tcp_pos - goal_pos

    row.update(
        {
            "object_name": entities["object_name"],
            "goal_name": entities["goal_name"],
            "tcp_x": tcp_pos[0],
            "tcp_y": tcp_pos[1],
            "tcp_z": tcp_pos[2],
            "object_x": object_pos[0],
            "object_y": object_pos[1],
            "object_z": object_pos[2],
            "goal_x": goal_pos[0],
            "goal_y": goal_pos[1],
            "goal_z": goal_pos[2],
            "support_x": np.asarray(entities["support"], dtype=np.float64)[0],
            "support_y": np.asarray(entities["support"], dtype=np.float64)[1],
            "support_z": np.asarray(entities["support"], dtype=np.float64)[2],
            "tcp_object_3d_dist": norm(tcp_object_world),
            "tcp_goal_3d_dist": norm(tcp_goal_world),
            "object_goal_3d_dist": norm(object_goal_world),
            "object_goal_xy_dist_world": norm(object_goal_world[:2]),
            "object_goal_height_error_world": abs(float(object_goal_world[2])),
            "object_cam_x": object_center_cam[0],
            "object_cam_y": object_center_cam[1],
            "object_cam_depth": object_center_cam[2],
            "goal_cam_x": goal_center_cam[0],
            "goal_cam_y": goal_center_cam[1],
            "goal_cam_depth": goal_center_cam[2],
            "tcp_cam_x": tcp_center_cam[0],
            "tcp_cam_y": tcp_center_cam[1],
            "tcp_cam_depth": tcp_center_cam[2],
            "object_goal_cam_lateral_error": norm(object_goal_cam[:2]),
            "object_goal_cam_depth_error": abs(float(object_goal_cam[2])),
            "tcp_object_cam_depth_error": abs(float(tcp_center_cam[2] - object_center_cam[2])),
            "left_object_depth": object_left_cam[2],
            "left_goal_depth": goal_left_cam[2],
            "left_tcp_depth": tcp_left_cam[2],
            "right_object_depth": object_right_cam[2],
            "right_goal_depth": goal_right_cam[2],
            "object_pseudo_disparity_px": pseudo_disparity(object_center_cam[2], rig),
            "goal_pseudo_disparity_px": pseudo_disparity(goal_center_cam[2], rig),
        }
    )


def replay_task(task: TaskSpec, rig: StereoRig) -> list[dict[str, Any]]:
    if not task.h5_path.exists():
        raise FileNotFoundError(task.h5_path)
    actions = read_actions(task.h5_path)
    seed = read_success_seed(task.summary_path)

    env = gym.make(
        task.env_id,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        render_mode=None,
        render_backend="cpu",
        sim_backend="cpu",
        max_episode_steps=max(1000, len(actions) + 10),
    )
    env.reset(seed=seed)
    rows: list[dict[str, Any]] = []

    def record(step: int) -> None:
        entities = get_entities(env, task.task_id)
        row: dict[str, Any] = {
            "task_id": task.task_id,
            "env_id": task.env_id,
            "seed": seed,
            "step": step,
            "time_progress": step / max(len(actions), 1),
            "success": bool(entities["success"]),
            "task_truth_1": entities["task_truth_1"],
            "task_truth_2": entities["task_truth_2"],
        }
        append_geometry(row, entities, rig)
        rows.append(row)

    record(0)
    for idx, action in enumerate(actions, start=1):
        env.step(action)
        record(idx)
    env.close()
    return rows


def normalized_progress(values: np.ndarray) -> np.ndarray:
    start = float(values[0])
    end = float(values[-1])
    denom = start - end
    if abs(denom) < 1e-8:
        return np.zeros_like(values, dtype=np.float64)
    progress = (start - values) / denom
    return np.clip(progress, 0.0, 1.0)


def add_progress_columns(rows: list[dict[str, Any]]) -> None:
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    for task_id in task_ids:
        idxs = [idx for idx, row in enumerate(rows) if row["task_id"] == task_id]
        object_goal = np.array([rows[idx]["object_goal_3d_dist"] for idx in idxs], dtype=np.float64)
        depth_error = np.array([rows[idx]["object_goal_cam_depth_error"] for idx in idxs], dtype=np.float64)
        lateral_error = np.array([rows[idx]["object_goal_cam_lateral_error"] for idx in idxs], dtype=np.float64)

        object_goal_progress = normalized_progress(object_goal)
        depth_progress = normalized_progress(depth_error)
        lateral_progress = normalized_progress(lateral_error)
        geometry_progress = np.clip(
            0.60 * object_goal_progress + 0.25 * lateral_progress + 0.15 * depth_progress,
            0.0,
            1.0,
        )

        prev_dist = object_goal[0]
        for local_i, row_idx in enumerate(idxs):
            rows[row_idx]["object_goal_progress"] = float(object_goal_progress[local_i])
            rows[row_idx]["depth_alignment_progress"] = float(depth_progress[local_i])
            rows[row_idx]["lateral_alignment_progress"] = float(lateral_progress[local_i])
            rows[row_idx]["geometry_progress_preview"] = float(geometry_progress[local_i])
            rows[row_idx]["delta_object_goal_dist"] = float(prev_dist - object_goal[local_i])
            prev_dist = object_goal[local_i]


def pairwise_order_accuracy(values: np.ndarray) -> float:
    total = 0
    correct = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            total += 1
            correct += int(values[j] >= values[i])
    return correct / total if total else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    for task_id in task_ids:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        time = np.array([row["time_progress"] for row in task_rows], dtype=np.float64)
        geom = np.array([row["geometry_progress_preview"] for row in task_rows], dtype=np.float64)
        dist = np.array([row["object_goal_3d_dist"] for row in task_rows], dtype=np.float64)
        depth = np.array([row["object_goal_cam_depth_error"] for row in task_rows], dtype=np.float64)
        corr = spearmanr(time, geom).correlation
        if math.isnan(float(corr)):
            corr = 0.0
        summary.append(
            {
                "task_id": task_id,
                "env_id": task_rows[0]["env_id"],
                "steps": len(task_rows) - 1,
                "success_final": bool(task_rows[-1]["success"]),
                "object_name": task_rows[0]["object_name"],
                "goal_name": task_rows[0]["goal_name"],
                "object_goal_dist_start": float(dist[0]),
                "object_goal_dist_end": float(dist[-1]),
                "object_goal_dist_min": float(np.min(dist)),
                "camera_depth_error_start": float(depth[0]),
                "camera_depth_error_end": float(depth[-1]),
                "camera_depth_error_min": float(np.min(depth)),
                "geometry_progress_spearman_vs_time": float(corr),
                "geometry_progress_pairwise_order_acc": float(pairwise_order_accuracy(geom)),
                "geometry_progress_end": float(geom[-1]),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: list[dict[str, Any]], out_path: Path) -> None:
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(10, 3.3 * len(task_ids)), sharex=False)
    if len(task_ids) == 1:
        axes = [axes]
    for ax, task_id in zip(axes, task_ids):
        task_rows = [row for row in rows if row["task_id"] == task_id]
        x = [row["time_progress"] for row in task_rows]
        ax.plot(x, [row["object_goal_progress"] for row in task_rows], label="object-goal 3D progress", linewidth=2)
        ax.plot(x, [row["depth_alignment_progress"] for row in task_rows], label="camera-depth alignment", linewidth=1.5)
        ax.plot(x, [row["lateral_alignment_progress"] for row in task_rows], label="camera-lateral alignment", linewidth=1.5)
        ax.plot(x, [row["geometry_progress_preview"] for row in task_rows], label="combined preview", linewidth=2.5)
        ax.plot(x, x, "--", color="#9ca3af", label="time baseline", linewidth=1)
        ax.set_title(task_id)
        ax.set_xlabel("normalized time")
        ax.set_ylabel("normalized signal")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_camera_paths(rows: list[dict[str, Any]], out_path: Path) -> None:
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    fig, axes = plt.subplots(1, len(task_ids), figsize=(5.0 * len(task_ids), 4.2), sharey=False)
    if len(task_ids) == 1:
        axes = [axes]
    for ax, task_id in zip(axes, task_ids):
        task_rows = [row for row in rows if row["task_id"] == task_id]
        ox = np.array([row["object_cam_x"] for row in task_rows])
        oz = np.array([row["object_cam_depth"] for row in task_rows])
        gx = np.array([row["goal_cam_x"] for row in task_rows])
        gz = np.array([row["goal_cam_depth"] for row in task_rows])
        tx = np.array([row["tcp_cam_x"] for row in task_rows])
        tz = np.array([row["tcp_cam_depth"] for row in task_rows])
        ax.plot(oz, ox, color="#f97316", label="object path")
        ax.plot(tz, tx, color="#2563eb", alpha=0.8, label="tcp path")
        ax.scatter([oz[0]], [ox[0]], color="#f97316", marker="o", s=45, label="object start")
        ax.scatter([oz[-1]], [ox[-1]], color="#f97316", marker="*", s=90, label="object end")
        ax.scatter([gz[-1]], [gx[-1]], color="#16a34a", marker="X", s=70, label="goal")
        ax.plot(gz, gx, color="#16a34a", alpha=0.4)
        ax.plot(tz, tx, color="#2563eb", alpha=0.6)
        ax.set_title(f"{task_id} camera-space x/depth")
        ax.set_xlabel("camera depth z (m)")
        ax.set_ylabel("camera horizontal x (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract fixed-stereo-view geometry features using ManiSkill environment truth.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("stereo_depth_truth") / "outputs" / "three_task_truth_depth")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out
    rig = make_stereo_rig()

    rows: list[dict[str, Any]] = []
    for task in default_tasks(root):
        print(f"Replaying {task.task_id} from {task.h5_path}")
        rows.extend(replay_task(task, rig))

    add_progress_columns(rows)
    summary_rows = summarize(rows)

    write_csv(out_dir / "truth_depth_features.csv", rows)
    write_csv(out_dir / "truth_depth_summary.csv", summary_rows)
    (out_dir / "truth_depth_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    config = {
        "note": "Environment-truth stand-in for future fixed-stereo reconstruction. All task rows use the same fixed stereo rig and generic entity relation features.",
        "stereo_rig": {
            "center_eye_world": rig.center_eye.tolist(),
            "look_at_world": rig.look_at.tolist(),
            "up_world": rig.up.tolist(),
            "baseline_m": rig.baseline_m,
            "focal_px": rig.focal_px,
            "left_eye_world": camera_centers(rig)[0].tolist(),
            "right_eye_world": camera_centers(rig)[1].tolist(),
        },
            "entity_mapping": {
            "stackcube": {"object": "cubeA center", "goal": "cubeB top-center"},
            "stackpyramid": {"object": "cubeC center", "goal": "current base cube pair top-center"},
            "peginsertion": {"object": "peg_head_pose.p", "goal": "box_hole_pose.p"},
        },
    }
    (out_dir / "stereo_truth_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_curves(rows, out_dir / "truth_depth_progress_curves.png")
    plot_camera_paths(rows, out_dir / "fixed_camera_geometry_paths.png")

    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    print(f"Saved features: {out_dir / 'truth_depth_features.csv'}")
    print(f"Saved summary: {out_dir / 'truth_depth_summary.csv'}")
    print(f"Saved curves: {out_dir / 'truth_depth_progress_curves.png'}")
    print(f"Saved camera paths: {out_dir / 'fixed_camera_geometry_paths.png'}")


if __name__ == "__main__":
    main()
