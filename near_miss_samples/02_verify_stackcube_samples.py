from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien


def to_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def scalar(value: Any) -> float:
    arr = to_np(value).reshape(-1)
    return float(arr[0]) if arr.size else float(value)


def bool_scalar(value: Any) -> bool:
    arr = to_np(value).reshape(-1)
    return bool(arr[0]) if arr.size else bool(value)


def read_actions(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f["traj_0/actions"], dtype=np.float32)


def make_env():
    return gym.make(
        "StackCube-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend="cpu",
        render_backend="cpu",
    )


def collect_state(env, sample_id: str, source_type: str, stop_step: int | None = None) -> dict[str, Any]:
    ue = env.unwrapped
    info = ue.evaluate()
    cube_a = to_np(ue.cubeA.pose.p).reshape(-1)[:3]
    cube_b = to_np(ue.cubeB.pose.p).reshape(-1)[:3]
    half = scalar(ue.cube_half_size)
    goal = cube_b + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64)
    return {
        "sample_id": sample_id,
        "source_type": source_type,
        "stop_step": "" if stop_step is None else stop_step,
        "success": bool_scalar(info["success"]),
        "is_cubeA_on_cubeB": bool_scalar(info["is_cubeA_on_cubeB"]),
        "is_cubeA_static": bool_scalar(info["is_cubeA_static"]),
        "cubeA_goal_3d_dist_m": float(np.linalg.norm(cube_a - goal)),
        "cubeA_goal_xy_dist_m": float(np.linalg.norm((cube_a - goal)[:2])),
        "cubeA_height_error_m": float(abs(cube_a[2] - goal[2])),
        "cubeA_x": float(cube_a[0]),
        "cubeA_y": float(cube_a[1]),
        "cubeA_z": float(cube_a[2]),
        "goal_x": float(goal[0]),
        "goal_y": float(goal[1]),
        "goal_z": float(goal[2]),
    }


def replay_to_step(actions: np.ndarray, stop_step: int) -> dict[str, Any]:
    env = make_env()
    try:
        env.reset(seed=0)
        for action in actions[:stop_step]:
            env.step(action)
        return collect_state(env, "manual", "truncated_success_trajectory", stop_step)
    finally:
        env.close()


def replay_perturbed(actions: np.ndarray) -> dict[str, Any]:
    env = make_env()
    try:
        env.reset(seed=0)
        for action in actions:
            env.step(action)
        ue = env.unwrapped
        half = scalar(ue.cube_half_size)
        cube_b = to_np(ue.cubeB.pose.p).reshape(-1)[:3]
        cube_a_quat = to_np(ue.cubeA.pose.q).reshape(-1)[:4]
        perturbed_pos = cube_b + np.array([0.055, 0.0, 2.0 * half], dtype=np.float64)
        ue.cubeA.set_pose(sapien.Pose(perturbed_pos, cube_a_quat))
        return collect_state(env, "manual", "perturbed_success_final_state", None)
    finally:
        env.close()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify StackCube near-miss/failure terminal states.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("near_miss_samples") / "outputs" / "stackcube_verification.csv")
    args = parser.parse_args()

    h5_path = (
        args.root
        / "paper_style_tasks"
        / "outputs"
        / "wsl_motionplanning"
        / "StackCube-v1"
        / "motionplanning"
        / "stackcube_wsl_mplib.h5"
    )
    actions = read_actions(h5_path)
    rows: list[dict[str, Any]] = []
    row = replay_to_step(actions, 100)
    row["sample_id"] = "SC-NM-01"
    rows.append(row)
    row = replay_to_step(actions, 23)
    row["sample_id"] = "SC-FAIL-01"
    rows.append(row)
    row = replay_perturbed(actions)
    row["sample_id"] = "SC-NM-02"
    rows.append(row)
    write_csv(args.out, rows)
    (args.out.with_suffix(".json")).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
