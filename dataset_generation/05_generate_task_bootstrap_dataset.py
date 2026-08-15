from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien
from mani_skill.examples.motionplanning.panda.solutions import solvePegInsertionSide, solveStackPyramid
from mani_skill.utils.wrappers.record import RecordEpisode


DEFAULT_OUT_ROOT = Path("/mnt/e/reward_model_dataset/raw_rollouts")


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    env_id: str
    sample_prefix: str
    solve: Callable[..., Any]
    offset_m: tuple[float, ...]
    perturb_directions: tuple[str, ...]
    source_desc: str


@dataclass(frozen=True)
class SuccessRecord:
    sample_id: str
    seed: int
    split: str
    h5_path: Path
    metadata_path: Path
    video_path: Path
    elapsed_steps: int


TASKS: dict[str, TaskConfig] = {
    "stackpyramid": TaskConfig(
        task_id="stackpyramid",
        env_id="StackPyramid-v1",
        sample_prefix="SP",
        solve=solveStackPyramid,
        offset_m=(0.055, 0.075, 0.095),
        perturb_directions=("posx", "negx", "posy", "negy"),
        source_desc="official ManiSkill StackPyramid Panda motion-planning success for reward-model dataset",
    ),
    "peginsertion": TaskConfig(
        task_id="peginsertion",
        env_id="PegInsertionSide-v1",
        sample_prefix="PEG",
        solve=solvePegInsertionSide,
        offset_m=(0.024, 0.036, 0.048),
        perturb_directions=("short", "side_pos", "side_neg", "up", "down"),
        source_desc="official ManiSkill PegInsertionSide Panda motion-planning success for reward-model dataset",
    ),
}


def scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def bool_scalar(value: Any) -> bool:
    value = scalar(value)
    if isinstance(value, list):
        arr = np.asarray(value).reshape(-1)
        return bool(arr[0]) if arr.size else False
    return bool(value)


def to_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def make_env(config: TaskConfig):
    return gym.make(
        config.env_id,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        sim_backend="cpu",
        render_backend="cpu",
    )


def render_frame(env) -> np.ndarray:
    frame = env.render()
    if isinstance(frame, list):
        frame = frame[0]
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame.astype(np.uint8)


def save_video(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError(f"No frames for {path}")
    imageio.mimsave(path, frames, fps=fps, quality=8)


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


def split_for_index(index: int, num_success: int) -> str:
    if num_success <= 2:
        return "train"
    val_start = max(1, int(round(num_success * 0.75)))
    test_start = max(val_start + 1, int(round(num_success * 0.875)))
    if index >= test_start:
        return "test"
    if index >= val_start:
        return "val"
    return "train"


def read_actions(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f["traj_0/actions"], dtype=np.float32)


def find_one(pattern: str, root: Path) -> Path | None:
    matches = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def normalize_video_name(sample_dir: Path, sample_id: str) -> Path:
    src = find_one("*.mp4", sample_dir)
    if src is None:
        raise RuntimeError(f"No video in {sample_dir}")
    dst = sample_dir / f"{sample_id}.mp4"
    if src != dst:
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
    return dst


def windows_path(path: Path) -> str:
    text = str(path)
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5].upper()
        rest = text[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return text


def env_success(info: dict[str, Any]) -> bool:
    return bool_scalar(info.get("success", False))


def run_success(config: TaskConfig, seed: int, sample_index: int, split: str, samples_dir: Path) -> SuccessRecord | None:
    sample_id = f"{config.sample_prefix}-SUCC-{seed:04d}"
    sample_dir = samples_dir / split / sample_id
    metadata_path = sample_dir / f"{sample_id}.json"
    if (sample_dir / f"{sample_id}.h5").exists() and (sample_dir / f"{sample_id}.mp4").exists():
        record = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return SuccessRecord(
            sample_id=sample_id,
            seed=seed,
            split=split,
            h5_path=sample_dir / f"{sample_id}.h5",
            metadata_path=metadata_path,
            video_path=sample_dir / f"{sample_id}.mp4",
            elapsed_steps=int(record.get("elapsed_steps", 0) or 0),
        )

    sample_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(config)
    env = RecordEpisode(
        env,
        output_dir=str(sample_dir),
        trajectory_name=sample_id,
        save_video=True,
        source_type="motionplanning_success",
        source_desc=config.source_desc,
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )
    record: dict[str, Any] = {}
    try:
        print(f"Running {config.env_id} success seed={seed}")
        result = config.solve(env, seed=seed, debug=False, vis=False)
        if result == -1:
            env.flush_trajectory(save=False)
            env.flush_video(save=False)
            return None
        final_info = result[-1]
        success = bool_scalar(final_info.get("success", False))
        record = {key: scalar(value) for key, value in final_info.items()}
        record.update(
            {
                "sample_id": sample_id,
                "task_id": config.task_id,
                "env_id": config.env_id,
                "seed": seed,
                "split": split,
                "source_type": "official_motionplanning_success",
                "success": success,
                "expected_success": True,
                "observed_success": success,
                "elapsed_steps": scalar(final_info.get("elapsed_steps")),
                "progress_rank_terminal": 3,
            }
        )
        env.flush_trajectory(save=success)
        env.flush_video(save=success)
    finally:
        env.close()

    if not record.get("success"):
        write_json(sample_dir / f"{sample_id}_failed_attempt.json", record)
        return None

    video_path = normalize_video_name(sample_dir, sample_id)
    h5_path = sample_dir / f"{sample_id}.h5"
    write_json(metadata_path, record)
    return SuccessRecord(
        sample_id=sample_id,
        seed=seed,
        split=split,
        h5_path=h5_path,
        metadata_path=metadata_path,
        video_path=video_path,
        elapsed_steps=int(record.get("elapsed_steps", 0) or 0),
    )


def stage_from_fraction(frac: float) -> str:
    if frac < 0.30:
        return "early_approach_or_pregrasp"
    if frac < 0.55:
        return "grasp_or_lift"
    if frac < 0.80:
        return "transport_or_align"
    return "place_before_stable_release"


def replay_truncated(config: TaskConfig, success: SuccessRecord, frac: float, samples_dir: Path) -> dict[str, Any]:
    actions = read_actions(success.h5_path)
    stop_step = max(1, min(len(actions), int(round(len(actions) * frac))))
    near_type = f"truncated_{stage_from_fraction(frac)}"
    sample_id = f"{success.sample_id}-TRUNC-{int(frac * 100):02d}"
    sample_dir = samples_dir / success.split / sample_id
    video_path = sample_dir / f"{sample_id}.mp4"
    metadata_path = sample_dir / f"{sample_id}.json"
    if video_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    env = make_env(config)
    frames: list[np.ndarray] = []
    info: dict[str, Any] = {}
    try:
        env.reset(seed=success.seed)
        frames.append(render_frame(env))
        for action in actions[:stop_step]:
            _, _, _, _, info = env.step(action)
            frames.append(render_frame(env))
    finally:
        env.close()

    save_video(video_path, frames)
    record = {
        "sample_id": sample_id,
        "task_id": config.task_id,
        "env_id": config.env_id,
        "split": success.split,
        "source_type": "truncated_success_trajectory",
        "seed": success.seed,
        "source_success_id": success.sample_id,
        "source_h5_path": str(success.h5_path),
        "video_path": str(video_path),
        "expected_success": False,
        "observed_success": env_success(info),
        "near_miss_type": near_type,
        "target_failure_mode": "trajectory is truncated before the full stable success condition",
        "stop_step": stop_step,
        "source_num_actions": len(actions),
        "fraction": frac,
        "num_frames": len(frames),
        "progress_rank_terminal": 1 if frac < 0.55 else 2,
    }
    write_json(metadata_path, record)
    return record


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.astype(np.float64)
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


def norm_vec(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= 1e-8:
        return fallback.astype(np.float64)
    return vec.astype(np.float64) / n


def evaluate_stackpyramid_snapshot(env) -> dict[str, Any]:
    ue = env.unwrapped
    info = ue.evaluate()
    half = float(to_np(ue.cube_half_size).reshape(-1)[0])
    cube_a = to_np(ue.cubeA.pose.p).reshape(-1)[:3]
    cube_b = to_np(ue.cubeB.pose.p).reshape(-1)[:3]
    cube_c = to_np(ue.cubeC.pose.p).reshape(-1)[:3]
    base_center = 0.5 * (cube_a + cube_b)
    goal = base_center + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64)
    return {
        "success": env_success(info),
        "cubeC_goal_3d_dist_m": float(np.linalg.norm(cube_c - goal)),
        "cubeC_goal_xy_dist_m": float(np.linalg.norm(cube_c[:2] - goal[:2])),
        "cubeC_height_error_m": float(abs(cube_c[2] - goal[2])),
        "base_cube_xy_dist_m": float(np.linalg.norm(cube_a[:2] - cube_b[:2])),
        "cubeC_x": float(cube_c[0]),
        "cubeC_y": float(cube_c[1]),
        "cubeC_z": float(cube_c[2]),
        "goal_x": float(goal[0]),
        "goal_y": float(goal[1]),
        "goal_z": float(goal[2]),
    }


def stackpyramid_perturb(env, direction_name: str, offset_m: float) -> dict[str, Any]:
    ue = env.unwrapped
    half = float(to_np(ue.cube_half_size).reshape(-1)[0])
    cube_a = to_np(ue.cubeA.pose.p).reshape(-1)[:3]
    cube_b = to_np(ue.cubeB.pose.p).reshape(-1)[:3]
    base_center = 0.5 * (cube_a + cube_b)
    cube_c_quat = to_np(ue.cubeC.pose.q).reshape(-1)[:4]
    directions = {
        "posx": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "negx": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "posy": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "negy": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    }
    offset = directions[direction_name] * offset_m + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64)
    ue.cubeC.set_pose(sapien.Pose(base_center + offset, cube_c_quat))
    snapshot = evaluate_stackpyramid_snapshot(env)
    snapshot["near_miss_type"] = "perturbed_top_cube_lateral_offset"
    snapshot["target_failure_mode"] = "top cube is near the pyramid height but laterally offset from base support"
    return snapshot


def evaluate_peg_snapshot(env) -> dict[str, Any]:
    ue = env.unwrapped
    info = ue.evaluate()
    peg_head = to_np(ue.peg_head_pose.p).reshape(-1)[:3]
    hole = to_np(ue.box_hole_pose.p).reshape(-1)[:3]
    peg = to_np(ue.peg.pose.p).reshape(-1)[:3]
    goal = to_np(ue.goal_pose.p).reshape(-1)[:3]
    peg_head_at_hole = to_np(info.get("peg_head_pos_at_hole", np.full(3, np.nan))).reshape(-1)[:3]
    return {
        "success": env_success(info),
        "peg_head_hole_3d_dist_m": float(np.linalg.norm(peg_head - hole)),
        "peg_head_hole_xy_dist_m": float(np.linalg.norm((peg_head - hole)[:2])),
        "peg_head_hole_z_error_m": float(abs(peg_head[2] - hole[2])),
        "peg_goal_3d_dist_m": float(np.linalg.norm(peg - goal)),
        "peg_head_at_hole_x": float(peg_head_at_hole[0]),
        "peg_head_at_hole_yz_norm": float(np.linalg.norm(peg_head_at_hole[1:3])),
        "peg_x": float(peg[0]),
        "peg_y": float(peg[1]),
        "peg_z": float(peg[2]),
        "peg_head_x": float(peg_head[0]),
        "peg_head_y": float(peg_head[1]),
        "peg_head_z": float(peg_head[2]),
        "hole_x": float(hole[0]),
        "hole_y": float(hole[1]),
        "hole_z": float(hole[2]),
        "goal_x": float(goal[0]),
        "goal_y": float(goal[1]),
        "goal_z": float(goal[2]),
    }


def peg_direction(env, direction_name: str) -> np.ndarray:
    ue = env.unwrapped
    hole = to_np(ue.box_hole_pose.p).reshape(-1)[:3]
    goal = to_np(ue.goal_pose.p).reshape(-1)[:3]
    axis = norm_vec(hole - goal, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    side = norm_vec(np.cross(axis, up), np.array([0.0, 1.0, 0.0], dtype=np.float64))
    directions = {
        "short": -axis,
        "side_pos": side,
        "side_neg": -side,
        "up": up,
        "down": -up,
    }
    return directions[direction_name]


def peg_perturb(env, direction_name: str, offset_m: float) -> dict[str, Any]:
    ue = env.unwrapped
    peg_pos = to_np(ue.peg.pose.p).reshape(-1)[:3]
    peg_quat = to_np(ue.peg.pose.q).reshape(-1)[:4]
    offset = peg_direction(env, direction_name) * offset_m
    ue.peg.set_pose(sapien.Pose(peg_pos + offset, peg_quat))
    snapshot = evaluate_peg_snapshot(env)
    if direction_name == "short":
        snapshot["near_miss_type"] = "peg_aligned_but_short_of_insertion"
        snapshot["target_failure_mode"] = "peg is aligned with the hole but not inserted deeply enough"
    else:
        snapshot["near_miss_type"] = "peg_lateral_or_vertical_misalignment"
        snapshot["target_failure_mode"] = "peg head is near the hole but misaligned with the insertion axis"
    return snapshot


def replay_perturbed(
    config: TaskConfig,
    success: SuccessRecord,
    offset_m: float,
    direction_name: str,
    samples_dir: Path,
    hold_frames: int,
) -> dict[str, Any]:
    sample_id = f"{success.sample_id}-OFFSET-{direction_name}-{int(round(offset_m * 1000)):03d}mm"
    sample_dir = samples_dir / success.split / sample_id
    video_path = sample_dir / f"{sample_id}.mp4"
    metadata_path = sample_dir / f"{sample_id}.json"
    if video_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    actions = read_actions(success.h5_path)
    env = make_env(config)
    frames: list[np.ndarray] = []
    snapshot: dict[str, Any] = {}
    try:
        env.reset(seed=success.seed)
        for action in actions:
            env.step(action)
        if config.task_id == "stackpyramid":
            snapshot = stackpyramid_perturb(env, direction_name, offset_m)
        elif config.task_id == "peginsertion":
            snapshot = peg_perturb(env, direction_name, offset_m)
        else:
            raise ValueError(config.task_id)
        for _ in range(hold_frames):
            frames.append(render_frame(env))
    finally:
        env.close()

    save_video(video_path, frames)
    record = {
        "sample_id": sample_id,
        "task_id": config.task_id,
        "env_id": config.env_id,
        "split": success.split,
        "source_type": "perturbed_success_final_state",
        "seed": success.seed,
        "source_success_id": success.sample_id,
        "source_h5_path": str(success.h5_path),
        "video_path": str(video_path),
        "expected_success": False,
        "observed_success": bool(snapshot["success"]),
        "lateral_offset_m": offset_m,
        "direction": direction_name,
        "num_frames": len(frames),
        "progress_rank_terminal": 2,
        **snapshot,
    }
    write_json(metadata_path, record)
    return record


def build_manifest_row(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record)
    if "video_path" in row:
        row["video_path_wsl"] = row["video_path"]
        row["video_path_windows"] = windows_path(Path(row["video_path"]))
    if "h5_path" in row:
        row["h5_path_wsl"] = row["h5_path"]
        row["h5_path_windows"] = windows_path(Path(row["h5_path"]))
    if "source_h5_path" in row:
        row["source_h5_path_wsl"] = row["source_h5_path"]
        row["source_h5_path_windows"] = windows_path(Path(row["source_h5_path"]))
    return row


def parse_floats(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    return tuple(float(x) for x in text.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate task bootstrap success/truncation/near-miss data.")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--num-success", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--truncate-fractions", default="0.20,0.40,0.60,0.78,0.92")
    parser.add_argument("--offsets-m", default=None)
    parser.add_argument("--hold-frames", type=int, default=28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TASKS[args.task]
    out_dir = args.out or (DEFAULT_OUT_ROOT / f"{config.task_id}_bootstrap_v1")
    samples_dir = out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    truncate_fractions = parse_floats(args.truncate_fractions, (0.20, 0.40, 0.60, 0.78, 0.92))
    offsets_m = parse_floats(args.offsets_m, config.offset_m)

    successes: list[SuccessRecord] = []
    attempts = 0
    while len(successes) < args.num_success and attempts < args.max_attempts:
        seed = args.seed_start + attempts
        split = split_for_index(len(successes), args.num_success)
        attempts += 1
        rec = run_success(config, seed, len(successes), split, samples_dir)
        if rec is not None:
            successes.append(rec)

    rows: list[dict[str, Any]] = []
    skipped_observed_success: list[str] = []
    for rec in successes:
        rows.append(
            build_manifest_row(
                {
                    "sample_id": rec.sample_id,
                    "task_id": config.task_id,
                    "env_id": config.env_id,
                    "split": rec.split,
                    "source_type": "official_motionplanning_success",
                    "seed": rec.seed,
                    "video_path": str(rec.video_path),
                    "h5_path": str(rec.h5_path),
                    "metadata_path": str(rec.metadata_path),
                    "expected_success": True,
                    "observed_success": True,
                    "near_miss_type": "",
                    "target_failure_mode": "",
                    "elapsed_steps": rec.elapsed_steps,
                    "progress_rank_terminal": 3,
                }
            )
        )
        for frac in truncate_fractions:
            record = replay_truncated(config, rec, frac, samples_dir)
            if record.get("observed_success"):
                skipped_observed_success.append(record["sample_id"])
            else:
                rows.append(build_manifest_row(record))
        for offset_m in offsets_m:
            for direction_name in config.perturb_directions:
                record = replay_perturbed(config, rec, offset_m, direction_name, samples_dir, args.hold_frames)
                if record.get("observed_success"):
                    skipped_observed_success.append(record["sample_id"])
                else:
                    rows.append(build_manifest_row(record))

    manifest_csv = out_dir / "trajectory_manifest.csv"
    manifest_json = out_dir / "trajectory_manifest.json"
    write_csv(manifest_csv, rows)
    write_json(manifest_json, rows)

    source_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    success_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row["source_type"]] = source_counts.get(row["source_type"], 0) + 1
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        success_counts[str(row["observed_success"])] = success_counts.get(str(row["observed_success"]), 0) + 1
    summary = {
        "task_id": config.task_id,
        "env_id": config.env_id,
        "out_dir": str(out_dir),
        "num_success_requested": args.num_success,
        "num_success_saved": len(successes),
        "attempts": attempts,
        "num_trajectories": len(rows),
        "source_counts": source_counts,
        "split_counts": split_counts,
        "observed_success_counts": success_counts,
        "truncate_fractions": truncate_fractions,
        "offsets_m": offsets_m,
        "perturb_directions": config.perturb_directions,
        "skipped_observed_success_count": len(skipped_observed_success),
        "skipped_observed_success_ids": skipped_observed_success,
    }
    write_json(out_dir / "dataset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
