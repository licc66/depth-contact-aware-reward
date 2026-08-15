from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien
from mani_skill.examples.motionplanning.panda.solutions import solveStackCube
from mani_skill.utils.wrappers.record import RecordEpisode


TASK_ID = "stackcube"
ENV_ID = "StackCube-v1"


@dataclass(frozen=True)
class SuccessRecord:
    sample_id: str
    seed: int
    split: str
    h5_path: Path
    metadata_path: Path
    video_path: Path
    elapsed_steps: int


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


def make_env():
    return gym.make(
        ENV_ID,
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
    with path.open("w", newline="", encoding="utf-8") as f:
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


def run_success(seed: int, sample_index: int, split: str, samples_dir: Path) -> SuccessRecord | None:
    sample_id = f"SC-SUCC-{seed:04d}"
    sample_dir = samples_dir / split / sample_id
    if (sample_dir / f"{sample_id}.h5").exists() and (sample_dir / f"{sample_id}.mp4").exists():
        metadata_path = sample_dir / f"{sample_id}.json"
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
    env = make_env()
    env = RecordEpisode(
        env,
        output_dir=str(sample_dir),
        trajectory_name=sample_id,
        save_video=True,
        source_type="motionplanning_success",
        source_desc="official ManiSkill StackCube Panda motion-planning success for reward-model dataset",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )
    try:
        print(f"Running StackCube success seed={seed}")
        result = solveStackCube(env, seed=seed, debug=False, vis=False)
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
                "task_id": TASK_ID,
                "env_id": ENV_ID,
                "seed": seed,
                "split": split,
                "source_type": "official_motionplanning_success",
                "success": success,
                "elapsed_steps": scalar(final_info.get("elapsed_steps")),
            }
        )
        env.flush_trajectory(save=success)
        env.flush_video(save=success)
    finally:
        env.close()

    if not record["success"]:
        write_json(sample_dir / f"{sample_id}_failed_attempt.json", record)
        return None

    video_path = normalize_video_name(sample_dir, sample_id)
    h5_path = sample_dir / f"{sample_id}.h5"
    metadata_path = sample_dir / f"{sample_id}.json"
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


def env_success(info: dict[str, Any]) -> bool:
    return bool_scalar(info.get("success", False))


def stage_from_fraction(frac: float) -> str:
    if frac < 0.30:
        return "early_approach_or_pregrasp"
    if frac < 0.55:
        return "grasp_or_lift"
    if frac < 0.80:
        return "transport_or_align"
    return "place_before_stable_release"


def replay_truncated(success: SuccessRecord, frac: float, samples_dir: Path) -> dict[str, Any]:
    actions = read_actions(success.h5_path)
    stop_step = max(1, min(len(actions), int(round(len(actions) * frac))))
    near_type = f"truncated_{stage_from_fraction(frac)}"
    sample_id = f"{success.sample_id}-TRUNC-{int(frac * 100):02d}"
    sample_dir = samples_dir / success.split / sample_id
    video_path = sample_dir / f"{sample_id}.mp4"
    metadata_path = sample_dir / f"{sample_id}.json"
    if video_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    env = make_env()
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
        "task_id": TASK_ID,
        "env_id": ENV_ID,
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


def evaluate_stackcube_snapshot(env) -> dict[str, Any]:
    ue = env.unwrapped
    info = ue.evaluate()
    half = float(to_np(ue.cube_half_size).reshape(-1)[0])
    cube_a = to_np(ue.cubeA.pose.p).reshape(-1)[:3]
    cube_b = to_np(ue.cubeB.pose.p).reshape(-1)[:3]
    goal = cube_b + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64)
    return {
        "success": env_success(info),
        "is_cubeA_on_cubeB": bool_scalar(info.get("is_cubeA_on_cubeB", False)),
        "is_cubeA_static": bool_scalar(info.get("is_cubeA_static", False)),
        "cubeA_goal_3d_dist_m": float(np.linalg.norm(cube_a - goal)),
        "cubeA_goal_xy_dist_m": float(np.linalg.norm(cube_a[:2] - goal[:2])),
        "cubeA_height_error_m": float(abs(cube_a[2] - goal[2])),
        "cubeA_x": float(cube_a[0]),
        "cubeA_y": float(cube_a[1]),
        "cubeA_z": float(cube_a[2]),
        "goal_x": float(goal[0]),
        "goal_y": float(goal[1]),
        "goal_z": float(goal[2]),
    }


def replay_perturbed(success: SuccessRecord, offset_m: float, direction: tuple[float, float], samples_dir: Path, hold_frames: int) -> dict[str, Any]:
    dx, dy = direction
    direction_name = {
        (1.0, 0.0): "posx",
        (-1.0, 0.0): "negx",
        (0.0, 1.0): "posy",
        (0.0, -1.0): "negy",
    }.get((dx, dy), f"{dx:g}_{dy:g}")
    sample_id = f"{success.sample_id}-OFFSET-{direction_name}-{int(round(offset_m * 1000)):03d}mm"
    sample_dir = samples_dir / success.split / sample_id
    video_path = sample_dir / f"{sample_id}.mp4"
    metadata_path = sample_dir / f"{sample_id}.json"
    if video_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    actions = read_actions(success.h5_path)
    env = make_env()
    frames: list[np.ndarray] = []
    snapshot: dict[str, Any] = {}
    try:
        env.reset(seed=success.seed)
        for action in actions:
            env.step(action)
        ue = env.unwrapped
        half = float(to_np(ue.cube_half_size).reshape(-1)[0])
        cube_b_pos = to_np(ue.cubeB.pose.p).reshape(-1)[:3]
        cube_a_quat = to_np(ue.cubeA.pose.q).reshape(-1)[:4]
        offset = np.array([dx * offset_m, dy * offset_m, 2.0 * half], dtype=np.float64)
        ue.cubeA.set_pose(sapien.Pose(cube_b_pos + offset, cube_a_quat))
        snapshot = evaluate_stackcube_snapshot(env)
        for _ in range(hold_frames):
            frames.append(render_frame(env))
    finally:
        env.close()

    save_video(video_path, frames)
    record = {
        "sample_id": sample_id,
        "task_id": TASK_ID,
        "env_id": ENV_ID,
        "split": success.split,
        "source_type": "perturbed_success_final_state",
        "seed": success.seed,
        "source_success_id": success.sample_id,
        "source_h5_path": str(success.h5_path),
        "video_path": str(video_path),
        "expected_success": False,
        "observed_success": bool(snapshot["success"]),
        "near_miss_type": "perturbed_cube_lateral_offset",
        "target_failure_mode": "cubeA is near target height but laterally offset from cubeB support",
        "lateral_offset_m": offset_m,
        "direction": direction_name,
        "num_frames": len(frames),
        "progress_rank_terminal": 2,
        **snapshot,
    }
    write_json(metadata_path, record)
    return record


def windows_path(path: Path) -> str:
    text = str(path)
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5].upper()
        rest = text[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return text


def build_manifest_row(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record)
    if "video_path" in row:
        row["video_path_wsl"] = row["video_path"]
        row["video_path_windows"] = windows_path(Path(row["video_path"]))
    if "source_h5_path" in row:
        row["source_h5_path_wsl"] = row["source_h5_path"]
        row["source_h5_path_windows"] = windows_path(Path(row["source_h5_path"]))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a StackCube bootstrap reward-model dataset.")
    parser.add_argument("--out", type=Path, default=Path("/mnt/e/reward_model_dataset/raw_rollouts/stackcube_bootstrap_v1"))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-success", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=40)
    parser.add_argument("--truncate-fractions", default="0.20,0.40,0.60,0.78,0.92")
    parser.add_argument("--offsets-m", default="0.035,0.055,0.075")
    parser.add_argument("--hold-frames", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    samples_dir = out_dir / "samples"
    reports_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    truncate_fractions = [float(x) for x in args.truncate_fractions.split(",") if x.strip()]
    offsets_m = [float(x) for x in args.offsets_m.split(",") if x.strip()]
    directions = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]

    success_records: list[SuccessRecord] = []
    attempts = 0
    seed = args.seed_start
    while len(success_records) < args.num_success and attempts < args.max_attempts:
        split = split_for_index(len(success_records), args.num_success)
        rec = run_success(seed=seed, sample_index=len(success_records), split=split, samples_dir=samples_dir)
        attempts += 1
        seed += 1
        if rec is not None:
            success_records.append(rec)

    rows: list[dict[str, Any]] = []
    for rec in success_records:
        success_row = {
            "sample_id": rec.sample_id,
            "task_id": TASK_ID,
            "env_id": ENV_ID,
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
        rows.append(build_manifest_row(success_row))

        for frac in truncate_fractions:
            rows.append(build_manifest_row(replay_truncated(rec, frac, samples_dir)))

        for offset_m in offsets_m:
            for direction in directions:
                rows.append(build_manifest_row(replay_perturbed(rec, offset_m, direction, samples_dir, args.hold_frames)))

    manifest_csv = out_dir / "trajectory_manifest.csv"
    manifest_json = out_dir / "trajectory_manifest.json"
    write_csv(manifest_csv, rows)
    write_json(manifest_json, rows)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_id": TASK_ID,
        "env_id": ENV_ID,
        "out_dir": str(out_dir),
        "num_success_requested": args.num_success,
        "num_success_saved": len(success_records),
        "attempts": attempts,
        "num_trajectories_total": len(rows),
        "truncate_fractions": truncate_fractions,
        "offsets_m": offsets_m,
        "directions": directions,
        "hold_frames": args.hold_frames,
    }
    write_json(out_dir / "dataset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved manifest: {manifest_csv}")


if __name__ == "__main__":
    main()
