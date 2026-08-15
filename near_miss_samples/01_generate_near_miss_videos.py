from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
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
from mani_skill.examples.motionplanning.panda.solutions import (
    solvePegInsertionSide,
    solveStackPyramid,
)
from mani_skill.utils.wrappers.record import RecordEpisode


SOLVERS = {
    "StackPyramid-v1": solveStackPyramid,
    "PegInsertionSide-v1": solvePegInsertionSide,
}


@dataclass(frozen=True)
class TruncatedSpec:
    sample_id: str
    task_id: str
    env_id: str
    h5_path: Path
    seed: int
    stop_step: int
    near_miss_type: str
    target_failure_mode: str
    notes: str


@dataclass(frozen=True)
class OfficialFailureSpec:
    sample_id: str
    task_id: str
    env_id: str
    seed: int
    near_miss_type: str
    target_failure_mode: str
    notes: str


@dataclass(frozen=True)
class PerturbedSpec:
    sample_id: str
    task_id: str
    env_id: str
    h5_path: Path
    seed: int
    lateral_offset_m: float
    hold_frames: int
    near_miss_type: str
    target_failure_mode: str
    notes: str


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


def read_actions(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f["traj_0/actions"], dtype=np.float32)


def env_success(info: dict[str, Any]) -> bool:
    return bool_scalar(info.get("success", False))


def make_env(env_id: str):
    return gym.make(
        env_id,
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
        raise RuntimeError(f"No frames to write for {path}")
    imageio.mimsave(path, frames, fps=fps, quality=8)


def replay_truncated(spec: TruncatedSpec, videos_dir: Path) -> dict[str, Any]:
    actions = read_actions(spec.h5_path)
    stop_step = min(spec.stop_step, len(actions))
    env = make_env(spec.env_id)
    frames: list[np.ndarray] = []
    info: dict[str, Any] = {}
    try:
        env.reset(seed=spec.seed)
        frames.append(render_frame(env))
        for action in actions[:stop_step]:
            _, _, _, _, info = env.step(action)
            frames.append(render_frame(env))
    finally:
        env.close()

    video_path = videos_dir / f"{spec.sample_id}.mp4"
    save_video(video_path, frames)
    return {
        "sample_id": spec.sample_id,
        "task_id": spec.task_id,
        "env_id": spec.env_id,
        "source_type": "truncated_success_trajectory",
        "seed": spec.seed,
        "video_path": str(video_path),
        "expected_success": False,
        "observed_success": env_success(info),
        "near_miss_type": spec.near_miss_type,
        "target_failure_mode": spec.target_failure_mode,
        "stop_step": stop_step,
        "num_frames": len(frames),
        "notes": spec.notes,
    }


def to_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def replay_and_perturb_stackcube(spec: PerturbedSpec, videos_dir: Path) -> dict[str, Any]:
    actions = read_actions(spec.h5_path)
    env = make_env(spec.env_id)
    frames: list[np.ndarray] = []
    info: dict[str, Any] = {}
    try:
        env.reset(seed=spec.seed)
        # Replay the successful trajectory first, but only keep the final
        # perturbed hold segment as the near-miss video.
        for action in actions:
            _, _, _, _, info = env.step(action)
        ue = env.unwrapped
        half = float(to_np(ue.cube_half_size).reshape(-1)[0])
        cube_b_pos = to_np(ue.cubeB.pose.p).reshape(-1)[:3]
        cube_a_quat = to_np(ue.cubeA.pose.q).reshape(-1)[:4]
        perturbed_pos = cube_b_pos + np.array([spec.lateral_offset_m, 0.0, 2.0 * half], dtype=np.float64)
        ue.cubeA.set_pose(sapien.Pose(perturbed_pos, cube_a_quat))
        info = ue.evaluate()
        for _ in range(spec.hold_frames):
            frames.append(render_frame(env))
    finally:
        env.close()

    video_path = videos_dir / f"{spec.sample_id}.mp4"
    save_video(video_path, frames)
    return {
        "sample_id": spec.sample_id,
        "task_id": spec.task_id,
        "env_id": spec.env_id,
        "source_type": "perturbed_success_final_state",
        "seed": spec.seed,
        "video_path": str(video_path),
        "expected_success": False,
        "observed_success": env_success(info),
        "near_miss_type": spec.near_miss_type,
        "target_failure_mode": spec.target_failure_mode,
        "stop_step": "",
        "num_frames": len(frames),
        "notes": spec.notes,
    }


def run_official_failure(spec: OfficialFailureSpec, out_dir: Path, videos_dir: Path) -> dict[str, Any]:
    tmp_dir = out_dir / "_raw_official_failures" / spec.sample_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(spec.env_id)
    env = RecordEpisode(
        env,
        output_dir=str(tmp_dir),
        trajectory_name=spec.sample_id,
        save_video=True,
        source_type="motionplanning_failure",
        source_desc="official ManiSkill Panda motion-planning failure saved as near-miss/failure sample",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )
    result: Any = None
    record: dict[str, Any]
    try:
        print(f"Running official failure {spec.sample_id} seed={spec.seed}")
        result = SOLVERS[spec.env_id](env, seed=spec.seed, debug=False, vis=False)
        if result == -1:
            record = {"success": False, "elapsed_steps": None, "reason": "motion_planner_failed"}
        else:
            final_info = result[-1]
            record = {key: scalar(value) for key, value in final_info.items()}
            record["success"] = bool_scalar(final_info.get("success", False))
            record["elapsed_steps"] = scalar(final_info.get("elapsed_steps"))
        env.flush_trajectory(save=True)
        env.flush_video(save=True)
    finally:
        env.close()

    candidates = sorted(tmp_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"No mp4 produced for {spec.sample_id}")
    src_video = candidates[-1]
    video_path = videos_dir / f"{spec.sample_id}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_video, video_path)

    return {
        "sample_id": spec.sample_id,
        "task_id": spec.task_id,
        "env_id": spec.env_id,
        "source_type": "official_motionplanning_failure",
        "seed": spec.seed,
        "video_path": str(video_path),
        "expected_success": False,
        "observed_success": bool(record.get("success", False)),
        "near_miss_type": spec.near_miss_type,
        "target_failure_mode": spec.target_failure_mode,
        "stop_step": "",
        "num_frames": "",
        "notes": spec.notes,
        "raw_record": json.dumps(record, ensure_ascii=False),
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate near-miss/failure videos for the three ManiSkill tasks.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("near_miss_samples") / "outputs")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out
    videos_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    base = root / "paper_style_tasks" / "outputs" / "wsl_motionplanning"
    trunc_specs = [
        TruncatedSpec(
            sample_id="SC-NM-01",
            task_id="stackcube",
            env_id="StackCube-v1",
            h5_path=base / "StackCube-v1" / "motionplanning" / "stackcube_wsl_mplib.h5",
            seed=0,
            stop_step=100,
            near_miss_type="truncated_place_before_stable_release",
            target_failure_mode="cube appears near/on support but task is not final stable success",
            notes="Truncate the successful StackCube trajectory just before it becomes static/successful.",
        ),
        TruncatedSpec(
            sample_id="SC-FAIL-01",
            task_id="stackcube",
            env_id="StackCube-v1",
            h5_path=base / "StackCube-v1" / "motionplanning" / "stackcube_wsl_mplib.h5",
            seed=0,
            stop_step=23,
            near_miss_type="early_grasp_or_lift_failure",
            target_failure_mode="cube is not stacked on cubeB",
            notes="Early truncated StackCube video; obvious non-completion control failure.",
        ),
        TruncatedSpec(
            sample_id="SP-NM-01",
            task_id="stackpyramid",
            env_id="StackPyramid-v1",
            h5_path=base / "StackPyramid-v1" / "motionplanning" / "stackpyramid_wsl_mplib.h5",
            seed=2,
            stop_step=141,
            near_miss_type="truncated_top_cube_place_before_stable_release",
            target_failure_mode="top cube appears close to pyramid but release/stability is not final",
            notes="Truncate successful StackPyramid trajectory around top-cube placement.",
        ),
        TruncatedSpec(
            sample_id="PEG-NM-03",
            task_id="peginsertion",
            env_id="PegInsertionSide-v1",
            h5_path=base / "PegInsertionSide-v1" / "motionplanning" / "peg_insertion_wsl_mplib.h5",
            seed=2,
            stop_step=142,
            near_miss_type="truncated_insert_before_release_stable",
            target_failure_mode="peg appears aligned/partly inserted but final release/stable condition is not reached",
            notes="Truncate successful PegInsertion trajectory before final stable release.",
        ),
    ]
    official_specs = [
        OfficialFailureSpec(
            sample_id="SP-FAIL-01",
            task_id="stackpyramid",
            env_id="StackPyramid-v1",
            seed=0,
            near_miss_type="official_motionplanning_failure",
            target_failure_mode="planner did not satisfy StackPyramid success",
            notes="Previously observed failed StackPyramid seed=0; saved now as a real planner failure video.",
        ),
        OfficialFailureSpec(
            sample_id="SP-FAIL-02",
            task_id="stackpyramid",
            env_id="StackPyramid-v1",
            seed=1,
            near_miss_type="official_motionplanning_failure",
            target_failure_mode="planner did not satisfy StackPyramid success",
            notes="Previously observed failed StackPyramid seed=1; saved now as a real planner failure video.",
        ),
        OfficialFailureSpec(
            sample_id="PEG-NM-01",
            task_id="peginsertion",
            env_id="PegInsertionSide-v1",
            seed=0,
            near_miss_type="official_depth_near_miss",
            target_failure_mode="peg roughly aligns in yz but remains far along insertion depth",
            notes="Previously observed failed PegInsertion seed=0 with about 10cm residual x insertion error.",
        ),
        OfficialFailureSpec(
            sample_id="PEG-NM-02",
            task_id="peginsertion",
            env_id="PegInsertionSide-v1",
            seed=1,
            near_miss_type="official_depth_near_miss",
            target_failure_mode="peg roughly aligns in yz but remains far along insertion depth",
            notes="Previously observed failed PegInsertion seed=1 with about 10cm residual x insertion error.",
        ),
    ]
    perturb_specs = [
        PerturbedSpec(
            sample_id="SC-NM-02",
            task_id="stackcube",
            env_id="StackCube-v1",
            h5_path=base / "StackCube-v1" / "motionplanning" / "stackcube_wsl_mplib.h5",
            seed=0,
            lateral_offset_m=0.055,
            hold_frames=48,
            near_miss_type="perturbed_cube_lateral_offset",
            target_failure_mode="cube appears near cubeB top but is laterally offset and not supported",
            notes="Replay success to the final scene, then move cubeA sideways by 5.5cm relative to cubeB top-center.",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for spec in trunc_specs:
        print(f"Rendering truncated near-miss {spec.sample_id}")
        rows.append(replay_truncated(spec, videos_dir))
    for spec in perturb_specs:
        print(f"Rendering perturbed near-miss {spec.sample_id}")
        rows.append(replay_and_perturb_stackcube(spec, videos_dir))
    for spec in official_specs:
        rows.append(run_official_failure(spec, out_dir, videos_dir))

    manifest_csv = out_dir / "failure_manifest.csv"
    manifest_json = out_dir / "failure_manifest.json"
    write_manifest(manifest_csv, rows)
    manifest_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest: {manifest_csv}")
    print(f"Saved videos to: {videos_dir}")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
