from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
from mani_skill.examples.motionplanning.panda.solutions import (
    solvePegInsertionSide,
    solvePickCube,
    solveStackCube,
    solveStackPyramid,
)
from mani_skill.utils.wrappers.record import RecordEpisode


SOLUTIONS = {
    "PickCube-v1": solvePickCube,
    "StackCube-v1": solveStackCube,
    "StackPyramid-v1": solveStackPyramid,
    "PegInsertionSide-v1": solvePegInsertionSide,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ManiSkill official Panda motion-planning demos in WSL2 with CPU rendering."
    )
    parser.add_argument("--env-id", default="StackCube-v1", choices=sorted(SOLUTIONS))
    parser.add_argument("--out-dir", default="paper_style_tasks/outputs/wsl_motionplanning")
    parser.add_argument("--name", default=None, help="Trajectory/video base name.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-traj", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--only-success", action="store_true")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--obs-mode", default="none")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--sim-backend", default="cpu")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--vis", action="store_true", help="Open live viewer if WSL GUI is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) / args.env_id / "motionplanning"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_name = args.name or f"{args.env_id.lower().replace('-v1', '')}_{time.strftime('%Y%m%d_%H%M%S')}"
    videos_before = set(out_dir.glob("*.mp4"))

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
        render_backend="cpu",
    )
    env = RecordEpisode(
        env,
        output_dir=str(out_dir),
        trajectory_name=traj_name,
        save_video=not args.no_video,
        source_type="motionplanning",
        source_desc="official ManiSkill Panda motion-planning solution, run in WSL2",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )

    solve = SOLUTIONS[args.env_id]
    successes: list[bool] = []
    attempts = 0
    saved = 0
    summary: list[dict[str, Any]] = []

    try:
        while saved < args.num_traj and attempts < args.max_attempts:
            seed = args.seed + attempts
            attempts += 1
            print(f"Running {args.env_id} seed={seed}")
            try:
                result = solve(env, seed=seed, debug=False, vis=args.vis)
            except Exception as exc:  # noqa: BLE001
                print(f"Motion planner error at seed={seed}: {exc}")
                result = -1

            if result == -1:
                record = {
                    "seed": seed,
                    "success": False,
                    "elapsed_steps": None,
                    "reason": "motion_planner_failed",
                }
            else:
                final_info = result[-1]
                record = {
                    "seed": seed,
                    "success": bool(scalar(final_info.get("success", False))),
                    "elapsed_steps": scalar(final_info.get("elapsed_steps")),
                }
                for key, value in final_info.items():
                    if key not in record:
                        record[key] = scalar(value)

            successes.append(record["success"])
            should_save = record["success"] or not args.only_success
            env.flush_trajectory(save=should_save)
            if not args.no_video:
                env.flush_video(save=should_save)

            record["saved"] = should_save
            summary.append(record)
            print(json.dumps(record, ensure_ascii=False))

            if should_save:
                saved += 1
    finally:
        env.close()

    new_videos = sorted(set(out_dir.glob("*.mp4")) - videos_before)
    renamed_videos: list[Path] = []
    for idx, path in enumerate(new_videos):
        suffix = "" if len(new_videos) == 1 else f"_{idx:02d}"
        target = out_dir / f"{traj_name}{suffix}.mp4"
        if path != target:
            if target.exists():
                target.unlink()
            path.replace(target)
        renamed_videos.append(target)

    videos = sorted(str(path) for path in out_dir.rglob("*.mp4"))
    h5_files = sorted(str(path) for path in out_dir.rglob("*.h5"))
    json_files = sorted(str(path) for path in out_dir.rglob("*.json"))
    run_summary = {
        "env_id": args.env_id,
        "attempts": attempts,
        "saved": saved,
        "success_rate_over_attempts": float(np.mean(successes)) if successes else 0.0,
        "records": summary,
        "new_videos": [str(path) for path in renamed_videos],
        "videos": videos,
        "trajectories": h5_files,
        "metadata": json_files,
    }
    summary_path = out_dir / f"{traj_name}_run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved run summary: {summary_path}")
    print("Videos:")
    for path in videos:
        print(f"  {path}")


if __name__ == "__main__":
    main()
