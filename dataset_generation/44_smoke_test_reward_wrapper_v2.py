"""Smoke-test the sensor-consistent v2 reward wrapper in real ManiSkill."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_wrapper_module():
    name = "maniskill_reward_wrapper_v2_smoke"
    spec = importlib.util.spec_from_file_location(
        name, SCRIPT_DIR / "43_maniskill_reward_wrapper_v2.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_stackcube_env(module):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        "StackCube-v1",
        obs_mode="state_dict",
        reward_mode="sparse",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend="cpu",
        render_backend="cpu",
        max_episode_steps=200,
        sensor_configs=module.fixed_stereo_sensor_configs_v1(),
    )


def run_dry_episode(
    module,
    scorer,
    emit_rgb: bool,
    max_steps: int,
    inference_interval: int,
    seed: int,
) -> dict[str, Any]:
    env = make_stackcube_env(module)
    adapter = module.make_stackcube_adapter(
        scorer, emit_rgb=emit_rgb, sensor_interval=inference_interval
    )
    wrapped = module.ManiSkillDenseRewardWrapperV2(
        env,
        scorer=scorer,
        observation_adapter=adapter,
        gamma=0.99,
        dry_run=True,
        inference_interval=inference_interval,
        allowed_frame_keys=scorer.allowed_frame_keys,
    )
    _, info = wrapped.reset(seed=seed)
    logs = [info["dense_reward_log"]]
    reward_mismatches = 0
    try:
        for _ in range(max_steps):
            action = wrapped.action_space.sample()
            _, reward, terminated, truncated, info = wrapped.step(action)
            log = info["dense_reward_log"]
            logs.append(log)
            if abs(float(reward) - log.sparse_reward) > 1e-8:
                reward_mismatches += 1
            if module.v1._single_bool(
                terminated, "terminated"
            ) or module.v1._single_bool(truncated, "truncated"):
                break
    finally:
        wrapped.close()

    if reward_mismatches:
        raise AssertionError(f"dry-run changed reward {reward_mismatches} times")
    if not any(log.extra.get("contact_fields_emitted", 0) > 0 for log in logs):
        raise AssertionError("contact adapter emitted no observable fields")

    adapter_times = [
        float(log.extra.get("observation_adapter_time_ms", 0.0)) for log in logs
    ]
    sensor_refreshed = [
        bool(log.extra.get("sensor_refreshed", True)) for log in logs
    ]
    refresh_times = [
        elapsed
        for elapsed, refreshed in zip(adapter_times, sensor_refreshed)
        if refreshed
    ]
    fast_only_times = [
        elapsed
        for elapsed, refreshed in zip(adapter_times, sensor_refreshed)
        if not refreshed
    ]
    scored = [log for log in logs if log.scored]
    return {
        "steps": len(logs) - 1,
        "dry_run_reward_mismatches": reward_mismatches,
        "scored_steps": len(scored),
        "potential_min": min(log.potential for log in logs),
        "potential_max": max(log.potential for log in logs),
        "max_depth_validity_ratio": max(
            float(log.depth_validity_ratio or 0.0) for log in logs
        ),
        "max_contact_validity_ratio": max(
            float(log.contact_validity_ratio or 0.0) for log in logs
        ),
        "stereo_statuses": sorted(
            {str(log.extra.get("stereo_status", "unknown")) for log in logs}
        ),
        "gripper_statuses": sorted(
            {str(log.extra.get("gripper_status", "unknown")) for log in logs}
        ),
        "max_depth_fields_emitted": max(
            int(log.extra.get("depth_fields_emitted", 0)) for log in logs
        ),
        "mean_scoring_latency_ms": statistics.fmean(
            log.inference_time_ms for log in scored
        ),
        "mean_observation_adapter_latency_ms": statistics.fmean(adapter_times),
        "median_observation_adapter_latency_ms": statistics.median(adapter_times),
        "sensor_refresh_steps": sum(sensor_refreshed),
        "mean_sensor_refresh_latency_ms": statistics.fmean(refresh_times),
        "mean_fast_only_latency_ms": (
            statistics.fmean(fast_only_times) if fast_only_times else None
        ),
        "terminal_cap_count": sum(
            bool(log.extra.get("terminal_cap_applied", False)) for log in logs
        ),
        "semantic_source": scored[-1].extra.get("semantic_source", "unknown"),
        "primary_scientific_result": bool(
            scored[-1].extra.get("primary_scientific_result", False)
        ),
    }


def make_reward_scorer(
    module,
    physical_checkpoint: Path,
    reward_checkpoint: Path,
    openclip_checkpoint: Path | None,
    task_goal_text: str,
    device: str,
):
    rgb_encoder = None
    if openclip_checkpoint is not None:
        rgb_encoder = module.OpenCLIPHistoryEncoderV1(
            openclip_checkpoint,
            task_goal_text=task_goal_text,
            device=device,
        )
    return module.FrozenRewardModelScorerV2(
        physical_checkpoint,
        reward_checkpoint,
        task_id="stackcube",
        rgb_encoder=rgb_encoder,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-checkpoint", type=Path, required=True)
    parser.add_argument("--physical-reward-checkpoint", type=Path, required=True)
    parser.add_argument("--fusion-reward-checkpoint", type=Path, default=None)
    parser.add_argument("--openclip-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--task-goal-text",
        default="A robot arm stacks one cube on top of another cube and releases it stably.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--fusion-max-steps", type=int, default=8)
    parser.add_argument("--inference-interval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module = load_wrapper_module()
    required = [args.physical_checkpoint, args.physical_reward_checkpoint]
    if args.fusion_reward_checkpoint is not None:
        required.append(args.fusion_reward_checkpoint)
    if args.openclip_checkpoint is not None:
        required.append(args.openclip_checkpoint)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")

    report: dict[str, Any] = {
        "dry_run": True,
        "inference_interval": args.inference_interval,
    }
    physical_scorer = make_reward_scorer(
        module,
        args.physical_checkpoint,
        args.physical_reward_checkpoint,
        None,
        args.task_goal_text,
        args.device,
    )
    report["physical_only"] = run_dry_episode(
        module,
        physical_scorer,
        emit_rgb=False,
        max_steps=args.max_steps,
        inference_interval=args.inference_interval,
        seed=args.seed,
    )

    if args.fusion_reward_checkpoint is not None and args.openclip_checkpoint is not None:
        fusion_scorer = make_reward_scorer(
            module,
            args.physical_checkpoint,
            args.fusion_reward_checkpoint,
            args.openclip_checkpoint,
            args.task_goal_text,
            args.device,
        )
        report["fusion"] = run_dry_episode(
            module,
            fusion_scorer,
            emit_rgb=True,
            max_steps=args.fusion_max_steps,
            inference_interval=args.inference_interval,
            seed=args.seed,
        )
    else:
        report["fusion"] = {
            "status": "SKIPPED",
            "reason": "fusion reward and OpenCLIP checkpoints are both required",
        }

    payload = json.dumps(report, indent=2, default=str)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
