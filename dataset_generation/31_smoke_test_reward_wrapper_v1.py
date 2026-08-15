"""Staged smoke tests for the frozen ManiSkill reward wrapper (Phase 5).

Stage 0 is torch-free. Stage 1 loads the physical checkpoint. Stage 2 runs a
real StackCube dry-run with fixed stereo SGBM and contact queries. Stage 3 is
optional and checks the complete OpenCLIP plus fusion-reward path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_wrapper_module():
    spec = importlib.util.spec_from_file_location(
        "maniskill_reward_wrapper_v1",
        SCRIPT_DIR / "30_maniskill_reward_wrapper_v1.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stage0_core(module) -> dict:
    potentials = iter([0.10, 0.40, 0.90, 0.90])

    def scorer(frames):  # noqa: ARG001
        return {
            "potential": next(potentials),
            "stage_probabilities": [0.7, 0.1, 0.1, 0.1],
            "confidence": 0.8,
            "depth_validity_ratio": 0.0,
            "contact_validity_ratio": 1.0,
        }

    gamma = 0.99
    core = module.RewardWrapperCoreV1(
        scorer,
        gamma=gamma,
        lambda_dense=1.0,
        dense_clip=0.5,
        history_window=4,
        inference_interval=2,
    )
    reset_log = core.reset({"gripper_width": 0.08})
    assert reset_log.potential == 0.10, reset_log
    logs = [
        core.step({"gripper_width": 0.08 - 0.01 * step}, sparse_reward=0.0)
        for step in range(1, 7)
    ]
    assert [log.scored for log in logs] == [False, True, False, True, False, True]
    assert abs(logs[0].dense_reward_raw - (gamma - 1.0) * 0.10) < 1e-9
    assert abs(logs[1].dense_reward_raw - (gamma * 0.40 - 0.10)) < 1e-9
    assert logs[4].dense_reward_raw <= 0.0, logs[4]
    assert core.history_length == 4

    dry_values = iter([0.2, 0.9])

    def dry_scorer(frames):  # noqa: ARG001
        return {"potential": next(dry_values)}

    dry = module.RewardWrapperCoreV1(dry_scorer, gamma=gamma, dry_run=True)
    dry.reset({"gripper_width": 0.05})
    dry_log = dry.step(
        {"gripper_width": 0.04}, sparse_reward=1.25, terminated=True
    )
    assert dry_log.total_reward == 1.25
    assert dry_log.dense_reward_raw != 0.0
    try:
        dry.step({"gripper_width": 0.03}, sparse_reward=0.0)
        raise AssertionError("step after terminal did not require reset")
    except RuntimeError:
        pass

    guard = module.RewardWrapperCoreV1(lambda frames: {"potential": 0.0})
    try:
        guard.reset({"env_success": 1.0})
        raise AssertionError("privileged key was not rejected")
    except ValueError:
        pass

    sampled = module.uniform_sample_history(
        [{"rgb": index} for index in range(16)], count=6
    )
    assert [frame["rgb"] for frame in sampled] == [0, 3, 6, 9, 12, 15]
    return {"status": "PASSED", "steps": len(logs)}


def stage1_physical(module, checkpoint: Path) -> dict:
    missing = [name for name in ("torch", "numpy") if not module_available(name)]
    if missing:
        return {"status": "SKIPPED", "missing_modules": missing}
    if not checkpoint.exists():
        return {"status": "SKIPPED", "missing_files": [str(checkpoint)]}
    scorer = module.FrozenPhysicalScorer(
        checkpoint, task_id="stackcube", device="cpu"
    )
    frames = [
        {"gripper_width": 0.08, "is_grasping_object": 0.0},
        {
            "gripper_width": 0.02,
            "is_grasping_object": 1.0,
            "finger_object_contact": 1.0,
        },
    ]
    result = scorer(frames)
    assert 0.0 <= result["potential"] <= 1.0
    assert len(result["stage_probabilities"]) == 4
    assert result["physical_observation_valid"] == 1.0
    return {
        "status": "PASSED",
        "potential": result["potential"],
        "depth_validity_ratio": result["depth_validity_ratio"],
        "contact_validity_ratio": result["contact_validity_ratio"],
    }


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


def make_online_adapter(module, scorer, emit_rgb: bool):
    allowed = scorer.allowed_frame_keys
    return module.CompositeAdapter(
        [
            module.ManiSkillStereoFeatureAdapterV1(
                "stackcube", allowed=allowed, emit_rgb=emit_rgb
            ),
            module.ManiSkillContactAdapterV1("stackcube", allowed=allowed),
            module.ActionHistoryAdapterV1(allowed=allowed),
        ],
        allowed=allowed,
    )


def run_dry_episode(module, env, scorer, adapter, max_steps: int) -> dict:
    wrapped = module.ManiSkillDenseRewardWrapper(
        env,
        scorer=scorer,
        observation_adapter=adapter,
        gamma=0.99,
        dry_run=True,
        inference_interval=4,
        allowed_frame_keys=scorer.allowed_frame_keys,
    )
    _, info = wrapped.reset(seed=0)
    logs = [info["dense_reward_log"]]
    mismatch_count = 0
    for _ in range(max_steps):
        action = wrapped.action_space.sample()
        _, reward, terminated, truncated, info = wrapped.step(action)
        log = info["dense_reward_log"]
        logs.append(log)
        if abs(float(reward) - log.sparse_reward) > 1e-8:
            mismatch_count += 1
        if module._single_bool(terminated, "terminated") or module._single_bool(
            truncated, "truncated"
        ):
            break
    wrapped.close()
    assert mismatch_count == 0, f"dry-run changed reward {mismatch_count} times"
    assert any(log.extra.get("contact_fields_emitted", 0) > 0 for log in logs)
    adapter_times = [
        float(log.extra.get("observation_adapter_time_ms", 0.0)) for log in logs
    ]
    steady_adapter_times = adapter_times[1:] or adapter_times
    return {
        "steps": len(logs) - 1,
        "dry_run_reward_mismatches": mismatch_count,
        "scored_steps": sum(log.scored for log in logs),
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
        "max_depth_fields_emitted": max(
            int(log.extra.get("depth_fields_emitted", 0)) for log in logs
        ),
        "mean_scoring_latency_ms": (
            sum(log.inference_time_ms for log in logs if log.scored)
            / max(1, sum(log.scored for log in logs))
        ),
        "mean_observation_adapter_latency_ms": (
            sum(adapter_times) / max(1, len(adapter_times))
        ),
        "steady_mean_observation_adapter_latency_ms": (
            sum(steady_adapter_times) / max(1, len(steady_adapter_times))
        ),
        "steady_median_observation_adapter_latency_ms": statistics.median(
            steady_adapter_times
        ),
    }


def stage2_maniskill(module, checkpoint: Path, max_steps: int) -> dict:
    required = ("torch", "numpy", "gymnasium", "mani_skill", "cv2")
    missing = [name for name in required if not module_available(name)]
    if missing:
        return {"status": "SKIPPED", "missing_modules": missing}
    if not checkpoint.exists():
        return {"status": "SKIPPED", "missing_files": [str(checkpoint)]}
    scorer = module.FrozenPhysicalScorer(
        checkpoint, task_id="stackcube", device="auto"
    )
    adapter = make_online_adapter(module, scorer, emit_rgb=False)
    result = run_dry_episode(
        module, make_stackcube_env(module), scorer, adapter, max_steps=max_steps
    )
    assert result["max_depth_fields_emitted"] > 0, result
    assert result["max_depth_validity_ratio"] > 0.0, result
    return {"status": "PASSED", **result}


def stage3_fusion(
    module,
    physical_checkpoint: Path,
    reward_checkpoint: Path | None,
    openclip_checkpoint: Path | None,
    task_goal_text: str,
    max_steps: int,
) -> dict:
    if reward_checkpoint is None or openclip_checkpoint is None:
        return {
            "status": "SKIPPED",
            "reason": "--reward-checkpoint and --openclip-checkpoint were not both provided",
        }
    missing_files = [
        str(path)
        for path in (physical_checkpoint, reward_checkpoint, openclip_checkpoint)
        if not path.exists()
    ]
    if missing_files:
        return {"status": "SKIPPED", "missing_files": missing_files}
    missing = [
        name
        for name in ("torch", "numpy", "gymnasium", "mani_skill", "cv2", "open_clip")
        if not module_available(name)
    ]
    if missing:
        return {"status": "SKIPPED", "missing_modules": missing}
    rgb_encoder = module.OpenCLIPHistoryEncoderV1(
        openclip_checkpoint,
        task_goal_text=task_goal_text,
        device="auto",
    )
    scorer = module.FrozenRewardModelScorer(
        physical_checkpoint,
        reward_checkpoint,
        task_id="stackcube",
        rgb_encoder=rgb_encoder,
        device="auto",
    )
    adapter = make_online_adapter(module, scorer, emit_rgb=True)
    result = run_dry_episode(
        module,
        make_stackcube_env(module),
        scorer,
        adapter,
        max_steps=max_steps,
    )
    return {"status": "PASSED", **result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SCRIPT_DIR.parent
        / "artifacts"
        / "physical_progress_branch_v1"
        / "best_model.pt",
    )
    parser.add_argument("--reward-checkpoint", type=Path, default=None)
    parser.add_argument("--openclip-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--task-goal-text",
        default="A robot arm stacks one cube on top of another cube and releases it stably.",
    )
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--fusion-max-steps", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module = load_wrapper_module()
    report: dict[str, dict] = {}
    failed = False

    stages = [
        ("stage0_core", lambda: stage0_core(module)),
        ("stage1_physical", lambda: stage1_physical(module, args.checkpoint)),
        (
            "stage2_maniskill_physical",
            lambda: stage2_maniskill(module, args.checkpoint, args.max_steps),
        ),
        (
            "stage3_maniskill_fusion",
            lambda: stage3_fusion(
                module,
                args.checkpoint,
                args.reward_checkpoint,
                args.openclip_checkpoint,
                args.task_goal_text,
                args.fusion_max_steps,
            ),
        ),
    ]
    for name, run in stages:
        try:
            report[name] = run()
        except Exception as exc:  # noqa: BLE001
            report[name] = {"status": "FAILED", "error": repr(exc)}
            failed = True
    print(json.dumps(report, indent=2, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
