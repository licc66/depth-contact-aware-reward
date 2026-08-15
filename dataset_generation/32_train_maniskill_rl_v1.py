"""Controlled ManiSkill SAC launcher for frozen reward-model conditions.

This script uses Stable-Baselines3 rather than a hand-written RL algorithm.
All conditions share the same sparse-reward ManiSkill environment, policy
state observation, SAC hyperparameters, seed, and environment-step budget.
Only the frozen dense-reward source changes.

The ``--smoke`` mode is an integration test, not a scientific result. In
particular, the physical checkpoint was trained from simulator-projected
geometry while the online adapter uses noisy SGBM plus segmentation. A proper
experiment must validate and, if needed, retrain/calibrate that sensor-domain
transition before interpreting policy performance.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


TASK_GOALS = {
    "stackcube": "A robot arm stacks one cube on top of another cube and releases it stably.",
    "stackpyramid": "A robot arm stacks the top cube onto two base cubes to form a stable pyramid.",
    "peginsertion": "A robot arm inserts the peg into the side hole of the box.",
}

REWARD_CHECKPOINTS = {
    "rgb_only": "reward_model_v1_rgb_only.pt",
    "physical_only": "reward_model_v1_physical_only.pt",
    "fusion": "reward_model_v1_fusion.pt",
}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_wrapper_module():
    spec = importlib.util.spec_from_file_location(
        "maniskill_reward_wrapper_v1_rl",
        SCRIPT_DIR / "30_maniskill_reward_wrapper_v1.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--env-id", default="StackCube-v1")
    parser.add_argument(
        "--task-id",
        default="stackcube",
        choices=sorted(TASK_GOALS),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["sparse_only", "rgb_only", "physical_only", "fusion"],
        choices=["sparse_only", "rgb_only", "physical_only", "fusion"],
    )
    parser.add_argument("--physical-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--reward-run-dir",
        type=Path,
        required=True,
        help="Script-28 output containing all reward_model_v1_*.pt files.",
    )
    parser.add_argument(
        "--openclip-checkpoint",
        type=Path,
        required=True,
        help="Local ViT-B-32.pt used by script 20; no weights are downloaded online.",
    )
    parser.add_argument("--task-goal-text", default=None)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-dense", type=float, default=1.0)
    parser.add_argument("--dense-clip", type=float, default=0.25)
    parser.add_argument("--inference-interval", type=int, default=4)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny integration budget and mark outputs non-scientific.",
    )
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.task_goal_text is None:
        args.task_goal_text = TASK_GOALS[args.task_id]
    if args.smoke:
        args.total_steps = min(args.total_steps, 64)
        args.eval_episodes = min(args.eval_episodes, 1)
        args.max_episode_steps = min(args.max_episode_steps, 64)
        args.learning_starts = min(args.learning_starts, 16)
        args.batch_size = min(args.batch_size, 16)
        args.buffer_size = min(args.buffer_size, 2_000)
    return args


def validate_inputs(args: argparse.Namespace) -> None:
    missing_modules = [
        name
        for name in (
            "torch",
            "numpy",
            "gymnasium",
            "mani_skill",
            "cv2",
            "stable_baselines3",
        )
        if not module_available(name)
    ]
    if any(condition in {"rgb_only", "fusion"} for condition in args.conditions):
        if not module_available("open_clip"):
            missing_modules.append("open_clip")
    if missing_modules:
        raise RuntimeError(
            "missing maintained RL/runtime dependencies: "
            + ", ".join(sorted(set(missing_modules)))
        )

    required_files = [args.physical_checkpoint]
    for condition in args.conditions:
        if condition in REWARD_CHECKPOINTS:
            required_files.append(args.reward_run_dir / REWARD_CHECKPOINTS[condition])
    if any(condition in {"rgb_only", "fusion"} for condition in args.conditions):
        required_files.append(args.openclip_checkpoint)
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError("missing experiment files: " + ", ".join(missing_files))


def make_raw_env(args: argparse.Namespace, wrapper_module):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        args.env_id,
        obs_mode="state",
        reward_mode="sparse",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend="cpu",
        render_backend="cpu",
        max_episode_steps=args.max_episode_steps,
        sensor_configs=wrapper_module.fixed_stereo_sensor_configs_v1(),
    )


def make_scorer(condition: str, args: argparse.Namespace, wrapper_module):
    if condition == "sparse_only":
        return lambda frames: {"potential": 0.0}

    rgb_encoder = None
    if condition in {"rgb_only", "fusion"}:
        rgb_encoder = wrapper_module.OpenCLIPHistoryEncoderV1(
            args.openclip_checkpoint,
            task_goal_text=args.task_goal_text,
            device=args.device,
        )
    scorer = wrapper_module.FrozenRewardModelScorer(
        physical_checkpoint=args.physical_checkpoint,
        reward_checkpoint=args.reward_run_dir / REWARD_CHECKPOINTS[condition],
        task_id=args.task_id,
        rgb_encoder=rgb_encoder,
        device=args.device,
    )
    if scorer.model.config.variant != condition:
        raise RuntimeError(
            f"condition {condition!r} loaded checkpoint variant "
            f"{scorer.model.config.variant!r}"
        )
    return scorer


def make_adapter(condition: str, scorer: Any, args: argparse.Namespace, wrapper_module):
    allowed = getattr(scorer, "allowed_frame_keys", None)
    if condition == "sparse_only":
        return wrapper_module.CompositeAdapter([], allowed=allowed)
    if condition == "rgb_only":
        return wrapper_module.CompositeAdapter(
            [
                wrapper_module.RGBObservationAdapterV1(allowed=allowed),
                wrapper_module.ActionHistoryAdapterV1(allowed=allowed),
            ],
            allowed=allowed,
        )
    return wrapper_module.CompositeAdapter(
        [
            wrapper_module.ManiSkillStereoFeatureAdapterV1(
                args.task_id,
                allowed=allowed,
                emit_rgb=condition == "fusion",
            ),
            wrapper_module.ManiSkillContactAdapterV1(
                args.task_id, allowed=allowed
            ),
            wrapper_module.ActionHistoryAdapterV1(allowed=allowed),
        ],
        allowed=allowed,
    )


def make_condition_env(
    condition: str,
    args: argparse.Namespace,
    wrapper_module,
    scorer: Any,
    monitor_path: Path | None = None,
):
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    env = CPUGymWrapper(make_raw_env(args, wrapper_module), record_metrics=True)
    adapter = make_adapter(condition, scorer, args, wrapper_module)
    env = wrapper_module.ManiSkillDenseRewardWrapper(
        env,
        scorer=scorer,
        observation_adapter=adapter,
        gamma=args.gamma,
        lambda_dense=args.lambda_dense,
        dense_clip=args.dense_clip,
        history_window=16,
        inference_interval=args.inference_interval,
        dry_run=condition == "sparse_only",
        allowed_frame_keys=getattr(scorer, "allowed_frame_keys", None),
    )
    if monitor_path is not None:
        from stable_baselines3.common.monitor import Monitor

        env = Monitor(env, filename=str(monitor_path), info_keywords=("success",))
    return env


def to_bool(value: Any) -> bool:
    import numpy as np

    array = np.asarray(value).reshape(-1)
    return bool(array[0]) if array.size else bool(value)


def evaluate_model(
    model: Any,
    condition: str,
    args: argparse.Namespace,
    wrapper_module,
    scorer: Any,
) -> dict[str, Any]:
    import numpy as np

    env = make_condition_env(condition, args, wrapper_module, scorer)
    episodes = []
    try:
        for episode in range(args.eval_episodes):
            obs, _ = env.reset(seed=args.seed + 10_000 + episode)
            done = False
            total_reward = 0.0
            sparse_return = 0.0
            dense_return = 0.0
            success = False
            first_success_step = None
            step = 0
            while not done and step < args.max_episode_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                total_reward += float(reward)
                log = info["dense_reward_log"]
                sparse_return += float(log.sparse_reward)
                dense_return += float(log.dense_reward_clipped)
                current_success = to_bool(info.get("success", False))
                if current_success and first_success_step is None:
                    first_success_step = step
                success = success or current_success
                done = to_bool(terminated) or to_bool(truncated)
            episodes.append(
                {
                    "episode": episode,
                    "success": success,
                    "steps": step,
                    "steps_to_success": first_success_step,
                    "total_return": total_reward,
                    "sparse_return": sparse_return,
                    "dense_clipped_sum": dense_return,
                }
            )
    finally:
        env.close()
    success_steps = [
        row["steps_to_success"]
        for row in episodes
        if row["steps_to_success"] is not None
    ]
    return {
        "episodes": episodes,
        "success_rate": float(np.mean([row["success"] for row in episodes])),
        "mean_return": float(np.mean([row["total_return"] for row in episodes])),
        "mean_sparse_return": float(
            np.mean([row["sparse_return"] for row in episodes])
        ),
        "mean_steps_to_success": (
            float(np.mean(success_steps)) if success_steps else None
        ),
    }


def train_condition(
    condition: str,
    args: argparse.Namespace,
    wrapper_module,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from stable_baselines3 import SAC

    run_dir = args.out_dir / condition
    run_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scorer = make_scorer(condition, args, wrapper_module)
    env = make_condition_env(
        condition,
        args,
        wrapper_module,
        scorer,
        monitor_path=run_dir / "monitor.csv",
    )
    model = SAC(
        "MlpPolicy",
        env,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [128, 128]},
        device=args.device,
        verbose=1,
    )
    started = datetime.now()
    try:
        model.learn(total_timesteps=args.total_steps, progress_bar=False)
        policy_path = run_dir / "sac_policy"
        model.save(policy_path)
    finally:
        env.close()
    elapsed = (datetime.now() - started).total_seconds()
    evaluation = evaluate_model(
        model, condition, args, wrapper_module, scorer
    )
    result = {
        "status": "COMPLETED",
        "condition": condition,
        "total_steps": args.total_steps,
        "elapsed_seconds": elapsed,
        "policy": str(run_dir / "sac_policy.zip"),
        "monitor": str(run_dir / "monitor.csv"),
        "evaluation": evaluation,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    del model, scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> int:
    args = resolve_args(parse_args())
    try:
        validate_inputs(args)
    except Exception as exc:  # noqa: BLE001
        print(f"DEPENDENCY/INPUT ERROR: {exc}", file=sys.stderr)
        return 3
    wrapper_module = load_wrapper_module()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    failed = False
    for condition in args.conditions:
        print(f"\n=== {condition} ===", flush=True)
        try:
            results[condition] = train_condition(condition, args, wrapper_module)
        except Exception as exc:  # noqa: BLE001
            results[condition] = {"status": "FAILED", "error": repr(exc)}
            failed = True
            print(f"{condition}: FAILED: {exc!r}", file=sys.stderr)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "env_id": args.env_id,
        "task_id": args.task_id,
        "conditions": args.conditions,
        "seed": args.seed,
        "total_steps_per_condition": args.total_steps,
        "eval_episodes": args.eval_episodes,
        "reward_mode": "sparse",
        "policy_observation": "ManiSkill state (identical for every condition)",
        "reward_model_frozen_for_entire_run": True,
        "commercial_vlm_called_online": False,
        "integration_smoke_only": bool(args.smoke),
        "scientific_result": not args.smoke,
        "sensor_domain_warning": (
            "The physical checkpoint was trained on simulator-projected geometry; "
            "the online adapter uses SGBM plus segmentation. Validate this domain "
            "transition before interpreting full RL comparisons. PegInsertion depth "
            "is missing until an observable hole locator is implemented."
        ),
        "results": results,
    }
    (args.out_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
