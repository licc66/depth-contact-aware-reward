"""Controlled ManiSkill SAC launcher for sensor-consistent reward model v2.

The maintained SAC setup and evaluation protocol come from script 32. This
launcher switches only the frozen reward stack to physical/reward v2 and uses
scheduled stereo/RGB acquisition. Checkpoints whose metadata says
``primary_scientific_result=false`` remain audit-only in every output file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_script(
    SCRIPT_DIR / "32_train_maniskill_rl_v1.py", "maniskill_rl_v1_base_for_v2"
)
base.REWARD_CHECKPOINTS = {
    "rgb_only": "reward_model_v2_rgb_only.pt",
    "physical_only": "reward_model_v2_physical_only.pt",
    "fusion": "reward_model_v2_fusion.pt",
}


def load_wrapper_module():
    return load_script(
        SCRIPT_DIR / "43_maniskill_reward_wrapper_v2.py",
        "maniskill_reward_wrapper_v2_rl",
    )


def make_scorer(condition: str, args: Any, wrapper_module: Any):
    if condition == "sparse_only":
        return lambda frames: {"potential": 0.0}

    rgb_encoder = None
    if condition in {"rgb_only", "fusion"}:
        rgb_encoder = wrapper_module.OpenCLIPHistoryEncoderV1(
            args.openclip_checkpoint,
            task_goal_text=args.task_goal_text,
            device=args.device,
        )
    scorer = wrapper_module.FrozenRewardModelScorerV2(
        physical_checkpoint=args.physical_checkpoint,
        reward_checkpoint=args.reward_run_dir / base.REWARD_CHECKPOINTS[condition],
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


def make_adapter(condition: str, scorer: Any, args: Any, wrapper_module: Any):
    allowed = getattr(scorer, "allowed_frame_keys", None)
    if condition == "sparse_only":
        return wrapper_module.CompositeAdapter([], allowed=allowed)
    if condition == "rgb_only":
        sensor = wrapper_module.RGBObservationAdapterV1(allowed=allowed)
        fast = [wrapper_module.ActionHistoryAdapterV1(allowed=allowed)]
        if args.inference_interval == 1:
            return wrapper_module.CompositeAdapter([sensor, *fast], allowed=allowed)
        return wrapper_module.ScheduledCompositeAdapterV2(
            sensor,
            fast,
            sensor_interval=args.inference_interval,
            allowed=allowed,
        )
    return wrapper_module.make_stackcube_adapter(
        scorer,
        emit_rgb=condition == "fusion",
        sensor_interval=args.inference_interval,
    )


def make_condition_env(
    condition: str,
    args: Any,
    wrapper_module: Any,
    scorer: Any,
    monitor_path: Path | None = None,
):
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    env = CPUGymWrapper(base.make_raw_env(args, wrapper_module), record_metrics=True)
    adapter = make_adapter(condition, scorer, args, wrapper_module)
    env = wrapper_module.ManiSkillDenseRewardWrapperV2(
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


base.load_wrapper_module = load_wrapper_module
base.make_scorer = make_scorer
base.make_adapter = make_adapter
base.make_condition_env = make_condition_env


def checkpoint_metadata(args: Any) -> dict[str, dict[str, Any]]:
    import torch

    metadata: dict[str, dict[str, Any]] = {}
    for condition in args.conditions:
        if condition == "sparse_only":
            metadata[condition] = {
                "semantic_source": "not_applicable",
                "primary_scientific_result": True,
            }
            continue
        path = args.reward_run_dir / base.REWARD_CHECKPOINTS[condition]
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        extra = payload.get("extra", {})
        metadata[condition] = {
            "checkpoint": str(path),
            "semantic_source": extra.get("semantic_source", "unknown"),
            "primary_scientific_result": bool(
                extra.get("primary_scientific_result", False)
            ),
        }
    return metadata


def main() -> int:
    args = base.resolve_args(base.parse_args())
    try:
        base.validate_inputs(args)
    except Exception as exc:  # noqa: BLE001
        print(f"DEPENDENCY/INPUT ERROR: {exc}", file=sys.stderr)
        return 3

    wrapper_module = load_wrapper_module()
    metadata = checkpoint_metadata(args)
    audit_only = any(
        not item["primary_scientific_result"]
        for condition, item in metadata.items()
        if condition != "sparse_only"
    )
    if audit_only:
        print(
            "AUDIT-ONLY: at least one reward checkpoint lacks primary semantic labels.",
            file=sys.stderr,
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    failed = False
    for condition in args.conditions:
        print(f"\n=== {condition} ===", flush=True)
        try:
            result = base.train_condition(condition, args, wrapper_module)
            result["reward_checkpoint_metadata"] = metadata[condition]
            result["primary_scientific_result"] = bool(
                not args.smoke
                and metadata[condition]["primary_scientific_result"]
            )
            (args.out_dir / condition / "result.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            results[condition] = result
        except Exception as exc:  # noqa: BLE001
            results[condition] = {"status": "FAILED", "error": repr(exc)}
            failed = True
            print(f"{condition}: FAILED: {exc!r}", file=sys.stderr)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline_version": "stackcube_sensor_reward_v2",
        "env_id": args.env_id,
        "task_id": args.task_id,
        "conditions": args.conditions,
        "seed": args.seed,
        "total_steps_per_condition": args.total_steps,
        "eval_episodes": args.eval_episodes,
        "reward_mode": "sparse_plus_frozen_potential_shaping",
        "policy_observation": "ManiSkill state (identical for every condition)",
        "reward_model_frozen_for_entire_run": True,
        "commercial_vlm_called_online": False,
        "sensor_interval": args.inference_interval,
        "integration_smoke_only": bool(args.smoke),
        "audit_only": audit_only,
        "scientific_result": bool(not args.smoke and not audit_only),
        "reward_checkpoint_metadata": metadata,
        "results": results,
    }
    (args.out_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
