"""Sensor-consistent physical/reward v2 scorer for the frozen ManiSkill wrapper."""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _load_v1_wrapper():
    path = SCRIPT_DIR / "30_maniskill_reward_wrapper_v1.py"
    spec = importlib.util.spec_from_file_location("reward_wrapper_v1_frozen_for_v2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v1 = _load_v1_wrapper()

from physical_progress_branch_v2 import PhysicalProgressRuntimeV2  # noqa: E402
from reward_model_v2 import load_checkpoint  # noqa: E402
from stereo_feature_adapter_v2 import ManiSkillStereoFeatureAdapterV2  # noqa: E402


RewardWrapperCoreV1 = v1.RewardWrapperCoreV1
ManiSkillDenseRewardWrapperV1 = v1.ManiSkillDenseRewardWrapper
CompositeAdapter = v1.CompositeAdapter
ManiSkillContactAdapterV1 = v1.ManiSkillContactAdapterV1
ActionHistoryAdapterV1 = v1.ActionHistoryAdapterV1
RGBObservationAdapterV1 = v1.RGBObservationAdapterV1
OpenCLIPHistoryEncoderV1 = v1.OpenCLIPHistoryEncoderV1
fixed_stereo_sensor_configs_v1 = v1.fixed_stereo_sensor_configs_v1


def _observed(value: Any) -> bool:
    if value is None or str(value).strip() == "":
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return isinstance(value, bool)


def terminal_consistency(frame: dict[str, Any]) -> str:
    """Narrow completion check used only to suppress unsupported terminal peaks."""

    support = float(frame.get("object_support_contacts", 0.0)) > 0.5
    released = float(frame.get("released_object", 0.0)) > 0.5
    raw_distance = frame.get("object_goal_3d_dist_m")
    distance = float(raw_distance) if _observed(raw_distance) else float("nan")
    if not support or not released:
        return "fail"
    if math.isfinite(distance):
        return "pass" if distance <= 0.045 else "fail"
    return "unknown_depth"


def constrain_terminal_potential(
    potential: float,
    stage_probabilities: list[float],
    consistency: str,
    terminal_cap: float,
) -> tuple[float, bool]:
    completion_claim = potential >= 0.75 or stage_probabilities[3] >= 0.5
    if completion_claim and consistency == "fail":
        constrained = min(potential, terminal_cap)
        return constrained, constrained != potential
    return potential, False


class FrozenPhysicalScorerV2:
    def __init__(self, checkpoint: Path, task_id: str, device: str = "auto") -> None:
        self.runtime = PhysicalProgressRuntimeV2.from_checkpoint(checkpoint, device)
        if task_id not in self.runtime.task_index:
            raise KeyError(f"task_id {task_id!r} not in physical checkpoint")
        self.task_id = task_id
        self.allowed_frame_keys = set(self.runtime.depth_feature_names) | set(
            self.runtime.contact_feature_names
        )

    @staticmethod
    def _ratio(frames: list[dict[str, Any]], keys: list[str]) -> float:
        total = len(frames) * len(keys)
        if not total:
            return 0.0
        return sum(_observed(frame.get(key)) for frame in frames for key in keys) / total

    def score(
        self, frames: list[dict[str, Any]], include_embedding: bool = False
    ) -> dict[str, Any]:
        result = self.runtime.score(
            self.task_id, frames, include_embedding=include_embedding
        )
        observed_count = sum(
            _observed(frame.get(key))
            for frame in frames
            for key in self.allowed_frame_keys
        )
        return {
            **result,
            "depth_validity_ratio": self._ratio(
                frames, self.runtime.depth_feature_names
            ),
            "contact_validity_ratio": self._ratio(
                frames, self.runtime.contact_feature_names
            ),
            "physical_observation_valid": float(observed_count > 0),
            "physical_observed_feature_count": observed_count,
        }

    def __call__(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        return self.score(frames)


class FrozenRewardModelScorerV2:
    def __init__(
        self,
        physical_checkpoint: Path,
        reward_checkpoint: Path,
        task_id: str,
        rgb_encoder: Callable[[list[dict[str, Any]]], Any] | None = None,
        device: str = "auto",
        terminal_cap: float = 0.749,
    ) -> None:
        if not 0.0 <= terminal_cap < 0.75:
            raise ValueError("terminal_cap must be in [0, 0.75)")
        resolved = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else device if device != "auto" else "cpu"
        )
        self.device = resolved
        self.physical = FrozenPhysicalScorerV2(
            physical_checkpoint, task_id, device=resolved
        )
        self.model, payload = load_checkpoint(reward_checkpoint, device=resolved)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        extra = payload.get("extra", {})
        self.physical_mean = np.asarray(extra["physical_mean"], dtype=np.float32)
        self.physical_std = np.asarray(extra["physical_std"], dtype=np.float32)
        self.rgb_encoder = rgb_encoder
        self.task_id = task_id
        self.terminal_cap = float(terminal_cap)
        self.allowed_frame_keys = set(self.physical.allowed_frame_keys) | {"rgb"}
        self.semantic_source = extra.get("semantic_source", "unknown")
        self.primary_scientific_result = bool(
            extra.get("primary_scientific_result", False)
        )

    def __call__(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        if not frames:
            raise ValueError("reward scorer requires at least one frame")
        physical_frames = [
            {key: value for key, value in frame.items() if key != "rgb"}
            for frame in frames
        ]
        physical = self.physical.score(physical_frames, include_embedding=True)
        vector = np.asarray(physical["embedding"], dtype=np.float32)
        scale = np.where(self.physical_std > 1e-8, self.physical_std, 1.0)
        vector = (vector - self.physical_mean) / scale
        physical_tensor = torch.from_numpy(vector).float().unsqueeze(0).to(self.device)
        physical_valid_value = float(physical["physical_observation_valid"])
        physical_valid = torch.tensor([physical_valid_value], device=self.device)

        rgb_feature = None
        rgb_frames = [frame for frame in frames if "rgb" in frame]
        if self.rgb_encoder is not None and rgb_frames:
            rgb_feature = np.asarray(
                self.rgb_encoder(rgb_frames), dtype=np.float32
            ).reshape(-1)
            if rgb_feature.size != self.model.config.rgb_dim:
                raise ValueError(
                    f"RGB encoder returned {rgb_feature.size}; expected {self.model.config.rgb_dim}"
                )
        rgb_valid_value = float(rgb_feature is not None)
        rgb_valid = torch.tensor([rgb_valid_value], device=self.device)
        rgb_tensor = (
            torch.from_numpy(rgb_feature).float().unsqueeze(0).to(self.device)
            if rgb_feature is not None
            else torch.zeros((1, self.model.config.rgb_dim), device=self.device)
        )
        with torch.inference_mode():
            output = self.model(
                rgb_tensor, physical_tensor, rgb_valid, physical_valid
            )
        potential_raw = float(output["potential"].item())
        stage_probabilities = output["stage_probs"][0].cpu().tolist()
        consistency = terminal_consistency(physical_frames[-1])
        potential, cap_applied = constrain_terminal_potential(
            potential_raw,
            stage_probabilities,
            consistency,
            self.terminal_cap,
        )
        return {
            "potential": potential,
            "potential_raw": potential_raw,
            "stage_probabilities": stage_probabilities,
            "confidence": float(output["confidence"].item()),
            "depth_validity_ratio": physical["depth_validity_ratio"],
            "contact_validity_ratio": physical["contact_validity_ratio"],
            "physical_potential": physical["potential"],
            "physical_stage": physical["stage"],
            "gate_rgb_weight": float(output["gate_rgb_weight"].item()),
            "rgb_valid": rgb_valid_value,
            "physical_valid": physical_valid_value,
            "terminal_consistency": consistency,
            "terminal_cap_applied": cap_applied,
            "reward_model_variant": self.model.config.variant,
            "semantic_source": self.semantic_source,
            "primary_scientific_result": self.primary_scientific_result,
        }


class ScheduledCompositeAdapterV2:
    """Refresh expensive stereo/RGB features less often than contact features."""

    def __init__(
        self,
        sensor_adapter: Any,
        fast_adapters: list[Any],
        sensor_interval: int,
        allowed: set[str] | None = None,
    ) -> None:
        if sensor_interval < 1:
            raise ValueError("sensor_interval must be >= 1")
        self.sensor_adapter = sensor_adapter
        self.fast_adapters = list(fast_adapters)
        self.sensor_interval = int(sensor_interval)
        self.allowed = allowed
        self._steps_since_sensor = 0
        self._sensor_refreshed = False
        self._sensor_refresh_count = 0

    @property
    def adapters(self) -> list[Any]:
        return [self.sensor_adapter, *self.fast_adapters]

    def reset(self) -> None:
        for adapter in self.adapters:
            reset = getattr(adapter, "reset", None)
            if callable(reset):
                reset()
        self._steps_since_sensor = 0
        self._sensor_refreshed = False
        self._sensor_refresh_count = 0

    def build_frame(
        self,
        obs: Any,
        env: Any,
        action: Any | None = None,
        force_sensor: bool = False,
    ) -> dict[str, Any]:
        self._steps_since_sensor += 1
        refresh = force_sensor or self._steps_since_sensor >= self.sensor_interval
        frame: dict[str, Any] = {}
        if refresh:
            frame.update(self.sensor_adapter.build_frame(obs, env, action))
            self._steps_since_sensor = 0
            self._sensor_refresh_count += 1
        for adapter in self.fast_adapters:
            frame.update(adapter.build_frame(obs, env, action))
        self._sensor_refreshed = refresh
        v1.validate_frame_keys(
            frame, self.allowed, context="ScheduledCompositeAdapterV2"
        )
        return frame

    @property
    def diagnostics(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for adapter in self.adapters:
            value = getattr(adapter, "diagnostics", None)
            if isinstance(value, dict):
                merged.update(value)
        merged.update(
            {
                "sensor_refreshed": self._sensor_refreshed,
                "sensor_age_steps": self._steps_since_sensor,
                "sensor_interval": self.sensor_interval,
                "sensor_refresh_count": self._sensor_refresh_count,
            }
        )
        return merged


class ManiSkillDenseRewardWrapperV2(ManiSkillDenseRewardWrapperV1):
    """V1 shaping core with terminal-aware scheduled sensor acquisition."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        interval = getattr(self.observation_adapter, "sensor_interval", None)
        if interval is not None and self.core.inference_interval % interval != 0:
            raise ValueError(
                "sensor_interval must divide inference_interval so every score "
                "uses a fresh sensor frame"
            )

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        reset = getattr(self.observation_adapter, "reset", None)
        if callable(reset):
            reset()
        started = time.perf_counter()
        if isinstance(self.observation_adapter, ScheduledCompositeAdapterV2):
            frame = self.observation_adapter.build_frame(
                obs, self.env, None, force_sensor=True
            )
        else:
            frame = self.observation_adapter.build_frame(obs, self.env, None)
        adapter_time_ms = (time.perf_counter() - started) * 1000.0
        log = self.core.reset(frame)
        self._attach_diagnostics(log)
        log.extra["observation_adapter_time_ms"] = adapter_time_ms
        info = dict(info)
        info["dense_reward_log"] = log
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminal = v1._single_bool(terminated, "terminated")
        timeout = v1._single_bool(truncated, "truncated")
        started = time.perf_counter()
        if isinstance(self.observation_adapter, ScheduledCompositeAdapterV2):
            frame = self.observation_adapter.build_frame(
                obs,
                self.env,
                action,
                force_sensor=terminal or timeout,
            )
        else:
            frame = self.observation_adapter.build_frame(obs, self.env, action)
        adapter_time_ms = (time.perf_counter() - started) * 1000.0
        log = self.core.step(
            frame,
            sparse_reward=v1._single_scalar(reward, "reward"),
            terminated=terminal,
            truncated=timeout,
        )
        self._attach_diagnostics(log)
        log.extra["observation_adapter_time_ms"] = adapter_time_ms
        info = dict(info)
        info["dense_reward_log"] = log
        return obs, log.total_reward, terminated, truncated, info


def make_stackcube_adapter(
    scorer: FrozenRewardModelScorerV2,
    emit_rgb: bool,
    sensor_interval: int = 1,
) -> Any:
    allowed = scorer.allowed_frame_keys
    sensor = ManiSkillStereoFeatureAdapterV2(
        "stackcube", allowed=allowed, emit_rgb=emit_rgb
    )
    fast = [
            ManiSkillContactAdapterV1("stackcube", allowed=allowed),
            ActionHistoryAdapterV1(allowed=allowed),
    ]
    if sensor_interval == 1:
        return CompositeAdapter([sensor, *fast], allowed=allowed)
    return ScheduledCompositeAdapterV2(
        sensor,
        fast,
        sensor_interval=sensor_interval,
        allowed=allowed,
    )


__all__ = [
    "ActionHistoryAdapterV1",
    "CompositeAdapter",
    "FrozenPhysicalScorerV2",
    "FrozenRewardModelScorerV2",
    "ManiSkillContactAdapterV1",
    "ManiSkillDenseRewardWrapperV1",
    "ManiSkillDenseRewardWrapperV2",
    "ManiSkillStereoFeatureAdapterV2",
    "OpenCLIPHistoryEncoderV1",
    "RGBObservationAdapterV1",
    "RewardWrapperCoreV1",
    "ScheduledCompositeAdapterV2",
    "fixed_stereo_sensor_configs_v1",
    "constrain_terminal_potential",
    "make_stackcube_adapter",
    "terminal_consistency",
]
