"""Frozen reward runtime and ManiSkill dense-reward wrapper (Phase 5).

The module keeps the online path deliberately small and auditable:

* observation, contact, stereo, RGB, and action handling live behind adapters;
* physical and reward checkpoints are loaded once and remain frozen;
* the commercial teacher is never called online;
* shaping uses ``gamma * Phi(s_next) - Phi(s)`` and supports a dry-run mode;
* privileged success flags, task evaluation outputs, poses, labels, and time
  proxies are rejected before they can reach either model.

The stereo adapter uses fixed rectified cameras, OpenCV StereoSGBM, and
renderer segmentation IDs. Segmentation is used only to locate visible task
entities; no object or goal pose is read. StackCube and StackPyramid have
observable support objects from which their goal point can be constructed
using known cube geometry. PegInsertion has no reliable hole locator in the
current sensor stack, so its depth fields remain missing instead of being
filled from simulator poses.
"""

from __future__ import annotations

import inspect
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reward_common_v1 import check_forbidden_feature  # noqa: E402

FrameDict = dict[str, Any]
ScorerResult = dict[str, Any]
Scorer = Callable[[list[FrameDict]], ScorerResult]


GEOMETRY_FEATURES = {
    "object_cam_x",
    "object_cam_y",
    "object_cam_depth_m",
    "goal_cam_x",
    "goal_cam_y",
    "goal_cam_depth_m",
    "support_cam_depth_m",
    "object_goal_3d_dist_m",
    "object_goal_xy_error_m",
    "object_goal_height_error_m",
    "delta_object_goal_dist_m",
    "object_moved_from_start_m",
    "object_goal_cam_lateral_error_m",
    "object_goal_cam_depth_error_m",
    "left_object_depth_m",
    "right_object_depth_m",
    "left_goal_depth_m",
    "object_pseudo_disparity_px",
    "goal_pseudo_disparity_px",
    "object_speed_proxy_m_per_step",
    "object_static_proxy",
}

CONTACT_FEATURES = {
    "is_grasping_object",
    "finger_object_contact_force_n",
    "finger_object_contact",
    "object_support_contact_force_n",
    "object_support_contacts",
    "released_object",
    "top_cube_cubeA_contact_force_n",
    "top_cube_cubeB_contact_force_n",
    "top_cube_contacts_both_base_cubes",
    "peg_box_contact_force_n",
    "left_finger_touch_object",
    "right_finger_touch_object",
    "both_fingers_touch_object",
    "left_finger_object_contact_force_n",
    "right_finger_object_contact_force_n",
    "gripper_width",
    "grasp_confirmed",
}


def _to_numpy(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _single_scalar(value: Any, name: str) -> float:
    array = _to_numpy(value).reshape(-1)
    if array.size != 1:
        raise ValueError(
            f"{name} must contain exactly one environment value; got shape "
            f"{tuple(_to_numpy(value).shape)}. Use a single-env wrapper."
        )
    return float(array[0])


def _single_bool(value: Any, name: str) -> bool:
    return bool(_single_scalar(value, name))


def validate_frame_keys(
    frame: FrameDict,
    allowed: set[str] | None,
    context: str = "observation adapter",
) -> None:
    for key in frame:
        reason = check_forbidden_feature(key)
        if reason:
            raise ValueError(
                f"{context} emitted privileged key {key!r} ({reason}); the online "
                "reward path must not use success/evaluate outputs, poses, time "
                "proxies, rules, or teacher labels"
            )
        if allowed is not None and key not in allowed:
            raise ValueError(
                f"{context} emitted key {key!r} outside the checkpoint feature "
                f"contract; declared keys start with {sorted(allowed)[:8]}"
            )


@dataclass
class StepLog:
    step: int
    scored: bool
    sparse_reward: float
    dense_reward_raw: float
    dense_reward_clipped: float
    total_reward: float
    potential: float
    previous_potential: float
    stage_probabilities: list[float] | None
    confidence: float | None
    depth_validity_ratio: float | None
    contact_validity_ratio: float | None
    inference_time_ms: float
    terminated: bool
    truncated: bool
    extra: dict[str, Any] = field(default_factory=dict)


class RewardWrapperCoreV1:
    """Torch-free history, cache, shaping, and logging state machine."""

    def __init__(
        self,
        scorer: Scorer,
        gamma: float = 0.99,
        lambda_dense: float = 1.0,
        dense_clip: float = 0.25,
        history_window: int = 16,
        inference_interval: int = 1,
        dry_run: bool = False,
        allowed_frame_keys: set[str] | None = None,
    ) -> None:
        if inference_interval < 1:
            raise ValueError("inference_interval must be >= 1")
        if history_window < 1:
            raise ValueError("history_window must be >= 1")
        if dense_clip < 0:
            raise ValueError("dense_clip must be non-negative")
        self.scorer = scorer
        self.gamma = float(gamma)
        self.lambda_dense = float(lambda_dense)
        self.dense_clip = float(dense_clip)
        self.history_window = int(history_window)
        self.inference_interval = int(inference_interval)
        self.dry_run = bool(dry_run)
        self.allowed_frame_keys = allowed_frame_keys

        self._history: deque[FrameDict] = deque(maxlen=self.history_window)
        self._step = 0
        self._steps_since_score = 0
        self._current_potential = 0.0
        self._current_result: ScorerResult = {}
        self._episode_logs: list[StepLog] = []
        self._in_episode = False

    @staticmethod
    def _result_extra(result: ScorerResult) -> dict[str, Any]:
        standard = {
            "potential",
            "stage_probabilities",
            "confidence",
            "depth_validity_ratio",
            "contact_validity_ratio",
        }
        return {key: value for key, value in result.items() if key not in standard}

    def _score_now(self) -> tuple[ScorerResult, float]:
        started = time.perf_counter()
        result = self.scorer(list(self._history))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if "potential" not in result:
            raise KeyError("scorer result must contain 'potential'")
        potential = float(result["potential"])
        if not math.isfinite(potential) or not 0.0 <= potential <= 1.0:
            raise ValueError(f"scorer returned invalid potential {potential}")
        return result, elapsed_ms

    def reset(self, initial_frame: FrameDict) -> StepLog:
        validate_frame_keys(initial_frame, self.allowed_frame_keys)
        self._history.clear()
        self._history.append(dict(initial_frame))
        self._step = 0
        self._steps_since_score = 0
        self._episode_logs = []
        result, elapsed = self._score_now()
        self._current_result = result
        self._current_potential = float(result["potential"])
        self._in_episode = True
        log = StepLog(
            step=0,
            scored=True,
            sparse_reward=0.0,
            dense_reward_raw=0.0,
            dense_reward_clipped=0.0,
            total_reward=0.0,
            potential=self._current_potential,
            previous_potential=self._current_potential,
            stage_probabilities=result.get("stage_probabilities"),
            confidence=result.get("confidence"),
            depth_validity_ratio=result.get("depth_validity_ratio"),
            contact_validity_ratio=result.get("contact_validity_ratio"),
            inference_time_ms=elapsed,
            terminated=False,
            truncated=False,
            extra=self._result_extra(result),
        )
        self._episode_logs.append(log)
        return log

    def step(
        self,
        frame: FrameDict,
        sparse_reward: float,
        terminated: bool = False,
        truncated: bool = False,
    ) -> StepLog:
        if not self._in_episode:
            raise RuntimeError("step() called before reset()")
        validate_frame_keys(frame, self.allowed_frame_keys)
        self._history.append(dict(frame))
        self._step += 1
        self._steps_since_score += 1

        previous_potential = self._current_potential
        scored = False
        elapsed = 0.0
        if self._steps_since_score >= self.inference_interval or terminated or truncated:
            result, elapsed = self._score_now()
            self._current_result = result
            self._current_potential = float(result["potential"])
            self._steps_since_score = 0
            scored = True

        dense_raw = self.gamma * self._current_potential - previous_potential
        dense_clipped = max(-self.dense_clip, min(self.dense_clip, dense_raw))
        total = (
            float(sparse_reward)
            if self.dry_run
            else float(sparse_reward) + self.lambda_dense * dense_clipped
        )
        log = StepLog(
            step=self._step,
            scored=scored,
            sparse_reward=float(sparse_reward),
            dense_reward_raw=dense_raw,
            dense_reward_clipped=dense_clipped,
            total_reward=total,
            potential=self._current_potential,
            previous_potential=previous_potential,
            stage_probabilities=self._current_result.get("stage_probabilities"),
            confidence=self._current_result.get("confidence"),
            depth_validity_ratio=self._current_result.get("depth_validity_ratio"),
            contact_validity_ratio=self._current_result.get("contact_validity_ratio"),
            inference_time_ms=elapsed,
            terminated=bool(terminated),
            truncated=bool(truncated),
            extra=self._result_extra(self._current_result),
        )
        self._episode_logs.append(log)
        if terminated or truncated:
            self._in_episode = False
        return log

    @property
    def episode_logs(self) -> list[StepLog]:
        return list(self._episode_logs)

    @property
    def history_length(self) -> int:
        return len(self._history)


class ObservationAdapter(Protocol):
    def reset(self) -> None: ...

    def build_frame(self, obs: Any, env: Any, action: Any | None = None) -> FrameDict: ...


def _call_extractor(extractor: Callable[..., Any], obs: Any, env: Any, action: Any):
    signature = inspect.signature(extractor)
    accepts_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if accepts_varargs or len(signature.parameters) >= 3:
        return extractor(obs, env, action)
    return extractor(obs, env)


class CallableFieldAdapter:
    """Build a frame from named observable field extractors."""

    def __init__(
        self,
        fields: dict[str, Callable[..., float | None]],
        allowed: set[str] | None = None,
        strict: bool = False,
    ) -> None:
        for name in fields:
            reason = check_forbidden_feature(name)
            if reason:
                raise ValueError(f"adapter field {name!r} is privileged ({reason})")
        self.fields = fields
        self.allowed = allowed
        self.strict = strict

    def reset(self) -> None:
        return None

    def build_frame(self, obs: Any, env: Any, action: Any | None = None) -> FrameDict:
        frame: FrameDict = {}
        for name, extractor in self.fields.items():
            try:
                value = _call_extractor(extractor, obs, env, action)
            except Exception:
                if self.strict:
                    raise
                value = None
            if value is not None:
                frame[name] = float(value)
        validate_frame_keys(frame, self.allowed, context="CallableFieldAdapter")
        return frame


class NullStereoAdapter:
    """Honest missing-depth adapter used when no sensor backend is configured."""

    diagnostics = {"stereo_status": "missing_backend"}

    def reset(self) -> None:
        return None

    def build_frame(self, obs: Any, env: Any, action: Any | None = None) -> FrameDict:  # noqa: ARG002
        return {}


class CompositeAdapter:
    def __init__(
        self,
        adapters: list[ObservationAdapter],
        allowed: set[str] | None = None,
    ) -> None:
        self.adapters = adapters
        self.allowed = allowed

    def reset(self) -> None:
        for adapter in self.adapters:
            reset = getattr(adapter, "reset", None)
            if callable(reset):
                reset()

    def build_frame(self, obs: Any, env: Any, action: Any | None = None) -> FrameDict:
        frame: FrameDict = {}
        for adapter in self.adapters:
            frame.update(adapter.build_frame(obs, env, action))
        validate_frame_keys(frame, self.allowed, context="CompositeAdapter")
        return frame

    @property
    def diagnostics(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for adapter in self.adapters:
            value = getattr(adapter, "diagnostics", None)
            if isinstance(value, dict):
                merged.update(value)
        return merged


class RGBObservationAdapterV1:
    """Extract RGB from a ManiSkill observation or scene sensor."""

    def __init__(self, camera_uid: str = "base_camera", allowed: set[str] | None = None):
        self.camera_uid = camera_uid
        self.allowed = allowed

    def reset(self) -> None:
        return None

    @staticmethod
    def _as_rgb(value: Any):
        import numpy as np

        array = _to_numpy(value)
        if array.ndim == 4:
            array = array[0]
        array = array[..., :3]
        if array.dtype != np.uint8:
            if np.nanmax(array) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def build_frame(self, obs: Any, env: Any, action: Any | None = None) -> FrameDict:  # noqa: ARG002
        try:
            rgb = obs["sensor_data"][self.camera_uid]["rgb"]
        except Exception:
            scene = env.unwrapped.scene
            scene.update_render(update_sensors=True, update_human_render_cameras=False)
            sensor = scene.sensors[self.camera_uid]
            sensor.capture()
            rgb = sensor.get_obs(
                rgb=True, depth=False, position=False, segmentation=False
            )["rgb"]
        frame = {"rgb": self._as_rgb(rgb)}
        validate_frame_keys(frame, self.allowed, context="RGBObservationAdapterV1")
        return frame


class ActionHistoryAdapterV1:
    """Replaceable action-history adapter.

    The current v1 physical checkpoint has no action fields, so this adapter
    keeps the history but emits only names present in ``allowed``. A future
    checkpoint can add ``action_0``, ``previous_action_0``, ``action_l2``, or
    ``action_delta_l2`` without changing the wrapper.
    """

    def __init__(self, history_window: int = 16, allowed: set[str] | None = None):
        self.history_window = int(history_window)
        self.allowed = allowed
        self._history: deque[Any] = deque(maxlen=self.history_window)

    def reset(self) -> None:
        self._history.clear()

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {"action_history_length": len(self._history)}

    def _put(self, frame: FrameDict, key: str, value: float) -> None:
        if self.allowed is None or key in self.allowed:
            frame[key] = float(value)

    def build_frame(self, obs: Any, env: Any, action: Any | None = None) -> FrameDict:  # noqa: ARG002
        import numpy as np

        if action is None:
            return {}
        current = _to_numpy(action).astype(np.float32).reshape(-1)
        previous = self._history[-1] if self._history else None
        self._history.append(current.copy())
        frame: FrameDict = {}
        for index, value in enumerate(current):
            self._put(frame, f"action_{index}", float(value))
        self._put(frame, "action_l2", float(np.linalg.norm(current)))
        if previous is not None and previous.shape == current.shape:
            for index, value in enumerate(previous):
                self._put(frame, f"previous_action_{index}", float(value))
            self._put(frame, "action_delta_l2", float(np.linalg.norm(current - previous)))
        validate_frame_keys(frame, self.allowed, context="ActionHistoryAdapterV1")
        return frame


class ManiSkillContactAdapterV1:
    """Task-aware contact sensor adapter with no pose/evaluate access."""

    OBJECT_ATTR = {
        "stackcube": "cubeA",
        "stackpyramid": "cubeC",
        "peginsertion": "peg",
    }

    def __init__(self, task_id: str, allowed: set[str] | None = None, threshold_n: float = 0.05):
        if task_id not in self.OBJECT_ATTR:
            raise ValueError(f"unsupported task_id {task_id!r}")
        self.task_id = task_id
        self.allowed = allowed
        self.threshold_n = float(threshold_n)
        self.diagnostics: dict[str, Any] = {}

    def reset(self) -> None:
        self.diagnostics = {}

    @staticmethod
    def _finger_links(ue: Any) -> tuple[Any | None, Any | None]:
        links = [link for link in ue.agent.robot.links if "finger" in link.name.lower()]
        left = next((link for link in links if "left" in link.name.lower()), None)
        right = next((link for link in links if "right" in link.name.lower()), None)
        remaining = [link for link in links if link is not left and link is not right]
        if left is None and remaining:
            left = remaining.pop(0)
        if right is None and remaining:
            right = remaining.pop(0)
        return left, right

    @staticmethod
    def _force_norm(ue: Any, actor_a: Any, actor_b: Any) -> float | None:
        import numpy as np

        if actor_a is None or actor_b is None:
            return None
        try:
            force = ue.scene.get_pairwise_contact_forces(actor_a, actor_b)
            value = float(np.linalg.norm(_to_numpy(force)))
            return value if math.isfinite(value) else None
        except Exception:
            return None

    @staticmethod
    def _gripper_width(ue: Any) -> float | None:
        try:
            qpos = _to_numpy(ue.agent.robot.get_qpos()).reshape(-1)
            joints = list(getattr(ue.agent.robot, "active_joints", []))
            values = [
                abs(float(qpos[index]))
                for index, joint in enumerate(joints)
                if index < len(qpos)
                and any(token in joint.name.lower() for token in ("finger", "gripper"))
            ]
            if len(values) >= 2:
                return float(sum(values[:2]))
            if len(values) == 1:
                return float(2.0 * values[0])
        except Exception:
            return None
        return None

    def _put(self, frame: FrameDict, key: str, value: Any) -> None:
        if value is None:
            return
        if self.allowed is None or key in self.allowed:
            frame[key] = float(value)

    def build_frame(self, obs: Any, env: Any, action: Any | None = None) -> FrameDict:  # noqa: ARG002
        ue = env.unwrapped
        obj = getattr(ue, self.OBJECT_ATTR[self.task_id])
        left_link, right_link = self._finger_links(ue)
        left_force = self._force_norm(ue, left_link, obj)
        right_force = self._force_norm(ue, right_link, obj)
        valid_finger_forces = [v for v in (left_force, right_force) if v is not None]
        total_finger_force = sum(valid_finger_forces) if valid_finger_forces else None
        left_touch = left_force is not None and left_force > self.threshold_n
        right_touch = right_force is not None and right_force > self.threshold_n
        try:
            is_grasping = _single_bool(ue.agent.is_grasping(obj), "is_grasping")
        except Exception:
            is_grasping = False
        width = self._gripper_width(ue)

        support_force: float | None
        frame: FrameDict = {}
        if self.task_id == "stackcube":
            support_force = self._force_norm(ue, ue.cubeA, ue.cubeB)
        elif self.task_id == "stackpyramid":
            force_a = self._force_norm(ue, ue.cubeC, ue.cubeA)
            force_b = self._force_norm(ue, ue.cubeC, ue.cubeB)
            support_force = (
                sum(v for v in (force_a, force_b) if v is not None)
                if force_a is not None or force_b is not None
                else None
            )
            self._put(frame, "top_cube_cubeA_contact_force_n", force_a)
            self._put(frame, "top_cube_cubeB_contact_force_n", force_b)
            if force_a is not None and force_b is not None:
                self._put(
                    frame,
                    "top_cube_contacts_both_base_cubes",
                    force_a > self.threshold_n and force_b > self.threshold_n,
                )
        else:
            support_force = self._force_norm(ue, ue.peg, ue.box)
            self._put(frame, "peg_box_contact_force_n", support_force)

        self._put(frame, "is_grasping_object", is_grasping)
        self._put(frame, "finger_object_contact_force_n", total_finger_force)
        if total_finger_force is not None:
            self._put(frame, "finger_object_contact", total_finger_force > self.threshold_n)
            self._put(
                frame,
                "released_object",
                (not is_grasping) and total_finger_force <= self.threshold_n,
            )
        self._put(frame, "left_finger_object_contact_force_n", left_force)
        self._put(frame, "right_finger_object_contact_force_n", right_force)
        if left_force is not None:
            self._put(frame, "left_finger_touch_object", left_touch)
        if right_force is not None:
            self._put(frame, "right_finger_touch_object", right_touch)
        if left_force is not None and right_force is not None:
            self._put(frame, "both_fingers_touch_object", left_touch and right_touch)
            self._put(frame, "grasp_confirmed", left_touch and right_touch and is_grasping)
        self._put(frame, "gripper_width", width)
        self._put(frame, "object_support_contact_force_n", support_force)
        if support_force is not None:
            self._put(frame, "object_support_contacts", support_force > self.threshold_n)

        self.diagnostics = {
            "contact_fields_emitted": len(frame),
            "contact_query_valid": bool(valid_finger_forces or support_force is not None),
        }
        validate_frame_keys(frame, self.allowed, context="ManiSkillContactAdapterV1")
        return frame


def default_gripper_contact_adapter(
    allowed: set[str] | None = None,
    task_id: str | None = None,
) -> ObservationAdapter:
    """Compatibility entry point; task-aware mode is preferred."""

    if task_id is not None:
        return ManiSkillContactAdapterV1(task_id=task_id, allowed=allowed)

    def gripper_width(obs: Any, env: Any, action: Any | None = None) -> float | None:  # noqa: ARG001
        try:
            qpos = _to_numpy(obs["agent"]["qpos"]).reshape(-1)
            return float(qpos[-1]) + float(qpos[-2])
        except Exception:
            return None

    return CallableFieldAdapter({"gripper_width": gripper_width}, allowed=allowed)


def fixed_stereo_sensor_configs_v1(
    width: int = 256,
    height: int = 256,
    baseline_m: float = 0.08,
    fov_y_rad: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Return two fixed, parallel ManiSkill camera configurations."""

    import numpy as np
    import sapien
    from mani_skill.utils import sapien_utils

    center_eye = np.array([0.55, -0.65, 0.42], dtype=np.float64)
    look_at = np.array([0.0, 0.0, 0.08], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_axis = look_at - center_eye
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, up)
    x_axis /= np.linalg.norm(x_axis)
    left_eye = center_eye - x_axis * (baseline_m / 2.0)
    right_eye = center_eye + x_axis * (baseline_m / 2.0)
    center_pose = sapien_utils.look_at(center_eye, look_at, up)
    quaternion = _to_numpy(center_pose.raw_pose).reshape(-1)[3:7]

    common = {
        "width": int(width),
        "height": int(height),
        "fov": float(fov_y_rad),
        "near": 0.01,
        "far": 10.0,
        "shader_pack": "minimal",
    }
    return {
        "base_camera": {"pose": sapien.Pose(left_eye, quaternion), **common},
        "hand_camera": {
            "pose": sapien.Pose(right_eye, quaternion),
            "mount": None,
            "entity_uid": None,
            **common,
        },
    }


class ManiSkillStereoFeatureAdapterV1:
    """SGBM plus segmentation-derived geometry for cube stacking tasks."""

    def __init__(
        self,
        task_id: str,
        allowed: set[str] | None = None,
        left_uid: str = "base_camera",
        right_uid: str = "hand_camera",
        baseline_m: float = 0.08,
        pseudo_focal_px: float = 520.0,
        num_disparities: int = 96,
        block_size: int = 5,
        min_mask_pixels: int = 8,
        emit_rgb: bool = False,
    ) -> None:
        if task_id not in {"stackcube", "stackpyramid", "peginsertion"}:
            raise ValueError(f"unsupported task_id {task_id!r}")
        self.task_id = task_id
        self.allowed = allowed
        self.left_uid = left_uid
        self.right_uid = right_uid
        self.baseline_m = float(baseline_m)
        self.pseudo_focal_px = float(pseudo_focal_px)
        self.num_disparities = int(math.ceil(num_disparities / 16.0) * 16)
        self.block_size = int(block_size)
        self.min_mask_pixels = int(min_mask_pixels)
        self.emit_rgb = bool(emit_rgb)
        self._matcher: Any = None
        self._initial_object_world: Any = None
        self._previous_object_world: Any = None
        self._previous_distance: float | None = None
        self.last_stereo_pair: tuple[Any, Any] | None = None
        self.diagnostics: dict[str, Any] = {}

    def reset(self) -> None:
        self._initial_object_world = None
        self._previous_object_world = None
        self._previous_distance = None
        self.last_stereo_pair = None
        self.diagnostics = {}

    def _make_matcher(self):
        import cv2

        if self._matcher is None:
            block = self.block_size
            self._matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=self.num_disparities,
                blockSize=block,
                P1=8 * block * block,
                P2=32 * block * block,
                disp12MaxDiff=1,
                preFilterCap=31,
                uniquenessRatio=6,
                speckleWindowSize=80,
                speckleRange=2,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )
        return self._matcher

    @staticmethod
    def _camera_obs(sensor: Any) -> dict[str, Any]:
        sensor.capture()
        obs = sensor.get_obs(
            rgb=True, depth=False, position=False, segmentation=True
        )
        params = sensor.get_params()
        rgb = _to_numpy(obs["rgb"])
        segmentation = _to_numpy(obs["segmentation"])
        intrinsic = _to_numpy(params["intrinsic_cv"])
        extrinsic = _to_numpy(params["extrinsic_cv"])
        if rgb.ndim == 4:
            rgb = rgb[0]
        if segmentation.ndim == 4:
            segmentation = segmentation[0]
        if intrinsic.ndim == 3:
            intrinsic = intrinsic[0]
        if extrinsic.ndim == 3:
            extrinsic = extrinsic[0]
        return {
            "rgb": rgb[..., :3],
            "segmentation": segmentation[..., 0],
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
        }

    @staticmethod
    def _actor_id(actor: Any) -> int:
        values = _to_numpy(actor.per_scene_id).reshape(-1)
        if values.size != 1:
            raise ValueError("stereo adapter supports one ManiSkill environment")
        return int(values[0])

    @staticmethod
    def _to_homogeneous(extrinsic: Any):
        import numpy as np

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :4] = np.asarray(extrinsic, dtype=np.float64)
        return matrix

    @classmethod
    def _cam_to_world(cls, point: Any, extrinsic: Any):
        import numpy as np

        homogeneous = np.append(np.asarray(point, dtype=np.float64), 1.0)
        return (np.linalg.inv(cls._to_homogeneous(extrinsic)) @ homogeneous)[:3]

    @classmethod
    def _world_to_cam(cls, point: Any, extrinsic: Any):
        import numpy as np

        homogeneous = np.append(np.asarray(point, dtype=np.float64), 1.0)
        return (cls._to_homogeneous(extrinsic) @ homogeneous)[:3]

    @classmethod
    def _center_extrinsic(cls, left: Any, right: Any):
        import numpy as np

        left4 = cls._to_homogeneous(left)
        right4 = cls._to_homogeneous(right)
        rotation = left4[:3, :3]
        left_center = -rotation.T @ left4[:3, 3]
        right_center = -rotation.T @ right4[:3, 3]
        center = 0.5 * (left_center + right_center)
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rotation
        result[:3, 3] = -rotation @ center
        return result[:3, :4]

    def _point_from_mask(self, disparity: Any, mask: Any, intrinsic: Any):
        import numpy as np

        yy, xx = np.nonzero(mask & np.isfinite(disparity) & (disparity > 0.0))
        if len(xx) < self.min_mask_pixels:
            return None
        disp = disparity[yy, xx].astype(np.float64)
        focal_x = float(intrinsic[0, 0])
        focal_y = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        depth = focal_x * self.baseline_m / disp
        valid = np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
        if int(valid.sum()) < self.min_mask_pixels:
            return None
        depth = depth[valid]
        xx = xx[valid]
        yy = yy[valid]
        x = (xx - cx) * depth / focal_x
        y = (yy - cy) * depth / focal_y
        return np.array(
            [np.median(x), np.median(y), np.median(depth)], dtype=np.float64
        )

    def _put(self, frame: FrameDict, key: str, value: Any) -> None:
        if value is None:
            return
        value = float(value)
        if not math.isfinite(value):
            return
        if self.allowed is None or key in self.allowed:
            frame[key] = value

    def _entity_actors(self, ue: Any):
        if self.task_id == "stackcube":
            return ue.cubeA, [ue.cubeB]
        if self.task_id == "stackpyramid":
            return ue.cubeC, [ue.cubeA, ue.cubeB]
        return None, []

    def build_frame(self, obs: Any, env: Any, action: Any | None = None) -> FrameDict:  # noqa: ARG002
        import cv2
        import numpy as np

        ue = env.unwrapped
        scene = ue.scene
        scene.update_render(update_sensors=True, update_human_render_cameras=False)
        left = self._camera_obs(scene.sensors[self.left_uid])
        right = self._camera_obs(scene.sensors[self.right_uid])
        self.last_stereo_pair = (left["rgb"].copy(), right["rgb"].copy())
        left_gray = cv2.cvtColor(left["rgb"], cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right["rgb"], cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        disparity = self._make_matcher().compute(
            clahe.apply(left_gray), clahe.apply(right_gray)
        ).astype(np.float32) / 16.0
        disparity[disparity <= 0.0] = np.nan

        frame: FrameDict = {}
        if self.emit_rgb:
            frame["rgb"] = left["rgb"].astype(np.uint8)
        object_actor, support_actors = self._entity_actors(ue)
        if object_actor is None:
            self.diagnostics = {
                "stereo_status": "missing_peg_hole_locator",
                "stereo_dense_valid_ratio": float(np.mean(np.isfinite(disparity))),
                "depth_fields_emitted": 0,
            }
            validate_frame_keys(frame, self.allowed, context="ManiSkillStereoFeatureAdapterV1")
            return frame

        object_mask = left["segmentation"] == self._actor_id(object_actor)
        object_left = self._point_from_mask(disparity, object_mask, left["intrinsic"])
        support_left_points = []
        for actor in support_actors:
            mask = left["segmentation"] == self._actor_id(actor)
            point = self._point_from_mask(disparity, mask, left["intrinsic"])
            if point is not None:
                support_left_points.append(point)
        if object_left is None or len(support_left_points) != len(support_actors):
            self.diagnostics = {
                "stereo_status": "entity_mask_or_disparity_missing",
                "stereo_dense_valid_ratio": float(np.mean(np.isfinite(disparity))),
                "object_mask_pixels": int(object_mask.sum()),
                "support_points_found": len(support_left_points),
                "depth_fields_emitted": 0,
            }
            validate_frame_keys(frame, self.allowed, context="ManiSkillStereoFeatureAdapterV1")
            return frame

        object_world = self._cam_to_world(object_left, left["extrinsic"])
        supports_world = [
            self._cam_to_world(point, left["extrinsic"])
            for point in support_left_points
        ]
        support_world = np.mean(np.stack(supports_world), axis=0)
        half_size = float(_to_numpy(ue.cube_half_size).reshape(-1)[0])
        goal_world = support_world + np.array([0.0, 0.0, 2.0 * half_size])
        center_extrinsic = self._center_extrinsic(
            left["extrinsic"], right["extrinsic"]
        )
        object_center = self._world_to_cam(object_world, center_extrinsic)
        goal_center = self._world_to_cam(goal_world, center_extrinsic)
        support_center = self._world_to_cam(support_world, center_extrinsic)
        object_right = self._world_to_cam(object_world, right["extrinsic"])
        goal_left = self._world_to_cam(goal_world, left["extrinsic"])

        world_delta = object_world - goal_world
        center_delta = object_center - goal_center
        distance = float(np.linalg.norm(world_delta))
        moved = (
            0.0
            if self._initial_object_world is None
            else float(np.linalg.norm(object_world - self._initial_object_world))
        )
        speed = (
            None
            if self._previous_object_world is None
            else float(np.linalg.norm(object_world - self._previous_object_world))
        )
        distance_delta = (
            None if self._previous_distance is None else self._previous_distance - distance
        )

        self._put(frame, "object_cam_x", object_center[0])
        self._put(frame, "object_cam_y", object_center[1])
        self._put(frame, "object_cam_depth_m", object_center[2])
        self._put(frame, "goal_cam_x", goal_center[0])
        self._put(frame, "goal_cam_y", goal_center[1])
        self._put(frame, "goal_cam_depth_m", goal_center[2])
        self._put(frame, "support_cam_depth_m", support_center[2])
        self._put(frame, "object_goal_3d_dist_m", distance)
        self._put(frame, "object_goal_xy_error_m", np.linalg.norm(world_delta[:2]))
        self._put(frame, "object_goal_height_error_m", abs(world_delta[2]))
        self._put(frame, "delta_object_goal_dist_m", distance_delta)
        self._put(frame, "object_moved_from_start_m", moved)
        self._put(frame, "object_goal_cam_lateral_error_m", np.linalg.norm(center_delta[:2]))
        self._put(frame, "object_goal_cam_depth_error_m", abs(center_delta[2]))
        self._put(frame, "left_object_depth_m", object_left[2])
        self._put(frame, "right_object_depth_m", object_right[2])
        self._put(frame, "left_goal_depth_m", goal_left[2])
        if object_center[2] > 0:
            self._put(
                frame,
                "object_pseudo_disparity_px",
                self.pseudo_focal_px * self.baseline_m / object_center[2],
            )
        if goal_center[2] > 0:
            self._put(
                frame,
                "goal_pseudo_disparity_px",
                self.pseudo_focal_px * self.baseline_m / goal_center[2],
            )
        self._put(frame, "object_speed_proxy_m_per_step", speed)
        if speed is not None:
            self._put(frame, "object_static_proxy", speed <= 0.003)

        if self._initial_object_world is None:
            self._initial_object_world = object_world.copy()
        self._previous_object_world = object_world.copy()
        self._previous_distance = distance
        self.diagnostics = {
            "stereo_status": "ok",
            "stereo_dense_valid_ratio": float(np.mean(np.isfinite(disparity))),
            "object_mask_pixels": int(object_mask.sum()),
            "depth_fields_emitted": sum(key in GEOMETRY_FEATURES for key in frame),
        }
        validate_frame_keys(frame, self.allowed, context="ManiSkillStereoFeatureAdapterV1")
        return frame


def uniform_sample_history(frames: list[FrameDict], count: int = 6) -> list[FrameDict]:
    """Uniformly select ``count`` RGB history frames, duplicating early frames."""

    import numpy as np

    rgb_frames = [frame for frame in frames if "rgb" in frame]
    if not rgb_frames:
        return []
    positions = np.linspace(0, len(rgb_frames) - 1, int(count)).round().astype(int)
    return [rgb_frames[int(index)] for index in positions]


class OpenCLIPHistoryEncoderV1:
    """Online encoder matching script 20 and script 28 exactly."""

    def __init__(
        self,
        checkpoint_path: Path,
        task_goal_text: str,
        model_name: str = "ViT-B-32",
        num_frames: int = 6,
        device: str = "auto",
    ) -> None:
        import open_clip
        import torch

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(checkpoint_path)
        resolved = "cuda" if device == "auto" and torch.cuda.is_available() else (
            device if device != "auto" else "cpu"
        )
        self.device = resolved
        self.num_frames = int(num_frames)
        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=str(checkpoint_path),
            device=resolved,
            weights_only=False,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        tokenizer = open_clip.get_tokenizer(model_name)
        with torch.inference_mode():
            text = self.model.encode_text(tokenizer([task_goal_text]).to(resolved)).float()
            self.text_feature = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def __call__(self, frames: list[FrameDict]):
        import numpy as np
        from PIL import Image

        selected = uniform_sample_history(frames, self.num_frames)
        if not selected:
            raise ValueError("OpenCLIPHistoryEncoderV1 received no RGB frames")
        images = [
            self.preprocess(Image.fromarray(np.asarray(frame["rgb"], dtype=np.uint8)).convert("RGB"))
            for frame in selected
        ]
        batch = self._torch.stack(images).to(self.device)
        with self._torch.inference_mode():
            image = self.model.encode_image(batch).float()
            image = image / image.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            image = image.mean(dim=0, keepdim=True)
            image = image / image.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            combined = self._torch.cat(
                [image, self.text_feature, image * self.text_feature], dim=-1
            )
        return combined[0].cpu().numpy().astype(np.float32)


class FrozenPhysicalScorer:
    def __init__(self, checkpoint_path: Path, task_id: str, device: str = "auto") -> None:
        from physical_progress_branch_v1 import PhysicalProgressRuntime

        self.runtime = PhysicalProgressRuntime.from_checkpoint(checkpoint_path, device=device)
        if task_id not in self.runtime.task_index:
            raise KeyError(f"task_id {task_id!r} not in checkpoint tasks")
        self.task_id = task_id
        self.allowed_frame_keys = set(self.runtime.feature_names)
        self.geometry_keys = sorted(self.allowed_frame_keys & GEOMETRY_FEATURES)
        self.contact_keys = sorted(self.allowed_frame_keys & CONTACT_FEATURES)

    @staticmethod
    def _ratio(frames: list[FrameDict], keys: list[str]) -> float:
        total = len(frames) * len(keys)
        if total == 0:
            return 0.0
        return sum(1 for frame in frames for key in keys if key in frame) / total

    def score(self, frames: list[FrameDict], return_embedding: bool = False) -> ScorerResult:
        result = self.runtime.score(
            self.task_id, frames, return_embedding=return_embedding
        )
        observed_count = sum(
            1 for frame in frames for key in self.allowed_frame_keys if key in frame
        )
        return {
            **result,
            "depth_validity_ratio": self._ratio(frames, self.geometry_keys),
            "contact_validity_ratio": self._ratio(frames, self.contact_keys),
            "physical_observation_valid": float(observed_count > 0),
            "physical_observed_feature_count": observed_count,
        }

    def __call__(self, frames: list[FrameDict]) -> ScorerResult:
        return self.score(frames, return_embedding=False)


class FrozenRewardModelScorer:
    """Frozen physical runtime plus one reward-model v1 checkpoint."""

    def __init__(
        self,
        physical_checkpoint: Path,
        reward_checkpoint: Path,
        task_id: str,
        rgb_encoder: Callable[[list[FrameDict]], Any] | None = None,
        device: str = "auto",
    ) -> None:
        import numpy as np
        import torch

        from reward_model_v1 import load_checkpoint

        self._np = np
        self._torch = torch
        self.physical = FrozenPhysicalScorer(physical_checkpoint, task_id, device=device)
        resolved = "cuda" if device == "auto" and torch.cuda.is_available() else (
            device if device != "auto" else "cpu"
        )
        self.device = resolved
        self.model, payload = load_checkpoint(reward_checkpoint, device=resolved)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        extra = payload.get("extra", {})
        self.physical_mean = np.asarray(extra.get("physical_mean", []), dtype=np.float32)
        self.physical_std = np.asarray(extra.get("physical_std", []), dtype=np.float32)
        self.physical_input_kind = extra.get("physical_input_kind", "summary")
        self.rgb_encoder = rgb_encoder
        self.task_id = task_id
        self.allowed_frame_keys = set(self.physical.allowed_frame_keys) | {"rgb"}

    def _physical_vector(self, physical: ScorerResult):
        np = self._np
        if self.physical_input_kind == "summary":
            task_onehot = [
                1.0 if self.task_id == task else 0.0
                for task in ("peginsertion", "stackcube", "stackpyramid")
            ]
            vector = np.asarray(
                list(physical["stage_probabilities"])
                + [
                    physical["local_progress"],
                    physical["potential"],
                    physical["confidence"],
                    physical["depth_validity_ratio"],
                    physical["contact_validity_ratio"],
                ]
                + task_onehot,
                dtype=np.float32,
            )
        elif self.physical_input_kind == "embedding":
            vector = np.asarray(physical["embedding"], dtype=np.float32)
        else:
            raise ValueError(f"unknown physical_input_kind {self.physical_input_kind!r}")
        if self.physical_mean.size:
            scale = np.where(self.physical_std > 1e-8, self.physical_std, 1.0)
            vector = (vector - self.physical_mean) / scale
        return vector

    def __call__(self, frames: list[FrameDict]) -> ScorerResult:
        np, torch = self._np, self._torch
        physical_frames = [
            {key: value for key, value in frame.items() if key != "rgb"}
            for frame in frames
        ]
        physical = self.physical.score(
            physical_frames,
            return_embedding=self.physical_input_kind == "embedding",
        )
        vector = self._physical_vector(physical)
        physical_tensor = torch.from_numpy(vector).float().unsqueeze(0).to(self.device)
        physical_valid_value = float(physical["physical_observation_valid"])
        physical_valid = torch.tensor([physical_valid_value], device=self.device)

        rgb_feature = None
        if self.rgb_encoder is not None and any("rgb" in frame for frame in frames):
            rgb_feature = np.asarray(self.rgb_encoder(frames), dtype=np.float32).reshape(-1)
            if rgb_feature.size != self.model.config.rgb_dim:
                raise ValueError(
                    f"RGB encoder returned {rgb_feature.size} values; checkpoint expects "
                    f"{self.model.config.rgb_dim}"
                )
        rgb_valid_value = float(rgb_feature is not None)
        rgb_valid = torch.tensor([rgb_valid_value], device=self.device)
        if rgb_feature is None:
            rgb_tensor = torch.zeros((1, self.model.config.rgb_dim), device=self.device)
        else:
            rgb_tensor = torch.from_numpy(rgb_feature).float().unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(
                rgb_tensor,
                physical_tensor,
                rgb_valid,
                physical_valid,
            )
        return {
            "potential": float(output["potential"].item()),
            "stage_probabilities": (
                output["stage_probs"][0].cpu().tolist()
                if "stage_probs" in output
                else physical["stage_probabilities"]
            ),
            "confidence": float(output["confidence"].item()),
            "depth_validity_ratio": physical["depth_validity_ratio"],
            "contact_validity_ratio": physical["contact_validity_ratio"],
            "physical_potential": physical["potential"],
            "gate_rgb_weight": float(output["gate_rgb_weight"].item()),
            "rgb_valid": rgb_valid_value,
            "physical_valid": physical_valid_value,
            "reward_model_variant": self.model.config.variant,
        }


try:  # pragma: no cover - exercised in the ManiSkill environment
    import gymnasium as _gym
except ImportError:  # pragma: no cover
    _gym = None


if _gym is not None:  # pragma: no cover

    class ManiSkillDenseRewardWrapper(_gym.Wrapper):
        """Add frozen potential-difference shaping to one ManiSkill env."""

        def __init__(
            self,
            env: Any,
            scorer: Scorer,
            observation_adapter: ObservationAdapter,
            gamma: float = 0.99,
            lambda_dense: float = 1.0,
            dense_clip: float = 0.25,
            history_window: int = 16,
            inference_interval: int = 1,
            dry_run: bool = False,
            allowed_frame_keys: set[str] | None = None,
        ) -> None:
            super().__init__(env)
            self.core = RewardWrapperCoreV1(
                scorer=scorer,
                gamma=gamma,
                lambda_dense=lambda_dense,
                dense_clip=dense_clip,
                history_window=history_window,
                inference_interval=inference_interval,
                dry_run=dry_run,
                allowed_frame_keys=allowed_frame_keys,
            )
            self.observation_adapter = observation_adapter

        def _attach_diagnostics(self, log: StepLog) -> None:
            diagnostics = getattr(self.observation_adapter, "diagnostics", None)
            if isinstance(diagnostics, dict):
                log.extra.update(diagnostics)

        def reset(self, **kwargs: Any):
            obs, info = self.env.reset(**kwargs)
            reset = getattr(self.observation_adapter, "reset", None)
            if callable(reset):
                reset()
            adapter_started = time.perf_counter()
            frame = self.observation_adapter.build_frame(obs, self.env, None)
            adapter_time_ms = (time.perf_counter() - adapter_started) * 1000.0
            log = self.core.reset(frame)
            self._attach_diagnostics(log)
            log.extra["observation_adapter_time_ms"] = adapter_time_ms
            info = dict(info)
            info["dense_reward_log"] = log
            return obs, info

        def step(self, action: Any):
            obs, reward, terminated, truncated, info = self.env.step(action)
            adapter_started = time.perf_counter()
            frame = self.observation_adapter.build_frame(obs, self.env, action)
            adapter_time_ms = (time.perf_counter() - adapter_started) * 1000.0
            log = self.core.step(
                frame,
                sparse_reward=_single_scalar(reward, "reward"),
                terminated=_single_bool(terminated, "terminated"),
                truncated=_single_bool(truncated, "truncated"),
            )
            self._attach_diagnostics(log)
            log.extra["observation_adapter_time_ms"] = adapter_time_ms
            info = dict(info)
            info["dense_reward_log"] = log
            return obs, log.total_reward, terminated, truncated, info
