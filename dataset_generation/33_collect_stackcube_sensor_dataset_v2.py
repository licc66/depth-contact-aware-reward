"""Collect StackCube v2 data through the exact online stereo/contact adapters.

Two tables are written and must remain separate:

* ``sensor_features.csv`` contains only online-observable SGBM geometry,
  simulator contact sensors, gripper state, actions, and non-model metadata.
* ``offline_supervision.csv`` contains simulator poses/evaluate outputs used
  only for offline labels and sensor-error auditing.

The collector replays the existing group-split-safe bootstrap trajectories.
Perturbed terminal samples can be replayed as physically settled rollouts
(default) or frozen snapshots. No privileged field is copied into the model
feature contract.

``--video-only`` replays an existing collection to add frame-aligned stereo
videos without recomputing or overwriting its sensor/supervision tables.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_wrapper_module():
    path = SCRIPT_DIR / "30_maniskill_reward_wrapper_v1.py"
    spec = importlib.util.spec_from_file_location("reward_wrapper_for_sensor_v2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wrapper = load_wrapper_module()


MODEL_METADATA_COLUMNS = [
    "meta_sample_id",
    "meta_task_id",
    "meta_split",
    "meta_source_type",
    "meta_near_miss_type",
    "meta_source_group_id",
    "meta_replay_step",
    "meta_saved_frame_index",
    "meta_action_source_step",
    "meta_perturb_mode",
]


def to_np(value: Any) -> np.ndarray:
    return wrapper._to_numpy(value)


def scalar(value: Any, default: float = float("nan")) -> float:
    try:
        array = to_np(value).reshape(-1)
        return float(array[0]) if array.size else default
    except Exception:
        return default


def boolean(value: Any) -> bool:
    try:
        array = to_np(value).reshape(-1)
        return bool(array[0]) if array.size else bool(value)
    except Exception:
        return bool(value)


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def resolve_h5_path(row: dict[str, str]) -> Path:
    keys = (
        "h5_path",
        "source_h5_path",
        "source_h5_path_wsl",
        "h5_path_wsl",
    )
    for key in keys:
        text = (row.get(key) or "").strip()
        if text and Path(text).exists():
            return Path(text)
    raise FileNotFoundError(
        f"no replay H5 exists for {row.get('sample_id')}: "
        + ", ".join(str(row.get(key, "")) for key in keys)
    )


def read_actions(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["traj_0/actions"], dtype=np.float32)


def source_group_id(row: dict[str, str]) -> str:
    return (row.get("source_success_id") or row.get("sample_id") or "").strip()


def select_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(args.manifest)
    if args.splits:
        allowed_splits = set(args.splits)
        rows = [row for row in rows if row.get("split") in allowed_splits]
    if args.source_types:
        allowed_sources = set(args.source_types)
        rows = [row for row in rows if row.get("source_type") in allowed_sources]
    if args.sample_ids:
        allowed_ids = set(args.sample_ids)
        rows = [row for row in rows if row.get("sample_id") in allowed_ids]
    rows.sort(key=lambda row: (row.get("split", ""), source_group_id(row), row.get("sample_id", "")))
    if args.max_trajectories <= 0 or len(rows) <= args.max_trajectories:
        return rows

    queues: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        queues[row.get("source_type", "unknown")].append(row)
    selected: list[dict[str, str]] = []
    source_names = sorted(queues)
    while len(selected) < args.max_trajectories and any(queues.values()):
        for name in source_names:
            if queues[name] and len(selected) < args.max_trajectories:
                selected.append(queues[name].pop(0))
    return selected


def make_env() -> gym.Env:
    return gym.make(
        "StackCube-v1",
        obs_mode="none",
        reward_mode="sparse",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend="cpu",
        render_backend="cpu",
        max_episode_steps=1000,
        sensor_configs=wrapper.fixed_stereo_sensor_configs_v1(),
    )


def actor_velocity(actor: Any, method_name: str) -> np.ndarray:
    try:
        method = getattr(actor, method_name)
        value = method() if callable(method) else method
        return to_np(value).reshape(-1)[:3].astype(np.float64)
    except Exception:
        return np.full(3, np.nan, dtype=np.float64)


def camera_params(ue: Any, uid: str) -> tuple[np.ndarray, np.ndarray]:
    params = ue.scene.sensors[uid].get_params()
    intrinsic = to_np(params["intrinsic_cv"])
    extrinsic = to_np(params["extrinsic_cv"])
    if intrinsic.ndim == 3:
        intrinsic = intrinsic[0]
    if extrinsic.ndim == 3:
        extrinsic = extrinsic[0]
    return intrinsic.astype(np.float64), extrinsic.astype(np.float64)


def offline_supervision(
    env: gym.Env,
    metadata: dict[str, Any],
    initial_object_world: np.ndarray | None,
    previous_object_world: np.ndarray | None,
    previous_distance: float | None,
) -> tuple[dict[str, Any], np.ndarray, float]:
    ue = env.unwrapped
    evaluation = ue.evaluate()
    cube_a = to_np(ue.cubeA.pose.p).reshape(-1)[:3].astype(np.float64)
    cube_b = to_np(ue.cubeB.pose.p).reshape(-1)[:3].astype(np.float64)
    half = float(to_np(ue.cube_half_size).reshape(-1)[0])
    goal = cube_b + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64)
    tcp = to_np(ue.agent.tcp.pose.p).reshape(-1)[:3].astype(np.float64)
    delta = cube_a - goal
    distance = float(np.linalg.norm(delta))
    moved = (
        0.0
        if initial_object_world is None
        else float(np.linalg.norm(cube_a - initial_object_world))
    )
    speed = (
        float("nan")
        if previous_object_world is None
        else float(np.linalg.norm(cube_a - previous_object_world))
    )
    distance_delta = (
        float("nan") if previous_distance is None else previous_distance - distance
    )
    left_intrinsic, left_extrinsic = camera_params(ue, "base_camera")
    _, right_extrinsic = camera_params(ue, "hand_camera")
    center_extrinsic = wrapper.ManiSkillStereoFeatureAdapterV1._center_extrinsic(
        left_extrinsic, right_extrinsic
    )
    object_center = wrapper.ManiSkillStereoFeatureAdapterV1._world_to_cam(
        cube_a, center_extrinsic
    )
    goal_center = wrapper.ManiSkillStereoFeatureAdapterV1._world_to_cam(
        goal, center_extrinsic
    )
    support_center = wrapper.ManiSkillStereoFeatureAdapterV1._world_to_cam(
        cube_b, center_extrinsic
    )
    object_left = wrapper.ManiSkillStereoFeatureAdapterV1._world_to_cam(
        cube_a, left_extrinsic
    )
    object_right = wrapper.ManiSkillStereoFeatureAdapterV1._world_to_cam(
        cube_a, right_extrinsic
    )
    goal_left = wrapper.ManiSkillStereoFeatureAdapterV1._world_to_cam(
        goal, left_extrinsic
    )
    camera_delta = object_center - goal_center
    try:
        is_grasping = boolean(ue.agent.is_grasping(ue.cubeA))
    except Exception:
        is_grasping = False
    success = boolean(evaluation.get("success", False))
    on_support = boolean(evaluation.get("is_cubeA_on_cubeB", False))
    is_static = boolean(evaluation.get("is_cubeA_static", False))
    if success:
        stage_candidate = 4
    elif on_support or (distance <= 0.035 and not is_grasping):
        stage_candidate = 3
    elif is_grasping or moved > 0.02:
        stage_candidate = 2
    else:
        stage_candidate = 1

    row = {
        **metadata,
        "gt_success": int(success),
        "gt_is_cubeA_on_cubeB": int(on_support),
        "gt_is_cubeA_static": int(is_static),
        "gt_is_cubeA_grasped": int(
            boolean(evaluation.get("is_cubeA_grasped", is_grasping))
        ),
        "gt_stage_candidate": stage_candidate,
        "gt_object_x": cube_a[0],
        "gt_object_y": cube_a[1],
        "gt_object_z": cube_a[2],
        "gt_goal_x": goal[0],
        "gt_goal_y": goal[1],
        "gt_goal_z": goal[2],
        "gt_tcp_x": tcp[0],
        "gt_tcp_y": tcp[1],
        "gt_tcp_z": tcp[2],
        "gt_tcp_object_dist_m": float(np.linalg.norm(tcp - cube_a)),
        "gt_object_goal_3d_dist_m": distance,
        "gt_object_goal_xy_error_m": float(np.linalg.norm(delta[:2])),
        "gt_object_goal_height_error_m": abs(float(delta[2])),
        "gt_delta_object_goal_dist_m": distance_delta,
        "gt_object_moved_from_start_m": moved,
        "gt_object_speed_proxy_m_per_step": speed,
        "gt_object_static_proxy": int(math.isfinite(speed) and speed <= 0.003),
        "gt_object_cam_x": object_center[0],
        "gt_object_cam_y": object_center[1],
        "gt_object_cam_depth_m": object_center[2],
        "gt_goal_cam_x": goal_center[0],
        "gt_goal_cam_y": goal_center[1],
        "gt_goal_cam_depth_m": goal_center[2],
        "gt_support_cam_depth_m": support_center[2],
        "gt_object_goal_cam_lateral_error_m": float(np.linalg.norm(camera_delta[:2])),
        "gt_object_goal_cam_depth_error_m": abs(float(camera_delta[2])),
        "gt_left_object_depth_m": object_left[2],
        "gt_right_object_depth_m": object_right[2],
        "gt_left_goal_depth_m": goal_left[2],
        "gt_object_pseudo_disparity_px": 520.0 * 0.08 / max(object_center[2], 1e-8),
        "gt_goal_pseudo_disparity_px": 520.0 * 0.08 / max(goal_center[2], 1e-8),
        "gt_object_linear_velocity_x": actor_velocity(ue.cubeA, "get_linear_velocity")[0],
        "gt_object_linear_velocity_y": actor_velocity(ue.cubeA, "get_linear_velocity")[1],
        "gt_object_linear_velocity_z": actor_velocity(ue.cubeA, "get_linear_velocity")[2],
        "gt_left_camera_focal_px": left_intrinsic[0, 0],
    }
    return row, cube_a.copy(), distance


def zero_actor_velocity(actor: Any) -> None:
    zeros = np.zeros(3, dtype=np.float32)
    for method_name in ("set_linear_velocity", "set_angular_velocity"):
        try:
            getattr(actor, method_name)(zeros)
        except Exception:
            pass


def metadata_columns(
    row: dict[str, str],
    replay_step: int,
    saved_frame_index: int,
    action_source_step: int,
    perturb_mode: str,
) -> dict[str, Any]:
    return {
        "meta_sample_id": row["sample_id"],
        "meta_task_id": "stackcube",
        "meta_split": row.get("split", ""),
        "meta_source_type": row.get("source_type", ""),
        "meta_near_miss_type": row.get("near_miss_type", ""),
        "meta_source_group_id": source_group_id(row),
        "meta_replay_step": replay_step,
        "meta_saved_frame_index": saved_frame_index,
        "meta_action_source_step": action_source_step,
        "meta_perturb_mode": perturb_mode,
    }


def add_action_columns(sensor_row: dict[str, Any], action: Any | None) -> list[str]:
    if action is None:
        return []
    values = to_np(action).astype(np.float32).reshape(-1)
    names = []
    for index, value in enumerate(values):
        name = f"action_{index}"
        sensor_row[name] = float(value)
        names.append(name)
    sensor_row["action_l2"] = float(np.linalg.norm(values))
    names.append("action_l2")
    return names


def save_preview(sample_dir: Path, left_frames: list[np.ndarray], right_frames: list[np.ndarray]) -> None:
    if not left_frames or not right_frames:
        return
    preview_dir = sample_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(preview_dir / "left_first.png", left_frames[0])
    imageio.imwrite(preview_dir / "left_last.png", left_frames[-1])
    imageio.imwrite(preview_dir / "right_first.png", right_frames[0])
    imageio.imwrite(preview_dir / "right_last.png", right_frames[-1])


def save_videos(sample_dir: Path, left_frames: list[np.ndarray], right_frames: list[np.ndarray], fps: int) -> None:
    imageio.mimwrite(sample_dir / "left_stereo.mp4", left_frames, fps=fps, codec="libx264", quality=8)
    imageio.mimwrite(sample_dir / "right_stereo.mp4", right_frames, fps=fps, codec="libx264", quality=8)


def capture_stereo_rgb(stereo: Any, env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Capture the configured stereo pair without running disparity inference."""

    scene = env.unwrapped.scene
    scene.update_render(update_sensors=True, update_human_render_cameras=False)
    left = stereo._camera_obs(scene.sensors[stereo.left_uid])["rgb"]
    right = stereo._camera_obs(scene.sensors[stereo.right_uid])["rgb"]
    return (
        np.asarray(left, dtype=np.uint8).copy(),
        np.asarray(right, dtype=np.uint8).copy(),
    )


def collect_sample(
    args: argparse.Namespace,
    row: dict[str, str],
    physical_feature_names: list[str],
) -> dict[str, Any]:
    sample_dir = args.out_dir / "samples" / row.get("split", "unknown") / row["sample_id"]
    completion_path = sample_dir / "collection_metadata.json"
    existing_payload: dict[str, Any] = {}
    if completion_path.exists():
        existing_payload = json.loads(completion_path.read_text(encoding="utf-8"))
        videos_exist = all(
            (sample_dir / name).exists()
            for name in ("left_stereo.mp4", "right_stereo.mp4")
        )
        if not args.force and (not args.video_only or videos_exist):
            payload = dict(existing_payload)
            payload["status"] = "SKIPPED_EXISTING"
            return payload
    elif args.video_only:
        raise FileNotFoundError(
            f"--video-only requires an existing collection: {completion_path}"
        )

    sample_dir.mkdir(parents=True, exist_ok=True)
    actions = read_actions(resolve_h5_path(row))
    env = make_env()
    allowed = set(physical_feature_names) | {"rgb"}
    if args.include_gripper_geometry:
        from stereo_feature_adapter_v2 import ManiSkillStereoFeatureAdapterV2

        stereo = ManiSkillStereoFeatureAdapterV2(
            "stackcube", allowed=allowed, emit_rgb=True
        )
    else:
        stereo = wrapper.ManiSkillStereoFeatureAdapterV1(
            "stackcube", allowed=allowed, emit_rgb=True
        )
    contact = wrapper.ManiSkillContactAdapterV1("stackcube", allowed=allowed)
    action_history = wrapper.ActionHistoryAdapterV1(allowed=allowed)
    adapter = wrapper.CompositeAdapter(
        [stereo, contact, action_history], allowed=allowed
    )

    sensor_rows: list[dict[str, Any]] = []
    supervision_rows: list[dict[str, Any]] = []
    left_frames: list[np.ndarray] = []
    right_frames: list[np.ndarray] = []
    initial_object_world: np.ndarray | None = None
    previous_object_world: np.ndarray | None = None
    previous_distance: float | None = None
    action_names: set[str] = set()
    started = time.perf_counter()

    def observe(
        action: Any | None,
        replay_step: int,
        action_source_step: int,
        contact_valid: bool,
    ) -> None:
        nonlocal initial_object_world, previous_object_world, previous_distance
        if args.video_only:
            if replay_step % args.frame_stride == 0:
                left, right = capture_stereo_rgb(stereo, env)
                left_frames.append(left)
                right_frames.append(right)
            return
        adapter_started = time.perf_counter()
        frame = adapter.build_frame(None, env, action)
        adapter_ms = (time.perf_counter() - adapter_started) * 1000.0
        rgb = frame.pop("rgb", None)
        if not contact_valid:
            for name in wrapper.CONTACT_FEATURES:
                frame.pop(name, None)
        if replay_step % args.frame_stride != 0:
            offline_row, current_object, current_distance = offline_supervision(
                env,
                {},
                initial_object_world,
                previous_object_world,
                previous_distance,
            )
            if initial_object_world is None:
                initial_object_world = current_object.copy()
            previous_object_world = current_object
            previous_distance = current_distance
            return

        meta = metadata_columns(
            row,
            replay_step=replay_step,
            saved_frame_index=len(sensor_rows),
            action_source_step=action_source_step,
            perturb_mode=args.perturb_mode if row.get("source_type") == "perturbed_success_final_state" else "none",
        )
        sensor_row = {**meta, **frame}
        action_names.update(add_action_columns(sensor_row, action))
        sensor_row.update(
            {
                "sensor_adapter_time_ms": adapter_ms,
                "sensor_stereo_status": stereo.diagnostics.get("stereo_status", "unknown"),
                "sensor_stereo_dense_valid_ratio": stereo.diagnostics.get(
                    "stereo_dense_valid_ratio", float("nan")
                ),
                "sensor_object_mask_pixels": stereo.diagnostics.get(
                    "object_mask_pixels", 0
                ),
                "sensor_gripper_status": stereo.diagnostics.get(
                    "gripper_status", "not_requested"
                ),
                "sensor_gripper_mask_pixels": stereo.diagnostics.get(
                    "gripper_mask_pixels", 0
                ),
                "sensor_depth_fields_emitted": stereo.diagnostics.get(
                    "depth_fields_emitted", 0
                ),
                "sensor_contact_fields_emitted": contact.diagnostics.get(
                    "contact_fields_emitted", 0
                )
                if contact_valid
                else 0,
                "sensor_contact_observation_valid": int(contact_valid),
            }
        )
        offline_row, current_object, current_distance = offline_supervision(
            env,
            meta,
            initial_object_world,
            previous_object_world,
            previous_distance,
        )
        offline_row["label_expected_success_from_manifest"] = row.get(
            "expected_success", ""
        )
        offline_row["label_terminal_rank_from_manifest"] = row.get(
            "progress_rank_terminal", ""
        )
        if initial_object_world is None:
            initial_object_world = current_object.copy()
        previous_object_world = current_object
        previous_distance = current_distance
        sensor_rows.append(sensor_row)
        supervision_rows.append(offline_row)
        if stereo.last_stereo_pair is not None:
            left, right = stereo.last_stereo_pair
            if rgb is not None:
                left = rgb
            left_frames.append(np.asarray(left, dtype=np.uint8))
            right_frames.append(np.asarray(right, dtype=np.uint8))

    try:
        env.reset(seed=as_int(row.get("seed"), 0))
        source_type = row.get("source_type", "")
        if source_type == "perturbed_success_final_state":
            for action in actions:
                env.step(action)
            ue = env.unwrapped
            target_position = np.array(
                [
                    as_float(row.get("cubeA_x")),
                    as_float(row.get("cubeA_y")),
                    as_float(row.get("cubeA_z")),
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(target_position)):
                half = float(to_np(ue.cube_half_size).reshape(-1)[0])
                base = to_np(ue.cubeB.pose.p).reshape(-1)[:3]
                offset = as_float(row.get("lateral_offset_m"), 0.03)
                direction = row.get("direction", "posx")
                dx, dy = {
                    "posx": (1.0, 0.0),
                    "negx": (-1.0, 0.0),
                    "posy": (0.0, 1.0),
                    "negy": (0.0, -1.0),
                }.get(direction, (1.0, 0.0))
                target_position = base + np.array([dx * offset, dy * offset, 2.0 * half])
            quaternion = to_np(ue.cubeA.pose.q).reshape(-1)[:4]
            ue.cubeA.set_pose(sapien.Pose(target_position, quaternion))
            zero_actor_velocity(ue.cubeA)
            adapter.reset()
            hold_steps = max(1, as_int(row.get("num_frames"), args.default_hold_steps))
            last_action = actions[-1] if len(actions) else np.zeros(env.action_space.shape, dtype=np.float32)
            for step in range(hold_steps):
                if args.perturb_mode == "settled":
                    env.step(last_action)
                    contact_valid = True
                else:
                    contact_valid = False
                observe(
                    last_action,
                    replay_step=step,
                    action_source_step=len(actions) - 1,
                    contact_valid=contact_valid,
                )
        else:
            adapter.reset()
            observe(None, replay_step=0, action_source_step=-1, contact_valid=True)
            action_limit = len(actions)
            if source_type == "truncated_success_trajectory":
                action_limit = max(1, min(len(actions), as_int(row.get("stop_step"), len(actions))))
            if args.max_steps_per_trajectory > 0:
                action_limit = min(action_limit, args.max_steps_per_trajectory)
            for action_index, action in enumerate(actions[:action_limit]):
                env.step(action)
                observe(
                    action,
                    replay_step=action_index + 1,
                    action_source_step=action_index,
                    contact_valid=True,
                )
    finally:
        env.close()

    if args.video_only:
        if not left_frames or len(left_frames) != len(right_frames):
            raise RuntimeError(
                f"invalid video-only replay for {row['sample_id']}: "
                f"left={len(left_frames)} right={len(right_frames)}"
            )
        sensor_path = sample_dir / "sensor_features.csv"
        expected_frames = len(read_csv(sensor_path))
        if expected_frames != len(left_frames):
            raise RuntimeError(
                f"video/table frame mismatch for {row['sample_id']}: "
                f"video={len(left_frames)} table={expected_frames}"
            )
        save_videos(sample_dir, left_frames, right_frames, args.video_fps)
        elapsed = time.perf_counter() - started
        payload = dict(existing_payload)
        payload.update(
            {
                "status": "COMPLETED_VIDEO_ONLY",
                "frames": len(left_frames),
                "video_only_elapsed_seconds": elapsed,
                "video_only_replay": True,
                "stereo_videos_saved": True,
                "video_fps": args.video_fps,
            }
        )
        write_json(completion_path, payload)
        return payload

    if not sensor_rows or len(sensor_rows) != len(supervision_rows):
        raise RuntimeError(
            f"invalid collection for {row['sample_id']}: sensor={len(sensor_rows)} "
            f"supervision={len(supervision_rows)}"
        )
    feature_names = set(physical_feature_names) | action_names
    forbidden_sensor_columns = [
        name
        for name in feature_names
        if name.startswith("gt_") or "success" in name or "pose" in name
    ]
    if forbidden_sensor_columns:
        raise RuntimeError(
            f"privileged columns entered sensor feature contract: {forbidden_sensor_columns}"
        )
    write_csv(sample_dir / "sensor_features.csv", sensor_rows)
    write_csv(sample_dir / "offline_supervision.csv", supervision_rows)
    save_preview(sample_dir, left_frames, right_frames)
    if args.save_stereo_videos:
        save_videos(sample_dir, left_frames, right_frames, args.video_fps)
    elapsed = time.perf_counter() - started
    payload = {
        "status": "COMPLETED",
        "sample_id": row["sample_id"],
        "split": row.get("split", ""),
        "source_type": row.get("source_type", ""),
        "source_group_id": source_group_id(row),
        "perturb_mode": args.perturb_mode,
        "frames": len(sensor_rows),
        "elapsed_seconds": elapsed,
        "mean_adapter_time_ms": float(
            np.mean([as_float(item.get("sensor_adapter_time_ms")) for item in sensor_rows])
        ),
        "feature_contract": sorted(feature_names),
        "physical_checkpoint_feature_contract": physical_feature_names,
        "sensor_table": str(sample_dir / "sensor_features.csv"),
        "supervision_table": str(sample_dir / "offline_supervision.csv"),
        "stereo_videos_saved": bool(args.save_stereo_videos),
    }
    write_json(completion_path, payload)
    return payload


def merge_outputs(args: argparse.Namespace, physical_feature_names: list[str]) -> dict[str, Any]:
    sensor_rows: list[dict[str, str]] = []
    supervision_rows: list[dict[str, str]] = []
    sample_metadata: list[dict[str, Any]] = []
    for path in sorted((args.out_dir / "samples").glob("*/*/collection_metadata.json")):
        sample_dir = path.parent
        sensor_path = sample_dir / "sensor_features.csv"
        supervision_path = sample_dir / "offline_supervision.csv"
        if not sensor_path.exists() or not supervision_path.exists():
            continue
        sensor_rows.extend(read_csv(sensor_path))
        supervision_rows.extend(read_csv(supervision_path))
        sample_metadata.append(json.loads(path.read_text(encoding="utf-8")))
    if not sensor_rows or len(sensor_rows) != len(supervision_rows):
        raise RuntimeError(
            f"global merge mismatch: sensor={len(sensor_rows)} supervision={len(supervision_rows)}"
        )
    write_csv(args.out_dir / "sensor_features.csv", sensor_rows)
    write_csv(args.out_dir / "offline_supervision.csv", supervision_rows)

    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in sensor_rows:
        splits_by_group[row["meta_source_group_id"]].add(row["meta_split"])
    leaking_groups = {
        group: sorted(splits) for group, splits in splits_by_group.items() if len(splits) > 1
    }
    if leaking_groups:
        raise RuntimeError(f"source-group leakage in collected dataset: {leaking_groups}")

    feature_validity = {}
    for name in physical_feature_names:
        feature_validity[name] = sum(
            1 for row in sensor_rows if (row.get(name) or "").strip() not in {"", "nan", "NaN"}
        ) / len(sensor_rows)
    summary = {
        "schema_version": (
            "stackcube_sensor_dataset_v3_gripper"
            if args.include_gripper_geometry
            else "stackcube_sensor_dataset_v2"
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "perturb_mode": args.perturb_mode,
        "include_gripper_geometry": bool(args.include_gripper_geometry),
        "num_samples": len(sample_metadata),
        "num_sensor_rows": len(sensor_rows),
        "num_supervision_rows": len(supervision_rows),
        "source_counts": dict(Counter(row["meta_source_type"] for row in sensor_rows)),
        "sample_source_counts": dict(Counter(row["source_type"] for row in sample_metadata)),
        "split_counts": dict(Counter(row["meta_split"] for row in sensor_rows)),
        "source_group_count": len(splits_by_group),
        "source_group_leakage_count": 0,
        "stereo_status_counts": dict(
            Counter(row.get("sensor_stereo_status", "") for row in sensor_rows)
        ),
        "mean_adapter_time_ms": float(
            np.mean([as_float(row.get("sensor_adapter_time_ms")) for row in sensor_rows])
        ),
        "physical_feature_validity": feature_validity,
        "physical_feature_names": physical_feature_names,
        "candidate_action_feature_names": sorted(
            {name for row in sensor_rows for name in row if name.startswith("action_")}
        ),
        "model_metadata_columns": MODEL_METADATA_COLUMNS,
        "sensor_and_supervision_tables_are_separate": True,
        "privileged_online_inputs": False,
        "notes": [
            "Simulator truth is stored only in offline_supervision.csv.",
            "Source-type and frame metadata must not be included in model features.",
            "Settled perturbations are new sensor-domain samples and do not exactly match the old frozen teacher videos.",
        ],
    }
    write_json(args.out_dir / "collection_summary.json", summary)
    write_json(
        args.out_dir / "feature_contract.json",
        {
            "schema_version": (
                "stackcube_sensor_feature_contract_v3_gripper"
                if args.include_gripper_geometry
                else "stackcube_sensor_feature_contract_v2"
            ),
            "model_input_features": physical_feature_names,
            "candidate_action_features_v2": summary["candidate_action_feature_names"],
            "metadata_excluded_from_model": MODEL_METADATA_COLUMNS
            + [
                "sensor_adapter_time_ms",
                "sensor_stereo_status",
                "sensor_stereo_dense_valid_ratio",
                "sensor_object_mask_pixels",
                "sensor_gripper_status",
                "sensor_gripper_mask_pixels",
                "sensor_depth_fields_emitted",
                "sensor_contact_fields_emitted",
                "sensor_contact_observation_valid",
            ],
            "offline_supervision_table": "offline_supervision.csv",
            "online_model_must_not_read": [
                "gt_*",
                "label_*",
                "meta_source_type",
                "meta_near_miss_type",
                "meta_replay_step",
                "meta_saved_frame_index",
            ],
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/mnt/e/reward_model_dataset/raw_rollouts/stackcube_bootstrap_v1/trajectory_manifest.csv"
        ),
    )
    parser.add_argument(
        "--physical-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/mnt/d/Users/User/Desktop/reward_model_dataset/sensor_consistent_v2/stackcube"
        ),
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--source-types", nargs="+", default=None)
    parser.add_argument("--sample-ids", nargs="+", default=None)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--max-steps-per-trajectory", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--default-hold-steps", type=int, default=12)
    parser.add_argument("--perturb-mode", choices=("settled", "frozen"), default="settled")
    parser.add_argument("--save-stereo-videos", action="store_true")
    parser.add_argument(
        "--video-only",
        action="store_true",
        help=(
            "Replay an existing collection only to add frame-aligned stereo "
            "videos; sensor and supervision CSVs are not overwritten."
        ),
    )
    parser.add_argument(
        "--include-gripper-geometry",
        action="store_true",
        help="Use the v2 stereo adapter to triangulate observable gripper geometry.",
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.video_only:
        args.save_stereo_videos = True
        if not args.out_dir.exists():
            raise FileNotFoundError(
                f"--video-only output collection does not exist: {args.out_dir}"
            )
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    scorer = wrapper.FrozenPhysicalScorer(
        args.physical_checkpoint, task_id="stackcube", device="cpu"
    )
    physical_feature_names = list(scorer.runtime.feature_names)
    if args.include_gripper_geometry:
        from stereo_feature_adapter_v2 import GRIPPER_GEOMETRY_FEATURES

        physical_feature_names = list(
            dict.fromkeys(physical_feature_names + sorted(GRIPPER_GEOMETRY_FEATURES))
        )
    rows = select_rows(args)
    if not rows:
        raise RuntimeError("no manifest rows selected")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": {key: str(value) for key, value in vars(args).items()},
        "selected_sample_ids": [row["sample_id"] for row in rows],
        "physical_feature_names": physical_feature_names,
    }
    write_json(args.out_dir / "collection_run_config.json", run_config)
    failures = []
    for index, row in enumerate(rows, start=1):
        print(
            f"[{index}/{len(rows)}] {row['sample_id']} "
            f"({row.get('source_type')}, {row.get('split')})",
            flush=True,
        )
        try:
            result = collect_sample(args, row, physical_feature_names)
            print(
                f"  {result['status']} frames={result.get('frames')} "
                f"elapsed={result.get('elapsed_seconds', 0):.2f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": row["sample_id"], "error": repr(exc)})
            print(f"  FAILED: {exc!r}", file=sys.stderr, flush=True)
    summary = merge_outputs(args, physical_feature_names)
    summary["failures"] = failures
    write_json(args.out_dir / "collection_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
