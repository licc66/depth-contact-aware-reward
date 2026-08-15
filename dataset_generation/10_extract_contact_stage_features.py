from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien


TASKS = {
    "stackcube": "StackCube-v1",
    "stackpyramid": "StackPyramid-v1",
    "peginsertion": "PegInsertionSide-v1",
}

OBJECT_ATTR = {
    "stackcube": "cubeA",
    "stackpyramid": "cubeC",
    "peginsertion": "peg",
}

CONTACT_ONLY_FIELDS = (
    "is_grasping_object",
    "finger_object_contact_force_n",
    "finger_object_contact",
    "left_finger_object_contact_force_n",
    "right_finger_object_contact_force_n",
    "left_finger_touch_object",
    "right_finger_touch_object",
    "both_fingers_touch_object",
    "grasp_confirmed",
    "object_support_contact_force_n",
    "object_support_contacts",
    "top_cube_cubeA_contact_force_n",
    "top_cube_cubeB_contact_force_n",
    "top_cube_contacts_both_base_cubes",
    "peg_box_contact_force_n",
    "released_object",
)


@dataclass(frozen=True)
class TaskPaths:
    task_id: str
    dataset: Path
    indices: Path
    out: Path


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def to_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def bool_scalar(value: Any) -> bool:
    arr = to_np(value).reshape(-1)
    return bool(arr[0]) if arr.size else bool(value)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def read_actions(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f["traj_0/actions"], dtype=np.float32)


def resolve_h5_path(row: dict[str, str]) -> Path:
    for key in ("h5_path", "h5_path_wsl", "source_h5_path", "source_h5_path_wsl"):
        value = (row.get(key) or "").strip()
        if value:
            return Path(value)
    raise ValueError(f"No h5/source_h5 path for {row.get('sample_id')}")


def make_env(task_id: str):
    return gym.make(
        TASKS[task_id],
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend="cpu",
        render_backend="cpu",
        max_episode_steps=1000,
    )


def actor(ue, name: str):
    return getattr(ue, name)


def finger_links(ue) -> list[Any]:
    return [link for link in ue.agent.robot.links if "finger" in link.name.lower()]


def named_finger_links(ue) -> tuple[Any | None, Any | None]:
    links = finger_links(ue)
    left = next((link for link in links if "left" in link.name.lower()), None)
    right = next((link for link in links if "right" in link.name.lower()), None)
    remaining = [link for link in links if link is not left and link is not right]
    if left is None and remaining:
        left = remaining.pop(0)
    if right is None and remaining:
        right = remaining.pop(0)
    return left, right


def contact_force_norm(ue, a: Any, b: Any) -> float:
    try:
        force = ue.scene.get_pairwise_contact_forces(a, b)
        return float(np.linalg.norm(to_np(force)))
    except Exception:
        return float("nan")


def gripper_width(ue) -> float:
    try:
        robot = ue.agent.robot
        qpos = to_np(robot.get_qpos()).reshape(-1)
        joints = list(getattr(robot, "active_joints", []))
        if not joints and hasattr(robot, "get_active_joints"):
            joints = list(robot.get_active_joints())
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
        pass
    return float("nan")


def object_position(ue, task_id: str) -> np.ndarray:
    if task_id == "peginsertion":
        return to_np(ue.peg_head_pose.p).reshape(-1)[:3].astype(np.float64)
    return to_np(getattr(ue, OBJECT_ATTR[task_id]).pose.p).reshape(-1)[:3].astype(np.float64)


def goal_position(ue, task_id: str) -> np.ndarray:
    if task_id == "stackcube":
        half = float(to_np(ue.cube_half_size).reshape(-1)[0])
        cube_b = to_np(ue.cubeB.pose.p).reshape(-1)[:3].astype(np.float64)
        return cube_b + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64)
    if task_id == "stackpyramid":
        half = float(to_np(ue.cube_half_size).reshape(-1)[0])
        cube_a = to_np(ue.cubeA.pose.p).reshape(-1)[:3].astype(np.float64)
        cube_b = to_np(ue.cubeB.pose.p).reshape(-1)[:3].astype(np.float64)
        return 0.5 * (cube_a + cube_b) + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64)
    if task_id == "peginsertion":
        return to_np(ue.box_hole_pose.p).reshape(-1)[:3].astype(np.float64)
    raise ValueError(task_id)


def support_contact_force(ue, task_id: str) -> dict[str, float]:
    if task_id == "stackcube":
        f = contact_force_norm(ue, ue.cubeA, ue.cubeB)
        return {"object_support_contact_force_n": f, "object_support_contacts": int(f > 0.05)}
    if task_id == "stackpyramid":
        f_a = contact_force_norm(ue, ue.cubeC, ue.cubeA)
        f_b = contact_force_norm(ue, ue.cubeC, ue.cubeB)
        return {
            "object_support_contact_force_n": float(np.nansum([f_a, f_b])),
            "top_cube_cubeA_contact_force_n": f_a,
            "top_cube_cubeB_contact_force_n": f_b,
            "top_cube_contacts_both_base_cubes": int(f_a > 0.05 and f_b > 0.05),
            "object_support_contacts": int(f_a > 0.05 or f_b > 0.05),
        }
    if task_id == "peginsertion":
        f = contact_force_norm(ue, ue.peg, ue.box)
        return {"object_support_contact_force_n": f, "peg_box_contact_force_n": f, "object_support_contacts": int(f > 0.05)}
    raise ValueError(task_id)


def grasp_features(ue, task_id: str) -> dict[str, Any]:
    obj = actor(ue, OBJECT_ATTR[task_id])
    left_link, right_link = named_finger_links(ue)
    left_force = contact_force_norm(ue, left_link, obj) if left_link is not None else float("nan")
    right_force = contact_force_norm(ue, right_link, obj) if right_link is not None else float("nan")
    link_forces = [left_force, right_force]
    if left_link is None or right_link is None:
        link_forces = [contact_force_norm(ue, link, obj) for link in finger_links(ue)]
    valid_forces = [f for f in link_forces if math.isfinite(f)]
    finger_force = float(sum(valid_forces)) if valid_forces else float("nan")
    left_touch = math.isfinite(left_force) and left_force > 0.05
    right_touch = math.isfinite(right_force) and right_force > 0.05
    both_touch = left_touch and right_touch
    try:
        is_grasping = bool_scalar(ue.agent.is_grasping(obj))
    except Exception:
        is_grasping = False
    width = gripper_width(ue)
    return {
        "is_grasping_object": bool(is_grasping),
        "finger_object_contact_force_n": finger_force,
        "finger_object_contact": int(finger_force > 0.05) if math.isfinite(finger_force) else 0,
        "left_finger_object_contact_force_n": left_force,
        "right_finger_object_contact_force_n": right_force,
        "left_finger_touch_object": int(left_touch),
        "right_finger_touch_object": int(right_touch),
        "both_fingers_touch_object": int(both_touch),
        "gripper_width": width,
        "grasp_confirmed": int(both_touch and is_grasping),
    }


def evaluate_features(ue, task_id: str) -> dict[str, Any]:
    info = ue.evaluate()
    out: dict[str, Any] = {"env_success": bool_scalar(info.get("success", False))}
    if task_id == "stackcube":
        out["eval_is_cubeA_on_cubeB"] = bool_scalar(info.get("is_cubeA_on_cubeB", False))
        out["eval_is_cubeA_static"] = bool_scalar(info.get("is_cubeA_static", False))
        out["eval_is_cubeA_grasped"] = bool_scalar(info.get("is_cubeA_grasped", False))
    elif task_id == "peginsertion":
        peg_head_at_hole = to_np(info.get("peg_head_pos_at_hole", np.full(3, np.nan))).reshape(-1)
        out["eval_peg_head_at_hole_x"] = float(peg_head_at_hole[0]) if peg_head_at_hole.size > 0 else float("nan")
        out["eval_peg_head_at_hole_yz_norm"] = float(np.linalg.norm(peg_head_at_hole[1:3])) if peg_head_at_hole.size >= 3 else float("nan")
    return out


def set_perturbed_pose(ue, row: dict[str, str], task_id: str) -> None:
    if task_id == "stackcube":
        pos = np.array([safe_float(row.get("cubeA_x")), safe_float(row.get("cubeA_y")), safe_float(row.get("cubeA_z"))], dtype=np.float64)
        quat = to_np(ue.cubeA.pose.q).reshape(-1)[:4]
        ue.cubeA.set_pose(sapien.Pose(pos, quat))
    elif task_id == "stackpyramid":
        pos = np.array([safe_float(row.get("cubeC_x")), safe_float(row.get("cubeC_y")), safe_float(row.get("cubeC_z"))], dtype=np.float64)
        quat = to_np(ue.cubeC.pose.q).reshape(-1)[:4]
        ue.cubeC.set_pose(sapien.Pose(pos, quat))
    elif task_id == "peginsertion":
        pos = np.array([safe_float(row.get("peg_x")), safe_float(row.get("peg_y")), safe_float(row.get("peg_z"))], dtype=np.float64)
        quat = to_np(ue.peg.pose.q).reshape(-1)[:4]
        ue.peg.set_pose(sapien.Pose(pos, quat))
    else:
        raise ValueError(task_id)


def contact_stage(task_id: str, feats: dict[str, Any]) -> tuple[int, str, float]:
    success = bool(feats["env_success"])
    contact_valid = bool(feats.get("contact_observation_valid", True))
    grasp = contact_valid and bool(feats["is_grasping_object"])
    finger_contact = contact_valid and bool(feats["finger_object_contact"])
    support_contact = contact_valid and bool(feats["object_support_contacts"])
    released = contact_valid and bool(feats["released_object"])
    static = bool(feats["object_static_proxy"])
    dist = safe_float(feats["object_goal_3d_dist_m"], 999.0)
    xy = safe_float(feats["object_goal_xy_error_m"], 999.0)
    height = safe_float(feats["object_goal_height_error_m"], 999.0)

    if task_id == "stackcube":
        if success or (support_contact and released and static and xy <= 0.03 and height <= 0.02):
            return 4, "stable_success", 4.0
        if not contact_valid and dist <= 0.12:
            return 3, "near_goal_contact_unobserved", 3.0
        if support_contact:
            return 3, "object_on_support_contact", 3.0
        if grasp or finger_contact or feats["object_moved_from_start_m"] > 0.03:
            return 2, "grasp_transport_or_align", 2.0
        return 1, "pre_contact_or_approach", 1.0

    if task_id == "stackpyramid":
        both = contact_valid and bool(feats.get("top_cube_contacts_both_base_cubes", 0))
        if success or (support_contact and released and static and xy <= 0.04 and height <= 0.025):
            return 4, "stable_pyramid_success", 4.0
        if not contact_valid and dist <= 0.14:
            return 3, "near_goal_contact_unobserved", 3.0
        if both:
            return 3, "top_cube_contacts_both_base_cubes", 3.1
        if support_contact:
            return 3, "top_cube_contacts_base", 3.0
        if grasp or finger_contact or feats["object_moved_from_start_m"] > 0.03:
            return 2, "grasp_transport_or_align", 2.0
        return 1, "pre_contact_or_approach", 1.0

    if task_id == "peginsertion":
        yz = safe_float(feats.get("eval_peg_head_at_hole_yz_norm"), 999.0)
        x = safe_float(feats.get("eval_peg_head_at_hole_x"), 999.0)
        if success:
            return 4, "inserted_success", 4.0
        if not contact_valid and (dist <= 0.09 or x < 0.05):
            return 3, "near_hole_contact_unobserved", 3.0
        if support_contact and yz <= 0.025:
            return 3, "peg_hole_contact_aligned", 3.0
        if support_contact:
            return 2, "peg_box_contact_misaligned", 2.4
        if grasp or finger_contact or dist <= 0.07 or x < 0.05:
            return 2, "grasp_transport_or_align", 2.0
        return 1, "pre_contact_or_approach", 1.0

    raise ValueError(task_id)


def record_features(
    ue,
    row: dict[str, str],
    task_id: str,
    frame_idx: int,
    prev_obj: np.ndarray | None,
    start_obj: np.ndarray,
    contact_observation_valid: bool = True,
) -> tuple[dict[str, Any], np.ndarray]:
    obj = object_position(ue, task_id)
    goal = goal_position(ue, task_id)
    delta = obj - goal
    speed = 0.0 if prev_obj is None else float(np.linalg.norm(obj - prev_obj))
    support = support_contact_force(ue, task_id)
    grasp = grasp_features(ue, task_id)
    evals = evaluate_features(ue, task_id)
    dist = float(np.linalg.norm(delta))
    xy = float(np.linalg.norm(delta[:2]))
    height = abs(float(delta[2]))
    moved = float(np.linalg.norm(obj - start_obj))
    feats: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "task_id": task_id,
        "split": row.get("split", ""),
        "source_type": row.get("source_type", ""),
        "near_miss_type": row.get("near_miss_type", ""),
        "frame_idx": frame_idx,
        "object_x": float(obj[0]),
        "object_y": float(obj[1]),
        "object_z": float(obj[2]),
        "goal_x": float(goal[0]),
        "goal_y": float(goal[1]),
        "goal_z": float(goal[2]),
        "object_goal_3d_dist_m": dist,
        "object_goal_xy_error_m": xy,
        "object_goal_height_error_m": height,
        "object_moved_from_start_m": moved,
        "object_speed_proxy_m_per_step": speed,
        "object_static_proxy": speed <= (0.003 if task_id != "peginsertion" else 0.006),
        "contact_observation_valid": contact_observation_valid,
        **grasp,
        **support,
        **evals,
    }
    feats["released_object"] = (not bool(feats["is_grasping_object"])) and safe_float(feats["finger_object_contact_force_n"], 0.0) <= 0.05
    stage_id, stage_name, stage_score = contact_stage(task_id, feats)
    if not contact_observation_valid:
        for key in CONTACT_ONLY_FIELDS:
            if key in feats:
                feats[key] = float("nan")
    feats["stage_id"] = stage_id
    feats["stage_name"] = stage_name
    feats["stage_score"] = stage_score
    feats["contact_feature_source"] = (
        "unavailable_for_frozen_teleport_snapshot"
        if not contact_observation_valid
        else "maniskill_scene_pairwise_contact_forces_and_task_evaluate"
    )
    return feats, obj


def replay_trajectory(env, row: dict[str, str], task_id: str) -> list[dict[str, Any]]:
    actions = read_actions(resolve_h5_path(row))
    seed = safe_int(row.get("seed"), 0)
    env.reset(seed=seed)
    ue = env.unwrapped
    start_obj = object_position(ue, task_id)
    prev_obj: np.ndarray | None = None
    rows: list[dict[str, Any]] = []

    source_type = row.get("source_type", "")
    if source_type == "truncated_success_trajectory":
        stop_step = max(1, min(len(actions), safe_int(row.get("stop_step"), len(actions))))
        actions_to_run = actions[:stop_step]
        rows.append(record_features(ue, row, task_id, 0, prev_obj, start_obj)[0])
        prev_obj = object_position(ue, task_id)
        for frame_idx, action in enumerate(actions_to_run, start=1):
            env.step(action)
            feats, prev_obj = record_features(ue, row, task_id, frame_idx, prev_obj, start_obj)
            rows.append(feats)
        return rows

    if source_type == "perturbed_success_final_state":
        for action in actions:
            env.step(action)
        set_perturbed_pose(ue, row, task_id)
        hold_frames = max(1, safe_int(row.get("num_frames"), 28))
        prev_obj = None
        for frame_idx in range(hold_frames):
            feats, prev_obj = record_features(
                ue,
                row,
                task_id,
                frame_idx,
                prev_obj,
                start_obj,
                contact_observation_valid=False,
            )
            rows.append(feats)
        return rows

    # Success trajectory: record reset plus each step.
    feats, prev_obj = record_features(ue, row, task_id, 0, prev_obj, start_obj)
    rows.append(feats)
    for frame_idx, action in enumerate(actions, start=1):
        env.step(action)
        feats, prev_obj = record_features(ue, row, task_id, frame_idx, prev_obj, start_obj)
        rows.append(feats)
    return rows


def aggregate_clip(clip: dict[str, str], frame_rows_by_sample: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    sample_id = clip["trajectory_id"]
    start = safe_int(clip["start_frame"])
    end = safe_int(clip["end_frame_exclusive"])
    source_rows = frame_rows_by_sample.get(sample_id, [])
    rows = source_rows[start:end]
    if not rows:
        rows = source_rows[-1:] if source_rows else []
    if not rows:
        return {"clip_id": clip["clip_id"], "has_contact_stage_features": False}
    end_row = rows[-1]

    def finite_mean(key: str) -> float:
        values = [safe_float(row.get(key)) for row in rows]
        finite = [value for value in values if math.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    def ratio(key: str) -> float:
        return finite_mean(key)

    return {
        "clip_id": clip["clip_id"],
        "trajectory_id": sample_id,
        "task_id": clip["task_id"],
        "split": clip["split"],
        "source_type": clip["source_type"],
        "near_miss_type": clip.get("near_miss_type", ""),
        "start_frame": start,
        "end_frame_exclusive": end,
        "num_contact_frames": len(rows),
        "has_contact_stage_features": True,
        "end_stage_id": end_row["stage_id"],
        "end_stage_name": end_row["stage_name"],
        "max_stage_id": int(max(safe_int(r["stage_id"]) for r in rows)),
        "mean_stage_score": float(np.mean([safe_float(r["stage_score"]) for r in rows])),
        "end_stage_score": safe_float(end_row["stage_score"]),
        "end_env_success": bool(end_row["env_success"]),
        "any_env_success": any(bool(r["env_success"]) for r in rows),
        "contact_observation_valid_ratio": ratio("contact_observation_valid"),
        "grasp_ratio": ratio("is_grasping_object"),
        "grasp_confirmed_ratio": ratio("grasp_confirmed"),
        "left_finger_touch_ratio": ratio("left_finger_touch_object"),
        "right_finger_touch_ratio": ratio("right_finger_touch_object"),
        "both_fingers_touch_ratio": ratio("both_fingers_touch_object"),
        "finger_object_contact_ratio": ratio("finger_object_contact"),
        "support_contact_ratio": ratio("object_support_contacts"),
        "release_ratio": ratio("released_object"),
        "static_ratio": ratio("object_static_proxy"),
        "end_object_support_contact_force_n": safe_float(end_row["object_support_contact_force_n"]),
        "mean_object_support_contact_force_n": finite_mean("object_support_contact_force_n"),
        "end_finger_object_contact_force_n": safe_float(end_row["finger_object_contact_force_n"]),
        "mean_finger_object_contact_force_n": finite_mean("finger_object_contact_force_n"),
        "end_gripper_width": safe_float(end_row.get("gripper_width")),
        "mean_gripper_width": finite_mean("gripper_width"),
        "end_object_goal_3d_dist_m": safe_float(end_row["object_goal_3d_dist_m"]),
        "end_object_goal_xy_error_m": safe_float(end_row["object_goal_xy_error_m"]),
        "end_object_goal_height_error_m": safe_float(end_row["object_goal_height_error_m"]),
        "contact_feature_source": "maniskill_scene_pairwise_contact_forces_and_task_evaluate",
    }


def build_pair_rows(pairs: list[dict[str, str]], clip_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_clip = {row["clip_id"]: row for row in clip_rows if row.get("has_contact_stage_features")}
    out: list[dict[str, Any]] = []
    for pair in pairs:
        a = by_clip[pair["clip_a_id"]]
        b = by_clip[pair["clip_b_id"]]
        score_a = safe_float(a["end_stage_score"])
        score_b = safe_float(b["end_stage_score"])
        diff = score_a - score_b
        if diff > 0.20:
            contact_label = "A>B"
        elif diff < -0.20:
            contact_label = "B>A"
        else:
            contact_label = "unsure"
        out.append(
            {
                **pair,
                "contact_stage_label_proxy": contact_label,
                "contact_stage_label_agrees_with_pair_label": contact_label == pair["label"],
                "clip_a_end_stage_id": a["end_stage_id"],
                "clip_b_end_stage_id": b["end_stage_id"],
                "clip_a_end_stage_name": a["end_stage_name"],
                "clip_b_end_stage_name": b["end_stage_name"],
                "clip_a_end_stage_score": score_a,
                "clip_b_end_stage_score": score_b,
                "stage_score_diff_a_minus_b": diff,
                "clip_a_support_contact_ratio": a["support_contact_ratio"],
                "clip_b_support_contact_ratio": b["support_contact_ratio"],
                "clip_a_grasp_ratio": a["grasp_ratio"],
                "clip_b_grasp_ratio": b["grasp_ratio"],
                "contact_feature_source": "maniskill_scene_pairwise_contact_forces_and_task_evaluate",
            }
        )
    return out


def summarize(task_id: str, trajectories: list[dict[str, str]], frame_rows: list[dict[str, Any]], clip_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts: dict[str, int] = {}
    pair_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in trajectories:
        source_counts[row["source_type"]] = source_counts.get(row["source_type"], 0) + 1
    for row in frame_rows:
        stage_counts[row["stage_name"]] = stage_counts.get(row["stage_name"], 0) + 1
    for row in pair_rows:
        pair_counts[row["pair_type"]] = pair_counts.get(row["pair_type"], 0) + 1
        label_counts[row["contact_stage_label_proxy"]] = label_counts.get(row["contact_stage_label_proxy"], 0) + 1
    return {
        "task_id": task_id,
        "num_trajectories": len(trajectories),
        "num_frame_rows": len(frame_rows),
        "num_clip_rows": len(clip_rows),
        "num_pair_rows": len(pair_rows),
        "source_counts": source_counts,
        "frame_stage_counts": stage_counts,
        "pair_type_counts": pair_counts,
        "contact_stage_label_proxy_counts": label_counts,
        "contact_unobserved_frame_count": sum(
            1 for row in frame_rows if not bool(row.get("contact_observation_valid", True))
        ),
        "pair_label_agreement_rate": (
            sum(1 for r in pair_rows if r["contact_stage_label_agrees_with_pair_label"]) / len(pair_rows) if pair_rows else 0.0
        ),
        "contact_feature_source": "maniskill_scene_pairwise_contact_forces_and_task_evaluate",
        "note": "True simulator contact forces are replayed for dynamical trajectories. Frozen teleport near-misses keep the matching geometry but mark stale contact fields missing. Stage labels are task-specific heuristic supervision, not model inputs.",
    }


def parse_task_entry(entry: str, default_out_root: Path) -> TaskPaths:
    parts = entry.split("=")
    if len(parts) != 3:
        raise ValueError("Task entry format must be task_id=dataset_dir=indices_dir")
    task_id, dataset, indices = parts
    task_id = task_id.strip().lower()
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id {task_id}")
    return TaskPaths(
        task_id=task_id,
        dataset=Path(dataset),
        indices=Path(indices),
        out=default_out_root / f"{task_id}_bootstrap_v1",
    )


def run_task(paths: TaskPaths, progress_every: int) -> dict[str, Any]:
    trajectories = load_csv(paths.dataset / "trajectory_manifest.csv")
    clips = load_csv(paths.indices / "clip_manifest.csv")
    pairs = load_csv(paths.indices / "pair_manifest.csv")
    env = make_env(paths.task_id)
    frame_rows: list[dict[str, Any]] = []
    frame_rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    try:
        for idx, traj in enumerate(trajectories, start=1):
            rows = replay_trajectory(env, traj, paths.task_id)
            frame_rows.extend(rows)
            frame_rows_by_sample[traj["sample_id"]] = rows
            if progress_every > 0 and idx % progress_every == 0:
                print(f"{paths.task_id}: replayed {idx}/{len(trajectories)} trajectories")
    finally:
        env.close()
    clip_rows = [aggregate_clip(clip, frame_rows_by_sample) for clip in clips]
    pair_rows = build_pair_rows(pairs, clip_rows)
    summary = summarize(paths.task_id, trajectories, frame_rows, clip_rows, pair_rows)

    write_csv(paths.out / "frame_contact_stage_features.csv", frame_rows)
    write_json(paths.out / "frame_contact_stage_features.json", frame_rows)
    write_csv(paths.out / "clip_contact_stage_features.csv", clip_rows)
    write_json(paths.out / "clip_contact_stage_features.json", clip_rows)
    write_csv(paths.out / "pair_contact_stage_labels.csv", pair_rows)
    write_json(paths.out / "pair_contact_stage_labels.json", pair_rows)
    write_json(paths.out / "contact_stage_summary.json", summary)
    (paths.out / "README.txt").write_text(
        "\n".join(
            [
                f"{paths.task_id} contact/stage features",
                "",
                "frame_contact_stage_features.csv/json: replay-level true contact force, grasp, release, stability, task-evaluate and stage labels.",
                "clip_contact_stage_features.csv/json: clip-level aggregation for reward model inputs.",
                "pair_contact_stage_labels.csv/json: pair preference proxy from end-stage score.",
                "",
                "Contact source: ManiSkill scene.get_pairwise_contact_forces + agent.is_grasping + env.evaluate().",
                "Stage labels are task-specific heuristic labels for training/fusion, not ground-truth task annotations from the original papers.",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay bootstrap datasets and extract true contact/stage features.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="Entries in format task_id=dataset_dir=indices_dir.",
    )
    parser.add_argument("--out-root", type=Path, default=Path("/mnt/e/reward_model_dataset/contact_stage_features"))
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for entry in args.tasks:
        paths = parse_task_entry(entry, args.out_root)
        summary = run_task(paths, args.progress_every)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    write_json(args.out_root / "contact_stage_bootstrap_v1_summary.json", summaries)


if __name__ == "__main__":
    main()
