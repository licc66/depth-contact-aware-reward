from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


TASK_DIRS = {
    "stackcube": "stackcube_bootstrap_v1",
    "stackpyramid": "stackpyramid_bootstrap_v1",
    "peginsertion": "peginsertion_bootstrap_v1",
}

PEG_HEAD_LOCAL_OFFSET = np.array([0.1024, 0.0, 0.0], dtype=np.float64)
PEG_HOLE_LOCAL_OFFSET = np.array([0.0, 0.0043, -0.0056], dtype=np.float64)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(-1)[:4]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 1e-8:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def remap_dataset_path(value: str, dataset_root: Path) -> Path:
    text = (value or "").strip()
    if not text:
        return Path()
    root = str(dataset_root)
    if text.lower().startswith(r"e:\reward_model_dataset".lower()):
        return Path(root + text[len(r"E:\reward_model_dataset") :])
    if text.startswith("/mnt/e/reward_model_dataset"):
        rest = text[len("/mnt/e/reward_model_dataset") :].replace("/", "\\")
        return Path(root + rest)
    if text:
        return Path(text)
    return Path()


def resolve_h5_path(traj: dict[str, str], dataset_root: Path) -> Path:
    keys = ("h5_path_windows", "h5_path", "source_h5_path_windows", "source_h5_path")
    for key in keys:
        value = (traj.get(key, "") or "").strip()
        if not value:
            continue
        path = remap_dataset_path(value, dataset_root)
        if str(path) not in {"", "."} and path.exists() and path.is_file():
            return path
    raise FileNotFoundError(f"No readable h5 path for {traj.get('sample_id')}")


def read_actor_arrays(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        actors = f["traj_0/env_states/actors"]
        return {name: np.asarray(actors[name], dtype=np.float64) for name in actors.keys()}


def frame_to_state_idx(frame_idx: int, video_frames: int, state_frames: int) -> int:
    if state_frames <= 1 or video_frames <= 1:
        return max(0, min(state_frames - 1, frame_idx))
    ratio = max(0.0, min(1.0, frame_idx / float(video_frames - 1)))
    return int(round(ratio * (state_frames - 1)))


def estimate_half_size(*positions: np.ndarray) -> float:
    z_values = []
    for pos in positions:
        if len(pos):
            z_values.extend(np.asarray(pos[: min(10, len(pos)), 2], dtype=np.float64).tolist())
    finite = [z for z in z_values if math.isfinite(z) and 0.005 <= z <= 0.05]
    if not finite:
        return 0.02
    return float(np.clip(np.median(finite), 0.015, 0.025))


def repeated(value: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(np.asarray(value, dtype=np.float64).reshape(1, -1), max(1, n), axis=0)


def trim_for_truncated(traj: dict[str, str], arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if traj.get("source_type") != "truncated_success_trajectory":
        return arrays
    first = next(iter(arrays.values()))
    stop = safe_int(traj.get("stop_step"), len(first) - 1)
    keep = max(1, min(len(first), stop + 1))
    return {name: value[:keep] for name, value in arrays.items()}


def stack_manifest_positions(traj: dict[str, str], task_id: str) -> dict[str, np.ndarray]:
    n = safe_int(traj.get("num_frames"), safe_int(traj.get("frame_count"), 1)) or 1
    if task_id == "stackcube":
        top = np.array([safe_float(traj["cubeA_x"]), safe_float(traj["cubeA_y"]), safe_float(traj["cubeA_z"])])
        goal = np.array([safe_float(traj["goal_x"]), safe_float(traj["goal_y"]), safe_float(traj["goal_z"])])
        base = goal - np.array([0.0, 0.0, 0.04])
        return {"cubeA": repeated(top, n), "cubeB": repeated(base, n)}
    top = np.array([safe_float(traj["cubeC_x"]), safe_float(traj["cubeC_y"]), safe_float(traj["cubeC_z"])])
    goal = np.array([safe_float(traj["goal_x"]), safe_float(traj["goal_y"]), safe_float(traj["goal_z"])])
    base_gap = safe_float(traj.get("base_cube_xy_dist_m"), 0.05)
    # Manifest stores only the pyramid target center. Reconstruct a symmetric base proxy.
    base_a = goal - np.array([base_gap / 2.0, 0.0, 0.04])
    base_b = goal + np.array([base_gap / 2.0, 0.0, -0.04])
    base_b[2] = goal[2] - 0.04
    return {"cubeA": repeated(base_a, n), "cubeB": repeated(base_b, n), "cubeC": repeated(top, n)}


def peg_manifest_positions(traj: dict[str, str]) -> dict[str, np.ndarray]:
    n = safe_int(traj.get("num_frames"), safe_int(traj.get("frame_count"), 1)) or 1
    x = safe_float(traj.get("peg_head_at_hole_x"), safe_float(traj.get("peg_goal_3d_dist_m"), 0.0))
    yz = safe_float(traj.get("peg_head_at_hole_yz_norm"), 0.0)
    if not math.isfinite(x):
        x = 0.0
    if not math.isfinite(yz):
        yz = 0.0
    return {
        "peg_hole_local": repeated(np.array([x, yz, 0.0], dtype=np.float64), n),
        "peg_axis_alignment": np.full((max(1, n),), 1.0, dtype=np.float64),
    }


def load_task_arrays(traj: dict[str, str], task_id: str, dataset_root: Path) -> dict[str, np.ndarray]:
    if traj.get("source_type") == "perturbed_success_final_state":
        if task_id in {"stackcube", "stackpyramid"}:
            return stack_manifest_positions(traj, task_id)
        return peg_manifest_positions(traj)
    arrays = trim_for_truncated(traj, read_actor_arrays(resolve_h5_path(traj, dataset_root)))
    if task_id == "stackcube":
        return {"cubeA": arrays["cubeA"][:, :3], "cubeB": arrays["cubeB"][:, :3]}
    if task_id == "stackpyramid":
        return {"cubeA": arrays["cubeA"][:, :3], "cubeB": arrays["cubeB"][:, :3], "cubeC": arrays["cubeC"][:, :3]}
    peg = arrays["peg"]
    box = arrays["box_with_hole"]
    peg_p, peg_q = peg[:, :3], peg[:, 3:7]
    box_p, box_q = box[:, :3], box[:, 3:7]
    local = np.zeros((len(peg_p), 3), dtype=np.float64)
    align = np.zeros((len(peg_p),), dtype=np.float64)
    for i in range(len(peg_p)):
        r_peg = quat_to_matrix(peg_q[i])
        r_box = quat_to_matrix(box_q[i])
        peg_head = peg_p[i] + r_peg @ PEG_HEAD_LOCAL_OFFSET
        hole = box_p[i] + r_box @ PEG_HOLE_LOCAL_OFFSET
        local[i] = r_box.T @ (peg_head - hole)
        align[i] = abs(float(np.dot(r_peg[:, 0], r_box[:, 0])))
    return {"peg_hole_local": local, "peg_axis_alignment": align}


def footprint_overlap(dx: float, dy: float, half: float) -> float:
    width = 2.0 * half
    ox = clip01((width - abs(dx)) / width)
    oy = clip01((width - abs(dy)) / width)
    return float(ox * oy)


def stackcube_features(cube_a: np.ndarray, cube_b: np.ndarray, start: np.ndarray, half: float) -> dict[str, float]:
    delta = cube_a - cube_b
    xy = float(np.linalg.norm(delta[:2]))
    dz = float(delta[2])
    height_err = abs(dz - 2.0 * half)
    overlap = footprint_overlap(float(delta[0]), float(delta[1]), half)
    xy_score = clip01(1.0 - xy / 0.060)
    height_score = clip01(1.0 - height_err / 0.035)
    moved_score = clip01(float(np.linalg.norm(cube_a - start)) / 0.080)
    support_score = 0.60 * overlap + 0.25 * xy_score + 0.15 * height_score
    score = max(0.25 * moved_score, 0.35 * moved_score + 0.65 * support_score)
    if xy <= 0.014 and height_err <= 0.014:
        score = max(score, 0.96)
    return {
        "task_progress_score_v1": clip01(score),
        "xy_error_m_v1": xy,
        "height_error_m_v1": height_err,
        "support_overlap_v1": overlap,
        "axis_or_base_score_v1": 1.0,
        "task_rule_stage_v1": 3 if xy <= 0.014 and height_err <= 0.014 else (2 if support_score > 0.55 else (1 if moved_score > 0.25 else 0)),
    }


def stackpyramid_features(cube_a: np.ndarray, cube_b: np.ndarray, cube_c: np.ndarray, start: np.ndarray, half: float) -> dict[str, float]:
    base_mid = 0.5 * (cube_a + cube_b)
    base_gap = float(np.linalg.norm(cube_a[:2] - cube_b[:2]))
    top_delta = cube_c - (base_mid + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64))
    top_xy = float(np.linalg.norm(top_delta[:2]))
    height_err = abs(float(top_delta[2]))
    moved_score = clip01(float(np.linalg.norm(cube_c - start)) / 0.100)
    base_gap_score = clip01(1.0 - abs(base_gap - 0.050) / 0.060)
    base_height_score = clip01(1.0 - abs(float(cube_a[2] - cube_b[2])) / 0.020)
    base_ready = 0.65 * base_gap_score + 0.35 * base_height_score
    top_xy_score = clip01(1.0 - top_xy / 0.065)
    height_score = clip01(1.0 - height_err / 0.040)
    support_score = 0.70 * top_xy_score + 0.30 * height_score
    score = 0.15 * base_ready + 0.25 * moved_score + 0.60 * support_score
    if top_xy <= 0.018 and height_err <= 0.014 and base_ready > 0.55:
        score = max(score, 0.95)
    return {
        "task_progress_score_v1": clip01(score),
        "xy_error_m_v1": top_xy,
        "height_error_m_v1": height_err,
        "support_overlap_v1": top_xy_score,
        "axis_or_base_score_v1": base_ready,
        "base_gap_m_v1": base_gap,
        "task_rule_stage_v1": 3 if top_xy <= 0.018 and height_err <= 0.014 else (2 if support_score > 0.55 else (1 if moved_score > 0.25 else 0)),
    }


def peginsertion_features(local: np.ndarray, axis_alignment: float) -> dict[str, float]:
    x = float(local[0])
    yz = float(np.linalg.norm(local[1:3]))
    dist = float(np.linalg.norm(local))
    axis_score = clip01((axis_alignment - 0.70) / 0.30)
    lateral_score = clip01(1.0 - yz / 0.045)
    depth_score = clip01(1.0 - abs(x) / 0.055)
    approach_score = clip01(1.0 - dist / 0.40)
    insert_score = lateral_score * axis_score * depth_score
    align_score = 0.65 * lateral_score + 0.35 * axis_score
    score = max(0.35 * approach_score, 0.45 * align_score + 0.55 * insert_score)
    if abs(x) <= 0.016 and yz <= 0.010 and axis_score >= 0.80:
        score = max(score, 0.96)
    return {
        "task_progress_score_v1": clip01(score),
        "insertion_x_at_hole_m_v1": x,
        "insertion_yz_error_m_v1": yz,
        "peg_axis_alignment_v1": float(axis_alignment),
        "xy_error_m_v1": yz,
        "height_error_m_v1": abs(x),
        "support_overlap_v1": lateral_score,
        "axis_or_base_score_v1": axis_score,
        "task_rule_stage_v1": 3 if abs(x) <= 0.016 and yz <= 0.010 else (2 if lateral_score > 0.60 and axis_score > 0.70 else (1 if approach_score > 0.30 else 0)),
    }


def load_peginsertion_evaluate_rows(dataset_root: Path) -> dict[str, list[dict[str, str]]]:
    path = dataset_root / "contact_stage_features" / "peginsertion_bootstrap_v1" / "frame_contact_stage_features.csv"
    if not path.exists():
        return {}
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in load_csv(path):
        by_sample[row["sample_id"]].append(row)
    for rows in by_sample.values():
        rows.sort(key=lambda item: safe_int(item.get("frame_idx")))
    return dict(by_sample)


def build_peginsertion_frame_rows_from_evaluate(traj: dict[str, str], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    total = len(rows)
    for i, row in enumerate(rows):
        x = safe_float(row.get("eval_peg_head_at_hole_x"), 0.0)
        yz = safe_float(row.get("eval_peg_head_at_hole_yz_norm"), 999.0)
        feats = peginsertion_features(np.array([x, yz, 0.0], dtype=np.float64), 1.0)
        out.append(
            {
                "sample_id": traj["sample_id"],
                "task_id": "peginsertion",
                "split": traj.get("split", ""),
                "source_type": traj.get("source_type", ""),
                "near_miss_type": traj.get("near_miss_type", ""),
                "frame_idx": safe_int(row.get("frame_idx"), i),
                "state_idx": safe_int(row.get("frame_idx"), i),
                "video_frames": total,
                "state_frames": total,
                "time_progress": i / max(1, total - 1),
                **feats,
                "feature_source_v1": "task_specific_peginsertion_evaluate_geometry_rules_v1",
            }
        )
    return out


def build_frame_rows(
    traj: dict[str, str],
    task_id: str,
    dataset_root: Path,
    peg_evaluate_by_sample: dict[str, list[dict[str, str]]] | None = None,
) -> list[dict[str, Any]]:
    if task_id == "peginsertion" and peg_evaluate_by_sample:
        eval_rows = peg_evaluate_by_sample.get(traj["sample_id"])
        if eval_rows:
            return build_peginsertion_frame_rows_from_evaluate(traj, eval_rows)
    arrays = load_task_arrays(traj, task_id, dataset_root)
    state_frames = len(next(iter(arrays.values())))
    video_frames = safe_int(traj.get("frame_count"), safe_int(traj.get("num_frames"), state_frames)) or state_frames
    rows: list[dict[str, Any]] = []
    if task_id == "stackcube":
        half = estimate_half_size(arrays["cubeA"], arrays["cubeB"])
        start = arrays["cubeA"][0]
    elif task_id == "stackpyramid":
        half = estimate_half_size(arrays["cubeA"], arrays["cubeB"], arrays["cubeC"])
        start = arrays["cubeC"][0]
    else:
        half = 0.0
        start = np.zeros(3, dtype=np.float64)

    for frame_idx in range(video_frames):
        state_idx = frame_to_state_idx(frame_idx, video_frames, state_frames)
        if task_id == "stackcube":
            feats = stackcube_features(arrays["cubeA"][state_idx], arrays["cubeB"][state_idx], start, half)
        elif task_id == "stackpyramid":
            feats = stackpyramid_features(
                arrays["cubeA"][state_idx],
                arrays["cubeB"][state_idx],
                arrays["cubeC"][state_idx],
                start,
                half,
            )
        else:
            feats = peginsertion_features(arrays["peg_hole_local"][state_idx], float(arrays["peg_axis_alignment"][state_idx]))
        rows.append(
            {
                "sample_id": traj["sample_id"],
                "task_id": task_id,
                "split": traj.get("split", ""),
                "source_type": traj.get("source_type", ""),
                "near_miss_type": traj.get("near_miss_type", ""),
                "frame_idx": frame_idx,
                "state_idx": state_idx,
                "video_frames": video_frames,
                "state_frames": state_frames,
                "time_progress": frame_idx / max(1, video_frames - 1),
                **feats,
                "feature_source_v1": "task_specific_stereo_geometry_rules_v1",
            }
        )
    return rows


def aggregate_clip(clip: dict[str, str], by_sample: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = by_sample.get(clip["trajectory_id"], [])
    start = safe_int(clip["start_frame"])
    end = safe_int(clip["end_frame_exclusive"])
    clip_rows = rows[start:end]
    if not clip_rows:
        return {"clip_id": clip["clip_id"], "has_stereo_task_rules_v1": False}
    end_row = clip_rows[-1]
    scores = np.array([safe_float(r["task_progress_score_v1"]) for r in clip_rows], dtype=np.float64)
    stages = np.array([safe_int(r["task_rule_stage_v1"]) for r in clip_rows], dtype=np.int64)
    return {
        "clip_id": clip["clip_id"],
        "trajectory_id": clip["trajectory_id"],
        "task_id": clip["task_id"],
        "split": clip.get("split", ""),
        "source_type": clip.get("source_type", ""),
        "near_miss_type": clip.get("near_miss_type", ""),
        "start_frame": start,
        "end_frame_exclusive": end,
        "num_frames": len(clip_rows),
        "has_stereo_task_rules_v1": True,
        "end_task_progress_score_v1": float(end_row["task_progress_score_v1"]),
        "mean_task_progress_score_v1": float(np.mean(scores)),
        "max_task_progress_score_v1": float(np.max(scores)),
        "end_task_rule_stage_v1": int(end_row["task_rule_stage_v1"]),
        "max_task_rule_stage_v1": int(np.max(stages)),
        "end_xy_error_m_v1": safe_float(end_row.get("xy_error_m_v1")),
        "end_height_error_m_v1": safe_float(end_row.get("height_error_m_v1")),
        "end_support_overlap_v1": safe_float(end_row.get("support_overlap_v1")),
        "end_axis_or_base_score_v1": safe_float(end_row.get("axis_or_base_score_v1")),
        "end_insertion_x_at_hole_m_v1": safe_float(end_row.get("insertion_x_at_hole_m_v1")),
        "end_insertion_yz_error_m_v1": safe_float(end_row.get("insertion_yz_error_m_v1")),
        "end_peg_axis_alignment_v1": safe_float(end_row.get("peg_axis_alignment_v1")),
        "feature_source_v1": "task_specific_stereo_geometry_rules_v1",
    }


def label_from_diff(diff: float, threshold: float = 0.08) -> str:
    if diff > threshold:
        return "A>B"
    if diff < -threshold:
        return "B>A"
    return "unsure"


def build_pair_rows(pairs: list[dict[str, str]], clip_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_clip = {row["clip_id"]: row for row in clip_rows if row.get("has_stereo_task_rules_v1")}
    out = []
    for pair in pairs:
        a = by_clip[pair["clip_a_id"]]
        b = by_clip[pair["clip_b_id"]]
        score_a = safe_float(a["end_task_progress_score_v1"])
        score_b = safe_float(b["end_task_progress_score_v1"])
        diff = score_a - score_b
        label = label_from_diff(diff)
        candidate = pair.get("label", pair.get("candidate_label", ""))
        out.append(
            {
                "pair_id": pair["pair_id"],
                "task_id": pair["task_id"],
                "split": pair.get("split", ""),
                "pair_type": pair["pair_type"],
                "candidate_label": candidate,
                "needs_vlm_preference_label": pair.get("needs_vlm_preference_label", ""),
                "clip_a_id": pair["clip_a_id"],
                "clip_b_id": pair["clip_b_id"],
                "stereo_geometry_label_proxy_v1": label,
                "stereo_geometry_label_agrees_with_pair_label_v1": label == candidate,
                "stereo_score_diff_a_minus_b_v1": diff,
                "clip_a_end_score_proxy_v1": score_a,
                "clip_b_end_score_proxy_v1": score_b,
                "clip_a_end_stage_proxy_v1": a["end_task_rule_stage_v1"],
                "clip_b_end_stage_proxy_v1": b["end_task_rule_stage_v1"],
                "clip_a_end_xy_error_m_v1": a["end_xy_error_m_v1"],
                "clip_b_end_xy_error_m_v1": b["end_xy_error_m_v1"],
                "clip_a_end_height_error_m_v1": a["end_height_error_m_v1"],
                "clip_b_end_height_error_m_v1": b["end_height_error_m_v1"],
                "clip_a_end_support_overlap_v1": a["end_support_overlap_v1"],
                "clip_b_end_support_overlap_v1": b["end_support_overlap_v1"],
                "clip_a_end_axis_or_base_score_v1": a["end_axis_or_base_score_v1"],
                "clip_b_end_axis_or_base_score_v1": b["end_axis_or_base_score_v1"],
                "clip_a_end_insertion_x_at_hole_m_v1": a["end_insertion_x_at_hole_m_v1"],
                "clip_b_end_insertion_x_at_hole_m_v1": b["end_insertion_x_at_hole_m_v1"],
                "clip_a_end_insertion_yz_error_m_v1": a["end_insertion_yz_error_m_v1"],
                "clip_b_end_insertion_yz_error_m_v1": b["end_insertion_yz_error_m_v1"],
                "feature_source_v1": "task_specific_stereo_geometry_rules_v1",
            }
        )
    return out


def update_joined_table(joined_rows: list[dict[str, str]], pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair = {row["pair_id"]: row for row in pair_rows}
    out: list[dict[str, Any]] = []
    for row in joined_rows:
        v1 = by_pair[row["pair_id"]]
        item: dict[str, Any] = dict(row)
        for key in [
            "stereo_geometry_label_proxy",
            "stereo_geometry_label_agrees_with_pair_label",
            "stereo_clip_a_end_stage_proxy",
            "stereo_clip_b_end_stage_proxy",
            "stereo_clip_a_end_score_proxy",
            "stereo_clip_b_end_score_proxy",
            "stereo_score_diff_a_minus_b",
            "stereo_clip_a_end_dist_m",
            "stereo_clip_b_end_dist_m",
            "stereo_clip_a_end_depth_error_m",
            "stereo_clip_b_end_depth_error_m",
            "stereo_feature_source",
        ]:
            item[f"{key}_v0"] = row.get(key, "")
        item["stereo_geometry_label_proxy"] = v1["stereo_geometry_label_proxy_v1"]
        item["stereo_geometry_label_agrees_with_pair_label"] = str(v1["stereo_geometry_label_agrees_with_pair_label_v1"])
        item["stereo_clip_a_end_stage_proxy"] = v1["clip_a_end_stage_proxy_v1"]
        item["stereo_clip_b_end_stage_proxy"] = v1["clip_b_end_stage_proxy_v1"]
        item["stereo_clip_a_end_score_proxy"] = v1["clip_a_end_score_proxy_v1"]
        item["stereo_clip_b_end_score_proxy"] = v1["clip_b_end_score_proxy_v1"]
        item["stereo_score_diff_a_minus_b"] = v1["stereo_score_diff_a_minus_b_v1"]
        item["stereo_clip_a_end_dist_m"] = v1["clip_a_end_xy_error_m_v1"]
        item["stereo_clip_b_end_dist_m"] = v1["clip_b_end_xy_error_m_v1"]
        item["stereo_clip_a_end_depth_error_m"] = v1["clip_a_end_height_error_m_v1"]
        item["stereo_clip_b_end_depth_error_m"] = v1["clip_b_end_height_error_m_v1"]
        item["stereo_feature_source"] = v1["feature_source_v1"]
        for key, value in v1.items():
            if key not in item:
                item[key] = value
        out.append(item)
    return out


def summarize_pair_rows(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    by_pair_type: dict[str, Counter[str]] = defaultdict(Counter)
    agree = 0
    clear = 0
    for row in pair_rows:
        label = row["stereo_geometry_label_proxy_v1"]
        by_task[row["task_id"]][label] += 1
        by_pair_type[row["pair_type"]][label] += 1
        agree += int(bool(row["stereo_geometry_label_agrees_with_pair_label_v1"]))
        clear += int(label != "unsure")
    return {
        "num_pair_rows": len(pair_rows),
        "label_distribution": dict(Counter(row["stereo_geometry_label_proxy_v1"] for row in pair_rows)),
        "candidate_agreement_rate_all": agree / len(pair_rows) if pair_rows else 0.0,
        "clear_label_rows": clear,
        "by_task": {key: dict(value) for key, value in by_task.items()},
        "by_pair_type": {key: dict(value) for key, value in by_pair_type.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build task-specific stereo/depth rule labels v1.")
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\Users\User\Desktop\reward_model_dataset"))
    parser.add_argument("--joined-input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_frame_rows: list[dict[str, Any]] = []
    all_clip_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    for task_id, task_dir in TASK_DIRS.items():
        dataset_dir = args.dataset_root / "raw_rollouts" / task_dir
        index_dir = args.dataset_root / "pair_indices" / task_dir
        trajectories = load_csv(dataset_dir / "trajectory_manifest.csv")
        clips = load_csv(index_dir / "clip_manifest.csv")
        pairs = load_csv(index_dir / "pair_manifest.csv")
        peg_evaluate_by_sample = load_peginsertion_evaluate_rows(args.dataset_root) if task_id == "peginsertion" else None
        frame_by_sample: dict[str, list[dict[str, Any]]] = {}
        for traj in trajectories:
            rows = build_frame_rows(traj, task_id, args.dataset_root, peg_evaluate_by_sample)
            frame_by_sample[traj["sample_id"]] = rows
            all_frame_rows.extend(rows)
        clip_rows = [aggregate_clip(clip, frame_by_sample) for clip in clips]
        pair_rows = build_pair_rows(pairs, clip_rows)
        all_clip_rows.extend(clip_rows)
        all_pair_rows.extend(pair_rows)

    joined = load_csv(args.joined_input)
    joined_v1 = update_joined_table(joined, all_pair_rows)
    summary = summarize_pair_rows(all_pair_rows)

    write_csv(args.out / "frame_stereo_task_rules_v1.csv", all_frame_rows)
    write_csv(args.out / "clip_stereo_task_rules_v1.csv", all_clip_rows)
    write_csv(args.out / "pair_stereo_task_rules_v1.csv", all_pair_rows)
    write_json(args.out / "pair_stereo_task_rules_v1.json", all_pair_rows)
    write_json(args.out / "stereo_task_rules_v1_summary.json", summary)
    write_csv(args.out / "training_pairs_joined_stereo_v1.csv", joined_v1)
    write_json(args.out / "training_pairs_joined_stereo_v1_summary.json", {"rows": len(joined_v1), **summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
