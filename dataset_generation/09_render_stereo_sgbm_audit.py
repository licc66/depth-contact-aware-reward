from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien


def load_sgbm_module():
    path = Path(__file__).resolve().parents[1] / "stereo_depth_truth" / "02_stereo_sgbm_compare.py"
    spec = importlib.util.spec_from_file_location("stereo_sgbm_compare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sgbm = load_sgbm_module()


ENV_IDS = {
    "stackcube": "StackCube-v1",
    "stackpyramid": "StackPyramid-v1",
    "peginsertion": "PegInsertionSide-v1",
}


@dataclass(frozen=True)
class RenderTask:
    task_id: str
    env_id: str


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


def read_actions(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f["traj_0/actions"], dtype=np.float32)


def resolve_h5_path(row: dict[str, str]) -> Path:
    for key in ("h5_path", "source_h5_path", "h5_path_wsl", "source_h5_path_wsl"):
        value = (row.get(key) or "").strip()
        if value:
            return Path(value)
    raise ValueError(f"No h5 path for {row.get('sample_id')}")


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


def make_env(task_id: str, rig: Any):
    task = RenderTask(task_id=task_id, env_id=ENV_IDS[task_id])
    return sgbm.make_env(task, rig)


def reset_and_replay(env, row: dict[str, str], task_id: str) -> None:
    actions = read_actions(resolve_h5_path(row))
    seed = safe_int(row.get("seed"), 0)
    env.reset(seed=seed)

    if row.get("source_type") == "truncated_success_trajectory":
        stop_step = max(1, min(len(actions), safe_int(row.get("stop_step"), len(actions))))
        actions_to_run = actions[:stop_step]
    else:
        actions_to_run = actions
    for action in actions_to_run:
        env.step(action)

    if row.get("source_type") != "perturbed_success_final_state":
        return

    ue = env.unwrapped
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


def get_entities(env, task_id: str) -> list[dict[str, Any]]:
    ue = env.unwrapped
    if task_id == "stackcube":
        half = float(to_np(ue.cube_half_size).reshape(-1)[0])
        obj = to_np(ue.cubeA.pose.p).reshape(-1)[:3]
        support = to_np(ue.cubeB.pose.p).reshape(-1)[:3]
        goal = support + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64)
        return [
            {"entity": "object", "point": obj},
            {"entity": "goal", "point": goal},
            {"entity": "support", "point": support},
        ]

    if task_id == "stackpyramid":
        half = float(to_np(ue.cube_half_size).reshape(-1)[0])
        obj = to_np(ue.cubeC.pose.p).reshape(-1)[:3]
        base_a = to_np(ue.cubeA.pose.p).reshape(-1)[:3]
        base_b = to_np(ue.cubeB.pose.p).reshape(-1)[:3]
        support = 0.5 * (base_a + base_b)
        goal = support + np.array([0.0, 0.0, 2.0 * half], dtype=np.float64)
        return [
            {"entity": "object", "point": obj},
            {"entity": "goal", "point": goal},
            {"entity": "support", "point": support},
        ]

    if task_id == "peginsertion":
        obj = to_np(ue.peg_head_pose.p).reshape(-1)[:3]
        goal = to_np(ue.box_hole_pose.p).reshape(-1)[:3]
        peg = to_np(ue.peg.pose.p).reshape(-1)[:3]
        return [
            {"entity": "peg_head", "point": obj},
            {"entity": "hole", "point": goal},
            {"entity": "peg_center", "point": peg},
        ]

    raise ValueError(task_id)


def choose_rows(rows: list[dict[str, str]], max_per_group: int, max_total: int) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = row.get("source_type", "")
        if row.get("near_miss_type"):
            key = f"{key}:{row['near_miss_type']}"
        groups[key].append(row)
    selected: list[dict[str, str]] = []
    for key in sorted(groups):
        selected.extend(groups[key][:max_per_group])
    return selected[:max_total]


def safe_mean(values: list[float]) -> float:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def summarize(frame_rows: list[dict[str, Any]], entity_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task_source: dict[str, dict[str, Any]] = {}
    keys = sorted({(row["task_id"], row["source_type"]) for row in frame_rows})
    for task_id, source_type in keys:
        frames = [row for row in frame_rows if row["task_id"] == task_id and row["source_type"] == source_type]
        entities = [row for row in entity_rows if row["task_id"] == task_id and row["source_type"] == source_type]
        valid_entities = [row for row in entities if np.isfinite(row["pred_depth_m"])]
        by_task_source[f"{task_id}/{source_type}"] = {
            "frames": len(frames),
            "dense_valid_ratio_mean": safe_mean([row["dense_valid_ratio"] for row in frames]),
            "dense_depth_mae_m_mean": safe_mean([row["dense_depth_mae_m"] for row in frames]),
            "dense_disp_mae_px_mean": safe_mean([row["dense_disp_mae_px"] for row in frames]),
            "entity_points": len(entities),
            "valid_entity_points": len(valid_entities),
            "entity_valid_rate": len(valid_entities) / max(1, len(entities)),
            "entity_depth_mae_m_mean": safe_mean([row["entity_depth_abs_error_m"] for row in valid_entities]),
            "entity_disp_mae_px_mean": safe_mean([row["entity_disp_abs_error_px"] for row in valid_entities]),
        }
    return {
        "num_frame_rows": len(frame_rows),
        "num_entity_rows": len(entity_rows),
        "by_task_source": by_task_source,
        "method": "OpenCV StereoSGBM on fixed rectified ManiSkill stereo cameras, compared with simulator depth buffer and projected entity centers.",
    }


def audit_task(args: argparse.Namespace, task_id: str, dataset: Path, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = choose_rows(load_csv(dataset / "trajectory_manifest.csv"), args.max_per_group, args.max_total_per_task)
    rig = sgbm.make_image_rig(width=args.width, height=args.height, baseline_m=args.baseline_m, fov_y_rad=args.fov_y_rad)
    env = make_env(task_id, rig)
    frame_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    try:
        for idx, row in enumerate(rows):
            reset_and_replay(env, row, task_id)
            frame = sgbm.read_sensor_pair(env)
            left_rgb = frame["left_rgb"]
            right_rgb = frame["right_rgb"]
            left_depth_m = frame["left_depth_m"]
            left_intrinsic = frame["left_intrinsic"]
            right_intrinsic = frame["right_intrinsic"]
            left_extrinsic = frame["left_extrinsic"]
            right_extrinsic = frame["right_extrinsic"]
            focal_px = float(left_intrinsic[0, 0])
            disparity = sgbm.compute_disparity(left_rgb, right_rgb, args.num_disparities, args.block_size)
            dense = sgbm.dense_metrics(disparity, left_depth_m, focal_px, rig.baseline_m, rig.far_m)
            frame_row = {
                "task_id": task_id,
                "sample_id": row["sample_id"],
                "split": row.get("split", ""),
                "source_type": row.get("source_type", ""),
                "near_miss_type": row.get("near_miss_type", ""),
                "observed_success": row.get("observed_success", ""),
                "focal_px": focal_px,
                **dense,
            }
            frame_rows.append(frame_row)

            if idx < args.preview_count:
                preview = out_dir / "previews" / f"{task_id}_{idx:03d}_{row['sample_id']}.png"
                sgbm.draw_preview(preview, left_rgb, right_rgb, disparity, left_depth_m, focal_px, rig.baseline_m)
            if args.save_stereo_pairs:
                pair_dir = out_dir / "stereo_pairs" / task_id
                pair_dir.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(pair_dir / f"{row['sample_id']}_left.png", left_rgb)
                imageio.imwrite(pair_dir / f"{row['sample_id']}_right.png", right_rgb)

            for entity in get_entities(env, task_id):
                point = np.asarray(entity["point"], dtype=np.float64)
                left_proj = sgbm.project_point(point, left_intrinsic, left_extrinsic)
                right_proj = sgbm.project_point(point, right_intrinsic, right_extrinsic)
                pred_disp, patch_valid_ratio = sgbm.patch_median(disparity, left_proj["u"], left_proj["v"], args.patch_radius)
                gt_disp = (
                    left_proj["u"] - right_proj["u"] if np.isfinite(left_proj["u"]) and np.isfinite(right_proj["u"]) else float("nan")
                )
                if np.isfinite(pred_disp) and pred_disp > 0:
                    pred_depth = focal_px * rig.baseline_m / pred_disp
                    depth_abs_error = abs(pred_depth - left_proj["z"])
                    disp_abs_error = abs(pred_disp - gt_disp)
                else:
                    pred_depth = float("nan")
                    depth_abs_error = float("nan")
                    disp_abs_error = float("nan")
                entity_rows.append(
                    {
                        "task_id": task_id,
                        "sample_id": row["sample_id"],
                        "split": row.get("split", ""),
                        "source_type": row.get("source_type", ""),
                        "near_miss_type": row.get("near_miss_type", ""),
                        "entity": entity["entity"],
                        "left_u": left_proj["u"],
                        "left_v": left_proj["v"],
                        "right_u": right_proj["u"],
                        "right_v": right_proj["v"],
                        "gt_depth_m": left_proj["z"],
                        "gt_disp_px": gt_disp,
                        "pred_disp_px": pred_disp,
                        "pred_depth_m": pred_depth,
                        "entity_disp_abs_error_px": disp_abs_error,
                        "entity_depth_abs_error_m": depth_abs_error,
                        "patch_valid_ratio": patch_valid_ratio,
                    }
                )
    finally:
        env.close()
    return frame_rows, entity_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a small fixed-stereo SGBM audit set from bootstrap datasets.")
    parser.add_argument("--datasets", nargs="+", required=True, help="task_id=dataset_path entries.")
    parser.add_argument("--out", type=Path, default=Path("/mnt/e/reward_model_dataset/stereo_sgbm_audit/bootstrap_v1"))
    parser.add_argument("--max-per-group", type=int, default=2)
    parser.add_argument("--max-total-per-task", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--baseline-m", type=float, default=0.08)
    parser.add_argument("--fov-y-rad", type=float, default=1.0)
    parser.add_argument("--num-disparities", type=int, default=96)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--patch-radius", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--save-stereo-pairs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    all_frame_rows: list[dict[str, Any]] = []
    all_entity_rows: list[dict[str, Any]] = []
    for entry in args.datasets:
        task_id, dataset_text = entry.split("=", 1)
        task_id = task_id.strip().lower()
        if task_id not in ENV_IDS:
            raise ValueError(f"Unknown task_id {task_id}")
        frame_rows, entity_rows = audit_task(args, task_id, Path(dataset_text), args.out)
        all_frame_rows.extend(frame_rows)
        all_entity_rows.extend(entity_rows)

    summary = summarize(all_frame_rows, all_entity_rows)
    write_csv(args.out / "stereo_sgbm_frame_metrics.csv", all_frame_rows)
    write_csv(args.out / "stereo_sgbm_entity_metrics.csv", all_entity_rows)
    write_json(args.out / "stereo_sgbm_audit_summary.json", summary)
    (args.out / "README.txt").write_text(
        "\n".join(
            [
                "Bootstrap stereo SGBM audit",
                "",
                "方法：固定左右相机重新渲染 RGB/depth，OpenCV StereoSGBM + CLAHE 估计视差，再和 ManiSkill depth buffer / object-goal 实体中心投影深度对比。",
                "用途：这是像素级双目质量审计，不是全量训练帧缓存。",
                "",
                "输出：",
                "- stereo_sgbm_frame_metrics.csv: dense disparity/depth metrics per sampled terminal state.",
                "- stereo_sgbm_entity_metrics.csv: object/goal/support entity-center patch metrics.",
                "- previews/: left/right/SGBM/GT disparity/depth preview images.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
