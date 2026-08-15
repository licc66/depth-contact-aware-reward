from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien
from mani_skill.utils import sapien_utils


def load_truth_module():
    path = Path(__file__).with_name("01_extract_truth_depth_features.py")
    spec = importlib.util.spec_from_file_location("truth_depth_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


truth = load_truth_module()


@dataclass(frozen=True)
class StereoImageRig:
    center_eye: np.ndarray
    look_at: np.ndarray
    up: np.ndarray
    baseline_m: float
    width: int
    height: int
    fov_y_rad: float
    near_m: float
    far_m: float


def make_image_rig(width: int, height: int, baseline_m: float, fov_y_rad: float) -> StereoImageRig:
    return StereoImageRig(
        center_eye=np.array([0.55, -0.65, 0.42], dtype=np.float64),
        look_at=np.array([0.00, 0.00, 0.08], dtype=np.float64),
        up=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        baseline_m=baseline_m,
        width=width,
        height=height,
        fov_y_rad=fov_y_rad,
        near_m=0.01,
        far_m=10.0,
    )


def to_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def camera_x_axis(rig: StereoImageRig) -> np.ndarray:
    z_axis = rig.look_at - rig.center_eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, rig.up)
    return x_axis / np.linalg.norm(x_axis)


def camera_poses(rig: StereoImageRig) -> tuple[sapien.Pose, sapien.Pose]:
    x_axis = camera_x_axis(rig)
    left_eye = rig.center_eye - x_axis * (rig.baseline_m / 2.0)
    right_eye = rig.center_eye + x_axis * (rig.baseline_m / 2.0)
    center_pose = sapien_utils.look_at(rig.center_eye, rig.look_at, rig.up)
    raw = to_np(center_pose.raw_pose).reshape(-1)
    quat = raw[3:7]
    return sapien.Pose(left_eye, quat), sapien.Pose(right_eye, quat)


def make_env(task: Any, rig: StereoImageRig):
    left_pose, right_pose = camera_poses(rig)
    return gym.make(
        task.env_id,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        render_backend="cpu",
        sim_backend="cpu",
        max_episode_steps=1000,
        sensor_configs={
            "base_camera": {
                "pose": left_pose,
                "width": rig.width,
                "height": rig.height,
                "fov": rig.fov_y_rad,
                "near": rig.near_m,
                "far": rig.far_m,
                "shader_pack": "minimal",
            },
            "hand_camera": {
                "pose": right_pose,
                "mount": None,
                "entity_uid": None,
                "width": rig.width,
                "height": rig.height,
                "fov": rig.fov_y_rad,
                "near": rig.near_m,
                "far": rig.far_m,
                "shader_pack": "minimal",
            },
        },
    )


def read_sensor_pair(env) -> dict[str, Any]:
    scene = env.unwrapped.scene
    scene.update_render(update_sensors=True, update_human_render_cameras=False)
    out: dict[str, Any] = {}
    for uid, alias in [("base_camera", "left"), ("hand_camera", "right")]:
        sensor = scene.sensors[uid]
        sensor.capture()
        obs = sensor.get_obs(rgb=True, depth=True, position=False, segmentation=False)
        out[f"{alias}_rgb"] = to_np(obs["rgb"])[0, ..., :3].astype(np.uint8)
        out[f"{alias}_depth_m"] = to_np(obs["depth"])[0, ..., 0].astype(np.float32) / 1000.0
        params = sensor.get_params()
        out[f"{alias}_intrinsic"] = to_np(params["intrinsic_cv"])[0].astype(np.float64)
        out[f"{alias}_extrinsic"] = to_np(params["extrinsic_cv"])[0].astype(np.float64)
    return out


def create_sgbm(num_disparities: int, block_size: int):
    num_disparities = int(math.ceil(num_disparities / 16.0) * 16)
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=6,
        speckleWindowSize=80,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity(left_rgb: np.ndarray, right_rgb: np.ndarray, num_disparities: int, block_size: int) -> np.ndarray:
    left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    left_eq = clahe.apply(left_gray)
    right_eq = clahe.apply(right_gray)
    matcher = create_sgbm(num_disparities=num_disparities, block_size=block_size)
    disparity = matcher.compute(left_eq, right_eq).astype(np.float32) / 16.0
    disparity[disparity <= 0.0] = np.nan
    return disparity


def project_point(point_world: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> dict[str, float]:
    point = np.append(np.asarray(point_world, dtype=np.float64), 1.0)
    xyz = extrinsic @ point
    if xyz[2] <= 1e-6:
        return {"u": float("nan"), "v": float("nan"), "z": float("nan")}
    uvw = intrinsic @ xyz
    return {
        "u": float(uvw[0] / uvw[2]),
        "v": float(uvw[1] / uvw[2]),
        "z": float(xyz[2]),
    }


def patch_median(disparity: np.ndarray, u: float, v: float, radius: int) -> tuple[float, float]:
    if not np.isfinite(u) or not np.isfinite(v):
        return float("nan"), 0.0
    x = int(round(u))
    y = int(round(v))
    if x < 0 or y < 0 or x >= disparity.shape[1] or y >= disparity.shape[0]:
        return float("nan"), 0.0
    x0 = max(0, x - radius)
    x1 = min(disparity.shape[1], x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(disparity.shape[0], y + radius + 1)
    patch = disparity[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0.0)
    if not np.any(valid):
        return float("nan"), 0.0
    return float(np.nanmedian(patch[valid])), float(np.mean(valid))


def dense_metrics(disparity: np.ndarray, left_depth_m: np.ndarray, focal_px: float, baseline_m: float, far_m: float) -> dict[str, float]:
    gt_depth = left_depth_m.astype(np.float64)
    gt_valid = (gt_depth > 0.05) & (gt_depth < min(far_m * 0.95, 3.0))
    gt_disp = np.full_like(gt_depth, np.nan, dtype=np.float64)
    gt_disp[gt_valid] = focal_px * baseline_m / gt_depth[gt_valid]
    pred_valid = np.isfinite(disparity) & (disparity > 0.0)
    valid = gt_valid & pred_valid
    if not np.any(valid):
        return {
            "dense_valid_ratio": 0.0,
            "dense_disp_mae_px": float("nan"),
            "dense_disp_rmse_px": float("nan"),
            "dense_depth_mae_m": float("nan"),
            "dense_depth_rmse_m": float("nan"),
            "dense_bad_1px_rate": float("nan"),
            "dense_bad_3px_rate": float("nan"),
        }
    disp_err = np.asarray(disparity, dtype=np.float64)[valid] - gt_disp[valid]
    pred_depth = focal_px * baseline_m / np.asarray(disparity, dtype=np.float64)[valid]
    depth_err = pred_depth - gt_depth[valid]
    return {
        "dense_valid_ratio": float(np.mean(valid & gt_valid) / max(float(np.mean(gt_valid)), 1e-9)),
        "dense_disp_mae_px": float(np.mean(np.abs(disp_err))),
        "dense_disp_rmse_px": float(np.sqrt(np.mean(disp_err**2))),
        "dense_depth_mae_m": float(np.mean(np.abs(depth_err))),
        "dense_depth_rmse_m": float(np.sqrt(np.mean(depth_err**2))),
        "dense_bad_1px_rate": float(np.mean(np.abs(disp_err) > 1.0)),
        "dense_bad_3px_rate": float(np.mean(np.abs(disp_err) > 3.0)),
    }


def sample_indices(num_actions: int, samples_per_task: int) -> set[int]:
    if samples_per_task <= 0:
        return set(range(num_actions + 1))
    raw = np.linspace(0, num_actions, samples_per_task)
    return set(int(round(x)) for x in raw)


def draw_preview(
    path: Path,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    disparity: np.ndarray,
    left_depth_m: np.ndarray,
    focal_px: float,
    baseline_m: float,
) -> None:
    gt_disp = np.where(left_depth_m > 0.05, focal_px * baseline_m / np.maximum(left_depth_m, 1e-6), np.nan)
    err = np.abs(disparity - gt_disp)
    vmax = np.nanpercentile(gt_disp, 98) if np.any(np.isfinite(gt_disp)) else 64.0
    err_max = np.nanpercentile(err, 95) if np.any(np.isfinite(err)) else 10.0
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes[0, 0].imshow(left_rgb)
    axes[0, 0].set_title("left RGB")
    axes[0, 1].imshow(right_rgb)
    axes[0, 1].set_title("right RGB")
    im0 = axes[0, 2].imshow(disparity, cmap="magma", vmin=0, vmax=vmax)
    axes[0, 2].set_title("SGBM disparity")
    fig.colorbar(im0, ax=axes[0, 2], fraction=0.046, pad=0.04)
    im1 = axes[1, 0].imshow(gt_disp, cmap="magma", vmin=0, vmax=vmax)
    axes[1, 0].set_title("GT disparity from depth")
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    im2 = axes[1, 1].imshow(err, cmap="inferno", vmin=0, vmax=err_max)
    axes[1, 1].set_title("absolute disparity error")
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 2].imshow(left_depth_m, cmap="viridis", vmin=0, vmax=2.0)
    axes[1, 2].set_title("left depth GT (m)")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rgb_pair(out_dir: Path, task_id: str, step: int, left_rgb: np.ndarray, right_rgb: np.ndarray) -> None:
    pair_dir = out_dir / "stereo_pairs" / task_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(pair_dir / f"{task_id}_step_{step:04d}_left.png", left_rgb)
    imageio.imwrite(pair_dir / f"{task_id}_step_{step:04d}_right.png", right_rgb)


def safe_mean(values: list[float]) -> float:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_median(values: list[float]) -> float:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def summarize(sample_rows: list[dict[str, Any]], entity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    task_ids = list(dict.fromkeys(row["task_id"] for row in sample_rows))
    for task_id in task_ids:
        task_samples = [row for row in sample_rows if row["task_id"] == task_id]
        task_entities = [row for row in entity_rows if row["task_id"] == task_id]
        valid_entities = [row for row in task_entities if np.isfinite(row["pred_disp_px"])]
        depth_errors = [row["entity_depth_abs_error_m"] for row in valid_entities]
        disp_errors = [row["entity_disp_abs_error_px"] for row in valid_entities]
        summary.append(
            {
                "task_id": task_id,
                "sample_frames": len(task_samples),
                "entity_points": len(task_entities),
                "valid_entity_points": len(valid_entities),
                "entity_valid_rate": len(valid_entities) / max(len(task_entities), 1),
                "entity_disp_mae_px": safe_mean(disp_errors),
                "entity_disp_median_abs_error_px": safe_median(disp_errors),
                "entity_depth_mae_m": safe_mean(depth_errors),
                "entity_depth_median_abs_error_m": safe_median(depth_errors),
                "dense_valid_ratio": safe_mean([row["dense_valid_ratio"] for row in task_samples]),
                "dense_disp_mae_px": safe_mean([row["dense_disp_mae_px"] for row in task_samples]),
                "dense_depth_mae_m": safe_mean([row["dense_depth_mae_m"] for row in task_samples]),
                "dense_bad_1px_rate": safe_mean([row["dense_bad_1px_rate"] for row in task_samples]),
                "dense_bad_3px_rate": safe_mean([row["dense_bad_3px_rate"] for row in task_samples]),
            }
        )
    return summary


def summarize_entities(entity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    keys = list(dict.fromkeys((row["task_id"], row["entity"]) for row in entity_rows))
    for task_id, entity in keys:
        rows = [row for row in entity_rows if row["task_id"] == task_id and row["entity"] == entity]
        valid = [row for row in rows if np.isfinite(row["pred_disp_px"])]
        summary.append(
            {
                "task_id": task_id,
                "entity": entity,
                "points": len(rows),
                "valid_points": len(valid),
                "valid_rate": len(valid) / max(len(rows), 1),
                "disp_mae_px": safe_mean([row["entity_disp_abs_error_px"] for row in valid]),
                "disp_median_abs_error_px": safe_median([row["entity_disp_abs_error_px"] for row in valid]),
                "depth_mae_m": safe_mean([row["entity_depth_abs_error_m"] for row in valid]),
                "depth_median_abs_error_m": safe_median([row["entity_depth_abs_error_m"] for row in valid]),
                "depth_buffer_vs_entity_center_mae_m": safe_mean(
                    [row["depth_buffer_vs_entity_center_abs_error_m"] for row in valid]
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_results(path: Path, summary_rows: list[dict[str, Any]], rig: StereoImageRig, method: dict[str, Any]) -> None:
    lines = [
        "Stereo SGBM disparity vs ManiSkill truth",
        "",
        "Method:",
        "OpenCV StereoSGBM with rectified, parallel fixed cameras and CLAHE preprocessing.",
        "",
        "Rig:",
        f"image_size={rig.width}x{rig.height}, baseline_m={rig.baseline_m}, fov_y_rad={rig.fov_y_rad}",
        f"center_eye={rig.center_eye.tolist()}, look_at={rig.look_at.tolist()}",
        "",
        "SGBM:",
        json.dumps(method, ensure_ascii=False),
        "",
        "Summary:",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"- {row['task_id']}:",
                f"  dense depth MAE = {row['dense_depth_mae_m']:.4f} m, dense disparity MAE = {row['dense_disp_mae_px']:.2f} px, valid ratio = {row['dense_valid_ratio']:.3f}",
                f"  entity depth MAE = {row['entity_depth_mae_m']:.4f} m, entity disparity MAE = {row['entity_disp_mae_px']:.2f} px, valid entity rate = {row['entity_valid_rate']:.3f}",
            ]
        )
    lines.extend(["", "Entity-level split is saved in stereo_sgbm_entity_summary.csv."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def replay_and_compare(task: Any, rig: StereoImageRig, args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actions = truth.read_actions(task.h5_path)
    seed = truth.read_success_seed(task.summary_path)
    wanted_steps = sample_indices(len(actions), args.samples_per_task)
    env = make_env(task, rig)
    env.reset(seed=seed)
    sample_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []

    def maybe_record(step: int) -> None:
        if step not in wanted_steps:
            return
        frame = read_sensor_pair(env)
        left_rgb = frame["left_rgb"]
        right_rgb = frame["right_rgb"]
        left_depth_m = frame["left_depth_m"]
        left_intrinsic = frame["left_intrinsic"]
        right_intrinsic = frame["right_intrinsic"]
        left_extrinsic = frame["left_extrinsic"]
        right_extrinsic = frame["right_extrinsic"]
        focal_px = float(left_intrinsic[0, 0])
        disparity = compute_disparity(left_rgb, right_rgb, args.num_disparities, args.block_size)
        dense = dense_metrics(disparity, left_depth_m, focal_px, rig.baseline_m, rig.far_m)
        row = {
            "task_id": task.task_id,
            "env_id": task.env_id,
            "seed": seed,
            "step": step,
            "time_progress": step / max(len(actions), 1),
            "focal_px": focal_px,
            **dense,
        }
        sample_rows.append(row)
        if args.save_pairs:
            save_rgb_pair(out_dir, task.task_id, step, left_rgb, right_rgb)
        if step in {min(wanted_steps), max(wanted_steps)} or len(sample_rows) <= args.preview_per_task:
            preview_path = out_dir / "previews" / f"{task.task_id}_step_{step:04d}.png"
            draw_preview(preview_path, left_rgb, right_rgb, disparity, left_depth_m, focal_px, rig.baseline_m)

        entities = truth.get_entities(env, task.task_id)
        for entity_name in ["tcp", "object", "goal"]:
            point = np.asarray(entities[entity_name], dtype=np.float64)
            left_proj = project_point(point, left_intrinsic, left_extrinsic)
            right_proj = project_point(point, right_intrinsic, right_extrinsic)
            pred_disp, patch_valid_ratio = patch_median(disparity, left_proj["u"], left_proj["v"], args.patch_radius)
            gt_disp = left_proj["u"] - right_proj["u"] if np.isfinite(left_proj["u"]) and np.isfinite(right_proj["u"]) else float("nan")
            if np.isfinite(pred_disp) and pred_disp > 0:
                pred_depth = focal_px * rig.baseline_m / pred_disp
                depth_abs_error = abs(pred_depth - left_proj["z"])
                disp_abs_error = abs(pred_disp - gt_disp)
            else:
                pred_depth = float("nan")
                depth_abs_error = float("nan")
                disp_abs_error = float("nan")
            px = int(round(left_proj["u"])) if np.isfinite(left_proj["u"]) else -1
            py = int(round(left_proj["v"])) if np.isfinite(left_proj["v"]) else -1
            if 0 <= px < left_depth_m.shape[1] and 0 <= py < left_depth_m.shape[0]:
                surface_depth_m = float(left_depth_m[py, px])
                surface_center_abs_error = abs(surface_depth_m - left_proj["z"]) if surface_depth_m > 0 else float("nan")
            else:
                surface_depth_m = float("nan")
                surface_center_abs_error = float("nan")
            entity_rows.append(
                {
                    "task_id": task.task_id,
                    "env_id": task.env_id,
                    "seed": seed,
                    "step": step,
                    "time_progress": step / max(len(actions), 1),
                    "entity": entity_name,
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
                    "left_depth_buffer_at_point_m": surface_depth_m,
                    "depth_buffer_vs_entity_center_abs_error_m": surface_center_abs_error,
                }
            )

    maybe_record(0)
    for idx, action in enumerate(actions, start=1):
        env.step(action)
        maybe_record(idx)
    env.close()
    return sample_rows, entity_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Render fixed stereo image pairs, compute SGBM disparity, and compare with ManiSkill truth.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("stereo_depth_truth") / "outputs" / "stereo_sgbm_compare")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--baseline-m", type=float, default=0.08)
    parser.add_argument("--fov-y-rad", type=float, default=1.0)
    parser.add_argument("--samples-per-task", type=int, default=12)
    parser.add_argument("--num-disparities", type=int, default=96)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--patch-radius", type=int, default=5)
    parser.add_argument("--preview-per-task", type=int, default=3)
    parser.add_argument("--save-pairs", action="store_true", help="Save left/right RGB pairs for every sampled frame.")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rig = make_image_rig(args.width, args.height, args.baseline_m, args.fov_y_rad)
    all_sample_rows: list[dict[str, Any]] = []
    all_entity_rows: list[dict[str, Any]] = []

    for task in truth.default_tasks(root):
        print(f"Replaying {task.task_id} and rendering stereo samples")
        sample_rows, entity_rows = replay_and_compare(task, rig, args, out_dir)
        all_sample_rows.extend(sample_rows)
        all_entity_rows.extend(entity_rows)

    summary_rows = summarize(all_sample_rows, all_entity_rows)
    method = {
        "algorithm": "OpenCV StereoSGBM",
        "preprocessing": "CLAHE on grayscale left/right images",
        "num_disparities": int(math.ceil(args.num_disparities / 16.0) * 16),
        "block_size": args.block_size,
        "patch_radius": args.patch_radius,
    }
    write_csv(out_dir / "stereo_sgbm_frame_metrics.csv", all_sample_rows)
    write_csv(out_dir / "stereo_sgbm_entity_metrics.csv", all_entity_rows)
    write_csv(out_dir / "stereo_sgbm_summary.csv", summary_rows)
    (out_dir / "stereo_sgbm_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    entity_summary_rows = summarize_entities(all_entity_rows)
    write_csv(out_dir / "stereo_sgbm_entity_summary.csv", entity_summary_rows)
    (out_dir / "stereo_sgbm_entity_summary.json").write_text(
        json.dumps(entity_summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "stereo_sgbm_config.json").write_text(
        json.dumps(
            {
                "rig": {
                    "center_eye": rig.center_eye.tolist(),
                    "look_at": rig.look_at.tolist(),
                    "up": rig.up.tolist(),
                    "baseline_m": rig.baseline_m,
                    "width": rig.width,
                    "height": rig.height,
                    "fov_y_rad": rig.fov_y_rad,
                    "near_m": rig.near_m,
                    "far_m": rig.far_m,
                },
                "method": method,
                "note": "base_camera is used as the fixed left camera; hand_camera is detached and reused as the fixed right camera.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_results(out_dir / "RESULTS.txt", summary_rows, rig, method)
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
