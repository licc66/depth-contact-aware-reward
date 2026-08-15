from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    env_id: str
    video_path: Path
    summary_path: Path
    goal_text: str
    negative_text: str
    stages: tuple[str, ...]


def default_tasks(root: Path) -> list[TaskSpec]:
    base = root / "paper_style_tasks" / "outputs" / "wsl_motionplanning"
    return [
        TaskSpec(
            task_id="stackcube",
            env_id="StackCube-v1",
            video_path=base / "StackCube-v1" / "motionplanning" / "stackcube_wsl_mplib.mp4",
            summary_path=base / "StackCube-v1" / "motionplanning" / "stackcube_wsl_mplib_run_summary.json",
            goal_text="A robot arm stacks one cube on top of another cube and releases it stably.",
            negative_text="The robot manipulation task is not completed yet.",
            stages=(
                "the robot arm approaches the cube on the table",
                "the gripper grasps the cube",
                "the robot lifts the cube above the table",
                "the robot moves the cube over the target cube",
                "the robot places the cube on top of the target cube",
                "the robot releases the cube and the stacked cubes are stable",
            ),
        ),
        TaskSpec(
            task_id="stackpyramid",
            env_id="StackPyramid-v1",
            video_path=base / "StackPyramid-v1" / "motionplanning" / "stackpyramid_wsl_mplib.mp4",
            summary_path=base / "StackPyramid-v1" / "motionplanning" / "stackpyramid_wsl_mplib_run_summary.json",
            goal_text="A robot arm builds a stable three-cube pyramid by placing the top cube on the base cubes.",
            negative_text="The robot has not finished building the cube pyramid.",
            stages=(
                "the robot observes the three cubes on the table",
                "the robot approaches the cube that will be moved",
                "the gripper grasps and lifts the cube",
                "the robot carries the cube above the base cubes",
                "the robot places the cube on the base cubes to form a pyramid",
                "the robot releases the cube and the pyramid is stable",
            ),
        ),
        TaskSpec(
            task_id="peginsertion",
            env_id="PegInsertionSide-v1",
            video_path=base / "PegInsertionSide-v1" / "motionplanning" / "peg_insertion_wsl_mplib.mp4",
            summary_path=base / "PegInsertionSide-v1" / "motionplanning" / "peg_insertion_wsl_mplib_run_summary.json",
            goal_text="A robot arm picks up a peg, aligns it with a hole, inserts the peg, and releases it stably.",
            negative_text="The peg insertion task is not completed yet.",
            stages=(
                "the robot approaches the peg",
                "the gripper grasps the peg",
                "the robot lifts the peg",
                "the robot transports the peg toward the hole",
                "the robot aligns the peg with the hole",
                "the robot inserts the peg into the hole",
                "the robot releases the inserted peg and the task is stable",
            ),
        ),
    ]


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def saved_record(summary: dict[str, Any]) -> dict[str, Any]:
    records = summary.get("records", [])
    success_records = [record for record in records if record.get("success")]
    return success_records[-1] if success_records else (records[-1] if records else {})


def load_video_frames(path: Path) -> tuple[list[np.ndarray], float]:
    meta = iio.immeta(path)
    fps = float(meta.get("fps") or 30.0)
    frames = [frame for frame in iio.imiter(path)]
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames, fps


def sample_indices(total_frames: int, num_samples: int) -> list[int]:
    if total_frames <= num_samples:
        return list(range(total_frames))
    return sorted(set(int(round(x)) for x in np.linspace(0, total_frames - 1, num_samples)))


def cosine_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def encode_images(model: torch.nn.Module, preprocess, frames: list[np.ndarray], batch_size: int) -> torch.Tensor:
    features: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            images = torch.stack([preprocess(Image.fromarray(frame).convert("RGB")) for frame in batch])
            feats = model.encode_image(images)
            features.append(cosine_normalize(feats.float()).cpu())
    return torch.cat(features, dim=0)


def encode_text(model: torch.nn.Module, tokenizer, texts: list[str]) -> torch.Tensor:
    with torch.no_grad():
        tokens = tokenizer(texts)
        feats = model.encode_text(tokens)
    return cosine_normalize(feats.float()).cpu()


def minmax(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if math.isclose(lo, hi):
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = (x - np.max(x)) / max(temperature, 1e-6)
    ez = np.exp(z)
    return ez / np.sum(ez)


def pairwise_order_accuracy(scores: np.ndarray) -> float:
    correct = 0
    total = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            correct += float(scores[j] >= scores[i])
            total += 1
    return correct / total if total else 0.0


def spearman_vs_time(time_progress: np.ndarray, scores: np.ndarray) -> float:
    corr = spearmanr(time_progress, scores).correlation
    return 0.0 if math.isnan(float(corr)) else float(corr)


def progress_mae(time_progress: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean(np.abs(scores - time_progress)))


def load_clip_model(model_name: str, pretrained: str, download_source: str):
    import open_clip

    if download_source == "direct":
        from open_clip.pretrained import download_pretrained, get_pretrained_cfg

        pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
        if not pretrained_cfg:
            raise RuntimeError(f"No pretrained config for {model_name}/{pretrained}")
        checkpoint_path = download_pretrained(pretrained_cfg, prefer_hf_hub=False)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=checkpoint_path,
            weights_only=False,
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            weights_only=False,
        )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
