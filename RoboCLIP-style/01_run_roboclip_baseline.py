from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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


def pairwise_order_accuracy(scores: np.ndarray) -> float:
    correct = 0
    total = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            correct += float(scores[j] >= scores[i])
            total += 1
    return correct / total if total else 0.0


def softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = (x - np.max(x)) / max(temperature, 1e-6)
    ez = np.exp(z)
    return ez / np.sum(ez)


def save_contact_sheet(samples: list[np.ndarray], labels: list[str], out_path: Path, cols: int = 6) -> None:
    thumbs = [Image.fromarray(frame).convert("RGB").resize((180, 180)) for frame in samples]
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 180, rows * 214), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 180
        y = (idx // cols) * 214
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + 184), labels[idx], fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def plot_curves(rows: list[dict[str, Any]], out_path: Path) -> None:
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(9, 3.2 * len(task_ids)), sharex=False)
    if len(task_ids) == 1:
        axes = [axes]
    for ax, task_id in zip(axes, task_ids):
        data = [row for row in rows if row["task_id"] == task_id]
        xs = [float(row["time_progress"]) for row in data]
        ax.plot(xs, [float(row["roboclip_score_norm"]) for row in data], marker="o", label="prefix RoboCLIP score")
        ax.plot(xs, [float(row["success_visual_similarity_norm"]) for row in data], marker="s", label="prefix-to-final visual sim")
        ax.plot(xs, [float(row["done_probability"]) for row in data], marker="^", label="prefix text done prob")
        ax.set_title(task_id)
        ax.set_xlabel("time-normalized progress")
        ax.set_ylabel("score")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_stage_heatmaps(task_stage_scores: dict[str, np.ndarray], stage_names: dict[str, list[str]], out_path: Path) -> None:
    task_ids = list(task_stage_scores)
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(10, 3.5 * len(task_ids)), sharex=False)
    if len(task_ids) == 1:
        axes = [axes]
    for ax, task_id in zip(axes, task_ids):
        scores = task_stage_scores[task_id]
        im = ax.imshow(scores.T, aspect="auto", cmap="viridis")
        ax.set_title(f"{task_id}: frame-to-stage text similarity")
        ax.set_xlabel("sample index")
        ax.set_ylabel("stage text")
        ax.set_yticks(range(len(stage_names[task_id])))
        ax.set_yticklabels([f"{idx}: {name[:36]}" for idx, name in enumerate(stage_names[task_id])], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboCLIP-style CLIP reward baseline on the three ManiSkill videos.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("RoboCLIP-style") / "outputs" / "three_task_roboclip")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--final-ref-frames", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--download-source", choices=("direct", "default"), default="direct")
    parser.add_argument("--temperature", type=float, default=0.03)
    args = parser.parse_args()

    try:
        import open_clip
    except ImportError as exc:
        raise SystemExit(
            "Missing open_clip. Install it in WSL with: "
            "source paper_style_tasks/wsl_env.sh && python -m pip install open_clip_torch==3.2.0"
        ) from exc

    root = args.root.resolve()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CLIP model {args.model} pretrained={args.pretrained} source={args.download_source}")
    if args.download_source == "direct":
        from open_clip.pretrained import download_pretrained, get_pretrained_cfg

        pretrained_cfg = get_pretrained_cfg(args.model, args.pretrained)
        if not pretrained_cfg:
            raise RuntimeError(f"No pretrained config for {args.model}/{args.pretrained}")
        checkpoint_path = download_pretrained(pretrained_cfg, prefer_hf_hub=False)
        print(f"Using checkpoint: {checkpoint_path}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model,
            pretrained=checkpoint_path,
            weights_only=False,
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model,
            pretrained=args.pretrained,
            weights_only=False,
        )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    prompt_config: dict[str, Any] = {
        "model": args.model,
        "pretrained": args.pretrained,
        "num_samples": args.num_samples,
        "final_ref_frames": args.final_ref_frames,
        "note": "RoboCLIP-style zero-shot baseline: no policy training and no CLIP fine-tuning.",
        "tasks": {},
    }
    heatmaps: dict[str, np.ndarray] = {}
    heatmap_stage_names: dict[str, list[str]] = {}

    for task in default_tasks(root):
        if not task.video_path.exists():
            raise FileNotFoundError(task.video_path)
        summary = read_summary(task.summary_path)
        frames, fps = load_video_frames(task.video_path)
        indices = sample_indices(len(frames), args.num_samples)
        sampled_frames = [frames[idx] for idx in indices]
        labels = [f"{i:02d} frame={idx}" for i, idx in enumerate(indices)]
        save_contact_sheet(sampled_frames, labels, out_dir / "contact_sheets" / f"{task.task_id}.jpg")

        image_features = encode_images(model, preprocess, sampled_frames, args.batch_size)
        prefix_features: list[torch.Tensor] = []
        running = torch.zeros_like(image_features[0])
        for idx in range(len(image_features)):
            running = running + image_features[idx]
            prefix_features.append(cosine_normalize((running / (idx + 1)).unsqueeze(0)).squeeze(0))
        prefix_features_tensor = torch.stack(prefix_features)
        ref_count = min(args.final_ref_frames, len(image_features))
        final_ref = cosine_normalize(image_features[-ref_count:].mean(dim=0, keepdim=True))

        text_features = encode_text(model, tokenizer, [task.goal_text, task.negative_text, *task.stages])
        goal_feature = text_features[0:1]
        negative_feature = text_features[1:2]
        stage_features = text_features[2:]

        frame_goal_sim = (image_features @ goal_feature.T).squeeze(1).numpy()
        goal_sim = (prefix_features_tensor @ goal_feature.T).squeeze(1).numpy()
        neg_sim = (prefix_features_tensor @ negative_feature.T).squeeze(1).numpy()
        done_logits = np.stack([neg_sim, goal_sim], axis=1) / max(args.temperature, 1e-6)
        done_logits = done_logits - done_logits.max(axis=1, keepdims=True)
        done_probs = np.exp(done_logits)
        done_probs = done_probs / done_probs.sum(axis=1, keepdims=True)
        done_probability = done_probs[:, 1]

        frame_success_visual = (image_features @ final_ref.T).squeeze(1).numpy()
        success_visual = (prefix_features_tensor @ final_ref.T).squeeze(1).numpy()
        stage_scores = (image_features @ stage_features.T).numpy()
        heatmaps[task.task_id] = stage_scores
        heatmap_stage_names[task.task_id] = list(task.stages)

        predicted_stage_ids = stage_scores.argmax(axis=1)
        weighted_stage_progress = []
        for score_row in stage_scores:
            weights = softmax(score_row, args.temperature)
            denom = max(len(task.stages) - 1, 1)
            weighted_stage_progress.append(float(np.dot(weights, np.arange(len(task.stages)) / denom)))
        weighted_stage_progress = np.array(weighted_stage_progress)

        roboclip_score_raw = 0.5 * goal_sim + 0.5 * success_visual
        goal_sim_norm = minmax(goal_sim)
        success_visual_norm = minmax(success_visual)
        roboclip_score_norm = minmax(roboclip_score_raw)

        time_progress = np.array(indices, dtype=np.float32) / max(len(frames) - 1, 1)
        spearman = spearmanr(time_progress, roboclip_score_norm).correlation
        if math.isnan(float(spearman)):
            spearman = 0.0
        order_acc = pairwise_order_accuracy(roboclip_score_norm)

        records = summary.get("records", [])
        success_records = [record for record in records if record.get("success")]
        saved_record = success_records[-1] if success_records else (records[-1] if records else {})

        for sample_i, frame_idx in enumerate(indices):
            sample_rows.append(
                {
                    "task_id": task.task_id,
                    "env_id": task.env_id,
                    "video_path": str(task.video_path),
                    "sample_index": sample_i,
                    "frame_index": frame_idx,
                    "time_sec": frame_idx / fps,
                    "time_progress": float(time_progress[sample_i]),
                    "frame_goal_text_similarity": float(frame_goal_sim[sample_i]),
                    "frame_success_visual_similarity": float(frame_success_visual[sample_i]),
                    "goal_text_similarity": float(goal_sim[sample_i]),
                    "goal_text_similarity_norm": float(goal_sim_norm[sample_i]),
                    "success_visual_similarity": float(success_visual[sample_i]),
                    "success_visual_similarity_norm": float(success_visual_norm[sample_i]),
                    "done_probability": float(done_probability[sample_i]),
                    "roboclip_score_raw": float(roboclip_score_raw[sample_i]),
                    "roboclip_score_norm": float(roboclip_score_norm[sample_i]),
                    "predicted_stage_id": int(predicted_stage_ids[sample_i]),
                    "predicted_stage_name": task.stages[int(predicted_stage_ids[sample_i])],
                    "weighted_stage_progress": float(weighted_stage_progress[sample_i]),
                    "max_stage_similarity": float(stage_scores[sample_i, predicted_stage_ids[sample_i]]),
                }
            )

        summary_rows.append(
            {
                "task_id": task.task_id,
                "env_id": task.env_id,
                "video_path": str(task.video_path),
                "num_video_frames": len(frames),
                "fps": fps,
                "duration_sec": len(frames) / fps,
                "sampled_frames": len(indices),
                "episode_success": bool(saved_record.get("success", False)),
                "first_success_seed": saved_record.get("seed"),
                "attempts": summary.get("attempts"),
                "elapsed_steps": saved_record.get("elapsed_steps"),
                "roboclip_spearman_vs_time": float(spearman),
                "roboclip_pairwise_order_acc_vs_time": float(order_acc),
                "roboclip_score_start": float(roboclip_score_norm[0]),
                "roboclip_score_end": float(roboclip_score_norm[-1]),
                "roboclip_score_delta": float(roboclip_score_norm[-1] - roboclip_score_norm[0]),
                "goal_text_sim_start": float(goal_sim[0]),
                "goal_text_sim_end": float(goal_sim[-1]),
                "success_visual_sim_start": float(success_visual[0]),
                "success_visual_sim_end": float(success_visual[-1]),
                "done_probability_start": float(done_probability[0]),
                "done_probability_end": float(done_probability[-1]),
                "predicted_final_stage_id": int(predicted_stage_ids[-1]),
                "predicted_final_stage_name": task.stages[int(predicted_stage_ids[-1])],
                "note": "Metrics are against time-normalized proxy progress, not environment-truth stage labels.",
            }
        )

        prompt_config["tasks"][task.task_id] = {
            "env_id": task.env_id,
            "goal_text": task.goal_text,
            "negative_text": task.negative_text,
            "stages": list(task.stages),
        }

    sample_csv = out_dir / "roboclip_samples.csv"
    summary_csv = out_dir / "roboclip_summary.csv"
    if sample_rows:
        with sample_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sample_rows[0]))
            writer.writeheader()
            writer.writerows(sample_rows)
    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

    (out_dir / "prompt_config.json").write_text(json.dumps(prompt_config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "roboclip_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_curves(sample_rows, out_dir / "roboclip_score_curves.png")
    plot_stage_heatmaps(heatmaps, heatmap_stage_names, out_dir / "stage_similarity_heatmaps.png")

    print(f"Saved samples: {sample_csv}")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved curves: {out_dir / 'roboclip_score_curves.png'}")
    print(f"Saved heatmaps: {out_dir / 'stage_similarity_heatmaps.png'}")
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
