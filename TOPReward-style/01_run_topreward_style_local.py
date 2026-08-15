from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from zero_shot_baseline_common import (
    cosine_normalize,
    default_tasks,
    encode_images,
    encode_text,
    load_clip_model,
    load_video_frames,
    pairwise_order_accuracy,
    progress_mae,
    read_summary,
    sample_indices,
    saved_record,
    spearman_vs_time,
    write_csv,
)


def build_binary_prompts(goal_text: str, negative_text: str) -> tuple[str, str]:
    done = f"The robot manipulation task is complete and successful: {goal_text}"
    not_done = f"The robot manipulation task is not complete yet: {negative_text}"
    return done, not_done


def prefix_average_features(image_features: torch.Tensor) -> torch.Tensor:
    prefix_features: list[torch.Tensor] = []
    running = torch.zeros_like(image_features[0])
    for idx in range(len(image_features)):
        running = running + image_features[idx]
        prefix_features.append(cosine_normalize((running / (idx + 1)).unsqueeze(0)).squeeze(0))
    return torch.stack(prefix_features)


def two_class_probability(negative_sim: np.ndarray, positive_sim: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.stack([negative_sim, positive_sim], axis=1) / max(temperature, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[:, 1]


def plot_curves(rows: list[dict[str, Any]], out_path: Path) -> None:
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(9, 3.1 * len(task_ids)), sharex=False)
    if len(task_ids) == 1:
        axes = [axes]
    for ax, task_id in zip(axes, task_ids):
        data = [row for row in rows if row["task_id"] == task_id]
        x = [float(row["time_progress"]) for row in data]
        ax.plot(x, [float(row["topreward_prefix_done_probability"]) for row in data], marker="o", label="prefix done prob")
        ax.plot(x, [float(row["topreward_frame_done_probability"]) for row in data], marker="s", label="frame done prob")
        ax.plot(x, x, "--", color="#9ca3af", linewidth=1, label="time proxy")
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


def write_results(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "TOPReward-style local CLIP baseline",
        "",
        "Method:",
        "For each sampled trajectory prefix, average OpenCLIP image embeddings and score the prefix against complete vs not-complete text prompts.",
        "This is a local weak reproduction of trajectory-prefix reward scoring. Metrics are against time-normalized proxy progress, not environment true progress.",
        "",
        "Summary:",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"- {row['task_id']}:",
                f"  Spearman vs time = {row['topreward_spearman_vs_time']:.3f}, pairwise order acc = {row['topreward_pairwise_order_acc_vs_time']:.3f}, MAE vs time = {row['topreward_progress_mae_vs_time']:.3f}",
                f"  start = {row['topreward_score_start']:.3f}, end = {row['topreward_score_end']:.3f}, delta = {row['topreward_score_delta']:.3f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="TOPReward-style local zero-shot trajectory-prefix baseline.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("TOPReward-style") / "outputs" / "three_task_topreward_local")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--download-source", choices=("direct", "default"), default="direct")
    parser.add_argument("--temperature", type=float, default=0.03)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CLIP model {args.model} pretrained={args.pretrained} source={args.download_source}")
    model, preprocess, tokenizer = load_clip_model(args.model, args.pretrained, args.download_source)

    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    config: dict[str, Any] = {
        "method": "TOPReward-style local weak reproduction with OpenCLIP trajectory-prefix binary prompts.",
        "model": args.model,
        "pretrained": args.pretrained,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "note": "Zero-shot baseline; no policy training, no CLIP fine-tuning, no environment truth as input.",
        "tasks": {},
    }

    for task in default_tasks(root):
        if not task.video_path.exists():
            raise FileNotFoundError(task.video_path)
        summary = read_summary(task.summary_path)
        record = saved_record(summary)
        frames, fps = load_video_frames(task.video_path)
        indices = sample_indices(len(frames), args.num_samples)
        sampled_frames = [frames[idx] for idx in indices]
        time_progress = np.array(indices, dtype=np.float64) / max(len(frames) - 1, 1)

        image_features = encode_images(model, preprocess, sampled_frames, args.batch_size)
        prefix_features = prefix_average_features(image_features)
        done_prompt, not_done_prompt = build_binary_prompts(task.goal_text, task.negative_text)
        text_features = encode_text(model, tokenizer, [not_done_prompt, done_prompt])
        negative_feature = text_features[0:1]
        positive_feature = text_features[1:2]

        frame_done_sim = (image_features @ positive_feature.T).squeeze(1).numpy()
        frame_not_done_sim = (image_features @ negative_feature.T).squeeze(1).numpy()
        prefix_done_sim = (prefix_features @ positive_feature.T).squeeze(1).numpy()
        prefix_not_done_sim = (prefix_features @ negative_feature.T).squeeze(1).numpy()
        frame_done_prob = two_class_probability(frame_not_done_sim, frame_done_sim, args.temperature)
        prefix_done_prob = two_class_probability(prefix_not_done_sim, prefix_done_sim, args.temperature)

        spearman = spearman_vs_time(time_progress, prefix_done_prob)
        order_acc = pairwise_order_accuracy(prefix_done_prob)
        mae = progress_mae(time_progress, prefix_done_prob)

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
                    "topreward_prefix_done_probability": float(prefix_done_prob[sample_i]),
                    "topreward_frame_done_probability": float(frame_done_prob[sample_i]),
                    "prefix_done_similarity": float(prefix_done_sim[sample_i]),
                    "prefix_not_done_similarity": float(prefix_not_done_sim[sample_i]),
                    "frame_done_similarity": float(frame_done_sim[sample_i]),
                    "frame_not_done_similarity": float(frame_not_done_sim[sample_i]),
                }
            )

        config["tasks"][task.task_id] = {
            "env_id": task.env_id,
            "done_prompt": done_prompt,
            "not_done_prompt": not_done_prompt,
        }
        summary_rows.append(
            {
                "task_id": task.task_id,
                "env_id": task.env_id,
                "video_path": str(task.video_path),
                "num_video_frames": len(frames),
                "fps": fps,
                "duration_sec": len(frames) / fps,
                "sampled_frames": len(indices),
                "episode_success": bool(record.get("success", False)),
                "first_success_seed": record.get("seed"),
                "attempts": summary.get("attempts"),
                "elapsed_steps": record.get("elapsed_steps"),
                "topreward_spearman_vs_time": float(spearman),
                "topreward_pairwise_order_acc_vs_time": float(order_acc),
                "topreward_progress_mae_vs_time": float(mae),
                "topreward_score_start": float(prefix_done_prob[0]),
                "topreward_score_end": float(prefix_done_prob[-1]),
                "topreward_score_delta": float(prefix_done_prob[-1] - prefix_done_prob[0]),
                "frame_done_probability_start": float(frame_done_prob[0]),
                "frame_done_probability_end": float(frame_done_prob[-1]),
                "note": "Metrics are against time-normalized proxy progress, not environment-truth stage labels.",
            }
        )

    write_csv(out_dir / "topreward_local_samples.csv", sample_rows)
    write_csv(out_dir / "topreward_local_summary.csv", summary_rows)
    (out_dir / "topreward_local_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "topreward_local_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_curves(sample_rows, out_dir / "topreward_local_progress_curves.png")
    write_results(out_dir / "RESULTS.txt", summary_rows)

    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
