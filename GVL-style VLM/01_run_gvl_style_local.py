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

from zero_shot_baseline_common import (
    default_tasks,
    encode_images,
    encode_text,
    load_clip_model,
    load_video_frames,
    minmax,
    pairwise_order_accuracy,
    progress_mae,
    read_summary,
    sample_indices,
    saved_record,
    softmax,
    spearman_vs_time,
    write_csv,
)


def build_stage_prompts(goal_text: str, stages: tuple[str, ...]) -> list[str]:
    return [f"Goal: {goal_text} Current robot manipulation stage: {stage}." for stage in stages]


def plot_curves(rows: list[dict[str, Any]], out_path: Path) -> None:
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(9, 3.1 * len(task_ids)), sharex=False)
    if len(task_ids) == 1:
        axes = [axes]
    for ax, task_id in zip(axes, task_ids):
        data = [row for row in rows if row["task_id"] == task_id]
        x = [float(row["time_progress"]) for row in data]
        ax.plot(x, [float(row["gvl_stage_progress"]) for row in data], marker="o", label="stage expected progress")
        ax.plot(x, [float(row["gvl_stage_progress_norm"]) for row in data], marker="s", label="minmax normalized")
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


def plot_stage_heatmaps(
    task_stage_scores: dict[str, np.ndarray],
    stage_names: dict[str, list[str]],
    out_path: Path,
) -> None:
    task_ids = list(task_stage_scores)
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(10, 3.5 * len(task_ids)), sharex=False)
    if len(task_ids) == 1:
        axes = [axes]
    for ax, task_id in zip(axes, task_ids):
        scores = task_stage_scores[task_id]
        im = ax.imshow(scores.T, aspect="auto", cmap="viridis")
        ax.set_title(f"{task_id}: GVL-style frame-to-stage similarity")
        ax.set_xlabel("sample index")
        ax.set_ylabel("stage text")
        ax.set_yticks(range(len(stage_names[task_id])))
        ax.set_yticklabels([f"{idx}: {name[:38]}" for idx, name in enumerate(stage_names[task_id])], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_results(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "GVL-style local CLIP baseline",
        "",
        "Method:",
        "For each sampled video frame, encode the image with OpenCLIP and compare it with ordered goal-conditioned stage prompts.",
        "The progress score is the softmax-expected stage index. Metrics are against time-normalized proxy progress, not environment true progress.",
        "",
        "Summary:",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"- {row['task_id']}:",
                f"  Spearman vs time = {row['gvl_spearman_vs_time']:.3f}, pairwise order acc = {row['gvl_pairwise_order_acc_vs_time']:.3f}, MAE vs time = {row['gvl_progress_mae_vs_time']:.3f}",
                f"  start = {row['gvl_progress_start']:.3f}, end = {row['gvl_progress_end']:.3f}, delta = {row['gvl_progress_delta']:.3f}",
                f"  final predicted stage = {row['predicted_final_stage_id']}: {row['predicted_final_stage_name']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="GVL-style local zero-shot stage/progress baseline.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("GVL-style VLM") / "outputs" / "three_task_gvl_local")
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
    heatmaps: dict[str, np.ndarray] = {}
    heatmap_stage_names: dict[str, list[str]] = {}
    config: dict[str, Any] = {
        "method": "GVL-style local weak reproduction with OpenCLIP stage prompts.",
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
        stage_prompts = build_stage_prompts(task.goal_text, task.stages)
        stage_features = encode_text(model, tokenizer, stage_prompts)
        stage_scores = (image_features @ stage_features.T).numpy()
        stage_probs = np.stack([softmax(row, args.temperature) for row in stage_scores], axis=0)
        stage_axis = np.arange(len(task.stages), dtype=np.float64) / max(len(task.stages) - 1, 1)
        stage_progress = stage_probs @ stage_axis
        stage_progress_norm = minmax(stage_progress)
        predicted_stage_ids = stage_scores.argmax(axis=1)

        heatmaps[task.task_id] = stage_scores
        heatmap_stage_names[task.task_id] = list(task.stages)
        spearman = spearman_vs_time(time_progress, stage_progress)
        order_acc = pairwise_order_accuracy(stage_progress)
        mae = progress_mae(time_progress, stage_progress)

        for sample_i, frame_idx in enumerate(indices):
            row: dict[str, Any] = {
                "task_id": task.task_id,
                "env_id": task.env_id,
                "video_path": str(task.video_path),
                "sample_index": sample_i,
                "frame_index": frame_idx,
                "time_sec": frame_idx / fps,
                "time_progress": float(time_progress[sample_i]),
                "gvl_stage_progress": float(stage_progress[sample_i]),
                "gvl_stage_progress_norm": float(stage_progress_norm[sample_i]),
                "predicted_stage_id": int(predicted_stage_ids[sample_i]),
                "predicted_stage_name": task.stages[int(predicted_stage_ids[sample_i])],
                "max_stage_similarity": float(stage_scores[sample_i, predicted_stage_ids[sample_i]]),
                "max_stage_probability": float(stage_probs[sample_i, predicted_stage_ids[sample_i]]),
            }
            for stage_i, stage_name in enumerate(task.stages):
                row[f"stage_{stage_i}_similarity"] = float(stage_scores[sample_i, stage_i])
                row[f"stage_{stage_i}_probability"] = float(stage_probs[sample_i, stage_i])
                if sample_i == 0:
                    config["tasks"].setdefault(task.task_id, {"env_id": task.env_id, "goal_text": task.goal_text, "stages": []})
                    config["tasks"][task.task_id]["stages"].append({"stage_id": stage_i, "stage_name": stage_name})
            sample_rows.append(row)

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
                "gvl_spearman_vs_time": float(spearman),
                "gvl_pairwise_order_acc_vs_time": float(order_acc),
                "gvl_progress_mae_vs_time": float(mae),
                "gvl_progress_start": float(stage_progress[0]),
                "gvl_progress_end": float(stage_progress[-1]),
                "gvl_progress_delta": float(stage_progress[-1] - stage_progress[0]),
                "predicted_final_stage_id": int(predicted_stage_ids[-1]),
                "predicted_final_stage_name": task.stages[int(predicted_stage_ids[-1])],
                "note": "Metrics are against time-normalized proxy progress, not environment-truth stage labels.",
            }
        )

    write_csv(out_dir / "gvl_local_samples.csv", sample_rows)
    write_csv(out_dir / "gvl_local_summary.csv", summary_rows)
    (out_dir / "gvl_local_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "gvl_local_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_curves(sample_rows, out_dir / "gvl_local_progress_curves.png")
    plot_stage_heatmaps(heatmaps, heatmap_stage_names, out_dir / "gvl_local_stage_heatmaps.png")
    write_results(out_dir / "RESULTS.txt", summary_rows)

    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
