from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mimo_vlm_common import (
    clamp01,
    default_base_url,
    extract_message_text,
    finite_or_zero,
    frame_to_data_url,
    parse_json_object,
    read_api_key,
    request_chat_completion,
    write_csv,
)
from zero_shot_baseline_common import (
    default_tasks,
    load_video_frames,
    pairwise_order_accuracy,
    progress_mae,
    read_summary,
    sample_indices,
    saved_record,
    spearman_vs_time,
)


def make_prompt(task: Any, indices: list[int]) -> str:
    frame_lines = "\n".join(f"prefix_index={i}, endpoint_frame={frame_idx}" for i, frame_idx in enumerate(indices))
    return f"""
You are reproducing a TOPReward-style zero-shot trajectory-prefix reward baseline.

Task id: {task.task_id}
Environment: {task.env_id}
Goal: {task.goal_text}
Negative/not-done description: {task.negative_text}

Images are sampled from one trajectory in chronological order:
{frame_lines}

TOPReward-style scoring rule for this evaluation:
- For each prefix endpoint image, judge whether the trajectory prefix up to that image has achieved the goal.
- Output a done_probability in [0, 1].
- Output a progress/reward-like score in [0, 1].
- Do not mark done just because the frame is late; use visual evidence.

Return JSON only, with this exact schema:
{{
  "task_id": "{task.task_id}",
  "samples": [
    {{
      "prefix_index": 0,
      "frame_index": {indices[0]},
      "topreward_score": 0.0,
      "done_probability": 0.0,
      "confidence": 0.0,
      "brief_reason": "max 10 words"
    }}
  ],
  "notes": "..."
}}
""".strip()


def build_messages(task: Any, frames: list[np.ndarray], indices: list[int], image_size: int, jpeg_quality: int) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": frame_to_data_url(frame, image_size, jpeg_quality)}} for frame in frames
    ]
    content.append({"type": "text", "text": make_prompt(task, indices)})
    return [
        {"role": "system", "content": "Return concise, valid JSON only. Do not include markdown fences."},
        {"role": "user", "content": content},
    ]


def normalize_samples(parsed: dict[str, Any], indices: list[int]) -> list[dict[str, Any]]:
    raw_samples = parsed.get("samples")
    if not isinstance(raw_samples, list):
        raise RuntimeError(f"MiMo JSON missing samples list: {json.dumps(parsed, ensure_ascii=False)[:500]}")
    by_index: dict[int, dict[str, Any]] = {}
    for item in raw_samples:
        if not isinstance(item, dict):
            continue
        try:
            prefix_index = int(item.get("prefix_index"))
        except (TypeError, ValueError):
            continue
        by_index[prefix_index] = item
    normalized: list[dict[str, Any]] = []
    for prefix_index, frame_index in enumerate(indices):
        item = by_index.get(prefix_index, {})
        normalized.append(
            {
                "prefix_index": prefix_index,
                "frame_index": frame_index,
                "topreward_score": clamp01(item.get("topreward_score")),
                "done_probability": clamp01(item.get("done_probability")),
                "confidence": clamp01(item.get("confidence")),
                "brief_reason": str(item.get("brief_reason", "")),
            }
        )
    return normalized


def plot_curves(rows: list[dict[str, Any]], out_path: Path) -> None:
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(9, 3.1 * len(task_ids)), sharex=False)
    if len(task_ids) == 1:
        axes = [axes]
    for ax, task_id in zip(axes, task_ids):
        data = [row for row in rows if row["task_id"] == task_id]
        x = [float(row["time_progress"]) for row in data]
        ax.plot(x, [float(row["mimo_topreward_score"]) for row in data], marker="o", label="MiMo TOPReward-style score")
        ax.plot(x, [float(row["mimo_done_probability"]) for row in data], marker="s", label="done probability")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="MiMo TOPReward-style baseline on sampled ManiSkill trajectory prefixes.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("TOPReward-style") / "outputs" / "three_task_mimo_topreward")
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--base-url", default=os.environ.get("MIMO_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("MIMO_MODEL", "mimo-v2.5"))
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    api_key = read_api_key(root, args.api_key_file)
    base_url = args.base_url or default_base_url(api_key)

    all_sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for task in default_tasks(root):
        print(f"Calling MiMo TOPReward-style for {task.task_id} with {args.num_samples} samples")
        frames, fps = load_video_frames(task.video_path)
        indices = sample_indices(len(frames), args.num_samples)
        sampled_frames = [frames[idx] for idx in indices]
        messages = build_messages(task, sampled_frames, indices, args.image_size, args.jpeg_quality)
        started = time.time()
        response = request_chat_completion(base_url, api_key, args.model, messages, args.max_tokens, args.temperature, args.timeout_sec)
        elapsed = time.time() - started
        raw_text = extract_message_text(response)
        raw_path = raw_dir / f"{task.task_id}_response.json"
        raw_payload = {"task_id": task.task_id, "model": args.model, "response": response, "message_text": raw_text}
        raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        parsed = parse_json_object(raw_text)
        normalized = normalize_samples(parsed, indices)
        raw_payload["parsed"] = parsed
        raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        summary = read_summary(task.summary_path)
        record = saved_record(summary)
        time_progress = np.array(indices, dtype=np.float64) / max(len(frames) - 1, 1)
        scores = finite_or_zero(np.array([row["topreward_score"] for row in normalized], dtype=np.float64))
        done = finite_or_zero(np.array([row["done_probability"] for row in normalized], dtype=np.float64))

        for sample_i, item in enumerate(normalized):
            all_sample_rows.append(
                {
                    "task_id": task.task_id,
                    "env_id": task.env_id,
                    "sample_index": sample_i,
                    "frame_index": item["frame_index"],
                    "time_sec": item["frame_index"] / fps,
                    "time_progress": float(time_progress[sample_i]),
                    "mimo_topreward_score": float(scores[sample_i]),
                    "mimo_done_probability": float(done[sample_i]),
                    "mimo_confidence": item["confidence"],
                    "brief_reason": item["brief_reason"],
                }
            )

        summary_rows.append(
            {
                "task_id": task.task_id,
                "env_id": task.env_id,
                "sampled_frames": len(indices),
                "episode_success": bool(record.get("success", False)),
                "elapsed_steps": record.get("elapsed_steps"),
                "mimo_model": args.model,
                "mimo_api_elapsed_sec": elapsed,
                "mimo_topreward_spearman_vs_time": float(spearman_vs_time(time_progress, scores)),
                "mimo_topreward_pairwise_order_acc_vs_time": float(pairwise_order_accuracy(scores)),
                "mimo_topreward_progress_mae_vs_time": float(progress_mae(time_progress, scores)),
                "mimo_topreward_score_start": float(scores[0]),
                "mimo_topreward_score_end": float(scores[-1]),
                "mimo_topreward_score_delta": float(scores[-1] - scores[0]),
                "mimo_done_probability_start": float(done[0]),
                "mimo_done_probability_end": float(done[-1]),
                "note": "Metrics are against time-normalized proxy progress, not environment-truth stage labels.",
            }
        )
        time.sleep(args.sleep_sec)

    write_csv(out_dir / "mimo_topreward_samples.csv", all_sample_rows)
    write_csv(out_dir / "mimo_topreward_summary.csv", summary_rows)
    (out_dir / "mimo_topreward_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_curves(all_sample_rows, out_dir / "mimo_topreward_progress_curves.png")
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    print(f"Saved MiMo TOPReward-style results to {out_dir}")


if __name__ == "__main__":
    main()
