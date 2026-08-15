from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v3 as iio
import numpy as np

from mimo_vlm_common import (
    clamp01,
    default_base_url,
    extract_message_text,
    frame_to_data_url,
    int_or_none,
    parse_json_object,
    read_api_key,
    request_chat_completion,
    write_csv,
)
from zero_shot_baseline_common import TaskSpec, default_tasks, sample_indices


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_video_frames(path: Path) -> list[np.ndarray]:
    frames = [frame[..., :3].astype(np.uint8) for frame in iio.imiter(path)]
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def task_by_id(root: Path) -> dict[str, TaskSpec]:
    return {task.task_id: task for task in default_tasks(root)}


def compact_stage_list(task: TaskSpec) -> str:
    return "\n".join(f"{i}: {stage}" for i, stage in enumerate(task.stages))


def make_common_image_content(frames: list[np.ndarray], image_size: int, jpeg_quality: int) -> list[dict[str, Any]]:
    return [
        {"type": "image_url", "image_url": {"url": frame_to_data_url(frame, image_size, jpeg_quality)}}
        for frame in frames
    ]


def make_prompt(method: str, task: TaskSpec, sample: dict[str, Any], indices: list[int], has_success_ref: bool) -> str:
    frame_lines = "\n".join(f"image_index={i}, frame_index={frame_idx}" for i, frame_idx in enumerate(indices))
    stages = compact_stage_list(task)
    base = f"""
Task id: {task.task_id}
Environment: {task.env_id}
Goal: {task.goal_text}
Candidate stages:
{stages}

This video is a failure or near-miss sample:
sample_id: {sample['sample_id']}
near_miss_type: {sample['near_miss_type']}
expected_success: false
target_failure_mode: {sample['target_failure_mode']}

Candidate video frames:
{frame_lines}
""".strip()

    if method == "roboclip":
        ref_text = "The first image is a successful reference final state. Candidate images start after that reference." if has_success_ref else ""
        return f"""
You are reproducing a RoboCLIP-style zero-shot reward baseline.
{ref_text}

{base}

For each candidate image, estimate how close it is to the task goal and to the successful visual state.
Return JSON only:
{{
  "samples": [
    {{
      "image_index": 0,
      "frame_index": {indices[0]},
      "score": 0.0,
      "done_probability": 0.0,
      "stage_id": 0,
      "stage_name": "...",
      "confidence": 0.0,
      "brief_reason": "..."
    }}
  ],
  "notes": "..."
}}
""".strip()

    if method == "gvl":
        return f"""
You are reproducing a GVL-style stage/progress VLM baseline.

{base}

For each image, infer the current stage and progress in [0, 1].
Do not mark the sample complete unless the visible state satisfies the full physical goal.
Return JSON only:
{{
  "samples": [
    {{
      "image_index": 0,
      "frame_index": {indices[0]},
      "score": 0.0,
      "done_probability": 0.0,
      "stage_id": 0,
      "stage_name": "...",
      "confidence": 0.0,
      "brief_reason": "..."
    }}
  ],
  "notes": "..."
}}
""".strip()

    if method == "topreward":
        return f"""
You are reproducing a TOPReward-style trajectory-prefix reward baseline.

{base}

For each prefix endpoint image, judge whether the trajectory prefix has achieved the goal.
Output score in [0, 1] and done_probability in [0, 1].
Return JSON only:
{{
  "samples": [
    {{
      "image_index": 0,
      "frame_index": {indices[0]},
      "score": 0.0,
      "done_probability": 0.0,
      "confidence": 0.0,
      "brief_reason": "..."
    }}
  ],
  "notes": "..."
}}
""".strip()

    raise ValueError(method)


def build_messages(
    method: str,
    task: TaskSpec,
    sample: dict[str, Any],
    frames: list[np.ndarray],
    indices: list[int],
    image_size: int,
    jpeg_quality: int,
    success_ref_frame: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if success_ref_frame is not None:
        content.append({"type": "image_url", "image_url": {"url": frame_to_data_url(success_ref_frame, image_size, jpeg_quality)}})
    content.extend(make_common_image_content(frames, image_size, jpeg_quality))
    content.append({"type": "text", "text": make_prompt(method, task, sample, indices, success_ref_frame is not None)})
    return [
        {"role": "system", "content": "Return concise valid JSON only. Do not include markdown fences."},
        {"role": "user", "content": content},
    ]


def normalize_samples(parsed: dict[str, Any], task: TaskSpec, indices: list[int]) -> tuple[list[dict[str, Any]], str]:
    raw_samples = parsed.get("samples")
    if not isinstance(raw_samples, list):
        raise RuntimeError(f"MiMo JSON missing samples list: {json.dumps(parsed, ensure_ascii=False)[:500]}")
    by_index: dict[int, dict[str, Any]] = {}
    for item in raw_samples:
        if isinstance(item, dict):
            image_index = int_or_none(item.get("image_index"))
            if image_index is not None:
                by_index[image_index] = item
    rows: list[dict[str, Any]] = []
    for image_index, frame_index in enumerate(indices):
        item = by_index.get(image_index, {})
        stage_id = int_or_none(item.get("stage_id"))
        stage_name = str(item.get("stage_name", ""))
        if stage_id is not None and 0 <= stage_id < len(task.stages):
            stage_name = task.stages[stage_id]
        rows.append(
            {
                "image_index": image_index,
                "frame_index": frame_index,
                "score": clamp01(item.get("score")),
                "done_probability": clamp01(item.get("done_probability")),
                "stage_id": "" if stage_id is None else stage_id,
                "stage_name": stage_name,
                "confidence": clamp01(item.get("confidence")),
                "brief_reason": str(item.get("brief_reason", "")),
            }
        )
    return rows, str(parsed.get("notes", ""))


def summarize_scores(scores: np.ndarray, dones: np.ndarray, threshold: float) -> dict[str, Any]:
    scores = np.nan_to_num(scores.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    dones = np.nan_to_num(dones.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    return {
        "score_start": float(scores[0]),
        "score_end": float(scores[-1]),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "done_start": float(dones[0]),
        "done_end": float(dones[-1]),
        "done_max": float(np.max(dones)),
        "false_positive_final": bool(scores[-1] >= threshold or dones[-1] >= 0.5),
        "false_positive_any": bool(np.max(scores) >= threshold or np.max(dones) >= 0.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MiMo-reproduced baselines on failure/near-miss videos.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("near_miss_samples") / "outputs" / "failure_manifest.csv")
    parser.add_argument("--out", type=Path, default=Path("near_miss_samples") / "outputs" / "mimo_failure_eval")
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--base-url", default=os.environ.get("MIMO_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("MIMO_MODEL", "mimo-v2.5"))
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tasks = task_by_id(root)
    manifest = load_manifest(args.manifest)
    api_key = read_api_key(root, args.api_key_file)
    base_url = args.base_url or default_base_url(api_key)
    methods = ["roboclip", "gvl", "topreward"]

    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    success_ref_cache: dict[str, np.ndarray] = {}
    for task_id, task in tasks.items():
        frames = load_video_frames(task.video_path)
        success_ref_cache[task_id] = frames[-1]

    for sample in manifest:
        task = tasks[sample["task_id"]]
        video_path = root / sample["video_path"]
        all_frames = load_video_frames(video_path)
        indices = sample_indices(len(all_frames), args.num_samples)
        frames = [all_frames[idx] for idx in indices]
        for method in methods:
            print(f"Calling MiMo {method} on {sample['sample_id']} ({len(indices)} frames)")
            ref = success_ref_cache[task.task_id] if method == "roboclip" else None
            messages = build_messages(method, task, sample, frames, indices, args.image_size, args.jpeg_quality, ref)
            started = time.time()
            response = request_chat_completion(base_url, api_key, args.model, messages, args.max_tokens, 0.0, args.timeout_sec)
            elapsed = time.time() - started
            text = extract_message_text(response)
            parsed = parse_json_object(text)
            normalized, notes = normalize_samples(parsed, task, indices)
            raw_path = raw_dir / f"{method}_{sample['sample_id']}.json"
            raw_path.write_text(
                json.dumps(
                    {"method": method, "sample_id": sample["sample_id"], "response": response, "message_text": text, "parsed": parsed},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            scores = np.array([row["score"] for row in normalized], dtype=np.float64)
            dones = np.array([row["done_probability"] for row in normalized], dtype=np.float64)
            stats = summarize_scores(scores, dones, args.threshold)
            final_row = normalized[-1]
            summary_rows.append(
                {
                    "method": f"MiMo-{method}",
                    "sample_id": sample["sample_id"],
                    "task_id": sample["task_id"],
                    "near_miss_type": sample["near_miss_type"],
                    "target_failure_mode": sample["target_failure_mode"],
                    "expected_success": sample["expected_success"],
                    "observed_success": sample["observed_success"],
                    "num_samples": len(indices),
                    "mimo_model": args.model,
                    "api_elapsed_sec": elapsed,
                    **stats,
                    "final_stage_id": final_row["stage_id"],
                    "final_stage_name": final_row["stage_name"],
                    "final_reason": final_row["brief_reason"],
                    "notes": notes,
                }
            )
            for idx, row in enumerate(normalized):
                sample_rows.append(
                    {
                        "method": f"MiMo-{method}",
                        "sample_id": sample["sample_id"],
                        "task_id": sample["task_id"],
                        "sample_index": idx,
                        "frame_index": row["frame_index"],
                        "time_progress": idx / max(len(normalized) - 1, 1),
                        "score": row["score"],
                        "done_probability": row["done_probability"],
                        "stage_id": row["stage_id"],
                        "stage_name": row["stage_name"],
                        "confidence": row["confidence"],
                        "brief_reason": row["brief_reason"],
                    }
                )
            time.sleep(args.sleep_sec)

    write_csv(out_dir / "mimo_failure_eval_summary.csv", summary_rows)
    write_csv(out_dir / "mimo_failure_eval_samples.csv", sample_rows)
    (out_dir / "mimo_failure_eval_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
