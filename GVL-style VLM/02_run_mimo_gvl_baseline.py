from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


def read_api_key(root: Path, explicit_path: Path | None) -> str:
    env_key = os.environ.get("MIMO_API_KEY") or os.environ.get("XIAOMI_MIMO_API_KEY")
    if env_key:
        return env_key.strip()
    key_path = explicit_path or (root / "mimo apikey.txt")
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    raise RuntimeError("No MiMo API key found. Set MIMO_API_KEY or provide --api-key-file.")


def default_base_url(api_key: str) -> str:
    if api_key.startswith("tp-"):
        return "https://token-plan-cn.xiaomimimo.com/v1"
    return "https://api.xiaomimimo.com/v1"


def frame_to_data_url(frame: np.ndarray, image_size: int, jpeg_quality: int) -> str:
    image = Image.fromarray(frame).convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def request_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "top_p": 0.95,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MiMo HTTP {exc.code}: {body}") from exc


def extract_message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"MiMo response has no choices: {json.dumps(response, ensure_ascii=False)[:500]}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.append(str(item.get("text", "")))
        return "\n".join(texts)
    return str(content)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def make_prompt(task: Any, indices: list[int]) -> str:
    stages = "\n".join(f"{i}: {name}" for i, name in enumerate(task.stages))
    frame_lines = "\n".join(f"image_index={i}, frame_index={frame_idx}" for i, frame_idx in enumerate(indices))
    return f"""
You are evaluating a robot manipulation trajectory from sampled RGB frames.

Task id: {task.task_id}
Environment: {task.env_id}
Goal: {task.goal_text}

Ordered candidate stages:
{stages}

Images are sampled from the same trajectory:
{frame_lines}

For every image, infer the current task stage and a progress score in [0, 1].
Use visual evidence from the frame. Do not rely only on the image index.
Return JSON only, with this exact schema:
{{
  "task_id": "{task.task_id}",
  "samples": [
    {{
      "image_index": 0,
      "frame_index": {indices[0]},
      "stage_id": 0,
      "stage_name": "...",
      "progress": 0.0,
      "done_probability": 0.0,
      "confidence": 0.0,
      "brief_reason": "..."
    }}
  ],
  "notes": "..."
}}
""".strip()


def build_messages(task: Any, frames: list[np.ndarray], indices: list[int], image_size: int, jpeg_quality: int) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": frame_to_data_url(frame, image_size, jpeg_quality)}})
    content.append({"type": "text", "text": make_prompt(task, indices)})
    return [
        {
            "role": "system",
            "content": "You are MiMo, an AI assistant developed by Xiaomi. Return concise, valid JSON only.",
        },
        {"role": "user", "content": content},
    ]


def clamp01(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return float("nan")


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_samples(parsed: dict[str, Any], task: Any, indices: list[int]) -> list[dict[str, Any]]:
    raw_samples = parsed.get("samples")
    if not isinstance(raw_samples, list):
        raise RuntimeError(f"MiMo JSON missing samples list: {json.dumps(parsed, ensure_ascii=False)[:500]}")
    by_index: dict[int, dict[str, Any]] = {}
    for item in raw_samples:
        if not isinstance(item, dict):
            continue
        image_index = int_or_none(item.get("image_index"))
        if image_index is not None:
            by_index[image_index] = item
    normalized: list[dict[str, Any]] = []
    for image_index, frame_index in enumerate(indices):
        item = by_index.get(image_index, {})
        stage_id = int_or_none(item.get("stage_id"))
        if stage_id is None or stage_id < 0 or stage_id >= len(task.stages):
            stage_id = -1
            stage_name = str(item.get("stage_name", "unknown"))
        else:
            stage_name = task.stages[stage_id]
        normalized.append(
            {
                "image_index": image_index,
                "frame_index": frame_index,
                "stage_id": stage_id,
                "stage_name": stage_name,
                "progress": clamp01(item.get("progress")),
                "done_probability": clamp01(item.get("done_probability")),
                "confidence": clamp01(item.get("confidence")),
                "brief_reason": str(item.get("brief_reason", "")),
            }
        )
    return normalized


def finite_or_zero(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)


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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: list[dict[str, Any]], out_path: Path) -> None:
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(9, 3.1 * len(task_ids)), sharex=False)
    if len(task_ids) == 1:
        axes = [axes]
    for ax, task_id in zip(axes, task_ids):
        data = [row for row in rows if row["task_id"] == task_id]
        x = [float(row["time_progress"]) for row in data]
        ax.plot(x, [float(row["mimo_progress"]) for row in data], marker="o", label="MiMo progress")
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


def write_results(path: Path, summary_rows: list[dict[str, Any]], model: str, base_url_label: str) -> None:
    lines = [
        "MiMo GVL-style VLM baseline",
        "",
        "Method:",
        "Each task sends sampled RGB frames to MiMo as base64 image inputs and asks for JSON stage/progress predictions.",
        "Metrics are against time-normalized proxy progress, not environment-truth progress or stage labels.",
        f"Model: {model}",
        f"Base URL class: {base_url_label}",
        "",
        "Summary:",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"- {row['task_id']}:",
                f"  Spearman vs time = {row['mimo_spearman_vs_time']:.3f}, pairwise order acc = {row['mimo_pairwise_order_acc_vs_time']:.3f}, MAE vs time = {row['mimo_progress_mae_vs_time']:.3f}",
                f"  start = {row['mimo_progress_start']:.3f}, end = {row['mimo_progress_end']:.3f}, delta = {row['mimo_progress_delta']:.3f}",
                f"  final predicted stage = {row['predicted_final_stage_id']}: {row['predicted_final_stage_name']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="MiMo VLM stage/progress baseline on sampled ManiSkill frames.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("GVL-style VLM") / "outputs" / "three_task_mimo_gvl")
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--base-url", default=os.environ.get("MIMO_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("MIMO_MODEL", "mimo-v2.5"))
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task ids: stackcube stackpyramid peginsertion")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    api_key = read_api_key(root, args.api_key_file)
    base_url = args.base_url or default_base_url(api_key)
    base_url_label = "token-plan" if "token-plan" in base_url else "pay-as-you-go/custom"

    all_sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.tasks) if args.tasks else None
    config = {
        "model": args.model,
        "base_url_class": base_url_label,
        "num_samples": args.num_samples,
        "image_size": args.image_size,
        "jpeg_quality": args.jpeg_quality,
        "temperature": args.temperature,
        "note": "API key is read from environment or local key file and is not stored in outputs.",
        "tasks": {},
    }

    for task in default_tasks(root):
        if selected and task.task_id not in selected:
            continue
        print(f"Calling MiMo for {task.task_id} with model={args.model}, samples={args.num_samples}")
        frames, fps = load_video_frames(task.video_path)
        indices = sample_indices(len(frames), args.num_samples)
        sampled_frames = [frames[idx] for idx in indices]
        messages = build_messages(task, sampled_frames, indices, args.image_size, args.jpeg_quality)
        started = time.time()
        response = request_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
        )
        elapsed_api_sec = time.time() - started
        raw_text = extract_message_text(response)
        raw_path = raw_dir / f"{task.task_id}_response.json"
        raw_payload: dict[str, Any] = {
            "task_id": task.task_id,
            "model": args.model,
            "response": response,
            "message_text": raw_text,
        }
        raw_path.write_text(
            json.dumps(raw_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        parsed = parse_json_object(raw_text)
        normalized = normalize_samples(parsed, task, indices)
        raw_payload["parsed"] = parsed
        raw_path.write_text(
            json.dumps(
                raw_payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        summary = read_summary(task.summary_path)
        record = saved_record(summary)
        time_progress = np.array(indices, dtype=np.float64) / max(len(frames) - 1, 1)
        pred_progress = finite_or_zero(np.array([row["progress"] for row in normalized], dtype=np.float64))
        pred_done = finite_or_zero(np.array([row["done_probability"] for row in normalized], dtype=np.float64))
        spearman = spearman_vs_time(time_progress, pred_progress)
        order_acc = pairwise_order_accuracy(pred_progress)
        mae = progress_mae(time_progress, pred_progress)

        for sample_i, item in enumerate(normalized):
            all_sample_rows.append(
                {
                    "task_id": task.task_id,
                    "env_id": task.env_id,
                    "video_path": str(task.video_path),
                    "sample_index": sample_i,
                    "frame_index": item["frame_index"],
                    "time_sec": item["frame_index"] / fps,
                    "time_progress": float(time_progress[sample_i]),
                    "mimo_progress": float(pred_progress[sample_i]),
                    "mimo_done_probability": float(pred_done[sample_i]),
                    "mimo_confidence": item["confidence"],
                    "predicted_stage_id": item["stage_id"],
                    "predicted_stage_name": item["stage_name"],
                    "brief_reason": item["brief_reason"],
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
                "episode_success": bool(record.get("success", False)),
                "first_success_seed": record.get("seed"),
                "attempts": summary.get("attempts"),
                "elapsed_steps": record.get("elapsed_steps"),
                "mimo_model": args.model,
                "mimo_api_elapsed_sec": elapsed_api_sec,
                "mimo_spearman_vs_time": float(spearman),
                "mimo_pairwise_order_acc_vs_time": float(order_acc),
                "mimo_progress_mae_vs_time": float(mae),
                "mimo_progress_start": float(pred_progress[0]),
                "mimo_progress_end": float(pred_progress[-1]),
                "mimo_progress_delta": float(pred_progress[-1] - pred_progress[0]),
                "mimo_done_probability_start": float(pred_done[0]),
                "mimo_done_probability_end": float(pred_done[-1]),
                "predicted_final_stage_id": normalized[-1]["stage_id"],
                "predicted_final_stage_name": normalized[-1]["stage_name"],
                "note": "Metrics are against time-normalized proxy progress, not environment-truth stage labels.",
            }
        )
        config["tasks"][task.task_id] = {
            "env_id": task.env_id,
            "goal_text": task.goal_text,
            "stages": list(task.stages),
        }
        time.sleep(args.sleep_sec)

    write_csv(out_dir / "mimo_gvl_samples.csv", all_sample_rows)
    write_csv(out_dir / "mimo_gvl_summary.csv", summary_rows)
    (out_dir / "mimo_gvl_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "mimo_gvl_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_curves(all_sample_rows, out_dir / "mimo_gvl_progress_curves.png")
    write_results(out_dir / "RESULTS.txt", summary_rows, args.model, base_url_label)
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    print(f"Saved MiMo results to {out_dir}")


if __name__ == "__main__":
    main()
