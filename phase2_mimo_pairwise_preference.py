from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimo_vlm_common import (  # noqa: E402
    default_base_url,
    extract_message_text,
    frame_to_data_url,
    parse_json_object,
    read_api_key,
    request_chat_completion,
    write_csv,
)


DEFAULT_QUEUE = Path(r"D:\Users\User\Desktop\reward_phase0_inventory\phase1_clip_dataset\pair_annotation_queue.csv")
DEFAULT_OUT = Path(r"D:\Users\User\Desktop\reward_phase0_inventory\phase2_mimo_pairwise_labels")

VALID_PREFS = {"A>B", "B>A", "unsure"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_frame_indices(text: str, fallback_start: int, fallback_end: int, max_frames: int = 6) -> list[int]:
    values = [int(x) for x in re.findall(r"\d+", text or "")]
    if values:
        return values[:max_frames]
    start = int(fallback_start)
    end = int(fallback_end)
    length = max(1, end - start)
    count = min(max_frames, length)
    if count == 1:
        return [start]
    return [start + round(i * (length - 1) / (count - 1)) for i in range(count)]


def read_video_frames(video_path: Path, indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not decode frame {idx} from {video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb.astype(np.uint8))
    cap.release()
    return frames


def clamp_confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def normalize_preference(value: Any) -> str:
    text = str(value or "").strip()
    compact = text.replace(" ", "").upper()
    if compact in {"A>B", "A"}:
        return "A>B"
    if compact in {"B>A", "B"}:
        return "B>A"
    if "A" in compact and "B" in compact and ">" in compact:
        return "A>B" if compact.index("A") < compact.index(">") < compact.index("B") else "B>A"
    if "UNSURE" in compact or "TIE" in compact or "UNKNOWN" in compact or "不确定" in text:
        return "unsure"
    return "unsure"


def safe_parse_pair_json(text: str) -> tuple[dict[str, Any], str, str]:
    try:
        return parse_json_object(text), "ok", ""
    except Exception as exc:  # MiMo occasionally returns almost-JSON with a small syntax issue.
        parsed: dict[str, Any] = {}
        pref_match = re.search(r'"?preference"?\s*:\s*"?([^",\n}]+)', text, flags=re.IGNORECASE)
        conf_match = re.search(r'"?confidence"?\s*:\s*([0-9.]+)', text, flags=re.IGNORECASE)
        for key in ["clip_a_stage", "clip_b_stage", "reason", "visible_failure_or_uncertainty"]:
            match = re.search(rf'"?{key}"?\s*:\s*"([^"]*)"', text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                parsed[key] = match.group(1).strip()
        if pref_match:
            parsed["preference"] = pref_match.group(1).strip()
        else:
            parsed["preference"] = "unsure"
        if conf_match:
            parsed["confidence"] = conf_match.group(1).strip()
        else:
            parsed["confidence"] = 0.0
        if "reason" not in parsed:
            parsed["reason"] = text.strip()[:500]
        parsed["visible_failure_or_uncertainty"] = parsed.get("visible_failure_or_uncertainty", "")
        return parsed, "fallback_regex", f"{type(exc).__name__}: {exc}"


def make_prompt(row: dict[str, str], a_indices: list[int], b_indices: list[int]) -> str:
    a_lines = "\n".join(f"A{i}: frame_index={idx}" for i, idx in enumerate(a_indices))
    b_lines = "\n".join(f"B{i}: frame_index={idx}" for i, idx in enumerate(b_indices))
    return f"""
You are an offline visual-semantic preference annotator for robot manipulation progress.

Task id: {row['task_id']}
Task goal: {row['task_goal_text']}

You will see two short clips from the same task.
- Images A0..A{len(a_indices) - 1} are clip A, in chronological order.
- Images B0..B{len(b_indices) - 1} are clip B, in chronological order.

Clip A frames:
{a_lines}

Clip B frames:
{b_lines}

Judge which clip shows more progress toward completing the task goal.
Use visual evidence only. Prefer the clip whose final state is closer to the full task goal.
If the clips show similar progress or the evidence is ambiguous, return "unsure".
Do not assume a clip is successful just because it looks close; release, stability, support, or insertion depth may matter when visible.

Return concise valid JSON only, with this exact schema:
{{
  "pair_id": "{row['pair_id']}",
  "preference": "A>B | B>A | unsure",
  "confidence": 0.0,
  "clip_a_stage": "short description of A's visible progress",
  "clip_b_stage": "short description of B's visible progress",
  "reason": "one or two sentences explaining the preference",
  "visible_failure_or_uncertainty": "brief note, or empty string"
}}
""".strip()


def build_messages(
    row: dict[str, str],
    a_frames: list[np.ndarray],
    b_frames: list[np.ndarray],
    a_indices: list[int],
    b_indices: list[int],
    image_size: int,
    jpeg_quality: int,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for frame in a_frames:
        content.append({"type": "image_url", "image_url": {"url": frame_to_data_url(frame, image_size, jpeg_quality)}})
    for frame in b_frames:
        content.append({"type": "image_url", "image_url": {"url": frame_to_data_url(frame, image_size, jpeg_quality)}})
    content.append({"type": "text", "text": make_prompt(row, a_indices, b_indices)})
    return [
        {"role": "system", "content": "Return concise valid JSON only. Do not include markdown fences."},
        {"role": "user", "content": content},
    ]


def raw_path_for(raw_dir: Path, pair_id: str) -> Path:
    return raw_dir / f"{pair_id}.json"


def existing_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = load_csv(path)
    return {row["pair_id"]: row for row in rows}


def call_one_pair(
    row: dict[str, str],
    root: Path,
    base_url: str,
    api_key: str,
    model: str,
    image_size: int,
    jpeg_quality: int,
    max_tokens: int,
    timeout_sec: int,
    raw_dir: Path,
) -> dict[str, Any]:
    a_indices = parse_frame_indices(row["clip_a_sample_frame_indices"], int(row["clip_a_start_frame"]), int(row["clip_a_end_frame_exclusive"]))
    b_indices = parse_frame_indices(row["clip_b_sample_frame_indices"], int(row["clip_b_start_frame"]), int(row["clip_b_end_frame_exclusive"]))
    a_video = root / row["clip_a_video_path"]
    b_video = root / row["clip_b_video_path"]
    a_frames = read_video_frames(a_video, a_indices)
    b_frames = read_video_frames(b_video, b_indices)

    messages = build_messages(row, a_frames, b_frames, a_indices, b_indices, image_size, jpeg_quality)
    started = time.time()
    response = request_chat_completion(base_url, api_key, model, messages, max_tokens, 0.0, timeout_sec)
    elapsed = time.time() - started
    text = extract_message_text(response)
    parsed, parse_status, parse_error = safe_parse_pair_json(text)
    preference = normalize_preference(parsed.get("preference"))
    confidence = clamp_confidence(parsed.get("confidence"))

    raw = {
        "pair_id": row["pair_id"],
        "model": model,
        "api_elapsed_sec": elapsed,
        "queue_row": row,
        "a_indices": a_indices,
        "b_indices": b_indices,
        "response": response,
        "message_text": text,
        "parsed": parsed,
        "parse_status": parse_status,
        "parse_error": parse_error,
    }
    write_json(raw_path_for(raw_dir, row["pair_id"]), raw)

    candidate_label = row.get("candidate_label", "")
    return {
        "pair_id": row["pair_id"],
        "task_id": row["task_id"],
        "pair_type": row["pair_type"],
        "candidate_label": candidate_label,
        "candidate_confidence": row.get("candidate_confidence", ""),
        "mimo_preference": preference,
        "mimo_confidence": confidence,
        "agrees_with_candidate": "true" if preference == candidate_label else ("unsure" if preference == "unsure" else "false"),
        "clip_a_id": row["clip_a_id"],
        "clip_b_id": row["clip_b_id"],
        "clip_a_stage": str(parsed.get("clip_a_stage", "")),
        "clip_b_stage": str(parsed.get("clip_b_stage", "")),
        "reason": str(parsed.get("reason", "")),
        "visible_failure_or_uncertainty": str(parsed.get("visible_failure_or_uncertainty", "")),
        "a_sample_frame_indices": ";".join(str(x) for x in a_indices),
        "b_sample_frame_indices": ";".join(str(x) for x in b_indices),
        "mimo_model": model,
        "api_elapsed_sec": round(elapsed, 3),
        "parse_status": parse_status,
        "parse_error": parse_error,
        "raw_response_path": str(raw_path_for(raw_dir, row["pair_id"])),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pref = Counter(row["mimo_preference"] for row in rows)
    by_task = defaultdict(lambda: Counter())
    by_pair_type = defaultdict(lambda: Counter())
    agreement = Counter(row["agrees_with_candidate"] for row in rows)
    for row in rows:
        by_task[row["task_id"]][row["mimo_preference"]] += 1
        by_pair_type[row["pair_type"]][row["mimo_preference"]] += 1
    return {
        "num_labels": len(rows),
        "preference_distribution": dict(by_pref),
        "candidate_agreement": dict(agreement),
        "by_task": {task: dict(counter) for task, counter in by_task.items()},
        "by_pair_type": {pair_type: dict(counter) for pair_type, counter in by_pair_type.items()},
    }


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "阶段 2：MiMo pairwise preference 离线标注",
        "生成时间：2026-06-11",
        "",
        "一、这一步做了什么",
        "读取阶段一的 pair_annotation_queue.csv，对每个 A/B clip pair 调用 MiMo，输出 A>B / B>A / unsure。",
        "本阶段仍然只是视觉语义偏好分支，不是最终训练标签；后续还需要物理分支和 Stage-Aware Fusion。",
        "",
        "二、标注规模",
        f"- 已标注 pair 数：{summary['num_labels']}",
        "",
        "三、MiMo preference 分布",
    ]
    for key, value in summary["preference_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "四、与阶段一候选弱标签的一致性",
    ]
    for key, value in summary["candidate_agreement"].items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "五、按任务统计",
    ]
    for task_id, counter in summary["by_task"].items():
        lines.append(f"- {task_id}: {counter}")
    lines += [
        "",
        "六、输出文件",
        "- mimo_pairwise_labels.csv/json：结构化 MiMo preference 标签。",
        "- raw_responses/*.json：每个 pair 的原始 API response 与解析结果。",
        "- phase2_summary.json：统计摘要。",
        "",
        "七、注意",
        "MiMo 的判断可能仍会偏向视觉接近目标的片段。不要直接把本文件作为最终 reward 标签，后续需要物理分支做约束。",
    ]
    (out_dir / "阶段2_MiMo_pairwise标注说明.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: call MiMo for pairwise clip preference labels.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--base-url", default=os.environ.get("MIMO_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("MIMO_MODEL", "mimo-v2.5"))
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Maximum new API calls; 0 means no limit.")
    parser.add_argument("--pair-type", default="", help="Optional pair_type filter.")
    parser.add_argument("--task-id", default="", help="Optional task_id filter.")
    parser.add_argument("--force", action="store_true", help="Re-call API even if pair_id already exists in output CSV.")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "mimo_pairwise_labels.csv"

    queue = load_csv(args.queue)
    if args.pair_type:
        queue = [row for row in queue if row["pair_type"] == args.pair_type]
    if args.task_id:
        queue = [row for row in queue if row["task_id"] == args.task_id]

    existing = {} if args.force else existing_labels(labels_path)
    labels: list[dict[str, Any]] = list(existing.values())

    api_key = read_api_key(root, args.api_key_file)
    base_url = args.base_url or default_base_url(api_key)

    calls = 0
    for row in queue:
        pair_id = row["pair_id"]
        if pair_id in existing and not args.force:
            continue
        if args.limit and calls >= args.limit:
            break
        print(f"Calling MiMo for {pair_id} {row['task_id']} {row['pair_type']}")
        try:
            label = call_one_pair(
                row=row,
                root=root,
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                image_size=args.image_size,
                jpeg_quality=args.jpeg_quality,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
                raw_dir=raw_dir,
            )
        except Exception as exc:
            label = {
                "pair_id": pair_id,
                "task_id": row["task_id"],
                "pair_type": row["pair_type"],
                "candidate_label": row.get("candidate_label", ""),
                "candidate_confidence": row.get("candidate_confidence", ""),
                "mimo_preference": "unsure",
                "mimo_confidence": 0.0,
                "agrees_with_candidate": "unsure",
                "clip_a_id": row["clip_a_id"],
                "clip_b_id": row["clip_b_id"],
                "clip_a_stage": "",
                "clip_b_stage": "",
                "reason": f"ERROR: {type(exc).__name__}: {exc}",
                "visible_failure_or_uncertainty": "",
                "a_sample_frame_indices": row.get("clip_a_sample_frame_indices", ""),
                "b_sample_frame_indices": row.get("clip_b_sample_frame_indices", ""),
                "mimo_model": args.model,
                "api_elapsed_sec": 0.0,
                "parse_status": "error",
                "parse_error": f"{type(exc).__name__}: {exc}",
                "raw_response_path": "",
            }
            write_json(raw_dir / f"{pair_id}_error.json", {"pair_id": pair_id, "queue_row": row, "error": label["parse_error"]})
        labels.append(label)
        labels.sort(key=lambda item: item["pair_id"])
        write_csv(labels_path, labels)
        write_json(out_dir / "mimo_pairwise_labels.json", labels)
        calls += 1
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    summary = summarize(labels)
    write_json(out_dir / "phase2_summary.json", summary)
    write_report(out_dir, summary)
    print(json.dumps({"new_calls": calls, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
