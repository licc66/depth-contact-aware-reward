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

ROOT = Path(__file__).resolve().parents[1]
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


VALID_PREFS = {"A>B", "B>A", "unsure"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_frame_indices(text: str, fallback_start: int, fallback_end: int, max_frames: int) -> list[int]:
    values = [int(x) for x in re.findall(r"\d+", text or "")]
    if values:
        if len(values) <= max_frames:
            return values
        idxs = np.linspace(0, len(values) - 1, max_frames)
        return [values[int(round(i))] for i in idxs]
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
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8))
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


def safe_parse_batch_json(text: str) -> tuple[list[dict[str, Any]], str, str]:
    try:
        obj = parse_json_object(text)
        if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
            return obj["labels"], "ok", ""
        if isinstance(obj, list):
            return obj, "ok_list", ""
        if isinstance(obj, dict) and "pair_id" in obj:
            return [obj], "ok_single", ""
        return [], "missing_labels", "JSON parsed but no labels list"
    except Exception as exc:
        labels: list[dict[str, Any]] = []
        for match in re.finditer(r"\{[^{}]*pair_id[^{}]*\}", text, flags=re.IGNORECASE | re.DOTALL):
            try:
                labels.append(json.loads(match.group(0)))
            except Exception:
                continue
        return labels, "fallback_regex", f"{type(exc).__name__}: {exc}"


def video_path(row: dict[str, str], side: str) -> Path:
    value = (
        row.get(f"clip_{side}_video_path_windows")
        or row.get(f"clip_{side}_video_path")
        or row.get(f"clip_{side}_video")
        or ""
    )
    return Path(value)


def build_batch_messages(
    rows: list[dict[str, str]],
    frames_per_clip: int,
    image_size: int,
    jpeg_quality: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    content: list[dict[str, Any]] = []
    batch_meta: dict[str, dict[str, Any]] = {}
    overview_lines = [
        "You are an offline visual-semantic preference annotator for robot manipulation progress.",
        "For each pair, compare clip A and clip B and decide which clip shows more valid progress toward the task goal.",
        "Use visual evidence only. If similar or ambiguous, return unsure.",
        "Important: do not mark a clip as complete just because it is visually close; release, stability, support, and insertion depth matter when visible.",
        "",
        "Return valid JSON only with this schema:",
        '{"labels":[{"pair_id":"...","preference":"A>B | B>A | unsure","confidence":0.0,"clip_a_stage":"...","clip_b_stage":"...","reason":"...","visible_failure_or_uncertainty":"..."}]}',
        "",
        "Pairs in this request:",
    ]
    for row in rows:
        overview_lines.append(
            f"- {row['pair_id']}: task={row['task_id']}; goal={row.get('task_goal_text','')}; pair_type={row.get('pair_type','')}"
        )
    content.append({"type": "text", "text": "\n".join(overview_lines)})

    for row in rows:
        pair_id = row["pair_id"]
        a_indices = parse_frame_indices(
            row.get("clip_a_sample_frame_indices", ""),
            int(float(row.get("clip_a_start_frame", 0))),
            int(float(row.get("clip_a_end_frame_exclusive", 1))),
            frames_per_clip,
        )
        b_indices = parse_frame_indices(
            row.get("clip_b_sample_frame_indices", ""),
            int(float(row.get("clip_b_start_frame", 0))),
            int(float(row.get("clip_b_end_frame_exclusive", 1))),
            frames_per_clip,
        )
        a_frames = read_video_frames(video_path(row, "a"), a_indices)
        b_frames = read_video_frames(video_path(row, "b"), b_indices)
        batch_meta[pair_id] = {
            "a_indices": a_indices,
            "b_indices": b_indices,
            "clip_a_video_path": str(video_path(row, "a")),
            "clip_b_video_path": str(video_path(row, "b")),
        }

        content.append(
            {
                "type": "text",
                "text": (
                    f"PAIR {pair_id}\n"
                    f"Task: {row['task_id']}\n"
                    f"Goal: {row.get('task_goal_text','')}\n"
                    f"Clip A frames follow in chronological order: {a_indices}"
                ),
            }
        )
        for i, frame in enumerate(a_frames):
            content.append({"type": "text", "text": f"{pair_id} A{i} frame_index={a_indices[i]}"})
            content.append({"type": "image_url", "image_url": {"url": frame_to_data_url(frame, image_size, jpeg_quality)}})
        content.append({"type": "text", "text": f"{pair_id} Clip B frames follow in chronological order: {b_indices}"})
        for i, frame in enumerate(b_frames):
            content.append({"type": "text", "text": f"{pair_id} B{i} frame_index={b_indices[i]}"})
            content.append({"type": "image_url", "image_url": {"url": frame_to_data_url(frame, image_size, jpeg_quality)}})

    content.append(
        {
            "type": "text",
            "text": (
                "Now return exactly one JSON object with a labels array containing one result per pair_id above. "
                "Do not include markdown fences or extra prose."
            ),
        }
    )
    return [
        {"role": "system", "content": "Return concise valid JSON only. Do not include markdown fences."},
        {"role": "user", "content": content},
    ], batch_meta


def existing_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = load_csv(path)
    return {row["pair_id"]: row for row in rows}


def label_from_parsed(row: dict[str, str], parsed: dict[str, Any], model: str, raw_path: Path, batch_id: str, elapsed: float) -> dict[str, Any]:
    preference = normalize_preference(parsed.get("preference"))
    confidence = clamp_confidence(parsed.get("confidence"))
    candidate_label = row.get("candidate_label", "")
    return {
        "pair_id": row["pair_id"],
        "task_id": row["task_id"],
        "split": row.get("split", ""),
        "pair_type": row["pair_type"],
        "candidate_label": candidate_label,
        "candidate_confidence": row.get("candidate_confidence", ""),
        "needs_vlm_preference_label": row.get("needs_vlm_preference_label", ""),
        "mimo_preference": preference,
        "mimo_confidence": confidence,
        "agrees_with_candidate": "true" if preference == candidate_label else ("unsure" if preference == "unsure" else "false"),
        "clip_a_id": row["clip_a_id"],
        "clip_b_id": row["clip_b_id"],
        "clip_a_stage": str(parsed.get("clip_a_stage", "")),
        "clip_b_stage": str(parsed.get("clip_b_stage", "")),
        "reason": str(parsed.get("reason", "")),
        "visible_failure_or_uncertainty": str(parsed.get("visible_failure_or_uncertainty", "")),
        "mimo_model": model,
        "batch_id": batch_id,
        "api_elapsed_sec": round(elapsed, 3),
        "raw_response_path": str(raw_path),
    }


def error_label(row: dict[str, str], model: str, raw_path: Path, batch_id: str, error: str) -> dict[str, Any]:
    candidate_label = row.get("candidate_label", "")
    return {
        "pair_id": row["pair_id"],
        "task_id": row["task_id"],
        "split": row.get("split", ""),
        "pair_type": row["pair_type"],
        "candidate_label": candidate_label,
        "candidate_confidence": row.get("candidate_confidence", ""),
        "needs_vlm_preference_label": row.get("needs_vlm_preference_label", ""),
        "mimo_preference": "unsure",
        "mimo_confidence": 0.0,
        "agrees_with_candidate": "unsure",
        "clip_a_id": row["clip_a_id"],
        "clip_b_id": row["clip_b_id"],
        "clip_a_stage": "",
        "clip_b_stage": "",
        "reason": f"ERROR: {error}",
        "visible_failure_or_uncertainty": "",
        "mimo_model": model,
        "batch_id": batch_id,
        "api_elapsed_sec": 0.0,
        "raw_response_path": str(raw_path),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pref = Counter(row["mimo_preference"] for row in rows)
    by_task = defaultdict(Counter)
    by_pair_type = defaultdict(Counter)
    agreement = Counter(row["agrees_with_candidate"] for row in rows)
    needs = defaultdict(Counter)
    for row in rows:
        by_task[row["task_id"]][row["mimo_preference"]] += 1
        by_pair_type[row["pair_type"]][row["mimo_preference"]] += 1
        needs[row.get("needs_vlm_preference_label", "")][row["mimo_preference"]] += 1
    return {
        "num_labels": len(rows),
        "preference_distribution": dict(by_pref),
        "candidate_agreement": dict(agreement),
        "by_task": {key: dict(value) for key, value in by_task.items()},
        "by_pair_type": {key: dict(value) for key, value in by_pair_type.items()},
        "by_needs_vlm_preference_label": {key: dict(value) for key, value in needs.items()},
    }


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "MiMo bootstrap pairwise preference labels",
        "",
        f"已标注 pair 数：{summary['num_labels']}",
        "",
        "Preference 分布：",
    ]
    lines.extend(f"- {k}: {v}" for k, v in summary["preference_distribution"].items())
    lines += ["", "候选标签一致性："]
    lines.extend(f"- {k}: {v}" for k, v in summary["candidate_agreement"].items())
    lines += ["", "按任务："]
    lines.extend(f"- {k}: {v}" for k, v in summary["by_task"].items())
    lines += [
        "",
        "注意：MiMo 标签是视觉语义分支，不是最终 reward 标签；后续应和 stereo/contact/stage 进行 fusion。",
    ]
    (out_dir / "MiMo_bootstrap_pairwise标注说明.txt").write_text("\n".join(lines), encoding="utf-8")


def load_queues(queue_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for queue_path in queue_paths:
        rows.extend(load_csv(queue_path))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-call MiMo for bootstrap A/B pairwise preference labels.")
    parser.add_argument("--queues", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(r"E:\reward_model_dataset\vlm_labels\mimo_bootstrap_pairwise_v1"))
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--base-url", default=os.environ.get("MIMO_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("MIMO_MODEL", "mimo-v2.5"))
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0, help="Maximum new pair labels to request; 0 means no limit.")
    parser.add_argument("--need-filter", default="true", choices=["true", "optional", "all"])
    parser.add_argument("--task-id", default="")
    parser.add_argument("--pair-type", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "mimo_pairwise_labels.csv"

    queue = load_queues(args.queues)
    if args.need_filter != "all":
        queue = [row for row in queue if row.get("needs_vlm_preference_label", "").lower() == args.need_filter]
    if args.task_id:
        queue = [row for row in queue if row["task_id"] == args.task_id]
    if args.pair_type:
        queue = [row for row in queue if row["pair_type"] == args.pair_type]

    existing = {} if args.force else existing_labels(labels_path)
    labels: list[dict[str, Any]] = list(existing.values())
    api_key = read_api_key(ROOT, args.api_key_file)
    base_url = args.base_url or default_base_url(api_key)

    pending = [row for row in queue if args.force or row["pair_id"] not in existing]
    if args.limit:
        pending = pending[: args.limit]

    processed_pairs = 0
    batch_index = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        batch_index += 1
        batch_id = f"batch_{int(time.time())}_{batch_index:06d}"
        raw_path = raw_dir / f"{batch_id}.json"
        pair_ids = [row["pair_id"] for row in batch]
        print(f"Calling MiMo {batch_id}: {pair_ids}")
        try:
            messages, batch_meta = build_batch_messages(
                batch,
                frames_per_clip=args.frames_per_clip,
                image_size=args.image_size,
                jpeg_quality=args.jpeg_quality,
            )
            started = time.time()
            response = request_chat_completion(base_url, api_key, args.model, messages, args.max_tokens, 0.0, args.timeout_sec)
            elapsed = time.time() - started
            text = extract_message_text(response)
            parsed_labels, parse_status, parse_error = safe_parse_batch_json(text)
            parsed_by_id = {str(item.get("pair_id", "")): item for item in parsed_labels}
            raw = {
                "batch_id": batch_id,
                "pair_ids": pair_ids,
                "model": args.model,
                "api_elapsed_sec": elapsed,
                "parse_status": parse_status,
                "parse_error": parse_error,
                "batch_meta": batch_meta,
                "response": response,
                "message_text": text,
                "parsed_labels": parsed_labels,
            }
            write_json(raw_path, raw)
            for row in batch:
                parsed = parsed_by_id.get(row["pair_id"])
                if parsed is None:
                    label = error_label(row, args.model, raw_path, batch_id, "MiMo response missing this pair_id")
                else:
                    label = label_from_parsed(row, parsed, args.model, raw_path, batch_id, elapsed)
                labels.append(label)
                processed_pairs += 1
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            write_json(raw_path, {"batch_id": batch_id, "pair_ids": pair_ids, "error": err, "queue_rows": batch})
            for row in batch:
                labels.append(error_label(row, args.model, raw_path, batch_id, err))
                processed_pairs += 1

        labels.sort(key=lambda item: item["pair_id"])
        write_csv(labels_path, labels)
        write_json(out_dir / "mimo_pairwise_labels.json", labels)
        summary = summarize(labels)
        write_json(out_dir / "phase2_summary.json", summary)
        write_report(out_dir, summary)
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    summary = summarize(labels)
    write_json(out_dir / "phase2_summary.json", summary)
    write_report(out_dir, summary)
    print(json.dumps({"new_pair_labels": processed_pairs, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
