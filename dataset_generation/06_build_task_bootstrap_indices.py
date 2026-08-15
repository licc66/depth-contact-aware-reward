from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


TASK_GOAL_TEXT = {
    "stackcube": "A robot arm stacks one cube on top of another cube and releases it stably.",
    "stackpyramid": "A robot arm stacks the top cube onto two base cubes to form a stable pyramid.",
    "peginsertion": "A robot arm inserts the peg into the side hole of the box.",
}

PAIR_PREFIX = {
    "stackcube": "SCB",
    "stackpyramid": "SPB",
    "peginsertion": "PEGB",
}

WINDOW_FRAMES = 16
STRIDE_FRAMES = 8
MIN_CLIP_FRAMES = 8
VLM_SAMPLE_FRAMES_PER_CLIP = 6


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def infer_task_id(rows: list[dict[str, str]], dataset: Path) -> str:
    for row in rows:
        task_id = (row.get("task_id") or "").strip().lower()
        if task_id:
            return task_id
    name = dataset.name.lower()
    if "stackpyramid" in name:
        return "stackpyramid"
    if "peginsertion" in name or "peg" in name:
        return "peginsertion"
    if "stackcube" in name:
        return "stackcube"
    raise ValueError(f"Cannot infer task_id from {dataset}")


def video_props(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"frame_count": 0, "fps": 0.0, "width": 0, "height": 0, "duration_sec": 0.0}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "frame_count": n,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "duration_sec": round(n / fps, 3) if fps > 0 else 0.0,
    }


def clip_frame_indices(start: int, end: int, n: int) -> str:
    length = max(0, end - start)
    if length <= 0:
        return ""
    count = min(VLM_SAMPLE_FRAMES_PER_CLIP, length)
    if count == 1:
        return str(start)
    return ";".join(str(min(n - 1, start + round(i * (length - 1) / (count - 1)))) for i in range(count))


def terminal_clip(clips: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(clips, key=lambda c: (int(c["end_frame_exclusive"]), int(c["start_frame"])))[-1]


def enrich_trajectories(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        video_path = Path(row.get("video_path_windows") or row.get("video_path") or row.get("video_path_wsl", ""))
        props = video_props(video_path)
        source_success_id = row.get("source_success_id") or row["sample_id"]
        rank = to_int(row.get("progress_rank_terminal"), 3 if boolish(row.get("observed_success")) else 1)
        out.append(
            {
                **row,
                "source_success_id": source_success_id,
                "video_path_windows": str(video_path),
                "frame_count": props["frame_count"],
                "fps": props["fps"],
                "width": props["width"],
                "height": props["height"],
                "duration_sec": props["duration_sec"],
                "progress_rank_terminal": rank,
            }
        )
    return out


def build_clips(task_id: str, trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for traj in trajectories:
        n = int(traj["frame_count"])
        if n <= 0:
            continue
        starts = list(range(0, max(1, n - WINDOW_FRAMES + 1), STRIDE_FRAMES))
        final_start = max(0, n - WINDOW_FRAMES)
        if final_start not in starts:
            starts.append(final_start)
        starts = sorted(set(starts))
        for idx, start in enumerate(starts):
            end = min(n, start + WINDOW_FRAMES)
            if end - start < MIN_CLIP_FRAMES and n >= MIN_CLIP_FRAMES:
                start = max(0, end - MIN_CLIP_FRAMES)
            center = (start + end - 1) / 2.0
            clips.append(
                {
                    "clip_id": f"{traj['sample_id']}-C{idx:03d}",
                    "trajectory_id": traj["sample_id"],
                    "source_success_id": traj["source_success_id"],
                    "task_id": task_id,
                    "split": traj.get("split", "train"),
                    "source_type": traj.get("source_type", ""),
                    "near_miss_type": traj.get("near_miss_type", ""),
                    "expected_success": traj.get("expected_success", ""),
                    "observed_success": traj.get("observed_success", ""),
                    "progress_rank_terminal": traj["progress_rank_terminal"],
                    "video_path_windows": traj["video_path_windows"],
                    "frame_count_video": n,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "num_frames": end - start,
                    "center_frame": round(center, 2),
                    "center_time_proxy": round(center / max(1, n - 1), 4),
                    "vlm_sample_frame_indices": clip_frame_indices(start, end, n),
                    "clip_window_frames": WINDOW_FRAMES,
                    "clip_stride_frames": STRIDE_FRAMES,
                }
            )
    return clips


def add_pair(
    task_id: str,
    pairs: list[dict[str, Any]],
    pair_type: str,
    a: dict[str, Any],
    b: dict[str, Any],
    label: str,
    confidence: str,
    reason: str,
    needs_vlm: str,
) -> None:
    prefix = PAIR_PREFIX.get(task_id, "TASK")
    pairs.append(
        {
            "pair_id": f"{prefix}-P{len(pairs):06d}",
            "task_id": task_id,
            "split": a["split"],
            "pair_type": pair_type,
            "clip_a_id": a["clip_id"],
            "clip_b_id": b["clip_id"],
            "trajectory_a_id": a["trajectory_id"],
            "trajectory_b_id": b["trajectory_id"],
            "source_success_a_id": a["source_success_id"],
            "source_success_b_id": b["source_success_id"],
            "label": label,
            "label_meaning": "A higher progress than B" if label == "A>B" else "B higher progress than A",
            "confidence": confidence,
            "label_source": f"{task_id}_bootstrap_index_rule",
            "needs_vlm_preference_label": needs_vlm,
            "rationale": reason,
        }
    )


def pair_type_for(high: dict[str, Any], low: dict[str, Any]) -> str:
    if high["source_type"] == "official_motionplanning_success" and low["source_type"] == "perturbed_success_final_state":
        if high["task_id"] == "peginsertion":
            return "success_vs_peg_near_miss"
        if high["task_id"] == "stackpyramid":
            return "success_vs_pyramid_near_miss"
        return "success_vs_offset_hard_negative"
    if high["source_type"] == "official_motionplanning_success":
        return "success_vs_truncated_terminal"
    if low["source_type"] == "truncated_success_trajectory":
        return "near_miss_vs_early_truncated"
    return "terminal_rank_cross_generated"


def build_pairs(task_id: str, trajectories: list[dict[str, Any]], clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    clips_by_traj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        clips_by_traj[clip["trajectory_id"]].append(clip)
    for key in list(clips_by_traj):
        clips_by_traj[key] = sorted(clips_by_traj[key], key=lambda c: int(c["start_frame"]))

    for traj in trajectories:
        if traj["source_type"] != "official_motionplanning_success":
            continue
        tclips = clips_by_traj[traj["sample_id"]]
        for gap in [2, 4, 8]:
            for i in range(0, max(0, len(tclips) - gap), max(1, gap // 2)):
                add_pair(
                    task_id,
                    pairs,
                    "intra_success_temporal_gap",
                    tclips[i],
                    tclips[i + gap],
                    "B>A",
                    "medium",
                    f"Same successful trajectory; clip B is {gap} clip windows later than clip A.",
                    "optional",
                )

    finals = {traj_id: terminal_clip(items) for traj_id, items in clips_by_traj.items()}
    trajs_by_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for traj in trajectories:
        trajs_by_source[(traj.get("split", "train"), traj["source_success_id"])].append(traj)

    for (_split, _source_id), items in trajs_by_source.items():
        for i, ta in enumerate(items):
            for tb in items[i + 1 :]:
                ra, rb = int(ta["progress_rank_terminal"]), int(tb["progress_rank_terminal"])
                if ra == rb:
                    continue
                high, low = (ta, tb) if ra > rb else (tb, ta)
                add_pair(
                    task_id,
                    pairs,
                    pair_type_for(high, low),
                    finals[high["sample_id"]],
                    finals[low["sample_id"]],
                    "A>B",
                    "high" if high["source_type"] == "official_motionplanning_success" else "medium",
                    f"Terminal rank {ra} vs {rb}; higher-rank trajectory should show more valid task progress.",
                    "true",
                )
    return pairs


def build_annotation_queue(task_id: str, pairs: list[dict[str, Any]], clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {clip["clip_id"]: clip for clip in clips}
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        a = by_id[pair["clip_a_id"]]
        b = by_id[pair["clip_b_id"]]
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "task_id": task_id,
                "split": pair["split"],
                "task_goal_text": TASK_GOAL_TEXT.get(task_id, ""),
                "pair_type": pair["pair_type"],
                "candidate_label": pair["label"],
                "candidate_confidence": pair["confidence"],
                "needs_vlm_preference_label": pair["needs_vlm_preference_label"],
                "clip_a_id": a["clip_id"],
                "clip_a_video_path_windows": a["video_path_windows"],
                "clip_a_start_frame": a["start_frame"],
                "clip_a_end_frame_exclusive": a["end_frame_exclusive"],
                "clip_a_sample_frame_indices": a["vlm_sample_frame_indices"],
                "clip_b_id": b["clip_id"],
                "clip_b_video_path_windows": b["video_path_windows"],
                "clip_b_start_frame": b["start_frame"],
                "clip_b_end_frame_exclusive": b["end_frame_exclusive"],
                "clip_b_sample_frame_indices": b["vlm_sample_frame_indices"],
                "prompt_instruction": "Compare clip A and clip B. Return A>B if clip A shows more valid task progress, B>A if clip B shows more valid task progress, or unsure.",
                "rationale": pair["rationale"],
            }
        )
    return rows


def summarize(trajectories: list[dict[str, Any]], clips: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = defaultdict(lambda: {"trajectories": 0, "clips": 0, "pairs": 0})
    for row in trajectories:
        by_split[row.get("split", "train")]["trajectories"] += 1
    for row in clips:
        by_split[row.get("split", "train")]["clips"] += 1
    for row in pairs:
        by_split[row.get("split", "train")]["pairs"] += 1
    by_pair_type = defaultdict(int)
    for row in pairs:
        by_pair_type[row["pair_type"]] += 1
    return {
        "task_id": trajectories[0].get("task_id") if trajectories else "",
        "num_trajectories": len(trajectories),
        "num_clips": len(clips),
        "num_pairs": len(pairs),
        "window_frames": WINDOW_FRAMES,
        "stride_frames": STRIDE_FRAMES,
        "by_split": dict(by_split),
        "by_pair_type": dict(by_pair_type),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clip and pair indices for a bootstrap dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.dataset / "trajectory_manifest.csv"
    trajectories = enrich_trajectories(load_csv(manifest))
    task_id = args.task_id or infer_task_id(trajectories, args.dataset)
    clips = build_clips(task_id, trajectories)
    pairs = build_pairs(task_id, trajectories, clips)
    queue = build_annotation_queue(task_id, pairs, clips)
    summary = summarize(trajectories, clips, pairs)
    summary["task_id"] = task_id

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "trajectory_manifest_enriched.csv", trajectories)
    write_json(args.out / "trajectory_manifest_enriched.json", trajectories)
    write_csv(args.out / "clip_manifest.csv", clips)
    write_json(args.out / "clip_manifest.json", clips)
    write_csv(args.out / "pair_manifest.csv", pairs)
    write_json(args.out / "pair_manifest.json", pairs)
    write_csv(args.out / "pair_annotation_queue.csv", queue)
    write_json(args.out / "pair_annotation_queue.json", queue)
    write_json(args.out / "index_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
