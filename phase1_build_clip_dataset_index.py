import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2


PROJECT_ROOT = Path.cwd()
OUT_DIR = Path(r"D:\Users\User\Desktop\reward_phase0_inventory\phase1_clip_dataset")

WINDOW_FRAMES = 16
STRIDE_FRAMES = 8
MIN_CLIP_FRAMES = 8
VLM_SAMPLE_FRAMES_PER_CLIP = 6
MAX_TEMPORAL_PAIRS_PER_SUCCESS_TRAJECTORY = 30

TASK_GOAL_TEXT = {
    "stackcube": "A robot arm stacks one cube on top of another cube and releases it stably.",
    "stackpyramid": "A robot arm builds a stable three-cube pyramid by placing the top cube on the base cubes.",
    "peginsertion": "A robot arm picks up a peg, aligns it with a hole, inserts the peg, and releases it stably.",
}


OFFICIAL_TASKS = [
    {
        "trajectory_id": "OFFICIAL-SC-SUCCESS-00",
        "task_id": "stackcube",
        "env_id": "StackCube-v1",
        "video_rel": "paper_style_tasks/outputs/wsl_motionplanning/StackCube-v1/motionplanning/stackcube_wsl_mplib.mp4",
        "h5_rel": "paper_style_tasks/outputs/wsl_motionplanning/StackCube-v1/motionplanning/stackcube_wsl_mplib.h5",
        "metadata_rel": "paper_style_tasks/outputs/wsl_motionplanning/StackCube-v1/motionplanning/stackcube_wsl_mplib.json",
        "run_summary_rel": "paper_style_tasks/outputs/wsl_motionplanning/StackCube-v1/motionplanning/stackcube_wsl_mplib_run_summary.json",
    },
    {
        "trajectory_id": "OFFICIAL-SP-SUCCESS-02",
        "task_id": "stackpyramid",
        "env_id": "StackPyramid-v1",
        "video_rel": "paper_style_tasks/outputs/wsl_motionplanning/StackPyramid-v1/motionplanning/stackpyramid_wsl_mplib.mp4",
        "h5_rel": "paper_style_tasks/outputs/wsl_motionplanning/StackPyramid-v1/motionplanning/stackpyramid_wsl_mplib.h5",
        "metadata_rel": "paper_style_tasks/outputs/wsl_motionplanning/StackPyramid-v1/motionplanning/stackpyramid_wsl_mplib.json",
        "run_summary_rel": "paper_style_tasks/outputs/wsl_motionplanning/StackPyramid-v1/motionplanning/stackpyramid_wsl_mplib_run_summary.json",
    },
    {
        "trajectory_id": "OFFICIAL-PEG-SUCCESS-02",
        "task_id": "peginsertion",
        "env_id": "PegInsertionSide-v1",
        "video_rel": "paper_style_tasks/outputs/wsl_motionplanning/PegInsertionSide-v1/motionplanning/peg_insertion_wsl_mplib.mp4",
        "h5_rel": "paper_style_tasks/outputs/wsl_motionplanning/PegInsertionSide-v1/motionplanning/peg_insertion_wsl_mplib.h5",
        "metadata_rel": "paper_style_tasks/outputs/wsl_motionplanning/PegInsertionSide-v1/motionplanning/peg_insertion_wsl_mplib.json",
        "run_summary_rel": "paper_style_tasks/outputs/wsl_motionplanning/PegInsertionSide-v1/motionplanning/peg_insertion_wsl_mplib_run_summary.json",
    },
]


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def video_props(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video_exists": video_path.exists(),
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "duration_sec": 0.0,
        }
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "video_exists": video_path.exists(),
        "frame_count": frame_count,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "duration_sec": round(frame_count / fps, 3) if fps > 0 else 0.0,
    }


def bool_text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    return "true" if text in {"true", "1", "yes"} else "false"


def official_success_record(summary):
    if not summary:
        return {}
    for record in summary.get("records", []):
        if record.get("success") is True and record.get("saved") is True:
            return record
    return {}


def build_trajectory_manifest():
    rows = []
    for spec in OFFICIAL_TASKS:
        video_path = PROJECT_ROOT / spec["video_rel"]
        h5_path = PROJECT_ROOT / spec["h5_rel"]
        metadata_path = PROJECT_ROOT / spec["metadata_rel"]
        summary_path = PROJECT_ROOT / spec["run_summary_rel"]
        props = video_props(video_path)
        summary = read_json(summary_path)
        record = official_success_record(summary)
        rows.append(
            {
                "trajectory_id": spec["trajectory_id"],
                "task_id": spec["task_id"],
                "env_id": spec["env_id"],
                "source_group": "official_success",
                "source_type": "official_motionplanning_success",
                "seed": record.get("seed", ""),
                "expected_success": "true",
                "observed_success": "true",
                "near_miss_type": "",
                "target_failure_mode": "",
                "progress_rank_terminal": 3,
                "video_path": spec["video_rel"],
                "h5_path": spec["h5_rel"] if h5_path.exists() else "",
                "metadata_path": spec["metadata_rel"] if metadata_path.exists() else "",
                "run_summary_path": spec["run_summary_rel"] if summary_path.exists() else "",
                "frame_count": props["frame_count"],
                "fps": props["fps"],
                "width": props["width"],
                "height": props["height"],
                "duration_sec": props["duration_sec"],
                "elapsed_steps": record.get("elapsed_steps", ""),
                "notes": "Official ManiSkill motion-planning success trajectory.",
            }
        )

    failure_manifest_path = PROJECT_ROOT / "near_miss_samples/outputs/failure_manifest.csv"
    if failure_manifest_path.exists():
        for row in read_csv(failure_manifest_path):
            video_rel = row["video_path"]
            video_path = PROJECT_ROOT / video_rel
            props = video_props(video_path)
            rank = terminal_progress_rank(row)
            manifest_frames = row.get("num_frames", "").strip()
            notes = row.get("notes", "")
            if manifest_frames and props["frame_count"] and int(float(manifest_frames)) != props["frame_count"]:
                notes = (notes + " " if notes else "") + f"Manifest num_frames={manifest_frames}, cv2 frame_count={props['frame_count']}."
            rows.append(
                {
                    "trajectory_id": row["sample_id"],
                    "task_id": row["task_id"],
                    "env_id": row["env_id"],
                    "source_group": "near_miss_failure",
                    "source_type": row["source_type"],
                    "seed": row.get("seed", ""),
                    "expected_success": bool_text(row.get("expected_success", "false")),
                    "observed_success": bool_text(row.get("observed_success", "false")),
                    "near_miss_type": row.get("near_miss_type", ""),
                    "target_failure_mode": row.get("target_failure_mode", ""),
                    "progress_rank_terminal": rank,
                    "video_path": video_rel,
                    "h5_path": "",
                    "metadata_path": "",
                    "run_summary_path": "",
                    "frame_count": props["frame_count"],
                    "fps": props["fps"],
                    "width": props["width"],
                    "height": props["height"],
                    "duration_sec": props["duration_sec"],
                    "elapsed_steps": "",
                    "notes": notes,
                }
            )
    return rows


def terminal_progress_rank(row):
    if bool_text(row.get("observed_success", "false")) == "true":
        return 3
    near_miss_type = row.get("near_miss_type", "")
    source_type = row.get("source_type", "")
    if "early_grasp_or_lift_failure" in near_miss_type:
        return 0
    if near_miss_type in {
        "truncated_place_before_stable_release",
        "perturbed_cube_lateral_offset",
        "truncated_top_cube_place_before_stable_release",
        "truncated_insert_before_release_stable",
        "official_depth_near_miss",
    }:
        return 2
    if source_type == "official_motionplanning_failure":
        return 1
    return 1


def clip_frame_indices(start, end_exclusive, n):
    length = max(0, end_exclusive - start)
    if length <= 0:
        return ""
    if length == 1:
        return str(start)
    samples = []
    count = min(VLM_SAMPLE_FRAMES_PER_CLIP, length)
    for i in range(count):
        pos = start + round(i * (length - 1) / max(1, count - 1))
        samples.append(min(max(int(pos), 0), n - 1))
    return ";".join(str(x) for x in samples)


def build_clips(trajectories):
    clips = []
    for traj in trajectories:
        n = int(traj["frame_count"])
        if n <= 0:
            continue
        starts = list(range(0, max(1, n - WINDOW_FRAMES + 1), STRIDE_FRAMES))
        if n >= MIN_CLIP_FRAMES:
            final_start = max(0, n - WINDOW_FRAMES)
            if final_start not in starts:
                starts.append(final_start)
        if not starts:
            starts = [0]
        starts = sorted(set(starts))
        for idx, start in enumerate(starts):
            end = min(n, start + WINDOW_FRAMES)
            if end - start < MIN_CLIP_FRAMES and n >= MIN_CLIP_FRAMES:
                start = max(0, end - MIN_CLIP_FRAMES)
            center = (start + end - 1) / 2.0
            clip_id = f"{traj['trajectory_id']}-C{idx:03d}"
            clips.append(
                {
                    "clip_id": clip_id,
                    "trajectory_id": traj["trajectory_id"],
                    "task_id": traj["task_id"],
                    "source_group": traj["source_group"],
                    "source_type": traj["source_type"],
                    "expected_success": traj["expected_success"],
                    "observed_success": traj["observed_success"],
                    "near_miss_type": traj["near_miss_type"],
                    "progress_rank_terminal": traj["progress_rank_terminal"],
                    "video_path": traj["video_path"],
                    "frame_count_video": n,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "num_frames": end - start,
                    "center_frame": round(center, 2),
                    "center_time_proxy": round(center / max(1, n - 1), 4),
                    "vlm_sample_frame_indices": clip_frame_indices(start, end, n),
                    "clip_window_frames": WINDOW_FRAMES,
                    "clip_stride_frames": STRIDE_FRAMES,
                    "has_h5_state": "true" if traj["h5_path"] else "false",
                    "auto_stage_label": "",
                    "notes": "",
                }
            )
    return clips


def final_clip_for_trajectory(clips):
    by_traj = defaultdict(list)
    for clip in clips:
        by_traj[clip["trajectory_id"]].append(clip)
    return {
        traj_id: sorted(items, key=lambda c: (int(c["end_frame_exclusive"]), int(c["start_frame"])))[-1]
        for traj_id, items in by_traj.items()
    }


def add_pair(pairs, pair_type, clip_a, clip_b, label, confidence, rationale, needs_vlm="true"):
    pairs.append(
        {
            "pair_id": f"P{len(pairs):05d}",
            "task_id": clip_a["task_id"],
            "pair_type": pair_type,
            "clip_a_id": clip_a["clip_id"],
            "clip_b_id": clip_b["clip_id"],
            "trajectory_a_id": clip_a["trajectory_id"],
            "trajectory_b_id": clip_b["trajectory_id"],
            "label": label,
            "label_meaning": "A higher progress than B" if label == "A>B" else "B higher progress than A",
            "confidence": confidence,
            "label_source": "phase1_auto_index_rule",
            "needs_vlm_preference_label": needs_vlm,
            "rationale": rationale,
        }
    )


def build_pairs(trajectories, clips):
    pairs = []
    clips_by_traj = defaultdict(list)
    for clip in clips:
        clips_by_traj[clip["trajectory_id"]].append(clip)
    for traj_id in clips_by_traj:
        clips_by_traj[traj_id] = sorted(clips_by_traj[traj_id], key=lambda c: int(c["start_frame"]))

    traj_by_id = {t["trajectory_id"]: t for t in trajectories}

    for traj in trajectories:
        if traj["source_group"] != "official_success":
            continue
        candidates = []
        tclips = clips_by_traj[traj["trajectory_id"]]
        for i in range(len(tclips) - 1):
            candidates.append((i, i + 1, "adjacent temporal order in a successful trajectory"))
        for gap in [2, 4, 8]:
            for i in range(0, max(0, len(tclips) - gap), max(1, gap)):
                candidates.append((i, i + gap, f"{gap}-clip temporal gap in a successful trajectory"))
        seen = set()
        kept = 0
        for i, j, reason in candidates:
            if (i, j) in seen:
                continue
            seen.add((i, j))
            add_pair(
                pairs,
                "intra_success_temporal",
                tclips[i],
                tclips[j],
                "B>A",
                "medium",
                reason,
                needs_vlm="optional",
            )
            kept += 1
            if kept >= MAX_TEMPORAL_PAIRS_PER_SUCCESS_TRAJECTORY:
                break

    finals = final_clip_for_trajectory(clips)
    trajs_by_task = defaultdict(list)
    for traj in trajectories:
        trajs_by_task[traj["task_id"]].append(traj)

    for task_id, task_trajs in trajs_by_task.items():
        for i, ta in enumerate(task_trajs):
            for tb in task_trajs[i + 1 :]:
                rank_a = int(ta["progress_rank_terminal"])
                rank_b = int(tb["progress_rank_terminal"])
                if rank_a == rank_b:
                    continue
                if rank_a > rank_b:
                    high, low = ta, tb
                else:
                    high, low = tb, ta
                high_clip = finals[high["trajectory_id"]]
                low_clip = finals[low["trajectory_id"]]
                confidence = "high" if high["source_group"] == "official_success" else "medium"
                add_pair(
                    pairs,
                    "terminal_rank_cross_trajectory",
                    high_clip,
                    low_clip,
                    "A>B",
                    confidence,
                    f"Terminal progress rank {high['progress_rank_terminal']} > {low['progress_rank_terminal']} within {task_id}.",
                    needs_vlm="true",
                )
    return pairs


def build_annotation_queue(pairs, clips):
    clip_by_id = {clip["clip_id"]: clip for clip in clips}
    rows = []
    for pair in pairs:
        a = clip_by_id[pair["clip_a_id"]]
        b = clip_by_id[pair["clip_b_id"]]
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "task_id": pair["task_id"],
                "task_goal_text": TASK_GOAL_TEXT.get(pair["task_id"], ""),
                "pair_type": pair["pair_type"],
                "candidate_label": pair["label"],
                "candidate_confidence": pair["confidence"],
                "needs_vlm_preference_label": pair["needs_vlm_preference_label"],
                "clip_a_id": a["clip_id"],
                "clip_a_video_path": a["video_path"],
                "clip_a_start_frame": a["start_frame"],
                "clip_a_end_frame_exclusive": a["end_frame_exclusive"],
                "clip_a_sample_frame_indices": a["vlm_sample_frame_indices"],
                "clip_b_id": b["clip_id"],
                "clip_b_video_path": b["video_path"],
                "clip_b_start_frame": b["start_frame"],
                "clip_b_end_frame_exclusive": b["end_frame_exclusive"],
                "clip_b_sample_frame_indices": b["vlm_sample_frame_indices"],
                "prompt_instruction": "Compare clip A and clip B for the same robot task. Return A>B if clip A shows more task progress, B>A if clip B shows more task progress, or unsure if the evidence is ambiguous.",
                "rationale": pair["rationale"],
            }
        )
    return rows


def summarize(trajectories, clips, pairs):
    by_task = defaultdict(lambda: {"trajectories": 0, "clips": 0, "pairs": 0})
    for row in trajectories:
        by_task[row["task_id"]]["trajectories"] += 1
    for row in clips:
        by_task[row["task_id"]]["clips"] += 1
    for row in pairs:
        by_task[row["task_id"]]["pairs"] += 1
    pair_types = defaultdict(int)
    for row in pairs:
        pair_types[row["pair_type"]] += 1
    return {
        "output_dir": str(OUT_DIR),
        "window_frames": WINDOW_FRAMES,
        "stride_frames": STRIDE_FRAMES,
        "min_clip_frames": MIN_CLIP_FRAMES,
        "vlm_sample_frames_per_clip": VLM_SAMPLE_FRAMES_PER_CLIP,
        "num_trajectories": len(trajectories),
        "num_clips": len(clips),
        "num_pairs": len(pairs),
        "by_task": dict(by_task),
        "pair_types": dict(pair_types),
    }


def write_report(summary):
    lines = [
        "阶段 1：clip 片段索引与 A/B pair 候选集",
        "生成时间：2026-06-11",
        "",
        "一、这一步做了什么",
        "本阶段没有复制或切出大量视频文件，只生成索引型数据集：轨迹 manifest、clip 帧区间索引、A/B pair 候选清单。",
        "后续 VLM 标注或物理规则分支可以按 video_path + start_frame/end_frame_exclusive 读取对应片段。",
        "",
        "二、切片参数",
        f"- window_frames：{summary['window_frames']}",
        f"- stride_frames：{summary['stride_frames']}",
        f"- min_clip_frames：{summary['min_clip_frames']}",
        f"- 每个 clip 建议给 VLM 抽帧数：{summary['vlm_sample_frames_per_clip']}",
        "",
        "三、数据规模",
        f"- trajectory 数：{summary['num_trajectories']}",
        f"- clip 数：{summary['num_clips']}",
        f"- A/B pair 候选数：{summary['num_pairs']}",
        "",
        "按任务统计：",
    ]
    for task_id, stats in summary["by_task"].items():
        lines.append(
            f"- {task_id}: trajectories={stats['trajectories']}, clips={stats['clips']}, pairs={stats['pairs']}"
        )
    lines += [
        "",
        "pair 类型统计：",
    ]
    for pair_type, count in summary["pair_types"].items():
        lines.append(f"- {pair_type}: {count}")
    lines += [
        "",
        "四、当前 pair 标签口径",
        "- intra_success_temporal：同一成功轨迹中，较晚 clip 默认比早期 clip 进度更高，标签为 B>A。",
        "- terminal_rank_cross_trajectory：同一任务内，成功终段 > near-miss 终段 > 明显失败终段，标签为 A>B。",
        "- 这些是阶段一的候选/弱标签，不等于最终训练标签；后续需要 MiMo preference、物理分支和融合规则再确认。",
        "",
        "五、输出文件",
        "- trajectory_manifest.csv/json：轨迹级清单。",
        "- clip_manifest.csv/json：clip 级帧区间索引。",
        "- pair_manifest.csv/json：A/B pair 候选清单。",
        "- pair_annotation_queue.csv/json：展开后的 VLM 标注队列，已包含 A/B 视频路径、帧区间、抽帧索引和任务文本。",
        "- phase1_summary.json：本阶段统计摘要。",
        "",
        "六、下一步建议",
        "先人工检查 pair_manifest 中每类 pair 的数量和口径；确认后再进入 VLM pairwise 标注或规则版物理分支。",
    ]
    (OUT_DIR / "阶段1_clip与pair索引说明.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectories = build_trajectory_manifest()
    clips = build_clips(trajectories)
    pairs = build_pairs(trajectories, clips)
    annotation_queue = build_annotation_queue(pairs, clips)
    summary = summarize(trajectories, clips, pairs)

    write_csv(OUT_DIR / "trajectory_manifest.csv", trajectories)
    write_json(OUT_DIR / "trajectory_manifest.json", trajectories)
    write_csv(OUT_DIR / "clip_manifest.csv", clips)
    write_json(OUT_DIR / "clip_manifest.json", clips)
    write_csv(OUT_DIR / "pair_manifest.csv", pairs)
    write_json(OUT_DIR / "pair_manifest.json", pairs)
    write_csv(OUT_DIR / "pair_annotation_queue.csv", annotation_queue)
    write_json(OUT_DIR / "pair_annotation_queue.json", annotation_queue)
    write_json(OUT_DIR / "phase1_summary.json", summary)
    write_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
