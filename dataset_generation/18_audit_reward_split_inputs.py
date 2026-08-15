from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2


SPLITS = ("train", "val", "test")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_indices(value: str) -> list[int]:
    if not value:
        return []
    out: list[int] = []
    for part in value.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        out.append(int(float(part)))
    return out


def inspect_video(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "openable": False,
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
        }
    cap = cv2.VideoCapture(str(path))
    openable = bool(cap.isOpened())
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if openable else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if openable else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if openable else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if openable else 0
    can_read_first = False
    if openable and frame_count > 0:
        ok, _ = cap.read()
        can_read_first = bool(ok)
    cap.release()
    return {
        "exists": True,
        "openable": openable,
        "can_read_first_frame": can_read_first,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def audit_split(split_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pair_rows: list[dict[str, str]] = []
    for split in SPLITS:
        path = split_dir / f"{split}_pairs.csv"
        rows = load_csv(path)
        for row in rows:
            row["_audit_split"] = split
        pair_rows.extend(rows)

    video_usage: dict[str, dict[str, Any]] = {}
    bad_pair_ids: set[str] = set()
    pair_issue_rows: list[dict[str, Any]] = []

    for row in pair_rows:
        for side in ("a", "b"):
            path_text = row.get(f"clip_{side}_video_path_local", "")
            indices = parse_indices(row.get(f"clip_{side}_sample_frame_indices", ""))
            if path_text not in video_usage:
                video_usage[path_text] = {
                    "video_path": path_text,
                    "splits": set(),
                    "pair_count": 0,
                    "max_sample_index": -1,
                    "sample_index_count": 0,
                }
            item = video_usage[path_text]
            item["splits"].add(row["_audit_split"])
            item["pair_count"] += 1
            item["max_sample_index"] = max(item["max_sample_index"], max(indices, default=-1))
            item["sample_index_count"] += len(indices)

    video_rows: list[dict[str, Any]] = []
    for path_text, usage in sorted(video_usage.items()):
        info = inspect_video(Path(path_text))
        max_index = int(usage["max_sample_index"])
        frame_count = int(info["frame_count"])
        indices_in_range = (
            bool(info["openable"])
            and bool(info.get("can_read_first_frame"))
            and frame_count > 0
            and max_index < frame_count
        )
        status = "ok" if info["exists"] and indices_in_range else "bad"
        row = {
            "video_path": path_text,
            "status": status,
            "splits": ";".join(sorted(usage["splits"])),
            "pair_count": usage["pair_count"],
            "sample_index_count": usage["sample_index_count"],
            "max_sample_index": max_index,
            **info,
            "indices_in_range": indices_in_range,
        }
        video_rows.append(row)

    bad_videos = {row["video_path"] for row in video_rows if row["status"] != "ok"}
    video_frame_counts = {row["video_path"]: int(row["frame_count"]) for row in video_rows}
    for row in pair_rows:
        issues: list[str] = []
        for side in ("a", "b"):
            path_text = row.get(f"clip_{side}_video_path_local", "")
            indices = parse_indices(row.get(f"clip_{side}_sample_frame_indices", ""))
            if path_text in bad_videos:
                issues.append(f"clip_{side}_video_bad")
            frame_count = video_frame_counts.get(path_text, 0)
            if indices and frame_count and max(indices) >= frame_count:
                issues.append(f"clip_{side}_sample_index_out_of_range")
        if issues:
            bad_pair_ids.add(row["pair_id"])
            pair_issue_rows.append(
                {
                    "split": row["_audit_split"],
                    "pair_id": row["pair_id"],
                    "task_id": row.get("task_id", ""),
                    "pair_type": row.get("pair_type", ""),
                    "issues": ";".join(issues),
                    "clip_a_video_path_local": row.get("clip_a_video_path_local", ""),
                    "clip_b_video_path_local": row.get("clip_b_video_path_local", ""),
                }
            )

    by_split = Counter(row["_audit_split"] for row in pair_rows)
    bad_by_split = Counter(row["split"] for row in pair_issue_rows)
    summary = {
        "split_dir": str(split_dir),
        "pair_rows": len(pair_rows),
        "unique_videos": len(video_rows),
        "bad_videos": sum(1 for row in video_rows if row["status"] != "ok"),
        "bad_pairs": len(bad_pair_ids),
        "by_split": dict(by_split),
        "bad_pairs_by_split": dict(bad_by_split),
        "video_status_counts": dict(Counter(row["status"] for row in video_rows)),
        "frame_count_min": min((int(row["frame_count"]) for row in video_rows), default=0),
        "frame_count_max": max((int(row["frame_count"]) for row in video_rows), default=0),
    }
    return video_rows, pair_issue_rows, summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Reward Split Input Audit",
        "",
        f"- split dir: `{summary['split_dir']}`",
        f"- pair rows: {summary['pair_rows']}",
        f"- unique videos: {summary['unique_videos']}",
        f"- bad videos: {summary['bad_videos']}",
        f"- bad pairs: {summary['bad_pairs']}",
        f"- frame count range: {summary['frame_count_min']} - {summary['frame_count_max']}",
        "",
        "## By Split",
        "",
        "| split | rows | bad pairs |",
        "| --- | ---: | ---: |",
    ]
    for split in SPLITS:
        lines.append(
            f"| {split} | {summary['by_split'].get(split, 0)} | "
            f"{summary['bad_pairs_by_split'].get(split, 0)} |"
        )
    lines += [
        "",
        "## Video Status",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(summary["video_status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit videos referenced by reward model split CSVs.")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.split_dir / "input_audit_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_rows, pair_issue_rows, summary = audit_split(args.split_dir)
    write_csv(out_dir / "video_audit.csv", video_rows)
    write_csv(out_dir / "pair_input_issues.csv", pair_issue_rows)
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "audit_report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
