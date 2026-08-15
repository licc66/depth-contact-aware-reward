"""Build a leakage-free 12-pair GPT-5.5 light video-annotation pilot package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SELECTION_PLAN = {
    "intra_success_temporal_gap": {"B>A": 2, "unsure": 1},
    "near_miss_vs_early_truncated": {"A>B": 2, "B>A": 1},
    "success_vs_offset_hard_negative": {"A>B": 3},
    "success_vs_truncated_terminal": {"A>B": 3},
}
VALID_LABELS = {"A>B", "B>A", "unsure"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def invert(label: str) -> str:
    return "B>A" if label == "A>B" else "A>B" if label == "B>A" else label


def select_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    train = [row for row in rows if row.get("split_v2") == "train"]
    selected: list[dict[str, str]] = []
    used_pair_ids: set[str] = set()
    for pair_type, label_counts in SELECTION_PLAN.items():
        for label, count in label_counts.items():
            candidates = [
                row
                for row in train
                if row.get("pair_type") == pair_type
                and row.get("reference_label_v2") == label
                and row.get("pair_id") not in used_pair_ids
            ]
            rng.shuffle(candidates)
            if len(candidates) < count:
                raise RuntimeError(
                    f"not enough {pair_type}/{label} rows: {len(candidates)} < {count}"
                )
            chosen = candidates[:count]
            selected.extend(chosen)
            used_pair_ids.update(row["pair_id"] for row in chosen)
    rng.shuffle(selected)
    if len(selected) != 12:
        raise AssertionError(f"selection plan produced {len(selected)} rows")
    return selected


def clip_fields(row: dict[str, str], side: str) -> tuple[Path, int, int]:
    path = Path(row[f"clip_{side}_video_path_windows"])
    start = int(float(row[f"clip_{side}_start_frame"]))
    end = int(float(row[f"clip_{side}_end_frame_exclusive"]))
    if not path.exists():
        raise FileNotFoundError(path)
    if end <= start:
        raise ValueError(f"invalid frame window {start}:{end} for {path}")
    return path, start, end


def extract_frames(path: Path, start: int, end: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[np.ndarray] = []
    for _ in range(start, end):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != end - start:
        raise RuntimeError(
            f"decoded {len(frames)} frames, expected {end - start}: {path}"
        )
    return frames


def write_video(path: Path, frames: list[np.ndarray], fps: float = 8.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def sampled_panels(frames: list[np.ndarray], label: str, count: int = 6) -> list[np.ndarray]:
    indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
    panels = []
    for order, index in enumerate(indices, start=1):
        panel = cv2.resize(frames[int(index)], (192, 192), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (68, 24), (20, 20, 20), -1)
        cv2.putText(
            panel,
            f"{label}{order}",
            (7, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    return panels


def write_contact_sheet(path: Path, a_frames: list[np.ndarray], b_frames: list[np.ndarray]) -> None:
    a_row = np.concatenate(sampled_panels(a_frames, "A"), axis=1)
    b_row = np.concatenate(sampled_panels(b_frames, "B"), axis=1)
    separator = np.full((10, a_row.shape[1], 3), 245, dtype=np.uint8)
    sheet = np.concatenate([a_row, separator, b_row], axis=0)
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise RuntimeError(f"cannot write contact sheet {path}")
    encoded.tofile(str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--private-mapping", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5202)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"output folder is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = args.out_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    rows = select_rows(read_csv(args.pairs), args.seed)

    non_unsure = [row for row in rows if row["reference_label_v2"] != "unsure"]
    desired_by_pair: dict[str, str] = {}
    for index, row in enumerate(non_unsure):
        desired_by_pair[row["pair_id"]] = "A>B" if index % 2 == 0 else "B>A"

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    output_template = {"labels": []}
    for index, row in enumerate(rows, start=1):
        blind_id = f"PILOT-{index:03d}"
        original_label = row["reference_label_v2"]
        desired = desired_by_pair.get(row["pair_id"], "unsure")
        swapped = original_label != "unsure" and desired != original_label
        source_a = "b" if swapped else "a"
        source_b = "a" if swapped else "b"
        a_path, a_start, a_end = clip_fields(row, source_a)
        b_path, b_start, b_end = clip_fields(row, source_b)
        a_frames = extract_frames(a_path, a_start, a_end)
        b_frames = extract_frames(b_path, b_start, b_end)

        a_name = f"{blind_id}_A.mp4"
        b_name = f"{blind_id}_B.mp4"
        sheet_name = f"{blind_id}_contact_sheet.jpg"
        write_video(media_dir / a_name, a_frames)
        write_video(media_dir / b_name, b_frames)
        write_contact_sheet(media_dir / sheet_name, a_frames, b_frames)

        public_rows.append(
            {
                "pair_id": blind_id,
                "task_id": "stackcube",
                "task_goal_text": row["task_goal_text"],
                "clip_a_file": f"media/{a_name}",
                "clip_b_file": f"media/{b_name}",
                "contact_sheet_file": f"media/{sheet_name}",
                "clip_a_frames": len(a_frames),
                "clip_b_frames": len(b_frames),
            }
        )
        blind_reference = invert(original_label) if swapped else original_label
        private_rows.append(
            {
                "blind_pair_id": blind_id,
                "original_pair_id": row["pair_id"],
                "swapped": int(swapped),
                "pair_type": row["pair_type"],
                "original_reference_label_v2": original_label,
                "blind_reference_label_v2": blind_reference,
                "source_group_id": row["source_group_id"],
            }
        )
        output_template["labels"].append(
            {
                "pair_id": blind_id,
                "preference": "",
                "confidence": 0.0,
                "clip_a_stage": "",
                "clip_b_stage": "",
                "reason": "",
                "visible_failure_or_uncertainty": "",
            }
        )

    write_csv(args.out_dir / "blind_manifest.csv", public_rows)
    write_csv(args.private_mapping, private_rows)
    (args.out_dir / "expected_output_template.json").write_text(
        json.dumps(output_template, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(args.prompt, args.out_dir / "GPT55_LIGHT_BLIND_VIDEO_LABELING_PROMPT.txt")
    readme = """GPT-5.5 light blind video-labeling pilot

1. Start a new project/conversation rooted at this folder.
2. Use GPT-5.5 with light reasoning.
3. Paste the full prompt file as the first message.
4. Ask the model to read blind_manifest.csv and inspect every contact sheet/video.
5. Save the strict JSON response as gpt55_light_labels_batch_001.json in this folder.

The folder intentionally contains no candidate, physical, legacy, or reference labels.
Do not add the original stackcube_pairs_v2.csv to this project.
"""
    (args.out_dir / "README_FIRST.txt").write_text(readme, encoding="utf-8")

    checksum_rows = []
    for path in sorted(args.out_dir.rglob("*")):
        if path.is_file():
            checksum_rows.append(
                {"file": path.relative_to(args.out_dir).as_posix(), "sha256": sha256(path)}
            )
    write_csv(args.out_dir / "CHECKSUMS.csv", checksum_rows)
    summary = {
        "schema_version": "gpt55_light_blind_pilot_v1",
        "pair_count": len(public_rows),
        "media_file_count": len(list(media_dir.iterdir())),
        "blind_reference_distribution_private": dict(
            Counter(row["blind_reference_label_v2"] for row in private_rows)
        ),
        "private_mapping": str(args.private_mapping),
        "public_output": str(args.out_dir),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
