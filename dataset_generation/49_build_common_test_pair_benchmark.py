"""Build a blinded, position-balanced benchmark from StackCube v2 test pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VALID_LABELS = {"A>B", "B>A", "unsure"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def invert(label: str) -> str:
    if label == "A>B":
        return "B>A"
    if label == "B>A":
        return "A>B"
    return label


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_indices(value: str) -> list[int]:
    indices = [int(item) for item in value.split(";") if item.strip()]
    if len(indices) != 6 or indices != sorted(indices):
        raise ValueError(f"expected six ordered frame indices, got {value!r}")
    return indices


def read_sampled_frames(path: Path, indices: list[int]) -> list[np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"cannot decode frame {index} from {path}")
            frames.append(frame)
    finally:
        capture.release()
    return frames


def panel(frame: np.ndarray, label: str) -> np.ndarray:
    output = cv2.resize(frame, (192, 192), interpolation=cv2.INTER_AREA)
    cv2.rectangle(output, (0, 0), (68, 25), (20, 20, 20), -1)
    cv2.putText(
        output,
        label,
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def write_contact_sheet(
    path: Path, a_frames: list[np.ndarray], b_frames: list[np.ndarray]
) -> None:
    a_row = np.concatenate(
        [panel(frame, f"A{index}") for index, frame in enumerate(a_frames, 1)], axis=1
    )
    b_row = np.concatenate(
        [panel(frame, f"B{index}") for index, frame in enumerate(b_frames, 1)], axis=1
    )
    separator = np.full((10, a_row.shape[1], 3), 245, dtype=np.uint8)
    sheet = np.concatenate([a_row, separator, b_row], axis=0)
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise RuntimeError(f"cannot encode contact sheet: {path}")
    encoded.tofile(str(path))


def oriented_physical(
    score: dict[str, str], swapped: bool
) -> dict[str, Any]:
    label = score["phys_preference_v2"]
    probability_a = float(score["phys_probability_a_better"])
    signed_margin = float(score["phys_signed_margin_v2"])
    source_a = "b" if swapped else "a"
    source_b = "a" if swapped else "b"
    return {
        "physical_label": invert(label) if swapped else label,
        "physical_basis": score["phys_preference_basis_v2"],
        "physical_probability_a_better": round(
            1.0 - probability_a if swapped else probability_a, 6
        ),
        "physical_pair_confidence": score["phys_pair_confidence"],
        "physical_signed_margin": round(-signed_margin if swapped else signed_margin, 6),
        "physical_a_stage": score[f"phys_{source_a}_stage"],
        "physical_b_stage": score[f"phys_{source_b}_stage"],
        "physical_a_expected_stage": score[f"phys_{source_a}_stage_expected"],
        "physical_b_expected_stage": score[f"phys_{source_b}_stage_expected"],
        "physical_a_potential": score[f"phys_{source_a}_potential"],
        "physical_b_potential": score[f"phys_{source_b}_potential"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--physical-scores", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8707)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"output folder is not empty: {args.out_dir}")

    all_pairs = read_csv(args.pairs)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in all_pairs:
        groups[row["split_v2"]].add(row["source_group_id"])
    overlap = groups["test"] & (groups["train"] | groups["val"])
    if overlap:
        raise RuntimeError(f"test source groups overlap train/val: {sorted(overlap)}")

    test_rows = [row for row in all_pairs if row["split_v2"] == "test"]
    if not test_rows:
        raise RuntimeError("no test pairs found")
    if any(row["reference_label_v2"] not in VALID_LABELS for row in test_rows):
        raise RuntimeError("unexpected reference label in test rows")

    score_by_id = {row["pair_id"]: row for row in read_csv(args.physical_scores)}
    missing_scores = sorted({row["pair_id"] for row in test_rows} - score_by_id.keys())
    if missing_scores:
        raise RuntimeError(f"missing physical scores for {len(missing_scores)} test pairs")

    rng = random.Random(args.seed)
    rng.shuffle(test_rows)
    decisive_index = 0
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    public_dir = args.out_dir / "public_qwen"
    media_dir = public_dir / "contact_sheets"
    private_dir = args.out_dir / "private"
    media_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(test_rows, start=1):
        original_reference = row["reference_label_v2"]
        if original_reference in {"A>B", "B>A"}:
            desired = "A>B" if decisive_index % 2 == 0 else "B>A"
            decisive_index += 1
            swapped = desired != original_reference
        else:
            swapped = bool(rng.getrandbits(1))

        source_a = "b" if swapped else "a"
        source_b = "a" if swapped else "b"
        a_frames = read_sampled_frames(
            Path(row[f"clip_{source_a}_video_path_windows"]),
            parse_indices(row[f"clip_{source_a}_sample_frame_indices"]),
        )
        b_frames = read_sampled_frames(
            Path(row[f"clip_{source_b}_video_path_windows"]),
            parse_indices(row[f"clip_{source_b}_sample_frame_indices"]),
        )

        blind_id = f"COMMON-TEST-{index:03d}"
        image_name = f"{blind_id}.jpg"
        write_contact_sheet(media_dir / image_name, a_frames, b_frames)
        public_rows.append(
            {
                "pair_id": blind_id,
                "task_id": "stackcube",
                "task_goal_text": row["task_goal_text"],
                "contact_sheet_file": f"contact_sheets/{image_name}",
                "clip_a_sampled_frames": 6,
                "clip_b_sampled_frames": 6,
            }
        )

        blind_reference = invert(original_reference) if swapped else original_reference
        physical = oriented_physical(score_by_id[row["pair_id"]], swapped)
        private_rows.append(
            {
                "blind_pair_id": blind_id,
                "original_pair_id": row["pair_id"],
                "swapped": int(swapped),
                "source_group_id": row["source_group_id"],
                "pair_type": row["pair_type"],
                "blind_reference_label": blind_reference,
                "reference_basis": row["reference_basis_v2"],
                **physical,
            }
        )

    write_csv(public_dir / "blind_manifest.csv", public_rows)
    write_csv(private_dir / "evaluation_key.csv", private_rows)
    template = {
        "model": "qwen3.7-plus",
        "labels": [
            {
                "pair_id": row["pair_id"],
                "preference": "",
                "confidence": 0.0,
                "reason": "",
                "visible_uncertainty": "",
            }
            for row in public_rows
        ],
    }
    (public_dir / "expected_output_template.json").write_text(
        json.dumps(template, indent=2) + "\n", encoding="utf-8"
    )
    (public_dir / "README_FIRST.txt").write_text(
        "This is a blind RGB-only StackCube preference benchmark.\n"
        "Do not add files from the sibling private directory to an annotation context.\n"
        "Each contact sheet contains six chronological frames for A on the top row and "
        "six for B on the bottom row.\n",
        encoding="utf-8",
    )

    checksums = [
        {
            "file": path.relative_to(public_dir).as_posix(),
            "sha256": sha256(path),
        }
        for path in sorted(public_dir.rglob("*"))
        if path.is_file()
    ]
    write_csv(public_dir / "CHECKSUMS.csv", checksums)

    summary = {
        "schema_version": "common_stackcube_test_pairs_v1",
        "seed": args.seed,
        "pair_count": len(private_rows),
        "source_groups": sorted({row["source_group_id"] for row in private_rows}),
        "group_disjoint_from_train_val": True,
        "test_set_status": "training-held-out but previously inspected development holdout",
        "blind_reference_distribution": dict(
            Counter(row["blind_reference_label"] for row in private_rows)
        ),
        "pair_type_distribution": dict(Counter(row["pair_type"] for row in private_rows)),
        "reference_provenance": "offline simulator stage/progress proxy",
        "qwen_visible_inputs": "RGB contact sheet and task text only",
        "physical_visible_inputs": "frozen stereo-depth/contact progress branch",
    }
    (args.out_dir / "benchmark_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
