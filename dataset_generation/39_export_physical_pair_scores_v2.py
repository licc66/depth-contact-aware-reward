"""Export frozen physical-progress v2 scores for sensor-aligned A/B pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from physical_progress_branch_v2 import PhysicalProgressRuntimeV2  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def observed(value: Any) -> bool:
    if value is None or str(value).strip() == "":
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return isinstance(value, bool)


def number(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grouped_sensor_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[row["meta_sample_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: integer(row["meta_saved_frame_index"]))
    return dict(grouped)


def validity_ratio(frames: list[dict[str, Any]], names: list[str]) -> float:
    if not frames or not names:
        return 0.0
    valid = sum(observed(frame.get(name)) for frame in frames for name in names)
    return valid / (len(frames) * len(names))


def completion_guard(frame: dict[str, Any]) -> str:
    support = number(frame.get("object_support_contacts"), 0.0) > 0.5
    released = number(frame.get("released_object"), 0.0) > 0.5
    distance = number(frame.get("object_goal_3d_dist_m"))
    if not support or not released:
        return "fail"
    if math.isfinite(distance):
        return "pass" if distance <= 0.045 else "fail"
    return "unknown_depth"


def preference_basis(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    stage_margin: float,
    progress_epsilon: float,
) -> tuple[str, str, float]:
    stage_axis = np.arange(1, 5, dtype=np.float64)
    expected_a = float(np.dot(stage_axis, result_a["stage_probabilities"]))
    expected_b = float(np.dot(stage_axis, result_b["stage_probabilities"]))
    stage_delta = expected_a - expected_b
    potential_delta = result_a["potential"] - result_b["potential"]
    if abs(stage_delta) >= stage_margin:
        return ("A>B" if stage_delta > 0 else "B>A"), "stage", stage_delta / 3.0
    if abs(potential_delta) >= progress_epsilon:
        return (
            ("A>B" if potential_delta > 0 else "B>A"),
            "progress",
            potential_delta,
        )
    return "unsure", "none", 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--sensor-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--stage-margin", type=float, default=0.25)
    parser.add_argument("--progress-epsilon", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = read_csv(args.pairs)
    sensors = grouped_sensor_rows(args.sensor_dataset / "sensor_features.csv")
    runtime = PhysicalProgressRuntimeV2.from_checkpoint(args.checkpoint, args.device)

    clip_cache: dict[str, dict[str, Any]] = {}

    def score_side(row: dict[str, str], side: str) -> dict[str, Any]:
        clip_id = row[f"clip_{side}_id"]
        if clip_id in clip_cache:
            return clip_cache[clip_id]
        sample_id = row[f"clip_{side}_sample_id"]
        end = integer(row[f"clip_{side}_end_frame_exclusive"])
        frames = sensors.get(sample_id, [])[:end]
        if not frames or len(frames) != end:
            raise RuntimeError(
                f"pair {row['pair_id']} side {side} expects {end} frames, got {len(frames)}"
            )
        result = runtime.score("stackcube", frames, include_embedding=True)
        selected = runtime._resample(frames)
        result["depth_validity_ratio"] = validity_ratio(
            selected, runtime.depth_feature_names
        )
        result["contact_validity_ratio"] = validity_ratio(
            selected, runtime.contact_feature_names
        )
        result["completion_guard"] = completion_guard(frames[-1])
        result["sample_id"] = sample_id
        result["end_frame_exclusive"] = end
        clip_cache[clip_id] = result
        return result

    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        a = score_side(pair, "a")
        b = score_side(pair, "b")
        stated, basis, signed_margin = preference_basis(
            a, b, args.stage_margin, args.progress_epsilon
        )
        min_confidence = min(a["confidence"], b["confidence"])
        min_validity = min(
            a["depth_validity_ratio"],
            b["depth_validity_ratio"],
            a["contact_validity_ratio"],
            b["contact_validity_ratio"],
        )
        raw_confidence = min(1.0, abs(signed_margin)) * min_confidence
        raw_confidence *= math.sqrt(max(0.0, min_validity))
        probability_a = (
            1.0 / (1.0 + math.exp(-signed_margin / 0.10))
            if stated != "unsure"
            else 0.5
        )
        record: dict[str, Any] = {
            "pair_id": pair["pair_id"],
            "task_id": pair.get("task_id", "stackcube"),
            "split_v2": pair.get("split_v2", pair.get("split", "")),
            "source_group_id": pair.get("source_group_id", ""),
            "pair_type": pair.get("pair_type", ""),
            "clip_a_id": pair["clip_a_id"],
            "clip_b_id": pair["clip_b_id"],
            "phys_preference_v2": stated,
            "phys_preference_basis_v2": basis,
            "phys_probability_a_better": round(probability_a, 6),
            "phys_pair_confidence": round(raw_confidence, 6),
            "phys_signed_margin_v2": round(signed_margin, 6),
        }
        for side, result in (("a", a), ("b", b)):
            expected_stage = sum(
                (stage + 1) * probability
                for stage, probability in enumerate(result["stage_probabilities"])
            )
            record.update(
                {
                    f"phys_{side}_stage": result["stage"],
                    f"phys_{side}_stage_expected": round(expected_stage, 6),
                    f"phys_{side}_local_progress": round(result["local_progress"], 6),
                    f"phys_{side}_potential": round(result["potential"], 6),
                    f"phys_{side}_confidence": round(result["confidence"], 6),
                    f"phys_{side}_depth_gate": round(result["depth_gate"], 6),
                    f"phys_{side}_contact_gate": round(result["contact_gate"], 6),
                    f"phys_{side}_depth_validity_ratio": round(
                        result["depth_validity_ratio"], 6
                    ),
                    f"phys_{side}_contact_validity_ratio": round(
                        result["contact_validity_ratio"], 6
                    ),
                    f"phys_{side}_completion_guard": result["completion_guard"],
                    f"phys_{side}_source_frame_count": result["source_frame_count"],
                    f"phys_{side}_model_frame_count": result["model_frame_count"],
                }
            )
            for stage, probability in enumerate(result["stage_probabilities"], start=1):
                record[f"phys_{side}_stage_p{stage}"] = round(probability, 7)
        rows.append(record)
        if index % 100 == 0 or index == len(pairs):
            print(f"scored {index}/{len(pairs)} pairs; unique_clips={len(clip_cache)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.out_dir / "physical_pair_scores_v2.csv"
    write_csv(score_path, rows)
    clip_ids = sorted(clip_cache)
    embeddings = np.asarray(
        [clip_cache[clip_id]["embedding"] for clip_id in clip_ids], dtype=np.float32
    )
    np.savez_compressed(
        args.out_dir / "physical_clip_embeddings_v2.npz",
        clip_ids=np.asarray(clip_ids),
        embeddings=embeddings,
    )
    summary = {
        "schema_version": "physical_pair_scores_v2.0",
        "pairs": len(rows),
        "unique_clips": len(clip_ids),
        "embedding_shape": list(embeddings.shape),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_depth_features": runtime.depth_feature_names,
        "checkpoint_contact_features": runtime.contact_feature_names,
        "preference_counts": dict(Counter(row["phys_preference_v2"] for row in rows)),
        "basis_counts": dict(Counter(row["phys_preference_basis_v2"] for row in rows)),
        "completion_guard_counts": {
            side: dict(Counter(row[f"phys_{side}_completion_guard"] for row in rows))
            for side in ("a", "b")
        },
        "online_privileged_inputs": False,
    }
    write_json(args.out_dir / "physical_pair_scores_v2_manifest.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
