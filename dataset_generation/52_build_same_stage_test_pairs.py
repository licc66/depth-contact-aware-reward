"""Build a deterministic same-stage StackCube hard-pair test table.

The builder leaves the existing common benchmark untouched. It creates new
pairs from held-out trajectories using three constructions:

1. Different prefix endpoints from the same successful video.
2. Terminal prefixes from different spatial-offset perturbations.
3. Successful prefixes compared with offset-perturbation prefixes.

All decisive pairs have the same endpoint stage and differ only in the
stage-local progress target. A small near-tie set tests abstention behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VALID_LABELS = {"A>B", "B>A", "unsure"}
TASK_GOAL = (
    "A robot arm stacks the movable red cube on the green support cube, "
    "releases it, and leaves it stably supported."
)


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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def physical_target(row: dict[str, str]) -> tuple[int, float, float]:
    """Match the offline StackCube target used by trainer 35."""

    stage_zero = max(1, min(4, integer(row.get("gt_stage_candidate"), 1))) - 1
    if stage_zero == 0:
        distance = number(row.get("gt_tcp_object_dist_m"), 0.35)
        local = max(0.0, min(1.0, (0.35 - distance) / 0.325))
    elif stage_zero == 1:
        distance = number(row.get("gt_object_goal_3d_dist_m"), 0.30)
        local = max(0.0, min(1.0, 1.0 - distance / 0.30))
    elif stage_zero == 2:
        distance = number(row.get("gt_object_goal_3d_dist_m"), 0.08)
        spatial = max(0.0, min(1.0, 1.0 - distance / 0.08))
        on_support = number(row.get("gt_is_cubeA_on_cubeB"), 0.0)
        static = number(row.get("gt_is_cubeA_static"), 0.0)
        released = 1.0 - number(row.get("gt_is_cubeA_grasped"), 0.0)
        local = max(
            0.0,
            min(
                1.0,
                0.45 * spatial
                + 0.30 * on_support
                + 0.15 * static
                + 0.10 * released,
            ),
        )
    else:
        local = 1.0
    return stage_zero + 1, local, (stage_zero + local) / 4.0


def sampled_indices(end_frame_exclusive: int, count: int = 6) -> list[int]:
    if end_frame_exclusive < count:
        raise ValueError(
            f"prefix has {end_frame_exclusive} frames but {count} are required"
        )
    last = end_frame_exclusive - 1
    result = [round(index * last / (count - 1)) for index in range(count)]
    if len(set(result)) != count:
        raise ValueError(f"sampled indices are not unique: {result}")
    return result


@dataclass(frozen=True)
class Endpoint:
    sample_id: str
    source_group_id: str
    split: str
    video_path: Path
    end: int
    stage: int
    local_progress: float
    potential: float
    variant: str

    @property
    def key(self) -> tuple[str, int]:
        return self.sample_id, self.end


@dataclass(frozen=True)
class Candidate:
    pair_type: str
    first: Endpoint
    second: Endpoint
    reference_label: str
    reference_basis: str

    @property
    def progress_delta(self) -> float:
        return abs(self.first.potential - self.second.potential)

    @property
    def key(self) -> tuple[tuple[str, int], tuple[str, int]]:
        return tuple(sorted((self.first.key, self.second.key)))  # type: ignore[return-value]


def classify_variant(sample_id: str, source_group_id: str) -> str:
    if sample_id == source_group_id:
        return "base_success"
    if "-OFFSET-" in sample_id:
        return "offset"
    if "-TRUNC-" in sample_id:
        return "truncated"
    return "other"


def label_for(first: Endpoint, second: Endpoint, epsilon: float) -> str:
    delta = first.potential - second.potential
    if delta >= epsilon:
        return "A>B"
    if delta <= -epsilon:
        return "B>A"
    return "unsure"


def delta_bin(delta: float, epsilon: float) -> str:
    if delta < epsilon:
        return "near_tie"
    if delta < 0.02:
        return "fine_0.01_to_0.02"
    if delta < 0.05:
        return "medium_0.02_to_0.05"
    return "coarse_ge_0.05"


def pair_candidates(
    endpoints: Iterable[Endpoint],
    *,
    pair_type: str,
    epsilon: float,
    min_frame_gap: int = 0,
    same_sample: bool | None = None,
    ties_only: bool = False,
) -> list[Candidate]:
    values = list(endpoints)
    output: list[Candidate] = []
    for left_index, first in enumerate(values):
        for second in values[left_index + 1 :]:
            if first.source_group_id != second.source_group_id:
                continue
            if first.stage != second.stage or first.stage == 4:
                continue
            if same_sample is True and first.sample_id != second.sample_id:
                continue
            if same_sample is False and first.sample_id == second.sample_id:
                continue
            if first.sample_id == second.sample_id and abs(first.end - second.end) < min_frame_gap:
                continue
            label = label_for(first, second, epsilon)
            if ties_only and label != "unsure":
                continue
            if not ties_only and label == "unsure":
                continue
            output.append(
                Candidate(
                    pair_type=pair_type,
                    first=first,
                    second=second,
                    reference_label=label,
                    reference_basis="none" if label == "unsure" else "progress",
                )
            )
    return output


def select_diverse(
    candidates: list[Candidate],
    limit: int,
    rng: random.Random,
    max_endpoint_reuse: int,
) -> list[Candidate]:
    if limit <= 0:
        return []
    bins: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        bins[delta_bin(candidate.progress_delta, 0.01)].append(candidate)
    for values in bins.values():
        rng.shuffle(values)

    ordered_bins = [
        "fine_0.01_to_0.02",
        "medium_0.02_to_0.05",
        "coarse_ge_0.05",
        "near_tie",
    ]
    selected: list[Candidate] = []
    endpoint_use: Counter[tuple[str, int]] = Counter()
    seen: set[tuple[tuple[str, int], tuple[str, int]]] = set()
    while len(selected) < limit:
        added = False
        for name in ordered_bins:
            values = bins.get(name, [])
            while values:
                candidate = values.pop()
                if candidate.key in seen:
                    continue
                if any(endpoint_use[key] >= max_endpoint_reuse for key in candidate.key):
                    continue
                selected.append(candidate)
                seen.add(candidate.key)
                endpoint_use.update(candidate.key)
                added = True
                break
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


def orient_candidate(
    candidate: Candidate, desired_label: str
) -> tuple[Endpoint, Endpoint, str]:
    if candidate.reference_label == "unsure":
        return candidate.first, candidate.second, "unsure"
    if candidate.reference_label == desired_label:
        return candidate.first, candidate.second, candidate.reference_label
    return candidate.second, candidate.first, desired_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--sensor-dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=8808)
    parser.add_argument("--min-prefix-frames", type=int, default=12)
    parser.add_argument("--endpoint-stride", type=int, default=4)
    parser.add_argument("--min-frame-gap", type=int, default=8)
    parser.add_argument("--progress-epsilon", type=float, default=0.01)
    parser.add_argument("--tie-max-delta", type=float, default=0.003)
    parser.add_argument("--temporal-per-stage", type=int, default=24)
    parser.add_argument("--offset-pairs", type=int, default=16)
    parser.add_argument("--success-offset-pairs", type=int, default=24)
    parser.add_argument("--tie-pairs", type=int, default=8)
    parser.add_argument("--max-endpoint-reuse", type=int, default=5)
    parser.add_argument(
        "--require-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"output folder is not empty: {args.out_dir}")
    if not 0.0 < args.tie_max_delta < args.progress_epsilon:
        raise ValueError("tie-max-delta must be positive and below progress-epsilon")

    supervision_path = args.sensor_dataset / "offline_supervision.csv"
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(supervision_path):
        if row.get("meta_split") == args.split:
            grouped[row["meta_sample_id"]].append(row)
    if not grouped:
        raise RuntimeError(f"no rows found for split {args.split!r}")
    for rows in grouped.values():
        rows.sort(key=lambda row: integer(row["meta_saved_frame_index"]))

    endpoints_by_group: dict[str, list[Endpoint]] = defaultdict(list)
    base_endpoints_by_group: dict[str, list[Endpoint]] = defaultdict(list)
    offset_terminals_by_group: dict[str, list[Endpoint]] = defaultdict(list)
    for sample_id, rows in grouped.items():
        first = rows[0]
        source_group = first["meta_source_group_id"]
        split = first["meta_split"]
        variant = classify_variant(sample_id, source_group)
        video_path = (
            args.sensor_dataset / "samples" / split / sample_id / "left_stereo.mp4"
        )
        if args.require_videos and not video_path.is_file():
            raise FileNotFoundError(video_path)

        if variant == "base_success":
            ends = list(
                range(
                    args.min_prefix_frames,
                    len(rows) + 1,
                    args.endpoint_stride,
                )
            )
            if len(rows) not in ends:
                ends.append(len(rows))
            for end in sorted(set(ends)):
                stage, local, potential = physical_target(rows[end - 1])
                endpoint = Endpoint(
                    sample_id=sample_id,
                    source_group_id=source_group,
                    split=split,
                    video_path=video_path,
                    end=end,
                    stage=stage,
                    local_progress=local,
                    potential=potential,
                    variant=variant,
                )
                endpoints_by_group[source_group].append(endpoint)
                base_endpoints_by_group[source_group].append(endpoint)
        elif variant == "offset":
            stage, local, potential = physical_target(rows[-1])
            endpoint = Endpoint(
                sample_id=sample_id,
                source_group_id=source_group,
                split=split,
                video_path=video_path,
                end=len(rows),
                stage=stage,
                local_progress=local,
                potential=potential,
                variant=variant,
            )
            endpoints_by_group[source_group].append(endpoint)
            offset_terminals_by_group[source_group].append(endpoint)

    rng = random.Random(args.seed)
    selected: list[Candidate] = []
    availability: dict[str, Any] = {}
    for source_group in sorted(endpoints_by_group):
        base = base_endpoints_by_group.get(source_group, [])
        offsets = offset_terminals_by_group.get(source_group, [])
        temporal = pair_candidates(
            base,
            pair_type="same_video_same_stage_prefix",
            epsilon=args.progress_epsilon,
            min_frame_gap=args.min_frame_gap,
            same_sample=True,
        )
        temporal_selected: list[Candidate] = []
        for stage in range(1, 4):
            stage_candidates = [
                candidate
                for candidate in temporal
                if candidate.first.stage == stage
            ]
            temporal_selected.extend(
                select_diverse(
                    stage_candidates,
                    args.temporal_per_stage,
                    rng,
                    args.max_endpoint_reuse,
                )
            )

        offset = pair_candidates(
            offsets,
            pair_type="same_stage_offset_terminal",
            epsilon=args.progress_epsilon,
            same_sample=False,
        )
        offset_selected = select_diverse(
            offset,
            args.offset_pairs,
            rng,
            args.max_endpoint_reuse,
        )

        cross_pool: list[Candidate] = []
        for base_endpoint in base:
            for offset_endpoint in offsets:
                if base_endpoint.stage != offset_endpoint.stage or base_endpoint.stage == 4:
                    continue
                label = label_for(
                    base_endpoint,
                    offset_endpoint,
                    args.progress_epsilon,
                )
                if label == "unsure":
                    continue
                cross_pool.append(
                    Candidate(
                        pair_type="same_stage_success_prefix_vs_offset",
                        first=base_endpoint,
                        second=offset_endpoint,
                        reference_label=label,
                        reference_basis="progress",
                    )
                )
        cross_selected = select_diverse(
            cross_pool,
            args.success_offset_pairs,
            rng,
            args.max_endpoint_reuse,
        )

        tie_pool: list[Candidate] = []
        for first_index, first_endpoint in enumerate(offsets):
            for second_endpoint in offsets[first_index + 1 :]:
                if first_endpoint.stage != second_endpoint.stage:
                    continue
                if abs(first_endpoint.potential - second_endpoint.potential) > args.tie_max_delta:
                    continue
                tie_pool.append(
                    Candidate(
                        pair_type="same_stage_offset_near_tie",
                        first=first_endpoint,
                        second=second_endpoint,
                        reference_label="unsure",
                        reference_basis="none",
                    )
                )
        tie_selected = select_diverse(
            tie_pool,
            args.tie_pairs,
            rng,
            args.max_endpoint_reuse,
        )

        selected.extend(temporal_selected)
        selected.extend(offset_selected)
        selected.extend(cross_selected)
        selected.extend(tie_selected)
        availability[source_group] = {
            "base_endpoints": len(base),
            "offset_terminals": len(offsets),
            "candidate_counts": {
                "same_video_same_stage_prefix": len(temporal),
                "same_stage_offset_terminal": len(offset),
                "same_stage_success_prefix_vs_offset": len(cross_pool),
                "same_stage_offset_near_tie": len(tie_pool),
            },
            "selected_counts": dict(
                Counter(
                    candidate.pair_type
                    for candidate in (
                        temporal_selected
                        + offset_selected
                        + cross_selected
                        + tie_selected
                    )
                )
            ),
        }

    if not selected:
        raise RuntimeError("no same-stage candidates survived selection")

    rows: list[dict[str, Any]] = []
    decisive_index = 0
    for pair_index, candidate in enumerate(selected):
        desired = "A>B" if decisive_index % 2 == 0 else "B>A"
        if candidate.reference_label != "unsure":
            decisive_index += 1
        first, second, label = orient_candidate(candidate, desired)
        delta = abs(first.potential - second.potential)
        pair_id = f"SC-HARD-P{pair_index:05d}"

        def side_fields(side: str, endpoint: Endpoint) -> dict[str, Any]:
            indices = sampled_indices(endpoint.end)
            return {
                f"clip_{side}_id": (
                    f"{endpoint.sample_id}-PREFIX-{endpoint.end:04d}-HARDV1"
                ),
                f"clip_{side}_sample_id": endpoint.sample_id,
                f"clip_{side}_video_path_windows": str(endpoint.video_path),
                f"clip_{side}_start_frame": 0,
                f"clip_{side}_end_frame_exclusive": endpoint.end,
                f"clip_{side}_sample_frame_indices": ";".join(
                    str(index) for index in indices
                ),
                f"reference_{side}_stage_v2": endpoint.stage,
                f"reference_{side}_local_progress_v2": round(
                    endpoint.local_progress, 6
                ),
                f"reference_{side}_potential_v2": round(endpoint.potential, 6),
                f"clip_{side}_variant": endpoint.variant,
            }

        row: dict[str, Any] = {
            "pair_id": pair_id,
            "task_id": "stackcube",
            "split": first.split,
            "split_v2": first.split,
            "source_group_id": first.source_group_id,
            "task_goal_text": TASK_GOAL,
            "pair_type": candidate.pair_type,
            "candidate_label": label,
            "candidate_confidence": (
                "unsure"
                if label == "unsure"
                else "medium"
                if delta < 0.02
                else "high"
            ),
            "needs_vlm_preference_label": "true",
            "reference_label_v2": label,
            "reference_basis_v2": candidate.reference_basis,
            "reference_progress_delta_abs_v2": round(delta, 6),
            "difficulty_bin": delta_bin(delta, args.progress_epsilon),
            "pair_construction_version": "same_stage_hard_pairs_v1",
            "independence_unit": "source_group_id",
            "rationale": (
                "Same endpoint stage; compare stage-local physical progress from "
                f"{candidate.pair_type}."
            ),
            **side_fields("a", first),
            **side_fields("b", second),
        }
        rows.append(row)

    # Fail closed on leakage, cross-stage pairs, duplicate pairs, or bad labels.
    duplicate_keys: set[tuple[tuple[str, int], tuple[str, int]]] = set()
    seen_keys: set[tuple[tuple[str, int], tuple[str, int]]] = set()
    for row in rows:
        if row["split_v2"] != args.split:
            raise RuntimeError(f"non-{args.split} row leaked into output: {row['pair_id']}")
        if row["reference_a_stage_v2"] != row["reference_b_stage_v2"]:
            raise RuntimeError(f"cross-stage pair generated: {row['pair_id']}")
        if row["reference_label_v2"] not in VALID_LABELS:
            raise RuntimeError(f"invalid label: {row['pair_id']}")
        key = tuple(
            sorted(
                (
                    (
                        row["clip_a_sample_id"],
                        integer(row["clip_a_end_frame_exclusive"]),
                    ),
                    (
                        row["clip_b_sample_id"],
                        integer(row["clip_b_end_frame_exclusive"]),
                    ),
                )
            )
        )
        if key in seen_keys:
            duplicate_keys.add(key)
        seen_keys.add(key)
    if duplicate_keys:
        raise RuntimeError(f"generated {len(duplicate_keys)} duplicate unordered pairs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.out_dir / "same_stage_test_pairs_v1.csv"
    write_csv(pair_path, rows)
    summary = {
        "schema_version": "same_stage_stackcube_test_pairs_v1",
        "sensor_dataset": str(args.sensor_dataset),
        "split": args.split,
        "seed": args.seed,
        "pair_count": len(rows),
        "all_pairs_same_stage": True,
        "source_groups": sorted({row["source_group_id"] for row in rows}),
        "unique_samples": len(
            {
                row[f"clip_{side}_sample_id"]
                for row in rows
                for side in ("a", "b")
            }
        ),
        "unique_prefix_clips": len(
            {
                row[f"clip_{side}_id"]
                for row in rows
                for side in ("a", "b")
            }
        ),
        "pair_type_counts": dict(Counter(row["pair_type"] for row in rows)),
        "stage_counts": dict(
            Counter(str(row["reference_a_stage_v2"]) for row in rows)
        ),
        "label_counts": dict(Counter(row["reference_label_v2"] for row in rows)),
        "difficulty_counts": dict(Counter(row["difficulty_bin"] for row in rows)),
        "progress_delta": {
            "minimum": min(row["reference_progress_delta_abs_v2"] for row in rows),
            "median": sorted(
                row["reference_progress_delta_abs_v2"] for row in rows
            )[len(rows) // 2],
            "maximum": max(row["reference_progress_delta_abs_v2"] for row in rows),
        },
        "candidate_availability": availability,
        "audit": {
            "duplicate_unordered_pairs": 0,
            "cross_stage_pairs": 0,
            "non_test_pairs": 0,
            "missing_video_policy": "error" if args.require_videos else "allowed",
        },
        "limitations": [
            "More pair rows do not create more independent source trajectories.",
            "The current held-out split contains one source group, SC-SUCC-0007.",
            "Available data support same-stage hard pairs mainly for stages 1 and 2.",
            "Reference labels remain simulator-derived physical progress targets.",
        ],
    }
    write_json(args.out_dir / "same_stage_test_pairs_manifest.json", summary)
    report = [
        "# Same-Stage StackCube Hard-Pair Test Set",
        "",
        f"- pairs: {summary['pair_count']}",
        f"- pair types: {summary['pair_type_counts']}",
        f"- stages: {summary['stage_counts']}",
        f"- labels: {summary['label_counts']}",
        f"- difficulty: {summary['difficulty_counts']}",
        f"- unique samples: {summary['unique_samples']}",
        f"- unique prefix clips: {summary['unique_prefix_clips']}",
        "",
        "Every pair has the same endpoint stage. This increases conditional test",
        "coverage but does not increase the number of independent source groups.",
    ]
    (args.out_dir / "RESULTS.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
