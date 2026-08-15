"""Calibrate and fuse semantic/physical pair preferences for StackCube v2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reward_common_v1 import PavCalibrator  # noqa: E402


SPLITS = ("train", "val", "test")
VALID_LABELS = {"A>B", "B>A"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def invert(label: str) -> str:
    return "B>A" if label == "A>B" else "A>B" if label == "B>A" else "unsure"


def preferred_side(label: str) -> str:
    return "a" if label == "A>B" else "b" if label == "B>A" else ""


def discount_toward_half(probability: float, factor: float) -> float:
    return 0.5 + (probability - 0.5) * max(0.0, min(1.0, factor))


def validity_factor(min_validity: float, floor: float, full: float = 0.90) -> float:
    if full <= floor:
        return 1.0
    return max(0.0, min(1.0, (min_validity - floor) / (full - floor)))


def completion_consistency_factor(row: dict[str, str], label: str) -> tuple[float, str]:
    """Apply only a narrow terminal safety cap, not a hand-coded progress model."""

    side = preferred_side(label)
    if not side:
        return 1.0, "not_applicable"
    expected_stage = number(row.get(f"phys_{side}_stage_expected"), 1.0)
    guard = row.get(f"phys_{side}_completion_guard", "")
    if expected_stage < 3.5:
        return 1.0, "preterminal"
    if guard == "fail":
        return 0.0, "terminal_guard_fail"
    if guard == "unknown_depth":
        return 0.75, "terminal_guard_unknown_depth"
    return 1.0, "terminal_guard_pass"


class FusionPolicyV2:
    def __init__(
        self,
        single_threshold: float,
        conflict_gap: float,
        conflict_penalty: float = 0.5,
    ) -> None:
        self.single_threshold = float(single_threshold)
        self.conflict_gap = float(conflict_gap)
        self.conflict_penalty = float(conflict_penalty)

    @staticmethod
    def _weight(probability: float) -> float:
        return max(0.05, min(1.0, 2.0 * (probability - 0.5)))

    @staticmethod
    def _abstain(reason: str) -> dict[str, Any]:
        return {
            "label": "abstain",
            "weight": 0.0,
            "confidence": 0.0,
            "reason": reason,
            "use_for_loss": False,
        }

    def decide(
        self,
        semantic_label: str,
        semantic_p: float,
        physical_label: str,
        physical_p: float,
    ) -> dict[str, Any]:
        semantic_stated = semantic_label in VALID_LABELS
        physical_stated = physical_label in VALID_LABELS
        if not semantic_stated and not physical_stated:
            return self._abstain("both_unsure")
        if semantic_stated and physical_stated and semantic_label == physical_label:
            semantic_evidence = max(0.0, 2.0 * (semantic_p - 0.5))
            physical_evidence = max(0.0, 2.0 * (physical_p - 0.5))
            combined = 0.5 + 0.5 * (
                1.0 - (1.0 - semantic_evidence) * (1.0 - physical_evidence)
            )
            return {
                "label": semantic_label,
                "weight": self._weight(combined),
                "confidence": combined,
                "reason": "semantic_physical_agree",
                "use_for_loss": True,
            }
        if semantic_stated and physical_stated:
            if semantic_p >= physical_p:
                label, strong_p, weak_p, branch = (
                    semantic_label,
                    semantic_p,
                    physical_p,
                    "semantic",
                )
            else:
                label, strong_p, weak_p, branch = (
                    physical_label,
                    physical_p,
                    semantic_p,
                    "physical",
                )
            if strong_p >= self.single_threshold and strong_p - weak_p >= self.conflict_gap:
                return {
                    "label": label,
                    "weight": self._weight(strong_p) * self.conflict_penalty,
                    "confidence": strong_p,
                    "reason": f"conflict_resolved_{branch}",
                    "use_for_loss": True,
                }
            return self._abstain("semantic_physical_conflict")
        label = semantic_label if semantic_stated else physical_label
        probability = semantic_p if semantic_stated else physical_p
        branch = "semantic" if semantic_stated else "physical"
        if probability >= self.single_threshold:
            return {
                "label": label,
                "weight": self._weight(probability),
                "confidence": probability,
                "reason": f"{branch}_only",
                "use_for_loss": True,
            }
        return self._abstain(f"{branch}_below_threshold")


def fit_calibrator(rows: list[dict[str, Any]], score_key: str, label_key: str) -> PavCalibrator:
    scores: list[float] = []
    correct: list[float] = []
    for row in rows:
        label = row.get(label_key, "")
        reference = row.get("reference_label_v2", "")
        if label not in VALID_LABELS or reference not in VALID_LABELS:
            continue
        scores.append(number(row.get(score_key), 0.0))
        correct.append(float(label == reference))
    if not scores:
        raise RuntimeError(f"no train rows available to calibrate {label_key}")
    return PavCalibrator().fit(scores, correct)


def calibrated_probability(calibrator: PavCalibrator, score: float) -> float:
    return max(0.5, min(1.0, calibrator.predict([score])[0]))


def fuse_rows(
    rows: list[dict[str, Any]],
    semantic_calibrator: PavCalibrator | None,
    physical_calibrator: PavCalibrator,
    policy: FusionPolicyV2,
    validity_floor: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        semantic_label = row.get("semantic_preference_v2", "unsure")
        if semantic_label not in VALID_LABELS:
            semantic_label = "unsure"
        semantic_raw = number(row.get("semantic_confidence_v2"), 0.0)
        semantic_p = (
            calibrated_probability(semantic_calibrator, semantic_raw)
            if semantic_calibrator is not None and semantic_label in VALID_LABELS
            else 0.5
        )
        semantic_guard_factor, semantic_guard_reason = completion_consistency_factor(
            row, semantic_label
        )
        semantic_p = discount_toward_half(semantic_p, semantic_guard_factor)

        physical_label = row.get("phys_preference_v2", "unsure")
        if physical_label not in VALID_LABELS:
            physical_label = "unsure"
        physical_raw = number(row.get("phys_pair_confidence"), 0.0)
        physical_p = (
            calibrated_probability(physical_calibrator, physical_raw)
            if physical_label in VALID_LABELS
            else 0.5
        )
        min_validity = min(
            number(row.get("phys_a_depth_validity_ratio"), 0.0),
            number(row.get("phys_b_depth_validity_ratio"), 0.0),
            number(row.get("phys_a_contact_validity_ratio"), 0.0),
            number(row.get("phys_b_contact_validity_ratio"), 0.0),
        )
        physical_p = discount_toward_half(
            physical_p, validity_factor(min_validity, validity_floor)
        )
        physical_guard_factor, physical_guard_reason = completion_consistency_factor(
            row, physical_label
        )
        physical_p = discount_toward_half(physical_p, physical_guard_factor)

        decision = policy.decide(
            semantic_label, semantic_p, physical_label, physical_p
        )
        output.append(
            {
                "pair_id": row["pair_id"],
                "task_id": row.get("task_id", "stackcube"),
                "split_v2": row.get("split_v2", row.get("split", "")),
                "source_group_id": row.get("source_group_id", ""),
                "pair_type": row.get("pair_type", ""),
                "clip_a_id": row.get("clip_a_id", ""),
                "clip_b_id": row.get("clip_b_id", ""),
                "reference_label_v2": row.get("reference_label_v2", ""),
                "semantic_preference_v2": semantic_label,
                "semantic_raw_confidence_v2": semantic_raw,
                "semantic_calibrated_p_v2": round(semantic_p, 6),
                "semantic_completion_constraint_v2": semantic_guard_reason,
                "phys_preference_v2": physical_label,
                "phys_raw_pair_confidence_v2": physical_raw,
                "phys_calibrated_p_v2": round(physical_p, 6),
                "phys_validity_min_v2": round(min_validity, 6),
                "phys_completion_constraint_v2": physical_guard_reason,
                "fusion_label_v2": decision["label"],
                "fusion_weight_v2": round(decision["weight"], 6),
                "fusion_confidence_v2": round(decision["confidence"], 6),
                "fusion_reason_v2": decision["reason"],
                "use_for_preference_loss_v2": decision["use_for_loss"],
            }
        )
    return output


def accuracy_coverage(rows: list[dict[str, Any]]) -> tuple[float, float, int]:
    reference_rows = [row for row in rows if row["reference_label_v2"] in VALID_LABELS]
    labeled = [row for row in reference_rows if row["fusion_label_v2"] in VALID_LABELS]
    if not labeled:
        return 0.0, 0.0, 0
    correct = sum(row["fusion_label_v2"] == row["reference_label_v2"] for row in labeled)
    return correct / len(labeled), len(labeled) / max(1, len(reference_rows)), len(labeled)


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    pairs = read_csv(args.pairs)
    physical = {row["pair_id"]: row for row in read_csv(args.physical_scores)}
    semantic: dict[str, dict[str, str]] = {}
    if args.semantic_source == "sensor_aligned":
        if args.semantic_labels is None:
            raise ValueError("--semantic-labels is required for sensor_aligned mode")
        semantic = {row["pair_id"]: row for row in read_csv(args.semantic_labels)}
        missing = sorted({row["pair_id"] for row in pairs} - semantic.keys())
        if missing:
            raise RuntimeError(f"sensor-aligned semantic labels missing {len(missing)} pair ids")

    joined: list[dict[str, Any]] = []
    for pair in pairs:
        score = physical.get(pair["pair_id"])
        if score is None:
            raise RuntimeError(f"missing physical score for {pair['pair_id']}")
        row: dict[str, Any] = {**pair, **score}
        if args.semantic_source == "sensor_aligned":
            label = semantic[pair["pair_id"]]
            row["semantic_preference_v2"] = label.get("mimo_preference", "unsure")
            row["semantic_confidence_v2"] = label.get("mimo_confidence", "0")
            row["semantic_model_v2"] = label.get("mimo_model", "")
            row["semantic_label_provenance_v2"] = "sensor_aligned_commercial_vlm"
        elif args.semantic_source == "legacy_audit":
            row["semantic_preference_v2"] = pair.get("legacy_mimo_preference", "unsure")
            row["semantic_confidence_v2"] = pair.get("legacy_mimo_confidence", "0")
            row["semantic_model_v2"] = pair.get("legacy_mimo_model", "")
            row["semantic_label_provenance_v2"] = "legacy_video_audit_only"
        else:
            row["semantic_preference_v2"] = "unsure"
            row["semantic_confidence_v2"] = 0.0
            row["semantic_model_v2"] = ""
            row["semantic_label_provenance_v2"] = "none"
        joined.append(row)
    return joined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--physical-scores", type=Path, required=True)
    parser.add_argument("--semantic-labels", type=Path, default=None)
    parser.add_argument(
        "--semantic-source",
        choices=("sensor_aligned", "legacy_audit", "none"),
        default="sensor_aligned",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-val-accuracy", type=float, default=0.90)
    parser.add_argument(
        "--single-threshold-grid", type=float, nargs="+", default=[0.60, 0.70, 0.80, 0.90]
    )
    parser.add_argument(
        "--conflict-gap-grid", type=float, nargs="+", default=[0.10, 0.20, 0.30]
    )
    parser.add_argument(
        "--validity-floor-grid", type=float, nargs="+", default=[0.00, 0.25, 0.50]
    )
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--test-split-status",
        choices=("touched", "untouched"),
        default="touched",
        help=(
            "Use untouched only for a newly frozen source-group holdout that was "
            "not inspected during method or hyperparameter development."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args)
    by_split = {split: [row for row in rows if row["split_v2"] == split] for split in SPLITS}
    physical_calibrator = fit_calibrator(
        by_split["train"], "phys_pair_confidence", "phys_preference_v2"
    )
    semantic_train_stated = any(
        row["semantic_preference_v2"] in VALID_LABELS for row in by_split["train"]
    )
    semantic_calibrator = (
        fit_calibrator(
            by_split["train"], "semantic_confidence_v2", "semantic_preference_v2"
        )
        if semantic_train_stated
        else None
    )

    best: dict[str, Any] | None = None
    for floor in args.validity_floor_grid:
        for single in args.single_threshold_grid:
            for gap in args.conflict_gap_grid:
                policy = FusionPolicyV2(single, gap)
                fused_val = fuse_rows(
                    by_split["val"],
                    semantic_calibrator,
                    physical_calibrator,
                    policy,
                    floor,
                )
                accuracy, coverage, labeled = accuracy_coverage(fused_val)
                feasible = accuracy >= args.min_val_accuracy and labeled > 0
                candidate = {
                    "validity_floor": floor,
                    "single_threshold": single,
                    "conflict_gap": gap,
                    "val_accuracy": accuracy,
                    "val_coverage": coverage,
                    "val_labeled": labeled,
                    "feasible": feasible,
                }
                key = (
                    feasible,
                    coverage if feasible else accuracy,
                    accuracy,
                )
                if best is None or key > (
                    best["feasible"],
                    best["val_coverage"] if best["feasible"] else best["val_accuracy"],
                    best["val_accuracy"],
                ):
                    best = candidate
    assert best is not None
    policy = FusionPolicyV2(best["single_threshold"], best["conflict_gap"])
    fused_by_split = {
        split: fuse_rows(
            by_split[split],
            semantic_calibrator,
            physical_calibrator,
            policy,
            best["validity_floor"],
        )
        for split in SPLITS
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, fused in fused_by_split.items():
        write_csv(args.out_dir / f"{split}_pairs_v2.csv", fused)
    write_csv(
        args.out_dir / "fusion_pairs_v2.csv",
        [row for split in SPLITS for row in fused_by_split[split]],
    )
    metrics = {}
    for split in SPLITS:
        if split == "test" and not args.evaluate_test:
            metrics[split] = {"withheld": True}
            continue
        accuracy, coverage, labeled = accuracy_coverage(fused_by_split[split])
        metrics[split] = {
            "accuracy": accuracy,
            "coverage": coverage,
            "labeled_rows": labeled,
            "reason_counts": dict(Counter(row["fusion_reason_v2"] for row in fused_by_split[split])),
        }
    summary = {
        "schema_version": "fusion_labels_v2.0",
        "semantic_source": args.semantic_source,
        "test_split_status": args.test_split_status,
        "primary_scientific_result": (
            args.semantic_source == "sensor_aligned"
            and args.test_split_status == "untouched"
        ),
        "selected_hyperparameters": best,
        "semantic_calibrator": semantic_calibrator.to_dict() if semantic_calibrator else None,
        "physical_calibrator": physical_calibrator.to_dict(),
        "metrics": metrics,
        "completion_constraint": (
            "Narrow terminal consistency cap only; stage/progress remain learned."
        ),
    }
    write_json(args.out_dir / "fusion_labels_v2_manifest.json", summary)
    report = [
        "# Fusion Labels v2",
        "",
        f"- semantic source: `{args.semantic_source}`",
        f"- test split status: `{summary['test_split_status']}`",
        f"- primary scientific result: `{summary['primary_scientific_result']}`",
        f"- selected: `{best}`",
        f"- validation: `{metrics['val']}`",
        f"- test: `{metrics['test']}`",
    ]
    (args.out_dir / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
