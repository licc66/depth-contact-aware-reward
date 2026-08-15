"""Build calibrated stage-aware fusion labels v1 (Phase 3).

Inputs
------
- a pair table carrying the offline semantic teacher columns
  (``mimo_preference`` in {A>B, B>A, unsure}, ``mimo_confidence`` bucket) —
  either the clean split files or, preferably, the FULL pre-filter fused table
  so that real conflicts exist (audit F1);
- the learned physical pair scores exported by
  ``26_export_physical_pair_scores_v1.py`` (stage distributions, potentials,
  confidence, sensor-validity ratios, stated preference);
- the split directory, used only to map source groups to train/val/test.

Method (all torch-free, deterministic)
--------------------------------------
1. Calibrate each branch on TRAIN rows only:
   semantic: Laplace-smoothed P(correct | confidence bucket);
   physical: isotonic (PAV) map from pair confidence to P(correct).
   Raw confidences from the two branches are never averaged or compared
   directly — only calibrated correctness probabilities are compared.
2. Physical direction is stage-first: soft expected-stage difference beyond a
   margin decides; otherwise within-stage progress (potential margin) decides;
   otherwise the physical branch is unsure.
3. Physical calibrated evidence is discounted toward 0.5 when depth/contact
   validity is poor (linear ramp with a validation-selected floor).
4. Fusion per pair: agree -> high-confidence label; one branch unsure -> the
   other branch only if its calibrated evidence passes a validation-selected
   threshold; conflict -> abstain unless the calibrated gap passes a
   validation-selected threshold. Abstentions are preserved as rows with
   ``fusion_label_v1 = abstain`` and zero preference-loss weight.
5. Thresholds (single-branch, conflict gap, stage margin, validity floor) are
   selected on VAL with a documented objective (max labeled coverage subject
   to a minimum val accuracy). TRAIN fits calibrators; TEST is untouched
   unless ``--evaluate-test`` is passed, and the manifest records whether it
   ever was.

Calibration reference caveat (audit F1): correctness during calibration and
threshold selection is measured against ``--calibration-reference`` (default
``candidate_label``), which is itself a constructed rule label; on the clean
split tables the semantic branch never disagrees with it. All outputs state
this. PSL is deliberately not added (master prompt: no terminology without a
tested implementation and an ablation).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reward_common_v1 import (  # noqa: E402
    PavCalibrator,
    SPLITS,
    VALID_LABELS,
    as_float,
    binned_reliability,
    build_group_split_map,
    default_split_dir,
    leakage_report,
    load_csv,
    pair_group_id,
    write_csv,
    write_json,
)

FUSION_VERSION = "fusion_labels_v1.0"
NUM_STAGES = 4

OUTPUT_COLUMNS = [
    "pair_id",
    "task_id",
    "pair_type",
    "source_group_id",
    "split_v1",
    "clip_a_id",
    "clip_b_id",
    "semantic_preference",
    "semantic_confidence_bucket",
    "semantic_calibrated_p",
    "phys_preference_stated",
    "phys_stage_expected_a",
    "phys_stage_expected_b",
    "phys_potential_a",
    "phys_potential_b",
    "phys_probability_a_better",
    "phys_raw_pair_confidence",
    "phys_validity_min",
    "phys_calibrated_p",
    "fusion_label_v1",
    "fusion_weight_v1",
    "fusion_confidence_v1",
    "fusion_reason_v1",
    "fusion_evidence_v1",
    "conflict_type_v1",
    "needs_manual_review_v1",
    "use_for_preference_loss_v1",
]


def opposite(label: str) -> str:
    if label == "A>B":
        return "B>A"
    if label == "B>A":
        return "A>B"
    return "unsure"


def expected_stage(record: dict[str, str], side: str) -> float:
    total = 0.0
    for stage in range(1, NUM_STAGES + 1):
        total += stage * as_float(record.get(f"phys_{side}_stage_p{stage}", "0"), 0.0)
    return total


def physical_direction(
    record: dict[str, str],
    stage_margin: float,
    progress_epsilon: float,
) -> tuple[str, str]:
    """Stage distributions first, within-stage progress second.

    Returns (direction, basis) with basis in {stage, progress, none}.
    """
    delta_stage = expected_stage(record, "a") - expected_stage(record, "b")
    if abs(delta_stage) >= stage_margin:
        return ("A>B" if delta_stage > 0 else "B>A", "stage")
    delta_potential = as_float(record.get("phys_a_potential"), 0.0) - as_float(
        record.get("phys_b_potential"), 0.0
    )
    if abs(delta_potential) >= progress_epsilon:
        return ("A>B" if delta_potential > 0 else "B>A", "progress")
    return ("unsure", "none")


def validity_factor(min_validity: float, validity_floor: float, validity_ok: float = 0.90) -> float:
    if validity_ok <= validity_floor:
        return 1.0
    return max(0.0, min(1.0, (min_validity - validity_floor) / (validity_ok - validity_floor)))


def discount_toward_half(probability: float, factor: float) -> float:
    return 0.5 + (probability - 0.5) * factor


class FusionPolicy:
    """Deterministic fusion decision given calibrated evidence.

    Kept as a small object so unit tests can exercise every branch of the
    decision table without file IO.
    """

    def __init__(
        self,
        single_threshold: float,
        conflict_gap: float,
        conflict_resolution_penalty: float = 0.5,
        min_weight: float = 0.05,
    ) -> None:
        self.single_threshold = single_threshold
        self.conflict_gap = conflict_gap
        self.conflict_resolution_penalty = conflict_resolution_penalty
        self.min_weight = min_weight

    def _weight(self, probability: float) -> float:
        return max(self.min_weight, min(1.0, 2.0 * (probability - 0.5)))

    def decide(
        self,
        semantic_pref: str,
        semantic_p: float | None,
        physical_pref: str,
        physical_p: float | None,
        stage_basis: str,
    ) -> dict[str, Any]:
        semantic_stated = semantic_pref in VALID_LABELS
        physical_stated = physical_pref in VALID_LABELS

        if not semantic_stated and not physical_stated:
            return self._abstain("no_signal", "both branches unsure", "")

        if semantic_stated and physical_stated:
            if semantic_pref == physical_pref:
                # Combine only evidence above chance. A branch calibrated at
                # p=0.5 is neutral and must not inflate the other branch.
                semantic_evidence = max(0.0, min(1.0, 2.0 * (semantic_p - 0.5)))
                physical_evidence = max(0.0, min(1.0, 2.0 * (physical_p - 0.5)))
                combined = 0.5 + 0.5 * (
                    1.0 - (1.0 - semantic_evidence) * (1.0 - physical_evidence)
                )
                return {
                    "label": semantic_pref,
                    "weight": self._weight(combined),
                    "confidence": combined,
                    "reason": "semantic_physical_agree",
                    "evidence": (
                        f"sem_p={semantic_p:.3f} phys_p={physical_p:.3f} "
                        f"combined(chance-corrected noisy-or)={combined:.3f}"
                    ),
                    "conflict_type": "",
                    "manual_review": False,
                    "use_for_loss": True,
                }
            conflict_type = (
                "semantic_vs_physical_stage"
                if stage_basis == "stage"
                else "semantic_vs_physical_progress"
            )
            strong_label, strong_p, weak_p, strong_name = (
                (semantic_pref, semantic_p, physical_p, "semantic")
                if semantic_p >= physical_p
                else (physical_pref, physical_p, semantic_p, "physical")
            )
            gap = strong_p - weak_p
            if gap >= self.conflict_gap and strong_p >= self.single_threshold:
                weight = self._weight(strong_p) * self.conflict_resolution_penalty
                return {
                    "label": strong_label,
                    "weight": max(self.min_weight, weight),
                    "confidence": strong_p,
                    "reason": f"conflict_resolved_{strong_name}",
                    "evidence": (
                        f"sem_p={semantic_p:.3f} phys_p={physical_p:.3f} gap={gap:.3f} "
                        f">= tau_gap={self.conflict_gap:.3f}"
                    ),
                    "conflict_type": conflict_type,
                    "manual_review": True,
                    "use_for_loss": True,
                }
            return self._abstain(
                "conflict_abstain",
                f"sem_p={semantic_p:.3f} phys_p={physical_p:.3f} gap={gap:.3f} "
                f"< tau_gap={self.conflict_gap:.3f}",
                conflict_type,
            )

        # exactly one branch stated a preference
        if semantic_stated:
            name, label, probability = "semantic", semantic_pref, semantic_p
        else:
            name, label, probability = "physical", physical_pref, physical_p
        if probability >= self.single_threshold:
            return {
                "label": label,
                "weight": self._weight(probability),
                "confidence": probability,
                "reason": f"{name}_only",
                "evidence": f"{name}_p={probability:.3f} >= tau_single={self.single_threshold:.3f}",
                "conflict_type": "",
                "manual_review": False,
                "use_for_loss": True,
            }
        return self._abstain(
            f"{name}_below_threshold",
            f"{name}_p={probability:.3f} < tau_single={self.single_threshold:.3f}",
            "",
        )

    def _abstain(self, reason: str, evidence: str, conflict_type: str) -> dict[str, Any]:
        return {
            "label": "abstain",
            "weight": 0.0,
            "confidence": 0.0,
            "reason": reason,
            "evidence": evidence,
            "conflict_type": conflict_type,
            "manual_review": True,
            "use_for_loss": False,
        }


def fuse_rows(
    rows: list[dict[str, str]],
    semantic_reliability: dict[str, float],
    physical_calibrator: PavCalibrator,
    policy: FusionPolicy,
    stage_margin: float,
    progress_epsilon: float,
    validity_floor: float,
) -> list[dict[str, Any]]:
    default_semantic_p = (
        sum(semantic_reliability.values()) / len(semantic_reliability)
        if semantic_reliability
        else 0.5
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        semantic_pref = row.get("mimo_preference", "").strip() or "unsure"
        if semantic_pref not in VALID_LABELS:
            semantic_pref = "unsure"
        bucket = str(row.get("mimo_confidence", "")).strip().lower()
        semantic_p = semantic_reliability.get(bucket, default_semantic_p)
        semantic_p = max(0.5, min(1.0, semantic_p))

        physical_pref, stage_basis = physical_direction(row, stage_margin, progress_epsilon)
        raw_conf = as_float(row.get("phys_pair_confidence"), 0.0)
        physical_p = physical_calibrator.predict([raw_conf])[0]
        physical_p = max(0.5, min(1.0, physical_p))
        min_validity = min(
            as_float(row.get("phys_a_depth_validity_ratio"), 1.0),
            as_float(row.get("phys_b_depth_validity_ratio"), 1.0),
            as_float(row.get("phys_a_contact_validity_ratio"), 1.0),
            as_float(row.get("phys_b_contact_validity_ratio"), 1.0),
        )
        factor = validity_factor(min_validity, validity_floor)
        physical_p = discount_toward_half(physical_p, factor)
        if physical_pref == "unsure":
            physical_p = 0.5

        decision = policy.decide(
            semantic_pref, semantic_p, physical_pref, physical_p, stage_basis
        )
        output.append(
            {
                "pair_id": row.get("pair_id", ""),
                "task_id": row.get("task_id", ""),
                "pair_type": row.get("pair_type", ""),
                "source_group_id": row.get("source_group_id") or pair_group_id(row),
                "split_v1": row.get("split_v1", ""),
                "clip_a_id": row.get("clip_a_id", ""),
                "clip_b_id": row.get("clip_b_id", ""),
                "semantic_preference": semantic_pref,
                "semantic_confidence_bucket": bucket,
                "semantic_calibrated_p": round(semantic_p, 6),
                "phys_preference_stated": physical_pref,
                "phys_stage_expected_a": round(expected_stage(row, "a"), 4),
                "phys_stage_expected_b": round(expected_stage(row, "b"), 4),
                "phys_potential_a": as_float(row.get("phys_a_potential"), float("nan")),
                "phys_potential_b": as_float(row.get("phys_b_potential"), float("nan")),
                "phys_probability_a_better": as_float(
                    row.get("phys_probability_a_better"), float("nan")
                ),
                "phys_raw_pair_confidence": raw_conf,
                "phys_validity_min": round(min_validity, 6),
                "phys_calibrated_p": round(physical_p, 6),
                "fusion_label_v1": decision["label"],
                "fusion_weight_v1": round(decision["weight"], 6),
                "fusion_confidence_v1": round(decision["confidence"], 6),
                "fusion_reason_v1": decision["reason"],
                "fusion_evidence_v1": decision["evidence"],
                "conflict_type_v1": decision["conflict_type"],
                "needs_manual_review_v1": decision["manual_review"],
                "use_for_preference_loss_v1": decision["use_for_loss"],
            }
        )
    return output


def accuracy_and_coverage(
    fused: list[dict[str, Any]],
    reference: dict[str, str],
) -> tuple[float, float, int]:
    labeled = [
        row
        for row in fused
        if row["fusion_label_v1"] in VALID_LABELS and reference.get(row["pair_id"], "") in VALID_LABELS
    ]
    total_with_reference = sum(
        1 for row in fused if reference.get(row["pair_id"], "") in VALID_LABELS
    )
    if not labeled:
        return 0.0, 0.0, 0
    correct = sum(
        1 for row in labeled if row["fusion_label_v1"] == reference[row["pair_id"]]
    )
    coverage = len(labeled) / max(total_with_reference, 1)
    return correct / len(labeled), coverage, len(labeled)


def fit_calibrators(
    train_rows: list[dict[str, str]],
    reference: dict[str, str],
) -> tuple[dict[str, float], PavCalibrator, dict[str, Any]]:
    semantic_keys: list[str] = []
    semantic_correct: list[float] = []
    physical_scores: list[float] = []
    physical_correct: list[float] = []
    for row in train_rows:
        ref = reference.get(row.get("pair_id", ""), "")
        if ref not in VALID_LABELS:
            continue
        pref = row.get("mimo_preference", "").strip()
        if pref in VALID_LABELS:
            semantic_keys.append(str(row.get("mimo_confidence", "")).lower())
            semantic_correct.append(1.0 if pref == ref else 0.0)
        stated = row.get("_phys_pref_for_fit", "")
        if stated in VALID_LABELS:
            physical_scores.append(as_float(row.get("phys_pair_confidence"), 0.0))
            physical_correct.append(1.0 if stated == ref else 0.0)
    if not physical_scores:
        raise RuntimeError("no train rows available to calibrate the physical branch")
    reliability = binned_reliability(semantic_keys, semantic_correct) if semantic_keys else {}
    calibrator = PavCalibrator().fit(physical_scores, physical_correct)
    info = {
        "semantic_train_rows": len(semantic_correct),
        "semantic_reliability_by_bucket": reliability,
        "physical_train_rows": len(physical_scores),
        "physical_calibrator": calibrator.to_dict(),
    }
    return reliability, calibrator, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--split-dir", type=Path, default=default_split_dir())
    parser.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="Optional full pre-filter pair table (recommended so conflicts exist; audit F1).",
    )
    parser.add_argument("--physical-scores", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-reference", default="candidate_label")
    parser.add_argument("--min-val-accuracy", type=float, default=0.90)
    parser.add_argument("--progress-epsilon", type=float, default=0.02)
    parser.add_argument(
        "--stage-margin-grid", type=float, nargs="+", default=[0.10, 0.25, 0.50]
    )
    parser.add_argument(
        "--single-threshold-grid",
        type=float,
        nargs="+",
        default=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )
    parser.add_argument(
        "--conflict-gap-grid", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30]
    )
    parser.add_argument(
        "--validity-floor-grid", type=float, nargs="+", default=[0.00, 0.25, 0.50]
    )
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args()


def load_joined_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.pairs is not None:
        pair_rows = load_csv(args.pairs)
        group_split = build_group_split_map(args.split_dir)
        for row in pair_rows:
            group = row.get("source_group_id") or pair_group_id(row)
            row["source_group_id"] = group
            row["split_v1"] = group_split.get(group, "unassigned")
    else:
        pair_rows = []
        for split in SPLITS:
            for row in load_csv(args.split_dir / f"{split}_pairs.csv"):
                row["split_v1"] = split
                pair_rows.append(row)
    physical: dict[str, dict[str, str]] = {}
    for score_row in load_csv(args.physical_scores):
        pair_id = score_row["pair_id"]
        if pair_id in physical:
            raise ValueError(f"duplicate pair_id {pair_id!r} in physical scores")
        physical[pair_id] = score_row
    joined: list[dict[str, str]] = []
    missing = 0
    for row in pair_rows:
        scores = physical.get(row.get("pair_id", ""))
        if scores is None:
            missing += 1
            continue
        merged = dict(row)
        for key, value in scores.items():
            # physical outputs own the phys_* namespace; everything else from
            # the scores file only fills gaps and never overrides pair columns
            if key.startswith("phys_"):
                merged[key] = value
            else:
                merged.setdefault(key, value)
        joined.append(merged)
    if missing:
        raise RuntimeError(
            f"{missing} pair rows have no physical scores; rerun script 26 on "
            "the same --pairs table instead of silently dropping supervision"
        )
    if not joined:
        raise RuntimeError("no rows after joining pairs with physical scores")
    unassigned = sum(row.get("split_v1") not in SPLITS for row in joined)
    if unassigned:
        raise RuntimeError(
            f"{unassigned} rows could not be mapped to canonical train/val/test "
            "source groups; check pair_group_id against script 17"
        )
    return joined


def main() -> int:
    args = parse_args()
    rows = load_joined_rows(args)
    reference = {
        row["pair_id"]: row.get(args.calibration_reference, "")
        for row in rows
    }
    by_split: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    unassigned: list[dict[str, str]] = []
    for row in rows:
        by_split.get(row.get("split_v1", ""), unassigned).append(row)

    # Grid search: stage margin affects the physical stated direction, so it
    # participates in calibration fitting as well (train only) and selection
    # (val only). Test is never read here.
    best: dict[str, Any] | None = None
    for stage_margin in args.stage_margin_grid:
        for split_rows in (by_split["train"], by_split["val"]):
            for row in split_rows:
                row["_phys_pref_for_fit"], _ = physical_direction(
                    row, stage_margin, args.progress_epsilon
                )
        reliability, calibrator, calibration_info = fit_calibrators(
            by_split["train"], reference
        )
        for validity_floor in args.validity_floor_grid:
            for single_threshold in args.single_threshold_grid:
                for conflict_gap in args.conflict_gap_grid:
                    policy = FusionPolicy(single_threshold, conflict_gap)
                    fused_val = fuse_rows(
                        by_split["val"],
                        reliability,
                        calibrator,
                        policy,
                        stage_margin,
                        args.progress_epsilon,
                        validity_floor,
                    )
                    accuracy, coverage, labeled = accuracy_and_coverage(fused_val, reference)
                    feasible = accuracy >= args.min_val_accuracy and labeled > 0
                    candidate = {
                        "stage_margin": stage_margin,
                        "validity_floor": validity_floor,
                        "single_threshold": single_threshold,
                        "conflict_gap": conflict_gap,
                        "val_accuracy": accuracy,
                        "val_coverage": coverage,
                        "val_labeled_rows": labeled,
                        "feasible": feasible,
                        "reliability": reliability,
                        "calibrator": calibrator,
                        "calibration_info": calibration_info,
                    }
                    if best is None:
                        best = candidate
                        continue
                    key = (
                        candidate["feasible"],
                        candidate["val_coverage"] if candidate["feasible"] else candidate["val_accuracy"],
                        candidate["val_accuracy"],
                    )
                    best_key = (
                        best["feasible"],
                        best["val_coverage"] if best["feasible"] else best["val_accuracy"],
                        best["val_accuracy"],
                    )
                    if key > best_key:
                        best = candidate
    assert best is not None
    if not best["feasible"]:
        print(
            f"WARNING: no threshold setting reached min val accuracy "
            f"{args.min_val_accuracy}; using the most accurate setting "
            f"(val_accuracy={best['val_accuracy']:.4f}).",
            file=sys.stderr,
        )

    policy = FusionPolicy(best["single_threshold"], best["conflict_gap"])
    fused_all: list[dict[str, Any]] = []
    for split in SPLITS:
        fused_all.extend(
            fuse_rows(
                by_split[split],
                best["reliability"],
                best["calibrator"],
                policy,
                best["stage_margin"],
                args.progress_epsilon,
                best["validity_floor"],
            )
        )
    fused_all.extend(
        fuse_rows(
            unassigned,
            best["reliability"],
            best["calibrator"],
            policy,
            best["stage_margin"],
            args.progress_epsilon,
            best["validity_floor"],
        )
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "fusion_pairs_v1.csv", fused_all, fieldnames=OUTPUT_COLUMNS)
    for split in SPLITS:
        split_rows = [row for row in fused_all if row["split_v1"] == split]
        write_csv(out_dir / f"{split}_pairs_v1.csv", split_rows, fieldnames=OUTPUT_COLUMNS)

    leakage = leakage_report(
        [row for row in fused_all if row["split_v1"] in SPLITS], split_key="split_v1"
    )
    metrics: dict[str, Any] = {}
    for split in ("train", "val"):
        split_fused = [row for row in fused_all if row["split_v1"] == split]
        accuracy, coverage, labeled = accuracy_and_coverage(split_fused, reference)
        metrics[split] = {
            "rows": len(split_fused),
            "labeled_rows": labeled,
            "accuracy_vs_reference": accuracy,
            "coverage": coverage,
            "label_distribution": dict(
                Counter(row["fusion_label_v1"] for row in split_fused)
            ),
            "reason_distribution": dict(
                Counter(row["fusion_reason_v1"] for row in split_fused)
            ),
        }
    test_evaluated = False
    if args.evaluate_test:
        split_fused = [row for row in fused_all if row["split_v1"] == "test"]
        accuracy, coverage, labeled = accuracy_and_coverage(split_fused, reference)
        metrics["test"] = {
            "rows": len(split_fused),
            "labeled_rows": labeled,
            "accuracy_vs_reference": accuracy,
            "coverage": coverage,
            "label_distribution": dict(
                Counter(row["fusion_label_v1"] for row in split_fused)
            ),
            "reason_distribution": dict(
                Counter(row["fusion_reason_v1"] for row in split_fused)
            ),
        }
        test_evaluated = True

    manifest = {
        "fusion_version": FUSION_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pairs_input": str(args.pairs) if args.pairs else str(args.split_dir),
        "physical_scores": str(args.physical_scores),
        "calibration_reference": args.calibration_reference,
        "calibration_reference_caveat": (
            "The reference is a constructed rule label; on the clean split "
            "tables the semantic branch never disagrees with it (audit F1). "
            "Accuracies against it are label-reconstruction scores."
        ),
        "selected_thresholds": {
            "stage_margin": best["stage_margin"],
            "validity_floor": best["validity_floor"],
            "single_threshold": best["single_threshold"],
            "conflict_gap": best["conflict_gap"],
        },
        "selection_objective": (
            "maximize val labeled coverage subject to val accuracy >= "
            f"{args.min_val_accuracy}; calibrators fit on train only"
        ),
        "selection_feasible": best["feasible"],
        "val_accuracy": best["val_accuracy"],
        "val_coverage": best["val_coverage"],
        "calibration": best["calibration_info"],
        "progress_epsilon": args.progress_epsilon,
        "rows_total": len(fused_all),
        "rows_unassigned_split": len(unassigned),
        "test_split_evaluated": test_evaluated,
        "metrics": metrics,
        "leakage_check": leakage,
        "psl_note": "PSL intentionally not added (no tested implementation/ablation).",
    }
    write_json(out_dir / "fusion_labels_v1_manifest.json", manifest)
    print(
        f"fusion rows={len(fused_all)} "
        f"labels={dict(Counter(row['fusion_label_v1'] for row in fused_all))}"
    )
    print(
        f"selected: stage_margin={best['stage_margin']} tau_single={best['single_threshold']} "
        f"tau_gap={best['conflict_gap']} validity_floor={best['validity_floor']} "
        f"(val acc={best['val_accuracy']:.4f}, cov={best['val_coverage']:.4f})"
    )
    print(f"leakage: groups={leakage['source_group_leakage_count']} bases={leakage['base_success_id_leakage_count']}")
    if leakage["source_group_leakage_count"] or leakage["base_success_id_leakage_count"]:
        print("ERROR: leakage detected in fusion v1 splits", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
