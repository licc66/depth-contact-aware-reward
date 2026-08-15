"""Evaluate Qwen, frozen physical branch, and fixed fusion on common test pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DECISIVE = {"A>B", "B>A"}
VALID_LABELS = DECISIVE | {"unsure"}


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


def fixed_stage_aware_fusion(
    qwen_label: str,
    qwen_confidence: float,
    physical_label: str,
    physical_basis: str,
) -> tuple[str, str]:
    """A test-label-free fusion rule fixed before benchmark evaluation."""

    qwen_decisive = qwen_label in DECISIVE
    physical_decisive = physical_label in DECISIVE
    if not qwen_decisive and not physical_decisive:
        return "unsure", "both_unsure"
    if qwen_decisive and physical_decisive and qwen_label == physical_label:
        return qwen_label, "semantic_physical_agree"
    if not qwen_decisive:
        return physical_label, "physical_only"
    if not physical_decisive:
        if qwen_confidence >= 0.70:
            return qwen_label, "semantic_only_confident"
        return "unsure", "semantic_only_low_confidence"
    if physical_basis == "stage":
        return physical_label, "physical_stage_override"
    return "unsure", "same_stage_progress_conflict"


def metric_block(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    decisive_reference = [row for row in rows if row["reference_label"] in DECISIVE]
    covered = [row for row in decisive_reference if row[field] in DECISIVE]
    correct_all = sum(row[field] == row["reference_label"] for row in rows)
    correct_decisive = sum(row[field] == row["reference_label"] for row in decisive_reference)
    correct_covered = sum(row[field] == row["reference_label"] for row in covered)
    recalls = []
    for label in ("A>B", "B>A"):
        members = [row for row in decisive_reference if row["reference_label"] == label]
        recalls.append(
            sum(row[field] == label for row in members) / len(members) if members else 0.0
        )
    return {
        "n_all": len(rows),
        "strict_accuracy_all": correct_all / len(rows) if rows else 0.0,
        "n_decisive_reference": len(decisive_reference),
        "effective_accuracy_decisive": (
            correct_decisive / len(decisive_reference)
            if decisive_reference
            else 0.0
        ),
        "coverage_decisive": (
            len(covered) / len(decisive_reference)
            if decisive_reference
            else 0.0
        ),
        "selective_accuracy": correct_covered / len(covered) if covered else 0.0,
        "balanced_accuracy_decisive": sum(recalls) / len(recalls),
        "prediction_counts": dict(Counter(row[field] for row in rows)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-key", type=Path, required=True)
    parser.add_argument("--qwen-labels", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key_rows = read_csv(args.evaluation_key)
    qwen_payload = json.loads(args.qwen_labels.read_text(encoding="utf-8-sig"))
    qwen_by_id = {row["pair_id"]: row for row in qwen_payload["labels"]}
    expected_ids = {row["blind_pair_id"] for row in key_rows}
    if set(qwen_by_id) != expected_ids:
        missing = sorted(expected_ids - qwen_by_id.keys())
        extra = sorted(qwen_by_id.keys() - expected_ids)
        raise RuntimeError(f"Qwen/key mismatch: missing={missing}, extra={extra}")

    joined: list[dict[str, Any]] = []
    for key in key_rows:
        qwen = qwen_by_id[key["blind_pair_id"]]
        if qwen["preference"] not in VALID_LABELS:
            raise ValueError(f"invalid Qwen preference: {qwen['preference']!r}")
        if key["physical_label"] not in VALID_LABELS:
            raise ValueError(f"invalid physical preference: {key['physical_label']!r}")
        fusion_label, fusion_reason = fixed_stage_aware_fusion(
            qwen["preference"],
            float(qwen["confidence"]),
            key["physical_label"],
            key["physical_basis"],
        )
        reference = key["blind_reference_label"]
        joined.append(
            {
                "pair_id": key["blind_pair_id"],
                "original_pair_id": key["original_pair_id"],
                "source_group_id": key["source_group_id"],
                "pair_type": key["pair_type"],
                "reference_label": reference,
                "reference_basis": key["reference_basis"],
                "qwen_label": qwen["preference"],
                "qwen_confidence": qwen["confidence"],
                "qwen_correct": qwen["preference"] == reference,
                "qwen_reason": qwen["reason"],
                "physical_label": key["physical_label"],
                "physical_basis": key["physical_basis"],
                "physical_probability_a_better": key["physical_probability_a_better"],
                "physical_pair_confidence": key["physical_pair_confidence"],
                "physical_correct": key["physical_label"] == reference,
                "fusion_label": fusion_label,
                "fusion_reason": fusion_reason,
                "fusion_correct": fusion_label == reference,
            }
        )

    methods = {
        "qwen": metric_block(joined, "qwen_label"),
        "physical": metric_block(joined, "physical_label"),
        "fusion": metric_block(joined, "fusion_label"),
    }
    pair_types: dict[str, dict[str, Any]] = {}
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_type[row["pair_type"]].append(row)
    for pair_type, rows in sorted(by_type.items()):
        pair_types[pair_type] = {
            name: metric_block(rows, field)
            for name, field in (
                ("qwen", "qwen_label"),
                ("physical", "physical_label"),
                ("fusion", "fusion_label"),
            )
        }

    decisive_rows = [row for row in joined if row["reference_label"] in DECISIVE]
    paired = {
        "qwen_wrong_physical_correct": sum(
            not row["qwen_correct"] and row["physical_correct"] for row in decisive_rows
        ),
        "qwen_correct_physical_wrong": sum(
            row["qwen_correct"] and not row["physical_correct"] for row in decisive_rows
        ),
        "qwen_wrong_fusion_correct": sum(
            not row["qwen_correct"] and row["fusion_correct"] for row in decisive_rows
        ),
        "qwen_correct_fusion_wrong_or_abstain": sum(
            row["qwen_correct"] and not row["fusion_correct"] for row in decisive_rows
        ),
        "qwen_physical_label_conflicts": sum(
            row["qwen_label"] in DECISIVE
            and row["physical_label"] in DECISIVE
            and row["qwen_label"] != row["physical_label"]
            for row in joined
        ),
        "fusion_reason_counts": dict(Counter(row["fusion_reason"] for row in joined)),
    }
    benchmark_manifest = json.loads(args.benchmark_manifest.read_text(encoding="utf-8"))
    summary = {
        "schema_version": "common_stackcube_test_pair_evaluation_v1",
        "qwen_model": qwen_payload.get("model", ""),
        "benchmark": benchmark_manifest,
        "fusion_policy": {
            "name": "fixed_stage_aware_fusion_v1",
            "uses_test_reference_for_decisions": False,
            "rules": [
                "agree -> accept",
                "one branch unsure -> use the decisive branch; Qwen-only requires confidence >= 0.70",
                "conflict with physical stage evidence -> physical label",
                "same-stage progress conflict -> abstain",
            ],
        },
        "methods": methods,
        "paired_comparison": paired,
        "pair_type_metrics": pair_types,
        "interpretation_limits": [
            "The source group is disjoint from model training, but this development holdout was inspected before this evaluation.",
            "Reference labels are offline simulator stage/progress proxies, not human preference ground truth.",
            "All test pairs come from one held-out source-success group, so confidence intervals would overstate trajectory diversity.",
            "Qwen sees six RGB frames per clip; the physical branch sees its frozen depth/contact sequence features.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "common_pair_predictions.csv", joined)
    (args.out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Common StackCube Test-Pair Comparison",
        "",
        f"- Pairs: {len(joined)}; decisive simulator references: {len(decisive_rows)}.",
        f"- Held-out source groups: {', '.join(benchmark_manifest['source_groups'])}.",
        "- The same blinded A/B orientation is used for Qwen, physical, fusion, and reference evaluation.",
        "",
        "| Method | Effective accuracy | Coverage | Selective accuracy | Balanced accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("qwen", "physical", "fusion"):
        metric = methods[name]
        lines.append(
            f"| {name} | {metric['effective_accuracy_decisive']:.3f} | "
            f"{metric['coverage_decisive']:.3f} | {metric['selective_accuracy']:.3f} | "
            f"{metric['balanced_accuracy_decisive']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Qwen/physical decisive conflicts: {paired['qwen_physical_label_conflicts']}.",
            f"Qwen errors corrected by fusion: {paired['qwen_wrong_fusion_correct']}.",
            f"Qwen-correct pairs lost or abstained by fusion: {paired['qwen_correct_fusion_wrong_or_abstain']}.",
            "",
            "## Interpretation boundary",
            "",
            "This is a source-group-held-out proxy benchmark, not a final untouched test. "
            "The simulator reference is derived from privileged stage/progress targets, and all pairs "
            "come from one source-success group. It is suitable for pipeline comparison and error analysis, "
            "but not yet for a paper-level generalization claim.",
        ]
    )
    (args.out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
