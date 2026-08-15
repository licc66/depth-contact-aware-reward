"""Aggregate source-group cross-validation for physical progress v2."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "stage_macro_f1",
    "potential_spearman",
    "terminal_pair_accuracy",
    "potential_mae",
    "mean_success_trajectory_spearman",
    "failure_stage4_false_positive_rate",
    "frame_stage4_false_positive_rate",
    "near_miss_frame_potential_ge_075_rate",
)

LOWER_IS_BETTER = {
    "potential_mae",
    "failure_stage4_false_positive_rate",
    "frame_stage4_false_positive_rate",
    "near_miss_frame_potential_ge_075_rate",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(values: list[float]) -> np.ndarray:
    return np.asarray([value for value in values if math.isfinite(value)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fold_dirs = sorted(
        [path for path in args.cv_root.glob("fold_*") if path.is_dir()],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not fold_dirs:
        raise RuntimeError(f"no fold directories found in {args.cv_root}")
    rows = []
    for fold_dir in fold_dirs:
        config = read_json(fold_dir / "run_config.json")
        fold = config.get("group_fold_definition")
        if not fold:
            raise RuntimeError(f"missing group fold definition in {fold_dir}")
        metrics = read_json(fold_dir / "all_metrics.json")
        for variant in ("observable_rule", "fusion"):
            test = metrics[variant]["test"]
            rows.append(
                {
                    "fold": fold["fold_index"],
                    "test_group": fold["test_group"],
                    "val_group": fold["val_group"],
                    "variant": variant,
                    **{metric: test[metric] for metric in METRICS},
                }
            )
    expected = len({row["fold"] for row in rows}) * 2
    if len(rows) != expected:
        raise RuntimeError("each fold must contain rule and fusion results")
    write_csv(args.cv_root / "fold_test_metrics.csv", rows)

    aggregate_rows = []
    for variant in ("observable_rule", "fusion"):
        variant_rows = [row for row in rows if row["variant"] == variant]
        for metric in METRICS:
            values = finite([float(row[metric]) for row in variant_rows])
            aggregate_rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "folds": len(values),
                    "mean": float(np.mean(values)) if len(values) else float("nan"),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)) if len(values) else float("nan"),
                    "max": float(np.max(values)) if len(values) else float("nan"),
                }
            )
    write_csv(args.cv_root / "aggregate_metrics.csv", aggregate_rows)

    paired_rows = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        rule = next(
            row for row in rows if int(row["fold"]) == fold and row["variant"] == "observable_rule"
        )
        fusion = next(
            row for row in rows if int(row["fold"]) == fold and row["variant"] == "fusion"
        )
        paired = {
            "fold": fold,
            "test_group": rule["test_group"],
        }
        for metric in METRICS:
            raw_delta = float(fusion[metric]) - float(rule[metric])
            paired[f"{metric}_fusion_minus_rule"] = raw_delta
            paired[f"{metric}_fusion_wins"] = int(
                raw_delta < 0 if metric in LOWER_IS_BETTER else raw_delta > 0
            )
        paired_rows.append(paired)
    write_csv(args.cv_root / "paired_fold_differences.csv", paired_rows)

    aggregate = {
        (row["variant"], row["metric"]): row for row in aggregate_rows
    }
    lines = [
        "# Physical Progress v2 Group Cross-Validation",
        "",
        f"- completed folds: {len(fold_dirs)}",
        "- each fold uses one source group for test, the next for validation, and six for training",
        "",
        "| metric | observable rule | fusion | fusion wins |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        rule = aggregate[("observable_rule", metric)]
        fusion = aggregate[("fusion", metric)]
        wins = sum(int(row[f"{metric}_fusion_wins"]) for row in paired_rows)
        lines.append(
            f"| {metric} | {rule['mean']:.3f} +/- {rule['std']:.3f} | "
            f"{fusion['mean']:.3f} +/- {fusion['std']:.3f} | {wins}/{len(paired_rows)} |"
        )
    lines += [
        "",
        "These folds reuse trajectories derived from eight successful source rollouts.",
        "They estimate source-group sensitivity, not broad policy or task generalization.",
    ]
    (args.cv_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.cv_root / "RESULTS.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

