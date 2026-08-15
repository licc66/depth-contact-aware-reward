"""Validate sensor-aligned commercial-VLM pair labels before v2 fusion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


VALID_PREFERENCES = {"A>B", "B>A", "unsure"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def duplicate_ids(rows: list[dict[str, str]]) -> list[str]:
    counts = Counter(row.get("pair_id", "") for row in rows)
    return sorted(pair_id for pair_id, count in counts.items() if count > 1)


def validate(
    pairs: list[dict[str, str]], labels: list[dict[str, str]]
) -> dict[str, Any]:
    pair_ids = {row.get("pair_id", "") for row in pairs}
    label_ids = {row.get("pair_id", "") for row in labels}
    invalid_preferences = sorted(
        row.get("pair_id", "")
        for row in labels
        if row.get("mimo_preference", "") not in VALID_PREFERENCES
    )
    invalid_confidence = []
    for row in labels:
        try:
            confidence = float(row.get("mimo_confidence", "nan"))
        except ValueError:
            confidence = float("nan")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            invalid_confidence.append(row.get("pair_id", ""))
    error_rows = sorted(
        row.get("pair_id", "")
        for row in labels
        if row.get("reason", "").startswith("ERROR:")
    )
    missing_model = sorted(
        row.get("pair_id", "")
        for row in labels
        if not row.get("mimo_model", "").strip()
    )
    report = {
        "schema_version": "sensor_aligned_label_validation_v2.0",
        "pair_rows": len(pairs),
        "label_rows": len(labels),
        "missing_pair_ids": sorted(pair_ids - label_ids),
        "unexpected_pair_ids": sorted(label_ids - pair_ids),
        "duplicate_pair_ids": duplicate_ids(labels),
        "invalid_preference_pair_ids": invalid_preferences,
        "invalid_confidence_pair_ids": sorted(invalid_confidence),
        "api_error_pair_ids": error_rows,
        "missing_model_pair_ids": missing_model,
        "preference_counts": dict(
            Counter(row.get("mimo_preference", "") for row in labels)
        ),
        "split_counts": dict(Counter(row.get("split", "") for row in labels)),
    }
    report["valid_for_fusion"] = not any(
        report[key]
        for key in (
            "missing_pair_ids",
            "unexpected_pair_ids",
            "duplicate_pair_ids",
            "invalid_preference_pair_ids",
            "invalid_confidence_pair_ids",
            "api_error_pair_ids",
            "missing_model_pair_ids",
        )
    ) and len(labels) == len(pairs)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate(read_csv(args.pairs), read_csv(args.labels))
    payload = json.dumps(report, indent=2)
    print(payload)
    output = args.out or args.labels.with_name("label_validation_v2.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["valid_for_fusion"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
