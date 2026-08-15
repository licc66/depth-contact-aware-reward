"""Evaluate reward-model v2 checkpoints once on the held-out StackCube group."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reward_model_v2 import load_checkpoint  # noqa: E402


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


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def macro_f1(prediction: np.ndarray, target: np.ndarray, classes: int = 4) -> float:
    values = []
    for cls in range(classes):
        pred = prediction == cls
        true = target == cls
        tp = int(np.sum(pred & true))
        fp = int(np.sum(pred & ~true))
        fn = int(np.sum(~pred & true))
        denominator = 2 * tp + fp + fn
        if denominator:
            values.append(2 * tp / denominator)
    return float(np.mean(values)) if values else 0.0


def completion_metrics(
    val_scores: np.ndarray,
    val_stages: np.ndarray,
    test_scores: np.ndarray,
    test_stages: np.ndarray,
    test_sample_ids: list[str],
    recall_target: float,
) -> dict[str, Any]:
    val_success = val_scores[val_stages == 3]
    if not len(val_success):
        return {"error": "no validation completion clips"}
    threshold = float(
        np.quantile(val_success, max(0.0, 1.0 - recall_target), method="lower")
    )
    test_positive = test_stages == 3
    test_negative = ~test_positive
    predicted = test_scores >= threshold
    near_miss = np.asarray(
        ["-OFFSET-" in sample_id or "-TRUNC-" in sample_id for sample_id in test_sample_ids]
    ) & test_negative
    return {
        "threshold_from_val": threshold,
        "val_completion_recall": float(np.mean(val_success >= threshold)),
        "test_completion_recall": (
            float(np.mean(predicted[test_positive])) if np.any(test_positive) else 0.0
        ),
        "test_noncompletion_false_positive_rate": (
            float(np.mean(predicted[test_negative])) if np.any(test_negative) else 0.0
        ),
        "test_near_miss_false_positive_rate": (
            float(np.mean(predicted[near_miss])) if np.any(near_miss) else 0.0
        ),
        "test_completion_clips": int(np.sum(test_positive)),
        "test_noncompletion_clips": int(np.sum(test_negative)),
        "test_near_miss_clips": int(np.sum(near_miss)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--physical-scores", type=Path, required=True)
    parser.add_argument("--physical-embeddings", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--completion-recall-target", type=float, default=0.95)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir or args.run_dir / "evaluation_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    fusion_manifest = json.loads(
        (args.fusion_dir / "fusion_labels_v2_manifest.json").read_text(encoding="utf-8")
    )
    pair_meta = {row["pair_id"]: row for row in read_csv(args.pair_table)}
    physical_pair = {
        row["pair_id"]: row for row in read_csv(args.physical_scores)
    }
    fusion_rows = {
        split: read_csv(args.fusion_dir / f"{split}_pairs_v2.csv")
        for split in ("val", "test")
    }

    clip_values = np.load(args.feature_dir / "clip_features.npy").astype(np.float32)
    text_values = np.load(args.feature_dir / "text_features.npy").astype(np.float32)
    clip_index = {
        row["clip_id"]: integer(row["clip_feature_index"])
        for row in read_csv(args.feature_dir / "clip_manifest.csv")
    }
    text_index = {
        row["task_goal_text"]: integer(row["text_feature_index"])
        for row in read_csv(args.feature_dir / "task_text_manifest.csv")
    }
    archive = np.load(args.physical_embeddings)
    physical_embedding = {
        str(clip_id): archive["embeddings"][index].astype(np.float32)
        for index, clip_id in enumerate(archive["clip_ids"].tolist())
    }

    def rgb(meta: dict[str, str], side: str) -> np.ndarray:
        image = clip_values[clip_index[meta[f"clip_{side}_id"]]]
        text = text_values[text_index[meta["task_goal_text"]]]
        return np.concatenate([image, text, image * text]).astype(np.float32)

    def build(split: str) -> dict[str, Any]:
        rows = []
        for fusion in fusion_rows[split]:
            pair_id = fusion["pair_id"]
            rows.append((fusion, pair_meta[pair_id], physical_pair[pair_id]))
        data: dict[str, Any] = {"rows": rows}
        for side in ("a", "b"):
            data[f"rgb_{side}"] = np.stack([rgb(meta, side) for _, meta, _ in rows])
            data[f"phys_{side}"] = np.stack(
                [physical_embedding[meta[f"clip_{side}_id"]] for _, meta, _ in rows]
            )
        return data

    datasets = {split: build(split) for split in ("val", "test")}

    def unique_clips(split: str) -> list[dict[str, Any]]:
        unique: dict[str, dict[str, Any]] = {}
        for _, meta, physical in datasets[split]["rows"]:
            for side in ("a", "b"):
                clip_id = meta[f"clip_{side}_id"]
                record = {
                    "clip_id": clip_id,
                    "sample_id": meta[f"clip_{side}_sample_id"],
                    "rgb": rgb(meta, side),
                    "physical": physical_embedding[clip_id],
                    "reference_stage": integer(meta[f"reference_{side}_stage_v2"], 1) - 1,
                    "reference_potential": number(
                        meta[f"reference_{side}_potential_v2"], 0.0
                    ),
                    "physical_stage": integer(physical[f"phys_{side}_stage"], 1) - 1,
                    "physical_potential": number(physical[f"phys_{side}_potential"], 0.0),
                }
                if clip_id in unique:
                    previous = unique[clip_id]
                    if (
                        previous["reference_stage"] != record["reference_stage"]
                        or abs(previous["reference_potential"] - record["reference_potential"]) > 1e-6
                    ):
                        raise RuntimeError(f"inconsistent target for {clip_id}")
                else:
                    unique[clip_id] = record
        return list(unique.values())

    unique = {split: unique_clips(split) for split in ("val", "test")}

    def reference_pair_metrics(
        split: str, potential_a: np.ndarray, potential_b: np.ndarray
    ) -> dict[str, float]:
        rows = datasets[split]["rows"]
        ref_mask = np.asarray(
            [fusion["reference_label_v2"] in VALID_LABELS for fusion, _, _ in rows]
        )
        ref_a = np.asarray(
            [fusion["reference_label_v2"] == "A>B" for fusion, _, _ in rows]
        )
        predicted_a = potential_a >= potential_b
        fusion_mask = np.asarray(
            [
                fusion["fusion_label_v2"] in VALID_LABELS
                and truthy(fusion["use_for_preference_loss_v2"])
                for fusion, _, _ in rows
            ]
        )
        fusion_a = np.asarray(
            [fusion["fusion_label_v2"] == "A>B" for fusion, _, _ in rows]
        )
        temporal = np.asarray(
            [meta.get("pair_type") == "intra_success_temporal_gap" for _, meta, _ in rows]
        )
        return {
            "reference_pair_accuracy": float(np.mean(predicted_a[ref_mask] == ref_a[ref_mask])),
            "fusion_pair_accuracy": (
                float(np.mean(predicted_a[fusion_mask] == fusion_a[fusion_mask]))
                if np.any(fusion_mask)
                else 0.0
            ),
            "temporal_order_accuracy": (
                float(np.mean(potential_b[temporal] >= potential_a[temporal]))
                if np.any(temporal)
                else 0.0
            ),
        }

    report: dict[str, Any] = {
        "schema_version": "reward_model_evaluation_v2.0",
        "semantic_source": fusion_manifest["semantic_source"],
        "primary_scientific_result": fusion_manifest["primary_scientific_result"],
        "test_split_touched": True,
        "variants": {},
    }
    prediction_rows: list[dict[str, Any]] = []
    for variant in ("rgb_only", "physical_only", "fusion"):
        checkpoint = args.run_dir / f"reward_model_v2_{variant}.pt"
        model, payload = load_checkpoint(checkpoint, device)
        extra = payload["extra"]
        mean = np.asarray(extra["physical_mean"], dtype=np.float32)
        std = np.asarray(extra["physical_std"], dtype=np.float32)

        def score_pairs(split: str, mode: str = "full") -> tuple[np.ndarray, np.ndarray]:
            data = datasets[split]
            rgb_a = torch.from_numpy(data["rgb_a"]).float().to(device)
            rgb_b = torch.from_numpy(data["rgb_b"]).float().to(device)
            phys_a = torch.from_numpy((data["phys_a"] - mean) / std).float().to(device)
            phys_b = torch.from_numpy((data["phys_b"] - mean) / std).float().to(device)
            count = len(rgb_a)
            rgb_valid = torch.zeros(count, device=device) if mode == "physical_only" else torch.ones(count, device=device)
            phys_valid = torch.zeros(count, device=device) if mode == "rgb_only" else torch.ones(count, device=device)
            with torch.inference_mode():
                a = model(rgb_a, phys_a, rgb_valid, phys_valid)["potential"].cpu().numpy()
                b = model(rgb_b, phys_b, rgb_valid, phys_valid)["potential"].cpu().numpy()
            return a, b

        test_a, test_b = score_pairs("test")
        pair_metrics = reference_pair_metrics("test", test_a, test_b)

        clip_scores: dict[str, np.ndarray] = {}
        clip_stages: dict[str, np.ndarray] = {}
        for split in ("val", "test"):
            clips = unique[split]
            rgb_tensor = torch.from_numpy(np.stack([item["rgb"] for item in clips])).float().to(device)
            phys_tensor = torch.from_numpy(
                (np.stack([item["physical"] for item in clips]) - mean) / std
            ).float().to(device)
            with torch.inference_mode():
                output = model(rgb_tensor, phys_tensor)
            clip_scores[split] = output["potential"].cpu().numpy()
            clip_stages[split] = output["stage_logits"].argmax(dim=-1).cpu().numpy()
        test_reference_potential = np.asarray(
            [item["reference_potential"] for item in unique["test"]]
        )
        test_reference_stage = np.asarray(
            [item["reference_stage"] for item in unique["test"]]
        )
        completion = completion_metrics(
            clip_scores["val"],
            np.asarray([item["reference_stage"] for item in unique["val"]]),
            clip_scores["test"],
            test_reference_stage,
            [item["sample_id"] for item in unique["test"]],
            args.completion_recall_target,
        )
        metrics = {
            **pair_metrics,
            "unique_test_clips": len(unique["test"]),
            "reference_potential_mae": float(
                np.mean(np.abs(clip_scores["test"] - test_reference_potential))
            ),
            "reference_stage_macro_f1": macro_f1(
                clip_stages["test"], test_reference_stage
            ),
            "completion": completion,
            "parameters": int(payload["parameter_count"]),
            "checkpoint_size_mb": checkpoint.stat().st_size / (1024 * 1024),
        }
        if variant == "fusion":
            for mode in ("rgb_only", "physical_only"):
                stressed_a, stressed_b = score_pairs("test", mode)
                metrics[f"missing_modality_{mode}"] = reference_pair_metrics(
                    "test", stressed_a, stressed_b
                )
        first = unique["test"][0]
        rgb_one = torch.from_numpy(first["rgb"]).float().unsqueeze(0).to(device)
        phys_one = torch.from_numpy((first["physical"] - mean) / std).float().unsqueeze(0).to(device)
        for _ in range(20):
            with torch.inference_mode():
                model(rgb_one, phys_one)
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(200):
            with torch.inference_mode():
                model(rgb_one, phys_one)
        if device == "cuda":
            torch.cuda.synchronize()
        metrics["head_only_latency_ms"] = (time.perf_counter() - started) * 1000 / 200
        report["variants"][variant] = metrics
        for item, potential, stage in zip(
            unique["test"], clip_scores["test"], clip_stages["test"]
        ):
            prediction_rows.append(
                {
                    "variant": variant,
                    "clip_id": item["clip_id"],
                    "sample_id": item["sample_id"],
                    "reference_stage": item["reference_stage"] + 1,
                    "reference_potential": item["reference_potential"],
                    "predicted_stage": int(stage) + 1,
                    "predicted_potential": float(potential),
                }
            )

    # Frozen physical progress branch teacher baseline on the same unique clips.
    val_physical = np.asarray([item["physical_potential"] for item in unique["val"]])
    test_physical = np.asarray([item["physical_potential"] for item in unique["test"]])
    test_physical_stage = np.asarray([item["physical_stage"] for item in unique["test"]])
    report["physical_teacher_baseline"] = {
        "reference_potential_mae": float(
            np.mean(
                np.abs(
                    test_physical
                    - np.asarray([item["reference_potential"] for item in unique["test"]])
                )
            )
        ),
        "reference_stage_macro_f1": macro_f1(
            test_physical_stage,
            np.asarray([item["reference_stage"] for item in unique["test"]]),
        ),
        "completion": completion_metrics(
            val_physical,
            np.asarray([item["reference_stage"] for item in unique["val"]]),
            test_physical,
            np.asarray([item["reference_stage"] for item in unique["test"]]),
            [item["sample_id"] for item in unique["test"]],
            args.completion_recall_target,
        ),
    }
    write_json(out_dir / "evaluation_metrics_v2.json", report)
    write_csv(out_dir / "test_clip_predictions_v2.csv", prediction_rows)
    lines = [
        "# Reward Model v2 Evaluation",
        "",
        f"- semantic source: `{report['semantic_source']}`",
        f"- primary scientific result: `{report['primary_scientific_result']}`",
        "",
        "| variant | ref pair acc | potential MAE | stage F1 | completion recall | completion FPR | near-miss FPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, metrics in report["variants"].items():
        completion = metrics["completion"]
        lines.append(
            f"| {variant} | {metrics['reference_pair_accuracy']:.3f} | "
            f"{metrics['reference_potential_mae']:.3f} | "
            f"{metrics['reference_stage_macro_f1']:.3f} | "
            f"{completion['test_completion_recall']:.3f} | "
            f"{completion['test_noncompletion_false_positive_rate']:.3f} | "
            f"{completion['test_near_miss_false_positive_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "This run is an engineering baseline until sensor-aligned commercial-VLM labels replace the audit source.",
        ]
    )
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
