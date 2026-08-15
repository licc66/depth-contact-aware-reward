"""Evaluate reward model v1 variants (Phase 4).

This is the single designated test-split evaluation. It reports, per variant:

- pair accuracy by split, task, and pair type (annotated with the audit F1/F3
  caveat: pair accuracy on this dataset is partially a label-reconstruction
  score);
- hard-negative / near-miss false-positive rates, two ways:
  (a) pairwise: near-miss side scores >= its matched success side,
  (b) completion: near-miss terminal potential >= a completion threshold
      selected on VAL success terminals at a target recall;
- success-trajectory Spearman (clip potentials vs temporal order) and
  temporal order accuracy — the offline temporal-behavior proxies; true
  frame-level backtracking probes require rollout replay and are produced by
  the wrapper dry-run (script 31), not fabricated here;
- confidence calibration (10-bin ECE of Bradley-Terry probabilities);
- missing-modality stress for the fusion variant (physical dropped, rgb
  dropped) and, if a second physical-score CSV from a different 26 run is
  given (e.g. --near-miss-contact-augment off vs train_like), a success-like
  contact stress comparison;
- latency, peak VRAM, parameter count, checkpoint size.

Requires torch; exits 3 with a precise dependency report otherwise.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reward_common_v1 import (  # noqa: E402
    SPLITS,
    VALID_LABELS,
    as_float,
    base_success_id,
    clip_trajectory_id,
    default_split_dir,
    load_csv,
    parse_indices,
    write_json,
)

NEAR_MISS_PAIR_TYPES = (
    "success_vs_offset_hard_negative",
    "success_vs_peg_near_miss",
    "success_vs_pyramid_near_miss",
)
HARD_PAIR_TOKENS = ("near_miss", "truncated", "offset")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor
        while end + 1 < len(order) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end) / 2.0
        for index in range(cursor, end + 1):
            output[order[index]] = rank
        cursor = end + 1
    return output


def spearman(values_a: list[float], values_b: list[float]) -> float:
    rank_a, rank_b = ranks(values_a), ranks(values_b)
    mean_a, mean_b = statistics.mean(rank_a), statistics.mean(rank_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(rank_a, rank_b))
    denominator = (
        sum((a - mean_a) ** 2 for a in rank_a) * sum((b - mean_b) ** 2 for b in rank_b)
    ) ** 0.5
    return numerator / denominator if denominator else 0.0


def ece(probabilities: list[float], labels: list[float], bins: int = 10) -> float:
    if not probabilities:
        return float("nan")
    total = len(probabilities)
    value = 0.0
    for bin_index in range(bins):
        lo, hi = bin_index / bins, (bin_index + 1) / bins
        members = [
            (p, y)
            for p, y in zip(probabilities, labels)
            if (lo <= p < hi) or (bin_index == bins - 1 and p == hi)
        ]
        if not members:
            continue
        mean_p = statistics.mean(p for p, _ in members)
        mean_y = statistics.mean(y for _, y in members)
        value += (len(members) / total) * abs(mean_p - mean_y)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="output dir of script 28")
    parser.add_argument("--fusion-dir", type=Path, required=True, help="output dir of script 27")
    parser.add_argument("--split-dir", type=Path, default=default_split_dir())
    parser.add_argument(
        "--pair-table",
        type=Path,
        default=None,
        help="Full pair metadata table used by scripts 26-28. Use this when "
        "fusion v1 includes rows rejected from the clean split tables.",
    )
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--physical-scores", type=Path, required=True)
    parser.add_argument("--physical-embeddings", type=Path, default=None)
    parser.add_argument(
        "--physical-scores-stress",
        type=Path,
        default=None,
        help="second 26 export (e.g. --near-miss-contact-augment off) for a "
        "success-like-contact / missing-contact comparison",
    )
    parser.add_argument("--variants", nargs="+", default=["rgb_only", "physical_only", "fusion"])
    parser.add_argument("--completion-recall-target", type=float, default=0.95)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        print(
            f"MISSING DEPENDENCY: torch/numpy required ({exc}). Use the Windows "
            "PyTorch environment (FABLE5_OPERATION_GUIDE.md §6). Nothing was fabricated.",
            file=sys.stderr,
        )
        return 3

    import importlib.util as _ilu

    train_spec = _ilu.spec_from_file_location(
        "train_reward_model_v1", SCRIPT_DIR / "28_train_reward_model_v1.py"
    )
    assert train_spec and train_spec.loader
    train_mod = _ilu.module_from_spec(train_spec)
    sys.modules[train_spec.name] = train_mod
    train_spec.loader.exec_module(train_mod)
    from reward_model_v1 import load_checkpoint

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    fusion_rows = {s: load_csv(args.fusion_dir / f"{s}_pairs_v1.csv") for s in SPLITS}
    metadata_rows: list[dict[str, str]] = []
    if args.pair_table is not None:
        metadata_rows = load_csv(args.pair_table)
    else:
        for split in SPLITS:
            metadata_rows.extend(load_csv(args.split_dir / f"{split}_pairs.csv"))
    pair_meta: dict[str, dict[str, str]] = {}
    for row in metadata_rows:
        pair_id = row["pair_id"]
        if pair_id in pair_meta:
            raise ValueError(f"duplicate pair_id {pair_id!r} in pair metadata")
        pair_meta[pair_id] = row
    physical_by_pair: dict[str, dict[str, str]] = {}
    for row in load_csv(args.physical_scores):
        pair_id = row["pair_id"]
        if pair_id in physical_by_pair:
            raise ValueError(f"duplicate pair_id {pair_id!r} in physical scores")
        physical_by_pair[pair_id] = row
    physical_stress = (
        {r["pair_id"]: r for r in load_csv(args.physical_scores_stress)}
        if args.physical_scores_stress
        else None
    )
    clip_features = np.load(args.feature_dir / "clip_features.npy").astype(np.float32)
    text_features = np.load(args.feature_dir / "text_features.npy").astype(np.float32)
    clip_index = {
        r["clip_uid"]: int(r["clip_feature_index"])
        for r in load_csv(args.feature_dir / "clip_manifest.csv")
    }
    text_index = {
        r["text_uid"]: int(r["text_feature_index"])
        for r in load_csv(args.feature_dir / "task_text_manifest.csv")
    }
    embeddings = np.load(args.physical_embeddings) if args.physical_embeddings else None

    required_pair_ids = {
        row["pair_id"] for split in SPLITS for row in fusion_rows[split]
    }
    missing_meta = sorted(required_pair_ids - pair_meta.keys())
    missing_physical = sorted(required_pair_ids - physical_by_pair.keys())
    if missing_meta or missing_physical:
        raise RuntimeError(
            "reward evaluation inputs are incomplete: "
            f"missing_pair_metadata={len(missing_meta)} "
            f"missing_physical_scores={len(missing_physical)}"
        )

    def rgb_feature(meta: dict[str, str], side: str) -> "np.ndarray":
        indices = parse_indices(meta[f"clip_{side}_sample_frame_indices"])
        uid = train_mod.clip_uid(
            meta[f"clip_{side}_id"], meta[f"clip_{side}_video_path_local"], indices
        )
        image = clip_features[clip_index[uid]]
        text = text_features[text_index[train_mod.text_uid(meta["task_goal_text"])]]
        return np.concatenate([image, text, image * text], axis=0).astype(np.float32)

    def physical_feature(kind: str, phys: dict[str, str], side: str) -> "np.ndarray":
        if kind == "summary":
            return np.asarray(
                [
                    as_float(phys.get(f"phys_{side}_stage_p1"), 0.0),
                    as_float(phys.get(f"phys_{side}_stage_p2"), 0.0),
                    as_float(phys.get(f"phys_{side}_stage_p3"), 0.0),
                    as_float(phys.get(f"phys_{side}_stage_p4"), 0.0),
                    as_float(phys.get(f"phys_{side}_local_progress"), 0.0),
                    as_float(phys.get(f"phys_{side}_potential"), 0.0),
                    as_float(phys.get(f"phys_{side}_confidence"), 0.0),
                    as_float(phys.get(f"phys_{side}_depth_validity_ratio"), 0.0),
                    as_float(phys.get(f"phys_{side}_contact_validity_ratio"), 0.0),
                ]
                + train_mod.task_onehot(phys.get("task_id", "")),
                dtype=np.float32,
            )
        uid = phys.get(f"clip_{side}_embedding_uid", "")
        if embeddings is None:
            raise RuntimeError(
                "checkpoint expects physical embeddings; pass --physical-embeddings"
            )
        return np.asarray(embeddings[uid], dtype=np.float32)  # type: ignore[index]

    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "caveat_pair_accuracy": (
            "Pair labels are construction-entangled (audit F1/F3); pair "
            "accuracy on this dataset is partially label reconstruction. "
            "Near-miss FP, temporal behavior, calibration and wrapper dry-run "
            "logs are the primary metrics."
        ),
        "distribution_note": (
            "All numbers below are in-distribution bootstrap evaluations. "
            "OOD/online behavior must come from the ManiSkill wrapper dry-run "
            "and RL experiments (scripts 30-32), not from this table."
        ),
        "test_split_evaluated": True,
        "variants": {},
    }

    for variant in args.variants:
        checkpoint_path = args.run_dir / f"reward_model_v1_{variant}.pt"
        if not checkpoint_path.exists():
            report["variants"][variant] = {"error": f"missing checkpoint {checkpoint_path}"}
            continue
        model, payload = load_checkpoint(checkpoint_path, device=device)
        extra = payload.get("extra", {})
        kind = extra.get("physical_input_kind", "summary")
        phys_mean = np.asarray(extra.get("physical_mean", []), dtype=np.float32)
        phys_std = np.asarray(extra.get("physical_std", []), dtype=np.float32)

        def score_pairs(
            rows: list[dict[str, str]],
            physical_source: dict[str, dict[str, str]],
            rgb_valid_value: float = 1.0,
            phys_valid_value: float = 1.0,
        ) -> list[dict[str, Any]]:
            batch_meta, rgb_a, rgb_b, phys_a, phys_b = [], [], [], [], []
            for row in rows:
                meta = pair_meta.get(row["pair_id"])
                phys = physical_source.get(row["pair_id"])
                if meta is None or phys is None:
                    raise RuntimeError(f"missing joined inputs for pair {row['pair_id']}")
                batch_meta.append((row, meta))
                rgb_a.append(rgb_feature(meta, "a"))
                rgb_b.append(rgb_feature(meta, "b"))
                pa = physical_feature(kind, phys, "a")
                pb = physical_feature(kind, phys, "b")
                if phys_mean.size:
                    pa = (pa - phys_mean) / phys_std
                    pb = (pb - phys_mean) / phys_std
                phys_a.append(pa)
                phys_b.append(pb)
            if not batch_meta:
                return []
            with torch.no_grad():
                ra = torch.from_numpy(np.stack(rgb_a)).float().to(device)
                rb = torch.from_numpy(np.stack(rgb_b)).float().to(device)
                pa = torch.from_numpy(np.stack(phys_a)).float().to(device)
                pb = torch.from_numpy(np.stack(phys_b)).float().to(device)
                rv = torch.full((ra.shape[0],), rgb_valid_value, device=device)
                pv = torch.full((ra.shape[0],), phys_valid_value, device=device)
                out_a = model(ra, pa, rv, pv)
                out_b = model(rb, pb, rv, pv)
            results = []
            for (row, meta), phi_a, phi_b, gate_a, gate_b in zip(
                batch_meta,
                out_a["potential"].cpu().tolist(),
                out_b["potential"].cpu().tolist(),
                out_a["gate_rgb_weight"].cpu().tolist(),
                out_b["gate_rgb_weight"].cpu().tolist(),
            ):
                results.append(
                    {
                        "pair_id": row["pair_id"],
                        "task_id": row["task_id"],
                        "pair_type": meta.get("pair_type", ""),
                        "split": row.get("split_v1", ""),
                        "label": row.get("fusion_label_v1", ""),
                        "clip_a_id": meta.get("clip_a_id", ""),
                        "clip_b_id": meta.get("clip_b_id", ""),
                        "clip_a_start": as_float(meta.get("clip_a_start_frame"), 0.0),
                        "clip_b_start": as_float(meta.get("clip_b_start_frame"), 0.0),
                        "phi_a": phi_a,
                        "phi_b": phi_b,
                        "gate_rgb_a": gate_a,
                        "gate_rgb_b": gate_b,
                    }
                )
            return results

        scored = {s: score_pairs(fusion_rows[s], physical_by_pair) for s in SPLITS}

        def pair_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
            labeled = [r for r in items if r["label"] in VALID_LABELS]
            if not labeled:
                return {"rows": 0}
            correct = [
                (r["phi_a"] >= r["phi_b"]) == (r["label"] == "A>B") for r in labeled
            ]
            by = {}
            for key in ("task_id", "pair_type"):
                groups = defaultdict(list)
                for r, c in zip(labeled, correct):
                    groups[r[key]].append(c)
                by[f"by_{key}"] = {
                    g: {"rows": len(v), "pair_accuracy": sum(v) / len(v)}
                    for g, v in sorted(groups.items())
                }
            hard = [
                c
                for r, c in zip(labeled, correct)
                if any(tok in r["pair_type"] for tok in HARD_PAIR_TOKENS)
            ]
            strict_near_miss = [
                c
                for r, c in zip(labeled, correct)
                if r["pair_type"] in NEAR_MISS_PAIR_TYPES
            ]
            probs = [
                1.0 / (1.0 + np.exp(-(r["phi_a"] - r["phi_b"]) / model.config.preference_temperature))
                for r in labeled
            ]
            ys = [1.0 if r["label"] == "A>B" else 0.0 for r in labeled]
            return {
                "rows": len(labeled),
                "pair_accuracy": sum(correct) / len(correct),
                "hard_negative_rows": len(hard),
                "hard_negative_accuracy": (sum(hard) / len(hard)) if hard else None,
                "hard_pair_error_rate": (1.0 - sum(hard) / len(hard)) if hard else None,
                "strict_near_miss_pair_rows": len(strict_near_miss),
                "near_miss_pairwise_fp_rate": (
                    1.0 - sum(strict_near_miss) / len(strict_near_miss)
                    if strict_near_miss
                    else None
                ),
                "ece_bt_probability": ece([float(p) for p in probs], ys),
                **by,
            }

        # Completion threshold on VAL success terminals; applied to TEST.
        def terminal_potentials(items: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
            success, near_miss = [], []
            for r in items:
                if r["pair_type"] in NEAR_MISS_PAIR_TYPES:
                    success.append(r["phi_a"])
                    near_miss.append(r["phi_b"])
                elif r["pair_type"] == "success_vs_truncated_terminal":
                    success.append(r["phi_a"])
            return success, near_miss

        val_success, _ = terminal_potentials(scored["val"])
        if val_success:
            sorted_success = sorted(val_success)
            k = max(0, int((1.0 - args.completion_recall_target) * len(sorted_success)) - 1)
            completion_threshold = sorted_success[max(k, 0)]
        else:
            completion_threshold = float("nan")
        test_success, test_near_miss = terminal_potentials(scored["test"])
        completion = {
            "threshold_selected_on": "val",
            "completion_threshold": completion_threshold,
            "target_success_recall": args.completion_recall_target,
            "test_success_terminals": len(test_success),
            "test_success_recall": (
                sum(1 for p in test_success if p >= completion_threshold) / len(test_success)
                if test_success
                else None
            ),
            "test_near_miss_terminals": len(test_near_miss),
            "test_near_miss_completion_fp_rate": (
                sum(1 for p in test_near_miss if p >= completion_threshold) / len(test_near_miss)
                if test_near_miss
                else None
            ),
        }

        # Success-trajectory Spearman from clip sequences (clean successes only).
        def trajectory_spearman(items: list[dict[str, Any]]) -> dict[str, Any]:
            clip_potential: dict[tuple[str, str], tuple[float, float]] = {}
            for r in items:
                for side in ("a", "b"):
                    cid = r[f"clip_{side}_id"]
                    traj = clip_trajectory_id(cid)
                    if traj != base_success_id(cid):
                        continue  # OFFSET/TRUNC derivatives are not clean successes
                    clip_potential[(r["task_id"], cid)] = (
                        r[f"clip_{side}_start"],
                        r[f"phi_{side}"],
                    )
            by_traj: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
            for (task, cid), (start, phi) in clip_potential.items():
                by_traj[(task, clip_trajectory_id(cid))].append((start, phi))
            per_task: dict[str, list[float]] = defaultdict(list)
            for (task, _), pairs in by_traj.items():
                if len(pairs) < 3:
                    continue
                pairs.sort()
                per_task[task].append(
                    spearman([p[0] for p in pairs], [p[1] for p in pairs])
                )
            return {
                task: {
                    "trajectories": len(vals),
                    "mean_spearman": statistics.mean(vals) if vals else None,
                }
                for task, vals in sorted(per_task.items())
            }

        variant_report: dict[str, Any] = {
            "checkpoint": str(checkpoint_path),
            "parameters": model.parameter_count(),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "splits": {s: pair_metrics(scored[s]) for s in SPLITS},
            "completion": completion,
            "success_trajectory_spearman_test": trajectory_spearman(scored["test"]),
            "backtracking_probe": (
                "requires frame-level rollout replay; produced by wrapper "
                "dry-run logs (script 31), not fabricated here"
            ),
        }

        if variant == "fusion":
            test_gates = [
                value
                for row in scored["test"]
                for value in (row["gate_rgb_a"], row["gate_rgb_b"])
            ]
            gate_by_task: dict[str, list[float]] = defaultdict(list)
            for row in scored["test"]:
                gate_by_task[row["task_id"]].extend(
                    [row["gate_rgb_a"], row["gate_rgb_b"]]
                )
            variant_report["fusion_gate"] = {
                "meaning": "1.0=RGB branch, 0.0=physical branch",
                "test_mean_rgb_weight": (
                    statistics.mean(test_gates) if test_gates else None
                ),
                "test_median_rgb_weight": (
                    statistics.median(test_gates) if test_gates else None
                ),
                "by_task_mean_rgb_weight": {
                    task: statistics.mean(values)
                    for task, values in sorted(gate_by_task.items())
                    if values
                },
            }

        if variant == "fusion":
            stress: dict[str, Any] = {}
            for name, kwargs in (
                ("physical_missing", {"phys_valid_value": 0.0}),
                ("rgb_missing", {"rgb_valid_value": 0.0}),
            ):
                scored_stress = score_pairs(fusion_rows["test"], physical_by_pair, **kwargs)
                stress[name] = pair_metrics(scored_stress)
                s_succ, s_nm = terminal_potentials(scored_stress)
                stress[name]["near_miss_completion_fp_rate"] = (
                    sum(1 for p in s_nm if p >= completion_threshold) / len(s_nm)
                    if s_nm
                    else None
                )
                stress[name]["success_terminal_mean_potential"] = (
                    statistics.mean(s_succ) if s_succ else None
                )
            if physical_stress is not None:
                scored_alt = score_pairs(fusion_rows["test"], physical_stress)
                stress["alternate_physical_export"] = pair_metrics(scored_alt)
                _, alt_nm = terminal_potentials(scored_alt)
                stress["alternate_physical_export"]["near_miss_completion_fp_rate"] = (
                    sum(1 for p in alt_nm if p >= completion_threshold) / len(alt_nm)
                    if alt_nm
                    else None
                )
            variant_report["stress"] = stress

        # Cost measurements
        with torch.no_grad():
            rgb1 = torch.zeros((1, model.config.rgb_dim), device=device)
            phys1 = torch.zeros((1, model.config.physical_dim), device=device)
            for _ in range(3):
                model(rgb1, phys1)
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for _ in range(50):
                model(rgb1, phys1)
            if device == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) / 50 * 1000.0
        variant_report["latency_ms_batch1"] = latency_ms
        variant_report["peak_vram_bytes"] = (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
        )
        report["variants"][variant] = variant_report
        print(
            f"{variant}: test_pair_acc="
            f"{variant_report['splits']['test'].get('pair_accuracy')} "
            f"nm_completion_fp={completion['test_near_miss_completion_fp_rate']} "
            f"latency={latency_ms:.2f}ms"
        )

    out_path = args.out or (args.run_dir / "metrics_full.json")
    write_json(out_path, report)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
