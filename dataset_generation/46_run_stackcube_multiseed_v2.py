"""Run and aggregate resumable multi-seed StackCube reward-v2 SAC jobs."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_SEED_SCRIPT = SCRIPT_DIR / "45_train_maniskill_rl_v2.py"
CONDITIONS = ("sparse_only", "rgb_only", "physical_only", "fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 7, 11])
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=["sparse_only", "physical_only", "fusion"],
    )
    parser.add_argument("--physical-checkpoint", type=Path, required=True)
    parser.add_argument("--reward-run-dir", type=Path, required=True)
    parser.add_argument("--openclip-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--inference-interval", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-dense", type=float, default=1.0)
    parser.add_argument("--dense-clip", type=float, default=0.25)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def completed_summary(path: Path, conditions: list[str], seed: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(payload.get("seed", -1)) == seed
        and payload.get("conditions") == conditions
        and all(
            payload.get("results", {}).get(condition, {}).get("status") == "COMPLETED"
            for condition in conditions
        )
    )


def command_for_seed(args: argparse.Namespace, seed: int, seed_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(SINGLE_SEED_SCRIPT),
        "--conditions",
        *args.conditions,
        "--physical-checkpoint",
        str(args.physical_checkpoint),
        "--reward-run-dir",
        str(args.reward_run_dir),
        "--openclip-checkpoint",
        str(args.openclip_checkpoint),
        "--out-dir",
        str(seed_dir),
        "--total-steps",
        str(args.total_steps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--seed",
        str(seed),
        "--inference-interval",
        str(args.inference_interval),
        "--gamma",
        str(args.gamma),
        "--lambda-dense",
        str(args.lambda_dense),
        "--dense-clip",
        str(args.dense_clip),
        "--learning-starts",
        str(args.learning_starts),
        "--batch-size",
        str(args.batch_size),
        "--buffer-size",
        str(args.buffer_size),
        "--learning-rate",
        str(args.learning_rate),
        "--device",
        str(args.device),
    ]
    if args.smoke:
        command.append("--smoke")
    return command


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate(summaries: list[dict[str, Any]], conditions: list[str]) -> dict[str, Any]:
    conditions_out: dict[str, Any] = {}
    for condition in conditions:
        rows = [summary["results"][condition] for summary in summaries]
        evaluations = [row["evaluation"] for row in rows]
        metrics = {
            "success_rate": mean_std(
                [float(item["success_rate"]) for item in evaluations]
            ),
            "mean_return": mean_std(
                [float(item["mean_return"]) for item in evaluations]
            ),
            "mean_sparse_return": mean_std(
                [float(item["mean_sparse_return"]) for item in evaluations]
            ),
            "elapsed_seconds": mean_std(
                [float(item["elapsed_seconds"]) for item in rows]
            ),
        }
        success_steps = [
            float(item["mean_steps_to_success"])
            for item in evaluations
            if item.get("mean_steps_to_success") is not None
        ]
        metrics["mean_steps_to_success"] = (
            mean_std(success_steps) if success_steps else None
        )
        conditions_out[condition] = {
            "seeds": [int(summary["seed"]) for summary in summaries],
            "metrics": metrics,
            "primary_scientific_result": all(
                bool(row.get("primary_scientific_result", False)) for row in rows
            ),
        }
    return conditions_out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# StackCube Multi-Seed Reward v2",
        "",
        f"- seeds: `{payload['completed_seeds']}`",
        f"- audit only: `{payload['audit_only']}`",
        f"- scientific result: `{payload['scientific_result']}`",
        f"- smoke: `{payload['smoke']}`",
        "",
        "| condition | success rate | mean return | mean sparse return | elapsed/seed (s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for condition, item in payload["aggregate"].items():
        metrics = item["metrics"]
        lines.append(
            f"| {condition} | "
            f"{metrics['success_rate']['mean']:.4f} +/- {metrics['success_rate']['std']:.4f} | "
            f"{metrics['mean_return']['mean']:.4f} +/- {metrics['mean_return']['std']:.4f} | "
            f"{metrics['mean_sparse_return']['mean']:.4f} +/- {metrics['mean_sparse_return']['std']:.4f} | "
            f"{metrics['elapsed_seconds']['mean']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed: dict[int, int] = {}
    for seed in args.seeds:
        seed_dir = args.out_dir / f"seed_{seed}"
        summary_path = seed_dir / "experiment_summary.json"
        if not args.force and completed_summary(summary_path, args.conditions, seed):
            print(f"seed {seed}: already complete, skipping", flush=True)
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        command = command_for_seed(args, seed, seed_dir)
        (seed_dir / "launch_command.json").write_text(
            json.dumps(command, indent=2) + "\n", encoding="utf-8"
        )
        print(f"seed {seed}: starting", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failed[seed] = result.returncode
            print(f"seed {seed}: failed with exit {result.returncode}", file=sys.stderr)

    summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        path = args.out_dir / f"seed_{seed}" / "experiment_summary.json"
        if completed_summary(path, args.conditions, seed):
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        print("No completed seed summaries to aggregate.", file=sys.stderr)
        return 1

    audit_only = any(bool(summary.get("audit_only", True)) for summary in summaries)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": "stackcube_multiseed_reward_v2.0",
        "requested_seeds": args.seeds,
        "completed_seeds": [int(summary["seed"]) for summary in summaries],
        "failed_seeds": failed,
        "conditions": args.conditions,
        "total_steps_per_condition": args.total_steps,
        "eval_episodes_per_seed": args.eval_episodes,
        "inference_interval": args.inference_interval,
        "smoke": bool(args.smoke),
        "audit_only": audit_only,
        "scientific_result": bool(
            not args.smoke
            and not audit_only
            and len(summaries) == len(args.seeds)
            and not failed
        ),
        "aggregate": aggregate(summaries, args.conditions),
    }
    (args.out_dir / "multiseed_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.out_dir / "RESULTS.md", payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
