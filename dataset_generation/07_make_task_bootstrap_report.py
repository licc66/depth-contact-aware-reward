from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_REPORT_ROOT = Path(r"E:\reward_model_dataset\reports")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_frame(video_path: Path, ratio: float = 0.92) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = max(0, min(n - 1, int(round((n - 1) * ratio)))) if n > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def choose_preview_rows(rows: list[dict[str, str]], max_rows: int = 12) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    groups = [
        ("official_motionplanning_success", None),
        ("truncated_success_trajectory", "truncated_early_approach_or_pregrasp"),
        ("truncated_success_trajectory", "truncated_grasp_or_lift"),
        ("truncated_success_trajectory", "truncated_transport_or_align"),
        ("truncated_success_trajectory", "truncated_place_before_stable_release"),
        ("perturbed_success_final_state", None),
    ]
    for source_type, near_type in groups:
        matches = [
            row
            for row in rows
            if row.get("source_type") == source_type and (near_type is None or row.get("near_miss_type") == near_type)
        ]
        selected.extend(matches[:2])
    if len(selected) < max_rows:
        seen = {row["sample_id"] for row in selected}
        for row in rows:
            if row["sample_id"] not in seen:
                selected.append(row)
                seen.add(row["sample_id"])
            if len(selected) >= max_rows:
                break
    return selected[:max_rows]


def make_preview(rows: list[dict[str, str]], out_path: Path) -> None:
    preview_rows = choose_preview_rows(rows)
    cols = 4
    rows_n = int(np.ceil(len(preview_rows) / cols)) if preview_rows else 1
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.0, rows_n * 2.45), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel(), preview_rows):
        video_path = Path(row.get("video_path_windows") or row.get("video_path") or "")
        frame = read_frame(video_path)
        if frame is not None:
            ax.imshow(frame)
        ax.set_title(
            "\n".join(
                [
                    row.get("sample_id", "")[:36],
                    row.get("source_type", ""),
                    row.get("near_miss_type", "")[:36],
                    f"success={row.get('observed_success', '')}",
                ]
            ),
            fontsize=7,
            loc="left",
        )
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(Counter(row.get(key, "") for row in rows))


def make_report(dataset: Path, indices: Path, out: Path) -> dict[str, Any]:
    traj_rows = load_csv(dataset / "trajectory_manifest.csv")
    clip_rows = load_csv(indices / "clip_manifest.csv") if (indices / "clip_manifest.csv").exists() else []
    pair_rows = load_csv(indices / "pair_manifest.csv") if (indices / "pair_manifest.csv").exists() else []
    task_id = traj_rows[0].get("task_id", dataset.name) if traj_rows else dataset.name
    preview_path = out / f"{dataset.name}_preview.png"
    make_preview(traj_rows, preview_path)

    summary = {
        "task_id": task_id,
        "dataset": str(dataset),
        "indices": str(indices),
        "num_trajectories": len(traj_rows),
        "num_clips": len(clip_rows),
        "num_pairs": len(pair_rows),
        "source_counts": counts(traj_rows, "source_type"),
        "split_counts": counts(traj_rows, "split"),
        "observed_success_counts": counts(traj_rows, "observed_success"),
        "pair_counts": counts(pair_rows, "pair_type"),
        "preview": str(preview_path),
    }
    write_json(out / f"{dataset.name}_report.json", summary)
    lines = [
        f"{dataset.name} report",
        "",
        f"dataset: {dataset}",
        f"indices: {indices}",
        f"preview: {preview_path}",
        "",
        f"trajectories: {len(traj_rows)}",
        f"clips: {len(clip_rows)}",
        f"pairs: {len(pair_rows)}",
        "",
        "source_counts:",
    ]
    lines.extend(f"- {k}: {v}" for k, v in summary["source_counts"].items())
    lines.append("")
    lines.append("observed_success_counts:")
    lines.extend(f"- {k}: {v}" for k, v in summary["observed_success_counts"].items())
    lines.append("")
    lines.append("pair_counts:")
    lines.extend(f"- {k}: {v}" for k, v in summary["pair_counts"].items())
    (out / f"{dataset.name}_report.txt").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make a visual/statistical report for a bootstrap dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or (DEFAULT_REPORT_ROOT / args.dataset.name)
    summary = make_report(args.dataset, args.indices, out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
