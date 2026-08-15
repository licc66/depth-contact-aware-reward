from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_indices(value: str) -> list[int]:
    if not value:
        return []
    return [int(float(part)) for part in value.replace(",", ";").split(";") if part.strip()]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def read_frame_rgb(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_multiline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    x, y = xy
    for line in text.splitlines():
        draw.text((x, y), line, font=font, fill=fill)
        y += 18


def make_pair_video(row: dict[str, Any], out_path: Path, fps: float = 2.0, size: tuple[int, int] = (360, 270)) -> None:
    a_indices = parse_indices(row["clip_a_sample_frame_indices"])
    b_indices = parse_indices(row["clip_b_sample_frame_indices"])
    n = min(len(a_indices), len(b_indices))
    if n == 0:
        raise RuntimeError(f"No sampled frames for {row['pair_id']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_w, panel_h = size
    header_h = 116
    frame_w = panel_w * 2
    frame_h = header_h + panel_h
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {out_path}")

    font = ImageFont.load_default()
    score_a = float(row["score_a"])
    score_b = float(row["score_b"])
    margin = float(row["margin_a_minus_b"])
    p_a = sigmoid(margin)
    for i in range(n):
        frame_a = Image.fromarray(read_frame_rgb(row["clip_a_video_path_local"], a_indices[i])).resize(size)
        frame_b = Image.fromarray(read_frame_rgb(row["clip_b_video_path_local"], b_indices[i])).resize(size)
        canvas = Image.new("RGB", (frame_w, frame_h), "white")
        draw = ImageDraw.Draw(canvas)
        title = (
            f"{row['audit_id']} | {row['pair_id']} | {row['task_id']} | {row['pair_type']}\n"
            f"label={row['final_preference_label_v0']} pred={row['predicted_label']} correct={row['correct']} "
            f"p(A>B)={p_a:.3f}\n"
            f"A score={score_a:.3f} | B score={score_b:.3f} | margin A-B={margin:.3f} | frame {i + 1}/{n}"
        )
        draw_multiline(draw, (8, 8), title, font, "#111111")
        draw.text((8, header_h - 20), "A", font=font, fill="#1B4E9B")
        draw.text((panel_w + 8, header_h - 20), "B", font=font, fill="#8A1F2D")
        canvas.paste(frame_a, (0, header_h))
        canvas.paste(frame_b, (panel_w, header_h))
        writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
    writer.release()


def join_predictions(run_dir: Path, variant: str, split_dir: Path, split: str) -> list[dict[str, Any]]:
    preds = {row["pair_id"]: row for row in load_csv(run_dir / variant / f"{split}_predictions.csv")}
    rows = []
    for row in load_csv(split_dir / f"{split}_pairs.csv"):
        pred = preds.get(row["pair_id"])
        if not pred:
            continue
        item: dict[str, Any] = dict(row)
        item.update({f"model_{key}": value for key, value in pred.items() if key not in row})
        # Keep the common names simple for audit output.
        for key in ("predicted_label", "correct", "score_a", "score_b", "margin_a_minus_b"):
            item[key] = pred[key]
        item["abs_margin"] = abs(float(pred["margin_a_minus_b"]))
        item["p_a_gt_b"] = sigmoid(float(pred["margin_a_minus_b"]))
        item["model_variant"] = variant
        item["eval_split"] = split
        rows.append(item)
    return rows


def pick_samples(run_dir: Path, split_dir: Path, variant: str, seed: int, max_samples: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = join_predictions(run_dir, variant, split_dir, "test") + join_predictions(run_dir, variant, split_dir, "val")
    rgb_test = join_predictions(run_dir, "rgb_only", split_dir, "test")
    rgb_wrong_ids = {row["pair_id"] for row in rgb_test if str(row["correct"]).lower() != "true"}

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(category: str, candidates: list[dict[str, Any]], k: int, shuffle: bool = True) -> None:
        if shuffle:
            rng.shuffle(candidates)
        for row in candidates:
            if len([item for item in selected if item["audit_category"] == category]) >= k:
                break
            if row["pair_id"] in seen:
                continue
            item = dict(row)
            item["audit_category"] = category
            selected.append(item)
            seen.add(row["pair_id"])

    add("low_margin_boundary", sorted(rows, key=lambda r: r["abs_margin"]), 6, shuffle=False)
    add("order_progress", [r for r in rows if r.get("use_for_order_loss_v0") == "true"], 6)
    add("hard_negative", [r for r in rows if any(key in r.get("pair_type", "") for key in ("near_miss", "truncated", "offset"))], 6)
    add("rgb_wrong_fusion_correct", sorted([r for r in rows if r["pair_id"] in rgb_wrong_ids], key=lambda r: r["abs_margin"]), 6, shuffle=False)
    add("high_margin_sanity", sorted(rows, key=lambda r: r["abs_margin"], reverse=True), 6, shuffle=False)

    selected = selected[:max_samples]
    for idx, row in enumerate(selected, start=1):
        row["audit_id"] = f"AUDIT-{idx:03d}"
        row["p_a_gt_b"] = f"{float(row['p_a_gt_b']):.6f}"
        row["abs_margin"] = f"{float(row['abs_margin']):.6f}"
    return selected


def slim_row(row: dict[str, Any], video_path: Path) -> dict[str, Any]:
    return {
        "audit_id": row["audit_id"],
        "audit_category": row["audit_category"],
        "eval_split": row["eval_split"],
        "pair_id": row["pair_id"],
        "task_id": row["task_id"],
        "pair_type": row["pair_type"],
        "final_preference_label_v0": row["final_preference_label_v0"],
        "predicted_label": row["predicted_label"],
        "correct": row["correct"],
        "score_a": row["score_a"],
        "score_b": row["score_b"],
        "margin_a_minus_b": row["margin_a_minus_b"],
        "p_a_gt_b": row["p_a_gt_b"],
        "clip_a_id": row["clip_a_id"],
        "clip_b_id": row["clip_b_id"],
        "clip_a_sample_frame_indices": row["clip_a_sample_frame_indices"],
        "clip_b_sample_frame_indices": row["clip_b_sample_frame_indices"],
        "clip_a_video_path_local": row["clip_a_video_path_local"],
        "clip_b_video_path_local": row["clip_b_video_path_local"],
        "audit_video": str(video_path),
    }


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Reward Model Manual Audit Samples",
        "",
        "Interpretation:",
        "",
        "- `score_a` and `score_b` are continuous reward/progress scores from the model.",
        "- `predicted_label` is derived only by comparing the scores.",
        "- `p_a_gt_b = sigmoid(score_a - score_b)` is a pairwise confidence, not an environment success probability.",
        "",
        "| audit_id | category | task | pair_type | label | pred | score_a | score_b | p(A>B) | video |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['audit_id']} | {row['audit_category']} | {row['task_id']} | {row['pair_type']} | "
            f"{row['final_preference_label_v0']} | {row['predicted_label']} | "
            f"{float(row['score_a']):.3f} | {float(row['score_b']):.3f} | {float(row['p_a_gt_b']):.3f} | "
            f"`{Path(row['audit_video']).name}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manual audit videos for reward model pair predictions.")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path(r"D:\Users\User\Desktop\reward_model_dataset\dataset_splits\bootstrap_v1_fusion_stereo_v1_clean"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(r"D:\Users\User\Desktop\reward_model_dataset\reward_model_runs\reward_model_v0_openclip"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"D:\Users\User\Desktop\reward_model_dataset\reward_model_audits\fusion_v0_manual_samples"),
    )
    parser.add_argument("--variant", default="fusion")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--no-videos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.out_dir / "videos"
    rows = pick_samples(args.run_dir, args.split_dir, args.variant, args.seed, args.max_samples)
    slim_rows: list[dict[str, Any]] = []
    for row in rows:
        video_path = videos_dir / f"{row['audit_id']}_{row['pair_id']}.mp4"
        if not args.no_videos:
            make_pair_video(row, video_path)
        slim_rows.append(slim_row(row, video_path))
    write_csv(args.out_dir / "audit_samples.csv", slim_rows)
    write_report(args.out_dir / "audit_report.md", slim_rows)
    print(f"Wrote {len(slim_rows)} audit samples to {args.out_dir}")
    print(args.out_dir / "audit_report.md")


if __name__ == "__main__":
    main()
