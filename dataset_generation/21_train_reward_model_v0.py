from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


SPLITS = ("train", "val", "test")
VALID_LABELS = {"A>B", "B>A"}
TASK_IDS = ("peginsertion", "stackcube", "stackpyramid")


PHYSICAL_FEATURES = (
    "stereo_end_score_proxy",
    "stereo_end_dist_m",
    "stereo_end_depth_error_m",
    "contact_end_stage_id",
    "contact_grasp_ratio",
    "contact_support_contact_ratio",
)


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_indices(value: str) -> list[int]:
    if not value:
        return []
    return [int(float(part)) for part in value.replace(",", ";").split(";") if part.strip()]


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def clip_uid(clip_id: str, video_path: str, indices: list[int]) -> str:
    return short_hash(f"{clip_id}|{video_path}|{';'.join(map(str, indices))}")


def text_uid(text: str) -> str:
    return short_hash(text)


def as_float(value: str, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def as_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def side_physical(row: dict[str, str], side: str) -> list[float]:
    prefix = f"clip_{side}"
    return [
        as_float(row.get(f"stereo_{prefix}_end_score_proxy", "0")),
        as_float(row.get(f"stereo_{prefix}_end_dist_m", "0")),
        as_float(row.get(f"stereo_{prefix}_end_depth_error_m", "0")),
        as_float(row.get(f"contact_{prefix}_end_stage_id", "0")),
        as_float(row.get(f"contact_{prefix}_grasp_ratio", "0")),
        as_float(row.get(f"contact_{prefix}_support_contact_ratio", "0")),
    ]


def task_onehot(task_id: str) -> list[float]:
    return [1.0 if task_id == item else 0.0 for item in TASK_IDS]


def hard_negative_pair(row: dict[str, str]) -> bool:
    pair_type = row.get("pair_type", "")
    return any(key in pair_type for key in ("near_miss", "truncated", "offset_hard_negative"))


def load_feature_maps(feature_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    clip_features = np.load(feature_dir / "clip_features.npy").astype(np.float32)
    text_features = np.load(feature_dir / "text_features.npy").astype(np.float32)
    clip_manifest = load_csv(feature_dir / "clip_manifest.csv")
    text_manifest = load_csv(feature_dir / "task_text_manifest.csv")
    clip_index = {row["clip_uid"]: int(row["clip_feature_index"]) for row in clip_manifest}
    text_index = {row["text_uid"]: int(row["text_feature_index"]) for row in text_manifest}
    return clip_features, text_features, clip_index, text_index


def load_splits(split_dir: Path) -> dict[str, list[dict[str, str]]]:
    return {split: load_csv(split_dir / f"{split}_pairs.csv") for split in SPLITS}


def fit_standardizer(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    values: list[list[float]] = []
    for row in rows:
        for side in ("a", "b"):
            values.append(side_physical(row, side) + task_onehot(row["task_id"]))
    arr = np.asarray(values, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


@dataclass
class PreparedSplit:
    rows: list[dict[str, str]]
    rgb_a: np.ndarray
    rgb_b: np.ndarray
    physical_a: np.ndarray
    physical_b: np.ndarray
    labels: np.ndarray
    weights: np.ndarray


def rgb_conditioned_feature(image_feat: np.ndarray, text_feat: np.ndarray) -> np.ndarray:
    return np.concatenate([image_feat, text_feat, image_feat * text_feat], axis=0).astype(np.float32)


def prepare_split(
    rows: list[dict[str, str]],
    clip_features: np.ndarray,
    text_features: np.ndarray,
    clip_index: dict[str, int],
    text_index: dict[str, int],
    physical_mean: np.ndarray,
    physical_std: np.ndarray,
) -> PreparedSplit:
    rgb_a: list[np.ndarray] = []
    rgb_b: list[np.ndarray] = []
    physical_a: list[np.ndarray] = []
    physical_b: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []

    for row in rows:
        if row["final_preference_label_v0"] not in VALID_LABELS:
            continue
        text_feat = text_features[text_index[text_uid(row["task_goal_text"])]]
        side_rgb: dict[str, np.ndarray] = {}
        side_phys: dict[str, np.ndarray] = {}
        for side in ("a", "b"):
            indices = parse_indices(row[f"clip_{side}_sample_frame_indices"])
            uid = clip_uid(row[f"clip_{side}_id"], row[f"clip_{side}_video_path_local"], indices)
            img_feat = clip_features[clip_index[uid]]
            side_rgb[side] = rgb_conditioned_feature(img_feat, text_feat)
            phys = np.asarray(side_physical(row, side) + task_onehot(row["task_id"]), dtype=np.float32)
            side_phys[side] = (phys - physical_mean) / physical_std
        rgb_a.append(side_rgb["a"])
        rgb_b.append(side_rgb["b"])
        physical_a.append(side_phys["a"])
        physical_b.append(side_phys["b"])
        labels.append(1.0 if row["final_preference_label_v0"] == "A>B" else 0.0)
        base_weight = max(as_float(row.get("preference_loss_weight_v0", "1"), 1.0), 1e-6)
        if as_bool(row.get("use_for_order_loss_v0", "false")):
            base_weight *= 0.7
        weights.append(base_weight)

    return PreparedSplit(
        rows=rows,
        rgb_a=np.stack(rgb_a).astype(np.float32),
        rgb_b=np.stack(rgb_b).astype(np.float32),
        physical_a=np.stack(physical_a).astype(np.float32),
        physical_b=np.stack(physical_b).astype(np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
    )


class PairDataset(Dataset):
    def __init__(self, split: PreparedSplit, variant: str):
        self.split = split
        self.variant = variant

    def __len__(self) -> int:
        return len(self.split.labels)

    def features(self, side: str) -> np.ndarray:
        rgb = self.split.rgb_a if side == "a" else self.split.rgb_b
        physical = self.split.physical_a if side == "a" else self.split.physical_b
        if self.variant == "rgb_only":
            return rgb
        if self.variant == "physical_only":
            return physical
        if self.variant == "fusion":
            return np.concatenate([rgb, physical], axis=1)
        raise ValueError(self.variant)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        feat_a = self.features("a")[idx]
        feat_b = self.features("b")[idx]
        return {
            "feat_a": torch.from_numpy(feat_a),
            "feat_b": torch.from_numpy(feat_b),
            "label": torch.tensor(self.split.labels[idx], dtype=torch.float32),
            "weight": torch.tensor(self.split.weights[idx], dtype=torch.float32),
        }


class RewardScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pairwise_loss(score_a: torch.Tensor, score_b: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    sign = labels.mul(2.0).sub(1.0)
    loss = nn.functional.softplus(-sign * (score_a - score_b))
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


@torch.no_grad()
def predict_split(model: RewardScorer, dataset: PairDataset, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    scores_a: list[np.ndarray] = []
    scores_b: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        feat_a = batch["feat_a"].to(device)
        feat_b = batch["feat_b"].to(device)
        scores_a.append(model(feat_a).cpu().numpy())
        scores_b.append(model(feat_b).cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(scores_a), np.concatenate(scores_b), np.concatenate(labels)


def metric_rows(rows: list[dict[str, str]], pred_labels: np.ndarray, labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict[str, Any]:
    correct = (pred_labels == labels).astype(np.float32)
    out: dict[str, Any] = {
        "rows": len(rows),
        "accuracy": float(correct.mean()) if len(correct) else 0.0,
        "mean_abs_margin": float(np.abs(scores_a - scores_b).mean()) if len(correct) else 0.0,
    }
    hard_mask = np.asarray([hard_negative_pair(row) for row in rows], dtype=bool)
    if hard_mask.any():
        out["hard_negative_rows"] = int(hard_mask.sum())
        out["hard_negative_accuracy"] = float(correct[hard_mask].mean())
        out["hard_negative_error_rate"] = float(1.0 - correct[hard_mask].mean())
    order_mask = np.asarray([as_bool(row.get("use_for_order_loss_v0", "false")) for row in rows], dtype=bool)
    if order_mask.any():
        out["order_rows"] = int(order_mask.sum())
        out["order_accuracy"] = float(correct[order_mask].mean())
    pref_mask = np.asarray([as_bool(row.get("use_for_preference_loss_v0", "false")) for row in rows], dtype=bool)
    if pref_mask.any():
        out["preference_rows"] = int(pref_mask.sum())
        out["preference_accuracy"] = float(correct[pref_mask].mean())
    return out


def grouped_metrics(
    rows: list[dict[str, str]],
    pred_labels: np.ndarray,
    labels: np.ndarray,
    group_key: str,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[row.get(group_key, "")].append(idx)
    result: dict[str, dict[str, float]] = {}
    for group, indices in sorted(grouped.items()):
        idxs = np.asarray(indices)
        correct = (pred_labels[idxs] == labels[idxs]).astype(np.float32)
        result[group] = {"rows": int(len(indices)), "accuracy": float(correct.mean())}
    return result


def evaluate(
    model: RewardScorer,
    prepared: PreparedSplit,
    variant: str,
    device: str,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = PairDataset(prepared, variant)
    scores_a, scores_b, labels = predict_split(model, dataset, device, batch_size)
    pred_labels = (scores_a >= scores_b).astype(np.float32)
    metrics = metric_rows(prepared.rows, pred_labels, labels, scores_a, scores_b)
    metrics["by_task"] = grouped_metrics(prepared.rows, pred_labels, labels, "task_id")
    metrics["by_pair_type"] = grouped_metrics(prepared.rows, pred_labels, labels, "pair_type")
    metrics["by_fusion_source"] = grouped_metrics(prepared.rows, pred_labels, labels, "fusion_label_source_v0")
    pred_rows: list[dict[str, Any]] = []
    for row, score_a, score_b, label, pred in zip(prepared.rows, scores_a, scores_b, labels, pred_labels):
        pred_rows.append(
            {
                "pair_id": row["pair_id"],
                "task_id": row["task_id"],
                "pair_type": row["pair_type"],
                "final_preference_label_v0": row["final_preference_label_v0"],
                "predicted_label": "A>B" if pred == 1.0 else "B>A",
                "correct": bool(pred == label),
                "score_a": float(score_a),
                "score_b": float(score_b),
                "margin_a_minus_b": float(score_a - score_b),
                "fusion_label_source_v0": row.get("fusion_label_source_v0", ""),
            }
        )
    return metrics, pred_rows


def train_variant(
    variant: str,
    splits: dict[str, PreparedSplit],
    out_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    train_dataset = PairDataset(splits["train"], variant)
    input_dim = train_dataset[0]["feat_a"].numel()
    model = RewardScorer(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    variant_dir = out_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        correct = 0
        total = 0
        pbar = tqdm(loader, desc=f"{variant} epoch {epoch:03d}/{args.epochs}", leave=False)
        for batch in pbar:
            feat_a = batch["feat_a"].to(device)
            feat_b = batch["feat_b"].to(device)
            labels = batch["label"].to(device)
            weights = batch["weight"].to(device)
            optimizer.zero_grad(set_to_none=True)
            score_a = model(feat_a)
            score_b = model(feat_b)
            loss = pairwise_loss(score_a, score_b, labels, weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                pred = (score_a >= score_b).float()
                correct += int((pred == labels).sum().item())
                total += int(labels.numel())
                batch_weight = float(weights.sum().item())
                total_loss += float(loss.item()) * batch_weight
                total_weight += batch_weight
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.3f}")

        train_loss = total_loss / max(total_weight, 1e-6)
        train_acc = correct / max(total, 1)
        val_metrics, _ = evaluate(model, splits["val"], variant, device, args.eval_batch_size)
        val_acc = float(val_metrics["accuracy"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "val_preference_accuracy": val_metrics.get("preference_accuracy", 0.0),
            "val_order_accuracy": val_metrics.get("order_accuracy", 0.0),
            "best_val_accuracy": best_val_acc,
        }
        history.append(row)
        tqdm.write(
            f"{variant} epoch {epoch:03d}: "
            f"loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f} best={best_val_acc:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "variant": variant,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "state_dict": model.state_dict(),
        },
        variant_dir / "best_model.pt",
    )
    write_csv(variant_dir / "train_history.csv", history)

    split_metrics: dict[str, Any] = {}
    for split_name in SPLITS:
        metrics, preds = evaluate(model, splits[split_name], variant, device, args.eval_batch_size)
        split_metrics[split_name] = metrics
        write_csv(variant_dir / f"{split_name}_predictions.csv", preds)
    write_json(variant_dir / "metrics.json", split_metrics)
    return {
        "variant": variant,
        "input_dim": input_dim,
        "best_val_accuracy": best_val_acc,
        "metrics": split_metrics,
        "model_path": str(variant_dir / "best_model.pt"),
    }


def write_summary(path: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Reward Model v0 Results",
        "",
        f"- created_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- split_dir: `{args.split_dir}`",
        f"- feature_dir: `{args.feature_dir}`",
        f"- epochs: {args.epochs}",
        f"- batch_size: {args.batch_size}",
        "",
        "## Overall",
        "",
        "| variant | train acc | val acc | test acc | test preference acc | test order acc | hard-neg test acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        metrics = result["metrics"]
        train = metrics["train"]
        val = metrics["val"]
        test = metrics["test"]
        lines.append(
            f"| {result['variant']} | "
            f"{train['accuracy']:.3f} | {val['accuracy']:.3f} | {test['accuracy']:.3f} | "
            f"{test.get('preference_accuracy', 0.0):.3f} | {test.get('order_accuracy', 0.0):.3f} | "
            f"{test.get('hard_negative_accuracy', 0.0):.3f} |"
        )
    for result in results:
        lines += ["", f"## {result['variant']} By Task", "", "| split | task | rows | accuracy |", "| --- | --- | ---: | ---: |"]
        for split in SPLITS:
            for task, metric in result["metrics"][split]["by_task"].items():
                lines.append(f"| {split} | {task} | {metric['rows']} | {metric['accuracy']:.3f} |")
        lines += ["", f"## {result['variant']} By Pair Type", "", "| split | pair_type | rows | accuracy |", "| --- | --- | ---: | ---: |"]
        for split in SPLITS:
            for pair_type, metric in result["metrics"][split]["by_pair_type"].items():
                lines.append(f"| {split} | {pair_type} | {metric['rows']} | {metric['accuracy']:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reward model v0 from clean pair splits and OpenCLIP features.")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path(r"D:\Users\User\Desktop\reward_model_dataset\dataset_splits\bootstrap_v1_fusion_stereo_v1_clean"),
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path(r"D:\Users\User\Desktop\reward_model_dataset\reward_model_features\openclip_vit_b32_v1"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"D:\Users\User\Desktop\reward_model_dataset\reward_model_runs\reward_model_v0_openclip"),
    )
    parser.add_argument("--variants", nargs="+", default=["rgb_only", "physical_only", "fusion"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"split_dir={args.split_dir}")
    print(f"feature_dir={args.feature_dir}")

    raw_splits = load_splits(args.split_dir)
    clip_features, text_features, clip_index, text_index = load_feature_maps(args.feature_dir)
    physical_mean, physical_std = fit_standardizer(raw_splits["train"])
    prepared = {
        split: prepare_split(raw_splits[split], clip_features, text_features, clip_index, text_index, physical_mean, physical_std)
        for split in SPLITS
    }
    write_json(
        args.out_dir / "run_config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "device": device,
            "split_dir": str(args.split_dir),
            "feature_dir": str(args.feature_dir),
            "variants": args.variants,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "physical_features": list(PHYSICAL_FEATURES),
            "task_ids": list(TASK_IDS),
            "physical_mean": physical_mean.tolist(),
            "physical_std": physical_std.tolist(),
            "split_rows": {split: len(prepared[split].labels) for split in SPLITS},
            "label_distribution": {
                split: dict(Counter(raw["final_preference_label_v0"] for raw in raw_splits[split]))
                for split in SPLITS
            },
        },
    )

    results = []
    for variant in tqdm(args.variants, desc="Training variants"):
        result = train_variant(variant, prepared, args.out_dir, args, device)
        results.append(result)
    write_json(args.out_dir / "all_metrics.json", results)
    write_summary(args.out_dir / "metrics_summary.md", results, args)
    print((args.out_dir / "metrics_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
