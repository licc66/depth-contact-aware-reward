"""Train lightweight StackCube reward-model v2 variants on fused preferences."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reward_model_v2 import (  # noqa: E402
    RewardModelV2,
    RewardModelV2Config,
    pairwise_preference_loss,
    save_checkpoint,
    temporal_order_loss,
    weighted_potential_loss,
    weighted_stage_loss,
)


TRAIN_SPLITS = ("train", "val")
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


def macro_f1(prediction: torch.Tensor, target: torch.Tensor, classes: int = 4) -> float:
    values = []
    for cls in range(classes):
        pred = prediction.eq(cls)
        true = target.eq(cls)
        tp = (pred & true).sum().item()
        fp = (pred & ~true).sum().item()
        fn = (~pred & true).sum().item()
        denominator = 2 * tp + fp + fn
        if denominator:
            values.append(2 * tp / denominator)
    return float(np.mean(values)) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--physical-scores", type=Path, required=True)
    parser.add_argument("--physical-embeddings", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=("rgb_only", "physical_only", "fusion"),
        default=["rgb_only", "physical_only", "fusion"]
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--preference-temperature", type=float, default=0.10)
    parser.add_argument("--preference-loss-weight", type=float, default=0.50)
    parser.add_argument("--order-loss-weight", type=float, default=0.15)
    parser.add_argument("--stage-loss-weight", type=float, default=0.20)
    parser.add_argument("--potential-loss-weight", type=float, default=0.50)
    parser.add_argument("--physical-dropout", type=float, default=0.15)
    parser.add_argument("--rgb-dropout", type=float, default=0.10)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fusion_manifest = json.loads(
        (args.fusion_dir / "fusion_labels_v2_manifest.json").read_text(encoding="utf-8")
    )
    fusion_rows = {
        split: read_csv(args.fusion_dir / f"{split}_pairs_v2.csv")
        for split in TRAIN_SPLITS
    }
    pair_meta = {row["pair_id"]: row for row in read_csv(args.pair_table)}
    physical_by_pair = {
        row["pair_id"]: row for row in read_csv(args.physical_scores)
    }

    clip_features = np.load(args.feature_dir / "clip_features.npy").astype(np.float32)
    text_features = np.load(args.feature_dir / "text_features.npy").astype(np.float32)
    clip_index: dict[str, int] = {}
    for row in read_csv(args.feature_dir / "clip_manifest.csv"):
        clip_id = row["clip_id"]
        if clip_id in clip_index:
            raise ValueError(f"duplicate OpenCLIP clip_id {clip_id}")
        clip_index[clip_id] = integer(row["clip_feature_index"])
    text_index = {
        row["task_goal_text"]: integer(row["text_feature_index"])
        for row in read_csv(args.feature_dir / "task_text_manifest.csv")
    }
    physical_archive = np.load(args.physical_embeddings)
    embedding_ids = [str(value) for value in physical_archive["clip_ids"].tolist()]
    embedding_values = physical_archive["embeddings"].astype(np.float32)
    physical_embedding = {
        clip_id: embedding_values[index] for index, clip_id in enumerate(embedding_ids)
    }

    def rgb_feature(meta: dict[str, str], side: str) -> np.ndarray:
        clip_id = meta[f"clip_{side}_id"]
        if clip_id not in clip_index:
            raise KeyError(f"OpenCLIP cache missing {clip_id}")
        image = clip_features[clip_index[clip_id]]
        text = text_features[text_index[meta["task_goal_text"]]]
        return np.concatenate([image, text, image * text], axis=0).astype(np.float32)

    def phys_feature(meta: dict[str, str], side: str) -> np.ndarray:
        clip_id = meta[f"clip_{side}_id"]
        if clip_id not in physical_embedding:
            raise KeyError(f"physical embedding cache missing {clip_id}")
        return physical_embedding[clip_id]

    def build_split(split: str) -> dict[str, Any]:
        joined = []
        for fusion in fusion_rows[split]:
            pair_id = fusion["pair_id"]
            if pair_id not in pair_meta or pair_id not in physical_by_pair:
                raise RuntimeError(f"incomplete reward-model inputs for {pair_id}")
            joined.append((fusion, pair_meta[pair_id], physical_by_pair[pair_id]))
        clip_occurrences = Counter(
            meta[f"clip_{side}_id"]
            for _, meta, _ in joined
            for side in ("a", "b")
        )
        result: dict[str, list[Any]] = {
            key: []
            for key in (
                "rows", "rgb_a", "rgb_b", "phys_a", "phys_b", "labels", "weights",
                "order_mask", "order_a_early", "target_potential_a", "target_potential_b",
                "target_stage_a", "target_stage_b", "reference_potential_a",
                "reference_potential_b", "reference_stage_a", "reference_stage_b",
                "point_weight_a", "point_weight_b",
            )
        }
        for fusion, meta, physical in joined:
            label = fusion.get("fusion_label_v2", "")
            use_preference = truthy(fusion.get("use_for_preference_loss_v2"))
            result["rows"].append({**fusion, "pair_type": meta.get("pair_type", "")})
            result["rgb_a"].append(rgb_feature(meta, "a"))
            result["rgb_b"].append(rgb_feature(meta, "b"))
            result["phys_a"].append(phys_feature(meta, "a"))
            result["phys_b"].append(phys_feature(meta, "b"))
            result["labels"].append(1.0 if label == "A>B" else 0.0)
            result["weights"].append(
                number(fusion.get("fusion_weight_v2"), 0.0)
                if use_preference and label in VALID_LABELS
                else 0.0
            )
            is_temporal = meta.get("pair_type") == "intra_success_temporal_gap"
            start_a = integer(meta.get("clip_a_start_frame"))
            start_b = integer(meta.get("clip_b_start_frame"))
            result["order_mask"].append(is_temporal and start_a != start_b)
            result["order_a_early"].append(start_a < start_b)
            for side in ("a", "b"):
                result[f"target_potential_{side}"].append(
                    number(physical.get(f"phys_{side}_potential"), 0.0)
                )
                result[f"target_stage_{side}"].append(
                    integer(physical.get(f"phys_{side}_stage"), 1) - 1
                )
                result[f"reference_potential_{side}"].append(
                    number(meta.get(f"reference_{side}_potential_v2"), 0.0)
                )
                result[f"reference_stage_{side}"].append(
                    integer(meta.get(f"reference_{side}_stage_v2"), 1) - 1
                )
                clip_id = meta[f"clip_{side}_id"]
                result[f"point_weight_{side}"].append(1.0 / clip_occurrences[clip_id])
        arrays: dict[str, Any] = {"rows": result["rows"]}
        for key in ("rgb_a", "rgb_b", "phys_a", "phys_b"):
            arrays[key] = np.stack(result[key]).astype(np.float32)
        for key in (
            "labels", "weights", "target_potential_a", "target_potential_b",
            "reference_potential_a", "reference_potential_b", "point_weight_a",
            "point_weight_b",
        ):
            arrays[key] = np.asarray(result[key], dtype=np.float32)
        for key in ("target_stage_a", "target_stage_b", "reference_stage_a", "reference_stage_b"):
            arrays[key] = np.asarray(result[key], dtype=np.int64)
        for key in ("order_mask", "order_a_early"):
            arrays[key] = np.asarray(result[key], dtype=bool)
        return arrays

    prepared = {split: build_split(split) for split in TRAIN_SPLITS}
    phys_train = np.concatenate(
        [prepared["train"]["phys_a"], prepared["train"]["phys_b"]], axis=0
    )
    phys_mean = phys_train.mean(axis=0)
    phys_std = phys_train.std(axis=0)
    phys_std = np.where(phys_std < 1e-6, 1.0, phys_std)
    for split in TRAIN_SPLITS:
        for key in ("phys_a", "phys_b"):
            prepared[split][key] = (prepared[split][key] - phys_mean) / phys_std

    def tensors(data: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(value)
            for key, value in data.items()
            if key != "rows"
        }

    tensor_data = {split: tensors(prepared[split]) for split in TRAIN_SPLITS}

    def evaluate(model: RewardModelV2, split: str) -> dict[str, float]:
        model.eval()
        data = tensor_data[split]
        with torch.inference_mode():
            out_a = model(data["rgb_a"].to(device), data["phys_a"].to(device))
            out_b = model(data["rgb_b"].to(device), data["phys_b"].to(device))
        phi_a = out_a["potential"].cpu()
        phi_b = out_b["potential"].cpu()
        preference_mask = data["weights"].gt(0)
        predicted_a = phi_a.ge(phi_b)
        fused_target_a = data["labels"].ge(0.5)
        pair_accuracy = (
            float(predicted_a[preference_mask].eq(fused_target_a[preference_mask]).float().mean())
            if preference_mask.any()
            else 0.0
        )
        rows = prepared[split]["rows"]
        reference_mask = torch.tensor(
            [row["reference_label_v2"] in VALID_LABELS for row in rows], dtype=torch.bool
        )
        reference_a = torch.tensor(
            [row["reference_label_v2"] == "A>B" for row in rows], dtype=torch.bool
        )
        reference_pair_accuracy = (
            float(predicted_a[reference_mask].eq(reference_a[reference_mask]).float().mean())
            if reference_mask.any()
            else 0.0
        )
        order = data["order_mask"]
        if order.any():
            early = torch.where(data["order_a_early"][order], phi_a[order], phi_b[order])
            late = torch.where(data["order_a_early"][order], phi_b[order], phi_a[order])
            order_accuracy = float(late.ge(early).float().mean())
        else:
            order_accuracy = 0.0
        reference_potential = torch.cat(
            [data["reference_potential_a"], data["reference_potential_b"]]
        )
        predicted_potential = torch.cat([phi_a, phi_b])
        potential_mae = float((predicted_potential - reference_potential).abs().mean())
        predicted_stage = torch.cat(
            [out_a["stage_logits"].argmax(dim=-1).cpu(), out_b["stage_logits"].argmax(dim=-1).cpu()]
        )
        reference_stage = torch.cat([data["reference_stage_a"], data["reference_stage_b"]])
        stage_f1 = macro_f1(predicted_stage, reference_stage)
        score = (
            0.35 * reference_pair_accuracy
            + 0.20 * order_accuracy
            + 0.25 * max(0.0, 1.0 - potential_mae)
            + 0.20 * stage_f1
        )
        return {
            "fused_pair_accuracy": pair_accuracy,
            "reference_pair_accuracy": reference_pair_accuracy,
            "order_accuracy": order_accuracy,
            "reference_potential_mae": potential_mae,
            "reference_stage_macro_f1": stage_f1,
            "selection_score": score,
            "mean_rgb_gate": float(
                torch.cat([out_a["gate_rgb_weight"].cpu(), out_b["gate_rgb_weight"].cpu()]).mean()
            ),
        }

    results: dict[str, Any] = {}
    for variant in args.variants:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        config = RewardModelV2Config(
            variant=variant,
            rgb_dim=int(prepared["train"]["rgb_a"].shape[1]),
            physical_dim=int(prepared["train"]["phys_a"].shape[1]),
            physical_input_kind="embedding",
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            preference_temperature=args.preference_temperature,
            physical_feature_contract=("physical_progress_v2_clip_embedding",),
        )
        model = RewardModelV2(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        train = tensor_data["train"]
        n = train["labels"].shape[0]
        best_score = -float("inf")
        best_state = None
        stale_epochs = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            permutation = torch.randperm(n)
            total_loss = 0.0
            for start in range(0, n, args.batch_size):
                indices = permutation[start : start + args.batch_size]
                rgb_a = train["rgb_a"][indices].float().to(device)
                rgb_b = train["rgb_b"][indices].float().to(device)
                phys_a = train["phys_a"][indices].float().to(device)
                phys_b = train["phys_b"][indices].float().to(device)
                batch = len(indices)
                rgb_valid = torch.ones(batch, device=device)
                phys_valid = torch.ones(batch, device=device)
                if variant == "fusion":
                    rgb_valid = (torch.rand(batch, device=device) >= args.rgb_dropout).float()
                    phys_valid = (
                        torch.rand(batch, device=device) >= args.physical_dropout
                    ).float()
                    both_missing = rgb_valid.add(phys_valid).eq(0)
                    rgb_valid = torch.where(both_missing, torch.ones_like(rgb_valid), rgb_valid)
                out_a = model(rgb_a, phys_a, rgb_valid, phys_valid)
                out_b = model(rgb_b, phys_b, rgb_valid, phys_valid)
                loss = args.preference_loss_weight * pairwise_preference_loss(
                    out_a["potential"],
                    out_b["potential"],
                    train["labels"][indices].float().to(device),
                    train["weights"][indices].float().to(device),
                    args.preference_temperature,
                )
                order = train["order_mask"][indices].to(device)
                if order.any():
                    a_early = train["order_a_early"][indices].to(device)[order]
                    early = torch.where(
                        a_early, out_a["potential"][order], out_b["potential"][order]
                    )
                    late = torch.where(
                        a_early, out_b["potential"][order], out_a["potential"][order]
                    )
                    loss = loss + args.order_loss_weight * temporal_order_loss(early, late)
                point_weight_a = train["point_weight_a"][indices].float().to(device)
                point_weight_b = train["point_weight_b"][indices].float().to(device)
                loss = loss + args.potential_loss_weight * 0.5 * (
                    weighted_potential_loss(
                        out_a["potential"],
                        train["target_potential_a"][indices].float().to(device),
                        point_weight_a,
                    )
                    + weighted_potential_loss(
                        out_b["potential"],
                        train["target_potential_b"][indices].float().to(device),
                        point_weight_b,
                    )
                )
                loss = loss + args.stage_loss_weight * 0.5 * (
                    weighted_stage_loss(
                        out_a["stage_logits"],
                        train["target_stage_a"][indices].long().to(device),
                        point_weight_a,
                    )
                    + weighted_stage_loss(
                        out_b["stage_logits"],
                        train["target_stage_b"][indices].long().to(device),
                        point_weight_b,
                    )
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.item()) * batch
            val_metrics = evaluate(model, "val")
            history.append(
                {"epoch": epoch, "train_loss": total_loss / max(1, n), **val_metrics}
            )
            score = val_metrics["selection_score"]
            if score > best_score + 1e-5:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"{variant} epoch={epoch:03d} loss={history[-1]['train_loss']:.4f} "
                    f"val_ref_pair={val_metrics['reference_pair_accuracy']:.3f} "
                    f"val_mae={val_metrics['reference_potential_mae']:.3f} "
                    f"score={score:.3f} best={best_score:.3f}"
                )
            if stale_epochs >= args.early_stopping_patience:
                break
        if best_state is None:
            raise RuntimeError(f"no checkpoint selected for {variant}")
        model.load_state_dict(best_state)
        checkpoint = args.out_dir / f"reward_model_v2_{variant}.pt"
        save_checkpoint(
            model,
            checkpoint,
            extra={
                "physical_mean": phys_mean.tolist(),
                "physical_std": phys_std.tolist(),
                "fusion_dir": str(args.fusion_dir),
                "feature_dir": str(args.feature_dir),
                "physical_scores": str(args.physical_scores),
                "physical_embeddings": str(args.physical_embeddings),
                "semantic_source": fusion_manifest["semantic_source"],
                "primary_scientific_result": fusion_manifest[
                    "primary_scientific_result"
                ],
                "seed": args.seed,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        write_csv(args.out_dir / f"train_history_{variant}.csv", history)
        results[variant] = {
            "checkpoint": str(checkpoint),
            "parameters": model.parameter_count(),
            "best_val_selection_score": best_score,
            "train": evaluate(model, "train"),
            "val": evaluate(model, "val"),
        }
        print(f"{variant}: params={model.parameter_count():,} best={best_score:.4f}")

    run_config = {
        "schema_version": "reward_model_training_v2.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "args": {key: str(value) for key, value in vars(args).items()},
        "semantic_source": fusion_manifest["semantic_source"],
        "primary_scientific_result": fusion_manifest["primary_scientific_result"],
        "test_split_touched": False,
        "row_counts": {split: len(prepared[split]["rows"]) for split in TRAIN_SPLITS},
        "fusion_label_counts": {
            split: dict(Counter(row["fusion_label_v2"] for row in prepared[split]["rows"]))
            for split in TRAIN_SPLITS
        },
        "results": results,
    }
    write_json(args.out_dir / "run_config.json", run_config)
    write_json(args.out_dir / "metrics_trainval.json", results)
    print(f"wrote {args.out_dir}; test split untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
