"""Train reward model v1 variants (Phase 4).

Trains ``rgb_only``, ``physical_only``, and ``fusion`` under the same splits,
seed, and budget. Supervision comes from the calibrated fusion labels built by
``27_build_fusion_labels_v1.py`` (abstain rows carry zero preference weight
and are excluded from the preference loss). The temporal order loss uses only
clean-success temporal pairs, with direction taken from clip start frames —
an offline dataset fact, not a model input. Optional stage auxiliary targets
come from the pair table's ``contact_clip_*_end_stage_id`` columns and are
TRAINING-ONLY targets (never features).

Model inputs are strictly: frozen OpenCLIP conditioned features (from the
script-20 cache) and frozen physical-branch outputs (script-26 export:
9-d observable summary, or 128-d clip embedding with --physical-input-kind
embedding). Teacher/candidate/fusion labels are never features; the deny list
is asserted on the feature contract at model construction.

Checkpoint selection uses validation pair accuracy only. The test split is
neither loaded nor evaluated here; use 29_evaluate_reward_model_v1.py.

Requires torch; exits with a precise dependency report (code 3) otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reward_common_v1 import (  # noqa: E402
    VALID_LABELS,
    as_float,
    default_split_dir,
    load_csv,
    parse_indices,
    write_csv,
    write_json,
)

TASK_IDS = ("peginsertion", "stackcube", "stackpyramid")
TRAIN_SPLITS = ("train", "val")


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def clip_uid(clip_id: str, video_path: str, indices: list[int]) -> str:
    return short_hash(f"{clip_id}|{video_path}|{';'.join(map(str, indices))}")


def text_uid(text: str) -> str:
    return short_hash(text)


def task_onehot(task_id: str) -> list[float]:
    return [1.0 if task_id == task else 0.0 for task in TASK_IDS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fusion-dir", type=Path, required=True, help="output dir of script 27")
    parser.add_argument("--split-dir", type=Path, default=default_split_dir())
    parser.add_argument(
        "--pair-table",
        type=Path,
        default=None,
        help="Full pair metadata table used by scripts 26/27. Required to retain "
        "conflict rows that were rejected from the clean split tables.",
    )
    parser.add_argument("--feature-dir", type=Path, required=True, help="OpenCLIP cache from script 20")
    parser.add_argument("--physical-scores", type=Path, required=True, help="CSV from script 26")
    parser.add_argument(
        "--physical-embeddings",
        type=Path,
        default=None,
        help="npz from script 26 --export-embeddings (needed for --physical-input-kind embedding)",
    )
    parser.add_argument("--physical-input-kind", choices=("summary", "embedding"), default="summary")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["rgb_only", "physical_only", "fusion"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--preference-temperature", type=float, default=0.10)
    parser.add_argument("--order-loss-weight", type=float, default=0.20)
    parser.add_argument("--stage-loss-weight", type=float, default=0.10)
    parser.add_argument("--physical-dropout", type=float, default=0.15)
    parser.add_argument("--rgb-dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
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

    from reward_model_v1 import (
        RewardModelV1,
        RewardModelV1Config,
        pairwise_preference_loss,
        save_checkpoint,
        stage_auxiliary_loss,
        temporal_order_loss,
        PHYSICAL_SUMMARY_FEATURES,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load tables (train/val ONLY — test is untouched here by design)
    # ------------------------------------------------------------------
    fusion_rows = {
        split: load_csv(args.fusion_dir / f"{split}_pairs_v1.csv") for split in TRAIN_SPLITS
    }
    metadata_rows: list[dict[str, str]] = []
    if args.pair_table is not None:
        metadata_rows = load_csv(args.pair_table)
    else:
        for split in TRAIN_SPLITS:
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

    required_pair_ids = {
        row["pair_id"] for split in TRAIN_SPLITS for row in fusion_rows[split]
    }
    missing_meta = sorted(required_pair_ids - pair_meta.keys())
    missing_physical = sorted(required_pair_ids - physical_by_pair.keys())
    if missing_meta or missing_physical:
        raise RuntimeError(
            "reward training inputs are incomplete: "
            f"missing_pair_metadata={len(missing_meta)} "
            f"missing_physical_scores={len(missing_physical)}. "
            "Use the same full --pair-table for scripts 26, 27, and 28."
        )

    clip_features = np.load(args.feature_dir / "clip_features.npy").astype(np.float32)
    text_features = np.load(args.feature_dir / "text_features.npy").astype(np.float32)
    clip_index = {
        row["clip_uid"]: int(row["clip_feature_index"])
        for row in load_csv(args.feature_dir / "clip_manifest.csv")
    }
    text_index = {
        row["text_uid"]: int(row["text_feature_index"])
        for row in load_csv(args.feature_dir / "task_text_manifest.csv")
    }
    embeddings = None
    if args.physical_input_kind == "embedding":
        if args.physical_embeddings is None:
            print("ERROR: --physical-embeddings required for embedding input kind", file=sys.stderr)
            return 2
        embeddings = np.load(args.physical_embeddings)

    def rgb_feature(meta: dict[str, str], side: str) -> np.ndarray:
        indices = parse_indices(meta[f"clip_{side}_sample_frame_indices"])
        uid = clip_uid(meta[f"clip_{side}_id"], meta[f"clip_{side}_video_path_local"], indices)
        image = clip_features[clip_index[uid]]
        text = text_features[text_index[text_uid(meta["task_goal_text"])]]
        return np.concatenate([image, text, image * text], axis=0).astype(np.float32)

    def physical_summary(phys: dict[str, str], side: str) -> np.ndarray:
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
            + task_onehot(phys.get("task_id", "")),
            dtype=np.float32,
        )

    def physical_embedding(phys: dict[str, str], side: str) -> np.ndarray:
        uid = phys.get(f"clip_{side}_embedding_uid", "")
        if not uid or uid not in embeddings:
            raise KeyError(
                f"missing physical embedding uid for pair {phys.get('pair_id')} side {side}; "
                "rerun 26 with --export-embeddings"
            )
        return np.asarray(embeddings[uid], dtype=np.float32)

    physical_fn = (
        physical_summary if args.physical_input_kind == "summary" else physical_embedding
    )
    physical_dim = (
        len(PHYSICAL_SUMMARY_FEATURES) + len(TASK_IDS)
        if args.physical_input_kind == "summary"
        else int(next(iter(embeddings.values())).shape[0])  # type: ignore[union-attr]
    )

    def build_split(split: str) -> dict[str, Any]:
        rgb_a, rgb_b, phys_a, phys_b = [], [], [], []
        labels, weights = [], []
        order_mask, order_a_is_early = [], []
        stage_a, stage_b = [], []
        kept_rows = []
        for row in fusion_rows[split]:
            pair_id = row["pair_id"]
            meta = pair_meta.get(pair_id)
            phys = physical_by_pair.get(pair_id)
            if meta is None or phys is None:
                raise RuntimeError(f"missing joined inputs for pair {pair_id}")
            label = row.get("fusion_label_v1", "")
            use_pref = str(row.get("use_for_preference_loss_v1", "")).lower() == "true"
            is_temporal = meta.get("pair_type", "") == "intra_success_temporal_gap"
            if label not in VALID_LABELS and not is_temporal:
                continue  # abstain, and no order signal either
            kept_rows.append({**row, "pair_type": meta.get("pair_type", "")})
            rgb_a.append(rgb_feature(meta, "a"))
            rgb_b.append(rgb_feature(meta, "b"))
            phys_a.append(physical_fn(phys, "a"))
            phys_b.append(physical_fn(phys, "b"))
            labels.append(1.0 if label == "A>B" else 0.0)
            weights.append(
                as_float(row.get("fusion_weight_v1"), 0.0)
                if (use_pref and label in VALID_LABELS)
                else 0.0
            )
            start_a = as_float(meta.get("clip_a_start_frame"), 0.0)
            start_b = as_float(meta.get("clip_b_start_frame"), 0.0)
            order_mask.append(is_temporal and start_a != start_b)
            order_a_is_early.append(start_a < start_b)
            stage_a.append(int(as_float(meta.get("contact_clip_a_end_stage_id"), 0.0)) - 1)
            stage_b.append(int(as_float(meta.get("contact_clip_b_end_stage_id"), 0.0)) - 1)
        if not kept_rows:
            raise RuntimeError(f"no usable rows in fusion split {split}")
        return {
            "rows": kept_rows,
            "rgb_a": np.stack(rgb_a),
            "rgb_b": np.stack(rgb_b),
            "phys_a": np.stack(phys_a),
            "phys_b": np.stack(phys_b),
            "labels": np.asarray(labels, dtype=np.float32),
            "weights": np.asarray(weights, dtype=np.float32),
            "order_mask": np.asarray(order_mask, dtype=bool),
            "order_a_is_early": np.asarray(order_a_is_early, dtype=bool),
            "stage_a": np.asarray(stage_a, dtype=np.int64),
            "stage_b": np.asarray(stage_b, dtype=np.int64),
        }

    prepared = {split: build_split(split) for split in TRAIN_SPLITS}

    # Standardize physical inputs on train only.
    phys_train = np.concatenate([prepared["train"]["phys_a"], prepared["train"]["phys_b"]], axis=0)
    phys_mean = phys_train.mean(axis=0)
    phys_std = np.where(phys_train.std(axis=0) < 1e-6, 1.0, phys_train.std(axis=0))
    for split in TRAIN_SPLITS:
        for key in ("phys_a", "phys_b"):
            prepared[split][key] = (prepared[split][key] - phys_mean) / phys_std

    def to_tensors(split: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            "rgb_a": torch.from_numpy(split["rgb_a"]).float(),
            "rgb_b": torch.from_numpy(split["rgb_b"]).float(),
            "phys_a": torch.from_numpy(split["phys_a"]).float(),
            "phys_b": torch.from_numpy(split["phys_b"]).float(),
            "labels": torch.from_numpy(split["labels"]).float(),
            "weights": torch.from_numpy(split["weights"]).float(),
            "order_mask": torch.from_numpy(split["order_mask"]),
            "order_a_is_early": torch.from_numpy(split["order_a_is_early"]),
            "stage_a": torch.from_numpy(split["stage_a"]),
            "stage_b": torch.from_numpy(split["stage_b"]),
        }

    tensors = {split: to_tensors(prepared[split]) for split in TRAIN_SPLITS}

    def evaluate_variant(model: RewardModelV1, split: str) -> dict[str, float]:
        model.eval()
        data = tensors[split]
        with torch.no_grad():
            out_a = model(data["rgb_a"].to(device), data["phys_a"].to(device))
            out_b = model(data["rgb_b"].to(device), data["phys_b"].to(device))
            phi_a = out_a["potential"].cpu()
            phi_b = out_b["potential"].cpu()
        pref = data["weights"] > 0
        predicted = (phi_a >= phi_b).float()
        accuracy = float((predicted[pref] == data["labels"][pref]).float().mean()) if pref.any() else float("nan")
        order = data["order_mask"]
        if order.any():
            early = torch.where(data["order_a_is_early"][order], phi_a[order], phi_b[order])
            late = torch.where(data["order_a_is_early"][order], phi_b[order], phi_a[order])
            order_accuracy = float((late >= early).float().mean())
        else:
            order_accuracy = float("nan")
        return {"pair_accuracy": accuracy, "order_accuracy": order_accuracy}

    results: dict[str, Any] = {}
    for variant in args.variants:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        config = RewardModelV1Config(
            variant=variant,
            rgb_dim=int(prepared["train"]["rgb_a"].shape[1]),
            physical_dim=physical_dim,
            physical_input_kind=args.physical_input_kind,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            preference_temperature=args.preference_temperature,
        )
        model = RewardModelV1(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train = tensors["train"]
        n = train["labels"].shape[0]
        best_val = -1.0
        best_state = None
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            permutation = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, args.batch_size):
                batch_idx = permutation[start : start + args.batch_size]
                rgb_a = train["rgb_a"][batch_idx].to(device)
                rgb_b = train["rgb_b"][batch_idx].to(device)
                phys_a = train["phys_a"][batch_idx].to(device)
                phys_b = train["phys_b"][batch_idx].to(device)
                bsz = rgb_a.shape[0]
                rgb_valid = torch.ones(bsz, device=device)
                phys_valid = torch.ones(bsz, device=device)
                if variant == "fusion":
                    phys_valid = (torch.rand(bsz, device=device) >= args.physical_dropout).float()
                    rgb_valid = (torch.rand(bsz, device=device) >= args.rgb_dropout).float()
                    # never drop both modalities of the same sample
                    both_dropped = (phys_valid + rgb_valid) == 0
                    rgb_valid = torch.where(both_dropped, torch.ones_like(rgb_valid), rgb_valid)
                out_a = model(rgb_a, phys_a, rgb_valid, phys_valid)
                out_b = model(rgb_b, phys_b, rgb_valid, phys_valid)
                loss = pairwise_preference_loss(
                    out_a["potential"],
                    out_b["potential"],
                    train["labels"][batch_idx].to(device),
                    train["weights"][batch_idx].to(device),
                    args.preference_temperature,
                )
                order = train["order_mask"][batch_idx].to(device)
                if order.any():
                    a_early = train["order_a_is_early"][batch_idx].to(device)[order]
                    phi_a = out_a["potential"][order]
                    phi_b = out_b["potential"][order]
                    early = torch.where(a_early, phi_a, phi_b)
                    late = torch.where(a_early, phi_b, phi_a)
                    loss = loss + args.order_loss_weight * temporal_order_loss(early, late)
                if args.stage_loss_weight > 0 and "stage_logits" in out_a:
                    loss = loss + args.stage_loss_weight * 0.5 * (
                        stage_auxiliary_loss(out_a["stage_logits"], train["stage_a"][batch_idx].to(device))
                        + stage_auxiliary_loss(out_b["stage_logits"], train["stage_b"][batch_idx].to(device))
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_loss += float(loss.item()) * bsz
            val_metrics = evaluate_variant(model, "val")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss / max(n, 1),
                    "val_pair_accuracy": val_metrics["pair_accuracy"],
                    "val_order_accuracy": val_metrics["order_accuracy"],
                }
            )
            if val_metrics["pair_accuracy"] > best_val:
                best_val = val_metrics["pair_accuracy"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        assert best_state is not None
        model.load_state_dict(best_state)
        checkpoint_path = args.out_dir / f"reward_model_v1_{variant}.pt"
        save_checkpoint(
            model,
            checkpoint_path,
            extra={
                "physical_mean": phys_mean.tolist(),
                "physical_std": phys_std.tolist(),
                "physical_input_kind": args.physical_input_kind,
                "fusion_dir": str(args.fusion_dir),
                "feature_dir": str(args.feature_dir),
                "physical_scores": str(args.physical_scores),
                "seed": args.seed,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        write_csv(args.out_dir / f"train_history_{variant}.csv", history)
        results[variant] = {
            "checkpoint": str(checkpoint_path),
            "parameters": model.parameter_count(),
            "best_val_pair_accuracy": best_val,
            "train": evaluate_variant(model, "train"),
            "val": evaluate_variant(model, "val"),
        }
        print(
            f"{variant}: params={results[variant]['parameters']:,} "
            f"best_val_pair_acc={best_val:.4f}"
        )

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "args": {k: str(v) for k, v in vars(args).items()},
        "row_counts": {split: len(prepared[split]["rows"]) for split in TRAIN_SPLITS},
        "preference_rows": {
            split: int((prepared[split]["weights"] > 0).sum()) for split in TRAIN_SPLITS
        },
        "order_rows": {
            split: int(prepared[split]["order_mask"].sum()) for split in TRAIN_SPLITS
        },
        "label_distribution": {
            split: dict(Counter(row["fusion_label_v1"] for row in prepared[split]["rows"]))
            for split in TRAIN_SPLITS
        },
        "test_split_touched": False,
        "results": results,
    }
    write_json(args.out_dir / "run_config.json", run_config)
    write_json(args.out_dir / "metrics_trainval.json", {v: results[v] for v in results})
    print(f"wrote {args.out_dir}/run_config.json (test split untouched; use script 29)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
