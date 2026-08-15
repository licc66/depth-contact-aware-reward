from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


SPLITS = ("train", "val", "test")


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


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def clip_uid(clip_id: str, video_path: str, indices: list[int]) -> str:
    return short_hash(f"{clip_id}|{video_path}|{';'.join(map(str, indices))}")


def text_uid(text: str) -> str:
    return short_hash(text)


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


def cosine_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def collect_clip_specs(
    split_dir: Path,
    pair_table: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    clips: dict[str, dict[str, Any]] = {}
    task_texts: dict[str, str] = {}
    row_groups = (
        [load_csv(pair_table)]
        if pair_table is not None
        else [load_csv(split_dir / f"{split}_pairs.csv") for split in SPLITS]
    )
    for rows in row_groups:
        for row in rows:
            task_texts[text_uid(row["task_goal_text"])] = row["task_goal_text"]
            for side in ("a", "b"):
                indices = parse_indices(row[f"clip_{side}_sample_frame_indices"])
                video_path = (
                    row.get(f"clip_{side}_video_path_local")
                    or row.get(f"clip_{side}_video_path_windows")
                    or ""
                )
                if not video_path:
                    raise KeyError(
                        f"pair {row.get('pair_id')} side {side} has no video path"
                    )
                uid = clip_uid(row[f"clip_{side}_id"], video_path, indices)
                if uid not in clips:
                    clips[uid] = {
                        "clip_uid": uid,
                        "clip_id": row[f"clip_{side}_id"],
                        "task_id": row["task_id"],
                        "video_path": video_path,
                        "sample_frame_indices": ";".join(map(str, indices)),
                        "num_frames": len(indices),
                    }
    ordered_clips = sorted(clips.values(), key=lambda item: (item["task_id"], item["clip_id"], item["clip_uid"]))
    return ordered_clips, task_texts


def encode_frames(
    model: torch.nn.Module,
    preprocess,
    frame_refs: list[tuple[str, int]],
    batch_size: int,
    device: str,
) -> np.ndarray:
    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(frame_refs), batch_size), desc="Encoding frames"):
            refs = frame_refs[start : start + batch_size]
            images = []
            for video_path, frame_index in refs:
                frame = read_frame_rgb(video_path, frame_index)
                images.append(preprocess(Image.fromarray(frame).convert("RGB")))
            batch = torch.stack(images).to(device)
            with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                feats = model.encode_image(batch)
            feats = cosine_normalize(feats.float()).cpu().numpy().astype(np.float32)
            features.append(feats)
    return np.concatenate(features, axis=0)


def encode_texts(model: torch.nn.Module, tokenizer, texts: list[str], batch_size: int, device: str) -> np.ndarray:
    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding task texts"):
            batch_text = texts[start : start + batch_size]
            tokens = tokenizer(batch_text).to(device)
            with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                feats = model.encode_text(tokens)
            feats = cosine_normalize(feats.float()).cpu().numpy().astype(np.float32)
            features.append(feats)
    return np.concatenate(features, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen OpenCLIP embeddings for reward model clips.")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path(r"D:\Users\User\Desktop\reward_model_dataset\dataset_splits\bootstrap_v1_fusion_stereo_v1_clean"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"D:\Users\User\Desktop\reward_model_dataset\reward_model_features\openclip_vit_b32_v1"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="Optional full pair table. Use this to cache clips from conflict/"
        "abstain rows that are absent from the clean split tables.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"D:\Users\User\Desktop\reward_model_dataset\model_cache\openclip\ViT-B-32.pt"),
    )
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=str(args.checkpoint),
        device=device,
        weights_only=False,
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    clip_rows, task_texts = collect_clip_specs(args.split_dir, args.pairs)
    frame_ref_to_idx: dict[tuple[str, int], int] = {}
    frame_refs: list[tuple[str, int]] = []
    clip_to_frame_indices: list[list[int]] = []
    for clip in clip_rows:
        indices = parse_indices(clip["sample_frame_indices"])
        current: list[int] = []
        for idx in indices:
            ref = (clip["video_path"], idx)
            if ref not in frame_ref_to_idx:
                frame_ref_to_idx[ref] = len(frame_refs)
                frame_refs.append(ref)
            current.append(frame_ref_to_idx[ref])
        clip_to_frame_indices.append(current)

    print(f"clips={len(clip_rows)} unique_frame_refs={len(frame_refs)} task_texts={len(task_texts)} device={device}")
    frame_features = encode_frames(model, preprocess, frame_refs, args.batch_size, device)
    clip_features = []
    for frame_indices in clip_to_frame_indices:
        if not frame_indices:
            raise RuntimeError("Empty clip frame indices")
        clip_features.append(frame_features[frame_indices].mean(axis=0))
    clip_features_np = np.stack(clip_features).astype(np.float32)
    clip_features_np /= np.maximum(np.linalg.norm(clip_features_np, axis=1, keepdims=True), 1e-8)

    text_rows = [{"text_uid": uid, "task_goal_text": text} for uid, text in sorted(task_texts.items())]
    text_features_np = encode_texts(
        model,
        tokenizer,
        [row["task_goal_text"] for row in text_rows],
        args.batch_size,
        device,
    )

    for idx, row in enumerate(clip_rows):
        row["clip_feature_index"] = idx
    for idx, row in enumerate(text_rows):
        row["text_feature_index"] = idx

    np.save(args.out_dir / "clip_features.npy", clip_features_np)
    np.save(args.out_dir / "text_features.npy", text_features_np)
    write_csv(args.out_dir / "clip_manifest.csv", clip_rows)
    write_csv(args.out_dir / "task_text_manifest.csv", text_rows)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split_dir": str(args.split_dir),
        "pairs": str(args.pairs) if args.pairs else None,
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "device": device,
        "num_clips": len(clip_rows),
        "num_unique_frame_refs": len(frame_refs),
        "num_task_texts": len(text_rows),
        "clip_feature_dim": int(clip_features_np.shape[1]),
        "text_feature_dim": int(text_features_np.shape[1]),
    }
    (args.out_dir / "feature_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
