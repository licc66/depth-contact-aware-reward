from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def default_cache_dir() -> Path:
    return Path(r"D:\Users\User\Desktop\reward_model_dataset\model_cache\openclip")


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify OpenCLIP weights in a fixed local cache.")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument(
        "--source",
        choices=("direct", "hf"),
        default="direct",
        help="direct uses the pretrained cfg URL with prefer_hf_hub=False; hf uses HuggingFace cache.",
    )
    parser.add_argument("--hf-endpoint", default="", help="Optional HuggingFace endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-load-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    import torch
    import open_clip
    from open_clip.pretrained import download_pretrained, get_pretrained_cfg

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path: str | None = None
    if args.source == "direct":
        pretrained_cfg = get_pretrained_cfg(args.model, args.pretrained)
        if not pretrained_cfg:
            raise RuntimeError(f"No pretrained config for {args.model}/{args.pretrained}")
        checkpoint_path = download_pretrained(
            pretrained_cfg,
            prefer_hf_hub=False,
            cache_dir=str(args.cache_dir),
        )
        model_pretrained_arg = checkpoint_path
    else:
        model_pretrained_arg = args.pretrained

    if not args.no_load_test:
        model, _, _ = open_clip.create_model_and_transforms(
            args.model,
            pretrained=model_pretrained_arg,
            cache_dir=str(args.cache_dir),
            device=device,
            weights_only=False,
        )
        model.eval()
        visual_output_dim = getattr(model.visual, "output_dim", None)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        visual_output_dim = None

    if checkpoint_path is None:
        # HF cache path is managed by huggingface_hub. Keep cache_dir in manifest;
        # future scripts can pass the same cache_dir and will reuse local files.
        checkpoint = {"path": "", "exists": False, "size_mb": 0.0}
    else:
        checkpoint = file_info(Path(checkpoint_path))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "pretrained": args.pretrained,
        "source": args.source,
        "cache_dir": str(args.cache_dir),
        "checkpoint": checkpoint,
        "device_tested": device if not args.no_load_test else "skipped",
        "torch_cuda_available": torch.cuda.is_available(),
        "visual_output_dim": visual_output_dim,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "how_to_reuse": {
            "direct": "Pass pretrained=manifest['checkpoint']['path'] to open_clip.create_model_and_transforms.",
            "hf": "Pass pretrained=manifest['pretrained'] and cache_dir=manifest['cache_dir'].",
        },
    }
    manifest_path = args.cache_dir / f"{args.model}_{args.pretrained}_{args.source}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
