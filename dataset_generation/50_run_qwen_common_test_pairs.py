#!/usr/bin/env python3
"""Label a blinded common-pair benchmark with Qwen3.7-Plus."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


VALID_LABELS = {"A>B", "B>A", "unsure"}
VALID_CONFIDENCE = {0.55, 0.70, 0.85, 0.95}
SYSTEM_PROMPT = (
    "You are a blinded visual-semantic preference annotator for robotic manipulation. "
    "Judge only visible RGB evidence and chronological order. Do not use filenames or "
    "hidden metadata. Do not infer contact, force, depth, release, or stability unless "
    "visibly supported. If the ordering is ambiguous, answer unsure. Return one strict "
    "JSON object with no markdown or extra prose."
)
USER_PROMPT = """Task: StackCube. The robot must place the movable red cube on the green support cube, release it, and leave it stably supported.

The image is chronological: TOP ROW is Clip A (A1 to A6, left to right); BOTTOM ROW is Clip B (B1 to B6, left to right).

Decide which clip shows more valid task progress. Use A>B, B>A, or unsure. Consider purposeful approach, credible grasp/hold, transport and alignment, placement, visible release, and stable support. A visually close near-miss is not completion. Dropping, slipping, collision, moving away, unstable placement, failed release, or losing the cube is regression. Within the same broad stage, prefer clearer controlled progress only when it is visibly supported.

Confidence must be one of 0.55, 0.70, 0.85, 0.95; use 0.0 for unsure.

Return exactly:
{"pair_id":"PAIR_ID","preference":"A>B|B>A|unsure","confidence":0.0,"reason":"one concise sentence using visible evidence only","visible_uncertainty":"concise text or empty string"}

The exact pair_id is: PAIR_ID"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def validate_label(label: dict[str, Any], pair_id: str) -> dict[str, Any]:
    required = {"pair_id", "preference", "confidence", "reason", "visible_uncertainty"}
    missing = required - label.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if label["pair_id"] != pair_id:
        raise ValueError(f"pair_id mismatch: {label['pair_id']!r} != {pair_id!r}")
    if label["preference"] not in VALID_LABELS:
        raise ValueError(f"invalid preference: {label['preference']!r}")
    confidence = float(label["confidence"])
    if label["preference"] == "unsure" and confidence != 0.0:
        raise ValueError("unsure must use confidence 0.0")
    if label["preference"] != "unsure" and confidence not in VALID_CONFIDENCE:
        raise ValueError(f"invalid confidence: {confidence!r}")
    label["confidence"] = confidence
    return label


def payload(model: str, pair_id: str, image_path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                    {"type": "text", "text": USER_PROMPT.replace("PAIR_ID", pair_id)},
                ],
            },
        ],
        "enable_thinking": False,
        "temperature": 0.0,
        "max_tokens": 260,
    }


def annotate_one(
    row: dict[str, str], args: argparse.Namespace, api_key: str, raw_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    pair_id = row["pair_id"]
    raw_path = raw_dir / f"{pair_id}.json"
    if raw_path.is_file() and not args.overwrite:
        saved = json.loads(raw_path.read_text(encoding="utf-8"))
        return saved["label"], saved.get("usage", {}), True

    image_path = args.base_dir / row["contact_sheet_file"]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 2):
        started = time.perf_counter()
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(
                    args.endpoint,
                    headers=headers,
                    json=payload(args.model, pair_id, image_path),
                    timeout=(30, args.timeout),
                )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            label = validate_label(extract_json(content), pair_id)
            record = {
                "pair_id": pair_id,
                "model": args.model,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "attempt": attempt,
                "label": label,
                "usage": body.get("usage", {}),
                "request_id": body.get("id", ""),
                "raw_content": content,
            }
            raw_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return label, record["usage"], False
        except Exception as exc:
            last_error = exc
            if attempt <= args.retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"{pair_id} failed after retries: {last_error}")


def sum_usage(usages: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for usage in usages:
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + value
    return totals


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY in the current shell before running.")
    with (args.base_dir / "blind_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if args.limit > 0:
        rows = rows[: args.limit]

    raw_dir = args.base_dir / "qwen_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    labels_by_id: dict[str, dict[str, Any]] = {}
    usages: list[dict[str, Any]] = []
    cache_hits = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(annotate_one, row, args, api_key, raw_dir): row["pair_id"]
            for row in rows
        }
        for future in as_completed(futures):
            pair_id = futures[future]
            label, usage, cached = future.result()
            labels_by_id[pair_id] = label
            usages.append(usage)
            cache_hits += int(cached)
            print(
                f"{pair_id}: {label['preference']} ({label['confidence']:.2f})"
                + (" [cached]" if cached else ""),
                flush=True,
            )

    labels = [labels_by_id[row["pair_id"]] for row in rows]
    aggregate = {
        "model": args.model,
        "annotation_mode": "blind RGB contact-sheet comparison; non-thinking",
        "labels": labels,
    }
    (args.base_dir / "qwen_labels.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    usage = {
        "model": args.model,
        "pairs": len(labels),
        "cache_hits": cache_hits,
        "usage": sum_usage(usages),
    }
    (args.base_dir / "qwen_usage.json").write_text(
        json.dumps(usage, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(usage, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
