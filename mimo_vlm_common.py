from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def read_api_key(root: Path, explicit_path: Path | None = None) -> str:
    env_key = os.environ.get("MIMO_API_KEY") or os.environ.get("XIAOMI_MIMO_API_KEY")
    if env_key:
        return env_key.strip()
    key_path = explicit_path or (root / "mimo apikey.txt")
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    raise RuntimeError("No MiMo API key found. Set MIMO_API_KEY or provide --api-key-file.")


def default_base_url(api_key: str) -> str:
    if api_key.startswith("tp-"):
        return "https://token-plan-cn.xiaomimimo.com/v1"
    return "https://api.xiaomimimo.com/v1"


def frame_to_data_url(frame: np.ndarray, image_size: int, jpeg_quality: int) -> str:
    image = Image.fromarray(frame).convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def request_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": False,
        "max_completion_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MiMo HTTP {exc.code}: {body}") from exc


def extract_message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"MiMo response has no choices: {json.dumps(response, ensure_ascii=False)[:500]}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def clamp01(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return float("nan")


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def finite_or_zero(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
