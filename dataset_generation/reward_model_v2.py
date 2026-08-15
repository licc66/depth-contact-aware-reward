"""Lightweight RGB/physical reward model v2 and checkpoint contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from reward_model_v1 import (
    RewardModelV1,
    RewardModelV1Config,
    pairwise_preference_loss,
    temporal_order_loss,
)


CHECKPOINT_FORMAT_VERSION = 2


@dataclass
class RewardModelV2Config(RewardModelV1Config):
    """The v1 lightweight architecture with a distinct v2 data contract."""


class RewardModelV2(RewardModelV1):
    config: RewardModelV2Config

    def __init__(self, config: RewardModelV2Config) -> None:
        super().__init__(config)


def weighted_potential_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    losses = F.smooth_l1_loss(prediction, target, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def weighted_stage_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    valid = target.ge(0) & weights.gt(0)
    if not valid.any():
        return logits.sum() * 0.0
    losses = F.cross_entropy(logits[valid], target[valid], reduction="none")
    selected_weights = weights[valid]
    return (losses * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)


def save_checkpoint(
    model: RewardModelV2,
    path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "pipeline_version": "stackcube_sensor_reward_v2",
        "config": asdict(model.config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "parameter_count": model.parameter_count(),
    }
    if extra:
        payload["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, Path(path))


def load_checkpoint(
    path: Path, device: str = "cpu"
) -> tuple[RewardModelV2, dict[str, Any]]:
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(Path(path), map_location=device)
    if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported reward model v2 checkpoint format: {payload.get('format_version')}"
        )
    config_payload = dict(payload["config"])
    for key in ("rgb_feature_contract", "physical_feature_contract"):
        if key in config_payload:
            config_payload[key] = tuple(config_payload[key])
    config = RewardModelV2Config(**config_payload)
    model = RewardModelV2(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "RewardModelV2",
    "RewardModelV2Config",
    "load_checkpoint",
    "pairwise_preference_loss",
    "save_checkpoint",
    "temporal_order_loss",
    "weighted_potential_loss",
    "weighted_stage_loss",
]
