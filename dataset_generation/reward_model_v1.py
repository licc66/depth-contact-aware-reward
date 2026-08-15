"""Lightweight multimodal reward/progress model v1 (Phase 4).

Architecture (all backbones frozen; only the small heads train):

    RGB/task branch : frozen OpenCLIP conditioned features
                      [img, text, img*text]  -> LayerNorm -> proj -> GELU
    Physical branch : frozen PhysicalProgressBranch v1 clip embedding (128-d)
                      or its 9-d observable summary vector
                      -> LayerNorm -> proj -> GELU
    Fusion          : gated blend g = sigmoid(W [rgb_h ; phys_h ; validity])
                      fused = g*rgb_h + (1-g)*phys_h
    Heads           : potential Phi(s) in [0,1] (sigmoid scalar),
                      optional stage logits (training-only targets),
                      confidence in [0,1].

Each modality input carries an explicit validity scalar so that a missing
modality (validity 0, features zeroed) is distinguishable from "sensors say
failure". Modality dropout during training zeroes a modality AND its validity
flag, teaching exactly that distinction (master prompt Phase 4 requirement).

Forbidden inputs (env_success, poses, evaluate outputs, time/frame index,
stage ids, teacher/candidate/fusion labels, rule proxy scores) are rejected by
``assert_features_allowed`` on the declared feature contract at construction
and checkpoint-load time.

This module requires torch. It must not be imported by the torch-free tests;
they import ``reward_common_v1`` instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from reward_common_v1 import assert_features_allowed

CHECKPOINT_FORMAT_VERSION = 1
VARIANTS = ("rgb_only", "physical_only", "fusion")

# Observable per-clip physical summary contract (order matters).
PHYSICAL_SUMMARY_FEATURES = (
    "stage_p1",
    "stage_p2",
    "stage_p3",
    "stage_p4",
    "local_progress",
    "potential",
    "confidence",
    "depth_validity_ratio",
    "contact_validity_ratio",
)
TASK_CONTEXT_FEATURES = (
    "task_is_peginsertion",
    "task_is_stackcube",
    "task_is_stackpyramid",
)
PHYSICAL_EMBEDDING_FEATURES = ("physical_clip_embedding",)


@dataclass
class RewardModelV1Config:
    variant: str = "fusion"
    rgb_dim: int = 1536  # 512*3 conditioned OpenCLIP ViT-B/32 features
    physical_dim: int = 128  # clip_embedding; 9 when physical_input_kind=summary
    physical_input_kind: str = "embedding"  # or "summary"
    hidden_dim: int = 128
    dropout: float = 0.10
    num_stages: int = 4
    use_stage_head: bool = True
    preference_temperature: float = 0.10
    rgb_feature_contract: tuple[str, ...] = ("openclip_image", "openclip_text", "openclip_image_x_text")
    physical_feature_contract: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.physical_input_kind not in ("embedding", "summary"):
            raise ValueError("physical_input_kind must be embedding|summary")
        if not self.physical_feature_contract:
            self.physical_feature_contract = (
                PHYSICAL_SUMMARY_FEATURES + TASK_CONTEXT_FEATURES
                if self.physical_input_kind == "summary"
                else PHYSICAL_EMBEDDING_FEATURES
            )
        expected_physical_dim = (
            len(PHYSICAL_SUMMARY_FEATURES) + len(TASK_CONTEXT_FEATURES)
            if self.physical_input_kind == "summary"
            else self.physical_dim
        )
        if self.physical_input_kind == "summary" and self.physical_dim != expected_physical_dim:
            raise ValueError(
                f"summary physical_dim must be {expected_physical_dim}, got {self.physical_dim}"
            )
        assert_features_allowed(self.rgb_feature_contract, "RewardModelV1 rgb contract")
        assert_features_allowed(
            self.physical_feature_contract, "RewardModelV1 physical contract"
        )


class RewardModelV1(nn.Module):
    def __init__(self, config: RewardModelV1Config) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim

        self.rgb_proj = (
            nn.Sequential(
                nn.LayerNorm(config.rgb_dim + 1),  # +1 validity flag
                nn.Linear(config.rgb_dim + 1, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            if config.variant != "physical_only"
            else None
        )
        self.phys_proj = (
            nn.Sequential(
                nn.LayerNorm(config.physical_dim + 1),
                nn.Linear(config.physical_dim + 1, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            if config.variant != "rgb_only"
            else None
        )
        self.gate = (
            nn.Linear(2 * hidden + 2, 1) if config.variant == "fusion" else None
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.potential_head = nn.Linear(hidden, 1)
        self.stage_head = (
            nn.Linear(hidden, config.num_stages) if config.use_stage_head else None
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        rgb: torch.Tensor | None,
        physical: torch.Tensor | None,
        rgb_valid: torch.Tensor | None = None,
        physical_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """rgb: [B, rgb_dim] or None; physical: [B, physical_dim] or None.

        Validity tensors are [B] in [0,1]; None means "fully valid if the
        tensor was provided, invalid otherwise". Variant masking is enforced
        here so a checkpointed variant cannot silently read the other
        modality.
        """
        if self.config.variant == "rgb_only":
            physical = None
        elif self.config.variant == "physical_only":
            rgb = None

        batch = None
        for tensor in (rgb, physical):
            if tensor is not None:
                batch = tensor.shape[0]
                device = tensor.device
                dtype = tensor.dtype
                break
        if batch is None:
            raise ValueError("at least one modality tensor is required")

        def prepared(
            tensor: torch.Tensor | None, dim: int, valid: torch.Tensor | None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if tensor is None:
                features = torch.zeros((batch, dim), device=device, dtype=dtype)
                validity = torch.zeros((batch,), device=device, dtype=dtype)
            else:
                if tensor.ndim != 2 or tensor.shape != (batch, dim):
                    raise ValueError(
                        f"expected modality tensor shape {(batch, dim)}, got "
                        f"{tuple(tensor.shape)}"
                    )
                features = tensor
                validity = (
                    torch.ones((batch,), device=device, dtype=dtype)
                    if valid is None
                    else valid.to(device=device, dtype=dtype)
                )
                if validity.shape != (batch,):
                    raise ValueError(
                        f"expected validity shape {(batch,)}, got {tuple(validity.shape)}"
                    )
                validity = validity.clamp(0.0, 1.0)
                features = features * validity.unsqueeze(-1)
            return features, validity

        rgb_features, rgb_validity = prepared(rgb, self.config.rgb_dim, rgb_valid)
        phys_features, phys_validity = prepared(
            physical, self.config.physical_dim, physical_valid
        )

        if self.config.variant == "rgb_only":
            assert self.rgb_proj is not None
            rgb_hidden = self.rgb_proj(
                torch.cat([rgb_features, rgb_validity.unsqueeze(-1)], dim=-1)
            )
            fused = rgb_hidden
            gate = torch.ones((batch,), device=rgb_hidden.device, dtype=rgb_hidden.dtype)
        elif self.config.variant == "physical_only":
            assert self.phys_proj is not None
            phys_hidden = self.phys_proj(
                torch.cat([phys_features, phys_validity.unsqueeze(-1)], dim=-1)
            )
            fused = phys_hidden
            gate = torch.zeros((batch,), device=phys_hidden.device, dtype=phys_hidden.dtype)
        else:
            assert self.rgb_proj is not None and self.phys_proj is not None
            assert self.gate is not None
            rgb_hidden = self.rgb_proj(
                torch.cat([rgb_features, rgb_validity.unsqueeze(-1)], dim=-1)
            )
            phys_hidden = self.phys_proj(
                torch.cat([phys_features, phys_validity.unsqueeze(-1)], dim=-1)
            )
            learned_gate = torch.sigmoid(
                self.gate(
                    torch.cat(
                        [
                            rgb_hidden,
                            phys_hidden,
                            rgb_validity.unsqueeze(-1),
                            phys_validity.unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                )
            ).squeeze(-1)
            rgb_available = rgb_validity > 0
            phys_available = phys_validity > 0
            gate = torch.where(
                rgb_available & ~phys_available,
                torch.ones_like(learned_gate),
                torch.where(
                    phys_available & ~rgb_available,
                    torch.zeros_like(learned_gate),
                    learned_gate,
                ),
            )
            fused = gate.unsqueeze(-1) * rgb_hidden + (1.0 - gate.unsqueeze(-1)) * phys_hidden

        trunk = self.trunk(fused)
        potential = torch.sigmoid(self.potential_head(trunk)).squeeze(-1)
        output: dict[str, torch.Tensor] = {
            "potential": potential,
            "gate_rgb_weight": gate,
        }
        if self.stage_head is not None:
            output["stage_logits"] = self.stage_head(trunk)
            output["stage_probs"] = F.softmax(output["stage_logits"], dim=-1)
            entropy = -(
                output["stage_probs"].clamp_min(1e-8).log()
                * output["stage_probs"]
            ).sum(dim=-1)
            max_entropy = torch.log(
                torch.tensor(
                    float(self.config.num_stages),
                    device=trunk.device,
                    dtype=trunk.dtype,
                )
            )
            stage_confidence = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
        else:
            stage_confidence = torch.ones(
                (batch,), device=trunk.device, dtype=trunk.dtype
            )
        modality_validity = torch.maximum(rgb_validity, phys_validity)
        output["confidence"] = stage_confidence * modality_validity
        return output

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------

def pairwise_preference_loss(
    potential_a: torch.Tensor,
    potential_b: torch.Tensor,
    labels_a_better: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = (potential_a - potential_b) / max(temperature, 1e-6)
    losses = F.binary_cross_entropy_with_logits(
        logits, labels_a_better.float(), reduction="none"
    )
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def temporal_order_loss(
    potential_early: torch.Tensor,
    potential_late: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Later clips of a clean successful trajectory must not score lower."""
    if potential_early.numel() == 0:
        return potential_early.sum() * 0.0
    return F.relu(potential_early - potential_late + margin).mean()


def stage_auxiliary_loss(
    stage_logits: torch.Tensor,
    stage_targets: torch.Tensor,
) -> torch.Tensor:
    valid = stage_targets.ge(0)
    if not valid.any():
        return stage_logits.sum() * 0.0
    return F.cross_entropy(stage_logits[valid], stage_targets[valid])


# ----------------------------------------------------------------------
# Checkpoint IO
# ----------------------------------------------------------------------

def save_checkpoint(
    model: RewardModelV1,
    path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "config": asdict(model.config),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "parameter_count": model.parameter_count(),
    }
    if extra:
        payload["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, Path(path))


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[RewardModelV1, dict[str, Any]]:
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=True)
    except Exception:
        # v1 checkpoints contain only tensors + plain types; fall back for
        # older torch versions, with the audit-noted caveat (F13).
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported reward model checkpoint format: {payload.get('format_version')}"
        )
    config_payload = dict(payload["config"])
    for key in ("rgb_feature_contract", "physical_feature_contract"):
        if key in config_payload:
            config_payload[key] = tuple(config_payload[key])
    config = RewardModelV1Config(**config_payload)
    model = RewardModelV1(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
