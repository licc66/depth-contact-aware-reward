"""Reliability-gated depth/contact progress model for sensor-domain training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PhysicalProgressOutputV2:
    stage_logits: torch.Tensor
    stage_probs: torch.Tensor
    local_progress: torch.Tensor
    potential: torch.Tensor
    modality_gates: torch.Tensor
    frame_confidence: torch.Tensor
    frame_embedding: torch.Tensor
    clip_stage_logits: torch.Tensor
    clip_stage_probs: torch.Tensor
    clip_stage: torch.Tensor
    clip_local_progress: torch.Tensor
    clip_potential: torch.Tensor
    clip_modality_gates: torch.Tensor
    clip_confidence: torch.Tensor
    clip_embedding: torch.Tensor


def prepare_observed_features(
    raw: torch.Tensor,
    valid: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    clip_value: float = 8.0,
) -> torch.Tensor:
    """Normalize observed values and append one validity bit per feature."""

    valid = valid.bool()
    safe = torch.where(valid, raw, center)
    normalized = ((safe - center) / scale.clamp_min(1e-6)).clamp(
        -clip_value, clip_value
    )
    return torch.cat([normalized, valid.to(raw.dtype)], dim=-1)


def last_valid(sequence: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    indices = frame_mask.long().sum(dim=1).sub(1).clamp_min(0)
    batch = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[batch, indices]


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class PhysicalProgressBranchV2(nn.Module):
    """Causal stage/progress estimator with reliability-aware modality fusion."""

    def __init__(
        self,
        depth_feature_count: int,
        contact_feature_count: int,
        num_tasks: int = 1,
        num_stages: int = 4,
        modality_hidden_dim: int = 96,
        temporal_hidden_dim: int = 128,
        task_embedding_dim: int = 16,
        num_gru_layers: int = 1,
        dropout: float = 0.10,
        use_depth: bool = True,
        use_contact: bool = True,
        direct_potential_head: bool = False,
    ) -> None:
        super().__init__()
        if not use_depth and not use_contact:
            raise ValueError("at least one modality must be enabled")
        if use_depth and depth_feature_count <= 0:
            raise ValueError("depth_feature_count must be positive when depth is enabled")
        if use_contact and contact_feature_count <= 0:
            raise ValueError("contact_feature_count must be positive when contact is enabled")
        self.depth_feature_count = int(depth_feature_count)
        self.contact_feature_count = int(contact_feature_count)
        self.num_tasks = int(num_tasks)
        self.num_stages = int(num_stages)
        self.modality_hidden_dim = int(modality_hidden_dim)
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.task_embedding_dim = int(task_embedding_dim)
        self.num_gru_layers = int(num_gru_layers)
        self.dropout_rate = float(dropout)
        self.use_depth = bool(use_depth)
        self.use_contact = bool(use_contact)
        self.direct_potential_head = bool(direct_potential_head)

        self.depth_encoder = (
            ModalityEncoder(2 * depth_feature_count, modality_hidden_dim, dropout)
            if use_depth
            else None
        )
        self.contact_encoder = (
            ModalityEncoder(2 * contact_feature_count, modality_hidden_dim, dropout)
            if use_contact
            else None
        )
        gate_input_dim = 2 * modality_hidden_dim + 2
        self.gate_network = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, modality_hidden_dim),
            nn.GELU(),
            nn.Linear(modality_hidden_dim, 2),
        )
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)
        self.temporal_encoder = nn.GRU(
            input_size=modality_hidden_dim + task_embedding_dim + 2,
            hidden_size=temporal_hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.output_norm = nn.LayerNorm(temporal_hidden_dim)
        self.stage_head = nn.Linear(temporal_hidden_dim, num_stages)
        self.progress_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim + num_stages, temporal_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden_dim, 1),
        )
        self.potential_head = (
            nn.Sequential(
                nn.Linear(temporal_hidden_dim + num_stages, temporal_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(temporal_hidden_dim, 1),
            )
            if direct_potential_head
            else None
        )

    def config(self) -> dict[str, Any]:
        return {
            "depth_feature_count": self.depth_feature_count,
            "contact_feature_count": self.contact_feature_count,
            "num_tasks": self.num_tasks,
            "num_stages": self.num_stages,
            "modality_hidden_dim": self.modality_hidden_dim,
            "temporal_hidden_dim": self.temporal_hidden_dim,
            "task_embedding_dim": self.task_embedding_dim,
            "num_gru_layers": self.num_gru_layers,
            "dropout": self.dropout_rate,
            "use_depth": self.use_depth,
            "use_contact": self.use_contact,
            "direct_potential_head": self.direct_potential_head,
        }

    @staticmethod
    def _valid_ratio(features: torch.Tensor, feature_count: int) -> torch.Tensor:
        if feature_count <= 0:
            return torch.zeros(features.shape[:2], device=features.device)
        return features[..., feature_count:].mean(dim=-1)

    def forward(
        self,
        depth_features: torch.Tensor,
        contact_features: torch.Tensor,
        task_ids: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        modality_dropout: float = 0.0,
    ) -> PhysicalProgressOutputV2:
        if depth_features.ndim != 3 or contact_features.ndim != 3:
            raise ValueError("modality tensors must have shape [batch, time, feature]")
        if depth_features.shape[:2] != contact_features.shape[:2]:
            raise ValueError("depth and contact sequences must align")
        batch_size, time_steps = depth_features.shape[:2]
        if task_ids.shape != (batch_size,):
            raise ValueError("task_ids must have shape [batch]")
        if frame_mask is None:
            frame_mask = torch.ones(
                (batch_size, time_steps), dtype=torch.bool, device=depth_features.device
            )
        else:
            frame_mask = frame_mask.bool()

        zero_embedding = torch.zeros(
            (batch_size, time_steps, self.modality_hidden_dim),
            dtype=depth_features.dtype,
            device=depth_features.device,
        )
        depth_ratio = (
            self._valid_ratio(depth_features, self.depth_feature_count)
            if self.use_depth
            else torch.zeros_like(frame_mask, dtype=depth_features.dtype)
        )
        contact_ratio = (
            self._valid_ratio(contact_features, self.contact_feature_count)
            if self.use_contact
            else torch.zeros_like(frame_mask, dtype=depth_features.dtype)
        )
        depth_token = (
            self.depth_encoder(depth_features) if self.depth_encoder else zero_embedding
        )
        contact_token = (
            self.contact_encoder(contact_features)
            if self.contact_encoder
            else zero_embedding
        )
        depth_available = depth_ratio.gt(0.0) & frame_mask & self.use_depth
        contact_available = contact_ratio.gt(0.0) & frame_mask & self.use_contact

        if self.training and modality_dropout > 0.0 and self.use_depth and self.use_contact:
            drop_depth = torch.rand(
                (batch_size, 1), device=depth_features.device
            ).lt(modality_dropout)
            drop_contact = torch.rand(
                (batch_size, 1), device=depth_features.device
            ).lt(modality_dropout)
            both = drop_depth & drop_contact
            drop_contact = drop_contact & ~both
            depth_available = depth_available & ~drop_depth
            contact_available = contact_available & ~drop_contact
            depth_ratio = torch.where(drop_depth, torch.zeros_like(depth_ratio), depth_ratio)
            contact_ratio = torch.where(
                drop_contact, torch.zeros_like(contact_ratio), contact_ratio
            )

        reliability = torch.stack([depth_ratio, contact_ratio], dim=-1)
        gate_input = torch.cat(
            [depth_token, contact_token, reliability], dim=-1
        )
        gate_logits = self.gate_network(gate_input)
        available = torch.stack([depth_available, contact_available], dim=-1)
        gate_logits = gate_logits.masked_fill(~available, -1e4)
        all_missing = ~available.any(dim=-1, keepdim=True)
        gate_logits = torch.where(all_missing, torch.zeros_like(gate_logits), gate_logits)
        gates = F.softmax(gate_logits, dim=-1) * available.to(gate_logits.dtype)
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        fused = gates[..., :1] * depth_token + gates[..., 1:] * contact_token

        task_token = self.task_embedding(task_ids).unsqueeze(1).expand(-1, time_steps, -1)
        temporal_input = torch.cat([fused, task_token, reliability], dim=-1)
        lengths = frame_mask.long().sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            temporal_input, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.temporal_encoder(packed)
        temporal_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=time_steps
        )
        hidden = self.output_norm(temporal_output)
        stage_logits = self.stage_head(hidden)
        stage_probs = F.softmax(stage_logits, dim=-1)
        local_progress = torch.sigmoid(
            self.progress_head(torch.cat([hidden, stage_probs], dim=-1)).squeeze(-1)
        )
        if self.potential_head is not None:
            potential = torch.sigmoid(
                self.potential_head(
                    torch.cat([hidden, stage_probs], dim=-1)
                ).squeeze(-1)
            )
        else:
            stage_axis = torch.arange(
                self.num_stages, dtype=hidden.dtype, device=hidden.device
            )
            expected_stage = (stage_probs * stage_axis).sum(dim=-1)
            potential = (expected_stage + local_progress) / float(self.num_stages)

        entropy = -(stage_probs.clamp_min(1e-8).log() * stage_probs).sum(dim=-1)
        entropy_scale = math.log(float(self.num_stages))
        stage_confidence = (1.0 - entropy / entropy_scale).clamp(0.0, 1.0)
        sensor_reliability = (gates * reliability).sum(dim=-1).clamp(0.0, 1.0)
        frame_confidence = stage_confidence * sensor_reliability
        numeric_mask = frame_mask.to(hidden.dtype)
        local_progress = local_progress * numeric_mask
        potential = potential * numeric_mask
        frame_confidence = frame_confidence * numeric_mask
        gates = gates * numeric_mask.unsqueeze(-1)
        frame_embedding = hidden * numeric_mask.unsqueeze(-1)

        clip_stage_logits = last_valid(stage_logits, frame_mask)
        clip_stage_probs = F.softmax(clip_stage_logits, dim=-1)
        return PhysicalProgressOutputV2(
            stage_logits=stage_logits,
            stage_probs=stage_probs,
            local_progress=local_progress,
            potential=potential,
            modality_gates=gates,
            frame_confidence=frame_confidence,
            frame_embedding=frame_embedding,
            clip_stage_logits=clip_stage_logits,
            clip_stage_probs=clip_stage_probs,
            clip_stage=clip_stage_probs.argmax(dim=-1),
            clip_local_progress=last_valid(local_progress, frame_mask),
            clip_potential=last_valid(potential, frame_mask),
            clip_modality_gates=last_valid(gates, frame_mask),
            clip_confidence=last_valid(frame_confidence, frame_mask),
            clip_embedding=last_valid(hidden, frame_mask),
        )


def _observed_float(value: Any) -> tuple[float, bool]:
    if value is None:
        return 0.0, False
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return 0.0, False
    if text in {"true", "yes"}:
        return 1.0, True
    if text in {"false", "no"}:
        return 0.0, True
    try:
        parsed = float(text)
        return (parsed, True) if math.isfinite(parsed) else (0.0, False)
    except (TypeError, ValueError):
        return 0.0, False


class PhysicalProgressRuntimeV2:
    """Frozen online scorer for checkpoints written by the v2 trainer."""

    def __init__(self, checkpoint: dict[str, Any], device: str) -> None:
        self.device = device
        self.model = PhysicalProgressBranchV2(**checkpoint["model_config"]).to(device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.depth_feature_names = list(checkpoint["depth_feature_names"])
        self.contact_feature_names = list(checkpoint["contact_feature_names"])
        self.depth_center = torch.as_tensor(
            checkpoint["depth_center"], dtype=torch.float32, device=device
        )
        self.depth_scale = torch.as_tensor(
            checkpoint["depth_scale"], dtype=torch.float32, device=device
        )
        self.contact_center = torch.as_tensor(
            checkpoint["contact_center"], dtype=torch.float32, device=device
        )
        self.contact_scale = torch.as_tensor(
            checkpoint["contact_scale"], dtype=torch.float32, device=device
        )
        self.feature_clip_value = float(checkpoint["feature_clip_value"])
        self.sequence_length = int(checkpoint["sequence_length"])
        self.history_window = int(checkpoint["history_window"])
        self.task_index = {
            task: index for index, task in enumerate(checkpoint["task_ids"])
        }

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, device: str = "auto"
    ) -> "PhysicalProgressRuntimeV2":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            checkpoint = torch.load(
                Path(checkpoint_path), map_location=device, weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(Path(checkpoint_path), map_location=device)
        if checkpoint.get("format_version") not in {6, 7}:
            raise ValueError("expected physical progress checkpoint format_version=6 or 7")
        return cls(checkpoint, device)

    def _resample(self, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        history = frames[-self.history_window :]
        if len(history) <= self.sequence_length:
            return history
        positions = torch.linspace(0, len(history) - 1, self.sequence_length)
        return [history[index] for index in positions.round().long().tolist()]

    def _modality_tensor(
        self,
        frames: list[dict[str, Any]],
        names: list[str],
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if not names:
            return torch.empty(
                (1, len(frames), 0), dtype=torch.float32, device=self.device
            )
        values = []
        validity = []
        for frame in frames:
            row_values, row_validity = zip(
                *[_observed_float(frame.get(name)) for name in names]
            )
            values.append(row_values)
            validity.append(row_validity)
        raw = torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)
        valid = torch.tensor(validity, dtype=torch.bool, device=self.device).unsqueeze(0)
        return prepare_observed_features(
            raw, valid, center, scale, self.feature_clip_value
        )

    @torch.inference_mode()
    def score(
        self,
        task_id: str,
        frames: list[dict[str, Any]],
        include_embedding: bool = False,
    ) -> dict[str, Any]:
        if not frames:
            raise ValueError("frames must contain at least one observation")
        if task_id not in self.task_index:
            raise KeyError(f"unknown task_id {task_id!r}")
        selected = self._resample(frames)
        depth = self._modality_tensor(
            selected,
            self.depth_feature_names,
            self.depth_center,
            self.depth_scale,
        )
        contact = self._modality_tensor(
            selected,
            self.contact_feature_names,
            self.contact_center,
            self.contact_scale,
        )
        frame_mask = torch.ones(
            (1, len(selected)), dtype=torch.bool, device=self.device
        )
        task = torch.tensor(
            [self.task_index[task_id]], dtype=torch.long, device=self.device
        )
        output = self.model(depth, contact, task, frame_mask)
        gates = output.clip_modality_gates[0].cpu().tolist()
        result = {
            "stage": int(output.clip_stage.item()) + 1,
            "stage_probabilities": output.clip_stage_probs[0].cpu().tolist(),
            "local_progress": float(output.clip_local_progress.item()),
            "potential": float(output.clip_potential.item()),
            "depth_gate": float(gates[0]),
            "contact_gate": float(gates[1]),
            "confidence": float(output.clip_confidence.item()),
            "source_frame_count": len(frames),
            "model_frame_count": len(selected),
        }
        if include_embedding:
            result["embedding"] = output.clip_embedding[0].cpu().tolist()
        return result


def pairwise_preference_loss(
    potential_high: torch.Tensor,
    potential_low: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = (potential_high - potential_low) / temperature
    return F.softplus(-logits).mean()


def temporal_order_loss(
    potential: torch.Tensor,
    targets: torch.Tensor,
    frame_mask: torch.Tensor,
    margin: float = 0.01,
) -> torch.Tensor:
    if potential.shape[1] < 2:
        return potential.sum() * 0.0
    valid = frame_mask[:, 1:] & frame_mask[:, :-1]
    target_delta = targets[:, 1:] - targets[:, :-1]
    informative = valid & target_delta.abs().gt(1e-4)
    if not informative.any():
        return potential.sum() * 0.0
    direction = target_delta.sign()
    prediction_delta = potential[:, 1:] - potential[:, :-1]
    return F.relu(margin - direction * prediction_delta)[informative].mean()


def potential_shaping_reward(
    potential_t: torch.Tensor, potential_next: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    return gamma * potential_next - potential_t
