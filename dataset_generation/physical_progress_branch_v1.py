from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PhysicalProgressOutput:
    stage_logits: torch.Tensor
    stage_probs: torch.Tensor
    local_progress: torch.Tensor
    potential: torch.Tensor
    frame_confidence: torch.Tensor
    clip_stage_logits: torch.Tensor
    clip_stage_probs: torch.Tensor
    clip_stage: torch.Tensor
    clip_local_progress: torch.Tensor
    clip_potential: torch.Tensor
    clip_confidence: torch.Tensor
    # Expose frozen temporal features to the downstream reward model without
    # changing the v5 checkpoint parameters or its existing outputs.
    frame_embedding: torch.Tensor | None = None
    clip_embedding: torch.Tensor | None = None


def prepare_physical_features(
    raw_features: torch.Tensor,
    valid_features: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    clip_value: float = 8.0,
) -> torch.Tensor:
    """Normalize observed features and append validity indicators."""
    valid = valid_features.to(dtype=raw_features.dtype)
    safe_raw = torch.where(valid_features, raw_features, center)
    normalized = ((safe_raw - center) / scale.clamp_min(1e-6)).clamp(-clip_value, clip_value)
    return torch.cat([normalized, valid], dim=-1)


def last_valid(sequence: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    if sequence.ndim < 2:
        raise ValueError("sequence must have shape [batch, time, ...]")
    if frame_mask.shape != sequence.shape[:2]:
        raise ValueError("frame_mask must match the first two sequence dimensions")
    indices = frame_mask.long().sum(dim=1).sub(1).clamp_min(0)
    batch_indices = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[batch_indices, indices]


class PhysicalProgressBranch(nn.Module):
    """Causal stage-conditioned physical progress estimator.

    The network consumes normalized per-frame depth/contact features. It predicts
    a stage distribution and local progress for every frame, then combines them
    into a globally ordered potential in [0, 1].
    """

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        num_stages: int = 4,
        frame_hidden_dim: int = 128,
        temporal_hidden_dim: int = 128,
        task_embedding_dim: int = 16,
        num_gru_layers: int = 1,
        dropout: float = 0.10,
        completion_threshold: float = 0.80,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        if num_stages < 2:
            raise ValueError("num_stages must be at least 2")
        if not 0.5 <= completion_threshold <= 1.0:
            raise ValueError("completion_threshold must be in [0.5, 1.0]")

        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.num_stages = num_stages
        self.frame_hidden_dim = frame_hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.task_embedding_dim = task_embedding_dim
        self.num_gru_layers = num_gru_layers
        self.dropout_rate = dropout
        self.completion_threshold = completion_threshold

        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, frame_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(frame_hidden_dim, frame_hidden_dim),
            nn.GELU(),
        )
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)
        self.temporal_encoder = nn.GRU(
            input_size=frame_hidden_dim + task_embedding_dim,
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

    def config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "num_tasks": self.num_tasks,
            "num_stages": self.num_stages,
            "frame_hidden_dim": self.frame_hidden_dim,
            "temporal_hidden_dim": self.temporal_hidden_dim,
            "task_embedding_dim": self.task_embedding_dim,
            "num_gru_layers": self.num_gru_layers,
            "dropout": self.dropout_rate,
            "completion_threshold": self.completion_threshold,
        }

    def forward(
        self,
        features: torch.Tensor,
        task_ids: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> PhysicalProgressOutput:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature]")
        batch_size, time_steps, _ = features.shape
        if task_ids.shape != (batch_size,):
            raise ValueError("task_ids must have shape [batch]")
        if frame_mask is None:
            frame_mask = torch.ones((batch_size, time_steps), dtype=torch.bool, device=features.device)
        else:
            frame_mask = frame_mask.bool()

        frame_tokens = self.frame_encoder(features)
        task_tokens = self.task_embedding(task_ids).unsqueeze(1).expand(-1, time_steps, -1)
        temporal_input = torch.cat([frame_tokens, task_tokens], dim=-1)

        lengths = frame_mask.long().sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            temporal_input,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.temporal_encoder(packed)
        temporal_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=time_steps,
        )
        hidden = self.output_norm(temporal_output)

        stage_logits = self.stage_head(hidden)
        stage_probs = F.softmax(stage_logits, dim=-1)
        progress_input = torch.cat([hidden, stage_probs], dim=-1)
        raw_local_progress = torch.sigmoid(self.progress_head(progress_input).squeeze(-1))

        stage_axis = torch.arange(self.num_stages, dtype=features.dtype, device=features.device)
        hard_stage = stage_probs.argmax(dim=-1)
        if not self.training:
            uncertain_terminal = (
                hard_stage.eq(self.num_stages - 1)
                & stage_probs[..., -1].lt(self.completion_threshold)
            )
            nonterminal_stage = stage_probs[..., :-1].argmax(dim=-1)
            hard_stage = torch.where(
                uncertain_terminal,
                nonterminal_stage,
                hard_stage,
            )
        hard_stage_probs = F.one_hot(hard_stage, num_classes=self.num_stages).to(features.dtype)
        routed_stage_probs = stage_probs if self.training else hard_stage_probs
        routed_stage = (routed_stage_probs * stage_axis).sum(dim=-1)
        terminal_gate = routed_stage_probs[..., -1]
        local_progress = (
            raw_local_progress
            + terminal_gate * (1.0 - raw_local_progress)
        )
        potential = (routed_stage + local_progress) / float(self.num_stages)

        entropy = -(stage_probs.clamp_min(1e-8).log() * stage_probs).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(self.num_stages), device=features.device))
        frame_confidence = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
        potential = potential * frame_mask.to(potential.dtype)
        local_progress = local_progress * frame_mask.to(local_progress.dtype)
        frame_confidence = frame_confidence * frame_mask.to(frame_confidence.dtype)

        clip_stage_logits = last_valid(stage_logits, frame_mask)
        clip_stage_probs = F.softmax(clip_stage_logits, dim=-1)
        frame_embedding = hidden * frame_mask.unsqueeze(-1).to(hidden.dtype)
        clip_embedding = last_valid(hidden, frame_mask)
        return PhysicalProgressOutput(
            stage_logits=stage_logits,
            stage_probs=stage_probs,
            local_progress=local_progress,
            potential=potential,
            frame_confidence=frame_confidence,
            clip_stage_logits=clip_stage_logits,
            clip_stage_probs=clip_stage_probs,
            clip_stage=last_valid(hard_stage, frame_mask),
            clip_local_progress=last_valid(local_progress, frame_mask),
            clip_potential=last_valid(potential, frame_mask),
            clip_confidence=last_valid(frame_confidence, frame_mask),
            frame_embedding=frame_embedding,
            clip_embedding=clip_embedding,
        )


def _as_observed_float(value: Any) -> tuple[float, bool]:
    if value is None:
        return 0.0, False
    if isinstance(value, bool):
        return float(value), True
    text = str(value).strip().lower()
    if not text:
        return 0.0, False
    if text in {"true", "yes"}:
        return 1.0, True
    if text in {"false", "no"}:
        return 0.0, True
    try:
        parsed = float(text)
        if not torch.isfinite(torch.tensor(parsed)):
            return 0.0, False
        return parsed, True
    except (TypeError, ValueError):
        return 0.0, False


class PhysicalProgressRuntime:
    """Small inference wrapper used by the future reward-model/RL integration."""

    def __init__(
        self,
        model: PhysicalProgressBranch,
        feature_names: list[str],
        center: torch.Tensor,
        scale: torch.Tensor,
        task_ids: list[str],
        feature_clip_value: float,
        preference_temperature: float,
        sequence_length: int,
        history_window: int,
        device: str,
    ):
        self.model = model.eval()
        self.feature_names = feature_names
        self.center = center.to(device)
        self.scale = scale.to(device)
        self.task_index = {task_id: index for index, task_id in enumerate(task_ids)}
        self.feature_clip_value = feature_clip_value
        self.preference_temperature = preference_temperature
        self.sequence_length = sequence_length
        self.history_window = history_window
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> "PhysicalProgressRuntime":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            checkpoint = torch.load(
                Path(checkpoint_path), map_location=device, weights_only=True
            )
        except TypeError:
            # PyTorch < 2.0 does not expose weights_only. The v5 checkpoint is
            # produced locally and contains tensors plus plain Python values.
            checkpoint = torch.load(Path(checkpoint_path), map_location=device)
        model = PhysicalProgressBranch(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return cls(
            model=model,
            feature_names=list(checkpoint["feature_names"]),
            center=torch.as_tensor(checkpoint["feature_center"], dtype=torch.float32),
            scale=torch.as_tensor(checkpoint["feature_scale"], dtype=torch.float32),
            task_ids=list(checkpoint["task_ids"]),
            feature_clip_value=float(checkpoint["feature_clip_value"]),
            preference_temperature=float(checkpoint["preference_temperature"]),
            sequence_length=int(checkpoint.get("sequence_length", 6)),
            history_window=int(checkpoint.get("history_window", 16)),
            device=device,
        )

    def _resample_history(
        self,
        frames: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not frames:
            raise ValueError("frames must contain at least one physical observation")
        history = frames[-self.history_window :]
        if len(history) <= self.sequence_length:
            return history
        positions = torch.linspace(
            0,
            len(history) - 1,
            self.sequence_length,
        ).round().long().tolist()
        return [history[position] for position in positions]

    def _prepare_sequence(
        self,
        task_id: str,
        frames: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if task_id not in self.task_index:
            raise KeyError(f"Unknown task_id: {task_id}")
        if not frames:
            raise ValueError("frames must contain at least one physical observation")
        values: list[list[float]] = []
        validity: list[list[bool]] = []
        for frame in frames:
            frame_values: list[float] = []
            frame_validity: list[bool] = []
            for name in self.feature_names:
                value, valid = _as_observed_float(frame.get(name))
                frame_values.append(value)
                frame_validity.append(valid)
            values.append(frame_values)
            validity.append(frame_validity)
        raw = torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)
        valid = torch.tensor(validity, dtype=torch.bool, device=self.device).unsqueeze(0)
        features = prepare_physical_features(
            raw,
            valid,
            self.center,
            self.scale,
            self.feature_clip_value,
        )
        task = torch.tensor([self.task_index[task_id]], dtype=torch.long, device=self.device)
        frame_mask = torch.ones((1, len(frames)), dtype=torch.bool, device=self.device)
        return features, task, frame_mask

    @torch.inference_mode()
    def score(
        self,
        task_id: str,
        frames: list[dict[str, Any]],
        return_embedding: bool = False,
    ) -> dict[str, Any]:
        source_frame_count = len(frames)
        model_frames = self._resample_history(frames)
        features, task, frame_mask = self._prepare_sequence(task_id, model_frames)
        output = self.model(features, task, frame_mask)
        extra: dict[str, Any] = {}
        if return_embedding and output.clip_embedding is not None:
            extra["embedding"] = output.clip_embedding[0].cpu().tolist()
        return {
            **extra,
            "source_frame_count": source_frame_count,
            "model_frame_count": len(model_frames),
            "stage": int(output.clip_stage.item()) + 1,
            "stage_probabilities": output.clip_stage_probs[0].cpu().tolist(),
            "local_progress": float(output.clip_local_progress.item()),
            "potential": float(output.clip_potential.item()),
            "confidence": float(output.clip_confidence.item()),
            "frame_potential": output.potential[0].cpu().tolist(),
            "frame_local_progress": output.local_progress[0].cpu().tolist(),
            "frame_confidence": output.frame_confidence[0].cpu().tolist(),
        }

    @torch.inference_mode()
    def compare(
        self,
        task_id: str,
        frames_a: list[dict[str, Any]],
        frames_b: list[dict[str, Any]],
    ) -> dict[str, Any]:
        result_a = self.score(task_id, frames_a)
        result_b = self.score(task_id, frames_b)
        potential_a = torch.tensor(result_a["potential"])
        potential_b = torch.tensor(result_b["potential"])
        probability = float(
            pair_preference_probability(
                potential_a,
                potential_b,
                self.preference_temperature,
            ).item()
        )
        confidence = (
            2.0
            * abs(probability - 0.5)
            * (result_a["confidence"] * result_b["confidence"]) ** 0.5
        )
        if probability > 0.5:
            preference = "A>B"
        elif probability < 0.5:
            preference = "B>A"
        else:
            preference = "tie"
        return {
            "preference": preference,
            "probability_a_better": probability,
            "confidence": confidence,
            "a": result_a,
            "b": result_b,
        }


def pair_preference_probability(
    potential_a: torch.Tensor,
    potential_b: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.sigmoid((potential_a - potential_b) / temperature)


def weighted_pairwise_loss(
    potential_a: torch.Tensor,
    potential_b: torch.Tensor,
    labels_a_better: torch.Tensor,
    weights: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    logits = (potential_a - potential_b) / temperature
    losses = F.binary_cross_entropy_with_logits(logits, labels_a_better.float(), reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def stage_classification_loss(
    stage_logits: torch.Tensor,
    stage_targets: torch.Tensor,
    frame_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = frame_mask.bool() & stage_targets.ge(0)
    if not valid.any():
        return stage_logits.sum() * 0.0
    return F.cross_entropy(stage_logits[valid], stage_targets[valid], weight=class_weights)


def progress_regression_loss(
    local_progress: torch.Tensor,
    progress_targets: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    valid = frame_mask.bool() & torch.isfinite(progress_targets)
    if not valid.any():
        return local_progress.sum() * 0.0
    return F.smooth_l1_loss(local_progress[valid], progress_targets[valid])


def monotonic_progress_loss(
    potential: torch.Tensor,
    frame_mask: torch.Tensor,
    eligible_clips: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    if potential.shape[1] < 2:
        return potential.sum() * 0.0
    valid_pairs = frame_mask[:, 1:] & frame_mask[:, :-1] & eligible_clips.bool().unsqueeze(1)
    if not valid_pairs.any():
        return potential.sum() * 0.0
    violations = F.relu(potential[:, :-1] - potential[:, 1:] + margin)
    return violations[valid_pairs].mean()


def potential_shaping_reward(
    potential_t: torch.Tensor,
    potential_next: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    return gamma * potential_next - potential_t
