from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dataset_generation"))

from reward_model_v2 import (  # noqa: E402
    RewardModelV2,
    RewardModelV2Config,
    load_checkpoint,
    save_checkpoint,
    weighted_potential_loss,
    weighted_stage_loss,
)


class RewardModelV2Test(unittest.TestCase):
    def test_weighted_losses_ignore_zero_weight_items(self) -> None:
        potential = weighted_potential_loss(
            torch.tensor([0.2, 0.9]),
            torch.tensor([0.2, 0.0]),
            torch.tensor([1.0, 0.0]),
        )
        self.assertAlmostEqual(float(potential), 0.0, places=6)

        logits = torch.tensor([[8.0, 0.0], [0.0, 8.0]])
        stage = weighted_stage_loss(
            logits,
            torch.tensor([0, 0]),
            torch.tensor([1.0, 0.0]),
        )
        self.assertLess(float(stage), 0.001)

    def test_checkpoint_round_trip(self) -> None:
        config = RewardModelV2Config(
            variant="fusion",
            rgb_dim=12,
            physical_dim=8,
            hidden_dim=16,
            dropout=0.0,
            physical_feature_contract=("physical_progress_v2_clip_embedding",),
        )
        model = RewardModelV2(config)
        rgb = torch.randn(2, 12)
        physical = torch.randn(2, 8)
        expected = model(rgb, physical)["potential"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward_v2.pt"
            save_checkpoint(model, path, extra={"semantic_source": "test"})
            restored, payload = load_checkpoint(path)
            actual = restored(rgb, physical)["potential"]
        self.assertEqual(payload["format_version"], 2)
        self.assertTrue(torch.allclose(expected, actual))


if __name__ == "__main__":
    unittest.main()
