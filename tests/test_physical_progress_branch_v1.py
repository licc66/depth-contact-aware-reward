from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dataset_generation"))

from physical_progress_branch_v1 import (  # noqa: E402
    PhysicalProgressBranch,
    PhysicalProgressRuntime,
    monotonic_progress_loss,
    pair_preference_probability,
    prepare_physical_features,
    progress_regression_loss,
)


class PhysicalProgressBranchTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.model = PhysicalProgressBranch(
            input_dim=8,
            num_tasks=3,
            num_stages=4,
            frame_hidden_dim=16,
            temporal_hidden_dim=16,
            task_embedding_dim=4,
            dropout=0.0,
        )

    def test_forward_shapes_and_ranges(self) -> None:
        features = torch.randn(2, 5, 8)
        task_ids = torch.tensor([0, 2])
        frame_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
            ]
        )
        output = self.model(features, task_ids, frame_mask)

        self.assertEqual(output.stage_logits.shape, (2, 5, 4))
        self.assertEqual(output.potential.shape, (2, 5))
        self.assertEqual(output.clip_potential.shape, (2,))
        self.assertTrue(torch.all(output.clip_potential.ge(0.0)))
        self.assertTrue(torch.all(output.clip_potential.le(1.0)))
        self.assertTrue(torch.allclose(output.potential[0, 3:], torch.zeros(2)))

    def test_missing_features_are_neutral_and_marked(self) -> None:
        raw = torch.tensor([[[1.0, 99.0]]])
        valid = torch.tensor([[[True, False]]])
        center = torch.tensor([0.0, 5.0])
        scale = torch.tensor([2.0, 4.0])
        prepared = prepare_physical_features(raw, valid, center, scale)

        self.assertTrue(torch.allclose(prepared[0, 0], torch.tensor([0.5, 0.0, 1.0, 0.0])))

    def test_terminal_stage_saturates_potential(self) -> None:
        self.model.eval()
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.stage_head.bias[-1] = 10.0

        output = self.model(
            torch.zeros((1, 2, 8)),
            torch.tensor([0]),
            torch.ones((1, 2), dtype=torch.bool),
        )

        self.assertTrue(torch.allclose(output.local_progress, torch.ones((1, 2))))
        self.assertTrue(torch.allclose(output.potential, torch.ones((1, 2))))

    def test_uncertain_terminal_stage_is_gated(self) -> None:
        self.model.eval()
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.stage_head.bias.copy_(
                torch.tensor([0.0, 0.0, 0.9, 1.0])
            )

        output = self.model(
            torch.zeros((1, 1, 8)),
            torch.tensor([0]),
            torch.ones((1, 1), dtype=torch.bool),
        )

        self.assertEqual(int(output.clip_stage.item()), 2)
        self.assertLess(float(output.clip_potential.item()), 0.75)

    def test_preference_and_monotonic_loss(self) -> None:
        probability = pair_preference_probability(
            torch.tensor([0.8]),
            torch.tensor([0.2]),
            temperature=0.1,
        )
        self.assertGreater(float(probability.item()), 0.99)

        potential = torch.tensor([[0.1, 0.4, 0.3]])
        mask = torch.ones((1, 3), dtype=torch.bool)
        eligible = torch.tensor([True])
        loss = monotonic_progress_loss(potential, mask, eligible)
        self.assertAlmostEqual(float(loss.item()), 0.05, places=6)

        targets = torch.tensor([[0.1, float("nan"), 0.8]])
        progress_loss = progress_regression_loss(potential, targets, mask)
        self.assertAlmostEqual(float(progress_loss.item()), 0.0625, places=6)

    def test_checkpoint_runtime_interface(self) -> None:
        checkpoint = {
            "model_config": self.model.config(),
            "state_dict": self.model.state_dict(),
            "feature_names": ["depth", "contact", "speed", "width"],
            "feature_center": [0.0, 0.0, 0.0, 0.0],
            "feature_scale": [1.0, 1.0, 1.0, 1.0],
            "task_ids": ["peginsertion", "stackcube", "stackpyramid"],
            "feature_clip_value": 8.0,
            "preference_temperature": 0.1,
            "sequence_length": 3,
            "history_window": 5,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.pt"
            torch.save(checkpoint, checkpoint_path)
            runtime = PhysicalProgressRuntime.from_checkpoint(checkpoint_path, device="cpu")
            frames = [
                {"depth": 0.8, "contact": False, "speed": 0.1, "width": 0.04},
                {"depth": 0.5, "contact": True, "speed": 0.0, "width": 0.02},
                {"depth": 0.4, "contact": True, "speed": 0.0, "width": 0.02},
                {"depth": 0.3, "contact": True, "speed": 0.0, "width": 0.02},
                {"depth": 0.2, "contact": True, "speed": 0.0, "width": 0.02},
            ]
            result = runtime.score("stackcube", frames)

        self.assertIn(result["stage"], (1, 2, 3, 4))
        self.assertGreaterEqual(result["potential"], 0.0)
        self.assertLessEqual(result["potential"], 1.0)
        self.assertEqual(result["source_frame_count"], 5)
        self.assertEqual(result["model_frame_count"], 3)
        self.assertEqual(len(result["frame_potential"]), 3)


if __name__ == "__main__":
    unittest.main()
