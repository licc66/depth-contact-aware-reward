from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dataset_generation"))

from physical_progress_branch_v2 import (  # noqa: E402
    PhysicalProgressBranchV2,
    PhysicalProgressRuntimeV2,
    pairwise_preference_loss,
    prepare_observed_features,
    temporal_order_loss,
)


class PhysicalProgressBranchV2Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = PhysicalProgressBranchV2(
            depth_feature_count=3,
            contact_feature_count=2,
            num_tasks=1,
            modality_hidden_dim=16,
            temporal_hidden_dim=20,
            task_embedding_dim=4,
            dropout=0.0,
        )

    @staticmethod
    def prepared(batch: int = 2, time: int = 5):
        depth = torch.randn(batch, time, 6)
        contact = torch.randn(batch, time, 4)
        depth[..., 3:] = 1.0
        contact[..., 2:] = 1.0
        return depth, contact

    def test_forward_shapes_ranges_and_gate_sum(self) -> None:
        depth, contact = self.prepared()
        mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]]
        )
        output = self.model(depth, contact, torch.zeros(2, dtype=torch.long), mask)

        self.assertEqual(output.stage_logits.shape, (2, 5, 4))
        self.assertEqual(output.clip_modality_gates.shape, (2, 2))
        self.assertTrue(torch.all(output.clip_potential.ge(0.0)))
        self.assertTrue(torch.all(output.clip_potential.le(1.0)))
        self.assertTrue(
            torch.allclose(output.clip_modality_gates.sum(dim=-1), torch.ones(2))
        )
        self.assertTrue(torch.allclose(output.potential[0, 3:], torch.zeros(2)))

    def test_missing_depth_collapses_gate_to_contact(self) -> None:
        depth, contact = self.prepared(batch=1, time=3)
        depth[..., 3:] = 0.0
        output = self.model(
            depth,
            contact,
            torch.zeros(1, dtype=torch.long),
            torch.ones((1, 3), dtype=torch.bool),
        )
        self.assertTrue(
            torch.allclose(output.modality_gates[..., 0], torch.zeros((1, 3)))
        )
        self.assertTrue(
            torch.allclose(output.modality_gates[..., 1], torch.ones((1, 3)))
        )

    def test_depth_only_variant_has_unit_depth_gate(self) -> None:
        model = PhysicalProgressBranchV2(
            depth_feature_count=3,
            contact_feature_count=2,
            modality_hidden_dim=12,
            temporal_hidden_dim=12,
            task_embedding_dim=4,
            dropout=0.0,
            use_depth=True,
            use_contact=False,
        )
        depth, contact = self.prepared(batch=1, time=2)
        output = model(
            depth,
            contact,
            torch.zeros(1, dtype=torch.long),
            torch.ones((1, 2), dtype=torch.bool),
        )
        self.assertTrue(
            torch.allclose(output.modality_gates[..., 0], torch.ones((1, 2)))
        )
        self.assertTrue(
            torch.allclose(output.modality_gates[..., 1], torch.zeros((1, 2)))
        )

    def test_modality_dropout_never_drops_both_modalities(self) -> None:
        depth, contact = self.prepared(batch=2, time=3)
        self.model.train()
        output = self.model(
            depth,
            contact,
            torch.zeros(2, dtype=torch.long),
            torch.ones((2, 3), dtype=torch.bool),
            modality_dropout=1.0,
        )
        self.assertTrue(
            torch.allclose(
                output.modality_gates.sum(dim=-1), torch.ones((2, 3)), atol=1e-6
            )
        )

    def test_direct_potential_head_is_independent_and_bounded(self) -> None:
        model = PhysicalProgressBranchV2(
            depth_feature_count=3,
            contact_feature_count=2,
            modality_hidden_dim=12,
            temporal_hidden_dim=12,
            task_embedding_dim=4,
            dropout=0.0,
            direct_potential_head=True,
        )
        depth, contact = self.prepared(batch=1, time=3)
        output = model(
            depth,
            contact,
            torch.zeros(1, dtype=torch.long),
            torch.ones((1, 3), dtype=torch.bool),
        )
        self.assertIsNotNone(model.potential_head)
        self.assertTrue(torch.all(output.potential.ge(0.0)))
        self.assertTrue(torch.all(output.potential.le(1.0)))

    def test_feature_preparation_neutralizes_missing_value(self) -> None:
        raw = torch.tensor([[[2.0, 100.0]]])
        valid = torch.tensor([[[True, False]]])
        center = torch.tensor([0.0, 4.0])
        scale = torch.tensor([2.0, 8.0])
        prepared = prepare_observed_features(raw, valid, center, scale)
        self.assertTrue(
            torch.allclose(prepared[0, 0], torch.tensor([1.0, 0.0, 1.0, 0.0]))
        )

    def test_pairwise_and_temporal_losses_prefer_correct_order(self) -> None:
        correct = pairwise_preference_loss(torch.tensor([0.8]), torch.tensor([0.2]))
        reversed_loss = pairwise_preference_loss(
            torch.tensor([0.2]), torch.tensor([0.8])
        )
        self.assertLess(float(correct.item()), float(reversed_loss.item()))

        mask = torch.ones((1, 3), dtype=torch.bool)
        target = torch.tensor([[0.1, 0.4, 0.8]])
        ordered = temporal_order_loss(torch.tensor([[0.1, 0.4, 0.8]]), target, mask)
        reversed_temporal = temporal_order_loss(
            torch.tensor([[0.8, 0.4, 0.1]]), target, mask
        )
        self.assertLess(float(ordered.item()), float(reversed_temporal.item()))

    def test_checkpoint_runtime_round_trip(self) -> None:
        checkpoint = {
            "format_version": 6,
            "model_config": self.model.config(),
            "state_dict": self.model.state_dict(),
            "task_ids": ["stackcube"],
            "depth_feature_names": ["d0", "d1", "d2"],
            "contact_feature_names": ["c0", "c1"],
            "depth_center": [0.0, 0.0, 0.0],
            "depth_scale": [1.0, 1.0, 1.0],
            "contact_center": [0.0, 0.0],
            "contact_scale": [1.0, 1.0],
            "feature_clip_value": 8.0,
            "sequence_length": 3,
            "history_window": 5,
        }
        frames = [
            {"d0": 0.8, "d1": 0.2, "d2": 0.1, "c0": False, "c1": 0.04},
            {"d0": 0.4, "d1": 0.1, "d2": 0.0, "c0": True, "c1": 0.02},
            {"d0": 0.2, "d1": 0.0, "d2": 0.0, "c0": True, "c1": 0.02},
            {"d0": 0.1, "d1": 0.0, "d2": 0.0, "c0": False, "c1": 0.04},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physical_v2.pt"
            torch.save(checkpoint, path)
            runtime = PhysicalProgressRuntimeV2.from_checkpoint(path, device="cpu")
            result = runtime.score("stackcube", frames)
            result_with_embedding = runtime.score(
                "stackcube", frames, include_embedding=True
            )

        self.assertIn(result["stage"], (1, 2, 3, 4))
        self.assertGreaterEqual(result["potential"], 0.0)
        self.assertLessEqual(result["potential"], 1.0)
        self.assertAlmostEqual(result["depth_gate"] + result["contact_gate"], 1.0, places=5)
        self.assertEqual(result["source_frame_count"], 4)
        self.assertEqual(result["model_frame_count"], 3)
        self.assertNotIn("embedding", result)
        self.assertEqual(len(result_with_embedding["embedding"]), 20)

    def test_parameter_budget(self) -> None:
        count = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertLess(count, 1_000_000)


if __name__ == "__main__":
    unittest.main()
