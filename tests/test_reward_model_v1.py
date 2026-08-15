"""Tests for reward_model_v1 (Phase 4).

Two tiers:
- Torch-free contract tests: parse the module source with ``ast`` (never
  importing torch) and verify the declared feature contracts contain no
  forbidden/privileged names, and that the checkpoint format constant exists.
- Torch-gated behavioural tests: skipped cleanly when torch is unavailable
  (per-class skip), covering variants, modality dropout, gating,
  losses, deny-list enforcement, and checkpoint round-trip.
"""
from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_GENERATION = PROJECT_ROOT / "dataset_generation"
sys.path.insert(0, str(DATASET_GENERATION))

from reward_common_v1 import check_forbidden_feature  # noqa: E402

MODEL_SOURCE = (DATASET_GENERATION / "reward_model_v1.py").read_text(encoding="utf-8")


def _module_tuple_assignment(source: str, name: str) -> tuple[str, ...]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = ast.literal_eval(node.value)
                    return tuple(value)
    raise AssertionError(f"module-level tuple {name!r} not found")


class TestContractsTorchFree(unittest.TestCase):
    def test_physical_summary_contract_has_no_forbidden_features(self) -> None:
        features = _module_tuple_assignment(MODEL_SOURCE, "PHYSICAL_SUMMARY_FEATURES")
        self.assertGreater(len(features), 0)
        for feature in features:
            reason = check_forbidden_feature(feature)
            self.assertIsNone(reason, f"{feature}: {reason}")

    def test_checkpoint_format_version_declared(self) -> None:
        tree = ast.parse(MODEL_SOURCE)
        versions = [
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == "CHECKPOINT_FORMAT_VERSION"
        ]
        self.assertEqual(versions, [1])

    def test_no_v0_privileged_proxies_referenced(self) -> None:
        # v0 privileged physical proxies must never appear in v1 model code.
        for forbidden in (
            "stereo_end_score_proxy",
            "stereo_end_dist_m",
            "contact_end_stage_id",
            "final_preference_label_v0",
            "candidate_label",
        ):
            self.assertNotIn(forbidden, MODEL_SOURCE, forbidden)

    def test_train_and_eval_scripts_never_read_env_success(self) -> None:
        for script in ("28_train_reward_model_v1.py", "29_evaluate_reward_model_v1.py"):
            source = (DATASET_GENERATION / script).read_text(encoding="utf-8")
            self.assertNotIn("env_success", source, script)


try:  # torch-gated tier
    import torch
    from reward_model_v1 import (
        PHYSICAL_SUMMARY_FEATURES,
        TASK_CONTEXT_FEATURES,
        RewardModelV1,
        RewardModelV1Config,
        load_checkpoint,
        pairwise_preference_loss,
        save_checkpoint,
        temporal_order_loss,
    )
    TORCH_AVAILABLE = True
    TORCH_SKIP_REASON = ""
except ImportError as exc:  # pragma: no cover - environment dependent
    TORCH_AVAILABLE = False
    TORCH_SKIP_REASON = f"torch not installed ({exc}); torch-free contract tests still run"


@unittest.skipUnless(TORCH_AVAILABLE, TORCH_SKIP_REASON)


class TestRewardModelV1(unittest.TestCase):
    @staticmethod
    def _batch(config, batch: int = 4):
        torch.manual_seed(0)
        rgb = torch.randn(batch, config.rgb_dim)
        phys = torch.randn(batch, config.physical_dim)
        return rgb, phys

    def test_variants_shapes_and_range(self) -> None:
        for variant in ("rgb_only", "physical_only", "fusion"):
            config = RewardModelV1Config(variant=variant, hidden_dim=32)
            model = RewardModelV1(config).eval()
            rgb, phys = self._batch(config)
            with torch.no_grad():
                out = model(
                    rgb if variant != "physical_only" else None,
                    phys if variant != "rgb_only" else None,
                )
            self.assertEqual(out["potential"].shape, (4,))
            self.assertTrue(bool((out["potential"] >= 0).all()))
            self.assertTrue(bool((out["potential"] <= 1).all()))
            self.assertIn("stage_probs", out)
            self.assertEqual(out["stage_probs"].shape, (4, config.num_stages))

    def test_missing_modality_does_not_crash_fusion(self) -> None:
        config = RewardModelV1Config(variant="fusion", hidden_dim=32)
        model = RewardModelV1(config).eval()
        rgb, phys = self._batch(config)
        with torch.no_grad():
            out_no_phys = model(rgb, None)
            out_no_rgb = model(None, phys)
        # Gate must collapse to the surviving modality.
        self.assertTrue(bool((out_no_phys["gate_rgb_weight"] == 1.0).all()))
        self.assertTrue(bool((out_no_rgb["gate_rgb_weight"] == 0.0).all()))

    def test_validity_flags_change_output(self) -> None:
        config = RewardModelV1Config(variant="fusion", hidden_dim=32)
        model = RewardModelV1(config).eval()
        rgb, phys = self._batch(config)
        with torch.no_grad():
            out_valid = model(rgb, phys)
            out_invalid = model(
                rgb, phys,
                rgb_valid=torch.ones(4),
                physical_valid=torch.zeros(4),
            )
        self.assertFalse(
            torch.allclose(out_valid["potential"], out_invalid["potential"])
        )

    def test_summary_input_kind_dimension(self) -> None:
        config = RewardModelV1Config(
            variant="physical_only",
            physical_input_kind="summary",
            physical_dim=len(PHYSICAL_SUMMARY_FEATURES) + len(TASK_CONTEXT_FEATURES),
            hidden_dim=32,
        )
        model = RewardModelV1(config).eval()
        phys = torch.randn(
            3, len(PHYSICAL_SUMMARY_FEATURES) + len(TASK_CONTEXT_FEATURES)
        )
        with torch.no_grad():
            out = model(None, phys)
        self.assertEqual(out["potential"].shape, (3,))

    def test_forbidden_feature_in_contract_raises(self) -> None:
        with self.assertRaises(ValueError):
            RewardModelV1Config(
                physical_feature_contract=("potential", "stereo_end_score_proxy"),
            )
        with self.assertRaises(ValueError):
            RewardModelV1Config(variant="not_a_variant")

    def test_parameter_budget_under_one_million(self) -> None:
        counts = {}
        for variant in ("rgb_only", "physical_only", "fusion"):
            model = RewardModelV1(RewardModelV1Config(variant=variant))
            self.assertLess(model.parameter_count(), 1_000_000, variant)
            counts[variant] = model.parameter_count()
        self.assertLess(counts["physical_only"], counts["rgb_only"])
        self.assertLess(counts["rgb_only"], counts["fusion"])

    def test_pairwise_preference_loss_prefers_correct_order(self) -> None:
        a = torch.tensor([0.9, 0.8])
        b = torch.tensor([0.1, 0.2])
        labels = torch.tensor([1.0, 1.0])
        weights = torch.ones(2)
        loss_correct = pairwise_preference_loss(a, b, labels, weights, 0.1)
        loss_wrong = pairwise_preference_loss(b, a, labels, weights, 0.1)
        self.assertLess(float(loss_correct), float(loss_wrong))

    def test_abstain_weight_zero_removes_contribution(self) -> None:
        a = torch.tensor([0.1, 0.9])
        b = torch.tensor([0.9, 0.1])
        labels = torch.tensor([1.0, 1.0])
        weights = torch.tensor([0.0, 1.0])  # first pair abstained
        loss = pairwise_preference_loss(a, b, labels, weights, 0.1)
        loss_only_second = pairwise_preference_loss(
            a[1:], b[1:], labels[1:], weights[1:], 0.1
        )
        self.assertAlmostEqual(float(loss), float(loss_only_second), places=6)

    def test_temporal_order_loss(self) -> None:
        early = torch.tensor([0.2, 0.3])
        late = torch.tensor([0.4, 0.6])
        self.assertEqual(float(temporal_order_loss(early, late)), 0.0)
        self.assertGreater(float(temporal_order_loss(late, early)), 0.0)
        empty = torch.zeros(0)
        self.assertEqual(float(temporal_order_loss(empty, empty)), 0.0)

    def test_checkpoint_round_trip(self) -> None:
        config = RewardModelV1Config(
            variant="fusion",
            physical_input_kind="summary",
            physical_dim=len(PHYSICAL_SUMMARY_FEATURES) + len(TASK_CONTEXT_FEATURES),
            hidden_dim=32,
        )
        model = RewardModelV1(config).eval()
        rgb, phys = self._batch(config)
        with torch.no_grad():
            before = model(rgb, phys)["potential"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reward_model_v1.pt"
            save_checkpoint(model, path, extra={"note": "test"})
            restored, payload = load_checkpoint(path)
        with torch.no_grad():
            after = restored(rgb, phys)["potential"]
        self.assertTrue(torch.allclose(before, after, atol=1e-6))
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(payload["extra"]["note"], "test")
        self.assertEqual(
            tuple(payload["config"]["physical_feature_contract"]),
            PHYSICAL_SUMMARY_FEATURES + TASK_CONTEXT_FEATURES,
        )


if __name__ == "__main__":
    unittest.main()
