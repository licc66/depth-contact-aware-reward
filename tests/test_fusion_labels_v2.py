from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "dataset_generation" / "40_build_fusion_labels_v2.py"
SPEC = importlib.util.spec_from_file_location("fusion_labels_v2_script", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FusionLabelsV2Test(unittest.TestCase):
    def test_agreement_combines_evidence(self) -> None:
        policy = MODULE.FusionPolicyV2(single_threshold=0.7, conflict_gap=0.2)
        result = policy.decide("A>B", 0.8, "A>B", 0.9)
        self.assertEqual(result["label"], "A>B")
        self.assertTrue(result["use_for_loss"])
        self.assertGreater(result["confidence"], 0.9)

    def test_close_conflict_abstains(self) -> None:
        policy = MODULE.FusionPolicyV2(single_threshold=0.7, conflict_gap=0.2)
        result = policy.decide("A>B", 0.82, "B>A", 0.75)
        self.assertEqual(result["label"], "abstain")
        self.assertEqual(result["weight"], 0.0)

    def test_terminal_guard_caps_only_terminal_claims(self) -> None:
        terminal = {
            "phys_a_stage_expected": "3.8",
            "phys_a_completion_guard": "fail",
        }
        factor, reason = MODULE.completion_consistency_factor(terminal, "A>B")
        self.assertEqual(factor, 0.0)
        self.assertEqual(reason, "terminal_guard_fail")

        preterminal = {
            "phys_a_stage_expected": "2.9",
            "phys_a_completion_guard": "fail",
        }
        factor, reason = MODULE.completion_consistency_factor(preterminal, "A>B")
        self.assertEqual(factor, 1.0)
        self.assertEqual(reason, "preterminal")


if __name__ == "__main__":
    unittest.main()
