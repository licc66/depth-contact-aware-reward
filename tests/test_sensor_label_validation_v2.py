from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "dataset_generation" / "47_validate_sensor_aligned_labels_v2.py"


def load_module():
    name = "sensor_label_validation_v2_test_module"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SensorLabelValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_complete_clean_labels_pass(self):
        pairs = [{"pair_id": "p1"}, {"pair_id": "p2"}]
        labels = [
            {
                "pair_id": "p1",
                "mimo_preference": "A>B",
                "mimo_confidence": "0.8",
                "mimo_model": "model",
                "reason": "visible progress",
                "split": "train",
            },
            {
                "pair_id": "p2",
                "mimo_preference": "unsure",
                "mimo_confidence": "0.3",
                "mimo_model": "model",
                "reason": "ambiguous",
                "split": "val",
            },
        ]
        self.assertTrue(self.module.validate(pairs, labels)["valid_for_fusion"])

    def test_missing_and_error_labels_fail(self):
        pairs = [{"pair_id": "p1"}, {"pair_id": "p2"}]
        labels = [
            {
                "pair_id": "p1",
                "mimo_preference": "unsure",
                "mimo_confidence": "0",
                "mimo_model": "model",
                "reason": "ERROR: timeout",
                "split": "train",
            }
        ]
        report = self.module.validate(pairs, labels)
        self.assertFalse(report["valid_for_fusion"])
        self.assertEqual(report["missing_pair_ids"], ["p2"])
        self.assertEqual(report["api_error_pair_ids"], ["p1"])


if __name__ == "__main__":
    unittest.main()
