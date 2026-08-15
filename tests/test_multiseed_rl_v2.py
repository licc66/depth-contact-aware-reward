from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "dataset_generation" / "46_run_stackcube_multiseed_v2.py"


def load_module():
    name = "multiseed_rl_v2_test_module"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MultiSeedAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_aggregate_uses_across_seed_statistics(self):
        summaries = []
        for seed, success, reward in ((3, 0.2, 1.0), (7, 0.6, 3.0)):
            summaries.append(
                {
                    "seed": seed,
                    "results": {
                        "physical_only": {
                            "elapsed_seconds": 10.0 + seed,
                            "primary_scientific_result": False,
                            "evaluation": {
                                "success_rate": success,
                                "mean_return": reward,
                                "mean_sparse_return": success,
                                "mean_steps_to_success": None,
                            },
                        }
                    },
                }
            )
        result = self.module.aggregate(summaries, ["physical_only"])[
            "physical_only"
        ]
        self.assertAlmostEqual(result["metrics"]["success_rate"]["mean"], 0.4)
        self.assertAlmostEqual(result["metrics"]["mean_return"]["mean"], 2.0)
        self.assertFalse(result["primary_scientific_result"])


if __name__ == "__main__":
    unittest.main()
