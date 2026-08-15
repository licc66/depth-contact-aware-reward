from __future__ import annotations

import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "dataset_generation"))

from reward_common_v1 import PavCalibrator, linspace_resample  # noqa: E402


class TestRewardCommonV1(unittest.TestCase):
    def test_pav_tied_scores_are_order_invariant(self) -> None:
        first = PavCalibrator().fit([0.2, 0.2, 0.8, 0.8], [0, 1, 1, 0])
        second = PavCalibrator().fit([0.8, 0.2, 0.8, 0.2], [0, 1, 1, 0])
        probes = [0.2, 0.5, 0.8]
        self.assertEqual(first.predict(probes), second.predict(probes))
        self.assertEqual(first.predict([0.2, 0.8]), [0.5, 0.5])

    def test_single_sample_resampling_is_defined(self) -> None:
        self.assertEqual(linspace_resample([1, 2, 3], 1), [3])
        with self.assertRaises(ValueError):
            linspace_resample([1], 0)


if __name__ == "__main__":
    unittest.main()
