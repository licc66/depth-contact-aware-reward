from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dataset_generation"))
TRAINING_PATH = REPO_ROOT / "dataset_generation" / "24_train_physical_progress_branch_v1.py"
SPEC = importlib.util.spec_from_file_location("train_physical_progress_branch_v1", TRAINING_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAINING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAINING
SPEC.loader.exec_module(TRAINING)


class PhysicalProgressTargetsTest(unittest.TestCase):
    def test_clean_success_targets_and_terminal_saturation(self) -> None:
        rows = [
            {
                "sample_id": "SUCCESS",
                "source_type": "official_success",
                "near_miss_type": "",
                "frame_idx": str(frame),
                "stage_id": str(stage),
            }
            for frame, stage in ((0, 1), (10, 1), (20, 4), (30, 4))
        ]
        rows.append(
            {
                "sample_id": "FAILURE",
                "source_type": "policy_failure",
                "near_miss_type": "",
                "frame_idx": "0",
                "stage_id": "1",
            }
        )

        targets = TRAINING.build_stage_local_progress_targets(rows)

        self.assertEqual(targets[("SUCCESS", 0)], 0.0)
        self.assertEqual(targets[("SUCCESS", 10)], 1.0)
        self.assertEqual(targets[("SUCCESS", 20)], 1.0)
        self.assertEqual(targets[("SUCCESS", 30)], 1.0)
        self.assertNotIn(("FAILURE", 0), targets)

    def test_canonical_split_map_overrides_legacy_frame_split(self) -> None:
        store = {}
        mapping = {}
        for split, sample_id in (("train", "S-TRAIN"), ("val", "S-VAL"), ("test", "S-TEST")):
            store[("stackcube", sample_id, 0)] = TRAINING.FrameRecord(
                values=np.asarray([0.0], dtype=np.float32),
                valid=np.asarray([True]),
                stage_target=0,
                local_progress_target=0.0,
                split="train",
            )
            mapping[("stackcube", sample_id)] = split

        result = TRAINING.build_auxiliary_splits(
            store,
            sequence_length=1,
            history_window=1,
            canonical_splits=mapping,
        )

        self.assertEqual(result["train"].sample_ids, ["S-TRAIN"])
        self.assertEqual(result["val"].sample_ids, ["S-VAL"])
        self.assertEqual(result["test"].sample_ids, ["S-TEST"])


if __name__ == "__main__":
    unittest.main()
