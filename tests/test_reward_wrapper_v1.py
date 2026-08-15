from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = ROOT / "dataset_generation" / "30_maniskill_reward_wrapper_v1.py"


def load_wrapper():
    name = "reward_wrapper_v1_test_module"
    spec = importlib.util.spec_from_file_location(name, WRAPPER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RewardWrapperCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_wrapper()

    def test_cached_potential_and_anti_farming(self):
        values = iter([0.1, 0.4, 0.9, 0.9])
        core = self.module.RewardWrapperCoreV1(
            lambda frames: {"potential": next(values)},
            gamma=0.99,
            inference_interval=2,
            dense_clip=1.0,
        )
        core.reset({"gripper_width": 0.08})
        logs = [
            core.step({"gripper_width": 0.08}, sparse_reward=0.0)
            for _ in range(6)
        ]
        self.assertEqual(
            [log.scored for log in logs],
            [False, True, False, True, False, True],
        )
        self.assertAlmostEqual(logs[1].dense_reward_raw, 0.99 * 0.4 - 0.1)
        self.assertLessEqual(logs[4].dense_reward_raw, 0.0)

    def test_dry_run_terminal_requires_reset(self):
        values = iter([0.2, 0.8])
        core = self.module.RewardWrapperCoreV1(
            lambda frames: {"potential": next(values)}, dry_run=True
        )
        core.reset({})
        log = core.step({}, sparse_reward=1.5, terminated=True)
        self.assertEqual(log.total_reward, 1.5)
        with self.assertRaises(RuntimeError):
            core.step({}, sparse_reward=0.0)

    def test_privileged_key_is_rejected(self):
        core = self.module.RewardWrapperCoreV1(
            lambda frames: {"potential": 0.0}
        )
        with self.assertRaises(ValueError):
            core.reset({"env_success": 1.0})

    def test_uniform_six_frame_history(self):
        frames = [{"rgb": index} for index in range(16)]
        sampled = self.module.uniform_sample_history(frames, count=6)
        self.assertEqual(
            [frame["rgb"] for frame in sampled], [0, 3, 6, 9, 12, 15]
        )
        early = self.module.uniform_sample_history([{"rgb": 7}], count=6)
        self.assertEqual([frame["rgb"] for frame in early], [7] * 6)

    def test_action_adapter_respects_checkpoint_contract(self):
        adapter = self.module.ActionHistoryAdapterV1(
            history_window=3,
            allowed={"action_0", "action_l2", "action_delta_l2"},
        )
        first = adapter.build_frame(None, None, [3.0, 4.0])
        second = adapter.build_frame(None, None, [0.0, 4.0])
        self.assertEqual(first, {"action_0": 3.0, "action_l2": 5.0})
        self.assertEqual(second["action_l2"], 4.0)
        self.assertEqual(second["action_delta_l2"], 3.0)
        self.assertEqual(adapter.diagnostics["action_history_length"], 2)
        adapter.reset()
        self.assertEqual(adapter.diagnostics["action_history_length"], 0)

    def test_online_source_has_no_privileged_runtime_queries(self):
        source = WRAPPER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("env.evaluate(", source)
        self.assertNotIn("ue.evaluate(", source)
        self.assertNotIn(".cubeA.pose", source)
        self.assertNotIn(".cubeB.pose", source)
        self.assertNotIn(".cubeC.pose", source)
        self.assertNotIn(".peg.pose", source)
        self.assertNotIn("requests.", source)


if __name__ == "__main__":
    unittest.main()
