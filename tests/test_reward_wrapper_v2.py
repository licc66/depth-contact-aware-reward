from __future__ import annotations

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER_V1_PATH = ROOT / "dataset_generation" / "30_maniskill_reward_wrapper_v1.py"
WRAPPER_V2_PATH = ROOT / "dataset_generation" / "43_maniskill_reward_wrapper_v2.py"
FROZEN_V1_SHA256 = (
    "5400b110069de8b48eacb67fe9378892b793ef19e20fbd255b901bd3d9323222"
)


def load_wrapper_v2():
    name = "reward_wrapper_v2_test_module"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, WRAPPER_V2_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RewardWrapperV2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_wrapper_v2()

    def test_frozen_v1_hash_is_unchanged(self):
        digest = hashlib.sha256(WRAPPER_V1_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, FROZEN_V1_SHA256)

    def test_v2_reuses_frozen_v1_core(self):
        self.assertIs(
            self.module.RewardWrapperCoreV1,
            self.module.v1.RewardWrapperCoreV1,
        )

    def test_terminal_consistency_requires_observed_release_and_support(self):
        check = self.module.terminal_consistency
        self.assertEqual(check({"object_goal_3d_dist_m": 0.01}), "fail")
        self.assertEqual(
            check(
                {
                    "object_support_contacts": 1.0,
                    "released_object": 0.0,
                    "object_goal_3d_dist_m": 0.01,
                }
            ),
            "fail",
        )

    def test_terminal_consistency_uses_depth_only_when_observed(self):
        check = self.module.terminal_consistency
        base = {"object_support_contacts": 1.0, "released_object": 1.0}
        self.assertEqual(check(base), "unknown_depth")
        self.assertEqual(check({**base, "object_goal_3d_dist_m": 0.045}), "pass")
        self.assertEqual(check({**base, "object_goal_3d_dist_m": 0.046}), "fail")

    def test_terminal_cap_catches_stage_or_potential_completion_claim(self):
        constrain = self.module.constrain_terminal_potential
        by_stage = constrain(0.70, [0.1, 0.1, 0.2, 0.6], "fail", 0.749)
        by_potential = constrain(0.90, [0.4, 0.3, 0.2, 0.1], "fail", 0.749)
        passed = constrain(0.90, [0.1, 0.1, 0.2, 0.6], "pass", 0.749)
        preterminal = constrain(0.70, [0.4, 0.3, 0.2, 0.1], "fail", 0.749)

        self.assertEqual(by_stage, (0.70, False))
        self.assertEqual(by_potential, (0.749, True))
        self.assertEqual(passed, (0.90, False))
        self.assertEqual(preterminal, (0.70, False))

    def test_terminal_cap_must_stay_below_completion_boundary(self):
        scorer = self.module.FrozenRewardModelScorerV2
        with self.assertRaisesRegex(ValueError, "terminal_cap"):
            scorer.__new__(scorer).__init__(
                Path("missing-physical.pt"),
                Path("missing-reward.pt"),
                "stackcube",
                terminal_cap=0.75,
            )

    def test_online_source_has_no_privileged_runtime_queries(self):
        source = WRAPPER_V2_PATH.read_text(encoding="utf-8")
        forbidden = (
            "env.evaluate(",
            "ue.evaluate(",
            ".cubeA.pose",
            ".cubeB.pose",
            ".cubeC.pose",
            ".peg.pose",
            "env_success",
            "success_flag",
            "requests.",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_scheduled_adapter_refreshes_sensor_at_interval_and_on_force(self):
        class FakeAdapter:
            def __init__(self, key):
                self.key = key
                self.calls = 0
                self.diagnostics = {f"{key}_diagnostic": True}

            def reset(self):
                self.calls = 0

            def build_frame(self, obs, env, action=None):  # noqa: ARG002
                self.calls += 1
                return {self.key: float(self.calls)}

        sensor = FakeAdapter("object_goal_3d_dist_m")
        contact = FakeAdapter("gripper_width")
        adapter = self.module.ScheduledCompositeAdapterV2(
            sensor,
            [contact],
            sensor_interval=3,
            allowed={"object_goal_3d_dist_m", "gripper_width"},
        )
        adapter.reset()
        reset_frame = adapter.build_frame(None, None, force_sensor=True)
        first = adapter.build_frame(None, None)
        second = adapter.build_frame(None, None)
        third = adapter.build_frame(None, None)
        forced = adapter.build_frame(None, None, force_sensor=True)

        self.assertIn("object_goal_3d_dist_m", reset_frame)
        self.assertNotIn("object_goal_3d_dist_m", first)
        self.assertNotIn("object_goal_3d_dist_m", second)
        self.assertIn("object_goal_3d_dist_m", third)
        self.assertIn("object_goal_3d_dist_m", forced)
        self.assertEqual(sensor.calls, 3)
        self.assertEqual(contact.calls, 5)
        self.assertTrue(adapter.diagnostics["sensor_refreshed"])
        self.assertEqual(adapter.diagnostics["sensor_refresh_count"], 3)

    def test_scheduled_adapter_rejects_misaligned_intervals(self):
        class FakeScorer:
            allowed_frame_keys = set()

            def __call__(self, frames):  # noqa: ARG002
                return {"potential": 0.0}

        class FakeAdapter:
            sensor_interval = 3

            def build_frame(self, obs, env, action=None):  # noqa: ARG002
                return {}

        class FakeEnv:
            observation_space = None
            action_space = None
            metadata = {}
            render_mode = None
            spec = None

        with self.assertRaises(ValueError):
            self.module.ManiSkillDenseRewardWrapperV2(
                FakeEnv(),
                scorer=FakeScorer(),
                observation_adapter=FakeAdapter(),
                inference_interval=4,
            )


if __name__ == "__main__":
    unittest.main()
