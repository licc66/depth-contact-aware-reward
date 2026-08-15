from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "dataset_generation"
sys.path.insert(0, str(SOURCE_DIR))

from reward_common_v1 import check_forbidden_feature  # noqa: E402


def literal_tuple(path: Path, name: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"tuple {name!r} not found in {path.name}")


def load_wrapper_module():
    path = SOURCE_DIR / "30_maniskill_reward_wrapper_v1.py"
    spec = importlib.util.spec_from_file_location("reward_wrapper_contract_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestFeatureDenyListV1(unittest.TestCase):
    def test_physical_feature_candidates_are_observable_contracts(self) -> None:
        sources = (
            (SOURCE_DIR / "24_train_physical_progress_branch_v1.py", "DEPTH_FEATURE_CANDIDATES"),
            (SOURCE_DIR / "24_train_physical_progress_branch_v1.py", "CONTACT_FEATURE_CANDIDATES"),
            (SOURCE_DIR / "26_export_physical_pair_scores_v1.py", "DEPTH_FEATURES"),
            (SOURCE_DIR / "26_export_physical_pair_scores_v1.py", "CONTACT_FEATURES"),
        )
        for path, tuple_name in sources:
            for feature in literal_tuple(path, tuple_name):
                self.assertIsNone(
                    check_forbidden_feature(feature),
                    f"{path.name}:{tuple_name}:{feature}",
                )

    def test_runtime_guard_rejects_privileged_keys(self) -> None:
        wrapper = load_wrapper_module()
        for key in ("env_success", "object_pose", "frame_idx", "candidate_label"):
            with self.assertRaises(ValueError, msg=key):
                wrapper.validate_frame_keys({key: 1.0}, allowed=None)

    def test_online_wrapper_has_no_commercial_vlm_client(self) -> None:
        source = (SOURCE_DIR / "30_maniskill_reward_wrapper_v1.py").read_text(
            encoding="utf-8"
        ).lower()
        for forbidden_import in (
            "import openai",
            "from openai",
            "import anthropic",
            "from anthropic",
            "mimo_vlm_common",
        ):
            self.assertNotIn(forbidden_import, source)


if __name__ == "__main__":
    unittest.main()
