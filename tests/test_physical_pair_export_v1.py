from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "dataset_generation"
sys.path.insert(0, str(SOURCE_DIR))

from reward_common_v1 import pair_group_id  # noqa: E402


def load_exporter_module():
    path = SOURCE_DIR / "26_export_physical_pair_scores_v1.py"
    spec = importlib.util.spec_from_file_location("physical_export_contract_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LabelReadVisitor(ast.NodeVisitor):
    LABELS = {
        "candidate_label",
        "mimo_preference",
        "mimo_confidence",
        "final_preference_label_v0",
        "fusion_label_v1",
    }

    def __init__(self) -> None:
        self.reads: list[tuple[str, int]] = []

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "row":
            try:
                key = ast.literal_eval(node.slice)
            except (ValueError, TypeError):
                key = None
            if key in self.LABELS:
                self.reads.append((str(key), node.lineno))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "row"
            and node.func.attr == "get"
            and node.args
        ):
            try:
                key = ast.literal_eval(node.args[0])
            except (ValueError, TypeError):
                key = None
            if key in self.LABELS:
                self.reads.append((str(key), node.lineno))
        self.generic_visit(node)


class TestPhysicalPairExporterV1(unittest.TestCase):
    def test_scoring_code_does_not_read_teacher_or_fusion_labels(self) -> None:
        path = SOURCE_DIR / "26_export_physical_pair_scores_v1.py"
        visitor = LabelReadVisitor()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        self.assertEqual(visitor.reads, [])

    def test_pair_group_id_matches_script17_contract(self) -> None:
        row = {
            "task_id": "stackcube",
            "clip_a_id": "SC-SUCC-0007-C001",
            "clip_b_id": "SC-SUCC-0007-OFFSET-X-C002",
        }
        self.assertEqual(pair_group_id(row), "stackcube::SC-SUCC-0007")

    def test_sampling_mismatch_fails_loudly(self) -> None:
        exporter = load_exporter_module()

        class Lookup:
            def get(self, task_id, trajectory_id, frame_idx):
                return {"gripper_width": float(frame_idx)}

        row = {
            "task_id": "stackcube",
            "clip_a_id": "SC-SUCC-0001-C000",
            "clip_a_start_frame": "0",
            "clip_a_end_frame_exclusive": "16",
            "clip_a_sample_frame_indices": "0;1;2;3;4;5",
        }
        clip_map = {
            "SC-SUCC-0001-C000": {"trajectory_id": "SC-SUCC-0001"}
        }
        with self.assertRaises(ValueError):
            exporter.clip_window_frames(
                row,
                "a",
                clip_map,
                Lookup(),
                history_window=16,
                sequence_length=6,
            )


if __name__ == "__main__":
    unittest.main()
