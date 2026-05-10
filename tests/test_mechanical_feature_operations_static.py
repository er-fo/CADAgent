from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAC_FEATURE_TOOLS = REPO_ROOT / "mac" / "CADAgent" / "feature_tools.py"
WIN_FEATURE_TOOLS = REPO_ROOT / "win" / "CADAgent" / "feature_tools.py"
MAC_CADAGENT = REPO_ROOT / "mac" / "CADAgent" / "CADAgent.py"
WIN_CADAGENT = REPO_ROOT / "win" / "CADAgent" / "CADAgent.py"

REQUIRED_HELPERS = {
    "create_countersink_hole",
    "apply_draft",
    "mirror_entities",
    "combine_bodies",
    "split_body",
    "split_face",
    "add_sketch_dimension",
    "add_sketch_constraint",
}

REQUIRED_OPERATIONS = {
    "create_countersink_hole",
    "apply_draft",
    "mirror_entities",
    "combine_bodies",
    "split_body",
    "split_face",
    "add_sketch_dimension",
    "add_sketch_constraint",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(_read(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class MechanicalFeatureOperationsStaticTests(unittest.TestCase):
    def test_feature_tools_parity(self) -> None:
        self.assertEqual(_read(MAC_FEATURE_TOOLS), _read(WIN_FEATURE_TOOLS))

    def test_cadagent_parity(self) -> None:
        self.assertEqual(_read(MAC_CADAGENT), _read(WIN_CADAGENT))

    def test_feature_tools_define_required_helpers(self) -> None:
        function_names = _function_names(MAC_FEATURE_TOOLS)
        missing = sorted(REQUIRED_HELPERS - function_names)
        self.assertFalse(missing, f"Missing helper definitions: {missing}")

    def test_cadagent_dispatches_required_feature_operations(self) -> None:
        source = _read(MAC_CADAGENT)
        for operation in sorted(REQUIRED_OPERATIONS):
            self.assertIn(f'operation == "{operation}"', source)
            self.assertIn(f"feature_tools.{operation}(", source)

    def test_cadagent_advertises_mechanical_feature_capability(self) -> None:
        self.assertIn('"mechanical_feature_operations_v1": True', _read(MAC_CADAGENT))
