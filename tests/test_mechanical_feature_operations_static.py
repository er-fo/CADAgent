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


def _operation_branch_source(path: Path, operation: str) -> str:
    source = _read(path)
    marker = f'elif operation == "{operation}":'
    start = source.index(marker)
    next_start = source.find('\n            elif operation == "', start + len(marker))
    if next_start == -1:
        next_start = len(source)
    return source[start:next_start]


def _function_source(path: Path, function_name: str) -> str:
    source = _read(path)
    marker = f"def {function_name}("
    start = source.index(marker)
    next_start = source.find("\ndef ", start + len(marker))
    if next_start == -1:
        next_start = len(source)
    return source[start:next_start]


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

    def test_cadagent_does_not_dispatch_create_sweep(self) -> None:
        source = _read(MAC_CADAGENT)
        self.assertNotIn('operation == "create_sweep"', source)
        self.assertNotIn("feature_tools.create_sweep(", source)

    def test_split_body_branch_supports_singular_target_aliases(self) -> None:
        branch = _operation_branch_source(MAC_CADAGENT, "split_body")
        self.assertIn('message.get("target_body_token")', branch)
        self.assertIn('params.get("target_body_token")', branch)
        self.assertIn('message.get("target_body_ref")', branch)
        self.assertIn('params.get("target_body_ref")', branch)
        self.assertIn("str(target_body_token)", branch)

    def test_strict_bool_helper_used_for_audit_sensitive_fields(self) -> None:
        source = _read(MAC_CADAGENT)
        self.assertIn("def _coerce_strict_bool(", source)
        self.assertIn('_coerce_strict_bool(is_tangent_chain, "is_tangent_chain")', source)
        self.assertIn('_coerce_strict_bool(keep_tools, "keep_tools")', source)
        self.assertIn(
            '_coerce_strict_bool(extend_splitting_tool, "extend_splitting_tool")',
            source,
        )
        self.assertNotIn("bool(keep_tools)", source)
        self.assertEqual(source.count("bool(extend_splitting_tool)"), 0)

    def test_create_countersink_hole_exception_cleans_up_temp_sketch(self) -> None:
        function_source = _function_source(MAC_FEATURE_TOOLS, "create_countersink_hole")
        self.assertIn("except Exception as exc:", function_source)
        self.assertIn("temp_sketch.deleteMe()", function_source)
        self.assertIn("temp_sketch.isVisible = False", function_source)
