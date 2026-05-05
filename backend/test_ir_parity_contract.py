"""Phase 0/1 parity contract tests for agent input → IR → target adapters."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

try:
    from .backends.build123d.translator import Build123dCapabilityError, translate_ir_document_to_build123d
    from .backends.fusion.translator import translate_ir_to_fusion_tool_call
    from .ir.capabilities import (
        CAD_TOOL_CAPABILITIES,
        NON_CAD_AGENT_TOOLS,
        SUPPORTED_BUILD123D_IR_OPERATIONS,
        missing_capability_rows,
    )
    from .ir.document import IRDocumentState
    from .ir.mapper import map_tool_call_to_ir
    from .ir.types import AddRectangleParams, CreateSketchParams, IRDocument, IROperation
    from .ir.validator import validate_operation
except ImportError:  # pragma: no cover
    from backend.backend.backends.build123d.translator import Build123dCapabilityError, translate_ir_document_to_build123d
    from backend.backend.backends.fusion.translator import translate_ir_to_fusion_tool_call
    from backend.backend.ir.capabilities import (
        CAD_TOOL_CAPABILITIES,
        NON_CAD_AGENT_TOOLS,
        SUPPORTED_BUILD123D_IR_OPERATIONS,
        missing_capability_rows,
    )
    from backend.backend.ir.document import IRDocumentState
    from backend.backend.ir.mapper import map_tool_call_to_ir
    from backend.backend.ir.types import AddRectangleParams, CreateSketchParams, IRDocument, IROperation
    from backend.backend.ir.validator import validate_operation


ROOT = Path(__file__).resolve().parent
LLM_CLIENT = ROOT / "llm_client.py"


def _fusion_tool_schema_names() -> List[str]:
    tree = ast.parse(LLM_CLIENT.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "TOOLS" for target in node.targets):
            continue
        names: List[str] = []
        for tool_node in getattr(node.value, "elts", []):
            if not isinstance(tool_node, ast.Dict):
                continue
            for key_node, value_node in zip(tool_node.keys, tool_node.values):
                if isinstance(key_node, ast.Constant) and key_node.value == "name":
                    if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                        names.append(value_node.value)
        return names
    raise AssertionError("TOOLS assignment not found in llm_client.py")


def test_parity_contract_covers_every_fusion_visible_cad_tool():
    tool_names = set(_fusion_tool_schema_names())
    missing = missing_capability_rows(tool_names)

    visible_cad_tools = tool_names - set(NON_CAD_AGENT_TOOLS)
    contract_tools = set(CAD_TOOL_CAPABILITIES)

    assert not missing
    assert contract_tools.isdisjoint(NON_CAD_AGENT_TOOLS)
    assert visible_cad_tools <= contract_tools
    assert contract_tools - visible_cad_tools == {"jump_to_timeline_position"}


@pytest.mark.parametrize("tool_name,capability", sorted(CAD_TOOL_CAPABILITIES.items()))
def test_parity_contract_rows_are_complete(tool_name, capability):
    assert capability.tool_name == tool_name
    assert capability.ir_operation
    assert capability.category
    assert capability.fusion_status == "supported"
    assert capability.build123d_status in {"supported", "unsupported", "fusion_only"}
    assert capability.build123d_reason


@pytest.mark.parametrize(
    "tool_call,expected_ir,expected_fusion,build123d_supported",
    [
        (
            {"name": "create_sketch", "input": {"plane_id": "XY", "sketch_id": "s0"}},
            "create_sketch",
            "create_sketch",
            True,
        ),
        (
            {"name": "add_rectangle", "input": {"sketch_id": "s0", "corner1_u": -1, "corner1_v": -0.5, "corner2_u": 1, "corner2_v": 0.5}},
            "add_rectangle",
            "add_rectangle",
            True,
        ),
        (
            {"name": "add_circle", "input": {"sketch_id": "s0", "center_u": 0, "center_v": 0, "radius": 1}},
            "add_circle",
            "add_circle",
            True,
        ),
        (
            {"name": "list_sketch_profiles", "input": {"sketch_id": "s0"}},
            "list_sketch_profiles",
            "list_sketch_profiles",
            True,
        ),
        (
            {"name": "extrude_profile", "input": {"sketch_id": "s0", "profile_index": 0, "distance": 2, "operation": "NewBody"}},
            "extrude",
            "extrude_profile",
            True,
        ),
        (
            {"name": "add_line", "input": {"sketch_id": "s0", "start_u": 0, "start_v": 0, "end_u": 1, "end_v": 0}},
            "add_line",
            "add_line",
            True,
        ),
        (
            {"name": "add_arc", "input": {"sketch_id": "s0", "center_u": 0, "center_v": 0, "start_u": 1, "start_v": 0, "end_u": 0, "end_v": 1}},
            "add_arc",
            "add_arc",
            True,
        ),
        (
            {"name": "revolve_profile", "input": {"sketch_id": "s0", "profile_index": 0, "axis": {"type": "construction", "axis": "z"}, "extent": {"mode": "full"}, "operation": "NewBody"}},
            "revolve",
            "revolve_profile",
            False,
        ),
        (
            {"name": "create_loft", "input": {"profile_ids": ["s0", "s1"], "operation": "NewBody"}},
            "loft",
            "create_loft",
            False,
        ),
        (
            {"name": "apply_fillet", "input": {"edge_refs": ["edge_0"], "radius": 2, "radius_unit": "mm"}},
            "fillet",
            "apply_fillet",
            False,
        ),
        (
            {"name": "apply_chamfer", "input": {"edge_refs": ["edge_0"], "distance": 2, "distance_unit": "mm"}},
            "chamfer",
            "apply_chamfer",
            False,
        ),
        (
            {"name": "create_shell", "input": {"mode": "open", "face_refs": ["face_0"], "inside_thickness": 1, "thickness_unit": "mm"}},
            "shell",
            "create_shell",
            False,
        ),
        (
            {"name": "create_simple_hole", "input": {"face_ref": "face_0", "center_x": 0, "center_y": 0, "center_z": 0, "diameter": 4, "diameter_unit": "mm", "extent_type": "through_all"}},
            "create_simple_hole",
            "create_simple_hole",
            False,
        ),
        (
            {"name": "create_counterbore_hole", "input": {"face_ref": "face_0", "center_x": 0, "center_y": 0, "center_z": 0, "hole_diameter": 4, "hole_depth": 5, "counterbore_diameter": 8, "counterbore_depth": 2, "diameter_unit": "mm"}},
            "create_counterbore_hole",
            "create_counterbore_hole",
            False,
        ),
        (
            {"name": "create_tapped_hole", "input": {"face_ref": "face_0", "center_x": 0, "center_y": 0, "center_z": 0, "thread_type": "metric", "thread_size": "M6", "thread_depth": 8}},
            "create_tapped_hole",
            "create_tapped_hole",
            False,
        ),
        (
            {"name": "create_external_thread", "input": {"face_ref": "face_0", "thread_type": "metric", "thread_size": "M6", "is_full_length": True}},
            "create_external_thread",
            "create_external_thread",
            False,
        ),
        (
            {"name": "create_pattern_feature", "input": {"pattern_type": "rectangular", "feature_refs": ["feature_0"], "count_x": 2, "spacing_x_cm": 1}},
            "pattern_feature",
            "create_pattern_feature",
            False,
        ),
        (
            {"name": "list_features", "input": {}},
            "list_features",
            "list_features",
            False,
        ),
        (
            {"name": "delete_feature", "input": {"feature_token": "feature_token_0", "expected_name": "Extrude1"}},
            "delete_feature",
            "delete_feature",
            False,
        ),
        (
            {"name": "adjust_feature_parameters", "input": {"feature_token": "feature_token_0", "parameters": {"distance": 4}}},
            "adjust_feature_parameters",
            "adjust_feature_parameters",
            False,
        ),
        (
            {"name": "suppress_feature", "input": {"feature_token": "feature_token_0"}},
            "set_feature_suppression",
            "suppress_feature",
            False,
        ),
        (
            {"name": "unsuppress_feature", "input": {"feature_token": "feature_token_0"}},
            "set_feature_suppression",
            "unsuppress_feature",
            False,
        ),
        (
            {"name": "select_edges", "input": {"edge_refs": ["edge_0"]}},
            "select_entities",
            "select_edges",
            False,
        ),
        (
            {"name": "select_faces", "input": {"face_refs": ["face_0"]}},
            "select_entities",
            "select_faces",
            False,
        ),
        (
            {"name": "select_bodies", "input": {"body_refs": ["body_0"]}},
            "select_entities",
            "select_bodies",
            False,
        ),
        (
            {"name": "clear_edge_selection", "input": {}},
            "clear_selection",
            "clear_edge_selection",
            False,
        ),
        (
            {"name": "clear_face_selection", "input": {}},
            "clear_selection",
            "clear_face_selection",
            False,
        ),
        (
            {"name": "clear_body_selection", "input": {}},
            "clear_selection",
            "clear_body_selection",
            False,
        ),
        (
            {"name": "jump_to_timeline_position", "input": {"target_index": 0, "reason": "rollback"}},
            "jump_to_timeline_position",
            "jump_to_timeline_position",
            False,
        ),
        (
            {"name": "create_construction_plane", "input": {"plane_id": "p0", "mode": "offset_from_datum", "base_datum_plane": "XY", "offset_cm": 1}},
            "create_construction_plane",
            "create_construction_plane",
            False,
        ),
    ],
)
def test_golden_tool_call_maps_validates_and_preserves_fusion_translation(
    tool_call: Dict[str, Any],
    expected_ir: str,
    expected_fusion: str,
    build123d_supported: bool,
):
    state = IRDocumentState()
    operation = map_tool_call_to_ir(tool_call, state)

    assert operation.type == expected_ir
    assert not validate_operation(operation)

    fusion_tool, _fusion_input = translate_ir_to_fusion_tool_call(operation)
    assert fusion_tool == expected_fusion

    operations = [operation]
    if expected_ir == "extrude":
        operations = [
            IROperation(
                id="op_sketch",
                type="create_sketch",
                params=CreateSketchParams(plane="XY", sketch="s0"),
            ),
            IROperation(
                id="op_rect",
                type="add_rectangle",
                params=AddRectangleParams(sketch="s0", center=[0.0, 0.0], width=10.0, height=10.0),
            ),
            operation,
        ]
    document = IRDocument(version="1.0", units="mm", operations=operations, metadata={"source": "test"})
    if build123d_supported:
        program = translate_ir_document_to_build123d(document)
        assert "from build123d import" in program.code
        assert expected_ir in SUPPORTED_BUILD123D_IR_OPERATIONS
    else:
        with pytest.raises(Build123dCapabilityError) as exc_info:
            translate_ir_document_to_build123d(document)
        assert exc_info.value.operation_type == expected_ir
        assert exc_info.value.reason


def test_build123d_capability_errors_are_precise_for_unsupported_ir():
    state = IRDocumentState()
    operation = map_tool_call_to_ir(
        {"name": "apply_fillet", "input": {"edge_refs": ["edge_0"], "radius": 1, "radius_unit": "mm"}},
        state,
    )
    document = IRDocument(version="1.0", units="mm", operations=[operation], metadata={"source": "test"})

    with pytest.raises(Build123dCapabilityError) as exc_info:
        translate_ir_document_to_build123d(document)

    assert exc_info.value.operation_type == "fillet"
    assert "portable edge selectors" in exc_info.value.reason
    assert "Unsupported IR operation for build123d translator: fillet" in str(exc_info.value)
