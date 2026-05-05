from dataclasses import replace

import pytest

try:
    from .ir.document import IRDocumentState
    from .ir.mapper import UnsupportedToolMappingError, map_tool_call_to_ir
    from .ir.types import (
        AddLineParams,
        ExtrudeParams,
        ExternalThreadParams,
        FeatureParameterEditParams,
        FeatureSuppressionParams,
        IROperation,
        PatternFeatureParams,
        SimpleHoleParams,
        TappedHoleParams,
    )
    from .ir.validator import validate_ir_candidate, validate_ir_sequence, validate_operation
except ImportError:  # pragma: no cover
    from backend.backend.ir.document import IRDocumentState
    from backend.backend.ir.mapper import UnsupportedToolMappingError, map_tool_call_to_ir
    from backend.backend.ir.types import (
        AddLineParams,
        ExtrudeParams,
        ExternalThreadParams,
        FeatureParameterEditParams,
        FeatureSuppressionParams,
        IROperation,
        PatternFeatureParams,
        SimpleHoleParams,
        TappedHoleParams,
    )
    from backend.backend.ir.validator import validate_ir_candidate, validate_ir_sequence, validate_operation


def test_mapper_converts_rectangle_and_extrude_to_shared_ir():
    state = IRDocumentState()

    sketch_op = map_tool_call_to_ir(
        {"name": "create_sketch", "input": {"plane_id": "XY", "sketch_id": "sketch_0"}},
        state,
    )
    rect_op = map_tool_call_to_ir(
        {
            "name": "add_rectangle",
            "input": {"sketch_id": "sketch_0", "corner1_u": -25, "corner1_v": -25, "corner2_u": 25, "corner2_v": 25},
        },
        state,
    )
    extrude_op = map_tool_call_to_ir(
        {"name": "extrude_profile", "input": {"sketch_id": "sketch_0", "profile_index": 0, "distance": 50, "operation": "NewBody"}},
        state,
    )

    assert sketch_op.type == "create_sketch"
    assert rect_op.type == "add_rectangle"
    assert extrude_op.type == "extrude"

    assert rect_op.params.center == [0.0, 0.0]
    assert rect_op.params.width == 500.0
    assert rect_op.params.height == 500.0
    assert rect_op.params.rectangle_id == "op_2_rectangle"

    assert extrude_op.params.profile == "sketch_0:profile_0"
    assert extrude_op.params.distance == 500.0
    assert extrude_op.params.direction == "positive"
    assert extrude_op.params.operation == "new"


def test_mapper_synthesizes_sketch_entity_ids_when_model_omits_them():
    state = IRDocumentState()
    map_tool_call_to_ir(
        {"name": "create_sketch", "input": {"plane_id": "XY", "sketch_id": "sketch_0"}},
        state,
    )

    line_op = map_tool_call_to_ir(
        {"name": "add_line", "input": {"sketch_id": "sketch_0", "start_u": 0, "start_v": 0, "end_u": 1, "end_v": 0}},
        state,
    )
    circle_op = map_tool_call_to_ir(
        {"name": "add_circle", "input": {"sketch_id": "sketch_0", "center_u": 0, "center_v": 0, "radius": 1}},
        state,
    )

    assert line_op.params.line_id == "op_2_line"
    assert circle_op.params.circle_id == "op_3_circle"


def test_mapper_preserves_profile_indices_for_multi_profile_extrude():
    state = IRDocumentState()
    op = map_tool_call_to_ir(
        {
            "name": "extrude_profile",
            "input": {
                "sketch_id": "sketch_0",
                "profile_indices": [0, 1, 2],
                "distance": -25,
                "operation": "Cut",
            },
        },
        state,
    )

    assert op.type == "extrude"
    assert op.params.profile_indices == [0, 1, 2]
    assert op.params.profile_index is None
    assert op.params.direction == "negative"
    assert op.params.distance == 250.0
    assert op.params.operation == "cut"
    assert op.params.profile == "sketch_0:profile_0"


def test_validator_rejects_invalid_extrude_distance():
    state = IRDocumentState()
    op = map_tool_call_to_ir(
        {"name": "extrude_profile", "input": {"sketch_id": "s0", "profile_index": 0, "distance": 0}},
        state,
    )

    errors = validate_operation(op)
    assert errors
    assert "distance must be > 0" in errors[0]


@pytest.mark.parametrize(
    "operation",
    [
        IROperation(
            id="op_nan_extrude",
            type="extrude",
            params=ExtrudeParams(
                profile="sketch_0:profile_0",
                distance=float("nan"),
                direction="positive",
                operation="new",
            ),
        ),
        IROperation(
            id="op_inf_line",
            type="add_line",
            params=AddLineParams(sketch="sketch_0", start=[float("inf"), 0.0], end=[1.0, 0.0]),
        ),
        IROperation(
            id="op_inf_hole",
            type="create_simple_hole",
            params=SimpleHoleParams(
                face_ref="face_0",
                center=[0.0, float("-inf"), 0.0],
                diameter=4.0,
                extent_type="distance",
                depth=10.0,
            ),
        ),
        IROperation(
            id="op_inf_tapped",
            type="create_tapped_hole",
            params=TappedHoleParams(
                face_ref="face_0",
                center=[0.0, 0.0, 0.0],
                thread_type="metric",
                thread_size="M6",
                thread_depth=float("inf"),
            ),
        ),
        IROperation(
            id="op_nan_thread",
            type="create_external_thread",
            params=ExternalThreadParams(
                face_ref="face_0",
                thread_type="metric",
                thread_size="M6",
                is_full_length=False,
                thread_length=float("nan"),
            ),
        ),
        IROperation(
            id="op_inf_pattern",
            type="pattern_feature",
            params=PatternFeatureParams(
                pattern_type="rectangular",
                feature_refs=["feature_token_0"],
                count_x=2,
                spacing_x=float("inf"),
            ),
        ),
    ],
)
def test_validator_rejects_nonfinite_numeric_values(operation):
    errors = validate_operation(operation)
    assert any("must be finite" in error or "must be a finite number" in error for error in errors)


def test_sequence_validator_enforces_sketch_dependencies():
    state = IRDocumentState()
    circle = map_tool_call_to_ir(
        {"name": "add_circle", "input": {"sketch_id": "missing_sketch", "center_u": 0, "center_v": 0, "radius": 10}},
        state,
    )
    state.append(circle)

    errors_by_op = validate_ir_sequence(state.operations)
    assert circle.id in errors_by_op
    assert "does not exist yet" in errors_by_op[circle.id][0]


def test_mapper_raises_for_unsupported_tool():
    state = IRDocumentState()
    with pytest.raises(UnsupportedToolMappingError):
        map_tool_call_to_ir({"name": "respond_to_user", "input": {}}, state)


def test_mapper_converts_feature_suppression_to_shared_ir():
    state = IRDocumentState()

    op = map_tool_call_to_ir(
        {
            "name": "suppress_feature",
            "input": {
                "feature_token": "feature-token-1",
                "expected_name": "Shell1",
                "expected_timeline_index": 7,
                "description": "Temporarily disable shell",
            },
        },
        state,
    )

    assert op.type == "set_feature_suppression"
    assert isinstance(op.params, FeatureSuppressionParams)
    assert op.params.feature_ref == "feature-token-1"
    assert op.params.suppress is True
    assert op.params.expected_name == "Shell1"
    assert op.params.expected_timeline_index == 7
    assert op.requires == ["parametric_timeline", "feature_lifecycle"]
    assert validate_operation(op) == []


def test_mapper_converts_feature_unsuppression_to_shared_ir():
    state = IRDocumentState()

    op = map_tool_call_to_ir(
        {
            "name": "unsuppress_feature",
            "input": {"feature_ref": "feature-token-2"},
        },
        state,
    )

    assert op.type == "set_feature_suppression"
    assert isinstance(op.params, FeatureSuppressionParams)
    assert op.params.feature_ref == "feature-token-2"
    assert op.params.suppress is False
    assert validate_ir_candidate(op, []) == []


def test_feature_suppression_ir_rejects_empty_ref_and_bad_expected_index():
    state = IRDocumentState()
    missing_ref = map_tool_call_to_ir({"name": "suppress_feature", "input": {}}, state)

    assert "set_feature_suppression feature_ref must be provided" in validate_operation(missing_ref)

    with pytest.raises(UnsupportedToolMappingError, match="expected_timeline_index"):
        map_tool_call_to_ir(
            {
                "name": "suppress_feature",
                "input": {"feature_token": "feature-token-1", "expected_timeline_index": True},
            },
            state,
        )

    negative_index = map_tool_call_to_ir(
        {
            "name": "suppress_feature",
            "input": {"feature_token": "feature-token-1", "expected_timeline_index": -1},
        },
        state,
    )

    assert "set_feature_suppression expected_timeline_index must be >= 0" in validate_operation(negative_index)


def test_mapper_converts_adjust_feature_parameters_to_shared_ir():
    state = IRDocumentState()

    op = map_tool_call_to_ir(
        {
            "name": "adjust_feature_parameters",
            "input": {
                "feature_token": "feature-token-3",
                "parameters": {
                    "distance": 1.25,
                    "distance_unit": "cm",
                    "radius": 2,
                    "radius_unit": "mm",
                    "circular_total_angle": 3.141592653589793,
                    "circular_total_angle_unit": "rad",
                    "rectangular_count_one": 4,
                },
                "expected_name": "Extrude1",
                "expected_timeline_index": 3,
                "description": "Change extrude distance",
            },
        },
        state,
    )

    assert op.type == "adjust_feature_parameters"
    assert isinstance(op.params, FeatureParameterEditParams)
    assert op.params.feature_ref == "feature-token-3"
    assert op.params.parameters["distance"] == 12.5
    assert op.params.parameters["radius"] == 2.0
    assert op.params.parameters["circular_total_angle"] == 180.0
    assert op.params.parameters["rectangular_count_one"] == 4
    assert op.params.expected_name == "Extrude1"
    assert op.params.expected_timeline_index == 3
    assert op.requires == ["parametric_timeline", "feature_lifecycle", "feature_parameter_edit"]
    assert validate_operation(op) == []


def test_adjust_feature_parameters_ir_rejects_invalid_parameters():
    state = IRDocumentState()

    missing_params = map_tool_call_to_ir(
        {"name": "adjust_feature_parameters", "input": {"feature_token": "feature-token-1"}},
        state,
    )
    assert "adjust_feature_parameters parameters must be a non-empty mapping" in validate_operation(missing_params)

    zero_distance = map_tool_call_to_ir(
        {
            "name": "adjust_feature_parameters",
            "input": {"feature_token": "feature-token-1", "parameters": {"distance": 0}},
        },
        state,
    )
    assert "adjust_feature_parameters distance must be > 0" in validate_operation(zero_distance)

    with pytest.raises(UnsupportedToolMappingError, match="does not support"):
        map_tool_call_to_ir(
            {
                "name": "adjust_feature_parameters",
                "input": {"feature_token": "feature-token-1", "parameters": {"script": "danger"}},
            },
            state,
        )

    with pytest.raises(UnsupportedToolMappingError, match="at least one geometry parameter"):
        map_tool_call_to_ir(
            {
                "name": "adjust_feature_parameters",
                "input": {"feature_token": "feature-token-1", "parameters": {"name": "Base Extrude"}},
            },
            state,
        )


def test_mapper_preserves_non_datum_create_sketch_planes_for_fusion():
    state = IRDocumentState()

    face_op = map_tool_call_to_ir(
        {"name": "create_sketch", "input": {"plane_id": "face_0", "sketch_id": "face_sketch"}},
        state,
        metadata={"source": "fusion"},
    )
    custom_plane_op = map_tool_call_to_ir(
        {"name": "create_sketch", "input": {"plane_id": "port_plane", "sketch_id": "port_sketch"}},
        state,
        metadata={"source": "fusion"},
    )

    assert face_op.params.plane == "face_0"
    assert custom_plane_op.params.plane == "port_plane"
    face_errors = validate_operation(face_op)
    assert any("alias refs like face_N are not allowed" in err for err in face_errors)
    assert not validate_operation(custom_plane_op)


def test_mapper_preserves_studio_non_datum_create_sketch_plane_for_validation():
    state = IRDocumentState()

    op = map_tool_call_to_ir(
        {"name": "create_sketch", "input": {"plane_id": "face_0", "sketch_id": "studio_sketch"}},
        state,
        metadata={"source": "studio"},
    )

    assert op.params.plane == "face_0"
    errors = validate_operation(op)
    assert errors
    assert any("create_sketch plane for studio target must be one of XY/XZ/YZ" in error for error in errors)


def test_mapper_populates_dependencies_for_sketch_flow():
    state = IRDocumentState()

    sketch_op = map_tool_call_to_ir(
        {"name": "create_sketch", "input": {"plane_id": "XY", "sketch_id": "sketch_0"}},
        state,
    )
    state.append(sketch_op)
    circle_op = map_tool_call_to_ir(
        {"name": "add_circle", "input": {"sketch_id": "sketch_0", "center_u": 0, "center_v": 0, "radius": 5}},
        state,
    )
    state.append(circle_op)
    extrude_op = map_tool_call_to_ir(
        {"name": "extrude_profile", "input": {"sketch_id": "sketch_0", "profile_index": 0, "distance": 10}},
        state,
    )

    assert circle_op.dependencies == [sketch_op.id]
    assert extrude_op.dependencies == [circle_op.id, sketch_op.id]


def test_candidate_validation_blocks_uncommitted_dependencies():
    state = IRDocumentState()

    sketch_op = map_tool_call_to_ir(
        {"name": "create_sketch", "input": {"plane_id": "XY", "sketch_id": "sketch_0"}},
        state,
    )
    circle_op = map_tool_call_to_ir(
        {"name": "add_circle", "input": {"sketch_id": "sketch_0", "center_u": 0, "center_v": 0, "radius": 5}},
        state,
        dependency_operations=[sketch_op],
    )

    # Simulate sketch creation failed and never committed.
    errors = validate_ir_candidate(circle_op, committed_operations=[])
    assert any("depends on uncommitted operation" in err for err in errors)


@pytest.mark.parametrize(
    "tool_call, expected_error_fragment",
    [
        (
            {
                "name": "add_rectangle",
                "input": {
                    "sketch_id": "sketch_0",
                    "corner1_u": "bad",
                    "corner1_v": 0,
                    "corner2_u": 10,
                    "corner2_v": 10,
                },
            },
            "corner1_u",
        ),
        (
            {
                "name": "add_circle",
                "input": {"sketch_id": "sketch_0", "center_u": 0, "center_v": 0, "radius": "bad"},
            },
            "radius",
        ),
        (
            {
                "name": "extrude_profile",
                "input": {"sketch_id": "sketch_0", "profile_index": "bad", "distance": 10},
            },
            "profile_index",
        ),
        (
            {
                "name": "extrude_profile",
                "input": {"sketch_id": "sketch_0", "profile_index": 0, "distance": "bad"},
            },
            "distance",
        ),
    ],
)
def test_mapper_rejects_invalid_numeric_inputs(tool_call, expected_error_fragment):
    state = IRDocumentState()

    with pytest.raises(UnsupportedToolMappingError, match=expected_error_fragment):
        map_tool_call_to_ir(tool_call, state)


def test_mapper_rejects_conflicting_profile_index_and_profile_indices():
    state = IRDocumentState()
    with pytest.raises(UnsupportedToolMappingError, match='profile_index" or "profile_indices'):
        map_tool_call_to_ir(
            {
                "name": "extrude_profile",
                "input": {
                    "sketch_id": "sketch_0",
                    "profile_index": 0,
                    "profile_indices": [0, 1],
                    "distance": 10,
                },
            },
            state,
        )


def test_mapper_rejects_empty_profile_indices():
    state = IRDocumentState()
    with pytest.raises(UnsupportedToolMappingError, match="cannot be empty"):
        map_tool_call_to_ir(
            {
                "name": "extrude_profile",
                "input": {
                    "sketch_id": "sketch_0",
                    "profile_indices": [],
                    "distance": 10,
                },
            },
            state,
        )


def test_mapper_canonicalizes_fusion_centimeters_to_ir_millimeters():
    state = IRDocumentState()

    line = map_tool_call_to_ir(
        {
            "name": "add_line",
            "input": {
                "sketch_id": "s0",
                "start_u": 1.25,
                "start_v": -2.0,
                "end_u": 3.0,
                "end_v": 4.5,
            },
        },
        state,
    )
    circle = map_tool_call_to_ir(
        {"name": "add_circle", "input": {"sketch_id": "s0", "center_u": 1, "center_v": 2, "radius": 0.75}},
        state,
    )

    assert line.params.start == [12.5, -20.0]
    assert line.params.end == [30.0, 45.0]
    assert circle.params.center == [10.0, 20.0]
    assert circle.params.radius == 7.5


def test_mapper_covers_revolve_loft_and_feature_operations():
    state = IRDocumentState()

    revolve = map_tool_call_to_ir(
        {
            "name": "revolve_profile",
            "input": {
                "sketch_id": "profile_sketch",
                "profile_index": 1,
                "axis": {"type": "construction", "axis": "z"},
                "extent": {"mode": "full"},
                "operation": "Intersect",
                "feature_name": "Turned Cut",
                "description": "revolve",
            },
        },
        state,
    )
    loft = map_tool_call_to_ir(
        {
            "name": "create_loft",
            "input": {"profile_ids": ["s0", "s1"], "operation": "Join", "description": "loft"},
        },
        state,
    )
    fillet = map_tool_call_to_ir(
        {
            "name": "apply_fillet",
            "input": {"edge_refs": ["edge_0"], "radius": 2.5, "radius_unit": "mm", "description": "round"},
        },
        state,
    )
    hole = map_tool_call_to_ir(
        {
            "name": "create_simple_hole",
            "input": {
                "face_ref": "face_0",
                "center_x": 10,
                "center_y": 20,
                "center_z": 30,
                "diameter": 0.4,
                "diameter_unit": "cm",
                "extent_type": "distance",
                "depth": 12,
                "description": "hole",
            },
        },
        state,
    )

    assert revolve.type == "revolve"
    assert revolve.params.operation == "intersect"
    assert revolve.params.profile == "profile_sketch:profile_1"
    assert loft.type == "loft"
    assert loft.params.operation == "join"
    assert fillet.type == "fillet"
    assert fillet.params.radius == 2.5
    assert fillet.effects.invalidates["edges"] == ["*"]
    assert hole.type == "create_simple_hole"
    assert hole.params.diameter == 4.0
    assert hole.params.depth == 12.0


def test_mapper_captures_phase_4_selector_metadata():
    state = IRDocumentState()

    fillet = map_tool_call_to_ir(
        {
            "name": "apply_fillet",
            "input": {
                "edge_refs": ["edge_0", "edge_1"],
                "radius": 2.0,
                "radius_unit": "mm",
                "include_tangent_edges": False,
            },
        },
        state,
    )
    shell = map_tool_call_to_ir(
        {
            "name": "create_shell",
            "input": {
                "mode": "open",
                "face_refs": ["face_0"],
                "inside_thickness": 1.5,
                "is_tangent_chain": False,
            },
        },
        state,
    )
    select_faces = map_tool_call_to_ir(
        {
            "name": "select_faces",
            "input": {"face_refs": ["face_0", "face_1"], "clear_existing": False},
        },
        state,
    )
    clear_faces = map_tool_call_to_ir(
        {"name": "clear_face_selection", "input": {}},
        state,
    )

    assert fillet.selectors == [{"kind": "edge", "refs": ["edge_0", "edge_1"], "tangent_chain": False}]
    assert shell.selectors == [{"kind": "face", "refs": ["face_0"]}]
    assert select_faces.type == "select_entities"
    assert select_faces.params.kind == "face"
    assert select_faces.params.refs == ["face_0", "face_1"]
    assert select_faces.params.clear_existing is False
    assert select_faces.selectors == [{"kind": "face", "refs": ["face_0", "face_1"]}]
    assert clear_faces.type == "clear_selection"
    assert clear_faces.effects.invalidates == {"selection": ["face"]}


def test_mapper_captures_phase_5_invalidation_semantics():
    state = IRDocumentState()

    delete_feature = map_tool_call_to_ir(
        {
            "name": "delete_feature",
            "input": {
                "feature_token": "feature-token-1",
                "expected_name": "Shell1",
                "expected_timeline_index": 8,
            },
        },
        state,
    )
    parameter_edit = map_tool_call_to_ir(
        {
            "name": "adjust_feature_parameters",
            "input": {
                "feature_token": "feature-token-2",
                "parameters": {"distance": 5.0},
                "expected_name": "Extrude1",
            },
        },
        state,
    )
    suppress_feature = map_tool_call_to_ir(
        {
            "name": "suppress_feature",
            "input": {"feature_token": "feature-token-3"},
        },
        state,
    )
    jump = map_tool_call_to_ir(
        {
            "name": "jump_to_timeline_position",
            "input": {"target_index": 4, "reason": "rollback failed branch"},
        },
        state,
    )

    for operation, feature_ref in (
        (delete_feature, "feature-token-1"),
        (parameter_edit, "feature-token-2"),
        (suppress_feature, "feature-token-3"),
    ):
        assert operation.effects.modifies == {"features": [feature_ref]}
        assert operation.effects.invalidates == {
            "features": ["downstream"],
            "faces": ["*"],
            "edges": ["*"],
            "bodies": ["topology_generation"],
        }

    assert jump.requires == ["parametric_timeline", "document_revision"]
    assert jump.effects.invalidates == {
        "operations": ["after_marker"],
        "features": ["after_marker"],
        "faces": ["*"],
        "edges": ["*"],
    }


def test_validator_rejects_invalid_widened_operations():
    state = IRDocumentState()
    bad_shell = map_tool_call_to_ir(
        {
            "name": "create_shell",
            "input": {
                "mode": "closed",
                "face_refs": ["face_0"],
                "inside_thickness": 2,
                "description": "bad shell",
            },
        },
        state,
    )
    bad_counterbore = map_tool_call_to_ir(
        {
            "name": "create_counterbore_hole",
            "input": {
                "face_ref": "face_0",
                "center_x": 0,
                "center_y": 0,
                "center_z": 0,
                "hole_diameter": 8,
                "hole_depth": 10,
                "counterbore_diameter": 6,
                "counterbore_depth": 2,
                "description": "bad counterbore",
            },
        },
        state,
    )

    assert any("closed mode must not include face refs" in err for err in validate_operation(bad_shell))
    assert any("counterbore_diameter must be larger" in err for err in validate_operation(bad_counterbore))


def test_list_sketch_profiles_does_not_satisfy_profile_geometry_requirement():
    state = IRDocumentState()
    sketch = map_tool_call_to_ir(
        {"name": "create_sketch", "input": {"plane_id": "XY", "sketch_id": "empty_sketch"}},
        state,
    )
    state.append(sketch)
    profiles = map_tool_call_to_ir(
        {"name": "list_sketch_profiles", "input": {"sketch_id": "empty_sketch"}},
        state,
    )
    extrude = map_tool_call_to_ir(
        {
            "name": "extrude_profile",
            "input": {"sketch_id": "empty_sketch", "profile_index": 0, "distance": 1.0},
        },
        state,
        dependency_operations=[sketch, profiles],
    )

    profile_errors = validate_ir_candidate(profiles, [sketch])
    extrude_errors = validate_ir_candidate(extrude, [sketch, profiles])

    assert any("requires committed sketch geometry" in err for err in profile_errors)
    assert any("requires at least one committed profile operation" in err for err in extrude_errors)


def test_validator_rejects_extrude_after_single_open_line():
    state = IRDocumentState()
    sketch = map_tool_call_to_ir(
        {"name": "create_sketch", "input": {"plane_id": "XY", "sketch_id": "open_sketch"}},
        state,
    )
    state.append(sketch)
    line = map_tool_call_to_ir(
        {
            "name": "add_line",
            "input": {
                "sketch_id": "open_sketch",
                "start_u": 0,
                "start_v": 0,
                "end_u": 1,
                "end_v": 0,
            },
        },
        state,
    )
    state.append(line)

    extrude = map_tool_call_to_ir(
        {
            "name": "extrude_profile",
            "input": {"sketch_id": "open_sketch", "profile_index": 0, "distance": 1.0},
        },
        state,
    )

    errors = validate_ir_candidate(extrude, state.operations)
    assert any("requires at least one committed profile operation" in err for err in errors)


def test_validator_accepts_successful_profile_inspection_as_profile_source():
    state = IRDocumentState()
    sketch = map_tool_call_to_ir(
        {"name": "create_sketch", "input": {"plane_id": "XY", "sketch_id": "closed_by_lines"}},
        state,
    )
    state.append(sketch)
    line = map_tool_call_to_ir(
        {
            "name": "add_line",
            "input": {
                "sketch_id": "closed_by_lines",
                "start_u": 0,
                "start_v": 0,
                "end_u": 1,
                "end_v": 0,
            },
        },
        state,
    )
    state.append(line)
    profiles = map_tool_call_to_ir(
        {"name": "list_sketch_profiles", "input": {"sketch_id": "closed_by_lines"}},
        state,
    )
    committed_profiles = replace(
        profiles,
        target_results=[
            {
                "target": "fusion",
                "success": True,
                "raw_result": {"profile_count": 1, "profiles": [{"index": 0}]},
            }
        ],
    )
    state.append(committed_profiles)

    extrude = map_tool_call_to_ir(
        {
            "name": "extrude_profile",
            "input": {"sketch_id": "closed_by_lines", "profile_index": 0, "distance": 1.0},
        },
        state,
    )

    errors = validate_ir_candidate(extrude, state.operations)
    assert not any("requires at least one committed profile operation" in err for err in errors)


def test_thread_ir_validation_uses_thread_catalog():
    state = IRDocumentState()
    tapped = map_tool_call_to_ir(
        {
            "name": "create_tapped_hole",
            "input": {
                "face_ref": "face_0",
                "center_x": 0,
                "center_y": 0,
                "center_z": 0,
                "thread_type": "metric",
                "thread_size": "1/4-20",
                "thread_depth": 10,
                "description": "wrong catalog",
            },
        },
        state,
    )
    external = map_tool_call_to_ir(
        {
            "name": "create_external_thread",
            "input": {
                "face_ref": "face_1",
                "thread_type": "unc",
                "thread_size": "M6",
                "is_full_length": True,
                "description": "wrong catalog",
            },
        },
        state,
    )

    assert any("thread spec invalid" in err for err in validate_operation(tapped))
    assert any("thread spec invalid" in err for err in validate_operation(external))
