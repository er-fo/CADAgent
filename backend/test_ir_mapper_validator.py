import pytest

try:
    from .ir.document import IRDocumentState
    from .ir.mapper import UnsupportedToolMappingError, map_tool_call_to_ir
    from .ir.validator import validate_ir_sequence, validate_operation
except ImportError:  # pragma: no cover
    from backend.backend.ir.document import IRDocumentState
    from backend.backend.ir.mapper import UnsupportedToolMappingError, map_tool_call_to_ir
    from backend.backend.ir.validator import validate_ir_sequence, validate_operation


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
    assert rect_op.params.width == 50.0
    assert rect_op.params.height == 50.0

    assert extrude_op.params.profile == "sketch_0:profile_0"
    assert extrude_op.params.direction == "positive"
    assert extrude_op.params.operation == "new"


def test_validator_rejects_invalid_extrude_distance():
    state = IRDocumentState()
    op = map_tool_call_to_ir(
        {"name": "extrude_profile", "input": {"sketch_id": "s0", "profile_index": 0, "distance": 0}},
        state,
    )

    errors = validate_operation(op)
    assert errors
    assert "distance must be > 0" in errors[0]


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
        map_tool_call_to_ir({"name": "apply_fillet", "input": {}}, state)


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
    assert not validate_operation(face_op)
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
