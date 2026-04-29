import asyncio

from pathlib import Path

import pytest

try:
    from .backends.build123d import Build123dTargetExecutor
    from .backends.build123d.translator import translate_ir_document_to_build123d
    from .backends.fusion.translator import translate_ir_to_fusion_tool_call
    from .ir.types import (
        AddCircleParams,
        AddRectangleParams,
        CreateSketchParams,
        ExtrudeParams,
        IRDocument,
        IROperation,
    )
except ImportError:  # pragma: no cover
    from backend.backend.backends.build123d import Build123dTargetExecutor
    from backend.backend.backends.build123d.translator import translate_ir_document_to_build123d
    from backend.backend.backends.fusion.translator import translate_ir_to_fusion_tool_call
    from backend.backend.ir.types import (
        AddCircleParams,
        AddRectangleParams,
        CreateSketchParams,
        ExtrudeParams,
        IRDocument,
        IROperation,
    )


try:
    import build123d  # noqa: F401
    HAS_BUILD123D = True
except Exception:
    HAS_BUILD123D = False


def test_fusion_translator_maps_shared_ir_to_existing_tool_schema():
    create_sketch = IROperation(
        id="op_1",
        type="create_sketch",
        params=CreateSketchParams(plane="XY", sketch="sketch_0"),
    )
    tool_name, tool_input = translate_ir_to_fusion_tool_call(create_sketch)
    assert tool_name == "create_sketch"
    assert tool_input["plane_id"] == "XY"
    assert tool_input["sketch_id"] == "sketch_0"

    extrude = IROperation(
        id="op_2",
        type="extrude",
        params=ExtrudeParams(
            profile="sketch_0:profile_0",
            distance=40.0,
            direction="positive",
            operation="new",
            sketch="sketch_0",
            profile_index=0,
        ),
    )
    tool_name, tool_input = translate_ir_to_fusion_tool_call(extrude)
    assert tool_name == "extrude_profile"
    assert tool_input["sketch_id"] == "sketch_0"
    assert tool_input["profile_index"] == 0
    assert tool_input["distance"] == 40.0
    assert tool_input["operation"] == "NewBody"


def test_fusion_translator_rejects_unsupported_extrude_operation():
    operation = IROperation(
        id="op_bad_op",
        type="extrude",
        params=ExtrudeParams(
            profile="sketch_0:profile_0",
            distance=5.0,
            direction="positive",
            operation="unsupported",  # type: ignore[arg-type]
            sketch="sketch_0",
            profile_index=0,
        ),
    )

    with pytest.raises(ValueError, match="Unsupported Fusion extrude operation"):
        translate_ir_to_fusion_tool_call(operation)


def test_fusion_translator_rejects_malformed_profile_reference():
    operation = IROperation(
        id="op_bad_profile",
        type="extrude",
        params=ExtrudeParams(
            profile="malformed_profile_ref",
            distance=5.0,
            direction="positive",
            operation="new",
            sketch=None,
            profile_index=None,
        ),
    )

    with pytest.raises(ValueError, match="Unsupported profile reference format"):
        translate_ir_to_fusion_tool_call(operation)


def test_fusion_translator_rejects_unresolved_face_alias_plane():
    operation = IROperation(
        id="op_face_alias",
        type="create_sketch",
        params=CreateSketchParams(plane="face_0", sketch="sketch_0"),
    )

    with pytest.raises(ValueError, match="Unresolved face alias for create_sketch plane"):
        translate_ir_to_fusion_tool_call(operation)


def test_build123d_translator_rejects_unsupported_plane():
    document = IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_bad_plane",
                type="create_sketch",
                params=CreateSketchParams(plane="AB", sketch="sketch_0"),  # type: ignore[arg-type]
            )
        ],
        metadata={"source": "test"},
    )

    with pytest.raises(ValueError, match="Unsupported sketch plane for build123d translator"):
        translate_ir_document_to_build123d(document)


def test_build123d_translator_rejects_unsupported_extrude_operation():
    document = IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_1",
                type="create_sketch",
                params=CreateSketchParams(plane="XY", sketch="sketch_0"),
            ),
            IROperation(
                id="op_2",
                type="add_rectangle",
                params=AddRectangleParams(sketch="sketch_0", center=[0.0, 0.0], width=10.0, height=10.0),
            ),
            IROperation(
                id="op_bad_mode",
                type="extrude",
                params=ExtrudeParams(
                    profile="sketch_0:profile_0",
                    distance=10.0,
                    direction="positive",
                    operation="unsupported",  # type: ignore[arg-type]
                    sketch="sketch_0",
                    profile_index=0,
                ),
            ),
        ],
        metadata={"source": "test"},
    )

    with pytest.raises(ValueError, match="Unsupported extrude operation for build123d translator"):
        translate_ir_document_to_build123d(document)


def test_build123d_translator_emits_fail_closed_sketch_plane_guards():
    document = IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_1",
                type="add_circle",
                params=AddCircleParams(sketch="missing_sketch", center=[0.0, 0.0], radius=2.0),
            )
        ],
        metadata={"source": "test"},
    )

    program = translate_ir_document_to_build123d(document)
    assert "not in _sketch_planes" in program.code
    assert "Sketch plane missing for 'missing_sketch'" in program.code
    assert "_sketch_planes.get('missing_sketch', Plane.XY)" not in program.code


def _build_cube_document() -> IRDocument:
    return IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_1",
                type="create_sketch",
                params=CreateSketchParams(plane="XY", sketch="sketch_0"),
            ),
            IROperation(
                id="op_2",
                type="add_rectangle",
                params=AddRectangleParams(sketch="sketch_0", center=[0.0, 0.0], width=50.0, height=50.0),
            ),
            IROperation(
                id="op_3",
                type="extrude",
                params=ExtrudeParams(
                    profile="sketch_0:profile_0",
                    distance=50.0,
                    direction="positive",
                    operation="new",
                    sketch="sketch_0",
                    profile_index=0,
                ),
            ),
        ],
        metadata={"source": "test"},
    )


def _build_cylinder_document() -> IRDocument:
    return IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_1",
                type="create_sketch",
                params=CreateSketchParams(plane="XY", sketch="sketch_0"),
            ),
            IROperation(
                id="op_2",
                type="add_circle",
                params=AddCircleParams(sketch="sketch_0", center=[0.0, 0.0], radius=10.0),
            ),
            IROperation(
                id="op_3",
                type="extrude",
                params=ExtrudeParams(
                    profile="sketch_0:profile_0",
                    distance=40.0,
                    direction="positive",
                    operation="new",
                    sketch="sketch_0",
                    profile_index=0,
                ),
            ),
        ],
        metadata={"source": "test"},
    )


@pytest.mark.skipif(not HAS_BUILD123D, reason="build123d is required for real-geometry adapter tests")
def test_build123d_executor_creates_real_cube_and_exports_step(tmp_path: Path):
    executor = Build123dTargetExecutor()
    result = asyncio.run(executor.execute_document("s_cube", _build_cube_document(), request_id="cube"))

    assert result.success, result.message
    entities = result.data["entities"]
    assert abs(entities["volume_mm3"] - 125000.0) < 1.0
    assert entities["faces"] == 6
    assert entities["edges"] == 12
    assert entities["vertices"] == 8

    out_path = executor.export_step("s_cube", str(tmp_path / "cube.step"))
    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 100

    content = Path(out_path).read_text(errors="ignore")
    assert "ISO-10303-21" in content


@pytest.mark.skipif(not HAS_BUILD123D, reason="build123d is required for real-geometry adapter tests")
def test_build123d_executor_creates_real_cylinder():
    executor = Build123dTargetExecutor()
    result = asyncio.run(executor.execute_document("s_cyl", _build_cylinder_document(), request_id="cylinder"))

    assert result.success, result.message
    entities = result.data["entities"]
    expected_volume = 3.141592653589793 * 10.0 * 10.0 * 40.0
    assert abs(entities["volume_mm3"] - expected_volume) < 10.0
    assert entities["faces"] == 3


@pytest.mark.skipif(not HAS_BUILD123D, reason="build123d is required for real-geometry adapter tests")
def test_build123d_executor_accepts_intermediate_ir_without_solid():
    executor = Build123dTargetExecutor()
    document = IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_1",
                type="create_sketch",
                params=CreateSketchParams(plane="XY", sketch="sketch_0"),
            )
        ],
        metadata={"source": "test"},
    )

    result = asyncio.run(executor.execute_document("s_partial", document, request_id="partial"))
    assert result.success
    assert result.data["entities"] == {}
