import asyncio

from pathlib import Path

import pytest

try:
    from .backends.build123d import Build123dTargetExecutor
    from .backends.build123d.translator import translate_ir_document_to_build123d
    from .backends.fusion.executor import FusionTargetExecutor
    from .backends.fusion.translator import translate_ir_to_fusion_tool_call
    from .entity_store import EntityStore
    from .ir.types import (
        AddCircleParams,
        AddLineParams,
        AddRectangleParams,
        CreateConstructionPlaneParams,
        CreateSketchParams,
        ExtrudeParams,
        ExternalThreadParams,
        FilletParams,
        IRDocument,
        IROperation,
        PatternFeatureParams,
        RevolveParams,
        SelectEntitiesParams,
        ShellParams,
        SimpleHoleParams,
    )
except ImportError:  # pragma: no cover
    from backend.backend.backends.build123d import Build123dTargetExecutor
    from backend.backend.backends.build123d.translator import translate_ir_document_to_build123d
    from backend.backend.backends.fusion.executor import FusionTargetExecutor
    from backend.backend.backends.fusion.translator import translate_ir_to_fusion_tool_call
    from backend.backend.entity_store import EntityStore
    from backend.backend.ir.types import (
        AddCircleParams,
        AddLineParams,
        AddRectangleParams,
        CreateConstructionPlaneParams,
        CreateSketchParams,
        ExtrudeParams,
        ExternalThreadParams,
        FilletParams,
        IRDocument,
        IROperation,
        PatternFeatureParams,
        RevolveParams,
        SelectEntitiesParams,
        ShellParams,
        SimpleHoleParams,
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
    assert tool_input["distance"] == 4.0
    assert tool_input["operation"] == "NewBody"


def test_fusion_translator_preserves_multi_profile_extrude():
    extrude = IROperation(
        id="op_multi",
        type="extrude",
        params=ExtrudeParams(
            profile="sketch_0:profile_0",
            distance=2.5,
            direction="negative",
            operation="cut",
            sketch="sketch_0",
            profile_indices=[0, 1, 2],
        ),
    )

    tool_name, tool_input = translate_ir_to_fusion_tool_call(extrude)
    assert tool_name == "extrude_profile"
    assert tool_input["sketch_id"] == "sketch_0"
    assert tool_input["profile_indices"] == [0, 1, 2]
    assert "profile_index" not in tool_input
    assert tool_input["distance"] == -0.25
    assert tool_input["operation"] == "Cut"


def test_fusion_translator_converts_canonical_mm_to_fusion_cm_for_sketch_tools():
    line = IROperation(
        id="op_line",
        type="add_line",
        params=AddLineParams(sketch="sketch_0", start=[10.0, -20.0], end=[35.0, 45.0], line_id="edge_a"),
    )
    tool_name, tool_input = translate_ir_to_fusion_tool_call(line)
    assert tool_name == "add_line"
    assert tool_input["start_u"] == 1.0
    assert tool_input["start_v"] == -2.0
    assert tool_input["end_u"] == 3.5
    assert tool_input["end_v"] == 4.5


def test_fusion_translator_covers_widened_ir_operations():
    plane = IROperation(
        id="op_plane",
        type="create_construction_plane",
        params=CreateConstructionPlaneParams(
            plane="offset_plane",
            mode="offset_from_datum",
            base_datum_plane="XY",
            offset=25.0,
            description="offset",
        ),
    )
    tool_name, tool_input = translate_ir_to_fusion_tool_call(plane)
    assert tool_name == "create_construction_plane"
    assert tool_input["offset_cm"] == 2.5

    revolve = IROperation(
        id="op_revolve",
        type="revolve",
        params=RevolveParams(
            profile="sketch_0:profile_0",
            sketch="sketch_0",
            profile_index=0,
            axis={"type": "construction", "axis": "z"},
            extent={"mode": "full"},
            operation="intersect",
        ),
    )
    tool_name, tool_input = translate_ir_to_fusion_tool_call(revolve)
    assert tool_name == "revolve_profile"
    assert tool_input["operation"] == "Intersect"

    fillet = IROperation(
        id="op_fillet",
        type="fillet",
        params=FilletParams(edge_refs=["edge_0"], radius=3.0),
    )
    tool_name, tool_input = translate_ir_to_fusion_tool_call(fillet)
    assert tool_name == "apply_fillet"
    assert tool_input["radius"] == 3.0
    assert tool_input["radius_unit"] == "mm"


class _FusionExecutorManager:
    def __init__(self, store: EntityStore, result: dict):
        self.store = store
        self.result = result
        self.sent_messages = []

    def get_entity_store(self, session_id: str) -> EntityStore:
        return self.store

    async def send_message(self, session_id: str, payload: dict) -> None:
        self.sent_messages.append(payload)

    async def wait_for_fusion_result(self, session_id: str, timeout=None):
        return self.result

    async def store_fusion_result(self, session_id: str, payload: dict) -> None:
        pass


def test_fusion_executor_resolves_ir_feature_refs_before_payload_send():
    async def _run():
        store = EntityStore()
        await store.register_entities(
            "edge",
            [{"entity_token": "edge_token_0", "length": 10.0}],
        )
        manager = _FusionExecutorManager(
            store,
            {"tool_use_id": "toolu_fillet", "success": True, "message": "fillet ok"},
        )
        executor = FusionTargetExecutor(manager)  # type: ignore[arg-type]
        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_fillet",
                type="fillet",
                params=FilletParams(edge_refs=["e0"], radius=2.0),
            ),
            tool_use_id="toolu_fillet",
        )

        assert result.success
        assert manager.sent_messages[0]["type"] == "feature_operation"
        assert manager.sent_messages[0]["parameters"]["entity_tokens"] == ["edge_token_0"]
        assert "edge_refs" not in manager.sent_messages[0]["parameters"]

    asyncio.run(_run())


def test_fusion_executor_resolves_ir_hole_face_ref_before_payload_send():
    async def _run():
        store = EntityStore()
        await store.register_entities(
            "face",
            [{"entity_token": "face_token_0", "surface_type": "planar", "area": 10.0}],
        )
        manager = _FusionExecutorManager(
            store,
            {"tool_use_id": "toolu_hole", "success": True, "message": "hole ok"},
        )
        executor = FusionTargetExecutor(manager)  # type: ignore[arg-type]
        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_hole",
                type="create_simple_hole",
                params=SimpleHoleParams(
                    face_ref="face_0",
                    center=[0.0, 0.0, 0.0],
                    diameter=4.0,
                    extent_type="through_all",
                ),
            ),
            tool_use_id="toolu_hole",
        )

        assert result.success
        assert manager.sent_messages[0]["parameters"]["face_token"] == "face_token_0"
        assert "face_ref" not in manager.sent_messages[0]["parameters"]

    asyncio.run(_run())


def test_fusion_executor_resolves_codegen_construction_plane_refs():
    async def _run():
        store = EntityStore()
        await store.register_entities(
            "face",
            [{"entity_token": "face_token_0", "surface_type": "planar", "area": 10.0}],
        )
        manager = _FusionExecutorManager(
            store,
            {"tool_use_id": "toolu_plane", "success": True, "message": "plane ok"},
        )
        executor = FusionTargetExecutor(manager)  # type: ignore[arg-type]

        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_plane",
                type="create_construction_plane",
                params=CreateConstructionPlaneParams(
                    plane="face_plane",
                    mode="face_normal",
                    description="plane on face",
                    face="face_0",
                ),
            ),
            tool_use_id="toolu_plane",
        )

        assert result.success
        assert manager.sent_messages[0]["type"] == "execute_code"
        assert 'face_token="face_token_0"' in manager.sent_messages[0]["code"]
        assert 'face_token="face_0"' not in manager.sent_messages[0]["code"]

    asyncio.run(_run())


def test_fusion_executor_preserves_custom_construction_plane_ids_for_sketches():
    async def _run():
        manager = _FusionExecutorManager(
            EntityStore(),
            {"tool_use_id": "toolu_sketch", "success": True, "message": "sketch ok"},
        )
        executor = FusionTargetExecutor(manager)  # type: ignore[arg-type]

        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_sketch",
                type="create_sketch",
                params=CreateSketchParams(plane="plane_0", sketch="sketch_on_plane"),
            ),
            tool_use_id="toolu_sketch",
        )

        assert result.success
        assert result.data["tool_input"]["plane_id"] == "plane_0"
        assert manager.sent_messages
        assert "plane_0" in manager.sent_messages[0]["code"]

    asyncio.run(_run())


def test_fusion_executor_fails_unresolved_face_alias_plane_for_sketches():
    async def _run():
        manager = _FusionExecutorManager(
            EntityStore(),
            {"tool_use_id": "toolu_sketch", "success": True, "message": "should not run"},
        )
        executor = FusionTargetExecutor(manager)  # type: ignore[arg-type]

        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_sketch",
                type="create_sketch",
                params=CreateSketchParams(plane="face_0", sketch="sketch_on_missing_face"),
            ),
            tool_use_id="toolu_sketch",
        )

        assert not result.success
        assert "Unresolved face alias for create_sketch plane" in result.message
        assert manager.sent_messages == []

    asyncio.run(_run())


def test_fusion_executor_resolves_codegen_revolve_refs():
    async def _run():
        store = EntityStore()
        await store.register_entities(
            "face",
            [{"entity_token": "face_token_0", "surface_type": "planar", "area": 10.0}],
        )
        await store.register_entities(
            "edge",
            [{"entity_token": "edge_token_0", "length": 10.0}],
        )
        manager = _FusionExecutorManager(
            store,
            {"tool_use_id": "toolu_revolve", "success": True, "message": "revolve ok"},
        )
        executor = FusionTargetExecutor(manager)  # type: ignore[arg-type]

        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_revolve",
                type="revolve",
                params=RevolveParams(
                    profile="sketch_0:profile_0",
                    sketch="sketch_0",
                    profile_index=0,
                    axis={"type": "edge", "edge_token": "e0"},
                    extent={"mode": "to", "to_entity_token": "face_0"},
                    operation="new",
                ),
            ),
            tool_use_id="toolu_revolve",
        )

        assert result.success
        code = manager.sent_messages[0]["code"]
        assert "'edge_token': 'edge_token_0'" in code
        assert "'to_entity_token': 'face_token_0'" in code
        assert "'edge_token': 'e0'" not in code
        assert "'to_entity_token': 'face_0'" not in code

    asyncio.run(_run())


def test_fusion_executor_resolves_shell_external_thread_and_selection_refs():
    async def _run():
        store = EntityStore()
        await store.register_entities(
            "face",
            [{"entity_token": "face_token_0", "surface_type": "planar", "area": 10.0}],
        )
        await store.register_entities(
            "edge",
            [{"entity_token": "edge_token_0", "length": 10.0}],
        )
        manager = _FusionExecutorManager(
            store,
            {"tool_use_id": "toolu_any", "success": True, "message": "ok"},
        )
        executor = FusionTargetExecutor(manager)  # type: ignore[arg-type]

        shell_result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_shell",
                type="shell",
                params=ShellParams(mode="open", face_refs=["face_0"], inside_thickness=2.0),
            ),
            tool_use_id="toolu_any",
        )
        thread_result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_thread",
                type="create_external_thread",
                params=ExternalThreadParams(
                    face_ref="face_0",
                    thread_type="metric",
                    thread_size="M6",
                    is_full_length=True,
                ),
            ),
            tool_use_id="toolu_any",
        )
        select_result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_select",
                type="select_entities",
                params=SelectEntitiesParams(kind="edge", refs=["e0"]),
            ),
            tool_use_id="toolu_any",
        )

        assert shell_result.success
        assert thread_result.success
        assert select_result.success
        assert manager.sent_messages[0]["parameters"]["entity_tokens"] == ["face_token_0"]
        assert "face_refs" not in manager.sent_messages[0]["parameters"]
        assert manager.sent_messages[1]["parameters"]["face_token"] == "face_token_0"
        assert "face_ref" not in manager.sent_messages[1]["parameters"]
        assert manager.sent_messages[2]["parameters"]["entity_tokens"] == ["edge_token_0"]
        assert "edge_refs" not in manager.sent_messages[2]["parameters"]

    asyncio.run(_run())


def test_fusion_executor_fails_closed_for_pattern_auto_last_without_workflow_preparation():
    async def _run():
        manager = _FusionExecutorManager(
            EntityStore(),
            {"tool_use_id": "toolu_pattern", "success": True, "message": "should not run"},
        )
        executor = FusionTargetExecutor(manager)  # type: ignore[arg-type]

        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_pattern",
                type="pattern_feature",
                params=PatternFeatureParams(pattern_type="rectangular", feature_refs=["auto_last"]),
            ),
            tool_use_id="toolu_pattern",
        )

        assert not result.success
        assert "auto_last" in result.message
        assert manager.sent_messages == []

    asyncio.run(_run())


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


def test_fusion_translator_rejects_conflicting_profile_index_fields():
    operation = IROperation(
        id="op_bad_profiles",
        type="extrude",
        params=ExtrudeParams(
            profile="sketch_0:profile_0",
            distance=5.0,
            direction="positive",
            operation="new",
            sketch="sketch_0",
            profile_index=0,
            profile_indices=[0, 1],
        ),
    )

    with pytest.raises(ValueError, match='both "profile_index" and "profile_indices"'):
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


def test_build123d_translator_fails_explicitly_for_unsupported_widened_ir():
    document = IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_line",
                type="add_line",
                params=AddLineParams(sketch="sketch_0", start=[0.0, 0.0], end=[10.0, 0.0]),
            )
        ],
        metadata={"source": "test"},
    )

    with pytest.raises(ValueError, match="Unsupported IR operation for build123d translator: add_line"):
        translate_ir_document_to_build123d(document)


def test_fusion_translator_synthesizes_missing_sketch_entity_ids():
    rect_name, rect_input = translate_ir_to_fusion_tool_call(
        IROperation(
            id="op_rect",
            type="add_rectangle",
            params=AddRectangleParams(sketch="sketch_0", center=[0.0, 0.0], width=10.0, height=6.0),
        )
    )
    line_name, line_input = translate_ir_to_fusion_tool_call(
        IROperation(
            id="op_line",
            type="add_line",
            params=AddLineParams(sketch="sketch_0", start=[0.0, 0.0], end=[10.0, 0.0]),
        )
    )
    circle_name, circle_input = translate_ir_to_fusion_tool_call(
        IROperation(
            id="op_circle",
            type="add_circle",
            params=AddCircleParams(sketch="sketch_0", center=[0.0, 0.0], radius=2.0),
        )
    )

    assert rect_name == "add_rectangle"
    assert rect_input["rectangle_id"] == "op_rect_rectangle"
    assert line_name == "add_line"
    assert line_input["line_id"] == "op_line_line"
    assert circle_name == "add_circle"
    assert circle_input["circle_id"] == "op_circle_circle"


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
