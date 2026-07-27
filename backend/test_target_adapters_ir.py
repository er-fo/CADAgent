import asyncio

from pathlib import Path

import pytest

try:
    from .backends.build123d import Build123dTargetExecutor
    from .backends.build123d.translator import Build123dCapabilityError, translate_ir_document_to_build123d
    from .backends.fusion.executor import FusionTargetExecutor
    from .backends.fusion.translator import translate_ir_to_fusion_tool_call
    from .entity_store import EntityStore
    from .ir.types import (
        AddArcParams,
        ChamferParams,
        AddCircleParams,
        AddLineParams,
        AddRectangleParams,
        CounterboreHoleParams,
        CreateConstructionPlaneParams,
        CreateSketchParams,
        DeleteFeatureParams,
        ExtrudeParams,
        ExternalThreadParams,
        FeatureParameterEditParams,
        FeatureSuppressionParams,
        FilletParams,
        IRDocument,
        IROperation,
        LoftParams,
        PatternFeatureParams,
        RevolveParams,
        SelectEntitiesParams,
        ShellParams,
        SimpleHoleParams,
        TappedHoleParams,
    )
except ImportError:  # pragma: no cover
    from backend.backend.backends.build123d import Build123dTargetExecutor
    from backend.backend.backends.build123d.translator import Build123dCapabilityError, translate_ir_document_to_build123d
    from backend.backend.backends.fusion.executor import FusionTargetExecutor
    from backend.backend.backends.fusion.translator import translate_ir_to_fusion_tool_call
    from backend.backend.entity_store import EntityStore
    from backend.backend.ir.types import (
        AddArcParams,
        ChamferParams,
        AddCircleParams,
        AddLineParams,
        AddRectangleParams,
        CounterboreHoleParams,
        CreateConstructionPlaneParams,
        CreateSketchParams,
        DeleteFeatureParams,
        ExtrudeParams,
        ExternalThreadParams,
        FeatureParameterEditParams,
        FeatureSuppressionParams,
        FilletParams,
        IRDocument,
        IROperation,
        LoftParams,
        PatternFeatureParams,
        RevolveParams,
        SelectEntitiesParams,
        ShellParams,
        SimpleHoleParams,
        TappedHoleParams,
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

    suppress = IROperation(
        id="op_suppress",
        type="set_feature_suppression",
        params=FeatureSuppressionParams(
            feature_ref="feature-token-1",
            suppress=True,
            description="Temporarily disable shell",
            expected_name="Shell1",
            expected_timeline_index=7,
        ),
    )
    tool_name, tool_input = translate_ir_to_fusion_tool_call(suppress)
    assert tool_name == "suppress_feature"
    assert tool_input["feature_token"] == "feature-token-1"
    assert tool_input["expected_name"] == "Shell1"
    assert tool_input["expected_timeline_index"] == 7

    unsuppress = IROperation(
        id="op_unsuppress",
        type="set_feature_suppression",
        params=FeatureSuppressionParams(feature_ref="feature-token-1", suppress=False),
    )
    tool_name, tool_input = translate_ir_to_fusion_tool_call(unsuppress)
    assert tool_name == "unsuppress_feature"
    assert tool_input["feature_token"] == "feature-token-1"

    parameter_edit = IROperation(
        id="op_parameter_edit",
        type="adjust_feature_parameters",
        params=FeatureParameterEditParams(
            feature_ref="feature-token-3",
            parameters={"distance": 12.5, "circular_total_angle": 180.0},
            description="Change extrude distance",
            expected_name="Extrude1",
            expected_timeline_index=3,
        ),
    )
    tool_name, tool_input = translate_ir_to_fusion_tool_call(parameter_edit)
    assert tool_name == "adjust_feature_parameters"
    assert tool_input["feature_token"] == "feature-token-3"
    assert tool_input["parameters"]["distance"] == 12.5
    assert tool_input["parameters"]["distance_unit"] == "mm"
    assert tool_input["parameters"]["circular_total_angle_unit"] == "deg"
    assert tool_input["expected_name"] == "Extrude1"
    assert tool_input["expected_timeline_index"] == 3


def test_fusion_translator_rejects_name_only_parameter_edit_ir():
    parameter_edit = IROperation(
        id="op_parameter_rename",
        type="adjust_feature_parameters",
        params=FeatureParameterEditParams(
            feature_ref="feature-token-3",
            parameters={"name": "Base Extrude"},
        ),
    )

    with pytest.raises(ValueError, match="at least one geometry parameter"):
        translate_ir_to_fusion_tool_call(parameter_edit)


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


def test_fusion_executor_sends_suppression_ir_as_feature_payload():
    async def _run():
        manager = _FusionExecutorManager(
            EntityStore(),
            {"tool_use_id": "toolu_suppress", "success": True, "message": "suppressed"},
        )
        executor = FusionTargetExecutor(manager)  # type: ignore[arg-type]
        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_suppress",
                type="set_feature_suppression",
                params=FeatureSuppressionParams(
                    feature_ref="feature-token-1",
                    suppress=True,
                    expected_name="Shell1",
                    expected_timeline_index=7,
                ),
            ),
            tool_use_id="toolu_suppress",
        )

        assert result.success
        payload = manager.sent_messages[0]
        assert payload["type"] == "feature_operation"
        assert payload["operation"] == "suppress_feature"
        assert payload["parameters"]["feature_token"] == "feature-token-1"
        assert payload["parameters"]["expected_name"] == "Shell1"
        assert payload["parameters"]["expected_timeline_index"] == 7

    asyncio.run(_run())


def test_fusion_executor_sends_unsuppression_ir_as_feature_payload():
    async def _run():
        manager = _FusionExecutorManager(
            EntityStore(),
            {"tool_use_id": "toolu_unsuppress", "success": True, "message": "unsuppressed"},
        )
        executor = FusionTargetExecutor(manager)  # type: ignore[arg-type]
        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_unsuppress",
                type="set_feature_suppression",
                params=FeatureSuppressionParams(feature_ref="feature-token-1", suppress=False),
            ),
            tool_use_id="toolu_unsuppress",
        )

        assert result.success
        payload = manager.sent_messages[0]
        assert payload["type"] == "feature_operation"
        assert payload["operation"] == "unsuppress_feature"
        assert payload["parameters"]["feature_token"] == "feature-token-1"

    asyncio.run(_run())


def test_fusion_executor_sends_parameter_edit_ir_as_structured_feature_payload():
    async def _run():
        manager = _FusionExecutorManager(
            EntityStore(),
            {"tool_use_id": "toolu_param", "success": True, "message": "parameters changed"},
        )
        executor = FusionTargetExecutor(
            manager,
            request={"client_capabilities": {"timeline_feature_parameter_edit": True}},
        )  # type: ignore[arg-type]
        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_parameter_edit",
                type="adjust_feature_parameters",
                params=FeatureParameterEditParams(
                    feature_ref="feature-token-3",
                    parameters={"distance": 12.5},
                    expected_name="Extrude1",
                    expected_timeline_index=3,
                ),
            ),
            tool_use_id="toolu_param",
        )

        assert result.success
        payload = manager.sent_messages[0]
        assert payload["type"] == "feature_operation"
        assert payload["operation"] == "adjust_feature_parameters"
        assert payload["feature_token"] == "feature-token-3"
        assert payload["parameters"] == {
            "distance": 12.5,
            "distance_unit": "mm",
        }
        assert "parameters" not in payload["parameters"]
        assert payload["expected_name"] == "Extrude1"
        assert payload["expected_timeline_index"] == 3

    asyncio.run(_run())


def test_fusion_executor_rejects_parameter_edit_without_client_capability():
    async def _run():
        manager = _FusionExecutorManager(EntityStore(), {})
        executor = FusionTargetExecutor(manager, request={})  # type: ignore[arg-type]
        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_parameter_edit",
                type="adjust_feature_parameters",
                params=FeatureParameterEditParams(
                    feature_ref="feature-token-3",
                    parameters={"distance": 12.5},
                ),
            ),
            tool_use_id="toolu_param",
        )

        assert not result.success
        assert "timeline_feature_parameter_edit" in result.message
        assert manager.sent_messages == []

    asyncio.run(_run())


def test_fusion_executor_rejects_parameter_edit_capability_aliases():
    async def _run():
        manager = _FusionExecutorManager(EntityStore(), {})
        executor = FusionTargetExecutor(
            manager,
            request={"client_capabilities": {"feature_parameter_edit": True, "parameter_edit": True}},
        )  # type: ignore[arg-type]
        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_parameter_edit",
                type="adjust_feature_parameters",
                params=FeatureParameterEditParams(
                    feature_ref="feature-token-3",
                    parameters={"distance": 12.5},
                ),
            ),
            tool_use_id="toolu_param",
        )

        assert not result.success
        assert "timeline_feature_parameter_edit" in result.message
        assert manager.sent_messages == []

    asyncio.run(_run())


def test_fusion_executor_falls_back_to_codegen_delete_without_client_capability():
    async def _run():
        manager = _FusionExecutorManager(
            EntityStore(),
            {"tool_use_id": "toolu_delete", "success": True, "message": "deleted via codegen"},
        )
        executor = FusionTargetExecutor(manager, request={})  # type: ignore[arg-type]
        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_delete",
                type="delete_feature",
                params=DeleteFeatureParams(
                    feature_ref="feature-token-1",
                    expected_name="Shell1",
                    expected_timeline_index=7,
                ),
            ),
            tool_use_id="toolu_delete",
        )

        assert result.success
        payload = manager.sent_messages[0]
        assert payload["type"] == "execute_code"
        assert payload["operation"] == "delete_feature"
        assert "feature-token-1" in payload["code"]

    asyncio.run(_run())


def test_fusion_executor_sends_delete_feature_payload_when_capability_declared():
    async def _run():
        manager = _FusionExecutorManager(
            EntityStore(),
            {"tool_use_id": "toolu_delete", "success": True, "message": "deleted"},
        )
        executor = FusionTargetExecutor(
            manager,
            request={"client_capabilities": {"timeline_feature_delete": True}},
        )  # type: ignore[arg-type]
        result = await executor.execute_operation(
            "s-fusion-ir",
            IROperation(
                id="op_delete",
                type="delete_feature",
                params=DeleteFeatureParams(
                    feature_ref="feature-token-1",
                    expected_name="Shell1",
                    expected_timeline_index=7,
                ),
            ),
            tool_use_id="toolu_delete",
        )

        assert result.success
        payload = manager.sent_messages[0]
        assert payload["type"] == "feature_operation"
        assert payload["operation"] == "delete_feature"
        assert payload["parameters"]["feature_token"] == "feature-token-1"
        assert payload["parameters"]["expected_name"] == "Shell1"
        assert payload["parameters"]["expected_timeline_index"] == 7

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


def test_build123d_translator_rejects_pattern_auto_last_seed():
    document = _build_cube_document()
    pattern_document = IRDocument(
        version=document.version,
        units=document.units,
        operations=[
            *document.operations,
            IROperation(
                id="op_pattern",
                type="pattern_feature",
                params=PatternFeatureParams(pattern_type="rectangular", feature_refs=["auto_last"], count_x=2, spacing_x=10.0),
            ),
        ],
        metadata=document.metadata,
    )

    with pytest.raises(Build123dCapabilityError, match="auto_last"):
        translate_ir_document_to_build123d(pattern_document)


def test_build123d_translator_rejects_pattern_non_replayable_seed():
    document = _build_cube_document()
    pattern_document = IRDocument(
        version=document.version,
        units=document.units,
        operations=[
            *document.operations,
            IROperation(
                id="op_pattern",
                type="pattern_feature",
                params=PatternFeatureParams(
                    pattern_type="rectangular",
                    feature_refs=["feature_0"],
                    count_x=2,
                    spacing_x=10.0,
                ),
            ),
        ],
        metadata=document.metadata,
    )

    with pytest.raises(Build123dCapabilityError, match="not a committed replayable feature"):
        translate_ir_document_to_build123d(pattern_document)


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


def test_build123d_translator_supports_portable_construction_planes():
    document = IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_plane",
                type="create_construction_plane",
                params=CreateConstructionPlaneParams(
                    plane="offset_plane",
                    mode="offset_from_datum",
                    base_datum_plane="XY",
                    offset=15.0,
                ),
            ),
            IROperation(
                id="op_sketch",
                type="create_sketch",
                params=CreateSketchParams(plane="offset_plane", sketch="sketch_offset"),
            ),
        ],
        metadata={"source": "test"},
    )

    program = translate_ir_document_to_build123d(document)

    assert "_construction_planes['offset_plane'] = Plane.XY.offset(15.0)" in program.code
    assert "_sketch_planes['sketch_offset'] = _construction_planes['offset_plane']" in program.code


def test_build123d_translator_rejects_nonportable_construction_plane_modes():
    document = IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_plane",
                type="create_construction_plane",
                params=CreateConstructionPlaneParams(
                    plane="face_plane",
                    mode="face_normal",
                    face="face_0",
                ),
            )
        ],
        metadata={"source": "test"},
    )

    with pytest.raises(ValueError, match="datum and offset_from_datum"):
        translate_ir_document_to_build123d(document)


def test_build123d_translator_supports_revolve_and_loft_code_generation():
    document = IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_revolve_sketch",
                type="create_sketch",
                params=CreateSketchParams(plane="XZ", sketch="revolve_sketch"),
            ),
            IROperation(
                id="op_revolve_rect",
                type="add_rectangle",
                params=AddRectangleParams(sketch="revolve_sketch", center=[20.0, 0.0], width=10.0, height=8.0),
            ),
            IROperation(
                id="op_revolve",
                type="revolve",
                params=RevolveParams(
                    profile="revolve_sketch:profile_0",
                    sketch="revolve_sketch",
                    axis={"type": "construction", "axis": "z"},
                    extent={"mode": "angle", "angle_degrees": 180.0},
                ),
            ),
            IROperation(
                id="op_loft_plane",
                type="create_construction_plane",
                params=CreateConstructionPlaneParams(
                    plane="loft_top",
                    mode="offset_from_datum",
                    base_datum_plane="XY",
                    offset=20.0,
                ),
            ),
            IROperation(
                id="op_loft_sketch_0",
                type="create_sketch",
                params=CreateSketchParams(plane="XY", sketch="loft_bottom"),
            ),
            IROperation(
                id="op_loft_rect_0",
                type="add_rectangle",
                params=AddRectangleParams(sketch="loft_bottom", center=[0.0, 0.0], width=10.0, height=10.0),
            ),
            IROperation(
                id="op_loft_sketch_1",
                type="create_sketch",
                params=CreateSketchParams(plane="loft_top", sketch="loft_top_sketch"),
            ),
            IROperation(
                id="op_loft_rect_1",
                type="add_rectangle",
                params=AddRectangleParams(sketch="loft_top_sketch", center=[0.0, 0.0], width=20.0, height=20.0),
            ),
            IROperation(
                id="op_loft",
                type="loft",
                params=LoftParams(profile_ids=["loft_bottom", "loft_top_sketch"]),
            ),
        ],
        metadata={"source": "test"},
    )

    program = translate_ir_document_to_build123d(document)

    assert "revolve(axis=Axis.Z, revolution_arc=180.0, mode=Mode.ADD)" in program.code
    assert "loft(mode=Mode.ADD)" in program.code
    assert "from build123d import Axis" in program.code


def test_build123d_translator_rejects_nonportable_revolve_axis():
    document = IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_sketch",
                type="create_sketch",
                params=CreateSketchParams(plane="XZ", sketch="sketch_0"),
            ),
            IROperation(
                id="op_rect",
                type="add_rectangle",
                params=AddRectangleParams(sketch="sketch_0", center=[20.0, 0.0], width=10.0, height=8.0),
            ),
            IROperation(
                id="op_revolve",
                type="revolve",
                params=RevolveParams(
                    profile="sketch_0:profile_0",
                    sketch="sketch_0",
                    axis={"type": "edge", "edge_token": "edge_0"},
                    extent={"mode": "full"},
                ),
            ),
        ],
        metadata={"source": "test"},
    )

    with pytest.raises(ValueError, match="construction axes"):
        translate_ir_document_to_build123d(document)


def test_build123d_translator_supports_line_and_arc_sketch_primitives():
    document = IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_sketch",
                type="create_sketch",
                params=CreateSketchParams(plane="XY", sketch="sketch_0"),
            ),
            IROperation(
                id="op_line",
                type="add_line",
                params=AddLineParams(sketch="sketch_0", start=[-10.0, 0.0], end=[10.0, 0.0]),
            ),
            IROperation(
                id="op_arc",
                type="add_arc",
                params=AddArcParams(sketch="sketch_0", center=[0.0, 0.0], start=[10.0, 0.0], end=[-10.0, 0.0]),
            ),
            IROperation(
                id="op_extrude",
                type="extrude",
                params=ExtrudeParams(
                    profile="sketch_0:profile_0",
                    distance=5.0,
                    direction="positive",
                    operation="new",
                    sketch="sketch_0",
                    profile_index=0,
                ),
            ),
        ],
        metadata={"source": "test"},
    )

    program = translate_ir_document_to_build123d(document)

    assert "BuildLine" in program.code
    assert "Line((-10.0, 0.0), (10.0, 0.0))" in program.code
    assert "CenterArc((0.0, 0.0), 10.0, 0.0, 180.0)" in program.code
    assert "make_face()" in program.code


def test_build123d_translator_rejects_open_line_arc_profiles_before_runtime():
    document = IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_sketch",
                type="create_sketch",
                params=CreateSketchParams(plane="XY", sketch="sketch_0"),
            ),
            IROperation(
                id="op_line",
                type="add_line",
                params=AddLineParams(sketch="sketch_0", start=[0.0, 0.0], end=[10.0, 0.0]),
            ),
            IROperation(
                id="op_extrude",
                type="extrude",
                params=ExtrudeParams(
                    profile="sketch_0:profile_0",
                    distance=5.0,
                    direction="positive",
                    operation="new",
                    sketch="sketch_0",
                    profile_index=0,
                ),
            ),
        ],
        metadata={"source": "test"},
    )

    with pytest.raises(ValueError, match="does not form a closed loop"):
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


def _build_line_arc_profile_document() -> IRDocument:
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
                type="add_line",
                params=AddLineParams(sketch="sketch_0", start=[-10.0, 0.0], end=[10.0, 0.0]),
            ),
            IROperation(
                id="op_3",
                type="add_arc",
                params=AddArcParams(sketch="sketch_0", center=[0.0, 0.0], start=[10.0, 0.0], end=[-10.0, 0.0]),
            ),
            IROperation(
                id="op_4",
                type="extrude",
                params=ExtrudeParams(
                    profile="sketch_0:profile_0",
                    distance=5.0,
                    direction="positive",
                    operation="new",
                    sketch="sketch_0",
                    profile_index=0,
                ),
            ),
        ],
        metadata={"source": "test"},
    )


def _build_revolve_document() -> IRDocument:
    return IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_1",
                type="create_sketch",
                params=CreateSketchParams(plane="XZ", sketch="sketch_0"),
            ),
            IROperation(
                id="op_2",
                type="add_rectangle",
                params=AddRectangleParams(sketch="sketch_0", center=[20.0, 0.0], width=10.0, height=20.0),
            ),
            IROperation(
                id="op_3",
                type="revolve",
                params=RevolveParams(
                    profile="sketch_0:profile_0",
                    sketch="sketch_0",
                    axis={"type": "construction", "axis": "z"},
                    extent={"mode": "full"},
                ),
            ),
        ],
        metadata={"source": "test"},
    )


def _build_loft_document() -> IRDocument:
    return IRDocument(
        version="1.0",
        units="mm",
        operations=[
            IROperation(
                id="op_1",
                type="create_sketch",
                params=CreateSketchParams(plane="XY", sketch="bottom"),
            ),
            IROperation(
                id="op_2",
                type="add_rectangle",
                params=AddRectangleParams(sketch="bottom", center=[0.0, 0.0], width=10.0, height=10.0),
            ),
            IROperation(
                id="op_3",
                type="create_construction_plane",
                params=CreateConstructionPlaneParams(
                    plane="top_plane",
                    mode="offset_from_datum",
                    base_datum_plane="XY",
                    offset=20.0,
                ),
            ),
            IROperation(
                id="op_4",
                type="create_sketch",
                params=CreateSketchParams(plane="top_plane", sketch="top"),
            ),
            IROperation(
                id="op_5",
                type="add_rectangle",
                params=AddRectangleParams(sketch="top", center=[0.0, 0.0], width=20.0, height=20.0),
            ),
            IROperation(
                id="op_6",
                type="loft",
                params=LoftParams(profile_ids=["bottom", "top"]),
            ),
        ],
        metadata={"source": "test"},
    )


def _build_cube_with_fillet_document() -> IRDocument:
    document = _build_cube_document()
    return IRDocument(
        version=document.version,
        units=document.units,
        operations=[
            *document.operations,
            IROperation(
                id="op_4",
                type="fillet",
                params=FilletParams(edge_refs=["edge_0"], radius=2.0),
            ),
        ],
        metadata=document.metadata,
    )


def _build_cube_with_chamfer_document() -> IRDocument:
    document = _build_cube_document()
    return IRDocument(
        version=document.version,
        units=document.units,
        operations=[
            *document.operations,
            IROperation(
                id="op_4",
                type="chamfer",
                params=ChamferParams(edge_refs=["edge_0"], distance=2.0),
            ),
        ],
        metadata=document.metadata,
    )


def _build_cube_with_shell_document() -> IRDocument:
    document = _build_cube_document()
    return IRDocument(
        version=document.version,
        units=document.units,
        operations=[
            *document.operations,
            IROperation(
                id="op_4",
                type="shell",
                params=ShellParams(mode="open", face_refs=["face_5"], inside_thickness=2.0),
            ),
        ],
        metadata=document.metadata,
    )


def _build_cube_with_simple_hole_document() -> IRDocument:
    document = _build_cube_document()
    return IRDocument(
        version=document.version,
        units=document.units,
        operations=[
            *document.operations,
            IROperation(
                id="op_4",
                type="create_simple_hole",
                params=SimpleHoleParams(
                    face_ref="face_5",
                    center=[0.0, 0.0, 50.0],
                    diameter=10.0,
                    extent_type="through_all",
                ),
            ),
        ],
        metadata=document.metadata,
    )


def _build_cube_with_simple_hole_pattern_document() -> IRDocument:
    document = _build_cube_document()
    return IRDocument(
        version=document.version,
        units=document.units,
        operations=[
            *document.operations,
            IROperation(
                id="op_4",
                type="create_simple_hole",
                params=SimpleHoleParams(
                    face_ref="face_5",
                    center=[-10.0, 0.0, 50.0],
                    diameter=6.0,
                    extent_type="through_all",
                ),
            ),
            IROperation(
                id="op_5",
                type="pattern_feature",
                params=PatternFeatureParams(
                    pattern_type="rectangular",
                    feature_refs=["op_4:simple_hole"],
                    count_x=3,
                    spacing_x=10.0,
                ),
            ),
        ],
        metadata=document.metadata,
    )


def _build_cube_with_circular_hole_pattern_document() -> IRDocument:
    document = _build_cube_document()
    return IRDocument(
        version=document.version,
        units=document.units,
        operations=[
            *document.operations,
            IROperation(
                id="op_4",
                type="create_simple_hole",
                params=SimpleHoleParams(
                    face_ref="face_5",
                    center=[10.0, 0.0, 50.0],
                    diameter=6.0,
                    extent_type="through_all",
                ),
            ),
            IROperation(
                id="op_5",
                type="pattern_feature",
                params=PatternFeatureParams(
                    pattern_type="circular",
                    feature_refs=["op_4:simple_hole"],
                    rotation_count=4,
                    rotation_angle_degrees=360.0,
                ),
            ),
        ],
        metadata=document.metadata,
    )


def _build_cube_with_hole_pattern_and_chamfer_document() -> IRDocument:
    document = _build_cube_with_simple_hole_pattern_document()
    return IRDocument(
        version=document.version,
        units=document.units,
        operations=[
            *document.operations,
            IROperation(
                id="op_6",
                type="chamfer",
                params=ChamferParams(edge_refs=["edge_0"], distance=1.0),
            ),
        ],
        metadata=document.metadata,
    )


def _build_cube_with_counterbore_document() -> IRDocument:
    document = _build_cube_document()
    return IRDocument(
        version=document.version,
        units=document.units,
        operations=[
            *document.operations,
            IROperation(
                id="op_4",
                type="create_counterbore_hole",
                params=CounterboreHoleParams(
                    face_ref="face_5",
                    center=[0.0, 0.0, 50.0],
                    hole_diameter=6.0,
                    hole_depth=50.0,
                    counterbore_diameter=12.0,
                    counterbore_depth=5.0,
                ),
            ),
        ],
        metadata=document.metadata,
    )


def _build_cube_with_tapped_hole_document() -> IRDocument:
    document = _build_cube_document()
    return IRDocument(
        version=document.version,
        units=document.units,
        operations=[
            *document.operations,
            IROperation(
                id="op_4",
                type="create_tapped_hole",
                params=TappedHoleParams(
                    face_ref="face_5",
                    center=[0.0, 0.0, 50.0],
                    thread_type="metric",
                    thread_size="M6",
                    thread_depth=20.0,
                ),
            ),
        ],
        metadata=document.metadata,
    )


def _build_cylinder_with_external_thread_document() -> IRDocument:
    document = _build_cylinder_document()
    return IRDocument(
        version=document.version,
        units=document.units,
        operations=[
            *document.operations,
            IROperation(
                id="op_4",
                type="create_external_thread",
                params=ExternalThreadParams(
                    face_ref="face_0",
                    thread_type="metric",
                    thread_size="M10",
                    is_full_length=True,
                ),
            ),
        ],
        metadata=document.metadata,
    )


def _assert_build123d_entity_metadata(entities: dict, *, face_types: set[str], edge_types: set[str]) -> None:
    assert entities["bodies"] == 1
    assert len(entities["bodies_metadata"]) == entities["bodies"]
    assert len(entities["faces_metadata"]) == entities["faces"]
    assert len(entities["edges_metadata"]) == entities["edges"]

    for collection_name, prefix in (
        ("bodies_metadata", "body"),
        ("faces_metadata", "face"),
        ("edges_metadata", "edge"),
    ):
        fingerprints = set()
        for index, entity in enumerate(entities[collection_name]):
            assert entity["id"] == f"{prefix}_{index}"
            assert entity["index"] == index
            assert entity["fingerprint"]
            assert entity["bounding_box"]["size"]
            assert entity["center"]
            fingerprints.add(entity["fingerprint"])
        assert fingerprints

    assert entities["bodies_metadata"][0]["topology_type"] == "Solid"
    assert face_types.issubset({face["geometry_type"] for face in entities["faces_metadata"]})
    assert edge_types.issubset({edge["geometry_type"] for edge in entities["edges_metadata"]})


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
    _assert_build123d_entity_metadata(entities, face_types={"PLANE"}, edge_types={"LINE"})

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
    _assert_build123d_entity_metadata(entities, face_types={"CYLINDER", "PLANE"}, edge_types={"CIRCLE", "LINE"})


@pytest.mark.skipif(not HAS_BUILD123D, reason="build123d is required for real-geometry adapter tests")
def test_build123d_executor_extrudes_closed_line_arc_profile():
    executor = Build123dTargetExecutor()
    result = asyncio.run(
        executor.execute_document("s_line_arc", _build_line_arc_profile_document(), request_id="line-arc")
    )

    assert result.success, result.message
    entities = result.data["entities"]
    expected_volume = 0.5 * 3.141592653589793 * 10.0 * 10.0 * 5.0
    assert abs(entities["volume_mm3"] - expected_volume) < 5.0
    assert entities["faces"] == 4


@pytest.mark.skipif(not HAS_BUILD123D, reason="build123d is required for real-geometry adapter tests")
def test_build123d_executor_revolves_profile():
    executor = Build123dTargetExecutor()
    result = asyncio.run(executor.execute_document("s_revolve", _build_revolve_document(), request_id="revolve"))

    assert result.success, result.message
    entities = result.data["entities"]
    expected_volume = 2.0 * 3.141592653589793 * 20.0 * 10.0 * 20.0
    assert abs(entities["volume_mm3"] - expected_volume) < 25.0
    assert entities["faces"] == 4
    _assert_build123d_entity_metadata(entities, face_types={"CYLINDER", "PLANE"}, edge_types={"CIRCLE", "LINE"})


@pytest.mark.skipif(not HAS_BUILD123D, reason="build123d is required for real-geometry adapter tests")
def test_build123d_executor_lofts_between_offset_profiles():
    executor = Build123dTargetExecutor()
    result = asyncio.run(executor.execute_document("s_loft", _build_loft_document(), request_id="loft"))

    assert result.success, result.message
    entities = result.data["entities"]
    assert abs(entities["volume_mm3"] - 4666.666666666667) < 5.0
    assert entities["faces"] == 6
    _assert_build123d_entity_metadata(entities, face_types={"BSPLINE", "PLANE"}, edge_types={"BSPLINE", "LINE"})


@pytest.mark.skipif(not HAS_BUILD123D, reason="build123d is required for real-geometry adapter tests")
@pytest.mark.parametrize(
    "document_factory",
    [
        _build_cube_with_fillet_document,
        _build_cube_with_chamfer_document,
        _build_cube_with_shell_document,
        _build_cube_with_simple_hole_document,
        _build_cube_with_counterbore_document,
    ],
)
def test_build123d_executor_supports_selector_dependent_features(document_factory):
    executor = Build123dTargetExecutor()
    result = asyncio.run(executor.execute_document("s_feature", document_factory(), request_id="feature"))

    assert result.success, result.message
    entities = result.data["entities"]
    assert entities["volume_mm3"] > 0
    assert entities["volume_mm3"] < 125000.0
    assert entities["faces"] > 0
    assert entities["edges"] > 0


@pytest.mark.skipif(not HAS_BUILD123D, reason="build123d is required for real-geometry adapter tests")
def test_build123d_executor_replays_rectangular_simple_hole_pattern():
    executor = Build123dTargetExecutor()
    result = asyncio.run(
        executor.execute_document(
            "s_hole_pattern",
            _build_cube_with_simple_hole_pattern_document(),
            request_id="hole-pattern",
        )
    )

    assert result.success, result.message
    entities = result.data["entities"]
    single_hole_volume = 3.141592653589793 * 3.0 * 3.0 * 50.0
    assert entities["volume_mm3"] < 125000.0 - (single_hole_volume * 2.5)
    assert entities["faces"] > 0
    assert entities["edges"] > 0


@pytest.mark.skipif(not HAS_BUILD123D, reason="build123d is required for real-geometry adapter tests")
def test_build123d_executor_replays_circular_simple_hole_pattern():
    executor = Build123dTargetExecutor()
    result = asyncio.run(
        executor.execute_document(
            "s_circular_hole_pattern",
            _build_cube_with_circular_hole_pattern_document(),
            request_id="circular-hole-pattern",
        )
    )

    assert result.success, result.message
    entities = result.data["entities"]
    single_hole_volume = 3.141592653589793 * 3.0 * 3.0 * 50.0
    assert entities["volume_mm3"] < 125000.0 - (single_hole_volume * 3.5)
    assert entities["faces"] > 0
    assert entities["edges"] > 0


@pytest.mark.skipif(not HAS_BUILD123D, reason="build123d is required for real-geometry adapter tests")
def test_build123d_executor_replays_hole_pattern_with_chamfer_finishing():
    executor = Build123dTargetExecutor()
    result = asyncio.run(
        executor.execute_document(
            "s_hole_pattern_chamfer",
            _build_cube_with_hole_pattern_and_chamfer_document(),
            request_id="hole-pattern-chamfer",
        )
    )

    assert result.success, result.message
    entities = result.data["entities"]
    assert entities["volume_mm3"] > 0
    assert entities["volume_mm3"] < 125000.0
    assert entities["faces"] > 0
    assert entities["edges"] > 0


@pytest.mark.skipif(not HAS_BUILD123D, reason="build123d is required for real-geometry adapter tests")
def test_build123d_executor_adds_tapped_hole_thread_metadata():
    executor = Build123dTargetExecutor()
    result = asyncio.run(
        executor.execute_document("s_tapped", _build_cube_with_tapped_hole_document(), request_id="tapped")
    )

    assert result.success, result.message
    entities = result.data["entities"]
    assert entities["thread_metadata"] == [
        {
            "operation_id": "op_4",
            "kind": "tapped_hole",
            "face_ref": "face_5",
            "thread_type": "metric",
            "thread_size": "M6",
            "thread_depth": 20.0,
            "pilot_hole_depth": 20.0,
        }
    ]


@pytest.mark.skipif(not HAS_BUILD123D, reason="build123d is required for real-geometry adapter tests")
def test_build123d_executor_adds_external_thread_metadata():
    executor = Build123dTargetExecutor()
    result = asyncio.run(
        executor.execute_document(
            "s_external_thread",
            _build_cylinder_with_external_thread_document(),
            request_id="external-thread",
        )
    )

    assert result.success, result.message
    entities = result.data["entities"]
    assert entities["thread_metadata"] == [
        {
            "operation_id": "op_4",
            "kind": "external_thread",
            "face_ref": "face_0",
            "thread_type": "metric",
            "thread_size": "M10",
            "is_full_length": True,
            "thread_length": None,
            "thread_offset": 0.0,
        }
    ]


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
