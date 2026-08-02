import asyncio
from dataclasses import replace
import json

import pytest

from backend.agent_workflow import (
    _capture_checkpoint,
    _deserialize_ir_document_state,
    _operation_checkpoint_conversation_snapshot,
    _serialize_ir_document_state,
    handle_revert_request,
    handle_resume_operation_request,
)
from backend.ir import IRDocumentState, map_tool_call_to_ir, validate_ir_candidate
from backend.ir.types import IRRef
from backend.websocket_manager import ConnectionManager


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


def _assistant_tool_ids(snapshot):
    assistant = next(message for message in snapshot if message.get("role") == "assistant")
    return [
        block.get("id")
        for block in assistant.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def _tool_result_ids(snapshot):
    result_ids = []
    for message in snapshot:
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                result_ids.append(block.get("tool_use_id"))
    return result_ids


def _create_sketch_ir_state(session_id: str, sketch_id: str = "base_square") -> IRDocumentState:
    ir_state = IRDocumentState(metadata={"source": "fusion", "session_id": session_id})
    create_op = map_tool_call_to_ir(
        {
            "name": "create_sketch",
            "input": {"sketch_id": sketch_id, "plane_id": "XY"},
        },
        ir_state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r1", "iteration": 1},
    )
    ir_state.append(create_op)
    return ir_state


def _assert_add_rectangle_valid_after_restore(manager: ConnectionManager, session_id: str, sketch_id: str = "base_square") -> None:
    restored = _deserialize_ir_document_state(manager.get_ir_document_state(session_id) or {})
    add_rectangle = map_tool_call_to_ir(
        {
            "name": "add_rectangle",
            "input": {
                "sketch_id": sketch_id,
                "corner1_u": -2.5,
                "corner1_v": -2.5,
                "corner2_u": 2.5,
                "corner2_v": 2.5,
            },
        },
        restored,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r2", "iteration": 1},
        dependency_operations=restored.operations,
    )

    assert [operation.type for operation in restored.operations] == ["create_sketch"]
    assert add_rectangle.dependencies == ["op_1"]
    assert validate_ir_candidate(add_rectangle, restored.operations) == []


def test_operation_checkpoint_snapshot_prunes_future_batched_tool_use() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "create a cube"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will create the base sketch and rectangle."},
                {"type": "tool_use", "id": "tool_create", "name": "create_sketch", "input": {}},
                {"type": "text", "text": "Then I will add the square."},
                {"type": "tool_use", "id": "tool_add", "name": "add_rectangle", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_create",
                    "content": [{"type": "text", "text": "created sketch"}],
                }
            ],
        },
    ]

    snapshot = _operation_checkpoint_conversation_snapshot(messages, "tool_create")

    assert _assistant_tool_ids(snapshot) == ["tool_create"]
    assert _tool_result_ids(snapshot) == ["tool_create"]
    assert "tool_add" not in json.dumps(snapshot)
    assert snapshot[1]["content"][-1]["id"] == "tool_create"


def test_operation_checkpoint_snapshot_keeps_completed_prior_sibling_tools() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "create a cube"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will create the base sketch and rectangle."},
                {"type": "tool_use", "id": "tool_create", "name": "create_sketch", "input": {}},
                {"type": "tool_use", "id": "tool_add", "name": "add_rectangle", "input": {}},
                {"type": "tool_use", "id": "tool_extrude", "name": "extrude_profile", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_create",
                    "content": [{"type": "text", "text": "created sketch"}],
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_add",
                    "content": [{"type": "text", "text": "added rectangle"}],
                }
            ],
        },
    ]

    snapshot = _operation_checkpoint_conversation_snapshot(messages, "tool_add")

    assert _assistant_tool_ids(snapshot) == ["tool_create", "tool_add"]
    assert _tool_result_ids(snapshot) == ["tool_create", "tool_add"]
    assert "tool_extrude" not in json.dumps(snapshot)


def test_operation_checkpoint_restore_sets_conversation_and_cached_context() -> None:
    manager = ConnectionManager()
    session_id = "session-op-restore"
    manager.conversation_history[session_id] = [{"role": "user", "content": "old"}]

    checkpoint = {
        "checkpoint_id": "opchk_1",
        "conversation_index": 2,
        "conversation_snapshot": [
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": [{"type": "text", "text": "tool"}]},
        ],
        "latest_entity_context": {"bodies": [{"name": "Body1"}], "faces": [], "edges": []},
        "feature_snapshot": {"success": True, "features": [], "timeline_count": 3, "marker_position": 3},
        "reasoning_summary": "Created base sketch and extruded the first body.",
    }

    manager.save_operation_checkpoint(session_id, checkpoint)
    restored = manager.restore_operation_checkpoint_state(session_id, checkpoint)

    assert restored == 2
    assert manager.get_conversation_history(session_id)[0]["content"] == "request"
    assert manager.get_latest_entity_context(session_id)["bodies"][0]["name"] == "Body1"
    assert manager.get_feature_snapshot(session_id)["timeline_count"] == 3
    assert "Created base sketch" in manager.get_reasoning_context(session_id).get_injection_text()


def test_operation_checkpoint_restore_sets_ir_state_for_followup_tools() -> None:
    manager = ConnectionManager()
    session_id = "session-op-ir-restore"

    ir_state = _create_sketch_ir_state(session_id)
    ir_state.entities["sketch"] = [
        IRRef(kind="sketch", id="base_square", source_operation_id="op_1", target_handles={"fusion": "Sketch1"})
    ]
    checkpoint = {
        "checkpoint_id": "opchk_ir",
        "conversation_index": 0,
        "conversation_snapshot": [],
        "ir_state": _serialize_ir_document_state(ir_state),
    }

    manager.restore_operation_checkpoint_state(session_id, checkpoint)
    restored_state = _deserialize_ir_document_state(manager.get_ir_document_state(session_id) or {})
    assert restored_state.entities["sketch"][0].target_handles["fusion"] == "Sketch1"
    _assert_add_rectangle_valid_after_restore(manager, session_id)


def test_message_checkpoint_capture_stores_runtime_state() -> None:
    manager = ConnectionManager()
    session_id = "session-message-capture"
    ir_state = _create_sketch_ir_state(session_id, "sketch_1")
    ir_state.entities["sketch"] = [
        IRRef(
            kind="sketch",
            id="sketch_1",
            source_operation_id="op_1",
            alias="Base sketch",
            target_handles={"fusion": "Sketch1"},
            fingerprint={"plane": "XY"},
        )
    ]
    manager.save_ir_document_state(session_id, _serialize_ir_document_state(ir_state))
    manager.set_feature_snapshot(session_id, {"success": True, "timeline_count": 1, "marker_position": 1})
    manager.set_latest_entity_context(session_id, {"bodies": [], "faces": [], "edges": [], "sketches": ["Sketch1"]})

    message_id = _capture_checkpoint(
        session_id,
        {"user_request": "continue", "timeline_state": {"marker_position": 1, "count": 1}},
        manager,
        conversation_index=3,
    )
    checkpoint = manager.get_checkpoint_by_message_id(session_id, message_id)

    assert checkpoint is not None
    assert checkpoint["ir_state"]["operations"][0]["params"]["sketch"] == "sketch_1"
    assert checkpoint["ir_state"]["entities"]["sketch"][0]["id"] == "sketch_1"
    assert checkpoint["ir_state"]["entities"]["sketch"][0]["target_handles"]["fusion"] == "Sketch1"
    assert checkpoint["feature_snapshot"]["timeline_count"] == 1
    assert checkpoint["latest_entity_context"]["sketches"] == ["Sketch1"]


@pytest.mark.asyncio
async def test_handle_revert_request_restores_ir_state_from_message_checkpoint() -> None:
    manager = ConnectionManager()
    session_id = "session-message-revert-ir"
    websocket = FakeWebSocket()
    manager.active_connections[session_id] = websocket
    manager.pending_results[session_id] = asyncio.Queue()
    manager.conversation_history[session_id] = [
        {"role": "user", "content": "create cube"},
        {"role": "assistant", "content": "create sketch"},
        {"role": "user", "content": "created sketch"},
        {"role": "user", "content": "future request"},
    ]

    checkpoint = {
        "message_id": "msg_revert",
        "conversation_index": 3,
        "marker_position": 1,
        "timeline_count": 1,
        "message_text": "future request",
        "ir_state": _serialize_ir_document_state(_create_sketch_ir_state(session_id)),
        "feature_snapshot": {"success": True, "timeline_count": 1, "marker_position": 1},
        "latest_entity_context": {"bodies": [], "faces": [], "edges": []},
    }
    manager.save_checkpoint(session_id, checkpoint)
    manager.save_ir_document_state(session_id, {"operations": [{"type": "future_bad_state"}]})

    await manager.pending_results[session_id].put({"success": True, "message_id": "msg_revert"})

    await handle_revert_request(session_id, {"message_id": "msg_revert"}, manager)

    assert websocket.sent[0]["type"] == "revert_timeline"
    assert any(message["type"] == "revert_applied" for message in websocket.sent)
    _assert_add_rectangle_valid_after_restore(manager, session_id)
    assert manager.get_conversation_history(session_id) == manager.conversation_history[session_id]
    assert len(manager.get_conversation_history(session_id)) == 3


@pytest.mark.asyncio
async def test_handle_revert_request_chat_only_when_timeline_unavailable() -> None:
    manager = ConnectionManager()
    session_id = "session-message-revert-chat-only"
    websocket = FakeWebSocket()
    manager.active_connections[session_id] = websocket
    manager.pending_results[session_id] = asyncio.Queue()
    manager.conversation_history[session_id] = [
        {"role": "user", "content": "create direct body"},
        {"role": "assistant", "content": "created direct body"},
        {"role": "user", "content": "future request"},
    ]

    checkpoint = {
        "message_id": "msg_chat_only",
        "conversation_index": 2,
        "marker_position": 0,
        "timeline_count": 0,
        "message_text": "future request",
        "feature_snapshot": {"success": True, "timeline_count": 0, "marker_position": 0},
        "latest_entity_context": {"bodies": [], "faces": [], "edges": []},
    }
    manager.save_checkpoint(session_id, checkpoint)
    manager.set_feature_snapshot(session_id, {"success": True, "timeline_count": 0, "current": "direct-body"})
    manager.set_latest_entity_context(session_id, {"bodies": [{"name": "current body"}], "faces": [], "edges": []})

    await manager.pending_results[session_id].put({
        "success": True,
        "message_id": "msg_chat_only",
        "geometry_reverted": "false",
        "message": "Chat history rollback only; this design does not expose a Fusion timeline.",
    })

    await handle_revert_request(session_id, {"message_id": "msg_chat_only"}, manager)

    assert websocket.sent[0]["type"] == "revert_timeline"
    revert_applied = next(message for message in websocket.sent if message["type"] == "revert_applied")
    assert revert_applied["timeline_reverted"] is False
    log_messages = [message for message in websocket.sent if message.get("type") == "log"]
    assert log_messages[-1]["level"] == "info"
    assert log_messages[-1]["message"] == 'Chat history reverted to: "future request"; model geometry was not changed.'
    assert not any(message.get("type") == "error" for message in websocket.sent)
    assert len(manager.get_conversation_history(session_id)) == 2
    assert manager.get_feature_snapshot(session_id) == {"success": True, "timeline_count": 0, "current": "direct-body"}
    assert manager.get_latest_entity_context(session_id) == {"bodies": [{"name": "current body"}], "faces": [], "edges": []}


@pytest.mark.asyncio
async def test_handle_revert_request_uses_operation_checkpoint_ir_fallback() -> None:
    manager = ConnectionManager()
    session_id = "session-message-revert-legacy"
    websocket = FakeWebSocket()
    manager.active_connections[session_id] = websocket
    manager.pending_results[session_id] = asyncio.Queue()
    manager.conversation_history[session_id] = [
        {"role": "user", "content": "create cube"},
        {"role": "assistant", "content": "create sketch"},
        {"role": "user", "content": "created sketch"},
        {"role": "user", "content": "future request"},
    ]

    manager.save_checkpoint(
        session_id,
        {
            "message_id": "legacy_msg_revert",
            "conversation_index": 3,
            "marker_position": 1,
            "timeline_count": 1,
            "message_text": "legacy future request",
        },
    )
    manager.save_operation_checkpoint(
        session_id,
        {
            "checkpoint_id": "opchk_create",
            "conversation_index": 3,
            "ir_state": _serialize_ir_document_state(_create_sketch_ir_state(session_id)),
        },
    )
    manager.save_operation_checkpoint(
        session_id,
        {
            "checkpoint_id": "opchk_future",
            "conversation_index": 8,
            "ir_state": {"operations": [{"type": "future_bad_state"}]},
        },
    )

    await manager.pending_results[session_id].put({"success": True, "message_id": "legacy_msg_revert"})

    await handle_revert_request(session_id, {"message_id": "legacy_msg_revert"}, manager)

    _assert_add_rectangle_valid_after_restore(manager, session_id)
    assert [cp["checkpoint_id"] for cp in manager.get_operation_checkpoints(session_id)] == ["opchk_create"]


def test_prune_operation_checkpoints_after_conversation_index() -> None:
    manager = ConnectionManager()
    session_id = "session-op-prune"
    manager.save_operation_checkpoint(session_id, {"checkpoint_id": "keep", "conversation_index": 2})
    manager.save_operation_checkpoint(session_id, {"checkpoint_id": "drop", "conversation_index": 5})

    removed = manager.prune_operation_checkpoints_after(session_id, 2)

    assert removed == 1
    assert [cp["checkpoint_id"] for cp in manager.get_operation_checkpoints(session_id)] == ["keep"]


def test_ir_document_state_round_trips_timeline_feature_suppression() -> None:
    session_id = "session-ir-suppression"
    state = IRDocumentState(metadata={"source": "fusion", "session_id": session_id})
    suppression = map_tool_call_to_ir(
        {
            "name": "suppress_feature",
            "input": {
                "feature_token": "feature-token-1",
                "expected_name": "Shell1",
                "expected_timeline_index": 7,
            },
        },
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r1", "iteration": 1},
    )
    state.append(suppression)

    restored = _deserialize_ir_document_state(_serialize_ir_document_state(state))

    assert len(restored.operations) == 1
    restored_op = restored.operations[0]
    assert restored_op.type == "set_feature_suppression"
    assert restored_op.params.feature_ref == "feature-token-1"
    assert restored_op.params.suppress is True
    assert restored_op.params.expected_name == "Shell1"


def test_ir_document_state_round_trips_feature_parameter_edit() -> None:
    session_id = "session-ir-parameter-edit"
    state = IRDocumentState(metadata={"source": "fusion", "session_id": session_id})
    parameter_edit = map_tool_call_to_ir(
        {
            "name": "adjust_feature_parameters",
            "input": {
                "feature_token": "feature-token-3",
                "parameters": {"distance": 1.25, "distance_unit": "cm"},
                "expected_name": "Extrude1",
                "expected_timeline_index": 3,
            },
        },
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r1", "iteration": 1},
    )
    state.append(parameter_edit)

    restored = _deserialize_ir_document_state(_serialize_ir_document_state(state))

    assert len(restored.operations) == 1
    restored_op = restored.operations[0]
    assert restored_op.type == "adjust_feature_parameters"
    assert restored_op.params.feature_ref == "feature-token-3"
    assert restored_op.params.parameters["distance"] == 12.5
    assert restored_op.params.expected_name == "Extrude1"
    assert restored_op.params.expected_timeline_index == 3
    assert restored.metadata["session_id"] == session_id


def test_ir_document_state_round_trips_shell_operation() -> None:
    session_id = "session-ir-shell"
    state = IRDocumentState(metadata={"source": "fusion", "session_id": session_id})
    shell = map_tool_call_to_ir(
        {
            "name": "create_shell",
            "input": {
                "mode": "open",
                "face_refs": ["face-token-1", "face-token-2"],
                "inside_thickness": 1.5,
                "outside_thickness": 0.25,
                "is_tangent_chain": False,
                "shell_type": "rounded",
                "feature_name": "Lightweight shell",
            },
        },
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r1", "iteration": 1},
    )
    state.append(shell)

    restored = _deserialize_ir_document_state(_serialize_ir_document_state(state))

    assert len(restored.operations) == 1
    restored_op = restored.operations[0]
    assert restored_op.type == "shell"
    assert restored_op.params.mode == "open"
    assert restored_op.params.face_refs == ["face-token-1", "face-token-2"]
    assert restored_op.params.body_refs == []
    assert restored_op.params.inside_thickness == 1.5
    assert restored_op.params.outside_thickness == 0.25
    assert restored_op.params.is_tangent_chain is False
    assert restored_op.params.shell_type == "rounded"
    assert restored_op.params.feature_name == "Lightweight shell"
    assert restored.metadata["session_id"] == session_id


def test_ir_document_state_round_trips_profile_inspection_results_for_restore_validation() -> None:
    session_id = "session-ir-profile-results"
    state = IRDocumentState(metadata={"source": "fusion", "session_id": session_id})

    create_sketch = map_tool_call_to_ir(
        {"name": "create_sketch", "input": {"plane_id": "XY", "sketch_id": "profile_sketch"}},
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r1", "iteration": 1},
    )
    state.append(create_sketch)

    add_circle = map_tool_call_to_ir(
        {
            "name": "add_circle",
            "input": {
                "sketch_id": "profile_sketch",
                "center_u": 0.0,
                "center_v": 0.0,
                "radius": 1.0,
            },
        },
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r1", "iteration": 2},
    )
    state.append(add_circle)

    profile_inspection = map_tool_call_to_ir(
        {"name": "list_sketch_profiles", "input": {"sketch_id": "profile_sketch"}},
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r1", "iteration": 3},
    )
    profile_inspection = replace(
        profile_inspection,
        target_results=[
            {
                "target": "fusion",
                "success": True,
                "raw_result": {
                    "profile_count": 1,
                    "profiles": [{"index": 0, "label": "profile_0"}],
                },
            }
        ],
        validation={"restored": True},
    )
    state.append(profile_inspection)

    restored = _deserialize_ir_document_state(_serialize_ir_document_state(state))

    restored_inspection = restored.operations[-1]
    assert restored_inspection.type == "list_sketch_profiles"
    assert restored_inspection.target_results[0]["raw_result"]["profile_count"] == 1
    assert restored_inspection.validation == {"restored": True}
    assert restored_inspection.effects.creates == {"profiles": ["profile_sketch:profiles"]}

    extrude = map_tool_call_to_ir(
        {
            "name": "extrude_profile",
            "input": {
                "sketch_id": "profile_sketch",
                "profile_index": 0,
                "distance": 1.0,
                "operation": "NewBody",
            },
        },
        restored,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r2", "iteration": 1},
        dependency_operations=restored.operations,
    )
    assert validate_ir_candidate(extrude, restored.operations) == []


def test_ir_document_state_round_trips_entity_registry_and_to_document() -> None:
    session_id = "session-ir-entity-registry"
    state = IRDocumentState(metadata={"source": "fusion", "session_id": session_id})
    state.entities["body"] = [
        IRRef(
            kind="body",
            id="body_1",
            source_operation_id="op_2",
            alias="Primary body",
            target_handles={"fusion": "body-token-1"},
            fingerprint={"volume": 125.0, "centroid": [0.0, 0.0, 2.5]},
            validity="valid",
            generation=2,
        )
    ]
    state.entities["face"] = [
        IRRef(
            kind="face",
            id="face_1",
            target_handles={"fusion": "face-token-1"},
            validity="unknown",
        )
    ]

    serialized = _serialize_ir_document_state(state)
    restored = _deserialize_ir_document_state(serialized)
    document = restored.to_document()

    assert serialized["entities"]["body"][0]["id"] == "body_1"
    assert restored.entities["body"][0].source_operation_id == "op_2"
    assert restored.entities["body"][0].target_handles["fusion"] == "body-token-1"
    assert restored.entities["body"][0].fingerprint["volume"] == 125.0
    assert restored.entities["body"][0].generation == 2
    assert restored.entities["face"][0].validity == "unknown"
    assert document.entities == restored.entities


def test_ir_document_state_round_trips_delete_hole_and_list_features_operations() -> None:
    session_id = "session-ir-delete-hole-list"
    state = IRDocumentState(metadata={"source": "fusion", "session_id": session_id})

    hole = map_tool_call_to_ir(
        {
            "name": "create_simple_hole",
            "input": {
                "face_ref": "face-token-1",
                "center_x": 1.0,
                "center_y": 2.0,
                "center_z": 3.0,
                "diameter": 4.0,
                "extent_type": "through_all",
                "feature_name": "Mount hole",
            },
        },
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r1", "iteration": 1},
    )
    state.append(hole)

    delete_feature = map_tool_call_to_ir(
        {
            "name": "delete_feature",
            "input": {
                "feature_token": "feature-token-9",
                "expected_name": "Hole1",
                "expected_timeline_index": 12,
            },
        },
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r1", "iteration": 2},
    )
    state.append(delete_feature)

    list_features = map_tool_call_to_ir(
        {"name": "list_features", "input": {"description": "Refresh features"}},
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r1", "iteration": 3},
    )
    state.append(list_features)

    restored = _deserialize_ir_document_state(_serialize_ir_document_state(state))

    restored_hole, restored_delete, restored_list = restored.operations
    assert restored_hole.type == "create_simple_hole"
    assert restored_hole.params.feature_name == "Mount hole"
    assert restored_delete.type == "delete_feature"
    assert restored_delete.params.expected_name == "Hole1"
    assert restored_delete.params.expected_timeline_index == 12
    assert restored_list.type == "list_features"
    assert restored_list.params.description == "Refresh features"


def test_ir_document_state_round_trips_selection_and_timeline_revision_metadata() -> None:
    session_id = "session-ir-selection-timeline"
    state = IRDocumentState(metadata={"source": "fusion", "session_id": session_id})

    select_faces = map_tool_call_to_ir(
        {
            "name": "select_faces",
            "input": {"face_refs": ["face-token-1", "face-token-2"], "clear_existing": False},
        },
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r1", "iteration": 1},
    )
    state.append(select_faces)

    clear_faces = map_tool_call_to_ir(
        {"name": "clear_face_selection", "input": {}},
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r1", "iteration": 2},
    )
    state.append(clear_faces)

    jump = map_tool_call_to_ir(
        {
            "name": "jump_to_timeline_position",
            "input": {"target_index": 6, "reason": "truncate regenerated branch"},
        },
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "r1", "iteration": 3},
    )
    state.append(jump)

    restored = _deserialize_ir_document_state(_serialize_ir_document_state(state))

    restored_select, restored_clear, restored_jump = restored.operations
    assert restored_select.type == "select_entities"
    assert restored_select.params.kind == "face"
    assert restored_select.params.refs == ["face-token-1", "face-token-2"]
    assert restored_select.params.clear_existing is False
    assert restored_select.selectors == [{"kind": "face", "refs": ["face-token-1", "face-token-2"]}]
    assert restored_select.requires == ["selection"]

    assert restored_clear.type == "clear_selection"
    assert restored_clear.params.kind == "face"
    assert restored_clear.effects.invalidates == {"selection": ["face"]}

    assert restored_jump.type == "jump_to_timeline_position"
    assert restored_jump.params.target_index == 6
    assert restored_jump.params.reason == "truncate regenerated branch"
    assert restored_jump.requires == ["parametric_timeline", "document_revision"]
    assert restored_jump.effects.invalidates == {
        "operations": ["after_marker"],
        "features": ["after_marker"],
        "faces": ["*"],
        "edges": ["*"],
    }
    assert restored.metadata["session_id"] == session_id


def test_ir_document_state_rejects_malformed_feature_suppression_bool() -> None:
    with pytest.raises(ValueError, match="Invalid boolean"):
        _deserialize_ir_document_state(
            {
                "operations": [
                    {
                        "id": "op_1",
                        "type": "set_feature_suppression",
                        "params": {"feature_ref": "feature-token-1", "suppress": "not-a-bool"},
                    }
                ]
            }
        )


@pytest.mark.asyncio
async def test_handle_resume_operation_request_rewinds_and_restores_backend_state() -> None:
    manager = ConnectionManager()
    session_id = "session-op-resume"
    websocket = FakeWebSocket()
    manager.active_connections[session_id] = websocket
    manager.pending_results[session_id] = asyncio.Queue()

    checkpoint = {
        "checkpoint_id": "opchk_resume",
        "tool_name": "extrude_profile",
        "display_label": "Extrude profile",
        "marker_position": 3,
        "timeline_count": 5,
        "conversation_index": 2,
        "conversation_snapshot": [
            {"role": "user", "content": "make a cube"},
            {"role": "assistant", "content": [{"type": "text", "text": "extruded"}]},
        ],
        "latest_entity_context": {},
        "feature_snapshot": {"success": True, "timeline_count": 5, "marker_position": 3},
        "reasoning_summary": "Created and extruded the cube base.",
    }
    manager.save_operation_checkpoint(session_id, checkpoint)
    manager.save_operation_checkpoint(session_id, {"checkpoint_id": "future", "conversation_index": 9})
    manager.save_checkpoint(
        session_id,
        {
            "message_id": "future_message",
            "conversation_index": 9,
            "marker_position": 9,
            "timeline_count": 9,
        },
    )

    await manager.pending_results[session_id].put({"success": True, "message_id": "opchk_resume"})

    await handle_resume_operation_request(
        session_id,
        {"checkpoint_id": "opchk_resume"},
        manager,
    )

    assert websocket.sent[0]["type"] == "revert_timeline"
    assert websocket.sent[0]["message_id"] == "opchk_resume"
    assert websocket.sent[0]["marker_position"] == 3
    assert any(message["type"] == "operation_resume_applied" for message in websocket.sent)
    assert manager.get_conversation_history(session_id)[0]["content"] == "make a cube"
    assert [cp["checkpoint_id"] for cp in manager.get_operation_checkpoints(session_id)] == ["opchk_resume"]


@pytest.mark.asyncio
async def test_handle_resume_operation_request_rejects_chat_only_revert() -> None:
    manager = ConnectionManager()
    session_id = "session-op-resume-chat-only"
    websocket = FakeWebSocket()
    manager.active_connections[session_id] = websocket
    manager.pending_results[session_id] = asyncio.Queue()
    manager.conversation_history[session_id] = [
        {"role": "user", "content": "future request"},
        {"role": "assistant", "content": "future response"},
    ]

    checkpoint = {
        "checkpoint_id": "opchk_chat_only",
        "tool_name": "extrude_profile",
        "display_label": "Extrude profile",
        "marker_position": 0,
        "timeline_count": 0,
        "conversation_index": 1,
        "conversation_snapshot": [{"role": "user", "content": "old request"}],
        "latest_entity_context": {"bodies": [{"name": "old body"}]},
        "feature_snapshot": {"success": True, "timeline_count": 0},
    }
    manager.save_operation_checkpoint(session_id, checkpoint)
    manager.set_latest_entity_context(session_id, {"bodies": [{"name": "current body"}]})
    manager.set_feature_snapshot(session_id, {"success": True, "timeline_count": 0, "current": True})

    await manager.pending_results[session_id].put({
        "success": True,
        "message_id": "opchk_chat_only",
        "geometry_reverted": False,
        "message": "Chat history rollback only; this design does not expose a Fusion timeline.",
    })

    await handle_resume_operation_request(
        session_id,
        {"checkpoint_id": "opchk_chat_only"},
        manager,
    )

    assert websocket.sent[0]["type"] == "revert_timeline"
    assert not any(message.get("type") == "operation_resume_applied" for message in websocket.sent)
    assert any(
        message.get("type") == "error" and message.get("message") == "Resume unavailable"
        for message in websocket.sent
    )
    assert manager.get_conversation_history(session_id)[0]["content"] == "future request"
    assert manager.get_latest_entity_context(session_id) == {"bodies": [{"name": "current body"}]}
    assert manager.get_feature_snapshot(session_id) == {"success": True, "timeline_count": 0, "current": True}


@pytest.mark.asyncio
async def test_handle_resume_operation_request_treats_string_false_as_chat_only() -> None:
    manager = ConnectionManager()
    session_id = "session-op-resume-string-false"
    websocket = FakeWebSocket()
    manager.active_connections[session_id] = websocket
    manager.pending_results[session_id] = asyncio.Queue()
    manager.conversation_history[session_id] = [{"role": "user", "content": "future request"}]

    checkpoint = {
        "checkpoint_id": "opchk_string_false",
        "tool_name": "extrude_profile",
        "marker_position": 0,
        "timeline_count": 0,
        "conversation_index": 1,
        "conversation_snapshot": [{"role": "user", "content": "old request"}],
    }
    manager.save_operation_checkpoint(session_id, checkpoint)

    await manager.pending_results[session_id].put({
        "success": True,
        "message_id": "opchk_string_false",
        "geometry_reverted": "false",
    })

    await handle_resume_operation_request(
        session_id,
        {"checkpoint_id": "opchk_string_false"},
        manager,
    )

    assert not any(message.get("type") == "operation_resume_applied" for message in websocket.sent)
    assert any(
        message.get("type") == "error" and message.get("message") == "Resume unavailable"
        for message in websocket.sent
    )
