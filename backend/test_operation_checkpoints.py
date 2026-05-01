import asyncio
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
    checkpoint = {
        "checkpoint_id": "opchk_ir",
        "conversation_index": 0,
        "conversation_snapshot": [],
        "ir_state": _serialize_ir_document_state(ir_state),
    }

    manager.restore_operation_checkpoint_state(session_id, checkpoint)
    _assert_add_rectangle_valid_after_restore(manager, session_id)


def test_message_checkpoint_capture_stores_runtime_state() -> None:
    manager = ConnectionManager()
    session_id = "session-message-capture"
    ir_state = _create_sketch_ir_state(session_id, "sketch_1")
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
