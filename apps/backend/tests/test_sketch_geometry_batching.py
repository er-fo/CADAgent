"""Regression tests for narrow sketch geometry batching."""

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import pytest

from backend import agent_workflow
from backend.entity_store import EntityStore
from backend.ir import IRDocumentState, map_tool_call_to_ir
from backend.prompt_structure import get_cluster_tools

@dataclass
class _FakeReasoningContext:
    entries: List[Any] = field(default_factory=list)

    def add_entry(self, entry: Any) -> None:
        self.entries.append(entry)

    def get_injection_text(self) -> str:
        return ""


@dataclass
class _FaceEntry:
    ref_id: str
    token: str


class _FaceEntityStore:
    def __init__(self) -> None:
        self._entries: Dict[str, _FaceEntry] = {}

    def add_face(self, ref_id: str, token: str) -> None:
        self._entries[ref_id] = _FaceEntry(ref_id=ref_id, token=token)

    def resolve_token(self, ref_or_token: str, expected_kind: Optional[str] = None):
        if expected_kind not in (None, "face"):
            return None, f"Unexpected kind '{expected_kind}'"
        entry = self._entries.get(ref_or_token)
        if entry:
            return entry.token, None
        for candidate in self._entries.values():
            if candidate.token == ref_or_token:
                return candidate.token, None
        return None, f"Could not resolve reference '{ref_or_token}'"

    def get_refs_by_kind(self, kind: str) -> List[str]:
        return list(self._entries) if kind == "face" else []

    def get_entry(self, ref_id: str) -> Optional[_FaceEntry]:
        return self._entries.get(ref_id)


class _FakeBatchManager:
    def __init__(self, fusion_results: Optional[List[Mapping[str, Any]]] = None) -> None:
        self.sent_messages: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.ir_state: Dict[str, Any] = {}
        self.operation_checkpoints: List[Dict[str, Any]] = []
        self.feature_snapshot: Dict[str, Any] = {
            "success": True,
            "timeline_count": 1,
            "marker_position": 1,
            "features": [],
        }
        self.latest_entity_context: Dict[str, Any] = {}
        self.reasoning_context = _FakeReasoningContext()
        self.fusion_results: List[Mapping[str, Any]] = list(fusion_results or [])
        self.entity_store: Optional[EntityStore] = None

    def get_user_token(self, session_id: str) -> str:
        return "token"

    def get_llm_api_keys(self, session_id: str) -> Dict[str, str]:
        return {}

    def get_reasoning_context(self, session_id: str) -> _FakeReasoningContext:
        return self.reasoning_context

    def get_active_build_plan(self, session_id: str) -> None:
        return None

    def get_entity_store(self, session_id: str) -> EntityStore:
        if self.entity_store is None:
            self.entity_store = EntityStore()
        return self.entity_store

    async def send_message(self, session_id: str, payload: Dict[str, Any]) -> None:
        self.sent_messages.append(payload)

    async def wait_for_fusion_result(self, session_id: str, timeout: Optional[float] = None) -> Mapping[str, Any]:
        if not self.fusion_results:
            raise AssertionError("No queued Fusion result")
        return self.fusion_results.pop(0)

    async def store_fusion_result(self, session_id: str, result: Dict[str, Any]) -> None:
        self.fusion_results.insert(0, result)

    def set_conversation_history(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self.history = list(messages)

    def save_ir_document_state(self, session_id: str, ir_state: Dict[str, Any]) -> None:
        self.ir_state = copy.deepcopy(ir_state)

    def get_ir_document_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        return copy.deepcopy(self.ir_state) if self.ir_state else None

    def save_operation_checkpoint(self, session_id: str, checkpoint_data: Dict[str, Any]) -> None:
        self.operation_checkpoints.append(copy.deepcopy(checkpoint_data))

    def get_operation_checkpoints(self, session_id: str) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.operation_checkpoints)

    def get_feature_snapshot(self, session_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self.feature_snapshot)

    def set_feature_snapshot(self, session_id: str, snapshot: Dict[str, Any]) -> None:
        self.feature_snapshot = copy.deepcopy(snapshot)

    def get_latest_entity_context(self, session_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self.latest_entity_context)

    def set_latest_entity_context(self, session_id: str, entity_context: Dict[str, Any]) -> None:
        self.latest_entity_context = copy.deepcopy(entity_context)


def _seed_committed_sketch(manager: _FakeBatchManager, session_id: str = "s-batch", sketch_id: str = "base_sketch") -> None:
    existing_state = manager.get_ir_document_state(session_id)
    state = (
        agent_workflow._deserialize_ir_document_state(existing_state)
        if existing_state
        else IRDocumentState(metadata={"source": "fusion", "session_id": session_id})
    )
    create_sketch = map_tool_call_to_ir(
        {
            "name": "create_sketch",
            "input": {"plane_id": "XY", "sketch_id": sketch_id},
        },
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "seed", "iteration": 1},
    )
    state.append(create_sketch)
    manager.save_ir_document_state(session_id, agent_workflow._serialize_ir_document_state(state))


def _seed_sketch_alias(
    manager: _FakeBatchManager,
    *,
    session_id: str = "s-batch",
    sketch_id: str = "base_sketch",
    sketch_name: str = "Base Sketch",
) -> None:
    sketch_store = agent_workflow._get_sketch_entity_store(session_id, manager)  # type: ignore[arg-type]
    sketch_store.register_sketch_metadata(sketch_id, {"sketch_name": sketch_name})


def _seed_face_entity(manager: _FakeBatchManager) -> None:
    manager.entity_store = _FaceEntityStore()
    manager.entity_store.add_face("face_0", "face_token_0")


def _seed_committed_profiled_sketch(
    manager: _FakeBatchManager,
    session_id: str = "s-batch",
    sketch_id: str = "profile_sketch",
) -> None:
    _seed_committed_sketch(manager, session_id=session_id, sketch_id=sketch_id)
    state = agent_workflow._deserialize_ir_document_state(manager.get_ir_document_state(session_id))
    add_rectangle = map_tool_call_to_ir(
        {
            "name": "add_rectangle",
            "input": {
                "sketch_id": sketch_id,
                "corner1_u": -1,
                "corner1_v": -1,
                "corner2_u": 1,
                "corner2_v": 1,
            },
        },
        state,
        metadata={"source": "fusion", "session_id": session_id, "request_id": "seed", "iteration": 1},
    )
    state.append(add_rectangle)
    manager.save_ir_document_state(session_id, agent_workflow._serialize_ir_document_state(state))


def _end_turn_response() -> Dict[str, Any]:
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Done."}]}


def _extract_tool_result_texts(messages: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                for content in block.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "text":
                        texts.append(str(content.get("text") or ""))
    return texts


def test_sketch_geometry_batch_executes_one_payload_and_commits_primitives(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeBatchManager(
        fusion_results=[
            {
                "tool_use_id": "toolu_batch",
                "success": True,
                "batch_size": 2,
                "success_count": 2,
                "failure_count": 0,
                "sketch_id": "base_sketch",
                "operations": [
                    {
                        "index": 0,
                        "operation": "add_line",
                        "success": True,
                        "result": {
                            "success": True,
                            "sketch_id": "base_sketch",
                            "kind": "line",
                            "entity_token": "line-token",
                            "line_id": "left_edge",
                            "point_tokens": {"start": "p0", "end": "p1"},
                        },
                    },
                    {
                        "index": 1,
                        "operation": "add_circle",
                        "success": True,
                        "result": {
                            "success": True,
                            "sketch_id": "base_sketch",
                            "kind": "circle",
                            "entity_token": "circle-token",
                            "circle_id": "center_hole",
                            "point_tokens": {"center": "pc"},
                        },
                    },
                ],
            }
        ]
    )
    _seed_committed_sketch(manager)
    responses = [
        {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_batch",
                    "name": "add_sketch_geometry_batch",
                    "input": {
                        "sketch_id": "base_sketch",
                        "description": "Add independent sketch primitives",
                        "operations": [
                            {
                                "operation": "add_line",
                                "start_u": 0,
                                "start_v": 0,
                                "end_u": 1,
                                "end_v": 0,
                                "line_id": "left_edge",
                            },
                            {
                                "operation": "add_circle",
                                "center_u": 0.5,
                                "center_v": 0.5,
                                "radius": 0.2,
                                "circle_id": "center_hole",
                            },
                        ],
                    },
                }
            ],
        },
        _end_turn_response(),
    ]

    async def fake_call_claude_with_tools(*args, **kwargs):
        return responses.pop(0)

    async def fake_runtime_sync(*args, **kwargs):
        return None

    async def fake_checkpoint(*args, **kwargs):
        return None

    async def fake_refresh_and_enrich(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)
    monkeypatch.setattr(agent_workflow, "_capture_operation_checkpoint", fake_checkpoint)
    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh_and_enrich)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-batch",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Add the sketch details"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "batch-1"},
            feature_snapshot=None,
        )
    )

    execute_payloads = [payload for payload in manager.sent_messages if payload.get("type") == "execute_code"]
    assert len(execute_payloads) == 1
    assert execute_payloads[0]["operation"] == "add_sketch_geometry_batch"
    assert "left_edge" in execute_payloads[0]["code"]
    assert "center_hole" in execute_payloads[0]["code"]

    assert [op["type"] for op in manager.ir_state["operations"]] == ["create_sketch", "add_line", "add_circle"]
    tool_result_texts = _extract_tool_result_texts(manager.history)
    assert any("Sketch geometry batch succeeded: 2/2 operations" in text for text in tool_result_texts)

    sketch_store = agent_workflow._get_sketch_entity_store("s-batch", manager)  # type: ignore[arg-type]
    refs, aliases = sketch_store.get_ref_and_alias_maps("base_sketch")
    assert refs == {"line_0": "line-token", "circle_0": "circle-token"}
    assert aliases["left_edge"] == "line_0"
    assert aliases["center_hole"] == "circle_0"


def test_sketch_geometry_batch_persists_operation_checkpoint_on_success(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeBatchManager(
        fusion_results=[
            {
                "tool_use_id": "toolu_batch",
                "success": True,
                "batch_size": 1,
                "success_count": 1,
                "failure_count": 0,
                "sketch_id": "base_sketch",
                "operations": [
                    {
                        "index": 0,
                        "operation": "add_line",
                        "success": True,
                        "result": {
                            "success": True,
                            "sketch_id": "base_sketch",
                            "kind": "line",
                            "entity_token": "line-token",
                            "line_id": "top_edge",
                            "point_tokens": {"start": "p0", "end": "p1"},
                        },
                    }
                ],
            }
        ]
    )
    _seed_committed_sketch(manager)
    responses = [
        {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_batch",
                    "name": "add_sketch_geometry_batch",
                    "input": {
                        "sketch_id": "base_sketch",
                        "description": "Add top edge",
                        "operations": [
                            {
                                "operation": "add_line",
                                "start_u": 0,
                                "start_v": 1,
                                "end_u": 1,
                                "end_v": 1,
                                "line_id": "top_edge",
                            }
                        ],
                    },
                }
            ],
        },
        _end_turn_response(),
    ]

    async def fake_call_claude_with_tools(*args, **kwargs):
        return responses.pop(0)

    async def fake_runtime_sync(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-batch",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Add top edge"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "batch-checkpoint"},
            feature_snapshot=None,
        )
    )

    checkpoints = manager.get_operation_checkpoints("s-batch")
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    assert checkpoint["tool_name"] == "add_sketch_geometry_batch"
    assert checkpoint["description"] == "Add top edge"
    assert [op["type"] for op in checkpoint["ir_state"]["operations"]] == ["create_sketch", "add_line"]
    assert checkpoint["conversation_snapshot"][1]["content"][-1]["id"] == "toolu_batch"
    assert any(
        payload.get("type") == "operation_checkpoint_created"
        and payload.get("checkpoint", {}).get("tool_name") == "add_sketch_geometry_batch"
        for payload in manager.sent_messages
    )


def test_sketch_geometry_batch_rejects_unsupported_operation(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeBatchManager()
    _seed_committed_sketch(manager)
    responses = [
        {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_batch",
                    "name": "add_sketch_geometry_batch",
                    "input": {
                        "sketch_id": "base_sketch",
                        "description": "Invalid batch",
                        "operations": [
                            {"operation": "add_line", "start_u": 0, "start_v": 0, "end_u": 1, "end_v": 0},
                            {"operation": "extrude_profile", "distance": 1},
                        ],
                    },
                }
            ],
        },
        _end_turn_response(),
    ]

    async def fake_call_claude_with_tools(*args, **kwargs):
        return responses.pop(0)

    async def fake_runtime_sync(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-batch",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Add sketch details"}]}],
            max_iterations=2,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "batch-invalid"},
            feature_snapshot=None,
        )
    )

    assert not any(payload.get("type") == "execute_code" for payload in manager.sent_messages)
    assert manager.get_operation_checkpoints("s-batch") == []
    tool_result_texts = _extract_tool_result_texts(manager.history)
    assert any("Unsupported batch operation 'extrude_profile'" in text for text in tool_result_texts)
    assert [op["type"] for op in manager.ir_state["operations"]] == ["create_sketch"]


def test_sketch_geometry_batch_without_committed_operations_skips_checkpoint(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeBatchManager(
        fusion_results=[
            {
                "tool_use_id": "toolu_batch",
                "success": False,
                "batch_size": 1,
                "success_count": 0,
                "failure_count": 1,
                "sketch_id": "base_sketch",
                "operations": [
                    {
                        "index": 0,
                        "operation": "add_line",
                        "success": False,
                        "error": "Sketch line creation failed",
                    }
                ],
            }
        ]
    )
    _seed_committed_sketch(manager)
    responses = [
        {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_batch",
                    "name": "add_sketch_geometry_batch",
                    "input": {
                        "sketch_id": "base_sketch",
                        "description": "Add failing edge",
                        "operations": [
                            {
                                "operation": "add_line",
                                "start_u": 0,
                                "start_v": 1,
                                "end_u": 1,
                                "end_v": 1,
                                "line_id": "top_edge",
                            }
                        ],
                    },
                }
            ],
        },
        _end_turn_response(),
    ]

    async def fake_call_claude_with_tools(*args, **kwargs):
        return responses.pop(0)

    async def fake_runtime_sync(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-batch",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Add top edge"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "batch-no-commit"},
            feature_snapshot=None,
        )
    )

    assert manager.get_operation_checkpoints("s-batch") == []
    assert not any(payload.get("type") == "operation_checkpoint_created" for payload in manager.sent_messages)


def test_sketch_geometry_batch_defers_downstream_extrude_in_same_turn(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeBatchManager(
        fusion_results=[
            {
                "tool_use_id": "toolu_batch",
                "success": True,
                "batch_size": 1,
                "success_count": 1,
                "failure_count": 0,
                "sketch_id": "base_sketch",
                "operations": [
                    {
                        "index": 0,
                        "operation": "add_rectangle",
                        "success": True,
                        "result": {
                            "success": True,
                            "sketch_id": "base_sketch",
                            "kind": "rectangle",
                            "rectangle_id": "base_rect",
                            "entities": [
                                {"kind": "line", "entity_token": "line-0", "point_tokens": {}},
                                {"kind": "line", "entity_token": "line-1", "point_tokens": {}},
                                {"kind": "line", "entity_token": "line-2", "point_tokens": {}},
                                {"kind": "line", "entity_token": "line-3", "point_tokens": {}},
                            ],
                        },
                    }
                ],
            }
        ]
    )
    _seed_committed_sketch(manager)
    _seed_sketch_alias(manager)
    responses = [
        {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_batch",
                    "name": "add_sketch_geometry_batch",
                    "input": {
                        "sketch_id": "Base Sketch",
                        "description": "Add base rectangle",
                        "operations": [
                            {
                                "operation": "add_rectangle",
                                "sketch_id": "Base Sketch",
                                "corner1_u": -1,
                                "corner1_v": -1,
                                "corner2_u": 1,
                                "corner2_v": 1,
                                "rectangle_id": "base_rect",
                            }
                        ],
                    },
                },
                {
                    "type": "tool_use",
                    "id": "toolu_extrude",
                    "name": "extrude_profile",
                    "input": {"sketch_id": "Base Sketch", "profile_index": 0, "distance": 1},
                },
            ],
        },
        _end_turn_response(),
    ]

    async def fake_call_claude_with_tools(*args, **kwargs):
        return responses.pop(0)

    async def fake_runtime_sync(*args, **kwargs):
        return None

    async def fake_checkpoint(*args, **kwargs):
        return None

    async def fake_refresh_and_enrich(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)
    monkeypatch.setattr(agent_workflow, "_capture_operation_checkpoint", fake_checkpoint)
    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh_and_enrich)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-batch",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Make a base block"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "batch-extrude"},
            feature_snapshot=None,
        )
    )

    execute_payloads = [payload for payload in manager.sent_messages if payload.get("type") == "execute_code"]
    assert [payload["operation"] for payload in execute_payloads] == ["add_sketch_geometry_batch"]
    assert [op["type"] for op in manager.ir_state["operations"]] == ["create_sketch", "add_rectangle"]
    tool_result_texts = _extract_tool_result_texts(manager.history)
    assert any("Deferred 'extrude_profile' after batched sketch geometry" in text for text in tool_result_texts)


def test_sketch_geometry_batch_defers_same_turn_face_sketch_followup_when_parent_uses_alias(
    monkeypatch: pytest.MonkeyPatch,
):
    manager = _FakeBatchManager(
        fusion_results=[
            {
                "tool_use_id": "toolu_face_sketch",
                "success": True,
                "sketch_id": "base_sketch",
                "sketch_name": "Base Sketch",
                "message": "create_sketch ok",
            }
        ]
    )
    _seed_face_entity(manager)
    responses = [
        {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_face_sketch",
                    "name": "create_sketch",
                    "input": {
                        "plane_id": "face_0",
                        "sketch_id": "base_sketch",
                        "sketch_name": "Base Sketch",
                    },
                },
                {
                    "type": "tool_use",
                    "id": "toolu_batch",
                    "name": "add_sketch_geometry_batch",
                    "input": {
                        "sketch_id": "Base Sketch",
                        "description": "Add slot outline",
                        "operations": [
                            {
                                "operation": "add_line",
                                "start_u": 0,
                                "start_v": 0,
                                "end_u": 1,
                                "end_v": 0,
                                "line_id": "slot_edge",
                            }
                        ],
                    },
                },
            ],
        },
        _end_turn_response(),
    ]

    async def fake_call_claude_with_tools(*args, **kwargs):
        return responses.pop(0)

    async def fake_runtime_sync(*args, **kwargs):
        return None

    async def fake_refresh(*args, **kwargs):
        return None

    async def fake_checkpoint(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)
    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh)
    monkeypatch.setattr(agent_workflow, "_capture_operation_checkpoint", fake_checkpoint)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-batch",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Sketch slot on selected face"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "batch-face-guard"},
            feature_snapshot=None,
        )
    )

    execute_payloads = [payload for payload in manager.sent_messages if payload.get("type") == "execute_code"]
    assert [payload["operation"] for payload in execute_payloads] == ["create_sketch"]
    tool_result_texts = _extract_tool_result_texts(manager.history)
    assert any("Deferred 'add_sketch_geometry_batch' for sketch 'base_sketch'" in text for text in tool_result_texts)


def test_sketch_geometry_batch_defers_downstream_loft_in_same_turn_when_profile_ids_reference_batched_sketch(
    monkeypatch: pytest.MonkeyPatch,
):
    manager = _FakeBatchManager(
        fusion_results=[
            {
                "tool_use_id": "toolu_batch",
                "success": True,
                "batch_size": 1,
                "success_count": 1,
                "failure_count": 0,
                "sketch_id": "base_sketch",
                "operations": [
                    {
                        "index": 0,
                        "operation": "add_rectangle",
                        "success": True,
                        "result": {
                            "success": True,
                            "sketch_id": "base_sketch",
                            "kind": "rectangle",
                            "rectangle_id": "base_rect",
                            "entities": [
                                {"kind": "line", "entity_token": "line-0", "point_tokens": {}},
                                {"kind": "line", "entity_token": "line-1", "point_tokens": {}},
                                {"kind": "line", "entity_token": "line-2", "point_tokens": {}},
                                {"kind": "line", "entity_token": "line-3", "point_tokens": {}},
                            ],
                        },
                    }
                ],
            },
            {
                "tool_use_id": "toolu_loft",
                "success": True,
                "message": "Loft created.",
            },
        ]
    )
    _seed_committed_sketch(manager, sketch_id="base_sketch")
    _seed_committed_profiled_sketch(manager, sketch_id="top_sketch")
    responses = [
        {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_batch",
                    "name": "add_sketch_geometry_batch",
                    "input": {
                        "sketch_id": "base_sketch",
                        "description": "Add loft base profile",
                        "operations": [
                            {
                                "operation": "add_rectangle",
                                "corner1_u": -1,
                                "corner1_v": -1,
                                "corner2_u": 1,
                                "corner2_v": 1,
                                "rectangle_id": "base_rect",
                            }
                        ],
                    },
                },
                {
                    "type": "tool_use",
                    "id": "toolu_loft",
                    "name": "create_loft",
                    "input": {
                        "profile_ids": ["base_sketch", "top_sketch"],
                        "operation": "NewBody",
                        "description": "Loft profiles",
                    },
                },
            ],
        },
        _end_turn_response(),
    ]

    async def fake_call_claude_with_tools(*args, **kwargs):
        return responses.pop(0)

    async def fake_runtime_sync(*args, **kwargs):
        return None

    async def fake_checkpoint(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)
    monkeypatch.setattr(agent_workflow, "_capture_operation_checkpoint", fake_checkpoint)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-batch",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Make loft profiles"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "batch-loft"},
            feature_snapshot=None,
        )
    )

    execute_payloads = [payload for payload in manager.sent_messages if payload.get("type") == "execute_code"]
    assert [payload["operation"] for payload in execute_payloads] == ["add_sketch_geometry_batch"]
    assert [op["type"] for op in manager.ir_state["operations"]] == [
        "create_sketch",
        "create_sketch",
        "add_rectangle",
        "add_rectangle",
    ]
    tool_result_texts = _extract_tool_result_texts(manager.history)
    assert any("Deferred 'create_loft' after batched sketch geometry" in text for text in tool_result_texts)


def test_sketch_tools_cluster_exposes_batch_tool():
    tool_names = [tool["name"] for tool in get_cluster_tools(["sketch_tools"])]
    assert "add_sketch_geometry_batch" in tool_names
