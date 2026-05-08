"""Regression tests for narrow sketch geometry batching."""

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import pytest

try:
    from . import agent_workflow
    from .ir import IRDocumentState, map_tool_call_to_ir
    from .prompt_structure import get_cluster_tools
except ImportError:  # pragma: no cover
    from backend.backend import agent_workflow
    from backend.backend.ir import IRDocumentState, map_tool_call_to_ir
    from backend.backend.prompt_structure import get_cluster_tools


@dataclass
class _FakeReasoningContext:
    entries: List[Any] = field(default_factory=list)

    def add_entry(self, entry: Any) -> None:
        self.entries.append(entry)

    def get_injection_text(self) -> str:
        return ""


class _FakeBatchManager:
    def __init__(self, fusion_results: Optional[List[Mapping[str, Any]]] = None) -> None:
        self.sent_messages: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.ir_state: Dict[str, Any] = {}
        self.reasoning_context = _FakeReasoningContext()
        self.fusion_results: List[Mapping[str, Any]] = list(fusion_results or [])

    def get_user_token(self, session_id: str) -> str:
        return "token"

    def get_llm_api_keys(self, session_id: str) -> Dict[str, str]:
        return {}

    def get_reasoning_context(self, session_id: str) -> _FakeReasoningContext:
        return self.reasoning_context

    def get_active_build_plan(self, session_id: str) -> None:
        return None

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


def _seed_committed_sketch(manager: _FakeBatchManager, session_id: str = "s-batch", sketch_id: str = "base_sketch") -> None:
    state = IRDocumentState(metadata={"source": "fusion", "session_id": session_id})
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

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)
    monkeypatch.setattr(agent_workflow, "_capture_operation_checkpoint", fake_checkpoint)

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
    tool_result_texts = _extract_tool_result_texts(manager.history)
    assert any("Unsupported batch operation 'extrude_profile'" in text for text in tool_result_texts)
    assert [op["type"] for op in manager.ir_state["operations"]] == ["create_sketch"]


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
                        "description": "Add base rectangle",
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
                    "id": "toolu_extrude",
                    "name": "extrude_profile",
                    "input": {"sketch_id": "base_sketch", "profile_index": 0, "distance": 1},
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


def test_sketch_tools_cluster_exposes_batch_tool():
    tool_names = [tool["name"] for tool in get_cluster_tools(["sketch_tools"])]
    assert "add_sketch_geometry_batch" in tool_names
