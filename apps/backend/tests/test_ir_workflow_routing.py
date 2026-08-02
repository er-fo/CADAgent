import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from backend import agent_workflow
from backend.backends.base import TargetExecutionResult
from backend.entity_store import EntityStore

@dataclass
class _FakeReasoningContext:
    entries: List[Any] = field(default_factory=list)

    def add_entry(self, entry: Any) -> None:
        self.entries.append(entry)

    def get_injection_text(self) -> str:
        return ""


class _FakeManager:
    def __init__(self) -> None:
        self.sent_messages: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.ir_state: Dict[str, Any] = {}
        self.reasoning_context = _FakeReasoningContext()
        self.active_build_plan: Optional[Dict[str, Any]] = None

    def get_user_token(self, session_id: str) -> Optional[str]:
        return "token"

    def get_llm_api_keys(self, session_id: str) -> Dict[str, str]:
        return {}

    def get_reasoning_context(self, session_id: str) -> _FakeReasoningContext:
        return self.reasoning_context

    async def send_message(self, session_id: str, payload: Dict[str, Any]) -> None:
        self.sent_messages.append(payload)

    def set_conversation_history(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self.history = list(messages)

    def save_ir_document_state(self, session_id: str, ir_state: Dict[str, Any]) -> None:
        self.ir_state = copy.deepcopy(ir_state)

    def get_ir_document_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        return copy.deepcopy(self.ir_state) if self.ir_state else None

    def clear_ir_document_state(self, session_id: str) -> None:
        self.ir_state = {}

    def get_active_build_plan(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.active_build_plan


def _tool_use_response() -> Dict[str, Any]:
    return {
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "Planning geometry."},
            {"type": "tool_use", "id": "toolu_1", "name": "create_sketch", "input": {"plane_id": "XY", "sketch_id": "sketch_0"}},
            {
                "type": "tool_use",
                "id": "toolu_2",
                "name": "add_circle",
                "input": {"sketch_id": "sketch_0", "center_u": 0, "center_v": 0, "radius": 10},
            },
            {
                "type": "tool_use",
                "id": "toolu_3",
                "name": "extrude_profile",
                "input": {"sketch_id": "sketch_0", "profile_index": 0, "distance": 40, "operation": "NewBody"},
            },
        ],
    }


def _end_turn_response() -> Dict[str, Any]:
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Done."}]}


def _extract_tool_result_texts(messages: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content_blocks = message.get("content")
        if not isinstance(content_blocks, list):
            continue
        for content_block in content_blocks:
            if not isinstance(content_block, dict) or content_block.get("type") != "tool_result":
                continue
            tool_content = content_block.get("content")
            if not isinstance(tool_content, list):
                continue
            for text_block in tool_content:
                if isinstance(text_block, dict) and text_block.get("type") == "text":
                    texts.append(str(text_block.get("text") or ""))
    return texts


def test_execute_workflow_routes_ir_tools_to_build123d_adapter(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeManager()
    call_count = {"llm": 0, "build_exec": 0}

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        return _tool_use_response() if call_count["llm"] == 1 else _end_turn_response()

    async def fake_execute_document(session_id, document, request_id=None):
        call_count["build_exec"] += 1
        return TargetExecutionResult(
            success=True,
            target="build123d",
            message="ok",
            data={"entities": {"faces": 3, "edges": 2, "vertices": 1, "volume_mm3": 1.0}},
        )

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow._BUILD123D_EXECUTOR, "execute_document", fake_execute_document)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s1",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Create a cylinder radius 10 height 40"}]}],
            max_iterations=4,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "build123d", "request_id": "r1"},
            feature_snapshot=None,
        )
    )

    assert call_count["build_exec"] == 3  # create_sketch + add_circle + extrude
    assert any(msg.get("type") == "ir_operation" for msg in manager.sent_messages)
    assert any(msg.get("type") == "studio_geometry_update" for msg in manager.sent_messages)


def test_execute_workflow_routes_ir_tools_to_fusion_adapter(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeManager()
    call_count = {"llm": 0, "fusion_exec": 0}

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        return _tool_use_response() if call_count["llm"] == 1 else _end_turn_response()

    async def fake_execute_operation(self, session_id, operation, tool_use_id, description=""):
        call_count["fusion_exec"] += 1
        return TargetExecutionResult(
            success=True,
            target="fusion",
            message="ok",
            raw_result={"success": True, "tool_use_id": tool_use_id, "message": f"{operation.type} ok"},
        )

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow.FusionTargetExecutor, "execute_operation", fake_execute_operation)

    refresh_payloads = []

    async def fake_refresh_after_success(session_id, manager, tool_name, result, messages):
        refresh_payloads.append({"tool_name": tool_name, "tool_use_id": result.get("tool_use_id")})

    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh_after_success)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s2",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Create a cylinder radius 10 height 40"}]}],
            max_iterations=4,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "r2"},
            feature_snapshot=None,
        )
    )

    assert call_count["fusion_exec"] == 3
    assert [item["tool_use_id"] for item in refresh_payloads] == ["toolu_1", "toolu_2", "toolu_3"]


def test_execute_workflow_reuses_committed_ir_state_across_fusion_requests(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeManager()
    responses = [
        {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_create",
                    "name": "create_sketch",
                    "input": {"plane_id": "XY", "sketch_id": "base_square"},
                }
            ],
        },
        _end_turn_response(),
        {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_rect",
                    "name": "add_rectangle",
                    "input": {
                        "sketch_id": "base_square",
                        "corner1_u": -2.5,
                        "corner1_v": -2.5,
                        "corner2_u": 2.5,
                        "corner2_v": 2.5,
                    },
                }
            ],
        },
        _end_turn_response(),
    ]
    executed_ops: List[str] = []

    async def fake_call_claude_with_tools(*args, **kwargs):
        return responses.pop(0)

    async def fake_execute_operation(self, session_id, operation, tool_use_id, description=""):
        executed_ops.append(operation.type)
        return TargetExecutionResult(
            success=True,
            target="fusion",
            message=f"{operation.type} ok",
            raw_result={"success": True, "tool_use_id": tool_use_id, "message": f"{operation.type} ok"},
        )

    async def fake_refresh_after_success(*args, **kwargs):
        return None

    async def fake_runtime_sync(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow.FusionTargetExecutor, "execute_operation", fake_execute_operation)
    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh_after_success)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-topology-defer",
            messages=[{"role": "user", "content": [{"type": "text", "text": "two extrudes"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "topology-1"},
            feature_snapshot=None,
        )
    )

    assert executed_ops == ["create_sketch"]
    assert [op["type"] for op in manager.ir_state["operations"]] == ["create_sketch"]

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-ir-continuity",
            messages=[{"role": "user", "content": [{"type": "text", "text": "add rectangle"}]}],
            max_iterations=2,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "r-rect"},
            feature_snapshot=None,
        )
    )

    assert executed_ops == ["create_sketch", "add_rectangle"]
    assert [op["type"] for op in manager.ir_state["operations"]] == ["create_sketch", "add_rectangle"]
    assert manager.ir_state["operations"][1]["dependencies"] == ["op_1"]
    assert not any("IR validation failed" in text for text in _extract_tool_result_texts(manager.history))


def test_execute_workflow_reuses_committed_ir_state_across_same_session_requests(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeManager()
    responses = [
        {
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "toolu_sketch", "name": "create_sketch", "input": {"plane_id": "XY", "sketch_id": "s0"}},
                {"type": "tool_use", "id": "toolu_circle", "name": "add_circle", "input": {"sketch_id": "s0", "center_u": 0, "center_v": 0, "radius": 1}},
            ],
        },
        _end_turn_response(),
        {
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "toolu_extrude", "name": "extrude_profile", "input": {"sketch_id": "s0", "profile_index": 0, "distance": 2}},
            ],
        },
        _end_turn_response(),
    ]
    executed_ops: List[tuple[str, str, List[str]]] = []

    async def fake_call_claude_with_tools(*args, **kwargs):
        return responses.pop(0)

    async def fake_execute_operation(self, session_id, operation, tool_use_id, description=""):
        executed_ops.append((operation.id, operation.type, list(operation.dependencies)))
        return TargetExecutionResult(
            success=True,
            target="fusion",
            message=f"{operation.type} ok",
            data={"tool_name": operation.type, "tool_input": {"sketch_id": "s0", "plane_id": "XY"}},
            raw_result={"success": True, "tool_use_id": tool_use_id, "message": f"{operation.type} ok"},
        )

    async def fake_refresh_after_success(*args, **kwargs):
        return None

    async def fake_runtime_sync(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow.FusionTargetExecutor, "execute_operation", fake_execute_operation)
    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh_after_success)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)

    for request_id, prompt in (("state-1", "Create circle sketch"), ("state-2", "Extrude it")):
        asyncio.run(
            agent_workflow._execute_workflow_loop(
                session_id="s-persist-ir",
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                max_iterations=3,
                model_name=None,
                manager=manager,  # type: ignore[arg-type]
                last_user_message_sent=None,
                request={"execution_target": "fusion", "request_id": request_id},
                feature_snapshot=None,
            )
        )

    assert [(op_id, op_type) for op_id, op_type, _deps in executed_ops] == [
        ("op_1", "create_sketch"),
        ("op_2", "add_circle"),
        ("op_3", "extrude"),
    ]
    assert executed_ops[-1][2] == ["op_2", "op_1"]
    state = manager.get_ir_document_state("s-persist-ir")
    assert state is not None
    assert [operation["id"] for operation in state["operations"]] == ["op_1", "op_2", "op_3"]
    committed_events = [msg for msg in manager.sent_messages if msg.get("type") == "ir_operation_committed"]
    assert [event["operation"]["id"] for event in committed_events] == ["op_1", "op_2", "op_3"]


def test_execute_workflow_commits_target_results_into_ir(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeManager()
    call_count = {"llm": 0}
    appended_ops: List[Any] = []
    original_append = agent_workflow.IRDocumentState.append

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        if call_count["llm"] == 1:
            return {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_sketch",
                        "name": "create_sketch",
                        "input": {"plane_id": "XY", "sketch_id": "sketch_0"},
                    }
                ],
            }
        return _end_turn_response()

    async def fake_execute_operation(self, session_id, operation, tool_use_id, description=""):
        return TargetExecutionResult(
            success=True,
            target="fusion",
            message="sketch ok",
            data={"tool_name": "create_sketch", "tool_input": {"plane_id": "XY", "sketch_id": "sketch_0"}},
            raw_result={
                "success": True,
                "tool_use_id": tool_use_id,
                "message": "sketch ok",
                "created_entities": {"sketches": ["sketch_0"]},
                "warnings": ["minor warning"],
            },
        )

    def capture_append(self, operation):
        appended_ops.append(operation)
        original_append(self, operation)

    async def fake_refresh_after_success(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow.FusionTargetExecutor, "execute_operation", fake_execute_operation)
    monkeypatch.setattr(agent_workflow.IRDocumentState, "append", capture_append)
    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh_after_success)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-target-results",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Create sketch"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "target-results-1"},
            feature_snapshot=None,
        )
    )

    assert appended_ops
    target_result = appended_ops[0].target_results[0]
    assert target_result["target"] == "fusion"
    assert target_result["success"] is True
    assert target_result["created_entities"] == {"sketches": ["sketch_0"]}
    assert target_result["warnings"] == ["minor warning"]
    assert target_result["raw_result"]["tool_use_id"] == "toolu_sketch"
    committed_events = [msg for msg in manager.sent_messages if msg.get("type") == "ir_operation_committed"]
    assert committed_events
    assert committed_events[0]["operation"]["target_results"][0]["raw_result"]["tool_use_id"] == "toolu_sketch"


def test_execute_workflow_commits_feature_target_results_into_ir(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeManager()
    call_count = {"llm": 0}
    appended_ops: List[Any] = []
    original_append = agent_workflow.IRDocumentState.append

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        if call_count["llm"] == 1:
            return {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_features",
                        "name": "list_features",
                        "input": {"description": "inspect timeline"},
                    }
                ],
            }
        return _end_turn_response()

    async def fake_execute_feature_tool_call(*args, **kwargs):
        return True, "features ok", {"features": [{"entity_token": "feature_token_0"}], "warnings": ["cached"]}

    async def fake_refresh_after_success(*args, **kwargs):
        return None

    def capture_append(self, operation):
        appended_ops.append(operation)
        original_append(self, operation)

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow, "_execute_feature_tool_call", fake_execute_feature_tool_call)
    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh_after_success)
    monkeypatch.setattr(agent_workflow.IRDocumentState, "append", capture_append)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-feature-target-results",
            messages=[{"role": "user", "content": [{"type": "text", "text": "List features"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "feature-target-results-1"},
            feature_snapshot=None,
        )
    )

    assert appended_ops
    target_result = appended_ops[0].target_results[0]
    assert target_result["target"] == "fusion"
    assert target_result["raw_result"]["features"][0]["entity_token"] == "feature_token_0"
    assert target_result["warnings"] == ["cached"]
    committed_events = [msg for msg in manager.sent_messages if msg.get("type") == "ir_operation_committed"]
    assert committed_events
    assert committed_events[0]["operation"]["target_results"][0]["raw_result"]["features"][0]["entity_token"] == "feature_token_0"


def test_execute_feature_tool_call_returns_list_features_snapshot():
    class _FeatureSnapshotManager(_FakeManager):
        def __init__(self) -> None:
            super().__init__()
            self.feature_snapshot: Optional[Dict[str, Any]] = None

        async def wait_for_fusion_result(self, session_id: str, timeout=None):
            request = next(msg for msg in reversed(self.sent_messages) if msg.get("type") == "feature_snapshot_request")
            return {
                "type": "feature_snapshot",
                "success": True,
                "message_id": request["message_id"],
                "features": [
                    {
                        "entity_token": "feature_token_0",
                        "name": "Extrude 1",
                        "timeline_index": 3,
                    }
                ],
            }

        async def store_fusion_result(self, session_id: str, result: Dict[str, Any]) -> None:
            pass

        def set_feature_snapshot(self, session_id: str, snapshot: Dict[str, Any]) -> None:
            self.feature_snapshot = dict(snapshot)

        def get_feature_snapshot(self, session_id: str) -> Optional[Dict[str, Any]]:
            return dict(self.feature_snapshot) if self.feature_snapshot else None

    async def _run():
        manager = _FeatureSnapshotManager()
        success, message, raw = await agent_workflow._execute_feature_tool_call(
            "s-list-features",
            manager,  # type: ignore[arg-type]
            "list_features",
            "toolu_features",
            {"description": "inspect timeline"},
        )

        assert success is True
        assert "Extrude 1" in message
        assert raw["features"][0]["entity_token"] == "feature_token_0"

    asyncio.run(_run())


def test_execute_workflow_commits_selection_raw_target_results(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeManager()
    appended_ops: List[Any] = []
    original_append = agent_workflow.IRDocumentState.append

    async def fake_call_claude_with_tools(*args, **kwargs):
        return {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_select",
                    "name": "select_edges",
                    "input": {"edge_refs": ["e0"], "clear_existing": True},
                }
            ],
        }

    async def fake_execute_geometry_tool_call(*args, **kwargs):
        return True, "selected ok", {"success": True, "selected_count": 1, "missing_tokens": []}

    def capture_append(self, operation):
        appended_ops.append(operation)
        original_append(self, operation)

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow, "_validate_entity_store_for_tool", lambda *args, **kwargs: (True, "ok"))
    monkeypatch.setattr(agent_workflow, "_execute_geometry_tool_call", fake_execute_geometry_tool_call)
    monkeypatch.setattr(agent_workflow.IRDocumentState, "append", capture_append)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-selection-target-results",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Select edge"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "selection-target-results-1"},
            feature_snapshot=None,
        )
    )

    assert appended_ops
    target_result = appended_ops[0].target_results[0]
    assert target_result["raw_result"]["selected_count"] == 1
    committed_events = [msg for msg in manager.sent_messages if msg.get("type") == "ir_operation_committed"]
    assert committed_events
    assert committed_events[0]["operation"]["target_results"][0]["raw_result"]["selected_count"] == 1


def test_execute_workflow_commits_resolved_pattern_feature_refs(monkeypatch: pytest.MonkeyPatch):
    class _PatternManager(_FakeManager):
        def __init__(self) -> None:
            super().__init__()
            self.feature_snapshot: Optional[Dict[str, Any]] = None

        async def wait_for_fusion_result(self, session_id: str, timeout=None):
            last = self.sent_messages[-1]
            if last.get("type") == "feature_snapshot_request":
                return {
                    "type": "feature_snapshot",
                    "success": True,
                    "message_id": last["message_id"],
                    "features": [
                        {
                            "entity_token": "feature_token_latest",
                            "name": "Hole 1",
                            "timeline_index": 5,
                            "bounds_cm": {"min": [0, 0, 0], "max": [1, 1, 1]},
                        }
                    ],
                }
            return {
                "tool_use_id": "toolu_pattern",
                "success": True,
                "message": "pattern ok",
                "pattern_token": "pattern_token_0",
            }

        async def store_fusion_result(self, session_id: str, result: Dict[str, Any]) -> None:
            pass

        def set_feature_snapshot(self, session_id: str, snapshot: Dict[str, Any]) -> None:
            self.feature_snapshot = dict(snapshot)

        def get_feature_snapshot(self, session_id: str) -> Optional[Dict[str, Any]]:
            return dict(self.feature_snapshot) if self.feature_snapshot else None

    manager = _PatternManager()
    appended_ops: List[Any] = []
    original_append = agent_workflow.IRDocumentState.append

    async def fake_call_claude_with_tools(*args, **kwargs):
        return {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_pattern",
                    "name": "create_pattern_feature",
                    "input": {
                        "pattern_type": "rectangular",
                        "feature_tokens": ["auto_last"],
                        "count_x": 2,
                        "description": "pattern latest feature",
                    },
                }
            ],
        }

    async def fake_refresh_after_success(*args, **kwargs):
        return None

    def capture_append(self, operation):
        appended_ops.append(operation)
        original_append(self, operation)

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh_after_success)
    monkeypatch.setattr(agent_workflow.IRDocumentState, "append", capture_append)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-pattern-target-results",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Pattern latest feature"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "pattern-target-results-1"},
            feature_snapshot=None,
        )
    )

    assert appended_ops
    assert appended_ops[0].params.feature_refs == ["feature_token_latest"]
    target_result = appended_ops[0].target_results[0]
    assert target_result["raw_result"]["resolved_feature_refs"] == ["feature_token_latest"]
    committed_events = [msg for msg in manager.sent_messages if msg.get("type") == "ir_operation_committed"]
    assert committed_events
    assert committed_events[0]["operation"]["params"]["feature_refs"] == ["feature_token_latest"]


@pytest.mark.parametrize(
    "tool_call",
    [
        {
            "id": "toolu_select",
            "name": "select_edges",
            "input": {"edge_refs": ["e0"], "description": "select"},
        },
        {
            "id": "toolu_features",
            "name": "list_features",
            "input": {"description": "inspect"},
        },
    ],
)
def test_build123d_rejects_fusion_only_feature_and_selection_tools(
    monkeypatch: pytest.MonkeyPatch,
    tool_call: Dict[str, Any],
):
    manager = _FakeManager()
    call_count = {"llm": 0, "feature_exec": 0, "geometry_exec": 0}

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        if call_count["llm"] == 1:
            return {"stop_reason": "tool_use", "content": [{"type": "tool_use", **tool_call}]}
        return _end_turn_response()

    async def fake_execute_feature_tool_call(*args, **kwargs):
        call_count["feature_exec"] += 1
        return True, "should not execute", {}

    async def fake_execute_geometry_tool_call(*args, **kwargs):
        call_count["geometry_exec"] += 1
        return True, "should not execute"

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow, "_execute_feature_tool_call", fake_execute_feature_tool_call)
    monkeypatch.setattr(agent_workflow, "_execute_geometry_tool_call", fake_execute_geometry_tool_call)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-build-unsupported",
            messages=[{"role": "user", "content": [{"type": "text", "text": "run unsupported"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "build123d", "request_id": "unsupported-1"},
            feature_snapshot=None,
        )
    )

    assert call_count["feature_exec"] == 0
    assert call_count["geometry_exec"] == 0
    assert any(msg.get("message") == "Unsupported build123d tool" for msg in manager.sent_messages)
    assert any("not supported by the build123d target" in text for text in _extract_tool_result_texts(manager.history))


def test_build123d_routes_supported_feature_tools_to_ir_adapter(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeManager()
    call_count = {"llm": 0, "build_exec": 0}

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        if call_count["llm"] == 1:
            return {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_fillet",
                        "name": "apply_fillet",
                        "input": {"edge_refs": ["edge_0"], "radius": 2, "description": "round edge"},
                    }
                ],
            }
        return _end_turn_response()

    async def fake_execute_document(session_id, document, request_id=None):
        call_count["build_exec"] += 1
        assert document.operations[-1].type == "fillet"
        return TargetExecutionResult(
            success=True,
            target="build123d",
            message="feature ok",
            data={"entities": {"faces": 6, "edges": 15, "vertices": 10, "volume_mm3": 1.0}},
        )

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow._BUILD123D_EXECUTOR, "execute_document", fake_execute_document)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-build-feature",
            messages=[{"role": "user", "content": [{"type": "text", "text": "round edge"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "build123d", "request_id": "feature-1"},
            feature_snapshot=None,
        )
    )

    assert call_count["build_exec"] == 1
    assert not any(msg.get("message") == "Unsupported build123d tool" for msg in manager.sent_messages)
    assert any(msg.get("type") == "ir_operation_committed" for msg in manager.sent_messages)


def test_ir_routed_topology_mutations_are_deferred_after_first_mutation(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeManager()
    call_count = {"llm": 0}
    executed_ops: List[str] = []

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        if call_count["llm"] == 1:
            return {
                "stop_reason": "tool_use",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "create_sketch", "input": {"plane_id": "XY", "sketch_id": "s0"}},
                    {"type": "tool_use", "id": "toolu_2", "name": "add_circle", "input": {"sketch_id": "s0", "center_u": 0, "center_v": 0, "radius": 1}},
                    {"type": "tool_use", "id": "toolu_3", "name": "extrude_profile", "input": {"sketch_id": "s0", "profile_index": 0, "distance": 2}},
                    {"type": "tool_use", "id": "toolu_4", "name": "extrude_profile", "input": {"sketch_id": "s0", "profile_index": 0, "distance": 1}},
                ],
            }
        return _end_turn_response()

    async def fake_execute_operation(self, session_id, operation, tool_use_id, description=""):
        executed_ops.append(operation.type)
        return TargetExecutionResult(
            success=True,
            target="fusion",
            message=f"{operation.type} ok",
            raw_result={"success": True, "tool_use_id": tool_use_id, "message": f"{operation.type} ok"},
        )

    async def fake_refresh_after_success(*args, **kwargs):
        return None

    async def fake_runtime_sync(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow.FusionTargetExecutor, "execute_operation", fake_execute_operation)
    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh_after_success)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-topology-defer",
            messages=[{"role": "user", "content": [{"type": "text", "text": "two extrudes"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "topology-1"},
            feature_snapshot=None,
        )
    )

    assert executed_ops == ["create_sketch", "add_circle", "extrude"]
    assert any("Deferred 'extrude_profile'" in text for text in _extract_tool_result_texts(manager.history))


def test_execute_workflow_resolves_ir_create_sketch_face_ref_for_fusion(monkeypatch: pytest.MonkeyPatch):
    class _FaceRefStore:
        def resolve_token(self, ref_or_token: str, expected_kind: Optional[str] = None):
            if expected_kind == "face" and ref_or_token == "face_0":
                return "face_token_abc", None
            return None, f"Could not resolve reference '{ref_or_token}'"

        def get_refs_by_kind(self, kind: str):
            if kind == "face":
                return ["face_0"]
            return []

    class _FaceRefManager(_FakeManager):
        def __init__(self) -> None:
            super().__init__()
            self._store = _FaceRefStore()

        def get_entity_store(self, session_id: str):
            return self._store

    manager = _FaceRefManager()
    call_count = {"llm": 0}
    captured_planes: List[str] = []

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        if call_count["llm"] == 1:
            return {
                "stop_reason": "tool_use",
                "content": [
                    {"type": "tool_use", "id": "toolu_face", "name": "create_sketch", "input": {"plane_id": "face_0", "sketch_id": "sketch_on_face"}}
                ],
            }
        return _end_turn_response()

    async def fake_execute_operation(self, session_id, operation, tool_use_id, description=""):
        captured_planes.append(str(getattr(operation.params, "plane", "")))
        return TargetExecutionResult(
            success=True,
            target="fusion",
            message="ok",
            raw_result={"success": True, "tool_use_id": tool_use_id, "message": f"{operation.type} ok"},
        )

    async def fake_refresh_after_success(session_id, manager, tool_name, result, messages):
        return None

    async def fake_runtime_sync(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow.FusionTargetExecutor, "execute_operation", fake_execute_operation)
    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh_after_success)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-face-ref",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Sketch on face_0"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "r-face-ref"},
            feature_snapshot=None,
        )
    )

    assert captured_planes == ["face_token_abc"]


def test_execute_workflow_resolves_ir_codegen_refs_for_construction_plane_and_revolve(monkeypatch: pytest.MonkeyPatch):
    async def _setup_store() -> EntityStore:
        store = EntityStore()
        await store.register_entities(
            "face",
            [{"entity_token": "face_token_0", "surface_type": "planar", "area": 10.0}],
        )
        await store.register_entities(
            "edge",
            [{"entity_token": "edge_token_0", "length": 10.0}],
        )
        return store

    class _StoreManager(_FakeManager):
        def __init__(self, store: EntityStore) -> None:
            super().__init__()
            self._store = store

        def get_entity_store(self, session_id: str) -> EntityStore:
            return self._store

    manager = _StoreManager(asyncio.run(_setup_store()))
    call_count = {"llm": 0}
    captured: Dict[str, Any] = {}

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        if call_count["llm"] == 1:
            return {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_plane",
                        "name": "create_construction_plane",
                        "input": {
                            "plane_id": "face_plane",
                            "mode": "face_normal",
                            "face_token": "face_0",
                            "description": "plane on face",
                        },
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_sketch",
                        "name": "create_sketch",
                        "input": {"plane_id": "XY", "sketch_id": "profile_sketch"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_circle",
                        "name": "add_circle",
                        "input": {"sketch_id": "profile_sketch", "center_u": 1, "center_v": 0, "radius": 0.5},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_revolve",
                        "name": "revolve_profile",
                        "input": {
                            "sketch_id": "profile_sketch",
                            "profile_index": 0,
                            "axis": {"type": "edge", "edge_token": "e0"},
                            "extent": {"mode": "to", "to_entity_token": "face_0"},
                            "operation": "NewBody",
                            "description": "revolve to face",
                        },
                    },
                ],
            }
        return _end_turn_response()

    async def fake_execute_operation(self, session_id, operation, tool_use_id, description=""):
        if operation.type == "create_construction_plane":
            captured["plane_face"] = operation.params.face
        if operation.type == "revolve":
            captured["revolve_axis"] = dict(operation.params.axis)
            captured["revolve_extent"] = dict(operation.params.extent)
        return TargetExecutionResult(
            success=True,
            target="fusion",
            message=f"{operation.type} ok",
            raw_result={"success": True, "tool_use_id": tool_use_id, "message": f"{operation.type} ok"},
        )

    async def fake_refresh_after_success(*args, **kwargs):
        return None

    async def fake_runtime_sync(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow.FusionTargetExecutor, "execute_operation", fake_execute_operation)
    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh_after_success)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-codegen-ref-resolution",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Resolve codegen refs"}]}],
            max_iterations=4,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "r-codegen-ref-resolution"},
            feature_snapshot=None,
        )
    )

    assert captured["plane_face"] == "face_token_0"
    assert captured["revolve_axis"]["edge_token"] == "edge_token_0"
    assert captured["revolve_extent"]["to_entity_token"] == "face_token_0"


def test_execute_workflow_defers_same_turn_face_sketch_geometry(monkeypatch: pytest.MonkeyPatch):
    async def _seed_store(store: EntityStore) -> None:
        await store.register_entities(
            "face",
            [
                {
                    "entity_token": "face_token_0",
                    "normal": [1, 0, 0],
                    "centroid": [45.5, 0.0, 10.0],
                    "surface_type": "planar",
                    "area": 800.0,
                }
            ],
        )

    class _FaceStoreManager(_FakeManager):
        def __init__(self, store: EntityStore) -> None:
            super().__init__()
            self._store = store

        def get_entity_store(self, session_id: str) -> EntityStore:
            return self._store

    async def _setup_store() -> EntityStore:
        store = EntityStore()
        await _seed_store(store)
        return store

    store = asyncio.run(_setup_store())
    manager = _FaceStoreManager(store)

    call_count = {"llm": 0}
    executed_ops: List[str] = []

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        if call_count["llm"] == 1:
            return {
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text", "text": "Creating face sketch and ports."},
                    {
                        "type": "tool_use",
                        "id": "toolu_face_create",
                        "name": "create_sketch",
                        "input": {"plane_id": "face_0", "sketch_id": "usb_ports"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_face_rect",
                        "name": "add_rectangle",
                        "input": {
                            "sketch_id": "usb_ports",
                            "corner1_u": -1.0,
                            "corner1_v": -0.4,
                            "corner2_u": 1.0,
                            "corner2_v": 0.4,
                        },
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_face_extrude",
                        "name": "extrude_profile",
                        "input": {
                            "sketch_id": "usb_ports",
                            "profile_index": 0,
                            "distance": -1.0,
                            "operation": "Cut",
                        },
                    },
                ],
            }
        return _end_turn_response()

    async def fake_execute_operation(self, session_id, operation, tool_use_id, description=""):
        executed_ops.append(operation.type)
        return TargetExecutionResult(
            success=True,
            target="fusion",
            message=f"{operation.type} ok",
            raw_result={
                "success": True,
                "tool_use_id": tool_use_id,
                "sketch_id": "usb_ports" if operation.type == "create_sketch" else "usb_ports",
                "plane_id_input": "face_token_0" if operation.type == "create_sketch" else None,
            },
            data={"tool_name": operation.type, "tool_input": {"sketch_id": "usb_ports", "plane_id": "face_token_0"}},
        )

    async def fake_refresh_after_success(session_id, manager, tool_name, result, messages):
        return None

    async def fake_runtime_sync(*args, **kwargs):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow.FusionTargetExecutor, "execute_operation", fake_execute_operation)
    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh_after_success)
    monkeypatch.setattr(agent_workflow, "_ensure_runtime_entity_context_synced", fake_runtime_sync)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-face-seq",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Add usb ports on wall"}]}],
            max_iterations=4,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "r-face-seq"},
            feature_snapshot=None,
        )
    )

    assert executed_ops == ["create_sketch"]
    tool_result_texts = _extract_tool_result_texts(manager.history)
    assert any("Deferred 'add_rectangle'" in text for text in tool_result_texts)
    assert any("Deferred 'extrude_profile'" in text for text in tool_result_texts)


def test_execute_workflow_initial_routing_includes_active_build_plan(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeManager()
    manager.active_build_plan = {
        "design_name": "Bracket",
        "steps": [{"step_number": 1, "operation": "create_shell", "description": "Shell the body"}],
        "completed_steps": 0,
    }
    captured: Dict[str, Any] = {}

    async def fake_route_request(user_request, conversation_history, build_plan=None, api_keys=None):
        captured["build_plan"] = build_plan
        return {"required": ["core"], "optional": [], "reasoning": "test"}

    def fake_build_prompt(routing_result):
        return "system", []

    async def fake_call_claude_with_tools(*args, **kwargs):
        return _end_turn_response()

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", True)
    monkeypatch.setattr(agent_workflow, "route_request", fake_route_request)
    monkeypatch.setattr(agent_workflow, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-route",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Continue the active plan"}]}],
            max_iterations=2,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "route-1"},
            feature_snapshot=None,
        )
    )

    assert captured["build_plan"] == manager.active_build_plan


def test_target_resolution_and_deterministic_fallback_helpers():
    assert agent_workflow._resolve_execution_target({"execution_target": "studio"}) == "build123d"
    assert agent_workflow._resolve_execution_target({"execution_target": "build123d"}) == "build123d"
    assert agent_workflow._resolve_execution_target({"execution_target": "fusion"}) == "fusion"
    assert agent_workflow._resolve_execution_target({}) == "fusion"
    with pytest.raises(agent_workflow.UnsupportedExecutionTargetError):
        agent_workflow._resolve_execution_target({"execution_target": "unknown"})

    cube_calls = agent_workflow._deterministic_mvp_tool_calls("Create a 50mm cube")
    assert cube_calls and len(cube_calls) == 3
    assert cube_calls[0]["name"] == "create_sketch"
    assert cube_calls[1]["name"] == "add_rectangle"
    assert cube_calls[2]["name"] == "extrude_profile"


def test_handle_studio_export_request_uses_shared_build123d_executor(monkeypatch: pytest.MonkeyPatch, tmp_path):
    manager = _FakeManager()
    exported = tmp_path / "part.step"

    def fake_export_step(session_id: str, output_path: str) -> str:
        exported.write_text("ISO-10303-21;\nENDSEC;\n")
        return str(exported)

    monkeypatch.setattr(agent_workflow._BUILD123D_EXECUTOR, "export_step", fake_export_step)

    asyncio.run(
        agent_workflow.handle_studio_export_request(
            "studio_session",
            {"type": "studio_export_request", "format": "step", "output_path": str(exported)},
            manager,  # type: ignore[arg-type]
        )
    )

    assert any(msg.get("type") == "studio_export_complete" for msg in manager.sent_messages)


def test_deterministic_fallback_runs_once_then_ends_turn(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeManager()
    call_count = {"llm": 0, "build_exec": 0}

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        raise ValueError("No OpenAI API key available for this session")

    async def fake_execute_document(session_id, document, request_id=None):
        call_count["build_exec"] += 1
        return TargetExecutionResult(
            success=True,
            target="build123d",
            message="ok",
            data={"entities": {"faces": 6, "edges": 12, "vertices": 8, "volume_mm3": 125000.0}},
        )

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow._BUILD123D_EXECUTOR, "execute_document", fake_execute_document)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-fallback",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Create a 50mm cube"}]}],
            max_iterations=5,
            model_name="gpt-5",
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "build123d", "request_id": "fallback-1"},
            feature_snapshot=None,
        )
    )

    # LLM called twice: first produces deterministic tool_use fallback, second forced end_turn.
    assert call_count["llm"] == 2
    assert call_count["build_exec"] == 3
    assert len([m for m in manager.sent_messages if m.get("type") == "runtime_fallback"]) == 1
    assert any(m.get("type") == "completed" for m in manager.sent_messages)


def test_execute_workflow_rejects_unknown_execution_target_without_fallback(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeManager()
    call_count = {"llm": 0}

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        return _end_turn_response()

    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-target-error",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Create a cube"}]}],
            max_iterations=2,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "unknown-target", "request_id": "r-target-error"},
            feature_snapshot=None,
        )
    )

    assert call_count["llm"] == 0
    error_messages = [m for m in manager.sent_messages if m.get("type") == "error"]
    assert error_messages
    assert error_messages[0]["message"] == "Invalid execution target"
    assert "unknown-target" in error_messages[0]["details"]


def test_execute_workflow_reports_ir_mapping_errors_for_invalid_numeric_inputs(monkeypatch: pytest.MonkeyPatch):
    manager = _FakeManager()
    call_count = {"llm": 0, "build_exec": 0}

    def tool_use_with_invalid_numeric() -> Dict[str, Any]:
        return {
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Planning geometry."},
                {
                    "type": "tool_use",
                    "id": "toolu_invalid",
                    "name": "add_circle",
                    "input": {"sketch_id": "sketch_0", "center_u": 0, "center_v": 0, "radius": "bad"},
                },
            ],
        }

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        return tool_use_with_invalid_numeric() if call_count["llm"] == 1 else _end_turn_response()

    async def fake_execute_document(session_id, document, request_id=None):
        call_count["build_exec"] += 1
        return TargetExecutionResult(success=True, target="build123d", message="ok", data={"entities": {}})

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow._BUILD123D_EXECUTOR, "execute_document", fake_execute_document)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-mapping-error",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Create a circle"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "build123d", "request_id": "r-mapping-error"},
            feature_snapshot=None,
        )
    )

    assert call_count["build_exec"] == 0
    error_messages = [m for m in manager.sent_messages if m.get("type") == "error"]
    assert error_messages
    assert any(msg.get("message") == "IR mapping failed" for msg in error_messages)
    assert any("radius" in msg.get("details", "") for msg in error_messages)
