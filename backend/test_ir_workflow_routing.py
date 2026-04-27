import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

try:
    from . import agent_workflow
    from .backends.base import TargetExecutionResult
except ImportError:  # pragma: no cover
    from backend.backend import agent_workflow
    from backend.backend.backends.base import TargetExecutionResult


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
