"""Integration/regression matrix for fail-closed IR execution gating."""

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

    def get_user_token(self, session_id: str) -> Optional[str]:
        return "token"

    def get_llm_api_keys(self, session_id: str) -> Dict[str, str]:
        return {}

    def get_reasoning_context(self, session_id: str) -> _FakeReasoningContext:
        return self.reasoning_context

    def get_active_build_plan(self, session_id: str) -> Optional[Dict[str, Any]]:
        return None

    async def send_message(self, session_id: str, payload: Dict[str, Any]) -> None:
        self.sent_messages.append(payload)

    def set_conversation_history(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self.history = list(messages)


def _tool_use_response() -> Dict[str, Any]:
    return {
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "Proceeding with case geometry."},
            {"type": "tool_use", "id": "toolu_1", "name": "create_sketch", "input": {"plane_id": "XY", "sketch_id": "s0"}},
            {"type": "tool_use", "id": "toolu_2", "name": "add_circle", "input": {"sketch_id": "s0", "center_u": 0, "center_v": 0, "radius": 3}},
            {"type": "tool_use", "id": "toolu_3", "name": "extrude_profile", "input": {"sketch_id": "s0", "profile_index": 0, "distance": 8}},
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


@pytest.mark.parametrize(
    "failed_operation_type, expected_executed_ops, expected_blocked_count",
    [
        ("create_sketch", ["create_sketch"], 2),
        ("add_circle", ["create_sketch", "add_circle"], 1),
        (None, ["create_sketch", "add_circle", "extrude"], 0),
    ],
)
def test_ir_execution_gating_matrix(
    monkeypatch: pytest.MonkeyPatch,
    failed_operation_type: Optional[str],
    expected_executed_ops: List[str],
    expected_blocked_count: int,
):
    manager = _FakeManager()
    call_count = {"llm": 0}
    executed_ops: List[str] = []

    async def fake_call_claude_with_tools(*args, **kwargs):
        call_count["llm"] += 1
        return _tool_use_response() if call_count["llm"] == 1 else _end_turn_response()

    async def fake_execute_operation(self, session_id, operation, tool_use_id, description=""):
        executed_ops.append(operation.type)
        if failed_operation_type and operation.type == failed_operation_type:
            return TargetExecutionResult(
                success=False,
                target="fusion",
                message=f"{operation.type} failed intentionally",
                raw_result={"success": False, "tool_use_id": tool_use_id, "error": f"{operation.type} failed intentionally"},
            )
        return TargetExecutionResult(
            success=True,
            target="fusion",
            message=f"{operation.type} ok",
            raw_result={"success": True, "tool_use_id": tool_use_id, "message": f"{operation.type} ok"},
        )

    async def fake_refresh_after_success(session_id, manager, tool_name, result, messages):
        return None

    monkeypatch.setattr(agent_workflow, "USE_PROMPT_ROUTING", False)
    monkeypatch.setattr(agent_workflow, "call_claude_with_tools", fake_call_claude_with_tools)
    monkeypatch.setattr(agent_workflow.FusionTargetExecutor, "execute_operation", fake_execute_operation)
    monkeypatch.setattr(agent_workflow, "_refresh_and_enrich_after_success", fake_refresh_after_success)

    asyncio.run(
        agent_workflow._execute_workflow_loop(
            session_id="s-gating-matrix",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Model the standoff"}]}],
            max_iterations=4,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "gating-matrix"},
            feature_snapshot=None,
        )
    )

    assert executed_ops == expected_executed_ops

    error_messages = [msg for msg in manager.sent_messages if msg.get("type") == "error"]
    blocked_messages = [msg for msg in error_messages if msg.get("message") == "IR dependency blocked"]
    assert len(blocked_messages) == expected_blocked_count

    if failed_operation_type:
        assert any(msg.get("message") == "fusion execution failed" for msg in error_messages)
    else:
        assert not error_messages

    tool_result_texts = _extract_tool_result_texts(manager.history)
    if expected_blocked_count > 0:
        assert any("depends on uncommitted operation" in text for text in tool_result_texts)
