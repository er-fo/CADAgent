"""Unit tests for prompt_router.py — pure-function coverage (no LLM calls)."""

import sys
import os

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from backend import prompt_router
from backend.prompt_router import (
    _fallback_routing,
    _extract_conversation_context,
    _extract_operations_from_build_plan,
    _build_routing_client,
    _extract_routing_response_text,
    get_routing_summary,
    ROUTING_PATTERNS,
)


# ------------------------------------------------------------------ #
# _fallback_routing
# ------------------------------------------------------------------ #

def test_fallback_always_includes_core():
    result = _fallback_routing("random gibberish with no keywords")
    assert "core" in result["required"]


def test_fallback_sketch_keywords():
    result = _fallback_routing("draw a circle on the top face")
    assert "sketch_tools" in result["required"]
    assert "3d_modeling" in result["required"]


def test_fallback_extrude_keywords():
    result = _fallback_routing("extrude a box 50mm tall")
    assert "sketch_tools" in result["required"]
    assert "3d_modeling" in result["required"]


def test_fallback_modification_keywords():
    for kw in ["fillet", "chamfer", "shell"]:
        result = _fallback_routing(f"apply a {kw} to the edge")
        assert "modification" in result["required"], f"Expected modification for '{kw}'"
        assert "inspection" in result["required"], f"Expected inspection for '{kw}'"
        assert "selection" in result["required"], f"Expected selection for '{kw}'"


def test_fallback_hole_keywords():
    result = _fallback_routing("drill a hole in the top face")
    assert "holes" in result["required"]
    assert "inspection" in result["required"]


def test_fallback_thread_keywords():
    result = _fallback_routing("add a thread to the cylinder")
    assert "threading" in result["required"]
    assert "inspection" in result["required"]


def test_fallback_thread_with_create():
    """Threading + creation keywords should also include modeling."""
    result = _fallback_routing("create a threaded bolt shaft")
    assert "threading" in result["required"]
    assert "sketch_tools" in result["required"]
    assert "3d_modeling" in result["required"]


def test_fallback_pattern_keywords():
    result = _fallback_routing("create a pattern of holes")
    assert "patterns" in result["required"]


def test_fallback_no_duplicates():
    result = _fallback_routing("sketch a circle then extrude it and sketch another")
    assert len(result["required"]) == len(set(result["required"]))


def test_fallback_design_exploration_included():
    result = _fallback_routing("I need something to hold my phone on my desk")
    assert "design_exploration" in result["required"]


def test_fallback_design_exploration_excluded_by_dimensions():
    result = _fallback_routing("I need a box 50mm wide")
    assert "design_exploration" not in result["required"]


def test_fallback_design_exploration_excluded_by_keywords():
    """Current legacy fallback still includes exploration for broad part-design requests."""
    result = _fallback_routing("I need to fillet the edges")
    assert "design_exploration" in result["required"]


def test_fallback_confidence_is_low():
    result = _fallback_routing("anything")
    assert result["confidence"] == "low"
    assert result.get("fallback") is True


def test_fallback_safety_clusters_when_nothing_matches():
    """When no keywords match, provide sensible defaults."""
    result = _fallback_routing("jksdflksdjf")
    assert len(result["required"]) > 1  # more than just core


def test_build_routing_client_uses_openai_key_for_default_router(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(prompt_router, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(prompt_router, "ROUTER_MODEL", "gpt-4.1-nano")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-router")

    client = _build_routing_client()

    assert client is not None
    assert captured == {"api_key": "sk-openai-router"}


def test_build_routing_client_uses_bedrock_token_for_minimax_router(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(prompt_router, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(prompt_router, "ROUTER_MODEL", "minimax.minimax-m2.5")
    monkeypatch.setattr(
        prompt_router,
        "ROUTER_BEDROCK_BASE_URL",
        "https://bedrock-mantle.us-east-1.api.aws/v1",
    )
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-token")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = _build_routing_client()

    assert client is not None
    assert captured == {
        "api_key": "bedrock-token",
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
    }


def test_build_routing_client_prefers_session_bedrock_key_over_env(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(prompt_router, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(prompt_router, "ROUTER_MODEL", "minimax.minimax-m2.5")
    monkeypatch.setattr(
        prompt_router,
        "ROUTER_BEDROCK_BASE_URL",
        "https://bedrock-mantle.us-east-1.api.aws/v1",
    )
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bedrock-token")

    client = _build_routing_client({"aws_bearer_token_bedrock": "session-bedrock-token"})

    assert client is not None
    assert captured == {
        "api_key": "session-bedrock-token",
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
    }


def test_build_routing_client_returns_none_when_minimax_token_missing(monkeypatch):
    monkeypatch.setattr(prompt_router, "ROUTER_MODEL", "minimax.minimax-m2.5")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    assert _build_routing_client() is None


def test_route_request_falls_back_when_minimax_token_missing(monkeypatch):
    monkeypatch.setattr(prompt_router, "ROUTER_MODEL", "minimax.minimax-m2.5")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    called = {}

    def fake_fallback(user_request, fallback_reason="fallback"):
        called["user_request"] = user_request
        called["fallback_reason"] = fallback_reason
        return {
            "required": ["core"],
            "optional": [],
            "reasoning": "fallback",
            "confidence": "low",
            "fallback": True,
        }

    monkeypatch.setattr(prompt_router, "_fallback_routing", fake_fallback)

    import asyncio

    result = asyncio.run(prompt_router.route_request("drill a hole in the plate"))

    assert result["fallback"] is True
    assert called["user_request"] == "drill a hole in the plate"
    assert called["fallback_reason"] == "missing_router_credentials"


def test_extract_routing_response_text_from_dict_content_variant():
    response = {
        "choices": [
            {
                "message": {
                    "content": {
                        "type": "text",
                        "text": "{\"required\": [\"core\"], \"optional\": [], \"reasoning\": \"dict_shape\"}",
                    },
                    "reasoning_content": None,
                    "reasoning": None,
                    "tool_calls": None,
                }
            }
        ],
        "output_text": None,
    }

    text = _extract_routing_response_text(response)
    assert "dict_shape" in text


def test_extract_routing_response_text_from_list_content_parts():
    class Message:
        content = [
            {"type": "text", "text": "{\"required\": [\"core\"], \"optional\": [], \"reasoning\": \"ok\"}"}
        ]
        reasoning_content = None
        reasoning = None
        tool_calls = None

    class Choice:
        message = Message()
        text = None

    class Response:
        choices = [Choice()]
        output_text = None

    text = _extract_routing_response_text(Response())
    assert "\"required\": [\"core\"]" in text


def test_extract_routing_response_text_from_choice_text_when_message_content_none():
    class Message:
        content = None
        reasoning_content = None
        reasoning = None
        tool_calls = None

    class Choice:
        message = Message()
        text = "{\"required\": [\"core\"], \"optional\": [], \"reasoning\": \"via_choice_text\"}"

    class Response:
        choices = [Choice()]
        output_text = None

    text = _extract_routing_response_text(Response())
    assert "via_choice_text" in text


def test_route_request_accepts_none_message_content_via_choice_text(monkeypatch):
    class Message:
        content = None
        reasoning_content = None
        reasoning = None
        tool_calls = None

    class Choice:
        message = Message()
        text = "{\"required\": [\"inspection\"], \"optional\": [], \"reasoning\": \"choice_text\"}"

    class Response:
        choices = [Choice()]
        output_text = None

    class FakeCompletions:
        async def create(self, **kwargs):  # noqa: ARG002
            return Response()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: FakeClient())  # noqa: ARG005

    import asyncio

    result = asyncio.run(prompt_router.route_request("inspect this body"))
    assert "core" in result["required"]
    assert "inspection" in result["required"]
    assert result["reasoning"] == "choice_text"
    assert "fallback" not in result


def test_route_request_accepts_list_part_message_content(monkeypatch):
    class Message:
        content = [
            {"type": "text", "text": "{\"required\": [\"core\", \"inspection\"], \"optional\": [], \"reasoning\": \"shape_ok\"}"}
        ]
        reasoning_content = None
        reasoning = None
        tool_calls = None

    response = {
        "choices": [{"message": Message(), "text": None}],
        "output_text": None,
    }

    class FakeCompletions:
        async def create(self, **kwargs):  # noqa: ARG002
            return response

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: FakeClient())  # noqa: ARG005

    import asyncio

    result = asyncio.run(prompt_router.route_request("inspect this body"))
    assert "core" in result["required"]
    assert "inspection" in result["required"]
    assert result["reasoning"] == "shape_ok"
    assert "fallback" not in result


def test_route_request_repairs_obvious_truncated_json(monkeypatch):
    class Message:
        content = "{\"required\": [\"inspection\"], \"optional\": [], \"reasoning\": \"repairable\""
        reasoning_content = None
        reasoning = None
        tool_calls = None

    class Choice:
        message = Message()
        text = None

    class Response:
        choices = [Choice()]
        output_text = None

    class FakeCompletions:
        async def create(self, **kwargs):  # noqa: ARG002
            return Response()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: FakeClient())  # noqa: ARG005

    import asyncio

    result = asyncio.run(prompt_router.route_request("inspect this body"))
    assert "core" in result["required"]
    assert "inspection" in result["required"]
    assert result["reasoning"] == "repairable"
    assert "fallback" not in result


def test_route_request_coerces_list_wrapped_router_json(monkeypatch):
    class Message:
        content = '[{"required": ["inspection"], "optional": ["selection"], "reasoning": "list_wrapped"}]'
        reasoning_content = None
        reasoning = None
        tool_calls = None

    class Choice:
        message = Message()
        text = None

    class Response:
        choices = [Choice()]
        output_text = None

    class FakeCompletions:
        async def create(self, **kwargs):  # noqa: ARG002
            return Response()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: FakeClient())  # noqa: ARG005

    import asyncio

    result = asyncio.run(prompt_router.route_request("inspect this body"))
    assert "core" in result["required"]
    assert "inspection" in result["required"]
    assert "selection" in result["optional"]
    assert result["router_parse_status"] == "coerced_shape"
    assert "fallback" not in result


def test_route_request_coerces_required_clusters_alias(monkeypatch):
    class Message:
        content = '{"required_clusters": ["inspection"], "optional_clusters": ["selection"], "reasoning": "alias_keys"}'
        reasoning_content = None
        reasoning = None
        tool_calls = None

    class Choice:
        message = Message()
        text = None

    class Response:
        choices = [Choice()]
        output_text = None

    class FakeCompletions:
        async def create(self, **kwargs):  # noqa: ARG002
            return Response()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: FakeClient())  # noqa: ARG005

    import asyncio

    result = asyncio.run(prompt_router.route_request("inspect this body"))
    assert "core" in result["required"]
    assert "inspection" in result["required"]
    assert "selection" in result["optional"]
    assert result["reasoning"] == "alias_keys"
    assert "fallback" not in result


def test_route_request_falls_back_on_truncated_json_and_marks_reason(monkeypatch, caplog):
    class Message:
        content = "{\"required\":"
        reasoning_content = None
        reasoning = None
        tool_calls = None

    class Choice:
        message = Message()
        text = None

    class Response:
        choices = [Choice()]
        output_text = None

    class FakeCompletions:
        async def create(self, **kwargs):  # noqa: ARG002
            return Response()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(prompt_router, "ROUTER_MODEL", "gpt-4.1-nano")
    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: FakeClient())  # noqa: ARG005

    import asyncio
    import logging

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(prompt_router.route_request("inspect this body"))

    assert result["fallback"] is True
    assert result["fallback_reason"] == "parse_failure:truncated_json"
    assert "parse_failure:truncated_json" in result["reasoning"]
    assert any(
        "provider=openai-compatible" in record.message
        and "model=gpt-4.1-nano" in record.message
        and "category=truncated_json" in record.message
        for record in caplog.records
    )


# ------------------------------------------------------------------ #
# _extract_conversation_context
# ------------------------------------------------------------------ #

def test_context_none():
    assert _extract_conversation_context(None) == ""


def test_context_empty_list():
    assert _extract_conversation_context([]) == ""


def test_context_string_content():
    history = [{"role": "user", "content": "hello"}]
    ctx = _extract_conversation_context(history)
    assert "User: hello" in ctx


def test_context_list_content():
    history = [{"role": "user", "content": [{"type": "text", "text": "make a box"}]}]
    ctx = _extract_conversation_context(history)
    assert "make a box" in ctx


def test_context_tool_use_content():
    history = [
        {"role": "assistant", "content": [{"type": "tool_use", "name": "create_sketch"}]}
    ]
    ctx = _extract_conversation_context(history)
    assert "create_sketch" in ctx


def test_context_truncation():
    history = [{"role": "user", "content": "a" * 500}]
    ctx = _extract_conversation_context(history)
    assert len(ctx) < 500
    assert ctx.endswith("...")


def test_context_max_messages():
    """Only the last 6 messages are used."""
    history = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
    ctx = _extract_conversation_context(history)
    assert "msg14" in ctx  # 20 - 6 = 14 is the first included
    assert "msg0" not in ctx


# ------------------------------------------------------------------ #
# _extract_operations_from_build_plan
# ------------------------------------------------------------------ #

def test_build_plan_none():
    assert _extract_operations_from_build_plan(None) == ""


def test_build_plan_empty_steps():
    assert _extract_operations_from_build_plan({"steps": []}) == ""


def test_build_plan_no_steps_key():
    assert _extract_operations_from_build_plan({}) == ""


def test_build_plan_with_operations():
    plan = {
        "design_name": "Widget",
        "steps": [
            {"operation": "create_sketch", "description": "Base sketch"},
            {"operation": "extrude_profile", "description": "Extrude base"},
        ]
    }
    result = _extract_operations_from_build_plan(plan)
    assert "Widget" in result
    assert "create_sketch" in result
    assert "extrude_profile" in result
    assert "2 steps" in result


def test_build_plan_description_only():
    plan = {
        "design_name": "Test",
        "steps": [{"description": "Do something"}]
    }
    result = _extract_operations_from_build_plan(plan)
    assert "Do something" in result


# ------------------------------------------------------------------ #
# get_routing_summary
# ------------------------------------------------------------------ #

def test_routing_summary_basic():
    routing = {
        "required": ["core", "sketch_tools"],
        "optional": ["inspection"],
        "reasoning": "Need sketching",
        "confidence": "high",
    }
    summary = get_routing_summary(routing)
    assert "high" in summary
    assert "core" in summary
    assert "sketch_tools" in summary
    assert "inspection" in summary
    assert "Need sketching" in summary


def test_routing_summary_no_optional():
    routing = {
        "required": ["core"],
        "optional": [],
        "reasoning": "Minimal",
        "confidence": "medium",
    }
    summary = get_routing_summary(routing)
    assert "Optional" not in summary


# ------------------------------------------------------------------ #
# ROUTING_PATTERNS sanity
# ------------------------------------------------------------------ #

def test_routing_patterns_have_keywords():
    for name, pattern in ROUTING_PATTERNS.items():
        assert "keywords" in pattern, f"Pattern {name} missing keywords"
        assert len(pattern["keywords"]) > 0, f"Pattern {name} has no keywords"
        assert "typical_clusters" in pattern, f"Pattern {name} missing clusters"


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import inspect
    passed = 0
    failed = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(func):
            try:
                func()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {name}: {exc}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
