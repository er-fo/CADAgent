"""Regression tests for production-like router payload variants and failures."""

import asyncio
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from backend import prompt_router


def _fake_client_with_responses(responses):
    class FakeCompletions:
        def __init__(self, queued):
            self._queued = list(queued)
            self.calls = 0

        async def create(self, **kwargs):  # noqa: ARG002
            self.calls += 1
            if not self._queued:
                raise AssertionError("No queued fake responses left")
            return self._queued.pop(0)

    class FakeChat:
        def __init__(self, queued):
            self.completions = FakeCompletions(queued)

    class FakeClient:
        def __init__(self, queued):
            self.chat = FakeChat(queued)

    return FakeClient(responses)


def test_route_request_extracts_json_from_tool_call_arguments(monkeypatch):
    class Message:
        content = None
        reasoning_content = None
        reasoning = None
        tool_calls = [
            {
                "function": {
                    "arguments": "{\"required\": [\"inspection\"], \"optional\": [\"selection\"], \"reasoning\": \"tool_call_args\"}"
                }
            }
        ]

    class Choice:
        message = Message()
        text = None

    class Response:
        choices = [Choice()]
        output_text = None

    fake_client = _fake_client_with_responses([Response()])
    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: fake_client)  # noqa: ARG005

    result = asyncio.run(prompt_router.route_request("inspect and select features"))

    assert "fallback" not in result
    assert "inspection" in result["required"]
    assert "selection" in result["optional"]
    assert result["reasoning"] == "tool_call_args"
    assert result["router_retry_attempted"] is False
    assert result["router_retry_count"] == 0


def test_route_request_extracts_nested_parts_payload(monkeypatch):
    response = {
        "choices": [
            {
                "message": {
                    "content": {
                        "parts": [
                            {
                                "type": "text",
                                "text": "{\"required\": [\"inspection\"], \"optional\": [], \"reasoning\": \"nested_parts\"}",
                            }
                        ]
                    }
                }
            }
        ]
    }

    fake_client = _fake_client_with_responses([response])
    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: fake_client)  # noqa: ARG005

    result = asyncio.run(prompt_router.route_request("inspect the model"))
    assert "inspection" in result["required"]
    assert result["reasoning"] == "nested_parts"
    assert result["router_retry_attempted"] is False
    assert result["router_parse_status"] == "ok"


def test_route_request_uses_second_choice_when_first_is_empty(monkeypatch):
    response = {
        "choices": [
            {"message": {"content": None}},
            {
                "message": {
                    "content": "{\"required\": [\"inspection\"], \"optional\": [], \"reasoning\": \"second_choice\"}"
                }
            },
        ]
    }

    fake_client = _fake_client_with_responses([response])
    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: fake_client)  # noqa: ARG005

    result = asyncio.run(prompt_router.route_request("inspect the model"))
    assert "inspection" in result["required"]
    assert result["reasoning"] == "second_choice"


def test_route_request_retries_once_and_recovers_from_non_json(monkeypatch):
    first = {
        "choices": [
            {
                "message": {
                    "content": "Use core and inspection clusters.",
                }
            }
        ]
    }
    second = {
        "choices": [
            {
                "message": {
                    "content": "{\"required\": [\"inspection\"], \"optional\": [], \"reasoning\": \"repaired_on_retry\"}",
                }
            }
        ]
    }

    fake_client = _fake_client_with_responses([first, second])
    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: fake_client)  # noqa: ARG005

    result = asyncio.run(prompt_router.route_request("inspect geometry"))
    assert "fallback" not in result
    assert "inspection" in result["required"]
    assert result["reasoning"] == "repaired_on_retry"
    assert result["router_retry_attempted"] is True
    assert result["router_retry_count"] == 1
    assert result["router_initial_parse_status"] == "non_json_text"
    assert result["router_parse_status"].startswith("retry_")
    assert result["router_retry_parse_status"] == "ok"
    assert fake_client.chat.completions.calls == 2


def test_route_request_extracts_json_after_wrapped_bracket_noise(monkeypatch):
    response = {
        "choices": [
            {
                "message": {
                    "content": (
                        "MiniMax routing notes [not-json]\n"
                        "```json\n"
                        "{\"required\": [\"inspection\"], \"optional\": [\"selection\"], "
                        "\"reasoning\": \"wrapped_json\"}\n"
                        "```\n"
                        "Use these clusters."
                    ),
                }
            }
        ]
    }

    fake_client = _fake_client_with_responses([response])
    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: fake_client)  # noqa: ARG005

    result = asyncio.run(prompt_router.route_request("inspect geometry"))
    assert "fallback" not in result
    assert "inspection" in result["required"]
    assert "selection" in result["optional"]
    assert result["reasoning"] == "wrapped_json"
    assert result["router_retry_attempted"] is False
    assert result["router_parse_status"] == "ok"
    assert fake_client.chat.completions.calls == 1


def test_route_request_skips_valid_non_router_json_before_router_json(monkeypatch):
    response = {
        "choices": [
            {
                "message": {
                    "content": (
                        "MiniMax scratchpad [1]\n"
                        "{\"required\": [\"inspection\"], \"optional\": [], "
                        "\"reasoning\": \"second_json\"}"
                    ),
                }
            }
        ]
    }

    fake_client = _fake_client_with_responses([response])
    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: fake_client)  # noqa: ARG005

    result = asyncio.run(prompt_router.route_request("inspect geometry"))
    assert "fallback" not in result
    assert "inspection" in result["required"]
    assert result["reasoning"] == "second_json"
    assert result["router_retry_attempted"] is False
    assert fake_client.chat.completions.calls == 1


def test_route_request_repairs_truncated_reasoning_string_without_retry(monkeypatch):
    response = {
        "choices": [
            {
                "message": {
                    "content": (
                        "{\"required\": [\"inspection\"], \"optional\": [], "
                        "\"reasoning\": \"MiniMax stopped mid reasoning"
                    ),
                }
            }
        ]
    }

    fake_client = _fake_client_with_responses([response])
    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: fake_client)  # noqa: ARG005

    result = asyncio.run(prompt_router.route_request("inspect geometry"))
    assert "fallback" not in result
    assert "inspection" in result["required"]
    assert result["reasoning"] == "MiniMax stopped mid reasoning"
    assert result["router_retry_attempted"] is False
    assert result["router_parse_status"] == "repaired"
    assert fake_client.chat.completions.calls == 1


def test_route_request_coerces_labeled_non_json_text(monkeypatch):
    response = {
        "choices": [
            {
                "message": {
                    "content": (
                        "Required clusters: core, inspection\n"
                        "Optional clusters: selection\n"
                        "Reasoning: MiniMax returned labels instead of JSON."
                    ),
                }
            }
        ]
    }

    fake_client = _fake_client_with_responses([response])
    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: fake_client)  # noqa: ARG005

    result = asyncio.run(prompt_router.route_request("inspect geometry"))
    assert "fallback" not in result
    assert "inspection" in result["required"]
    assert "selection" in result["optional"]
    assert result["reasoning"] == "MiniMax returned labels instead of JSON."
    assert result["router_retry_attempted"] is False
    assert result["router_parse_status"] == "coerced_text_labels"
    assert fake_client.chat.completions.calls == 1


def test_route_request_retries_once_then_falls_back_deterministically(monkeypatch):
    first = {"choices": [{"message": {"content": "{\"required\":"}}]}
    second = {"choices": [{"message": {"content": "{\"required\":"}}]}

    fake_client = _fake_client_with_responses([first, second])
    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: fake_client)  # noqa: ARG005
    monkeypatch.setattr(prompt_router, "ROUTER_MODEL", "gpt-4.1-nano")

    result = asyncio.run(prompt_router.route_request("inspect geometry"))
    assert result["fallback"] is True
    assert result["fallback_reason"] == "parse_failure:truncated_json"
    assert result["router_initial_parse_status"] == "truncated_json"
    assert result["router_parse_status"].startswith("retry_failed:")
    assert result["router_retry_attempted"] is True
    assert result["router_retry_count"] == 1
    assert result["router_retry_parse_status"] == "truncated_json"
    assert fake_client.chat.completions.calls == 2


def test_route_request_missing_credentials_populates_router_parse_status(monkeypatch):
    monkeypatch.setattr(prompt_router, "_build_routing_client", lambda api_keys=None: None)  # noqa: ARG005
    monkeypatch.setattr(prompt_router, "ROUTER_MODEL", "gpt-4.1-nano")

    result = asyncio.run(prompt_router.route_request("inspect geometry"))
    assert result["fallback"] is True
    assert result["fallback_reason"] == "missing_router_credentials"
    assert result["router_parse_status"] == "missing_router_credentials"
    assert result["router_retry_attempted"] is False
    assert result["router_retry_count"] == 0
    assert result["router_retry_parse_status"] == "not_attempted"
