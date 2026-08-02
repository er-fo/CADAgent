"""Unit coverage for CADAgent-managed Bedrock model routing."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from backend import llm_client


def test_normalize_model_name_preserves_managed_bedrock_models():
    assert llm_client.normalize_model_name("minimax.minimax-m2.5") == llm_client.MODEL_MINIMAX_M25
    assert llm_client.normalize_model_name("moonshotai.kimi-k2.5") == llm_client.MODEL_KIMI_K25
    assert llm_client.normalize_model_name("Kimi K2.5") == llm_client.MODEL_KIMI_K25


def test_gateway_provider_for_managed_bedrock_models():
    assert llm_client._gateway_provider_for_model(llm_client.MODEL_MINIMAX_M25) == "bedrock"
    assert llm_client._gateway_provider_for_model(llm_client.MODEL_KIMI_K25) == "bedrock"
    assert llm_client._gateway_provider_for_model(llm_client.MODEL_GPT_5) == "openai"
    assert llm_client._gateway_provider_for_model(llm_client.MODEL_CLAUDE_SONNET_45) == "anthropic"


def test_managed_bedrock_client_uses_bedrock_openai_compatible_env(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    llm_client._openai_compatible_clients.clear()
    monkeypatch.setattr(llm_client, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-token")
    monkeypatch.setenv("BEDROCK_OPENAI_BASE_URL", "https://bedrock-mantle.example/v1")

    client = llm_client._get_managed_bedrock_client()

    assert client is not None
    assert captured == {
        "api_key": "bedrock-token",
        "base_url": "https://bedrock-mantle.example/v1",
    }


def test_managed_bedrock_chat_converter_rejects_images():
    with pytest.raises(ValueError, match="Image attachments are not supported"):
        llm_client._convert_messages_to_openai_chat_format(
            "system",
            [{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {"media_type": "image/png", "data": "abc"},
                }],
            }],
        )


def test_openai_chat_converter_keeps_text_and_tool_calls_in_one_assistant_message():
    converted = llm_client._convert_messages_to_openai_chat_format(
        "system",
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll create the sketch."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "create_sketch",
                        "input": {"plane": "XY"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [{"type": "text", "text": "ok"}],
                    },
                ],
            },
        ],
    )

    assert converted[0] == {"role": "system", "content": "system"}
    assert converted[1]["role"] == "assistant"
    assert converted[1]["content"] == "I'll create the sketch."
    assert converted[1]["tool_calls"][0]["id"] == "call_1"
    assert converted[2] == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}


def test_extract_openai_chat_reasoning_from_minimax_fields():
    class Message:
        reasoning = "primary reasoning"
        reasoning_content = None
        thinking = ""
        reasoning_details = [{"text": "detail reasoning"}]

    assert llm_client._extract_openai_chat_reasoning(Message()) == "primary reasoning\n\ndetail reasoning"


def test_openai_chat_converter_preserves_length_finish_reason_with_text():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content="Partial answer", tool_calls=[]),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )

    converted = llm_client._convert_openai_chat_response_to_anthropic_format(response)

    assert converted["stop_reason"] == "max_tokens"
    assert converted["content"] == [{"type": "text", "text": "Partial answer"}]


def test_openai_chat_converter_preserves_length_finish_reason_with_tool_calls():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="create_sketch", arguments='{"plane_id":"XY"}'),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )

    converted = llm_client._convert_openai_chat_response_to_anthropic_format(response)

    assert converted["stop_reason"] == "max_tokens"
    assert converted["content"][0]["type"] == "tool_use"
    assert converted["content"][0]["name"] == "create_sketch"


def test_openai_response_converter_preserves_incomplete_status_with_tool_calls():
    response = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=[
            SimpleNamespace(
                type="function_call",
                name="create_sketch",
                call_id="call_1",
                arguments='{"plane_id":"XY"}',
            )
        ],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )

    converted = llm_client._convert_openai_response_to_anthropic_format(response)

    assert converted["stop_reason"] == "max_tokens"
    assert converted["content"][0]["type"] == "tool_use"
    assert converted["content"][0]["name"] == "create_sketch"


def test_openai_response_converter_preserves_incomplete_status_with_text():
    response = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text="Partial answer"),
                ],
            )
        ],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )

    converted = llm_client._convert_openai_response_to_anthropic_format(response)

    assert converted["stop_reason"] == "max_tokens"
    assert converted["content"] == [{"type": "text", "text": "Partial answer"}]
