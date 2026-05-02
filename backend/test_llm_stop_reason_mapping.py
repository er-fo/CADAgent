"""Provider adapter stop-reason safety coverage."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from backend import llm_client


def test_openai_chat_missing_finish_reason_is_unsafe_for_text():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=None,
                message=SimpleNamespace(content="Looks complete but finish reason is missing.", tool_calls=[]),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )

    converted = llm_client._convert_openai_chat_response_to_anthropic_format(response)

    assert converted["stop_reason"] == "unknown"
    assert converted["content"][0]["type"] == "text"


def test_openai_chat_missing_finish_reason_is_unsafe_for_tool_calls():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=None,
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
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )

    converted = llm_client._convert_openai_chat_response_to_anthropic_format(response)

    assert converted["stop_reason"] == "unknown"
    assert converted["content"][0]["type"] == "tool_use"


def test_openai_chat_stop_finish_reason_is_unsafe_with_tool_calls():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
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
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )

    converted = llm_client._convert_openai_chat_response_to_anthropic_format(response)

    assert converted["stop_reason"] == "stop_with_tool_calls"
    assert converted["content"][0]["type"] == "tool_use"


def test_openai_responses_nonterminal_status_is_unsafe():
    response = SimpleNamespace(
        status="in_progress",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Partial answer")],
            )
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )

    converted = llm_client._convert_openai_response_to_anthropic_format(response)

    assert converted["stop_reason"] == "in_progress"
    assert converted["content"][0]["type"] == "text"


def test_openai_responses_completed_status_is_safe_for_tool_calls():
    response = SimpleNamespace(
        status="completed",
        output=[
            SimpleNamespace(
                type="function_call",
                name="create_sketch",
                call_id="call_1",
                arguments='{"plane_id":"XY"}',
            )
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )

    converted = llm_client._convert_openai_response_to_anthropic_format(response)

    assert converted["stop_reason"] == "tool_use"
    assert converted["content"][0]["type"] == "tool_use"


def test_gemini_max_tokens_finish_reason_is_unsafe_for_text():
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="MAX_TOKENS",
                content=SimpleNamespace(parts=[SimpleNamespace(text="Partial answer")]),
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=1, response_token_count=1),
    )

    converted = llm_client._convert_gemini_response_to_anthropic_format(response)

    assert converted["stop_reason"] == "max_tokens"
    assert converted["content"][0]["type"] == "text"


def test_gemini_safety_finish_reason_is_unsafe_for_tool_calls():
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="SAFETY",
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(
                                id="call_1",
                                name="create_sketch",
                                args={"plane_id": "XY"},
                            )
                        )
                    ]
                ),
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=1, response_token_count=1),
    )

    converted = llm_client._convert_gemini_response_to_anthropic_format(response)

    assert converted["stop_reason"] == "safety"
    assert converted["content"][0]["type"] == "tool_use"


def test_gemini_stop_finish_reason_is_safe_for_tool_calls():
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(
                                id="call_1",
                                name="create_sketch",
                                args={"plane_id": "XY"},
                            )
                        )
                    ]
                ),
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=1, response_token_count=1),
    )

    converted = llm_client._convert_gemini_response_to_anthropic_format(response)

    assert converted["stop_reason"] == "tool_use"
    assert converted["content"][0]["type"] == "tool_use"
