"""Tests for thread spec normalization and LLM guardrails."""

import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from backend.llm_client import TOOLS, extract_tool_calls
from backend.thread_specs import (
    ALL_THREAD_SIZES,
    normalize_thread_size,
    normalize_thread_type,
    validate_thread_spec,
)


def _tool_schema(tool_name: str):
    for tool in TOOLS:
        if tool.get("name") == tool_name:
            return tool["input_schema"]
    raise AssertionError(f"Tool not found: {tool_name}")


def test_normalize_thread_type_aliases():
    assert normalize_thread_type("metric") == "metric"
    assert normalize_thread_type("ISO") == "metric"
    assert normalize_thread_type(" iso-metric ") == "metric"
    assert normalize_thread_type("UNC") == "unc"
    assert normalize_thread_type("unified fine") == "unf"
    assert normalize_thread_type("acme") is None


def test_normalize_thread_size_formats():
    assert normalize_thread_size("m6") == "M6"
    assert normalize_thread_size("M 6 x 1") == "M6"
    assert normalize_thread_size("#8 - 32") == "#8-32"
    assert normalize_thread_size("10-24") == "#10-24"
    assert normalize_thread_size(" 1 / 4 - 20 ") == "1/4-20"


def test_validate_thread_spec_accepts_supported():
    thread_type, thread_size = validate_thread_spec("ISO", "m 6 x 1")
    assert thread_type == "metric"
    assert thread_size == "M6"


def test_validate_thread_spec_rejects_unsupported_metric_size():
    with pytest.raises(ValueError, match="not supported"):
        validate_thread_spec("metric", "M2.5")


def test_validate_thread_spec_rejects_thread_type_size_mismatch():
    with pytest.raises(ValueError, match="belongs to 'unc'"):
        validate_thread_spec("metric", "1/4-20")


def test_thread_schema_uses_explicit_thread_size_enum():
    tapped_schema = _tool_schema("create_tapped_hole")
    external_schema = _tool_schema("create_external_thread")

    tapped_enum = tapped_schema["properties"]["thread_size"]["enum"]
    external_enum = external_schema["properties"]["thread_size"]["enum"]

    assert tapped_enum == list(ALL_THREAD_SIZES)
    assert external_enum == list(ALL_THREAD_SIZES)


def test_extract_tool_calls_rewrites_invalid_thread_specs():
    response = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_bad_thread",
                "name": "create_tapped_hole",
                "input": {
                    "face_ref": "face_0",
                    "center_x": 0.0,
                    "center_y": 0.0,
                    "center_z": 0.0,
                    "thread_type": "metric",
                    "thread_size": "M2.5",
                    "thread_depth": 6.0,
                    "description": "tap hole",
                },
            }
        ]
    }

    calls = extract_tool_calls(response)
    assert len(calls) == 1
    assert calls[0]["id"] == "toolu_bad_thread"
    assert calls[0]["name"] == "respond_to_user"
    assert "not supported" in calls[0]["input"]["message"]


def test_extract_tool_calls_normalizes_supported_thread_specs():
    response = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_good_thread",
                "name": "create_external_thread",
                "input": {
                    "face_ref": "face_1",
                    "thread_type": "ISO",
                    "thread_size": "M 6 x 1",
                    "description": "external thread",
                },
            }
        ]
    }

    calls = extract_tool_calls(response)
    assert len(calls) == 1
    assert calls[0]["name"] == "create_external_thread"
    assert calls[0]["input"]["thread_type"] == "metric"
    assert calls[0]["input"]["thread_size"] == "M6"
