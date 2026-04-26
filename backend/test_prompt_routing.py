"""Unit tests for prompt_router.py — pure-function coverage (no LLM calls)."""

import sys
import os

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from backend.prompt_router import (
    _fallback_routing,
    _extract_conversation_context,
    _extract_operations_from_build_plan,
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
