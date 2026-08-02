"""Regression tests for failure-intent retry guards and extrude diagnostics."""

from backend.agent_workflow import (
    _build_failure_intent_key,
    _summarise_execution_result,
)
from backend.code_generator import translate_tool_call

def _cut_extrude_input(distance: float, operation: str = "Cut"):
    return {
        "sketch_id": "sketch_0",
        "profile_index": 0,
        "distance": distance,
        "operation": operation,
    }


def test_failure_intent_key_collapses_cut_sign_flip_for_no_intersection():
    key_pos = _build_failure_intent_key(
        "extrude_profile",
        _cut_extrude_input(12.0, "Cut"),
        "No target body found to cut or intersect",
    )
    key_neg = _build_failure_intent_key(
        "extrude_profile",
        _cut_extrude_input(-12.0, "cut"),
        "extrude_profile failure_mode=no_intersection_after_direction_retry",
    )
    assert key_pos == key_neg


def test_failure_intent_key_keeps_cut_sign_flip_distinct_for_other_failures():
    key_pos = _build_failure_intent_key(
        "extrude_profile",
        _cut_extrude_input(12.0),
        "Distance extent must be non-zero",
    )
    key_neg = _build_failure_intent_key(
        "extrude_profile",
        _cut_extrude_input(-12.0),
        "Distance extent must be non-zero",
    )
    assert key_pos != key_neg


def test_failure_intent_key_does_not_collapse_non_cut_ops():
    key_pos = _build_failure_intent_key(
        "extrude_profile",
        _cut_extrude_input(8.0, "Join"),
        "No target body found to cut or intersect",
    )
    key_neg = _build_failure_intent_key(
        "extrude_profile",
        _cut_extrude_input(-8.0, "Join"),
        "No target body found to cut or intersect",
    )
    assert key_pos != key_neg


def test_no_intersection_marker_still_gets_guidance_hint():
    success, message = _summarise_execution_result(
        "extrude_profile",
        {
            "success": False,
            "error": (
                "extrude_profile failure_mode=no_intersection_after_direction_retry; "
                "operation=Cut; sketch_id=sketch_0"
            ),
        },
    )
    assert success is False
    assert "Hint: This usually means your sketch/profile does not intersect any solid" in message


def test_codegen_extrude_cut_has_explicit_no_intersection_diagnostics():
    code = translate_tool_call(
        "extrude_profile",
        {
            "sketch_id": "sketch_0",
            "distance": 10,
            "operation": "Cut",
        },
    )
    assert "failure_mode=no_intersection_after_direction_retry" in code
    assert "cut/intersect retry failed after direction flip" in code
    assert "action=Reposition sketch/profile to intersect solid" in code
