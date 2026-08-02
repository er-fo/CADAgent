import pytest

from backend.agent_workflow import (
    SelectionToolCallError,
    _validate_adjust_feature_parameters,
    _validate_feature_identity_tool_input,
)
from backend.llm_client import TOOLS
from backend.prompt_structure import CLUSTER_TOOL_MAPPING


def test_adjust_feature_parameters_accepts_supported_extrude_distance() -> None:
    cleaned = _validate_adjust_feature_parameters(
        {
            "feature_token": "token-1",
            "parameters": {"distance": 12.5, "distance_unit": "mm", "name": "Base extrude"},
            "expected_name": "Extrude1",
            "expected_timeline_index": 3,
            "description": "Make base longer",
        }
    )

    assert cleaned["feature_token"] == "token-1"
    assert cleaned["parameters"]["distance"] == 12.5
    assert cleaned["parameters"]["distance_unit"] == "mm"
    assert cleaned["expected_timeline_index"] == 3


def test_adjust_feature_parameters_accepts_fillet_shell_and_pattern_fields() -> None:
    cleaned = _validate_adjust_feature_parameters(
        {
            "feature_token": "token-7",
            "parameters": {
                "radius": 1.5,
                "radius_unit": "mm",
                "inside_thickness": 2.0,
                "inside_thickness_unit": "cm",
                "rectangular_count_one": 4.0,
                "rectangular_spacing_one": 12,
                "rectangular_spacing_one_unit": "mm",
                "circular_count": 6,
                "circular_total_angle": 180,
                "circular_total_angle_unit": "deg",
            },
            "description": "Exercise widened parameter validation",
        }
    )

    assert cleaned["parameters"]["radius"] == 1.5
    assert cleaned["parameters"]["inside_thickness_unit"] == "cm"
    assert cleaned["parameters"]["rectangular_count_one"] == 4
    assert cleaned["parameters"]["circular_count"] == 6
    assert cleaned["parameters"]["circular_total_angle_unit"] == "deg"


def test_adjust_feature_parameters_rejects_unsupported_parameter() -> None:
    with pytest.raises(SelectionToolCallError, match="does not support"):
        _validate_adjust_feature_parameters(
            {
                "feature_token": "token-1",
                "parameters": {"script": "danger"},
                "description": "Unsafe edit",
            }
        )


def test_adjust_feature_parameters_requires_positive_numeric_values() -> None:
    with pytest.raises(SelectionToolCallError, match="positive number"):
        _validate_adjust_feature_parameters(
            {
                "feature_token": "token-1",
                "parameters": {"diameter": 0},
                "description": "Invalid diameter",
            }
        )


def test_adjust_feature_parameters_rejects_invalid_count_and_angle_fields() -> None:
    invalid_payloads = [
        ({"rectangular_count_one": 2.5}, "positive integer"),
        ({"circular_count": 0}, "positive integer"),
        ({"circular_total_angle": 0}, "finite non-zero"),
        ({"circular_total_angle": 45, "circular_total_angle_unit": "turn"}, "one of"),
        ({"radius": 2, "radius_unit": "ft"}, "one of"),
    ]

    for params, expected in invalid_payloads:
        with pytest.raises(SelectionToolCallError, match=expected):
            _validate_adjust_feature_parameters(
                {
                    "feature_token": "token-1",
                    "parameters": params,
                    "description": "Invalid widened parameter",
                }
            )


def test_suppress_feature_identity_validation_accepts_guardrails() -> None:
    cleaned = _validate_feature_identity_tool_input(
        "suppress_feature",
        {
            "feature_token": "token-2",
            "expected_name": "Shell1",
            "expected_timeline_index": "7",
            "description": "Temporarily disable shell",
        },
    )

    assert cleaned == {
        "feature_token": "token-2",
        "expected_name": "Shell1",
        "expected_timeline_index": 7,
    }


def test_unsuppress_feature_identity_validation_rejects_extra_parameters() -> None:
    with pytest.raises(SelectionToolCallError, match="unexpected parameter"):
        _validate_feature_identity_tool_input(
            "unsuppress_feature",
            {
                "feature_token": "token-2",
                "force": True,
                "description": "Restore shell",
            },
        )


def test_feature_identity_validation_rejects_bool_or_negative_expected_index() -> None:
    for bad_index in (True, -1):
        with pytest.raises(SelectionToolCallError, match="expected_timeline_index"):
            _validate_feature_identity_tool_input(
                "suppress_feature",
                {
                    "feature_token": "token-2",
                    "expected_timeline_index": bad_index,
                    "description": "Invalid guardrail",
                },
            )


def test_suppress_tools_are_available_in_timeline_cluster() -> None:
    tool_names = {tool["name"] for tool in TOOLS}
    timeline_tools = set(CLUSTER_TOOL_MAPPING["timeline"]["tools"])

    assert {"suppress_feature", "unsuppress_feature"} <= tool_names
    assert {"suppress_feature", "unsuppress_feature"} <= timeline_tools
