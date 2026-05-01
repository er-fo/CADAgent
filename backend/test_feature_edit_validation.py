import pytest

from backend.agent_workflow import SelectionToolCallError, _validate_adjust_feature_parameters


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
