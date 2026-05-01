import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from backend.code_generator import OPERATION_TEMPLATES
from backend.prompt_structure import CLUSTER_TOOL_MAPPING, CORE_INSTRUCTIONS


def test_core_instructions_do_not_claim_list_features_refreshes_entity_refs():
    assert "Pattern after refreshing list_features?" not in CORE_INSTRUCTIONS
    assert "Call `list_features` to refresh, then retry." not in CORE_INSTRUCTIONS
    assert "If Design Entities is empty, call list_features before attempting modifications" not in CORE_INSTRUCTIONS
    assert "If design_entities is empty or missing, HALT and call list_features" not in CORE_INSTRUCTIONS

    assert "Do NOT treat `list_features` as an entity-ref refresh" in CORE_INSTRUCTIONS
    assert "list_features does not refresh entity refs" in CORE_INSTRUCTIONS


def test_delete_feature_template_scopes_list_features_to_feature_snapshot():
    delete_feature_template = OPERATION_TEMPLATES["delete_feature"]

    assert "call list_features to get current tokens." not in delete_feature_template
    assert "Call list_features for current tokens." not in delete_feature_template
    assert "Call list_features to get current state." not in delete_feature_template

    assert "latest feature snapshot/tokens" in delete_feature_template
    assert "does not refresh face/edge/body refs" in delete_feature_template


def test_prompt_uses_consistent_spatial_units_and_hole_schema():
    assert "Spatial Context| mm" in CORE_INSTRUCTIONS
    assert "Design Entities uses WORLD coordinates (x, y, z) in mm" in CORE_INSTRUCTIONS
    assert "Design Entities uses WORLD coordinates (x, y, z) in cm" not in CORE_INSTRUCTIONS

    hole_docs = CLUSTER_TOOL_MAPPING["holes"]["documentation"]
    assert 'extent_type="through_all"' in hole_docs
    assert 'extent_type="distance"' in hole_docs
    assert "through_all=true" not in hole_docs
    assert "through_all=false" not in hole_docs
    assert "create_counterbore_hole (with angle)" not in hole_docs
    assert "Countersink | unsupported as a dedicated tool" in hole_docs


def test_line_loop_prompt_requires_profile_inspection_before_extrude():
    assert "This creates ONE closed loop = ONE profile" not in CORE_INSTRUCTIONS
    assert "call list_sketch_profiles to inspect the closed loop" in CORE_INSTRUCTIONS
    assert "then extrude the returned profile index" in CORE_INSTRUCTIONS
