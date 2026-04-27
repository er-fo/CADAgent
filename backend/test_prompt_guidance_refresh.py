import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from backend.code_generator import OPERATION_TEMPLATES
from backend.prompt_structure import CORE_INSTRUCTIONS


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
