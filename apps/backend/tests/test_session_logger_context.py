"""Tests for session context extraction metadata and flattened counts."""

from backend.session_logger import _extract_session_context

def test_extract_session_context_includes_capture_phase_and_state_counts():
    context = _extract_session_context(
        user_request="Help me build a case",
        timeline_state=None,
        messages=[],
        feature_snapshot={
            "success": True,
            "features": [
                {"entity_token": "f0", "name": "Extrude1", "type": "extrude"},
                {"entity_token": "f1", "name": "Fillet1", "type": "fillet"},
            ],
        },
        entity_store_data={
            "selected_bodies": [{"entity_ref": "body_0"}],
            "selected_faces": [{"entity_ref": "face_0"}, {"entity_ref": "face_1"}],
            "selected_edges": [{"entity_ref": "e0"}],
        },
        routing_result=None,
        loaded_tools=None,
        model_name="gpt-5",
        reasoning_effort="high",
        iteration=2,
        max_iterations=30,
        capture_phase="pre_llm",
        system_prompt=None,
        tools=None,
        spatial_context=None,
        entity_context={
            "bodies": [{"entity_token": "b0"}],
            "faces": [{"entity_token": "fa0"}, {"entity_token": "fa1"}],
            "edges": [{"entity_token": "e0"}],
            "vertices": [{"entity_token": "v0"}],
        },
    )

    assert context["metadata"]["capture_phase"] == "pre_llm"
    counts = context["state_counts"]
    assert counts["feature_count"] == 2
    assert counts["selected_body_count"] == 1
    assert counts["selected_face_count"] == 2
    assert counts["selected_edge_count"] == 1
    assert counts["entity_context_body_count"] == 1
    assert counts["entity_context_face_count"] == 2
    assert counts["entity_context_edge_count"] == 1
    assert counts["entity_context_vertex_count"] == 1


def test_extract_session_context_includes_router_telemetry_fields():
    context = _extract_session_context(
        user_request="Route this request",
        timeline_state=None,
        messages=[],
        feature_snapshot={"success": True, "features": []},
        entity_store_data={"selected_bodies": [], "selected_faces": [], "selected_edges": []},
        routing_result={
            "required": ["core", "inspection"],
            "optional": ["selection"],
            "reasoning": "Router decision",
            "confidence": "medium",
            "routing_source": "llm_router",
            "fallback_reason": None,
            "router_provider": "bedrock",
            "router_model": "minimax.minimax-m2.5",
            "router_api_key_source": "session_byok",
            "router_parse_status": "coerced_shape",
        },
        loaded_tools=None,
        model_name="claude-sonnet-4.6",
        reasoning_effort="low",
        iteration=1,
        max_iterations=30,
        capture_phase="pre_llm",
        system_prompt=None,
        tools=None,
        spatial_context=None,
        entity_context={"bodies": [], "faces": [], "edges": [], "vertices": []},
    )

    routing = context["routing"]
    assert routing["required_clusters"] == ["core", "inspection"]
    assert routing["optional_clusters"] == ["selection"]
    assert routing["routing_source"] == "llm_router"
    assert routing["router_provider"] == "bedrock"
    assert routing["router_model"] == "minimax.minimax-m2.5"
    assert routing["router_api_key_source"] == "session_byok"
    assert routing["router_parse_status"] == "coerced_shape"
