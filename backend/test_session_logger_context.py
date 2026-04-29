"""Tests for session context extraction metadata and flattened counts."""

try:
    from .session_logger import _extract_session_context
except ImportError:  # pragma: no cover
    from backend.backend.session_logger import _extract_session_context


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
