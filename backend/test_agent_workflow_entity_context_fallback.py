"""Regression tests for flat fallback when spatial_context.bodies is empty."""

import asyncio
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

# agent_workflow imports llm_client which instantiates an OpenAI client at import-time.
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from backend.agent_workflow import _format_unified_context, _prepopulate_entity_store  # noqa: E402
from backend.websocket_manager import ConnectionManager  # noqa: E402
from backend.entity_store import EntityStore  # noqa: E402
from backend.sketch_entity_store import SketchEntityStore  # noqa: E402


def test_unified_context_uses_flat_faces_when_spatial_bodies_empty() -> None:
    context = {
        "bodies": [],
        "faces": [
            {
                "id": "face_0",
                "entity_token": "face-token-0",
                "body_name": "BodyA",
                "surface_type": "plane",
                "normal": [0, 0, 1],
                "centroid": [0, 0, 10],
            }
        ],
        "edges": [],
        "spatial_context": {"units": "mm", "bodies": []},
    }

    text = _format_unified_context(context)

    assert "design_entities:" in text
    assert "faces:" in text
    assert "face_0:" in text


def test_prepopulate_falls_back_to_flat_entities_when_spatial_bodies_empty() -> None:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        manager = ConnectionManager()
        session_id = "test-session"

        manager.pending_results[session_id] = asyncio.Queue()
        manager.pending_entity_context[session_id] = asyncio.Queue()
        manager.conversation_history[session_id] = []
        manager.entity_stores[session_id] = EntityStore()
        manager.sketch_entity_stores[session_id] = SketchEntityStore()

        context = {
            "bodies": [],
            "faces": [
                {
                    "id": "face_0",
                    "entity_token": "face-token-0",
                    "body_name": "BodyA",
                    "surface_type": "plane",
                    "normal": [0, 0, 1],
                    "centroid": [0, 0, 10],
                }
            ],
            "edges": [],
            "vertices": [],
            "spatial_context": {"units": "mm", "bodies": []},
        }

        loop.run_until_complete(_prepopulate_entity_store(session_id, manager, context))

        counts = manager.get_entity_store(session_id).get_entity_counts()
        assert counts["face"] == 1
        assert context["faces"][0].get("entity_ref")
    finally:
        asyncio.set_event_loop(None)
        loop.close()
