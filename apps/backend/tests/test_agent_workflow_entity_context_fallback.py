"""Regression tests for flat fallback when spatial_context.bodies is empty."""

import asyncio
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

# agent_workflow imports llm_client which instantiates an OpenAI client at import-time.
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from backend.agent_workflow import (  # noqa: E402
    _format_design_entities_xml,
    _format_unified_context,
    _prepopulate_entity_store,
    _refresh_entity_context_with_retry,
    _validate_entity_context,
)
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


def test_validate_entity_context_accepts_nested_spatial_payload() -> None:
    context = {
        "spatial_context": {
            "units": "mm",
            "bodies": [
                {
                    "id": "body_0",
                    "token": "body-token-0",
                    "faces": [
                        {"id": "face_0", "token": "face-token-0"},
                    ],
                    "edges": [
                        {"id": "e0", "token": "edge-token-0"},
                    ],
                }
            ],
        }
    }

    assert _validate_entity_context(context)


def test_refresh_accepts_nested_spatial_payload_for_signature() -> None:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        manager = ConnectionManager()
        session_id = "test-session-refresh"

        manager.pending_results[session_id] = asyncio.Queue()
        manager.pending_entity_context[session_id] = asyncio.Queue()
        manager.conversation_history[session_id] = []
        manager.entity_stores[session_id] = EntityStore()
        manager.sketch_entity_stores[session_id] = SketchEntityStore()

        captured_request_ids: list[str] = []

        async def _capture_send_message(session: str, message: dict) -> None:
            if message.get("type") == "request_entity_context":
                captured_request_ids.append(message.get("context_request_id", ""))
                context = {
                    "spatial_context": {
                        "units": "mm",
                        "bodies": [
                            {
                                "id": "body_0",
                                "token": "body-token-0",
                                "faces": [
                                    {
                                        "id": "face_0",
                                        "token": "face-token-0",
                                        "normal": {"x": 0, "y": 1, "z": 0},
                                        "centroid": {"x": 10, "y": 12, "z": 14},
                                    }
                                ],
                                "edges": [
                                    {"id": "e0", "token": "edge-token-0"},
                                ],
                            }
                        ],
                    },
                    "context_request_id": message.get("context_request_id"),
                }
                await manager.pending_entity_context[session_id].put(context)

        manager.send_message = _capture_send_message  # type: ignore[method-assign]

        refreshed = loop.run_until_complete(
            _refresh_entity_context_with_retry(
                session_id,
                manager,
                "create_extrude",
                prev_signature="",
                max_attempts=1,
                operation_was_noop=False,
            )
        )

        assert refreshed is not None
        assert "spatial_context" in refreshed
        assert len(captured_request_ids) == 1
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_design_entities_xml_handles_dict_vectors_in_spatial_context() -> None:
    context = {
        "spatial_context": {
            "units": "mm",
            "bodies": [
                {
                    "id": "body_0",
                    "token": "body-token-0",
                    "bbox": {
                        "min": {"x": 0, "y": 5, "z": 0},
                        "max": {"x": 20, "y": 25, "z": 30},
                    },
                    "vertices": [
                        {
                            "id": "v0",
                            "token": "vertex-token-0",
                            "p": {"x": 1.25, "y": 6.5, "z": 2.0},
                        }
                    ],
                    "faces": [
                        {
                            "id": "face_0",
                            "token": "face-token-0",
                            "normal": {"x": 0, "y": 1, "z": 0},
                            "centroid": {"x": 10, "y": 15, "z": 20},
                            "frame": {
                                "u": {"x": 1, "y": 0, "z": 0},
                                "v": {"x": 0, "y": 0, "z": 1},
                                "n": {"x": 0, "y": 1, "z": 0},
                            },
                        }
                    ],
                    "edges": [
                        {
                            "id": "e0",
                            "token": "edge-token-0",
                            "length": "12.5",
                            "adjacent_faces": ["face_0"],
                        }
                    ],
                }
            ],
        }
    }

    entity_xml = _format_design_entities_xml(context)

    assert "<design_entities>" in entity_xml
    assert '<body ref="body_0"' in entity_xml
    assert 'bbox_min="[0.0,5.0,0.0]"' in entity_xml
    assert '<vertex ref="v0" p="[1.25,6.50,2.00]" />' in entity_xml
    assert '<face ref="face_0"' in entity_xml
