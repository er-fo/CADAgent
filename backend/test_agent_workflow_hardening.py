"""Regression tests for workflow hardening safeguards."""
import os
import sys
import asyncio

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from backend.agent_workflow import (
    SelectionToolCallError,
    _build_tool_intent_key,
    _preflight_hole_center_on_face,
    _resolve_entity_tokens_or_refs,
    _resolve_single_entity_ref,
)
from backend.entity_store import EntityStore


class _ManagerStub:
    def __init__(self, store: EntityStore):
        self._store = store

    def get_entity_store(self, session_id: str) -> EntityStore:
        return self._store


def _build_store_with_stale_face(*, ambiguous: bool = False) -> tuple[EntityStore, _ManagerStub]:
    async def _run() -> tuple[EntityStore, _ManagerStub]:
        store = EntityStore()
        manager = _ManagerStub(store)

        await store.register_entities("body", [{"entity_token": "b0", "name": "B"}])
        await store.register_entities(
            "face",
            [
                {
                    "entity_token": "old_face_token",
                    "body": "B",
                    "normal": [0, 0, 1],
                    "centroid": [0, 0, 2.5],
                    "surface_type": "planar",
                    "area": 100.0,
                }
            ],
        )

        # Keep persistent cache while forcing active refs to be rebuilt.
        store.soft_clear()
        await store.register_entities("body", [{"entity_token": "b0", "name": "B"}])
        new_faces = [
            {
                "entity_token": "new_face_token_a",
                "body": "B",
                "normal": [0, 0, 1],
                "centroid": [0, 0, 2.8],  # >0.05mm so face_0 is stale
                "surface_type": "planar",
                "area": 100.0,
            }
        ]
        if ambiguous:
            new_faces.append(
                {
                    "entity_token": "new_face_token_b",
                    "body": "B",
                    "normal": [0, 0, 1],
                    "centroid": [0, 0, 2.2],  # Similar distance from stale face centroid
                    "surface_type": "planar",
                    "area": 100.0,
                }
            )
        await store.register_entities("face", new_faces)
        return store, manager

    return asyncio.run(_run())


def test_stale_face_single_ref_auto_recovers_when_unambiguous():
    store, manager = _build_store_with_stale_face(ambiguous=False)
    token = _resolve_single_entity_ref(
        "s1",
        manager,
        "face_0",
        expected_kind="face",
        context="face_token",
    )
    assert token == "new_face_token_a"


def test_stale_face_array_auto_recovers_when_unambiguous():
    store, manager = _build_store_with_stale_face(ambiguous=False)
    tokens = _resolve_entity_tokens_or_refs(
        "s1",
        manager,
        ["face_0"],
        expected_kind="face",
        context="face_refs",
    )
    assert tokens == ["new_face_token_a"]


def test_stale_face_recovery_errors_when_ambiguous():
    store, manager = _build_store_with_stale_face(ambiguous=True)
    try:
        _resolve_single_entity_ref(
            "s1",
            manager,
            "face_0",
            expected_kind="face",
            context="face_token",
        )
        assert False, "Expected SelectionToolCallError for ambiguous stale face recovery"
    except SelectionToolCallError as exc:
        msg = str(exc)
        assert "Candidate replacements" in msg
        assert "face_" in msg


def test_duplicate_intent_key_ignores_noise_and_param_aliases():
    key_a = _build_tool_intent_key(
        "create_tapped_hole",
        {
            "face_ref": "face_5",
            "center_x": 10.0,
            "center_y": 20.0,
            "center_z": 30.0,
            "thread_type": "metric",
            "thread_size": "M6",
            "thread_depth": 8.0,
            "description": "first wording",
            "feature_name": "A",
        },
    )
    key_b = _build_tool_intent_key(
        "create_tapped_hole",
        {
            "face_token": "face_5",
            "center_x": 10.0,
            "center_y": 20.0,
            "center_z": 30.0,
            "thread_type": "metric",
            "thread_size": "M6",
            "thread_depth": 8.0,
            "description": "different wording",
            "feature_name": "B",
        },
    )
    assert key_a == key_b


def test_hole_preflight_rejects_point_far_from_face_plane():
    async def _run() -> tuple[EntityStore, _ManagerStub]:
        store = EntityStore()
        manager = _ManagerStub(store)
        await store.register_entities("face", [
            {
                "entity_token": "face_token_a",
                "normal": [0, 0, 1],
                "centroid": [0, 0, 0],
                "surface_type": "planar",
                "area": 100.0,
            }
        ])
        return store, manager

    _store, manager = asyncio.run(_run())
    error = _preflight_hole_center_on_face(
        "s1",
        manager,
        tool_name="create_tapped_hole",
        face_token="face_token_a",
        center_x=0.0,
        center_y=0.0,
        center_z=25.0,
    )
    assert error is not None
    assert "away from the selected face plane" in error
