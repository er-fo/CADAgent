"""Tests for create_sketch unresolved face-ref diagnostics."""
import os
import sys
import asyncio

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from backend.agent_workflow import _resolve_codegen_entity_refs, SelectionToolCallError
from backend.entity_store import EntityStore


class _ManagerStub:
    def __init__(self, store: EntityStore):
        self._store = store

    def get_entity_store(self, session_id: str) -> EntityStore:
        return self._store

    def get_feature_snapshot(self, session_id: str):
        return None


def test_create_sketch_unresolved_face_ref_includes_candidate_refs():
    async def _run():
        store = EntityStore()
        manager = _ManagerStub(store)
        session_id = "s1"

        await store.register_entities("body", [{"entity_token": "b0", "name": "B"}])
        await store.register_entities("face", [
            {
                "entity_token": "old_face_token",
                "body": "B",
                "normal": [0, 0, 1],
                "centroid": [0, 0, 2.5],
                "surface_type": "planar",
                "area": 100.0,
            }
        ])

        # Drop active refs while preserving persistent cache, then register a similar
        # but not identical face so stale face_0 is unresolved and candidate search can kick in.
        store.soft_clear()
        await store.register_entities("body", [{"entity_token": "b0", "name": "B"}])
        refs = await store.register_entities("face", [
            {
                "entity_token": "new_face_token",
                "body": "B",
                "normal": [0, 0, 1],
                "centroid": [0, 0, 2.8],  # >0.05mm so strict fingerprint reuse does not apply
                "surface_type": "planar",
                "area": 100.0,
            }
        ])
        assert refs[0].ref_id == "face_1"

        try:
            _resolve_codegen_entity_refs(
                session_id,
                manager,
                "create_sketch",
                {
                    "plane_id": "face_0",
                    "sketch_id": "s",
                    "sketch_name": "s",
                    "description": "x",
                },
            )
            assert False, "Expected SelectionToolCallError for stale face ref"
        except SelectionToolCallError as exc:
            msg = str(exc)
            assert "create_sketch plane_id could not be resolved: face_0" in msg
            assert "Nearest candidates: face_1" in msg

    asyncio.run(_run())


def test_create_sketch_unresolved_face_ref_without_entities_raises_instead_of_xy_fallback():
    async def _run():
        store = EntityStore()
        manager = _ManagerStub(store)

        try:
            _resolve_codegen_entity_refs(
                "s1",
                manager,
                "create_sketch",
                {
                    "plane_id": "face_0",
                    "sketch_id": "s",
                    "sketch_name": "s",
                    "description": "x",
                },
            )
            assert False, "Expected SelectionToolCallError for unresolved face ref with empty context"
        except SelectionToolCallError as exc:
            msg = str(exc)
            assert "create_sketch plane_id could not be resolved: face_0" in msg
            assert "No design entities are loaded in the current entity context." in msg
            assert "Wait for refreshed Design Entities" in msg

    asyncio.run(_run())
