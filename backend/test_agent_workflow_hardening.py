"""Regression tests for workflow hardening safeguards."""
import os
import sys
import asyncio
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from backend.agent_workflow import (
    SelectionToolCallError,
    _build_tool_intent_key,
    _execute_feature_tool_call,
    _execute_geometry_tool_call,
    _execute_workflow_loop,
    _get_sketch_entity_store,
    handle_execute_request,
    _preflight_face_sketch_uv_bounds,
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


class _RequestManagerStub:
    def __init__(self, store: EntityStore):
        self._store = store
        self.history = []
        self.sent_messages = []
        self.saved_checkpoints = []

    def get_entity_store(self, session_id: str) -> EntityStore:
        return self._store

    def get_conversation_history(self, session_id: str):
        return list(self.history)

    def save_checkpoint(self, session_id: str, checkpoint_data):
        self.saved_checkpoints.append(checkpoint_data)

    async def send_message(self, session_id: str, payload):
        self.sent_messages.append(payload)

    def set_conversation_history(self, session_id: str, messages):
        self.history = list(messages)

    def get_feature_snapshot(self, session_id: str):
        return None

    def get_active_build_plan(self, session_id: str):
        return None


class _ResultQueueManager:
    def __init__(self, store: EntityStore, results):
        self._store = store
        self._results = list(results)
        self.sent_messages = []
        self.requeued = []
        self.history = []
        self._reasoning_context = type(
            "_ReasoningContext",
            (),
            {
                "entries": [],
                "add_entry": lambda self, _entry: None,
                "get_injection_text": lambda self: "",
            },
        )()

    def get_entity_store(self, session_id: str) -> EntityStore:
        return self._store

    async def send_message(self, session_id: str, payload):
        self.sent_messages.append(payload)

    async def wait_for_fusion_result(self, session_id: str, timeout=30):
        if not self._results:
            raise asyncio.TimeoutError
        return self._results.pop(0)

    async def store_fusion_result(self, session_id: str, result):
        self.requeued.append(result)

    def get_user_token(self, session_id: str):
        return "token"

    def get_llm_api_keys(self, session_id: str):
        return {}

    def get_reasoning_context(self, session_id: str):
        return self._reasoning_context

    def set_conversation_history(self, session_id: str, messages):
        self.history = list(messages)

    def get_active_build_plan(self, session_id: str):
        return None


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


def test_face_sketch_uv_preflight_rejects_out_of_bounds_circle():
    async def _setup() -> _ManagerStub:
        store = EntityStore()
        return _ManagerStub(store)

    manager = asyncio.run(_setup())
    sketch_store = _get_sketch_entity_store("s1", manager)  # type: ignore[arg-type]
    sketch_store.register_sketch_metadata(
        "port_sketch",
        {
            "plane_kind": "face",
            "face_ref": "face_0",
            "uv_bounds": {"u_min": -2.0, "u_max": 2.0, "v_min": -1.0, "v_max": 1.0},
        },
    )

    error = _preflight_face_sketch_uv_bounds(
        "s1",
        manager,  # type: ignore[arg-type]
        tool_name="add_circle",
        tool_input={
            "sketch_id": "port_sketch",
            "center_u": 4.0,
            "center_v": 0.0,
            "radius": 0.3,
        },
    )
    assert error is not None
    assert "face-bounds preflight" in error
    assert "Requested UV extents" in error


def test_face_sketch_uv_preflight_blocks_when_bounds_missing():
    async def _setup() -> _ManagerStub:
        store = EntityStore()
        return _ManagerStub(store)

    manager = asyncio.run(_setup())
    sketch_store = _get_sketch_entity_store("s2", manager)  # type: ignore[arg-type]
    sketch_store.register_sketch_metadata(
        "wall_sketch",
        {
            "plane_kind": "face",
            "face_ref": "face_3",
        },
    )

    error = _preflight_face_sketch_uv_bounds(
        "s2",
        manager,  # type: ignore[arg-type]
        tool_name="add_rectangle",
        tool_input={
            "sketch_id": "wall_sketch",
            "corner1_u": -1.0,
            "corner1_v": -0.4,
            "corner2_u": 1.0,
            "corner2_v": 0.4,
        },
    )
    assert error is not None
    assert "UV bounds are unavailable" in error


def test_handle_execute_request_preserves_store_when_entity_context_missing(monkeypatch: pytest.MonkeyPatch):
    async def _run():
        store = EntityStore()
        await store.register_entities("face", [{"entity_token": "face_token_existing"}])
        manager = _RequestManagerStub(store)

        soft_clear_calls = {"count": 0}
        original_soft_clear = store.soft_clear

        def wrapped_soft_clear():
            soft_clear_calls["count"] += 1
            original_soft_clear()

        async def fake_execute_loop(*args, **kwargs):
            return None

        async def fake_snapshot(*args, **kwargs):
            return None

        monkeypatch.setattr(store, "soft_clear", wrapped_soft_clear)
        monkeypatch.setattr("backend.agent_workflow.initialize_session", lambda _sid: None)
        monkeypatch.setattr("backend.agent_workflow._execute_workflow_loop", fake_execute_loop)
        monkeypatch.setattr("backend.agent_workflow._ensure_feature_snapshot", fake_snapshot)

        await handle_execute_request("s-preserve", {"user_request": "continue"}, manager)  # type: ignore[arg-type]

        assert soft_clear_calls["count"] == 0
        assert store.get_entity_counts()["face"] == 1

    asyncio.run(_run())


def test_handle_execute_request_clears_when_entity_context_present(monkeypatch: pytest.MonkeyPatch):
    async def _run():
        store = EntityStore()
        await store.register_entities("face", [{"entity_token": "face_token_existing"}])
        manager = _RequestManagerStub(store)

        soft_clear_calls = {"count": 0}
        prepopulate_calls = {"count": 0}
        original_soft_clear = store.soft_clear

        def wrapped_soft_clear():
            soft_clear_calls["count"] += 1
            original_soft_clear()

        async def fake_prepopulate(*args, **kwargs):
            prepopulate_calls["count"] += 1

        async def fake_execute_loop(*args, **kwargs):
            return None

        async def fake_snapshot(*args, **kwargs):
            return None

        monkeypatch.setattr(store, "soft_clear", wrapped_soft_clear)
        monkeypatch.setattr("backend.agent_workflow.initialize_session", lambda _sid: None)
        monkeypatch.setattr("backend.agent_workflow._prepopulate_entity_store", fake_prepopulate)
        monkeypatch.setattr("backend.agent_workflow._execute_workflow_loop", fake_execute_loop)
        monkeypatch.setattr("backend.agent_workflow._ensure_feature_snapshot", fake_snapshot)

        await handle_execute_request(
            "s-clear",
            {
                "user_request": "continue",
                "entity_context": {"bodies": [], "faces": [{"entity_token": "face_token_new"}], "edges": []},
            },
            manager,  # type: ignore[arg-type]
        )

        assert soft_clear_calls["count"] == 1
        assert prepopulate_calls["count"] == 1

    asyncio.run(_run())


def test_geometry_wait_ignores_mismatched_tool_use_id():
    async def _run():
        store = EntityStore()
        manager = _ResultQueueManager(
            store,
            results=[
                {"tool_use_id": "other", "success": True, "message": "wrong"},
                {"tool_use_id": "expected", "success": True, "message": "matched"},
            ],
        )

        success, message = await _execute_geometry_tool_call(
            "s-geo",
            manager,  # type: ignore[arg-type]
            "clear_face_selection",
            "expected",
            {},
            "face",
        )

        assert success is True
        assert "matched" in message
        assert len(manager.requeued) == 1
        assert manager.requeued[0].get("tool_use_id") == "other"

    asyncio.run(_run())


def test_feature_wait_ignores_mismatched_tool_use_id():
    async def _run():
        store = EntityStore()
        manager = _ResultQueueManager(
            store,
            results=[
                {"tool_use_id": "other", "success": True, "message": "wrong"},
                {
                    "tool_use_id": "expected",
                    "success": True,
                    "message": "matched",
                    "edge_count": 1,
                    "radius": 2.0,
                    "radius_unit": "mm",
                },
            ],
        )

        success, message, _raw = await _execute_feature_tool_call(
            "s-feature",
            manager,  # type: ignore[arg-type]
            "apply_fillet",
            "expected",
            {"entity_tokens": ["edge_token_1"], "radius": 2.0},
        )

        assert success is True
        assert "matched" in message
        assert len(manager.requeued) == 1
        assert manager.requeued[0].get("tool_use_id") == "other"

    asyncio.run(_run())


def test_execute_code_wait_ignores_mismatched_tool_use_id(monkeypatch: pytest.MonkeyPatch):
    async def _run():
        manager = _ResultQueueManager(
            EntityStore(),
            results=[
                {"tool_use_id": "other", "success": True, "message": "wrong"},
                {"tool_use_id": "toolu_expected", "success": True, "message": "matched"},
            ],
        )
        llm_calls = {"count": 0}

        async def fake_call_claude_with_tools(*args, **kwargs):
            llm_calls["count"] += 1
            if llm_calls["count"] == 1:
                return {
                    "stop_reason": "tool_use",
                    "content": [
                        {"type": "text", "text": "do it"},
                        {"type": "tool_use", "id": "toolu_expected", "name": "dummy_tool", "input": {"description": "x"}},
                    ],
                }
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Done"}]}

        monkeypatch.setattr("backend.agent_workflow.USE_PROMPT_ROUTING", False)
        monkeypatch.setattr("backend.agent_workflow.call_claude_with_tools", fake_call_claude_with_tools)
        monkeypatch.setattr("backend.agent_workflow.translate_tool_call", lambda _name, _input: "print('ok')")
        monkeypatch.setattr("backend.agent_workflow._resolve_codegen_entity_refs", lambda *_args: dict(_args[-1]))

        await _execute_workflow_loop(
            session_id="s-direct",
            messages=[{"role": "user", "content": [{"type": "text", "text": "run dummy"}]}],
            max_iterations=3,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "direct-1"},
            feature_snapshot=None,
        )

        assert len(manager.requeued) == 1
        assert manager.requeued[0].get("tool_use_id") == "other"

    asyncio.run(_run())
