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
    _format_current_ref_table,
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
        self.latest_entity_context = None
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

    def get_latest_entity_context(self, session_id: str):
        return self.latest_entity_context

    def set_latest_entity_context(self, session_id: str, entity_context):
        self.latest_entity_context = dict(entity_context)

    def clear_latest_entity_context(self, session_id: str):
        self.latest_entity_context = None

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


def _extract_tool_result_texts(messages):
    texts = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content_blocks = message.get("content")
        if not isinstance(content_blocks, list):
            continue
        for content_block in content_blocks:
            if not isinstance(content_block, dict) or content_block.get("type") != "tool_result":
                continue
            tool_content = content_block.get("content")
            if not isinstance(tool_content, list):
                continue
            for text_block in tool_content:
                if isinstance(text_block, dict) and text_block.get("type") == "text":
                    texts.append(str(text_block.get("text") or ""))
    return texts


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


def test_unresolved_face_ref_error_lists_current_valid_refs():
    async def _run() -> _ManagerStub:
        store = EntityStore()
        manager = _ManagerStub(store)
        await store.register_entities(
            "face",
            [
                {
                    "entity_token": "front_face_token",
                    "normal": [0, 1, 0],
                    "centroid": [0, 18, 6],
                    "surface_type": "planar",
                    "area": 800.0,
                }
            ],
        )
        return manager

    manager = asyncio.run(_run())
    with pytest.raises(SelectionToolCallError) as exc_info:
        _resolve_single_entity_ref(
            "s1",
            manager,
            "face_5",
            expected_kind="face",
            context="face_token",
        )

    msg = str(exc_info.value)
    assert "Unknown entity ref: face_5" in msg
    assert "Current valid face refs (1): face_0" in msg
    assert "list_features does not refresh entity refs" in msg


def test_unresolved_edge_refs_error_lists_current_valid_refs():
    async def _run() -> _ManagerStub:
        store = EntityStore()
        manager = _ManagerStub(store)
        await store.register_entities(
            "edge",
            [
                {"entity_token": "edge_token_0", "edge_type": "line", "length": 10.0},
                {"entity_token": "edge_token_1", "edge_type": "line", "length": 12.0},
            ],
        )
        return manager

    manager = asyncio.run(_run())
    with pytest.raises(SelectionToolCallError) as exc_info:
        _resolve_entity_tokens_or_refs(
            "s1",
            manager,
            ["e0", "e3"],
            expected_kind="edge",
            context="entity_tokens (edges)",
        )

    msg = str(exc_info.value)
    assert "e3: Unknown entity ref: e3" in msg
    assert "Current valid edge refs (2): e0, e1" in msg
    assert "list_features does not refresh entity refs" in msg


def test_unresolved_edge_ref_error_lists_stale_fingerprint_candidate():
    async def _run() -> _ManagerStub:
        store = EntityStore()
        await store.register_entities(
            "edge",
            [
                {"entity_token": "old_edge_0", "edge_type": "line", "length": 8.0, "midpoint": [0, 0, 0]},
                {"entity_token": "old_edge_1", "edge_type": "line", "length": 10.0, "midpoint": [2, 0, 0]},
                {"entity_token": "old_edge_2", "edge_type": "line", "length": 12.0, "midpoint": [4, 0, 0]},
                {"entity_token": "old_edge_3", "edge_type": "line", "length": 24.0, "midpoint": [10, 0, 0]},
            ],
        )
        store.soft_clear()
        await store.register_entities(
            "edge",
            [
                {"entity_token": "new_edge_target", "edge_type": "line", "length": 24.1, "midpoint": [10.05, 0, 0]},
                {"entity_token": "new_edge_other", "edge_type": "line", "length": 6.0, "midpoint": [50, 0, 0]},
            ],
        )
        return _ManagerStub(store)

    manager = asyncio.run(_run())
    with pytest.raises(SelectionToolCallError) as exc_info:
        _resolve_entity_tokens_or_refs(
            "s1",
            manager,
            ["e3"],
            expected_kind="edge",
            context="entity_tokens (edges)",
        )

    msg = str(exc_info.value)
    assert "Current valid edge refs (2): e4, e5" in msg
    assert "Candidate replacements for e3: e4" in msg


def test_current_ref_table_is_compact_and_authoritative():
    text = _format_current_ref_table(
        {
            "units": "mm",
            "bodies": [
                {
                    "entity_token": "body_token_0",
                    "entity_ref": "body_0",
                    "name": "Case",
                    "bounding_box": {"min": [0, 0, 0], "max": [10, 20, 5]},
                }
            ],
            "faces": [
                {
                    "entity_token": "face_token_0",
                    "entity_ref": "face_0",
                    "body_name": "Case",
                    "surface_type": "planar",
                    "normal": [0, 1, 0],
                    "centroid": [5, 20, 2.5],
                }
            ],
            "edges": [
                {"entity_token": "edge_token_0", "entity_ref": "e0"},
                {"entity_token": "edge_token_1", "entity_ref": "e1"},
            ],
        }
    )

    assert "current_design_refs:" in text
    assert "do not use refs absent from this table" in text
    assert "list_features shows timeline features only" in text
    assert "face_0: body=Case" in text
    assert "edges_1_2: e0, e1" in text


def test_execute_loop_injects_current_ref_table_each_llm_call(monkeypatch: pytest.MonkeyPatch):
    async def _run():
        store = EntityStore()
        await store.register_entities("body", [{"entity_token": "body_token_0", "name": "Case"}])
        await store.register_entities(
            "face",
            [
                {
                    "entity_token": "face_token_0",
                    "normal": [0, 1, 0],
                    "centroid": [0, 18, 6],
                    "surface_type": "planar",
                }
            ],
        )
        await store.register_entities("edge", [{"entity_token": "edge_token_0", "edge_type": "line"}])
        manager = _ResultQueueManager(store, results=[])
        manager.latest_entity_context = {
            "bodies": [{"entity_token": "body_token_0", "entity_ref": "body_0", "name": "Case"}],
            "faces": [
                {
                    "entity_token": "face_token_0",
                    "entity_ref": "face_0",
                    "body_name": "Case",
                    "normal": [0, 1, 0],
                    "centroid": [0, 18, 6],
                    "surface_type": "planar",
                }
            ],
            "edges": [{"entity_token": "edge_token_0", "entity_ref": "e0", "edge_type": "line"}],
        }
        captured_messages = []

        async def fake_call_claude_with_tools(messages, *args, **kwargs):
            captured_messages.append(messages)
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Done"}]}

        monkeypatch.setattr("backend.agent_workflow.USE_PROMPT_ROUTING", False)
        monkeypatch.setattr("backend.agent_workflow.call_claude_with_tools", fake_call_claude_with_tools)

        await _execute_workflow_loop(
            session_id="s-current-refs",
            messages=[{"role": "user", "content": [{"type": "text", "text": "continue"}]}],
            max_iterations=1,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "refs-1"},
            feature_snapshot=None,
        )

        assert captured_messages
        last_message = captured_messages[0][-1]
        text = last_message["content"][0]["text"]
        assert "current_design_refs:" in text
        assert "face_0" in text
        assert "e0" in text

    asyncio.run(_run())


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


def test_face_sketch_uv_preflight_checks_line_and_arc_extents():
    async def _setup() -> _ManagerStub:
        return _ManagerStub(EntityStore())

    manager = asyncio.run(_setup())
    sketch_store = _get_sketch_entity_store("s-line-arc", manager)  # type: ignore[arg-type]
    sketch_store.register_sketch_metadata(
        "wall_sketch",
        {
            "plane_kind": "face",
            "face_ref": "face_0",
            "uv_bounds": {"u_min": -1.0, "u_max": 1.0, "v_min": -1.0, "v_max": 1.0},
        },
    )

    assert _preflight_face_sketch_uv_bounds(
        "s-line-arc",
        manager,  # type: ignore[arg-type]
        tool_name="add_line",
        tool_input={"sketch_id": "wall_sketch", "start_u": -0.5, "start_v": 0.0, "end_u": 0.5, "end_v": 0.0},
    ) is None
    line_error = _preflight_face_sketch_uv_bounds(
        "s-line-arc",
        manager,  # type: ignore[arg-type]
        tool_name="add_line",
        tool_input={"sketch_id": "wall_sketch", "start_u": -0.5, "start_v": 0.0, "end_u": 2.0, "end_v": 0.0},
    )
    assert line_error is not None
    assert "add_line rejected by face-bounds preflight" in line_error

    assert _preflight_face_sketch_uv_bounds(
        "s-line-arc",
        manager,  # type: ignore[arg-type]
        tool_name="add_arc",
        tool_input={
            "sketch_id": "wall_sketch",
            "center_u": 0.0,
            "center_v": 0.0,
            "start_u": 0.4,
            "start_v": 0.0,
            "end_u": 0.0,
            "end_v": 0.4,
        },
    ) is None
    arc_error = _preflight_face_sketch_uv_bounds(
        "s-line-arc",
        manager,  # type: ignore[arg-type]
        tool_name="add_arc",
        tool_input={
            "sketch_id": "wall_sketch",
            "center_u": 0.0,
            "center_v": 0.0,
            "start_u": 1.5,
            "start_v": 0.0,
            "end_u": 0.0,
            "end_v": 1.5,
        },
    )
    assert arc_error is not None
    assert "add_arc rejected by face-bounds preflight" in arc_error


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

        success, message, raw_result = await _execute_geometry_tool_call(
            "s-geo",
            manager,  # type: ignore[arg-type]
            "clear_face_selection",
            "expected",
            {},
            "face",
        )

        assert success is True
        assert "matched" in message
        assert raw_result["tool_use_id"] == "expected"
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


def test_topology_deferred_hole_retry_is_not_suppressed_as_duplicate(monkeypatch: pytest.MonkeyPatch):
    async def _run():
        store = EntityStore()
        await store.register_entities(
            "face",
            [
                {
                    "entity_token": "face_token_0",
                    "normal": [0, 0, 1],
                    "centroid": [0, 0, 0],
                    "surface_type": "planar",
                    "area": 100.0,
                }
            ],
        )
        manager = _ResultQueueManager(store, results=[])

        holes = [
            {
                "face_ref": "face_0",
                "center_x": 0.0,
                "center_y": 0.0,
                "center_z": 0.0,
                "diameter": 4.0,
                "extent_type": "through_all",
            },
            {
                "face_ref": "face_0",
                "center_x": 12.0,
                "center_y": 0.0,
                "center_z": 0.0,
                "diameter": 4.0,
                "extent_type": "through_all",
            },
            {
                "face_ref": "face_0",
                "center_x": 0.0,
                "center_y": 12.0,
                "center_z": 0.0,
                "diameter": 4.0,
                "extent_type": "through_all",
            },
            {
                "face_ref": "face_0",
                "center_x": 12.0,
                "center_y": 12.0,
                "center_z": 0.0,
                "diameter": 4.0,
                "extent_type": "through_all",
            },
        ]
        llm_calls = {"count": 0}
        executed_tool_ids = []

        def hole_call(tool_use_id, tool_input):
            return {
                "type": "tool_use",
                "id": tool_use_id,
                "name": "create_simple_hole",
                "input": dict(tool_input),
            }

        async def fake_call_claude_with_tools(*args, **kwargs):
            llm_calls["count"] += 1
            if llm_calls["count"] == 1:
                return {
                    "stop_reason": "tool_use",
                    "content": [
                        {"type": "text", "text": "Create four mounting holes."},
                        *[
                            hole_call(f"toolu_hole_{idx}", hole_input)
                            for idx, hole_input in enumerate(holes, start=1)
                        ],
                    ],
                }
            if llm_calls["count"] == 2:
                return {
                    "stop_reason": "tool_use",
                    "content": [
                        {"type": "text", "text": "Retry the first deferred hole after topology refresh."},
                        hole_call("toolu_hole_2_retry", holes[1]),
                    ],
                }
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Done"}]}

        async def fake_execute_feature_tool_call(session_id, manager, tool_name, tool_use_id, tool_input, description=""):
            executed_tool_ids.append(tool_use_id)
            return True, f"{tool_name} {tool_use_id} ok", {
                "success": True,
                "tool_use_id": tool_use_id,
                "message": "ok",
            }

        async def fake_refresh_after_success(*args, **kwargs):
            return None

        async def fake_runtime_sync(*args, **kwargs):
            return None

        monkeypatch.setattr("backend.agent_workflow.USE_PROMPT_ROUTING", False)
        monkeypatch.setattr("backend.agent_workflow.call_claude_with_tools", fake_call_claude_with_tools)
        monkeypatch.setattr("backend.agent_workflow._execute_feature_tool_call", fake_execute_feature_tool_call)
        monkeypatch.setattr("backend.agent_workflow._refresh_and_enrich_after_success", fake_refresh_after_success)
        monkeypatch.setattr("backend.agent_workflow._ensure_runtime_entity_context_synced", fake_runtime_sync)

        await _execute_workflow_loop(
            session_id="s-hole-deferral",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Add four holes"}]}],
            max_iterations=4,
            model_name=None,
            manager=manager,  # type: ignore[arg-type]
            last_user_message_sent=None,
            request={"execution_target": "fusion", "request_id": "holes-1"},
            feature_snapshot=None,
        )

        tool_result_texts = _extract_tool_result_texts(manager.history)

        assert executed_tool_ids == ["toolu_hole_1", "toolu_hole_2_retry"]
        assert sum("Deferred 'create_simple_hole'" in text for text in tool_result_texts) == 3
        assert not any("Skipped duplicate create_simple_hole" in text for text in tool_result_texts)

    asyncio.run(_run())
