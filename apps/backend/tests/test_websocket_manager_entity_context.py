import asyncio

from backend.websocket_manager import ConnectionManager


class _DummyWebSocket:
    def __init__(self) -> None:
        self.sent_messages = []

    async def send_json(self, payload):
        self.sent_messages.append(payload)


def test_send_message_injects_entity_context_correlation_id() -> None:
    async def _run() -> None:
        manager = ConnectionManager()
        session_id = "session-send"
        ws = _DummyWebSocket()
        manager.active_connections[session_id] = ws

        await manager.send_message(session_id, {"type": "request_entity_context"})

        assert len(ws.sent_messages) == 1
        sent = ws.sent_messages[0]
        assert sent["type"] == "request_entity_context"
        assert sent.get("context_request_id")
        assert sent.get("message_id") == sent.get("context_request_id")

    asyncio.run(_run())


def test_wait_for_entity_context_filters_by_expected_correlation_id() -> None:
    async def _run() -> None:
        manager = ConnectionManager()
        session_id = "session-filter"
        manager.pending_entity_context[session_id] = asyncio.Queue()

        await manager.store_entity_context(
            session_id,
            {"bodies": [{"name": "stale"}], "faces": [], "edges": [], "context_request_id": "ctx-old"},
        )
        await manager.store_entity_context(
            session_id,
            {"bodies": [{"name": "fresh"}], "faces": [], "edges": [], "context_request_id": "ctx-new"},
        )

        result = await manager.wait_for_entity_context(
            session_id,
            timeout=0.5,
            expected_context_request_id="ctx-new",
        )

        assert result is not None
        assert result.get("context_request_id") == "ctx-new"
        assert result.get("message_id") == "ctx-new"
        assert result.get("bodies") == [{"name": "fresh"}]
        assert manager.pending_entity_context[session_id].empty()

    asyncio.run(_run())


def test_wait_for_entity_context_without_expected_id_returns_first_payload() -> None:
    async def _run() -> None:
        manager = ConnectionManager()
        session_id = "session-legacy"
        manager.pending_entity_context[session_id] = asyncio.Queue()

        await manager.store_entity_context(
            session_id,
            {"bodies": [{"name": "first"}], "faces": [], "edges": [], "message_id": "msg-1"},
        )
        await manager.store_entity_context(
            session_id,
            {"bodies": [{"name": "second"}], "faces": [], "edges": [], "message_id": "msg-2"},
        )

        result = await manager.wait_for_entity_context(session_id, timeout=0.5)

        assert result is not None
        assert result.get("bodies") == [{"name": "first"}]
        assert result.get("message_id") == "msg-1"
        assert result.get("context_request_id") == "msg-1"

    asyncio.run(_run())


def test_flush_pending_entity_context_drains_queue() -> None:
    async def _run() -> None:
        manager = ConnectionManager()
        session_id = "session-flush"
        manager.pending_entity_context[session_id] = asyncio.Queue()

        await manager.store_entity_context(
            session_id,
            {"bodies": [], "faces": [], "edges": [], "context_request_id": "ctx-1"},
        )
        await manager.store_entity_context(
            session_id,
            {"bodies": [], "faces": [], "edges": [], "context_request_id": "ctx-2"},
        )

        flushed = manager.flush_pending_entity_context(session_id)

        assert flushed == 2
        assert manager.pending_entity_context[session_id].empty()
        assert manager.flush_pending_entity_context(session_id) == 0
        assert manager.flush_pending_entity_context("missing-session") == 0

    asyncio.run(_run())
