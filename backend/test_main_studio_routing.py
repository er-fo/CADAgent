import pytest
from fastapi.testclient import TestClient

try:
    from . import main
except ImportError:  # pragma: no cover
    from backend.backend import main


pytestmark = pytest.mark.xfail(
    reason="The legacy production backend does not route studio_* websocket messages.",
    strict=False,
)


def test_websocket_studio_execute_routes_to_unified_workflow(monkeypatch):
    captured = {"request": None}

    async def fake_handle_execute_request(session_id, request, manager):
        captured["request"] = request

    async def fake_launch(session_id, coro):
        await coro

    monkeypatch.setattr(main, "handle_execute_request", fake_handle_execute_request)
    monkeypatch.setattr(main, "_launch_session_task", fake_launch)

    with TestClient(main.app) as client:
        with client.websocket_connect("/ws/test-studio-exec") as ws:
            ws.send_json({"type": "studio_execute_request", "user_request": "Create a cube"})

    assert captured["request"] is not None
    assert captured["request"]["execution_target"] == "build123d"
    assert captured["request"]["user_request"] == "Create a cube"


def test_websocket_studio_export_routes_to_unified_workflow(monkeypatch):
    captured = {"request": None}

    async def fake_handle_studio_export_request(session_id, request, manager):
        captured["request"] = request

    async def fake_launch(session_id, coro):
        await coro

    monkeypatch.setattr(main, "handle_studio_export_request", fake_handle_studio_export_request)
    monkeypatch.setattr(main, "_launch_session_task", fake_launch)

    with TestClient(main.app) as client:
        with client.websocket_connect("/ws/test-studio-export") as ws:
            ws.send_json({"type": "studio_export_request", "format": "step"})

    assert captured["request"] is not None
    assert captured["request"]["type"] == "studio_export_request"
    assert captured["request"]["format"] == "step"
