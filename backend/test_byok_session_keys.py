import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.websocket_manager import ConnectionManager

try:
    from . import main
except ImportError:  # pragma: no cover
    from backend import main


def _configure_main_for_ws_test(monkeypatch: pytest.MonkeyPatch) -> ConnectionManager:
    manager = main.ConnectionManager()
    monkeypatch.setattr(main, "manager", manager)
    monkeypatch.setattr(main, "AUTH_BYPASS", False)
    return manager


def _configure_main_for_auth_bypass_ws_test(monkeypatch: pytest.MonkeyPatch) -> ConnectionManager:
    manager = main.ConnectionManager()
    monkeypatch.setattr(main, "manager", manager)
    monkeypatch.setattr(main, "AUTH_BYPASS", True)
    return manager


def test_connection_manager_set_llm_api_keys_sanitizes_and_cleans_up() -> None:
    manager = ConnectionManager()
    session_id = "session-unit"

    manager.set_llm_api_keys(
        session_id,
        {
            "openai_api_key": "  sk-openai  ",
            "anthropic_api_key": "sk-anthropic",
            "google_api_key": "",  # empty values should be discarded
            "ignored": 123,  # non-string values should be discarded
        },
    )

    assert manager.get_llm_api_keys(session_id) == {
        "openai_api_key": "sk-openai",
        "anthropic_api_key": "sk-anthropic",
    }

    manager.disconnect(session_id)
    assert manager.get_llm_api_keys(session_id) == {}


def test_websocket_auth_bypass_authenticate_preserves_llm_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _configure_main_for_auth_bypass_ws_test(monkeypatch)
    session_id = "session-auth-bypass-keys"

    with TestClient(main.app) as client:
        with client.websocket_connect(f"/ws/{session_id}") as ws:
            ws.send_json(
                {
                    "type": "authenticate",
                    "api_keys": {
                        "anthropic_api_key": "  sk-ant-dev  ",
                        "openai_api_key": "",
                    },
                }
            )

            assert ws.receive_json() == {"type": "authentication_ack", "authenticated": True}
            assert manager.is_authenticated(session_id) is True
            assert manager.get_llm_api_keys(session_id) == {"anthropic_api_key": "sk-ant-dev"}


def test_websocket_update_api_keys_llm_payload_pre_auth_and_execute_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _configure_main_for_ws_test(monkeypatch)
    session_id = "session-ws-llm-payload"

    with TestClient(main.app) as client:
        with client.websocket_connect(f"/ws/{session_id}") as ws:
            ws.send_json(
                {
                    "type": "update_api_keys",
                    "llm_api_keys": {
                        "openai_api_key": "  sk-openai  ",
                        "anthropic_api_key": "sk-anthropic",
                        "google_api_key": "   ",
                        "ignored": 123,
                    },
                }
            )

            assert ws.receive_json() == {"type": "api_keys_updated"}
            assert manager.get_llm_api_keys(session_id) == {
                "openai_api_key": "sk-openai",
                "anthropic_api_key": "sk-anthropic",
            }
            assert manager.is_authenticated(session_id) is False

            ws.send_json({"type": "execute_request", "user_request": "Create a cube"})
            assert ws.receive_json() == {
                "type": "authentication_error",
                "message": "Authenticate first.",
            }
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws.receive_json()
            assert exc_info.value.code == 1008


def test_websocket_update_api_keys_api_keys_alias_pre_auth_stores_sanitized_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _configure_main_for_ws_test(monkeypatch)
    session_id = "session-ws-alias-payload"

    with TestClient(main.app) as client:
        with client.websocket_connect(f"/ws/{session_id}") as ws:
            ws.send_json(
                {
                    "type": "update_api_keys",
                    "api_keys": {
                        "openai_api_key": "sk-openai",
                        "anthropic_api_key": "  sk-anthropic  ",
                        "google_api_key": "",
                        "ignored": None,
                    },
                }
            )

            assert ws.receive_json() == {"type": "api_keys_updated"}
            assert manager.get_llm_api_keys(session_id) == {
                "openai_api_key": "sk-openai",
                "anthropic_api_key": "sk-anthropic",
            }
            assert manager.is_authenticated(session_id) is False
