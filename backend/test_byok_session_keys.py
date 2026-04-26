from backend.websocket_manager import ConnectionManager


def test_llm_api_keys_store_and_cleanup() -> None:
    manager = ConnectionManager()
    session_id = "session-1"

    manager.set_llm_api_keys(
        session_id,
        {
            "openai_api_key": "sk-openai",
            "anthropic_api_key": "sk-anthropic",
            "google_api_key": "",  # empty values should be discarded
            "ignored": 123,  # non-string values should be discarded
        },
    )

    stored = manager.get_llm_api_keys(session_id)
    assert stored == {
        "openai_api_key": "sk-openai",
        "anthropic_api_key": "sk-anthropic",
    }

    manager.disconnect(session_id)
    assert manager.get_llm_api_keys(session_id) == {}
