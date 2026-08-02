"""
Asynchronous WebSocket client tailored for the Fusion 360 add-in.

Runs an asyncio event loop on a background thread so the Fusion main thread stays
responsive. Incoming messages are pushed to a thread-safe queue and surfaced via
callbacks, while outbound messages are scheduled on the loop in a thread-safe way.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from typing import Callable, List, Optional

# Add bundled websockets to path if not already available
try:
    import websockets
except ImportError:
    # Add lib directory to path to use bundled websockets
    lib_path = os.path.join(os.path.dirname(__file__), 'lib')
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    import websockets

logger = logging.getLogger(__name__)


def _fusion_log_probe(message: str) -> None:
    """Best-effort bridge to Fusion's Text Command log for field diagnostics."""
    try:
        import adsk.core  # type: ignore

        app = adsk.core.Application.get()
        if app:
            app.log(message)
    except Exception:
        # Swallow any Fusion logging failures; we don't want telemetry to crash the add-in.
        pass


class FusionWebSocketClient:
    """Manages the lifecycle of a WebSocket connection on a background loop."""

    def __init__(self, url: str, user_token: Optional[str] = None, api_keys: Optional[dict] = None):
        self._url = url
        self._user_token = user_token
        self._api_keys = api_keys or {}  # BYOK: User's API keys for LLM providers
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="CADAgentWebSocketLoop",
            daemon=True,
        )
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._message_handlers: List[Callable[[dict], None]] = []
        self._state_handlers: List[Callable[[bool], None]] = []
        self._connected_event = threading.Event()
        self._stopping = threading.Event()

    def start(self) -> None:
        """Start the background event loop and connect to the server."""
        if self._thread.is_alive():
            return
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        fut.result()

    def stop(self) -> None:
        """Close the socket and stop the event loop."""
        self._stopping.set()
        if self._websocket:
            asyncio.run_coroutine_threadsafe(self._websocket.close(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def add_message_handler(self, handler: Callable[[dict], None]) -> None:
        """Register a callback invoked when a JSON message arrives."""
        self._message_handlers.append(handler)

    def add_state_handler(self, handler: Callable[[bool], None]) -> None:
        """Register a callback invoked on connect/disconnect with a bool flag."""
        self._state_handlers.append(handler)

    def send_json(self, payload: dict) -> None:
        """Send a JSON payload to the backend asynchronously."""
        async def _send():
            if not self._websocket:
                raise ConnectionError("WebSocket is not connected.")
            await self._websocket.send(json.dumps(payload))

        future = asyncio.run_coroutine_threadsafe(_send(), self._loop)
        future.result()

    def is_connected(self) -> bool:
        return self._connected_event.is_set() and self._websocket is not None

    def set_user_token(self, token: Optional[str]) -> None:
        """
        Set or update the user's JWT token for authentication.

        Args:
            token: JWT access token from Supabase authentication
        """
        self._user_token = token
        # If already connected, send authentication message with API keys
        if self.is_connected():
            self._send_auth_message()
            logger.info("Sent authentication update with %s token", "valid" if token else "no")

    def set_api_keys(self, api_keys: dict) -> None:
        """
        Set or update the user's API keys for LLM providers (BYOK).

        Args:
            api_keys: Dictionary with keys like 'anthropic_api_key', 'openai_api_key'
        """
        self._api_keys = api_keys or {}
        logger.info(
            "[api_keys] Client keys set anthropic=%s openai=%s google=%s",
            bool(self._api_keys.get("anthropic_api_key")),
            bool(self._api_keys.get("openai_api_key")),
            bool(self._api_keys.get("google_api_key")),
        )
        # If already connected, re-authenticate with new keys
        if self.is_connected() and self._user_token:
            self._send_auth_message()
            logger.info("Sent authentication update with API keys")

    def _send_auth_message(self) -> None:
        """Send authentication message with token and API keys."""
        auth_payload = {
            "type": "authenticate",
            "token": self._user_token,
            "api_keys": self._api_keys,
            # Compatibility: newer backends expect llm_api_keys
            "llm_api_keys": self._api_keys,
        }
        self.send_json(auth_payload)

    # Internal helpers -----------------------------------------------------
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        pending = asyncio.all_tasks(loop=self._loop)
        for task in pending:
            task.cancel()
        try:
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        finally:
            self._loop.close()

    async def _connect(self) -> None:
        try:
            logger.info("Connecting to backend WebSocket at %s", self._url)
            self._websocket = await websockets.connect(self._url, ping_interval=20, ping_timeout=20)
            self._connected_event.set()
            self._notify_state(True)

            # Send authentication message. Token is optional for anonymous sessions.
            auth_payload = {
                "type": "authenticate",
                "token": self._user_token,
                "api_keys": self._api_keys,
                # Compatibility: newer backends expect llm_api_keys
                "llm_api_keys": self._api_keys,
            }
            logger.info(
                "[api_keys] Sending authenticate with keys anthropic=%s openai=%s google=%s",
                bool(self._api_keys.get("anthropic_api_key")),
                bool(self._api_keys.get("openai_api_key")),
                bool(self._api_keys.get("google_api_key")),
            )
            await self._websocket.send(json.dumps(auth_payload))
            logger.info(
                "Sent authentication (authenticated=%s) with %d API keys",
                bool(self._user_token),
                len([k for k, v in self._api_keys.items() if v]),
            )

            asyncio.create_task(self._receiver())
        except Exception:
            logger.exception("Failed to connect to backend WebSocket.")
            raise

    async def _receiver(self) -> None:
        assert self._websocket is not None
        try:
            async for message in self._websocket:
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    logger.error("Received non-JSON message: %s", message)
                    continue

                # INSTRUMENTATION: Track message arrival in WS pipeline
                msg_type = payload.get("type", "<no type>")
                logger.info(f"[WS_RECEIVE] Message received: type={msg_type}")
                _fusion_log_probe(f"[WS_RECEIVE] type={msg_type}")

                # NOTE: This background thread must not touch adsk.* APIs; it only
                # queues data and signals the main thread through handlers.
                for handler in list(self._message_handlers):
                    try:
                        handler(payload)
                    except Exception as e:
                        # INSTRUMENTATION: Show which message type failed in handler
                        logger.exception(f"WebSocket message handler failed for type={msg_type}: {e}")
                        _fusion_log_probe(f"[WS_RECEIVE] handler failed type={msg_type}: {e}")
        except websockets.ConnectionClosed:
            logger.info("WebSocket connection closed by server.")
            _fusion_log_probe("[WS_RECEIVE] connection closed by server")
        except Exception:
            logger.exception("WebSocket receiver crashed.")
            _fusion_log_probe("[WS_RECEIVE] receiver crashed")
        finally:
            self._connected_event.clear()
            self._notify_state(False)
            if not self._stopping.is_set():
                logger.warning("WebSocket disconnected unexpectedly.")

    def _notify_state(self, connected: bool) -> None:
        for handler in list(self._state_handlers):
            try:
                handler(connected)
            except Exception:
                logger.exception("WebSocket state handler failed.")
