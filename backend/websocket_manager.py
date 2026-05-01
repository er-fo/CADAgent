"""
WebSocket Connection Manager for Fusion 360 LLM CAD Agent

Manages WebSocket connections from Fusion 360 add-ins and handles
bidirectional communication with result waiting mechanisms.
"""

import asyncio
import copy
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4
from fastapi import WebSocket
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

# Maximum concurrent WebSocket connections to prevent resource exhaustion.
# Clamp invalid values to safe defaults to avoid accidentally rejecting all clients.
_DEFAULT_MAX_CONNECTIONS = 200
_DEFAULT_MAX_HISTORY_MESSAGES = 200


def _parse_positive_int_env(var_name: str, default: int) -> int:
    raw_value = (os.environ.get(var_name) or str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid %s value %r; using default %d",
            var_name,
            raw_value,
            default,
        )
        return default

    if value < 1:
        logger.warning(
            "Non-positive %s value %d; using default %d",
            var_name,
            value,
            default,
        )
        return default

    return value


MAX_CONNECTIONS = _parse_positive_int_env("MAX_WEBSOCKET_CONNECTIONS", _DEFAULT_MAX_CONNECTIONS)

# Maximum conversation history messages per session to prevent memory exhaustion
# Each session keeps only the last N messages (sliding window)
MAX_HISTORY_MESSAGES = _parse_positive_int_env("MAX_HISTORY_MESSAGES", _DEFAULT_MAX_HISTORY_MESSAGES)


class ConnectionManager:
    """
    Manages WebSocket connections and message routing for Fusion 360 sessions.

    Each session is identified by a unique session_id and maintains:
    - An active WebSocket connection
    - A queue for pending results from Fusion 360
    - Conversation history for multi-turn interactions
    """

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.pending_results: Dict[str, asyncio.Queue] = {}
        self.pending_entity_context: Dict[str, asyncio.Queue] = {}
        self.latest_entity_contexts: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        self.message_checkpoints: Dict[str, List[Dict[str, Any]]] = {}
        self.feature_snapshots: Dict[str, Dict[str, Any]] = {}
        self.ir_document_states: Dict[str, Any] = {}
        self.entity_stores: Dict[str, "EntityStore"] = {}
        self.sketch_entity_stores: Dict[str, "SketchEntityStore"] = {}
        self.user_tokens: Dict[str, Optional[str]] = {}  # JWT tokens for usage tracking
        self.llm_api_keys: Dict[str, Dict[str, str]] = {}  # per-session BYOK provider keys
        self.user_ids: Dict[str, Optional[str]] = {}     # User IDs (sub claim) for rate limiting
        self.authenticated: Dict[str, bool] = {}         # auth status per session
        self.active_build_plans: Dict[str, Optional[Dict[str, Any]]] = {}  # active build plans per session
        self.reasoning_contexts: Dict[str, "ReasoningContext"] = {}  # reasoning history per session

    def _is_socket_connected(self, websocket: WebSocket) -> bool:
        """Check whether a tracked websocket still appears connected."""
        return (
            websocket.client_state == WebSocketState.CONNECTED
            and websocket.application_state == WebSocketState.CONNECTED
        )

    def _reap_stale_connections(self) -> int:
        """Remove stale/disconnected sockets so they do not consume capacity."""
        stale_session_ids = [
            sid
            for sid, ws in self.active_connections.items()
            if not self._is_socket_connected(ws)
        ]
        for stale_id in stale_session_ids:
            self.disconnect(stale_id)

        if stale_session_ids:
            logger.warning(
                "Reaped %d stale WebSocket session(s) before capacity check",
                len(stale_session_ids),
            )

        return len(stale_session_ids)

    async def connect(self, session_id: str, websocket: WebSocket) -> bool:
        """
        Accept and store a new WebSocket connection for a session.

        Args:
            session_id: Unique identifier for the Fusion 360 session
            websocket: WebSocket connection object
        """
        self._reap_stale_connections()

        # If the same session reconnects, replace its previous socket to avoid stale
        # occupancy and to guarantee the newest connection is authoritative.
        previous_websocket = self.active_connections.get(session_id)
        if previous_websocket is not None:
            if self._is_socket_connected(previous_websocket):
                try:
                    await previous_websocket.close(
                        code=1001,
                        reason="Session replaced by reconnect",
                    )
                except RuntimeError as exc:
                    logger.debug(
                        "Best-effort close failed for prior session %s socket: %s",
                        session_id,
                        exc,
                    )
            self.disconnect(session_id)

        # Enforce connection limit to prevent resource exhaustion
        if len(self.active_connections) >= MAX_CONNECTIONS:
            await websocket.accept()  # Accept to send close message
            await websocket.close(code=1008, reason="Server at capacity")
            logger.warning(
                f"Connection rejected for session {session_id}: "
                f"max connections ({MAX_CONNECTIONS}) reached"
            )
            return False

        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.pending_results[session_id] = asyncio.Queue()
        self.pending_entity_context[session_id] = asyncio.Queue()
        self.conversation_history[session_id] = []
        from .entity_store import EntityStore  # late import to avoid circulars
        from .sketch_entity_store import SketchEntityStore  # late import to avoid circulars
        self.entity_stores[session_id] = EntityStore()
        self.sketch_entity_stores[session_id] = SketchEntityStore()
        from .reasoning_context import ReasoningContext  # late import to avoid circulars
        self.reasoning_contexts[session_id] = ReasoningContext(session_id=session_id)
        logger.info(f"Session {session_id} connected. Active sessions: {len(self.active_connections)}")
        # Explicitly start unauthenticated; main.py will gate messages
        self.authenticated[session_id] = False
        self.active_build_plans[session_id] = None
        return True

    def disconnect(self, session_id: str):
        """
        Remove a session's connection and cleanup resources.

        Args:
            session_id: Unique identifier for the session to disconnect
        """
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"Session {session_id} disconnected. Active sessions: {len(self.active_connections)}")

        if session_id in self.pending_results:
            del self.pending_results[session_id]

        if session_id in self.pending_entity_context:
            del self.pending_entity_context[session_id]

        if session_id in self.latest_entity_contexts:
            del self.latest_entity_contexts[session_id]

        if session_id in self.conversation_history:
            del self.conversation_history[session_id]

        if session_id in self.message_checkpoints:
            del self.message_checkpoints[session_id]

        if session_id in self.feature_snapshots:
            del self.feature_snapshots[session_id]
        if session_id in self.ir_document_states:
            del self.ir_document_states[session_id]
        if session_id in self.entity_stores:
            del self.entity_stores[session_id]
        if session_id in self.sketch_entity_stores:
            del self.sketch_entity_stores[session_id]

        if session_id in self.user_tokens:
            del self.user_tokens[session_id]

        if session_id in self.llm_api_keys:
            del self.llm_api_keys[session_id]

        if session_id in self.user_ids:
            del self.user_ids[session_id]

        if session_id in self.authenticated:
            del self.authenticated[session_id]

        if session_id in self.active_build_plans:
            del self.active_build_plans[session_id]

        if session_id in self.reasoning_contexts:
            del self.reasoning_contexts[session_id]

    def set_user_token(self, session_id: str, token: Optional[str]) -> None:
        """
        Set the user's JWT token for a session.

        Args:
            session_id: Session identifier
            token: JWT access token from Supabase authentication
        """
        self.user_tokens[session_id] = token
        logger.debug(f"Set user token for session {session_id}: {'present' if token else 'none'}")

    def set_user_id(self, session_id: str, user_id: Optional[str]) -> None:
        """
        Set the user's ID (sub claim from JWT) for a session.

        Args:
            session_id: Session identifier
            user_id: User identifier from JWT sub claim (for rate limiting)
        """
        self.user_ids[session_id] = user_id
        logger.debug(f"Set user ID for session {session_id}: {user_id[:8]}..." if user_id else "none")

    def set_llm_api_keys(self, session_id: str, llm_api_keys: Optional[Dict[str, str]]) -> None:
        """Store per-session BYOK provider keys sent by the add-in at auth time."""
        sanitized: Dict[str, str] = {}
        for key, value in (llm_api_keys or {}).items():
            if not isinstance(key, str):
                continue
            if not isinstance(value, str):
                continue
            value = value.strip()
            if not value:
                continue
            sanitized[key] = value
        self.llm_api_keys[session_id] = sanitized
        logger.debug(
            "Set BYOK keys for session %s: providers=%s",
            session_id,
            sorted(sanitized.keys()),
        )

    def get_llm_api_keys(self, session_id: str) -> Dict[str, str]:
        """Return copy of per-session provider keys for LLM/routing calls."""
        return dict(self.llm_api_keys.get(session_id, {}))

    def get_user_id(self, session_id: str) -> Optional[str]:
        """
        Get the user's ID for a session.

        Args:
            session_id: Session identifier

        Returns:
            User ID from JWT sub claim, or None if not authenticated
        """
        return self.user_ids.get(session_id)

    def mark_authenticated(self, session_id: str, value: bool = True) -> None:
        """Record that a session has successfully authenticated."""
        self.authenticated[session_id] = value

    def is_authenticated(self, session_id: str) -> bool:
        """Check whether a session has authenticated."""
        return self.authenticated.get(session_id, False)

    def get_user_token(self, session_id: str) -> Optional[str]:
        """
        Get the user's JWT token for a session.

        Args:
            session_id: Session identifier

        Returns:
            JWT access token or None if not authenticated
        """
        return self.user_tokens.get(session_id)

    def get_entity_store(self, session_id: str) -> "EntityStore":
        """
        Retrieve the entity store for a session, creating it if needed.
        """
        if session_id not in self.entity_stores:
            from .entity_store import EntityStore  # late import to avoid circulars
            self.entity_stores[session_id] = EntityStore()
        return self.entity_stores[session_id]

    def get_sketch_entity_store(self, session_id: str) -> "SketchEntityStore":
        """Retrieve the sketch entity store for a session, creating it if needed."""
        if session_id not in self.sketch_entity_stores:
            from .sketch_entity_store import SketchEntityStore  # late import to avoid circulars
            self.sketch_entity_stores[session_id] = SketchEntityStore()
        return self.sketch_entity_stores[session_id]

    def clear_entity_store(self, session_id: str) -> None:
        """Reset all cached entity references for a session."""
        store = self.entity_stores.get(session_id)
        if store:
            store.clear()
        sketch_store = self.sketch_entity_stores.get(session_id)
        if sketch_store:
            sketch_store.clear()
        self.clear_latest_entity_context(session_id)

    def set_feature_snapshot(self, session_id: str, snapshot: Dict[str, Any]) -> None:
        """Store the most recent feature snapshot for a session."""
        self.feature_snapshots[session_id] = copy.deepcopy(snapshot)

    def get_feature_snapshot(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the cached feature snapshot for a session."""
        snapshot = self.feature_snapshots.get(session_id)
        return copy.deepcopy(snapshot) if snapshot is not None else None

    def clear_feature_snapshot(self, session_id: str) -> None:
        """Remove any cached feature snapshot for a session."""
        if session_id in self.feature_snapshots:
            del self.feature_snapshots[session_id]

    def get_ir_document_state(self, session_id: str) -> Any:
        """Retrieve the mutable committed IR document state for a session."""
        return self.ir_document_states.get(session_id)

    def set_ir_document_state(self, session_id: str, state: Any) -> None:
        """Store the mutable committed IR document state for a session."""
        self.ir_document_states[session_id] = state

    def clear_ir_document_state(self, session_id: str) -> None:
        """Remove committed IR document state for a session."""
        if session_id in self.ir_document_states:
            del self.ir_document_states[session_id]

    def set_latest_entity_context(self, session_id: str, entity_context: Dict[str, Any]) -> None:
        """Store the most recent full entity context for a session."""
        self.latest_entity_contexts[session_id] = copy.deepcopy(entity_context)

    def get_latest_entity_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the latest known full entity context for a session."""
        context = self.latest_entity_contexts.get(session_id)
        return copy.deepcopy(context) if context is not None else None

    def clear_latest_entity_context(self, session_id: str) -> None:
        """Remove any cached latest entity context for a session."""
        if session_id in self.latest_entity_contexts:
            del self.latest_entity_contexts[session_id]

    def get_connection_stats(self) -> Dict[str, int]:
        """Expose capacity stats for health/debug endpoints."""
        active_count = len(self.active_connections)
        return {
            "websocket_active_connections": active_count,
            "websocket_max_connections": MAX_CONNECTIONS,
            "websocket_available_slots": max(MAX_CONNECTIONS - active_count, 0),
        }

    async def send_message(self, session_id: str, message: dict):
        """
        Send a message to a specific Fusion 360 session.

        Args:
            session_id: Target session identifier
            message: Dictionary containing the message data (will be JSON serialized)

        Raises:
            KeyError: If session_id is not found in active connections
            Exception: If sending fails (connection error, etc.)
        """
        if session_id not in self.active_connections:
            logger.error(f"Attempted to send message to non-existent session: {session_id}")
            raise KeyError(f"Session {session_id} not found in active connections")

        try:
            websocket = self.active_connections[session_id]
            outbound_message = dict(message)
            msg_type = outbound_message.get('type', 'unknown')

            # Ensure entity-context requests carry a correlation id.
            if msg_type == "request_entity_context":
                correlation_id = outbound_message.get("context_request_id") or outbound_message.get("message_id")
                if not correlation_id:
                    correlation_id = f"ctx-{uuid4().hex}"
                outbound_message.setdefault("context_request_id", correlation_id)
                outbound_message.setdefault("message_id", correlation_id)
            elif msg_type == "feature_snapshot_request":
                message_id = outbound_message.get("message_id")
                if not message_id:
                    outbound_message["message_id"] = f"fs-{uuid4().hex}"

            logger.info("→ [SEND] session=%s type=%s payload_keys=%s", session_id, msg_type, list(outbound_message.keys()))
            await websocket.send_json(outbound_message)
            logger.info("✓ [SEND_OK] session=%s type=%s", session_id, msg_type)
        except Exception as e:
            logger.error(f"Failed to send message to session {session_id}: type={message.get('type')} err={str(e)}")
            self.disconnect(session_id)
            raise

    async def wait_for_fusion_result(
        self,
        session_id: str,
        timeout: int = 30,
        *,
        expected_message_id: Optional[str] = None,
    ) -> dict:
        """
        Wait for a result from Fusion 360 with timeout.

        This method blocks until a result with the expected message_id is available
        in the session's queue or the timeout is reached. Non-matching results are
        discarded with a warning to prevent queue misrouting when operations are
        superseded (e.g., a revert arriving before a cancelled tool completes).

        Args:
            session_id: Session identifier to wait for results from
            timeout: Maximum seconds to wait for a result (default: 30)
            expected_message_id: If provided, only return results with matching
                message_id field; discard mismatched results.

        Returns:
            dict: Result data from Fusion 360

        Raises:
            asyncio.TimeoutError: If no result received within timeout period
            KeyError: If session_id is not found
        """
        if session_id not in self.pending_results:
            logger.error(f"Attempted to wait for result from non-existent session: {session_id}")
            raise KeyError(f"Session {session_id} not found")

        queue = self.pending_results[session_id]
        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.error(f"Timeout waiting for result from session {session_id}")
                raise asyncio.TimeoutError()

            try:
                logger.debug(f"Waiting for result from session {session_id} (timeout: {remaining:.1f}s)")
                result = await asyncio.wait_for(queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for result from session {session_id}")
                raise

            # If no expected_message_id, return immediately (legacy behavior)
            if expected_message_id is None:
                logger.debug(f"Result received from session {session_id}")
                return result

            # Check for matching message_id
            result_message_id = result.get("message_id")
            if result_message_id == expected_message_id:
                logger.debug(
                    f"Result received from session {session_id} with matching message_id={expected_message_id}"
                )
                return result

            # Non-matching result: discard and continue waiting
            logger.warning(
                f"Discarding stale result for session {session_id}: "
                f"expected message_id={expected_message_id}, got {result_message_id}. "
                f"Result type={result.get('type')}, success={result.get('success')}"
            )

    def flush_pending_results(self, session_id: str) -> int:
        """
        Drain all pending results from a session's queue without processing them.

        Called when a task is cancelled or times out to prevent orphaned results
        from being consumed by subsequent operations.

        Args:
            session_id: Session identifier

        Returns:
            Number of results flushed from the queue
        """
        if session_id not in self.pending_results:
            return 0

        queue = self.pending_results[session_id]
        flushed = 0

        while not queue.empty():
            try:
                result = queue.get_nowait()
                flushed += 1
                logger.debug(
                    f"Flushed stale result for session {session_id}: "
                    f"type={result.get('type')}, message_id={result.get('message_id')}"
                )
            except asyncio.QueueEmpty:
                break

        if flushed:
            logger.info(f"Flushed {flushed} stale result(s) for session {session_id}")

        return flushed

    async def store_fusion_result(self, session_id: str, result: dict):
        """
        Store a result from Fusion 360 into the session's queue.

        This unblocks any coroutine waiting on wait_for_fusion_result().

        Args:
            session_id: Session identifier that sent the result
            result: Result data from Fusion 360

        Raises:
            KeyError: If session_id is not found
        """
        if session_id not in self.pending_results:
            logger.error(f"Attempted to store result for non-existent session: {session_id}")
            raise KeyError(f"Session {session_id} not found")

        await self.pending_results[session_id].put(result)
        logger.debug(f"Result stored for session {session_id}")

    async def wait_for_entity_context(
        self,
        session_id: str,
        timeout: float = 5.0,
        *,
        expected_context_request_id: Optional[str] = None,
        expected_message_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for fresh entity context from Fusion 360 with timeout.

        This method blocks until entity context is available in the session's queue
        or the timeout is reached.

        Args:
            session_id: Session identifier to wait for entity context from
            timeout: Maximum seconds to wait for entity context (default: 5.0)

        Returns:
            dict: Entity context data from Fusion 360, or None if timeout

        Raises:
            KeyError: If session_id is not found
        """
        if session_id not in self.pending_entity_context:
            logger.error(f"Attempted to wait for entity context from non-existent session: {session_id}")
            raise KeyError(f"Session {session_id} not found")

        expected_correlation_id = expected_context_request_id or expected_message_id
        queue = self.pending_entity_context[session_id]
        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning(f"Timeout waiting for entity context from session {session_id}")
                return None

            try:
                logger.debug(f"Waiting for entity context from session {session_id} (timeout: {remaining:.2f}s)")
                entity_context = await asyncio.wait_for(queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for entity context from session {session_id}")
                return None

            if expected_correlation_id is None:
                logger.debug(f"Entity context received from session {session_id}")
                return entity_context

            actual_correlation_id = None
            if isinstance(entity_context, dict):
                actual_correlation_id = (
                    entity_context.get("context_request_id")
                    or entity_context.get("message_id")
                )

            if actual_correlation_id == expected_correlation_id:
                logger.debug(
                    "Entity context received from session %s with matching correlation id=%s",
                    session_id,
                    expected_correlation_id,
                )
                return entity_context

            logger.warning(
                "Discarding stale entity context for session %s: expected correlation id=%s, got %s",
                session_id,
                expected_correlation_id,
                actual_correlation_id,
            )

    async def store_entity_context(self, session_id: str, entity_context: Dict[str, Any]):
        """
        Store entity context from Fusion 360 into the session's queue.

        This unblocks any coroutine waiting on wait_for_entity_context().

        Args:
            session_id: Session identifier that sent the entity context
            entity_context: Entity context data from Fusion 360

        Raises:
            KeyError: If session_id is not found
        """
        if session_id not in self.pending_entity_context:
            logger.error(f"Attempted to store entity context for non-existent session: {session_id}")
            raise KeyError(f"Session {session_id} not found")

        payload = dict(entity_context)
        correlation_id = payload.get("context_request_id") or payload.get("message_id")
        if correlation_id is not None:
            payload.setdefault("context_request_id", correlation_id)
            payload.setdefault("message_id", correlation_id)

        self.set_latest_entity_context(session_id, payload)
        await self.pending_entity_context[session_id].put(payload)
        logger.debug(f"Entity context stored for session {session_id}")

    def flush_pending_entity_context(self, session_id: str) -> int:
        """
        Drain all pending entity-context payloads from a session's queue.

        Called when a task is cancelled or superseded to prevent stale context
        responses from being consumed by subsequent operations.
        """
        if session_id not in self.pending_entity_context:
            return 0

        queue = self.pending_entity_context[session_id]
        flushed = 0

        while not queue.empty():
            try:
                context = queue.get_nowait()
                flushed += 1
                logger.debug(
                    "Flushed stale entity context for session %s: context_request_id=%s message_id=%s",
                    session_id,
                    context.get("context_request_id"),
                    context.get("message_id"),
                )
            except asyncio.QueueEmpty:
                break

        if flushed:
            logger.info(f"Flushed {flushed} stale entity-context payload(s) for session {session_id}")

        return flushed

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of message dictionaries in chronological order
        """
        return self.conversation_history.get(session_id, []).copy()

    def set_conversation_history(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Set the complete conversation history for a session (replaces existing).
        Automatically trims to MAX_HISTORY_MESSAGES to prevent memory exhaustion.

        Args:
            session_id: Session identifier
            messages: List of message dictionaries
        """
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        self.conversation_history[session_id] = messages.copy()
        self._trim_conversation_history(session_id)
        logger.debug(f"Set conversation history for session {session_id}: {len(self.conversation_history[session_id])} messages")

    def _trim_conversation_history(self, session_id: str) -> None:
        """
        Enforce MAX_HISTORY_MESSAGES limit by keeping only the most recent messages (sliding window).

        Args:
            session_id: Session identifier
        """
        history = self.conversation_history.get(session_id)
        if history and len(history) > MAX_HISTORY_MESSAGES:
            original_len = len(history)
            self.conversation_history[session_id] = history[-MAX_HISTORY_MESSAGES:]
            logger.debug(
                f"Trimmed conversation history for session {session_id}: "
                f"{original_len} → {len(self.conversation_history[session_id])} messages"
            )

    def clear_conversation(self, session_id: str) -> None:
        """
        Clear the conversation history for a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.conversation_history:
            self.conversation_history[session_id] = []
            logger.info(f"Cleared conversation history for session {session_id}")

    def trim_conversation_to_index(
        self,
        session_id: str,
        conversation_index: Optional[int],
        *,
        include_current_message: bool = True,
    ) -> int:
        """
        Truncate the stored conversation history at the checkpoint boundary.

        Args:
            session_id: Session identifier
            conversation_index: Conversation length captured at checkpoint
                (messages that existed before the user message was appended).
            include_current_message: When True, retain the user message that
                triggered the checkpoint. When False, revert to the state before
                it was sent.

        Returns:
            New conversation length after trimming.
        """
        history = self.conversation_history.get(session_id)
        if history is None:
            return 0

        try:
            index_value = int(conversation_index) if conversation_index is not None else 0
        except (TypeError, ValueError):
            index_value = 0

        if index_value < 0:
            index_value = 0

        retain_count = index_value + (1 if include_current_message else 0)
        retain_count = max(0, min(retain_count, len(history)))
        original_len = len(history)

        if original_len > retain_count:
            self.conversation_history[session_id] = history[:retain_count]
            logger.info(
                "Trimmed conversation history for session %s from %d to %d messages",
                session_id,
                original_len,
                retain_count,
            )
        else:
            retain_count = original_len

        return retain_count

    def save_checkpoint(self, session_id: str, checkpoint_data: Dict[str, Any]) -> None:
        """
        Save a message checkpoint for timeline revert functionality.

        Args:
            session_id: Session identifier
            checkpoint_data: Dictionary containing checkpoint metadata:
                - message_id: Unique identifier for this checkpoint
                - marker_position: Timeline marker position when checkpoint was created
                - timeline_count: Number of timeline items at checkpoint
                - message_text: User's message text
                - timestamp: When checkpoint was created
                - conversation_index: Position in conversation history
        """
        if session_id not in self.message_checkpoints:
            self.message_checkpoints[session_id] = []

        self.message_checkpoints[session_id].append(checkpoint_data.copy())

        # Limit checkpoint history to last 50 messages to prevent memory bloat
        max_checkpoints = 50
        if len(self.message_checkpoints[session_id]) > max_checkpoints:
            self.message_checkpoints[session_id] = self.message_checkpoints[session_id][-max_checkpoints:]

        logger.debug(
            f"Saved checkpoint for session {session_id}: message_id={checkpoint_data.get('message_id')}, "
            f"marker_position={checkpoint_data.get('marker_position')}"
        )

    def get_checkpoints(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all checkpoints for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of checkpoint dictionaries in chronological order
        """
        return self.message_checkpoints.get(session_id, []).copy()

    def get_checkpoint_by_message_id(self, session_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific checkpoint by message ID.

        Args:
            session_id: Session identifier
            message_id: Unique message identifier

        Returns:
            Checkpoint dictionary if found, None otherwise
        """
        checkpoints = self.message_checkpoints.get(session_id, [])
        for checkpoint in checkpoints:
            if checkpoint.get("message_id") == message_id:
                return checkpoint.copy()
        return None

    def clear_checkpoints(self, session_id: str) -> None:
        """
        Clear all checkpoints for a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.message_checkpoints:
            self.message_checkpoints[session_id] = []
            logger.info(f"Cleared checkpoints for session {session_id}")

    def prune_checkpoints_after(
        self,
        session_id: str,
        max_conversation_index: Optional[int],
    ) -> int:
        """
        Drop checkpoints captured after the retained conversation index.

        Args:
            session_id: Session identifier
            max_conversation_index: Highest conversation index to keep

        Returns:
            Number of checkpoints removed.
        """
        checkpoints = self.message_checkpoints.get(session_id)
        if not checkpoints:
            return 0

        try:
            max_index = int(max_conversation_index) if max_conversation_index is not None else 0
        except (TypeError, ValueError):
            max_index = 0

        if max_index < 0:
            max_index = 0

        initial_len = len(checkpoints)
        self.message_checkpoints[session_id] = [
            checkpoint
            for checkpoint in checkpoints
            if int(checkpoint.get("conversation_index", 0) or 0) <= max_index
        ]
        removed = initial_len - len(self.message_checkpoints[session_id])

        if removed:
            logger.info(
                "Pruned %d checkpoint(s) for session %s beyond conversation index %d",
                removed,
                session_id,
                max_index,
            )

        return removed

    def set_active_build_plan(self, session_id: str, build_plan: Optional[Dict[str, Any]]) -> None:
        """
        Set the active build plan for a session.

        Args:
            session_id: Session identifier
            build_plan: Build plan data with design_name, steps, and completed_steps counter
        """
        self.active_build_plans[session_id] = build_plan
        if build_plan:
            logger.info(
                f"Set active build plan for session {session_id}: "
                f"{build_plan.get('design_name')} with {len(build_plan.get('steps', []))} steps"
            )
        else:
            logger.info(f"Cleared active build plan for session {session_id}")

    def get_active_build_plan(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the active build plan for a session.

        Args:
            session_id: Session identifier

        Returns:
            Build plan data if active, None otherwise
        """
        return self.active_build_plans.get(session_id)

    def increment_build_plan_step(self, session_id: str) -> Optional[int]:
        """
        Increment the completed steps counter for the active build plan.

        Args:
            session_id: Session identifier

        Returns:
            New completed_steps count, or None if no active plan
        """
        plan = self.active_build_plans.get(session_id)
        if not plan:
            return None

        plan["completed_steps"] = plan.get("completed_steps", 0) + 1
        completed = plan["completed_steps"]
        total = len(plan.get("steps", []))

        logger.debug(f"Build plan progress for session {session_id}: {completed}/{total} steps")

        # Auto-clear plan when all steps completed
        if completed >= total:
            logger.info(f"Build plan completed for session {session_id}: {total} steps")
            # Don't clear yet - let the workflow handle it after sending completion message

        return completed

    def get_reasoning_context(self, session_id: str) -> "ReasoningContext":
        """
        Retrieve the reasoning context for a session, creating it if needed.

        Args:
            session_id: Session identifier

        Returns:
            ReasoningContext instance for the session
        """
        if session_id not in self.reasoning_contexts:
            from .reasoning_context import ReasoningContext  # late import to avoid circulars
            self.reasoning_contexts[session_id] = ReasoningContext(session_id=session_id)
        return self.reasoning_contexts[session_id]

    def clear_reasoning_context(self, session_id: str) -> None:
        """
        Clear the reasoning context for a session (e.g., on task reset).

        Args:
            session_id: Session identifier
        """
        ctx = self.reasoning_contexts.get(session_id)
        if ctx:
            ctx.clear()
            logger.debug(f"Cleared reasoning context for session {session_id}")

    def format_build_plan_context(self, session_id: str) -> Optional[str]:
        """
        Format the active build plan as context to inject into tool results.

        Args:
            session_id: Session identifier

        Returns:
            Formatted plan context string, or None if no active plan
        """
        plan = self.active_build_plans.get(session_id)
        if not plan:
            return None

        steps = plan.get("steps", [])
        completed = plan.get("completed_steps", 0)
        total = len(steps)
        design_name = plan.get("design_name", "Design")

        lines = [
            "",
            "=" * 50,
            f"ACTIVE BUILD PLAN: {design_name} ({completed}/{total} steps complete)",
            "-" * 50,
        ]

        for i, step in enumerate(steps):
            step_num = i + 1
            description = step.get("description", step.get("operation", f"Step {step_num}"))

            if step_num <= completed:
                marker = "✓"
                suffix = ""
            elif step_num == completed + 1:
                marker = "→"
                suffix = "  ← CURRENT"
            else:
                marker = " "
                suffix = ""

            lines.append(f"  {marker} {step_num}. {description}{suffix}")

        lines.append("=" * 50)
        lines.append("")

        return "\n".join(lines)
