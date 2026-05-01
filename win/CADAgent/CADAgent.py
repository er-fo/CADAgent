"""
Entry point for the CADAgent Fusion 360 add-in.

This is the main module that Fusion 360 loads when starting the add-in.
It initializes all commands and manages the add-in lifecycle following
Fusion 360 best practices.

Key features:
- WebSocket connection to AI backend for natural language CAD requests
- Multi-command architecture with Execute, Settings, and Status commands
- Asynchronous code execution with real-time feedback
- Planning mode with user approval workflow
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import threading
import uuid
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import adsk.core
import adsk.fusion

from . import config
from .code_executor import CodeExecutor
from .websocket_client import FusionWebSocketClient
from .palette_manager import PaletteManager
from .supabase_auth import SupabaseAuthClient
from .api_key_manager import get_api_key_manager
from . import body_tools
from . import edge_tools
from . import face_tools
from . import feature_tools
from . import camera_tools
from . import spatial_analyzer
from . import general_utils
from .selection_extractor import extract_selection_context
import time

# Configure logging based on config settings
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='[%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Message types that can arrive in high volume (streaming) and should not spam logs.
_QUIET_MESSAGE_TYPES = {"reasoning_chunk", "plan_chunk"}

DEFAULT_SUPABASE_URL = "https://fcxnngctkfwpfbbhrmbs.supabase.co"
DEFAULT_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_9pBlFZWV0LzXWNqYHgULpg_Gy86vf2j"


def _fusion_probe(message: str) -> None:
    """Best-effort bridge to Fusion's Text Commands log for field diagnostics."""
    try:
        import adsk.core  # type: ignore

        app = adsk.core.Application.get()
        if app:
            app.log(message)
    except Exception:
        # Don't let probe logging crash the add-in
        pass

# Minimal .env loader (avoids dependency on python-dotenv in Fusion sandbox)
def _strip_inline_env_comment(value: str) -> str:
    """Strip inline comments from unquoted env values."""
    in_single = False
    in_double = False
    for i, ch in enumerate(value):
        if ch == '\"' and not in_single:
            in_double = not in_double
        elif ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '#' and not in_single and not in_double:
            # Treat as comment only when preceded by whitespace (KEY=VALUE # comment)
            if i > 0 and value[i - 1].isspace():
                return value[:i].rstrip()
    return value


def _load_env_file(path: Path) -> None:
    if not path.exists():
        logger.warning(f"Env file not found: {path}")
        return
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            value = _strip_inline_env_comment(value)
            value = value.strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        logger.info(f"Loaded environment from {path}")
    except Exception as exc:
        logger.warning(f"Failed to load env file {path}: {exc}")

# Load only the CADAgent-specific env file to avoid pulling unrelated secrets.
cadagent_env_path = Path(__file__).resolve().parent / ".env.cadagent"
_load_env_file(cadagent_env_path)

# Global references maintained by Fusion 360
_app: Optional[adsk.core.Application] = None
_controller: Optional["AgentController"] = None

SELECTION_PREVIEW_LIMIT = 6


class PlanApprovalEventHandler(adsk.core.CustomEventHandler):
    """Handles plan approval on the main UI thread via CustomEvent."""

    def __init__(self, controller: "AgentController"):
        super().__init__()
        self._controller = controller

    def notify(self, args: adsk.core.CustomEventArgs) -> None:
        """Called on main thread when plan approval event fires."""
        try:
            event_args = adsk.core.CustomEventArgs.cast(args)
            data = json.loads(event_args.additionalInfo)
            plan_text = data.get("plan_text", "")
            doc_id = data.get("doc_id")

            logger.info("Plan approval event received on main thread")
            self._controller.handle_plan_approval_on_main_thread(plan_text, doc_id)

        except Exception as e:
            logger.exception(f"Failed to handle plan approval event: {e}")


class _InboundMessageEventHandler(adsk.core.CustomEventHandler):
    """Dispatch queued backend messages on Fusion's main thread."""

    def __init__(self, controller: "AgentController"):
        super().__init__()
        self._controller = controller

    def notify(self, args: adsk.core.CustomEventArgs) -> None:
        """Drain inbound queue on the UI thread."""
        try:
            self._controller.process_pending_messages()
        except Exception as exc:
            logger.exception("Failed to process inbound messages on main thread: %s", exc)


class AgentController:
    """
    Coordinates the add-in components and routes messages between them.

    This controller manages:
    - WebSocket connection to the backend
    - Code execution in the Fusion context
    - Message routing between UI and backend
    - Custom event handling for thread-safe operations
    """

    def __init__(self, app: adsk.core.Application):
        self._app = app
        self._ui = app.userInterface
        # When the add-in is loaded on Fusion startup there may be no document
        # (or the document is still initializing), so defer binding to a design
        # until a real document is activated. This prevents
        # InternalValidationError: document exceptions from activeProduct.
        self._design: Optional[adsk.fusion.Design] = None
        try:
            self._design = self._resolve_design_reference(getattr(app, "activeDocument", None))
        except Exception as exc:
            logger.debug("Deferring design resolution until document activation: %s", exc)
        
        # Per-document session management
        # doc_id -> { 'session_id': str, 'ws_client': FusionWebSocketClient, 'created_at': float, 'last_active': float, 'doc_name': str }
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._active_doc_id: Optional[str] = None
        self._synthetic_ids: Dict[int, str] = {}
        self._last_sent_token: Optional[str] = None

        # Core components
        self._code_executor = CodeExecutor(app)
        self._palette_manager = PaletteManager(app, self)
        self._incoming_messages: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue()
        self._event_handlers: List[Any] = []
        self._running = False
        self._plan_chunks: Dict[str, List[str]] = {}
        self._pending_plan_full: Dict[str, str] = {}

        self._auth_bypass = (
            os.environ.get("CADAGENT_AUTH_BYPASS", os.environ.get("AUTH_BYPASS", "false")).lower()
            in ("1", "true", "yes", "on")
        )

        # Supabase auth client
        supabase_url = os.environ.get("SUPABASE_URL", DEFAULT_SUPABASE_URL)
        supabase_key = os.environ.get("SUPABASE_PUBLISHABLE_KEY", DEFAULT_SUPABASE_PUBLISHABLE_KEY)
        self._auth_client: Optional[SupabaseAuthClient] = None
        self._auth_error_emitted: bool = False  # Throttle repetitive auth_error spam

        if supabase_url and supabase_key:
            try:
                self._auth_client = SupabaseAuthClient(supabase_url, supabase_key)
                # Try to restore session on startup
                if self._auth_client.restore_session():
                    logger.info("Restored Supabase session on startup")

                    # Validate the session by testing with a profile fetch
                    # This ensures tokens are actually valid, not just present
                    try:
                        profile = self._auth_client.get_profile()
                        if profile:
                            logger.info(f"Session validated successfully for {profile.get('email', 'unknown')}")
                            try:
                                # Push profile to palette immediately; queued until handshake/UI ready
                                self._palette_manager.send_message('user_profile', profile=profile)
                            except Exception as push_err:
                                logger.debug(f"Deferred profile push failed (will retry on UI request): {push_err}")
                        else:
                            logger.warning("Session restored but validation failed - session cleared")
                    except Exception as validation_error:
                        logger.warning(f"Session validation error: {validation_error}")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase auth client: {e}")
        else:
            logger.warning("Supabase credentials not configured; falling back to auth bypass")
            # Mirror backend behavior: when Supabase isn't configured, operate in auth bypass mode.
            self._auth_bypass = True

        # Custom event for plan approval (must be on main thread)
        self._plan_approval_event = app.registerCustomEvent('CADAgentPlanApproval')
        self._plan_approval_handler = PlanApprovalEventHandler(self)
        self._plan_approval_event.add(self._plan_approval_handler)
        self._event_handlers.append(self._plan_approval_handler)

        # Custom event for inbound message dispatch
        self._inbound_event_id = config.EVENT_ID_INBOUND_MESSAGE
        self._inbound_event = app.registerCustomEvent(self._inbound_event_id)
        self._inbound_handler = _InboundMessageEventHandler(self)
        self._inbound_event.add(self._inbound_handler)
        self._event_handlers.append(self._inbound_handler)

        logger.info("AgentController initialized (per-document sessions)")

        # Subscribe to workspace activation event (critical for palette visibility on startup)
        try:
            workspace_handler = _WorkspaceActivatedHandler(self)
            self._ui.workspaceActivated.add(workspace_handler)
            self._event_handlers.append(workspace_handler)
            logger.info("✓ Subscribed to workspaceActivated event")
        except Exception as e:
            logger.warning(f"Unable to subscribe to workspaceActivated: {e}")

        # Subscribe to document events
        try:
            handler_activated = _DocumentActivatedHandler(self)
            self._app.documentActivated.add(handler_activated)
            self._event_handlers.append(handler_activated)
        except Exception as e:
            logger.warning(f"Unable to subscribe to documentActivated: {e}")

        # Optional events; not critical but useful for cleanup/logging
        for evt_name, Handler in (
            ('documentClosed', _DocumentClosedHandler),
            ('documentOpened', _DocumentOpenedHandler),
            ('documentCreated', _DocumentCreatedHandler),
        ):
            try:
                evt = getattr(self._app, evt_name, None)
                if evt is not None:
                    h = Handler(self)
                    evt.add(h)
                    self._event_handlers.append(h)
            except Exception as e:
                logger.debug(f"Event subscription failed for {evt_name}: {e}")

    # ------------------------------------------------------------------ Lifecycle
    def start(self) -> None:
        """Start the controller and all its components."""
        logger.info("=" * 70)
        logger.info("STARTING CADAGENT CONTROLLER")
        logger.info("=" * 70)

        # Set running flag before starting timer
        self._running = True

        # Initialize per-document session for current active document (if any)
        # Note: During startup, the document may not be fully initialized yet, which can
        # cause InternalValidationError. This is not fatal - the documentActivated event
        # will create the session when a document is actually ready.
        logger.info("Initializing session for active document (if available)...")
        try:
            self._activate_session_for_current_document()
            logger.info("✓ Document session initialized")
        except Exception as e:
            logger.warning(f"⚠ Could not initialize session for active document (will retry on documentActivated): {e}")
            logger.debug("Document session initialization deferred", exc_info=True)

        # Initialize palette UI (creates palette but doesn't show it yet)
        logger.info("Initializing palette UI...")
        try:
            self._palette_manager.start()
            logger.info("✓ Palette UI created (will be shown when workspace activates)")
        except Exception as e:
            logger.error(f"❌ Failed to create palette: {e}", exc_info=True)
            raise

        # If Fusion already has an active workspace at startup, show the palette immediately
        # so launch-on-start users see the UI without waiting for a workspaceActivated event.
        self._show_palette_if_workspace_ready()

        # Push stored API keys to any session we just initialized (best effort)
        try:
            self._push_api_keys_all_sessions(reason="startup")
        except Exception as e:
            logger.debug(f"[api_keys] startup push failed: {e}")

        # If already authenticated, proactively sync key status to UI
        try:
            if self.is_authenticated():
                status = self.get_api_keys_status()
                self._palette_manager.send_message('api_keys_status', status=status)
        except Exception as e:
            logger.debug(f"[api_keys] startup status push failed: {e}")

        logger.info("=" * 70)
        logger.info("CADAGENT CONTROLLER READY")
        logger.info("=" * 70)

    def _show_palette_if_workspace_ready(self) -> None:
        """Show palette immediately when a workspace is already active on startup."""
        try:
            active_ws = getattr(self._ui, "activeWorkspace", None)
            if active_ws:
                logger.info("Active workspace detected at startup; showing palette immediately")
                self._palette_manager.show_palette()
            else:
                logger.debug("No active workspace at startup; palette will show on workspaceActivated")
        except Exception as exc:
            logger.warning(f"Unable to show palette at startup: {exc}")

    def stop(self) -> None:
        """Stop the controller and clean up resources."""
        logger.info("Stopping CADAgent controller")
        self._running = False

        # Stop all WebSocket connections
        try:
            for doc_id, info in list(self._sessions.items()):
                client = info.get('ws_client')
                if client:
                    client.stop()
                    logger.info(f"WebSocket client stopped for doc {doc_id}")
        except Exception as e:
            logger.error(f"Failed to stop WebSocket clients: {e}")

        # Stop palette UI
        try:
            self._palette_manager.stop()
            logger.info("Palette UI stopped")
        except Exception as e:
            logger.error(f"Failed to stop palette: {e}")

        # Unregister custom event
        try:
            if self._plan_approval_event:
                self._app.unregisterCustomEvent('CADAgentPlanApproval')
            if self._inbound_event_id:
                self._app.unregisterCustomEvent(self._inbound_event_id)
        except Exception as e:
            logger.error(f"Failed to unregister custom event: {e}")

        self._event_handlers.clear()
        logger.info("CADAgent controller stopped")

    # ------------------------------------------------------------------ Public API
    def submit_user_request(
        self,
        request_text: str,
        planning_mode: bool,
        include_visual_context: bool = False,
        model_name: str = "claude-sonnet-4.5",
        request_id: Optional[str] = None,
        image_data: Optional[str] = None,
        image_format: str = "png",
        attachments: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None
    ) -> None:
        """
        Submit a user request to the backend for processing.

        Args:
            request_text: Natural language CAD request
            planning_mode: Whether to generate a plan for approval first
            include_visual_context: Whether to attach a viewport snapshot for context
            model_name: AI model to use for processing
            request_id: Optional request ID for checkpoint correlation
            image_data: Optional base64-encoded image data for sketch translation
            image_format: Image format (png, jpg, jpeg)
            attachments: Optional attachment list from the palette
            reasoning_effort: User-selected reasoning effort level (off/low/medium/high/xhigh/on)
        """
        # For production Supabase-backed deployments, block command execution until
        # we have a valid JWT. This prevents opening unauthenticated WS sessions
        # that backend v2 rejects with "no token provided".
        token_required = bool(self._auth_client) and not self._auth_bypass
        if token_required and not self._get_user_token():
            self._palette_manager.send_message('auth_error', message='Please sign in before sending requests.')
            general_utils.show_message_box(
                "CADAgent Authentication Required",
                "Please sign in to CADAgent before sending requests.",
                adsk.core.MessageBoxIconTypes.WarningIconType
            )
            return

        client = self._get_active_ws_client()
        if not client or not client.is_connected():
            # Try to establish/recover the active document session just-in-time.
            try:
                self._activate_session_for_current_document()
            except Exception as e:
                logger.warning(f"Failed to activate document session before submit: {e}")
            client = self._get_active_ws_client()
        if not client or not client.is_connected():
            general_utils.show_message_box(
                "CADAgent Connection Error",
                "Backend connection is not available. Please check your network and backend status.",
                adsk.core.MessageBoxIconTypes.WarningIconType
            )
            return

        doc_id = self._active_doc_id

        visual_context_payload: Optional[Dict[str, Any]] = None
        selection_context: Optional[Dict[str, Any]] = None

        if include_visual_context and not planning_mode:
            visual_context_payload = self._capture_visual_context_snapshot()
            if visual_context_payload:
                logger.info(
                    "Attaching visual context snapshot (%dx%d)",
                    visual_context_payload.get("width", 0),
                    visual_context_payload.get("height", 0)
                )
            else:
                self._palette_manager.send_log('warning', 'Unable to capture visual context snapshot.', doc_id=doc_id)

        try:
            selection_context = extract_selection_context(self._app)
        except Exception as exc:
            logger.warning("Selection context extraction failed: %s", exc)
            selection_context = None

        # Extract spatial relationships (parallel faces, body distances)
        spatial_context = None
        try:
            spatial_context = spatial_analyzer.extract_spatial_context(self._app)
        except Exception as exc:
            logger.warning("Spatial context extraction failed: %s", exc)

        # Extract entity context (bodies, faces, edges) - always include for LLM awareness
        entity_context = self._extract_entity_context()

        payload: Dict[str, Any] = {
            "type": "planning_request" if planning_mode else "execute_request",
            "user_request": request_text,
            "timeline_state": self._collect_timeline_state(),
            "session_id": self.get_session_id(),
            "model_name": model_name,
            "reasoning_effort": reasoning_effort,
        }

        # Include request_id if provided for checkpoint correlation
        if request_id:
            payload["request_id"] = request_id

        if visual_context_payload:
            payload["visual_context"] = visual_context_payload
        if selection_context:
            entities = selection_context.get("entities") or []
            payload["selection_context"] = selection_context

            entity_count = selection_context.get("count") or len(entities)
            if entity_count:
                self._palette_manager.send_log(
                    'info',
                    f"Detected {entity_count} selected entit{'y' if entity_count == 1 else 'ies'} for context.",
                    doc_id=doc_id
                )
        if spatial_context:
            payload["spatial_context"] = spatial_context
        if entity_context:
            payload["entity_context"] = entity_context
        
        # Add image data if present for vision translation
        if image_data:
            payload["image_data"] = image_data
            payload["image_format"] = image_format
            logger.info(f"Including sketch image for vision translation (format={image_format})")
        if attachments:
            payload["attachments"] = attachments
            logger.info("Including %d attachment(s) in request", len(attachments))
        logger.info(f"Submitting {'planning' if planning_mode else 'execution'} request with model {model_name}")
        debug_payload = dict(payload)
        if debug_payload.get("image_data"):
            debug_payload["image_data"] = "<base64 omitted>"
        if "attachments" in debug_payload:
            redacted_attachments = []
            for attachment in debug_payload.get("attachments") or []:
                if isinstance(attachment, dict):
                    redacted = dict(attachment)
                    if "data" in redacted:
                        redacted["data"] = "<base64 omitted>"
                    redacted_attachments.append(redacted)
                else:
                    redacted_attachments.append("<non-object attachment>")
            debug_payload["attachments"] = redacted_attachments
        if visual_context_payload:
            redacted = {key: ("<base64 omitted>" if key == "data" else value)
                        for key, value in visual_context_payload.items()}
            debug_payload["visual_context"] = redacted
        if selection_context:
            entities = selection_context.get("entities") or []
            debug_payload["selection_context"] = {
                "count": selection_context.get("count"),
                "entities": [
                    {
                        "type": entity.get("type"),
                        "geometry_type": entity.get("geometry_type"),
                    }
                    for entity in entities
                ]
            }
        logger.debug(f"Request payload: {debug_payload}")

        try:
            client.send_json(payload)
        except Exception as e:
            logger.error(f"Failed to send request: {e}")
            general_utils.show_message_box(
                "CADAgent Error",
                f"Failed to send request to backend:\n{str(e)}",
                adsk.core.MessageBoxIconTypes.CriticalIconType
            )

    def is_connected(self, doc_id: Optional[str] = None) -> bool:
        """Check if the WebSocket connection is active for a specific document or the active one."""
        if doc_id is None:
            doc_id = self._active_doc_id

        if doc_id and doc_id in self._sessions:
            client = self._sessions[doc_id].get('ws_client')
            return bool(client and client.is_connected())
        return False

    def get_session_id(self) -> str:
        """Get the current session ID."""
        if self._active_doc_id and self._active_doc_id in self._sessions:
            return self._sessions[self._active_doc_id]['session_id']
        return ""

    def get_session_id_for_doc(self, doc_id: Optional[str] = None) -> str:
        """Get the session ID for a specific document or the active one."""
        if doc_id is None:
            doc_id = self._active_doc_id

        if doc_id and doc_id in self._sessions:
            return self._sessions[doc_id].get('session_id', '')
        return ""

    def get_active_doc_id(self) -> Optional[str]:
        """Expose the currently active document id."""
        return self._active_doc_id

    def handle_reconnect_request(self) -> None:
        """Handle manual reconnection request from the UI."""
        logger.info("Manual reconnection requested")

        try:
            # Force reconnection by ensuring session for current document
            doc = self._app.activeDocument if self._app else None
            if not doc:
                logger.warning("No active document available for reconnection")
                self._palette_manager.send_log('error', 'No active document found', doc_id=self._active_doc_id)
                return

            doc_id, doc_name = self._doc_identity(doc)
            # Update active doc_id to ensure messages reach the UI
            self._active_doc_id = doc_id

            logger.info(f"Forcing reconnection for document '{doc_name}' ({doc_id})")
            self._palette_manager.send_log('info', 'Reconnecting to backend...', doc_id=doc_id)

            info = self._ensure_session_for_doc(doc)
            session_id = info.get('session_id', '')
            client = info.get('ws_client')

            # Always tell the palette which doc/session we're on
            logger.info(f"[reconnect] nudging palette: document_switched {doc_name} ({doc_id}) sid={session_id}")
            self._palette_manager.send_document_switched(doc_id, doc_name, session_id)

            # CRITICAL: Force retry via send_message to ensure document_switched reaches UI
            # Reconnect creates a new session/doc_id; if this message is dropped, all subsequent
            # responses carry the new doc_id and are deferred indefinitely by the frontend.
            # Queue via send_message (even though send_document_switched already sent) and force flush.
            logger.info(f"[reconnect] queueing document_switched retry for bulletproof delivery")
            self._palette_manager.send_message('document_switched', doc_id=doc_id, doc_name=doc_name, session_id=session_id)
            self._palette_manager._fire_flush_event()

            # Then send the latest status (may still be "connecting")
            self._palette_manager.send_connection_status(doc_id=doc_id)

            deadline = time.time() + 3.0
            while client and not client.is_connected() and time.time() < deadline:
                time.sleep(0.1)

            if self.is_connected():
                self._palette_manager.send_log('success', 'Reconnected successfully!', doc_id=doc_id)
            else:
                self._palette_manager.send_log('warning', 'Reconnection still in progress...', doc_id=doc_id)
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            self._palette_manager.send_log('error', f'Reconnection failed: {str(e)}', doc_id=self._active_doc_id)
            self._palette_manager.send_connection_status(doc_id=self._active_doc_id)

    # ------------------------------------------------------------------ Message Handling
    def _enqueue_message(self, doc_id: str, message: Dict[str, Any]) -> None:
        """Thread-safe message enqueueing from WebSocket thread."""
        # INSTRUMENTATION: Track message arrival in CADAgent enqueue
        msg_type = message.get("type", "<no type>")
        if msg_type not in _QUIET_MESSAGE_TYPES:
            logger.debug("[CADAGENT_ENQUEUE] Message enqueued: doc_id=%s, type=%s", doc_id, msg_type)
            _fusion_probe(f"[CADAGENT_ENQUEUE] doc_id={doc_id}, type={msg_type}")

        self._incoming_messages.put((doc_id, message))
        logger.debug(f"Message enqueued for doc {doc_id}, queue size: {self._incoming_messages.qsize()}")
        if self._running and self._app:
            try:
                self._app.fireCustomEvent(self._inbound_event_id, "")
            except Exception as exc:
                logger.exception("Failed to signal inbound message event: %s", exc)

    def process_pending_messages(self) -> None:
        """Process all pending messages on the Fusion main thread."""
        current_thread = threading.current_thread()
        logger.debug(
            "Processing inbound queue on thread %s (name=%s)",
            current_thread.ident,
            current_thread.name,
        )
        while not self._incoming_messages.empty():
            doc_id, message = self._incoming_messages.get()
            try:
                self._handle_message(doc_id, message)
            except Exception as e:
                logger.exception(f"Failed to handle message: {message}")
                general_utils.log_error("Message handling error", e)

    def _handle_message(self, doc_id: str, message: Dict[str, Any]) -> None:
        """Route incoming messages to appropriate handlers."""
        message_type = message.get("type")
        # INSTRUMENTATION: Track message handling in CADAgent
        if message_type not in _QUIET_MESSAGE_TYPES:
            logger.debug("[CADAGENT_HANDLE] Handling message: doc_id=%s, type=%s", doc_id, message_type)
            _fusion_probe(f"[CADAGENT_HANDLE] doc_id={doc_id}, type={message_type}")
        logger.debug(f"Handling message type: {message_type}")

        if message_type == "execute_code":
            self._handle_execute_code(doc_id, message)
        elif message_type == "llm_message":
            text = message.get("message", "")
            message_format = message.get("format")
            # Force run-scope so the palette starts/continues the run log timeline even after reconnect.
            self._palette_manager.send_log('agent', text, doc_id=doc_id, scope='run', message_format=message_format)
        elif message_type == "plan_chunk":
            # Accumulate plan chunks silently - don't display individually
            chunk = message.get("content", "")
            doc_chunks = self._plan_chunks.setdefault(doc_id, [])
            if len(doc_chunks) == 0:
                # First chunk - show "Generating plan..." indicator
                self._palette_manager.send_progress("Generating plan...", doc_id=doc_id)
            doc_chunks.append(chunk)
        elif message_type == "reasoning_chunk":
            chunk = message.get("content", "")
            if chunk:
                self._palette_manager.send_message(
                    'reasoning_chunk',
                    doc_id=doc_id,
                    content=chunk
                )
        elif message_type == "plan_complete":
            # Hide progress and fire custom event to show approval dialog on main thread
            self._palette_manager.hide_progress(doc_id=doc_id)
            full_plan = message.get("full_plan", "")
            display_plan = message.get("display_plan") or full_plan
            display_plan_plain = message.get("display_plan_plain") or full_plan

            self._pending_plan_full[doc_id] = full_plan

            # Show plan in palette first
            if full_plan:
                plan_log_message = f"📋 Plan generated:\n\n{full_plan}"
                self._palette_manager.send_log(
                    'agent',
                    plan_log_message,
                    doc_id=doc_id,
                    message_format='markdown'
                )
            else:
                self._palette_manager.send_log('agent', '📋 Plan generated.', doc_id=doc_id)
            self._palette_manager.send_log('info', 'Please review the plan in the dialog that will appear...', doc_id=doc_id)

            # Fire custom event with plan data (handler runs on main thread)
            event_payload = {"plan_text": display_plan_plain, "doc_id": doc_id}
            self._app.fireCustomEvent('CADAgentPlanApproval', json.dumps(event_payload))
            self._plan_chunks.pop(doc_id, None)
        elif message_type == "completed":
            # Don't send any message - frontend will update run summary to "Designed"
            self._palette_manager.send_completed("", doc_id=doc_id)
        elif message_type == "error":
            error_msg = message.get("message", "An error occurred")
            details = message.get("details", "")
            full_error = f"{error_msg}\n{details}" if details else error_msg
            self._palette_manager.send_error(full_error, doc_id=doc_id)
        elif message_type == "log":
            self._palette_manager.send_log(
                message.get("level", "info"),
                message.get("message", ""),
                doc_id=doc_id,
                scope=message.get("scope", "auto"),
                message_format=message.get("format"),
            )
        elif message_type == "cancelled":
            cancel_msg = message.get("message", "Request cancelled")
            self._palette_manager.send_message('cancelled', doc_id=doc_id, message=cancel_msg)
        elif message_type == "body_operation":
            self._handle_body_operation(doc_id, message)
        elif message_type == "edge_operation":
            self._handle_edge_operation(doc_id, message)
        elif message_type == "face_operation":
            self._handle_face_operation(doc_id, message)
        elif message_type == "feature_operation":
            self._handle_feature_operation(doc_id, message)
        elif message_type == "revert_timeline":
            self._handle_revert_timeline(doc_id, message)
        elif message_type == "checkpoint_created":
            # Forward checkpoint notification to palette for UI association
            message_id = message.get("message_id")
            request_id = message.get("request_id")  # Extract request_id for correlation
            if message_id:
                # Include request_id if present for exact checkpoint-message matching
                kwargs = {"message_id": message_id}
                if request_id:
                    kwargs["request_id"] = request_id
                self._palette_manager.send_message('checkpoint_created', doc_id=doc_id, **kwargs)
        elif message_type == "operation_checkpoint_created":
            checkpoint = message.get("checkpoint") or {}
            if checkpoint:
                self._palette_manager.send_message(
                    'operation_checkpoint_created',
                    doc_id=doc_id,
                    checkpoint=checkpoint,
                )
        elif message_type == "revert_applied":
            # Forward revert confirmation to palette for UI cleanup
            self._palette_manager.send_message(
                'revert_applied',
                doc_id=doc_id,
                message_id=message.get("message_id"),
                conversation_index=message.get("conversation_index"),
                conversation_length=message.get("conversation_length"),
                include_message=message.get("include_message", True),
            )
        elif message_type == "operation_resume_applied":
            self._palette_manager.send_message(
                'operation_resume_applied',
                doc_id=doc_id,
                checkpoint_id=message.get("checkpoint_id") or message.get("operation_checkpoint_id"),
                conversation_index=message.get("conversation_index"),
                conversation_length=message.get("conversation_length"),
                tool_name=message.get("tool_name"),
                display_label=message.get("display_label"),
            )
        elif message_type == "feature_snapshot_request":
            self._handle_feature_snapshot_request(doc_id, message)
        elif message_type == "request_entity_context":
            self._handle_entity_context_request(doc_id, message)
        elif message_type == "question_tree_generated":
            # Forward design-exploration question tree to the palette UI
            self._palette_manager.send_message(
                'question_tree_generated',
                doc_id=doc_id,
                data=message.get("data", {}),
            )
        elif message_type == "designs_proposed":
            # Forward design proposal cards to the palette UI
            self._palette_manager.send_message(
                'designs_proposed',
                doc_id=doc_id,
                data=message.get("data", {}),
            )
        elif message_type == "build_plan_generated":
            # Forward build plan to the palette UI
            self._palette_manager.send_message(
                'build_plan_generated',
                doc_id=doc_id,
                data=message.get("data", {}),
            )
        elif message_type == "build_step_completed":
            # Forward build step progress to the palette UI
            self._palette_manager.send_message(
                'build_step_completed',
                doc_id=doc_id,
                data=message.get("data", {}),
            )
        elif message_type == "build_plan_completed":
            # Forward build plan completion to the palette UI
            self._palette_manager.send_message(
                'build_plan_completed',
                doc_id=doc_id,
                data=message.get("data", {}),
            )
        elif message_type == "authentication_ack":
            # Forward authentication acknowledgment to palette with API keys status
            self._palette_manager.send_message(
                'authentication_ack',
                doc_id=doc_id,
                authenticated=message.get("authenticated", False),
                needs_api_keys=message.get("needs_api_keys", True),
                has_anthropic=message.get("has_anthropic", False),
                has_openai=message.get("has_openai", False),
                has_google=message.get("has_google", False),
            )
            # Auto-heal: if backend says keys are missing but we have them locally, resend immediately
            try:
                needs_keys = message.get("needs_api_keys")
                logger.info(
                    "[api_keys] authentication_ack received (needs_api_keys=%s, has_anthropic=%s, has_openai=%s)",
                    needs_keys,
                    message.get("has_anthropic"),
                    message.get("has_openai"),
                )
                if needs_keys:
                    self._send_api_keys_to_backend(doc_id, reason="auth_ack_needs_keys")
                # Always refresh UI with local view of stored keys
                try:
                    status = self.get_api_keys_status()
                    self._palette_manager.send_message('api_keys_status', status=status)
                except Exception as status_err:
                    logger.debug(f"[api_keys] Failed to send status after auth_ack: {status_err}")
            except Exception as e:
                logger.warning(f"[api_keys] Auto-push on auth_ack failed: {e}")
        elif message_type == "api_keys_updated":
            # Backend confirms it received updated keys; refresh UI status
            try:
                status = self.get_api_keys_status()
                self._palette_manager.send_message('api_keys_status', status=status)
                self._palette_manager.send_message('api_keys_saved', success=True, status=status)
                logger.info("[api_keys] api_keys_updated received; UI status refreshed")
            except Exception as e:
                logger.warning(f"[api_keys] Failed to handle api_keys_updated: {e}")
        else:
            logger.warning(f"Unhandled message type: {message_type}")

    def _handle_execute_code(self, doc_id: str, message: Dict[str, Any]) -> None:
        """Execute code received from the backend."""
        code = message.get("code", "")
        operation = message.get("operation", "unknown")
        tool_use_id = message.get("tool_use_id")

        logger.debug(
            "Executing code message on thread %s (name=%s)",
            threading.get_ident(),
            threading.current_thread().name,
        )

        if not code:
            logger.error("Received execute_code message without code")
            return

        logger.info(f"Executing operation: {operation}")
        logger.debug(f"Code to execute:\n{code}")

        session_info = self._get_session_info_by_doc(doc_id)
        if not session_info:
            logger.warning(f"Session info missing for document {doc_id}; dropping execute_code request")
            return

        session_id = session_info.get("session_id")
        client = session_info.get("ws_client")
        
        # Always force fresh design resolution before code execution to avoid stale references
        # This is critical because cached design references can become invalid after document switches
        design = self._resolve_design_reference(session_info.get('document'))
        if design:
            session_info['design'] = design

        if not design:
            logger.error("Unable to resolve design context for document %s", doc_id)
            self._palette_manager.send_log('error', 'Unable to resolve design context for this document.', doc_id=doc_id)
            error_payload = {
                "type": "execution_result",
                "session_id": session_id,
                "operation": operation,
                "tool_use_id": tool_use_id,
                "success": False,
                "error": "Design context unavailable",
            }
            if client:
                client.send_json(error_payload)
            return

        # Send to palette
        description = message.get("description", "")
        self._palette_manager.send_message(
            'execute_code',
            doc_id=doc_id,
            operation=operation,
            description=description,
        )
        self._palette_manager.send_progress(f"Executing {operation}...", doc_id=doc_id)

        result = self._code_executor.execute_code(doc_id, design, code, operation)

        payload = {
            "type": "execution_result",
            "session_id": session_id,
            "operation": operation,
            "tool_use_id": tool_use_id,
            "success": result.get("success", False),
            "message": result.get("message"),
            "error": result.get("error"),
        }

        # Pass through any additional result fields (orientation, created_sketches, etc.)
        # without dropping the core status keys above. This ensures Fusion data produced by
        # create_sketch/extrude/etc. reaches the backend unchanged for enrichment.
        for key, value in result.items():
            if key in {"success", "message", "error", "error_type", "traceback"}:
                continue
            if key not in payload:
                payload[key] = value

        logger.info(f"Operation {operation} {'succeeded' if result.get('success') else 'failed'}")

        if result.get("success"):
            self._palette_manager.send_log('success', f"✓ {operation} completed", doc_id=doc_id)
        else:
            self._palette_manager.send_log('error', f"✗ {operation} failed: {result.get('error', 'Unknown error')}", doc_id=doc_id)

        if client:
            client.send_json(payload)
        else:
            logger.warning(f"No active WebSocket client for document {doc_id}; execution result not sent")

    def _send_selection_feedback(
        self,
        doc_id: Optional[str],
        geometry: str,
        operation: str,
        *,
        success: bool,
        message: str,
        **extra: Any,
    ) -> None:
        """Send structured selection feedback to the palette UI."""
        payload: Dict[str, Any] = {
            "geometry": geometry,
            "operation": operation,
            "success": bool(success),
            "message": message or "",
            "timestamp": time.time(),
        }

        for key, value in extra.items():
            if value is not None:
                payload[key] = value

        try:
            self._palette_manager.send_message('selection_feedback', doc_id=doc_id, **payload)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to send selection feedback to palette", exc_info=True)

    @staticmethod
    def _build_edge_preview(edges: List[Dict[str, Any]], limit: int = SELECTION_PREVIEW_LIMIT) -> List[Dict[str, Any]]:
        """Return a trimmed list of edge metadata suitable for UI display."""
        preview: List[Dict[str, Any]] = []
        for edge in edges[:max(0, limit)]:
            preview.append({
                "id": edge.get("id"),
                "body": edge.get("body_name") or f"Body {edge.get('body_index', '?')}",
                "geometry_type": edge.get("geometry_type"),
                "length": edge.get("length"),
                "start": edge.get("start_coords"),
                "end": edge.get("end_coords"),
                "entity_token": edge.get("entity_token"),
            })
        return preview

    @staticmethod
    def _build_body_preview(bodies: List[Dict[str, Any]], limit: int = SELECTION_PREVIEW_LIMIT) -> List[Dict[str, Any]]:
        """Return a trimmed list of body metadata suitable for UI display."""
        preview: List[Dict[str, Any]] = []
        for body in bodies[:max(0, limit)]:
            preview.append({
                "id": body.get("id"),
                "name": body.get("name") or f"Body {body.get('body_index', '?')}",
                "component": body.get("component"),
                "volume": body.get("volume"),
                "volume_units": body.get("volume_units"),
                "area": body.get("area"),
                "area_units": body.get("area_units"),
                "entity_token": body.get("entity_token"),
            })
        return preview

    @staticmethod
    def _build_face_preview(faces: List[Dict[str, Any]], limit: int = SELECTION_PREVIEW_LIMIT) -> List[Dict[str, Any]]:
        """Return a trimmed list of face metadata suitable for UI display."""
        preview: List[Dict[str, Any]] = []
        for face in faces[:max(0, limit)]:
            preview.append({
                "id": face.get("id"),
                "body": face.get("body_name") or f"Body {face.get('body_index', '?')}",
                "geometry_type": face.get("geometry_type"),
                "area": face.get("area"),
                "centroid": face.get("centroid"),
                "edge_count": len(face.get("edges") or []),
                "entity_token": face.get("entity_token"),
            })
        return preview

    def _handle_body_operation(self, doc_id: str, message: Dict[str, Any]) -> None:
        """Execute body-specific operations requested by the backend."""
        operation = message.get("operation")
        tool_use_id = message.get("tool_use_id")
        params = message.get("parameters") or {}
        description = message.get("description", "")

        logger.info("Handling body operation '%s' (tool_use_id=%s)", operation, tool_use_id)

        session_info = self._get_session_info_by_doc(doc_id)
        if not session_info:
            logger.warning("Body operation requested for unknown document %s", doc_id)
            return

        session_id = session_info.get("session_id")
        client: Optional[FusionWebSocketClient] = session_info.get("ws_client")

        # Send description to UI (matching execute_code pattern)
        if description:
            self._palette_manager.send_message(
                'body_operation',
                doc_id=doc_id,
                operation=operation,
                description=description,
            )

        payload: Dict[str, Any] = {
            "type": "execution_result",
            "session_id": session_id,
            "operation": operation or "body_operation",
            "tool_use_id": tool_use_id,
        }

        success = False
        message_text = "Unknown body operation."

        try:
            if not operation:
                raise body_tools.BodyOperationError("Body operation type was not specified.")

            if operation == "list_bodies":
                bodies, units = body_tools.list_bodies(self._app)
                success = True
                message_text = f"Found {len(bodies)} body(s)."
                payload.update({
                    "success": success,
                    "message": message_text,
                    "bodies": bodies,
                    "units": units,
                })
                self._palette_manager.send_log('info', message_text, doc_id=doc_id)
                preview = self._build_body_preview(bodies)
                self._send_selection_feedback(
                    doc_id,
                    "body",
                    operation,
                    success=success,
                    message=message_text,
                    counts={"total": len(bodies)},
                    units=units,
                    preview=preview,
                    truncated=len(bodies) > len(preview),
                )

            elif operation == "select_bodies":
                incoming_tokens = message.get("entity_tokens", params.get("entity_tokens"))
                if incoming_tokens is None:
                    incoming_tokens = params.get("entity_tokens") or []

                if isinstance(incoming_tokens, list):
                    tokens = incoming_tokens
                elif isinstance(incoming_tokens, tuple):
                    tokens = list(incoming_tokens)
                elif isinstance(incoming_tokens, set):
                    tokens = list(incoming_tokens)
                elif incoming_tokens:
                    tokens = [incoming_tokens]
                else:
                    tokens = []

                clear_existing = bool(message.get("clear_existing", params.get("clear_existing", True)))
                result = body_tools.select_bodies(self._app, tokens, clear_existing=clear_existing)
                success = True
                missing = result.get("missing_tokens", [])
                message_text = f"Selected {result.get('selected_count', 0)} body(s)."
                if missing:
                    message_text += f" {len(missing)} token(s) not found."
                log_level = 'success' if not missing else 'warning'
                payload.update({
                    "success": success,
                    "message": message_text,
                    "selected_count": result.get("selected_count", 0),
                    "missing_tokens": missing,
                    "cleared_existing": result.get("cleared_existing", clear_existing),
                })
                self._palette_manager.send_log(log_level, message_text, doc_id=doc_id)
                self._send_selection_feedback(
                    doc_id,
                    "body",
                    operation,
                    success=success,
                    message=message_text,
                    counts={
                        "selected": result.get("selected_count", 0),
                        "missing": len(missing),
                        "provided": len(tokens),
                    },
                    missing_tokens=missing,
                    cleared_existing=result.get("cleared_existing", clear_existing),
                )

            elif operation == "clear_body_selection":
                result = body_tools.clear_body_selection(self._app)
                success = True
                cleared = result.get("cleared_count", 0)
                message_text = f"Cleared {cleared} selection(s)."
                payload.update({
                    "success": success,
                    "message": message_text,
                    "cleared_count": cleared,
                })
                self._palette_manager.send_log('info', message_text, doc_id=doc_id)
                self._send_selection_feedback(
                    doc_id,
                    "body",
                    operation,
                    success=success,
                    message=message_text,
                    counts={"cleared": cleared},
                )

            else:
                raise body_tools.BodyOperationError(f"Unsupported body operation '{operation}'.")

        except body_tools.BodyOperationError as exc:
            message_text = str(exc)
            payload.update({
                "success": False,
                "error": message_text,
            })
            self._palette_manager.send_log('error', message_text, doc_id=doc_id)
            if operation:
                self._send_selection_feedback(
                    doc_id,
                    "body",
                    operation,
                    success=False,
                    message=message_text,
                )
        except Exception as exc:  # pragma: no cover - defensive
            message_text = f"Unexpected error: {exc}"
            logger.exception("Unexpected error while handling body operation.")
            payload.update({
                "success": False,
                "error": message_text,
            })
            self._palette_manager.send_log('error', message_text, doc_id=doc_id)
            if operation:
                self._send_selection_feedback(
                    doc_id,
                    "body",
                    operation,
                    success=False,
                    message=message_text,
                )
        finally:
            payload.setdefault("success", success)
            payload.setdefault("message", message_text)

            if client:
                client.send_json(payload)
            else:
                logger.warning("Body operation result not sent; no active WebSocket client.")

    def _handle_edge_operation(self, doc_id: str, message: Dict[str, Any]) -> None:
        """Execute edge-specific operations requested by the backend."""
        operation = message.get("operation")
        tool_use_id = message.get("tool_use_id")
        params = message.get("parameters") or {}
        description = message.get("description", "")

        logger.info("Handling edge operation '%s' (tool_use_id=%s)", operation, tool_use_id)

        session_info = self._get_session_info_by_doc(doc_id)
        if not session_info:
            logger.warning("Edge operation requested for unknown document %s", doc_id)
            return

        session_id = session_info.get("session_id")
        client: Optional[FusionWebSocketClient] = session_info.get("ws_client")

        # Send description to UI (matching execute_code pattern)
        if description:
            self._palette_manager.send_message(
                'edge_operation',
                doc_id=doc_id,
                operation=operation,
                description=description,
            )

        payload: Dict[str, Any] = {
            "type": "execution_result",
            "session_id": session_id,
            "operation": operation or "edge_operation",
            "tool_use_id": tool_use_id,
        }

        success = False
        message_text = "Unknown edge operation."

        try:
            if not operation:
                raise edge_tools.EdgeOperationError("Edge operation type was not specified.")

            if operation == "list_edges":
                edges, units = edge_tools.list_edges(self._app)
                success = True
                message_text = f"Found {len(edges)} edge(s)."
                payload.update({
                    "success": success,
                    "message": message_text,
                    "edges": edges,
                    "units": units,
                })
                self._palette_manager.send_log('info', message_text, doc_id=doc_id)
                preview = self._build_edge_preview(edges)
                self._send_selection_feedback(
                    doc_id,
                    "edge",
                    operation,
                    success=success,
                    message=message_text,
                    counts={"total": len(edges)},
                    units={"length": units},
                    preview=preview,
                    truncated=len(edges) > len(preview),
                )

            elif operation == "select_edges":
                incoming_tokens = message.get("entity_tokens", params.get("entity_tokens"))
                if incoming_tokens is None:
                    incoming_tokens = params.get("entity_tokens") or []

                if isinstance(incoming_tokens, list):
                    tokens = incoming_tokens
                elif isinstance(incoming_tokens, tuple):
                    tokens = list(incoming_tokens)
                elif isinstance(incoming_tokens, set):
                    tokens = list(incoming_tokens)
                elif incoming_tokens:
                    tokens = [incoming_tokens]
                else:
                    tokens = []
                clear_existing = bool(message.get("clear_existing", params.get("clear_existing", True)))
                result = edge_tools.select_edges(self._app, tokens, clear_existing=clear_existing)
                success = True
                missing = result.get("missing_tokens", [])
                message_text = f"Selected {result.get('selected_count', 0)} edge(s)."
                if missing:
                    message_text += f" {len(missing)} token(s) not found."
                log_level = 'success' if not missing else 'warning'
                payload.update({
                    "success": success,
                    "message": message_text,
                    "selected_count": result.get("selected_count", 0),
                    "missing_tokens": missing,
                    "cleared_existing": result.get("cleared_existing", clear_existing),
                })
                self._palette_manager.send_log(log_level, message_text, doc_id=doc_id)
                self._send_selection_feedback(
                    doc_id,
                    "edge",
                    operation,
                    success=success,
                    message=message_text,
                    counts={
                        "selected": result.get("selected_count", 0),
                        "missing": len(missing),
                        "provided": len(tokens),
                    },
                    missing_tokens=missing,
                    cleared_existing=result.get("cleared_existing", clear_existing),
                )

            elif operation == "clear_edge_selection":
                result = edge_tools.clear_edge_selection(self._app)
                success = True
                cleared = result.get("cleared_count", 0)
                message_text = f"Cleared {cleared} selection(s)."
                payload.update({
                    "success": success,
                    "message": message_text,
                    "cleared_count": cleared,
                })
                self._palette_manager.send_log('info', message_text, doc_id=doc_id)
                self._send_selection_feedback(
                    doc_id,
                    "edge",
                    operation,
                    success=success,
                    message=message_text,
                    counts={"cleared": cleared},
                )

            else:
                raise edge_tools.EdgeOperationError(f"Unsupported edge operation '{operation}'.")

        except edge_tools.EdgeOperationError as exc:
            message_text = str(exc)
            payload.update({
                "success": False,
                "error": message_text,
            })
            self._palette_manager.send_log('error', message_text, doc_id=doc_id)
            if operation:
                self._send_selection_feedback(
                    doc_id,
                    "edge",
                    operation,
                    success=False,
                    message=message_text,
                )
        except Exception as exc:  # pragma: no cover - defensive
            message_text = f"Unexpected error: {exc}"
            logger.exception("Unexpected error while handling edge operation.")
            payload.update({
                "success": False,
                "error": message_text,
            })
            self._palette_manager.send_log('error', message_text, doc_id=doc_id)
            if operation:
                self._send_selection_feedback(
                    doc_id,
                    "edge",
                    operation,
                    success=False,
                    message=message_text,
                )
        finally:
            # Ensure payload has success flag and message
            payload.setdefault("success", success)
            payload.setdefault("message", message_text)

            if client:
                client.send_json(payload)
            else:
                logger.warning("Edge operation result not sent; no active WebSocket client.")

    def _handle_face_operation(self, doc_id: str, message: Dict[str, Any]) -> None:
        """Execute face-specific operations requested by the backend."""
        operation = message.get("operation")
        tool_use_id = message.get("tool_use_id")
        params = message.get("parameters") or {}
        description = message.get("description", "")

        logger.info("Handling face operation '%s' (tool_use_id=%s)", operation, tool_use_id)

        session_info = self._get_session_info_by_doc(doc_id)
        if not session_info:
            logger.warning("Face operation requested for unknown document %s", doc_id)
            return

        session_id = session_info.get("session_id")
        client: Optional[FusionWebSocketClient] = session_info.get("ws_client")

        # Send description to UI (matching execute_code pattern)
        if description:
            self._palette_manager.send_message(
                'face_operation',
                doc_id=doc_id,
                operation=operation,
                description=description,
            )

        payload: Dict[str, Any] = {
            "type": "execution_result",
            "session_id": session_id,
            "operation": operation or "face_operation",
            "tool_use_id": tool_use_id,
        }

        success = False
        message_text = "Unknown face operation."

        try:
            if not operation:
                raise face_tools.FaceOperationError("Face operation type was not specified.")

            if operation == "list_faces":
                faces, units = face_tools.list_faces(self._app)
                success = True
                message_text = f"Found {len(faces)} face(s)."
                payload.update({
                    "success": success,
                    "message": message_text,
                    "faces": faces,
                    "units": units,
                })
                self._palette_manager.send_log('info', message_text, doc_id=doc_id)
                preview = self._build_face_preview(faces)
                self._send_selection_feedback(
                    doc_id,
                    "face",
                    operation,
                    success=success,
                    message=message_text,
                    counts={"total": len(faces)},
                    units=units,
                    preview=preview,
                    truncated=len(faces) > len(preview),
                )

            elif operation == "select_faces":
                incoming_tokens = message.get("entity_tokens", params.get("entity_tokens"))
                if incoming_tokens is None:
                    incoming_tokens = params.get("entity_tokens") or []

                if isinstance(incoming_tokens, list):
                    tokens = incoming_tokens
                elif isinstance(incoming_tokens, tuple):
                    tokens = list(incoming_tokens)
                elif isinstance(incoming_tokens, set):
                    tokens = list(incoming_tokens)
                elif incoming_tokens:
                    tokens = [incoming_tokens]
                else:
                    tokens = []

                clear_existing = bool(message.get("clear_existing", params.get("clear_existing", True)))
                result = face_tools.select_faces(self._app, tokens, clear_existing=clear_existing)
                success = True
                missing = result.get("missing_tokens", [])
                message_text = f"Selected {result.get('selected_count', 0)} face(s)."
                if missing:
                    message_text += f" {len(missing)} token(s) not found."
                log_level = 'success' if not missing else 'warning'
                payload.update({
                    "success": success,
                    "message": message_text,
                    "selected_count": result.get("selected_count", 0),
                    "missing_tokens": missing,
                    "cleared_existing": result.get("cleared_existing", clear_existing),
                })
                self._palette_manager.send_log(log_level, message_text, doc_id=doc_id)
                self._send_selection_feedback(
                    doc_id,
                    "face",
                    operation,
                    success=success,
                    message=message_text,
                    counts={
                        "selected": result.get("selected_count", 0),
                        "missing": len(missing),
                        "provided": len(tokens),
                    },
                    missing_tokens=missing,
                    cleared_existing=result.get("cleared_existing", clear_existing),
                )

            elif operation == "clear_face_selection":
                result = face_tools.clear_face_selection(self._app)
                success = True
                cleared = result.get("cleared_count", 0)
                message_text = f"Cleared {cleared} selection(s)."
                payload.update({
                    "success": success,
                    "message": message_text,
                    "cleared_count": cleared,
                })
                self._palette_manager.send_log('info', message_text, doc_id=doc_id)
                self._send_selection_feedback(
                    doc_id,
                    "face",
                    operation,
                    success=success,
                    message=message_text,
                    counts={"cleared": cleared},
                )

            else:
                raise face_tools.FaceOperationError(f"Unsupported face operation '{operation}'.")

        except face_tools.FaceOperationError as exc:
            message_text = str(exc)
            payload.update({
                "success": False,
                "error": message_text,
            })
            self._palette_manager.send_log('error', message_text, doc_id=doc_id)
            if operation:
                self._send_selection_feedback(
                    doc_id,
                    "face",
                    operation,
                    success=False,
                    message=message_text,
                )
        except Exception as exc:  # pragma: no cover - defensive
            message_text = f"Unexpected error: {exc}"
            logger.exception("Unexpected error while handling face operation.")
            payload.update({
                "success": False,
                "error": message_text,
            })
            self._palette_manager.send_log('error', message_text, doc_id=doc_id)
            if operation:
                self._send_selection_feedback(
                    doc_id,
                    "face",
                    operation,
                    success=False,
                    message=message_text,
                )
        finally:
            payload.setdefault("success", success)
            payload.setdefault("message", message_text)

            if client:
                client.send_json(payload)
            else:
                logger.warning("Face operation result not sent; no active WebSocket client.")

    def _handle_feature_operation(self, doc_id: str, message: Dict[str, Any]) -> None:
        """Execute feature-specific operations (fillet, chamfer) requested by the backend."""
        operation = message.get("operation")
        tool_use_id = message.get("tool_use_id")
        params = message.get("parameters") or {}
        description = message.get("description", "")

        logger.info("Handling feature operation '%s' (tool_use_id=%s)", operation, tool_use_id)

        session_info = self._get_session_info_by_doc(doc_id)
        if not session_info:
            logger.warning("Feature operation requested for unknown document %s", doc_id)
            return

        session_id = session_info.get("session_id")
        client: Optional[FusionWebSocketClient] = session_info.get("ws_client")

        # Send description to UI (matching execute_code pattern)
        if description:
            self._palette_manager.send_message(
                'feature_operation',
                doc_id=doc_id,
                operation=operation,
                description=description,
            )

        payload: Dict[str, Any] = {
            "type": "execution_result",
            "session_id": session_id,
            "operation": operation or "feature_operation",
            "tool_use_id": tool_use_id,
        }

        success = False
        message_text = "Unknown feature operation."

        try:
            if not operation:
                raise feature_tools.FeatureOperationError("Feature operation type was not specified.")

            if operation == "adjust_feature_parameters":
                feature_token = message.get("feature_token") or params.get("feature_token")
                edit_params = message.get("parameters") or params
                expected_name = message.get("expected_name") or params.get("expected_name") or ""
                expected_timeline_index = message.get("expected_timeline_index")
                if expected_timeline_index is None:
                    expected_timeline_index = params.get("expected_timeline_index")

                if not feature_token:
                    raise feature_tools.FeatureOperationError("adjust_feature_parameters requires feature_token.")
                if not isinstance(edit_params, dict) or not edit_params:
                    raise feature_tools.FeatureOperationError("adjust_feature_parameters requires parameters.")

                result = feature_tools.adjust_feature_parameters(
                    self._app,
                    str(feature_token),
                    dict(edit_params),
                    str(expected_name),
                    int(expected_timeline_index) if expected_timeline_index is not None else None,
                )
                success = True
                message_text = result.get("message", "Feature parameters adjusted successfully.")
                payload.update({
                    "success": success,
                    "message": message_text,
                    "feature_type": result.get("feature_type"),
                    "feature_name": result.get("feature_name"),
                    "timeline_index": result.get("timeline_index"),
                    "changed_parameters": result.get("changed_parameters", {}),
                })
                self._palette_manager.send_log('success', message_text, doc_id=doc_id)

            elif operation == "apply_fillet":
                # Prefer top-level message keys over nested params to match backend payload structure
                entity_tokens = message.get("entity_tokens") or params.get("entity_tokens") or []
                radius = message.get("radius") if message.get("radius") is not None else params.get("radius")
                radius_unit = message.get("radius_unit") or params.get("radius_unit") or "mm"
                include_tangent_edges = message.get("include_tangent_edges", params.get("include_tangent_edges", True))

                if not entity_tokens:
                    raise feature_tools.FeatureOperationError("apply_fillet requires entity_tokens.")
                if radius is None:
                    raise feature_tools.FeatureOperationError("apply_fillet requires radius parameter.")

                result = feature_tools.apply_fillet(
                    self._app,
                    entity_tokens,
                    float(radius),
                    str(radius_unit),
                    bool(include_tangent_edges)
                )
                success = True
                message_text = result.get("message", "Fillet applied successfully.")
                payload.update({
                    "success": success,
                    "message": message_text,
                    "edge_count": result.get("edge_count", 0),
                    "radius": result.get("radius"),
                    "radius_unit": result.get("radius_unit"),
                    "missing_tokens": result.get("missing_tokens", []),
                })
                log_level = 'success' if not result.get("missing_tokens") else 'warning'
                self._palette_manager.send_log(log_level, message_text, doc_id=doc_id)

            elif operation == "apply_chamfer":
                # Prefer top-level message keys over nested params to match backend payload structure
                entity_tokens = message.get("entity_tokens") or params.get("entity_tokens") or []
                distance = message.get("distance") if message.get("distance") is not None else params.get("distance")
                distance_unit = message.get("distance_unit") or params.get("distance_unit") or "mm"
                include_tangent_edges = message.get("include_tangent_edges", params.get("include_tangent_edges", True))

                if not entity_tokens:
                    raise feature_tools.FeatureOperationError("apply_chamfer requires entity_tokens.")
                if distance is None:
                    raise feature_tools.FeatureOperationError("apply_chamfer requires distance parameter.")

                result = feature_tools.apply_chamfer(
                    self._app,
                    entity_tokens,
                    float(distance),
                    str(distance_unit),
                    bool(include_tangent_edges)
                )
                success = True
                message_text = result.get("message", "Chamfer applied successfully.")
                payload.update({
                    "success": success,
                    "message": message_text,
                    "edge_count": result.get("edge_count", 0),
                    "distance": result.get("distance"),
                    "distance_unit": result.get("distance_unit"),
                    "missing_tokens": result.get("missing_tokens", []),
                })
                log_level = 'success' if not result.get("missing_tokens") else 'warning'
                self._palette_manager.send_log(log_level, message_text, doc_id=doc_id)

            elif operation == "create_shell":
                # Prefer top-level message keys over nested params to match backend payload structure
                entity_tokens = message.get("entity_tokens") or params.get("entity_tokens") or []
                inside_thickness = message.get("inside_thickness") if message.get("inside_thickness") is not None else params.get("inside_thickness")
                outside_thickness = message.get("outside_thickness") if message.get("outside_thickness") is not None else params.get("outside_thickness")
                thickness_unit = message.get("thickness_unit") or params.get("thickness_unit") or "mm"
                is_tangent_chain = message.get("is_tangent_chain", params.get("is_tangent_chain", True))
                shell_type = message.get("shell_type") or params.get("shell_type") or "sharp"
                feature_name = message.get("feature_name") or params.get("feature_name") or ""

                if not entity_tokens:
                    raise feature_tools.FeatureOperationError("create_shell requires entity_tokens.")

                # Require at least one positive thickness; align with schema
                if inside_thickness is None and outside_thickness is None:
                    raise feature_tools.FeatureOperationError("Provide inside_thickness or outside_thickness (must be > 0).")

                try:
                    inside_value = float(inside_thickness) if inside_thickness is not None else 0.0
                    outside_value = float(outside_thickness) if outside_thickness is not None else 0.0
                except (TypeError, ValueError):
                    raise feature_tools.FeatureOperationError("inside_thickness and outside_thickness must be numbers.")

                if inside_value < 0 or outside_value < 0:
                    raise feature_tools.FeatureOperationError("inside_thickness and outside_thickness must be >= 0.")
                if inside_value <= 0 and outside_value <= 0:
                    raise feature_tools.FeatureOperationError("At least one thickness must be > 0.")

                thickness_unit_norm = (str(thickness_unit or "mm")).strip().lower()
                if thickness_unit_norm not in {"mm", "cm", "m", "in"}:
                    raise feature_tools.FeatureOperationError("thickness_unit must be one of ['mm', 'cm', 'm', 'in'].")

                if not isinstance(is_tangent_chain, bool):
                    raise feature_tools.FeatureOperationError("is_tangent_chain must be a boolean.")

                shell_type_norm = (str(shell_type or "sharp")).strip().lower()
                if shell_type_norm not in {"sharp", "rounded"}:
                    raise feature_tools.FeatureOperationError("shell_type must be 'sharp' or 'rounded'.")

                result = feature_tools.create_shell(
                    self._app,
                    entity_tokens,
                    inside_value,
                    outside_value,
                    thickness_unit_norm,
                    bool(is_tangent_chain),
                    shell_type_norm,
                    str(feature_name),
                )
                success = True
                message_text = result.get("message", "Shell created successfully.")
                payload.update({
                    "success": success,
                    "message": message_text,
                    "entity_count": result.get("entity_count"),
                    "entity_type": result.get("entity_type"),
                    "inside_thickness": result.get("inside_thickness"),
                    "outside_thickness": result.get("outside_thickness"),
                    "thickness_unit": result.get("thickness_unit"),
                    "is_tangent_chain": result.get("is_tangent_chain"),
                    "shell_type": result.get("shell_type"),
                    "feature_name": result.get("feature_name"),
                })
                log_level = 'success'
                self._palette_manager.send_log(log_level, message_text, doc_id=doc_id)

            elif operation == "create_simple_hole":
                face_token = params.get("face_token") or message.get("face_token")
                center_x = params.get("center_x") or message.get("center_x")
                center_y = params.get("center_y") or message.get("center_y")
                center_z = params.get("center_z") or message.get("center_z")
                diameter = params.get("diameter") or message.get("diameter")
                diameter_unit = params.get("diameter_unit") or message.get("diameter_unit", "mm")
                extent_type = params.get("extent_type") or message.get("extent_type")
                depth = params.get("depth") or message.get("depth")
                feature_name = params.get("feature_name") or message.get("feature_name", "")

                if not face_token:
                    raise feature_tools.FeatureOperationError("create_simple_hole requires face_token.")
                if center_x is None or center_y is None or center_z is None:
                    raise feature_tools.FeatureOperationError("create_simple_hole requires center coordinates (center_x, center_y, center_z).")
                if diameter is None:
                    raise feature_tools.FeatureOperationError("create_simple_hole requires diameter.")
                if not extent_type:
                    raise feature_tools.FeatureOperationError("create_simple_hole requires extent_type.")

                result = feature_tools.create_simple_hole(
                    self._app,
                    str(face_token),
                    float(center_x),
                    float(center_y),
                    float(center_z),
                    float(diameter),
                    str(diameter_unit),
                    str(extent_type),
                    float(depth) if depth is not None else None,
                    str(feature_name)
                )
                success = True
                message_text = result.get("message", "Hole created successfully.")
                payload.update({
                    "success": success,
                    "message": message_text,
                    "diameter": result.get("diameter"),
                    "diameter_unit": result.get("diameter_unit"),
                    "extent_type": result.get("extent_type"),
                    "depth": result.get("depth"),
                })
                log_level = 'success'
                self._palette_manager.send_log(log_level, message_text, doc_id=doc_id)

            elif operation == "create_counterbore_hole":
                face_token = params.get("face_token") or message.get("face_token")
                center_x = params.get("center_x") or message.get("center_x")
                center_y = params.get("center_y") or message.get("center_y")
                center_z = params.get("center_z") or message.get("center_z")
                hole_diameter = params.get("hole_diameter") or message.get("hole_diameter")
                hole_depth = params.get("hole_depth") or message.get("hole_depth")
                counterbore_diameter = params.get("counterbore_diameter") or message.get("counterbore_diameter")
                counterbore_depth = params.get("counterbore_depth") or message.get("counterbore_depth")
                diameter_unit = params.get("diameter_unit") or message.get("diameter_unit", "mm")
                feature_name = params.get("feature_name") or message.get("feature_name", "")

                if not face_token:
                    raise feature_tools.FeatureOperationError("create_counterbore_hole requires face_token.")
                if center_x is None or center_y is None or center_z is None:
                    raise feature_tools.FeatureOperationError("create_counterbore_hole requires center coordinates (center_x, center_y, center_z).")
                if hole_diameter is None:
                    raise feature_tools.FeatureOperationError("create_counterbore_hole requires hole_diameter.")
                if hole_depth is None:
                    raise feature_tools.FeatureOperationError("create_counterbore_hole requires hole_depth.")
                if counterbore_diameter is None:
                    raise feature_tools.FeatureOperationError("create_counterbore_hole requires counterbore_diameter.")
                if counterbore_depth is None:
                    raise feature_tools.FeatureOperationError("create_counterbore_hole requires counterbore_depth.")

                result = feature_tools.create_counterbore_hole(
                    self._app,
                    str(face_token),
                    float(center_x),
                    float(center_y),
                    float(center_z),
                    float(hole_diameter),
                    float(hole_depth),
                    float(counterbore_diameter),
                    float(counterbore_depth),
                    str(diameter_unit),
                    str(feature_name)
                )
                success = True
                message_text = result.get("message", "Counterbore hole created successfully.")
                payload.update({
                    "success": success,
                    "message": message_text,
                    "hole_diameter": result.get("hole_diameter"),
                    "counterbore_diameter": result.get("counterbore_diameter"),
                    "hole_depth": result.get("hole_depth"),
                    "counterbore_depth": result.get("counterbore_depth"),
                    "diameter_unit": result.get("diameter_unit"),
                })
                log_level = 'success'
                self._palette_manager.send_log(log_level, message_text, doc_id=doc_id)

            elif operation == "create_tapped_hole":
                face_token = params.get("face_token") or message.get("face_token")
                center_x = params.get("center_x") or message.get("center_x")
                center_y = params.get("center_y") or message.get("center_y")
                center_z = params.get("center_z") or message.get("center_z")
                thread_type = params.get("thread_type") or message.get("thread_type")
                thread_size = params.get("thread_size") or message.get("thread_size")
                thread_depth = params.get("thread_depth") or message.get("thread_depth")
                pilot_hole_depth = params.get("pilot_hole_depth") or message.get("pilot_hole_depth")
                diameter_unit = params.get("diameter_unit") or message.get("diameter_unit", "mm")
                feature_name = params.get("feature_name") or message.get("feature_name", "")

                if not face_token:
                    raise feature_tools.FeatureOperationError("create_tapped_hole requires face_token.")
                if center_x is None or center_y is None or center_z is None:
                    raise feature_tools.FeatureOperationError("create_tapped_hole requires center coordinates.")
                if not thread_type:
                    raise feature_tools.FeatureOperationError("create_tapped_hole requires thread_type.")
                if not thread_size:
                    raise feature_tools.FeatureOperationError("create_tapped_hole requires thread_size.")
                if thread_depth is None:
                    raise feature_tools.FeatureOperationError("create_tapped_hole requires thread_depth.")

                result = feature_tools.create_tapped_hole(
                    self._app,
                    str(face_token),
                    float(center_x),
                    float(center_y),
                    float(center_z),
                    str(thread_type),
                    str(thread_size),
                    float(thread_depth),
                    float(pilot_hole_depth) if pilot_hole_depth is not None else None,
                    str(diameter_unit),
                    str(feature_name)
                )
                success = True
                message_text = result.get("message", "Tapped hole created successfully.")
                payload.update({
                    "success": success,
                    "message": message_text,
                    "thread_type": result.get("thread_type"),
                    "thread_designation": result.get("thread_designation"),
                    "tap_drill_diameter": result.get("tap_drill_diameter"),
                    "thread_depth": result.get("thread_depth"),
                    "pilot_hole_depth": result.get("pilot_hole_depth"),
                    "diameter_unit": result.get("diameter_unit"),
                })
                log_level = 'success'
                self._palette_manager.send_log(log_level, message_text, doc_id=doc_id)

            elif operation == "create_external_thread":
                face_token = params.get("face_token") or message.get("face_token")
                thread_type = params.get("thread_type") or message.get("thread_type")
                thread_size = params.get("thread_size") or message.get("thread_size")
                thread_length = params.get("thread_length") or message.get("thread_length")
                thread_offset = params.get("thread_offset") or message.get("thread_offset", 0.0)
                # Fusion default is full-length threads; treat missing flag as True
                is_full_length = params.get("is_full_length", message.get("is_full_length", True))
                diameter_unit = params.get("diameter_unit") or message.get("diameter_unit", "mm")
                feature_name = params.get("feature_name") or message.get("feature_name", "")

                if not face_token:
                    raise feature_tools.FeatureOperationError("create_external_thread requires face_token.")
                if not thread_type:
                    raise feature_tools.FeatureOperationError("create_external_thread requires thread_type.")
                if not thread_size:
                    raise feature_tools.FeatureOperationError("create_external_thread requires thread_size.")
                if thread_length is None and not is_full_length:
                    raise feature_tools.FeatureOperationError("create_external_thread requires thread_length when is_full_length is False.")

                result = feature_tools.create_external_thread(
                    self._app,
                    str(face_token),
                    str(thread_type),
                    str(thread_size),
                    float(thread_length) if thread_length is not None else 0.0,
                    float(thread_offset),
                    bool(is_full_length),
                    str(diameter_unit),
                    str(feature_name)
                )
                success = True
                message_text = result.get("message", "External thread created successfully.")
                payload.update({
                    "success": success,
                    "message": message_text,
                    "thread_type": result.get("thread_type"),
                    "thread_designation": result.get("thread_designation"),
                    "nominal_diameter": result.get("nominal_diameter"),
                    "thread_length": result.get("thread_length"),
                    "thread_offset": result.get("thread_offset"),
                    "is_full_length": result.get("is_full_length"),
                    "diameter_unit": result.get("diameter_unit"),
                })
                log_level = 'success'
                self._palette_manager.send_log(log_level, message_text, doc_id=doc_id)

            elif operation == "create_pattern_feature":
                # Extract parameters from backend-prepared payload
                pattern_type = params.get("pattern_type")
                feature_tokens = params.get("feature_tokens", [])
                anchor_point_cm = params.get("anchor_point_cm")
                feature_name = params.get("feature_name", "")

                if not pattern_type:
                    raise feature_tools.FeatureOperationError("create_pattern_feature requires pattern_type.")
                if not feature_tokens:
                    raise feature_tools.FeatureOperationError("create_pattern_feature requires at least one feature_token.")
                if not anchor_point_cm or len(anchor_point_cm) != 3:
                    raise feature_tools.FeatureOperationError("create_pattern_feature requires anchor_point_cm as [x, y, z].")

                result = feature_tools.create_pattern_feature(
                    self._app,
                    str(pattern_type),
                    feature_tokens,
                    dict(params)  # Pass all params including pattern-specific parameters
                )
                success = True
                message_text = result.get("message", "Pattern feature created successfully.")
                payload.update({
                    "success": success,
                    "message": message_text,
                    "pattern_type": result.get("pattern_type"),
                    "instance_count": result.get("instance_count"),
                    "feature_token": result.get("feature_token"),
                })
                log_level = 'success'
                self._palette_manager.send_log(log_level, message_text, doc_id=doc_id)

            else:
                raise feature_tools.FeatureOperationError(f"Unsupported feature operation '{operation}'.")

        except feature_tools.FeatureOperationError as exc:
            message_text = str(exc)
            payload.update({
                "success": False,
                "error": message_text,
            })
            self._palette_manager.send_log('error', message_text, doc_id=doc_id)
        except Exception as exc:  # pragma: no cover - defensive
            message_text = f"Unexpected error: {exc}"
            logger.exception("Unexpected error while handling feature operation.")
            payload.update({
                "success": False,
                "error": message_text,
            })
            self._palette_manager.send_log('error', message_text, doc_id=doc_id)
        finally:
            # Ensure payload has success flag and message
            payload.setdefault("success", success)
            payload.setdefault("message", message_text)

            if client:
                client.send_json(payload)
            else:
                logger.warning("Feature operation result not sent; no active WebSocket client.")

    def _handle_feature_snapshot_request(self, doc_id: str, message: Dict[str, Any]) -> None:
        """Capture a snapshot of recent timeline features and send it to the backend."""
        session_info = self._get_session_info_by_doc(doc_id)
        if not session_info:
            logger.warning("Feature snapshot requested for unknown document %s", doc_id)
            return

        session_id = session_info.get("session_id")
        client: Optional[FusionWebSocketClient] = session_info.get("ws_client")
        design = session_info.get("design")
        if not design:
            design = self._resolve_design_reference(session_info.get("document"))
            if design:
                session_info["design"] = design

        def _parse_max_features(value: Any) -> int:
            try:
                parsed = int(value)
                if parsed <= 0:
                    return 25
                return min(parsed, 100)
            except (TypeError, ValueError):
                return 25

        max_features = _parse_max_features(message.get("max_features"))

        payload: Dict[str, Any] = {
            "type": "feature_snapshot",
            "session_id": session_id,
            "doc_id": doc_id,
            "message_id": message.get("message_id"),
            "requested_timeline_count": message.get("timeline_count"),
            "requested_marker_position": message.get("marker_position"),
            "max_features": max_features,
        }

        if not design:
            payload.update({
                "success": False,
                "error": "Design context unavailable for feature snapshot.",
            })
        else:
            try:
                snapshot = feature_tools.capture_feature_snapshot(design, max_features=max_features)
                payload.update(snapshot)
                payload.setdefault("success", True)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to capture feature snapshot")
                payload.update({
                    "success": False,
                    "error": f"Failed to capture feature snapshot: {exc}",
                })

        if client:
            client.send_json(payload)
        else:
            logger.warning("Feature snapshot result not sent; no active WebSocket client.")

    def _handle_entity_context_request(self, doc_id: str, message: Dict[str, Any]) -> None:
        """
        Extract fresh entity context (bodies, faces, edges) and send it to the backend.

        This is called after geometry-modifying operations to keep the LLM aware of
        the current state of all entities in the design.
        """
        session_info = self._get_session_info_by_doc(doc_id)
        if not session_info:
            logger.warning("Entity context requested for unknown document %s", doc_id)
            return

        session_id = session_info.get("session_id")
        client: Optional[FusionWebSocketClient] = session_info.get("ws_client")
        context_request_id = message.get("context_request_id")
        message_id = message.get("message_id")

        # Always force fresh design resolution before entity extraction to avoid stale references
        # This is critical because cached design references can become invalid after timeline operations
        design = self._resolve_design_reference(session_info.get('document'))
        if design:
            session_info['design'] = design

        # Extract fresh entity context with a short retry loop to avoid transient empties
        entity_context: Optional[Dict[str, Any]] = None
        for attempt in range(3):
            entity_context = self._extract_entity_context(design)
            total_entities = 0
            if entity_context:
                total_entities = (
                    len(entity_context.get("bodies", [])) +
                    len(entity_context.get("faces", [])) +
                    len(entity_context.get("edges", []))
                )
            if entity_context and total_entities > 0:
                break
            # Exponential-ish backoff: 0.2s, 0.4s
            wait = 0.2 * (attempt + 1)
            logger.debug(
                "Entity context empty on attempt %d for session %s (retrying in %.2fs)",
                attempt + 1, session_id, wait
            )
            time.sleep(wait)

        payload: Dict[str, Any] = {
            "type": "entity_context_response",
            "session_id": session_id,
            "doc_id": doc_id,
        }
        if context_request_id is not None:
            payload["context_request_id"] = context_request_id
            payload["message_id"] = context_request_id if message_id is None else message_id
        elif message_id is not None:
            payload["message_id"] = message_id
            payload["context_request_id"] = message_id

        if entity_context:
            payload["entity_context"] = entity_context
            logger.info(
                "Sending fresh entity context for session %s: %d bodies, %d faces, %d edges",
                session_id,
                len(entity_context.get("bodies", [])),
                len(entity_context.get("faces", [])),
                len(entity_context.get("edges", []))
            )
        else:
            logger.warning(
                "Failed to extract entity context for session %s after retries; sending empty context",
                session_id
            )
            payload["entity_context"] = {
                "bodies": [],
                "faces": [],
                "edges": [],
            }

        if client:
            client.send_json(payload)
        else:
            logger.warning("Entity context response not sent; no active WebSocket client.")

    def _handle_revert_timeline(self, doc_id: str, message: Dict[str, Any]) -> None:
        """
        Revert the timeline to a specific checkpoint position.

        Args:
            doc_id: Document identifier
            message: Message containing marker_position and message_id
        """
        message_id = message.get("message_id")
        marker_position = message.get("marker_position", 0)
        timeline_count = message.get("timeline_count", 0)

        logger.info(
            "Reverting timeline for doc %s to marker_position=%d (expected_count=%d, message_id=%s)",
            doc_id, marker_position, timeline_count, message_id
        )

        session_info = self._get_session_info_by_doc(doc_id)
        if not session_info:
            logger.warning("Revert timeline requested for unknown document %s", doc_id)
            return

        session_id = session_info.get("session_id")
        client: Optional[FusionWebSocketClient] = session_info.get("ws_client")
        design = session_info.get('design')

        if not design:
            design = self._resolve_design_reference(session_info.get('document'))
            if design:
                session_info['design'] = design

        payload: Dict[str, Any] = {
            "type": "execution_result",
            "session_id": session_id,
            "message_id": message_id,
        }

        try:
            if not design or not design.timeline:
                raise RuntimeError("Timeline not available for this design.")

            timeline = design.timeline
            current_position = timeline.markerPosition
            current_count = timeline.count

            logger.debug(
                "Timeline state - current: marker_position=%d, count=%d",
                current_position, current_count
            )

            # Clamp marker position to valid range (timeline may have shrunk since checkpoint)
            original_marker = marker_position
            if marker_position < 0:
                marker_position = 0
                logger.warning(
                    "Clamped marker position from %d to 0 (timeline count: %d)",
                    original_marker, current_count
                )
            elif marker_position > current_count:
                marker_position = current_count
                logger.warning(
                    "Clamped marker position from %d to %d (timeline shrunk since checkpoint)",
                    original_marker, current_count
                )

            # Set the timeline marker position
            timeline.markerPosition = marker_position
            removed_items = 0

            if hasattr(timeline, "deleteAllAfterMarker"):
                before_delete_count = timeline.count
                timeline.deleteAllAfterMarker()
                after_delete_count = timeline.count
                removed_items = max(0, before_delete_count - after_delete_count)
                logger.debug(
                    "Removed %d timeline item(s) beyond marker (count: %d → %d)",
                    removed_items,
                    before_delete_count,
                    after_delete_count,
                )

                # Build success message, noting if position was clamped
                if original_marker != marker_position:
                    success_message = f"Timeline reverted to position {marker_position} (requested {original_marker}, clamped to timeline end)"
                else:
                    success_message = f"Timeline reverted to position {marker_position}"
                logger.info("Successfully reverted timeline: %s", success_message)

                payload.update({
                    "success": True,
                    "message": success_message,
                    "marker_position": marker_position,
                    "previous_position": current_position,
                    "removed_items": removed_items,
                    "geometry_reverted": True,
                })
            else:
                # CRITICAL: Missing deleteAllAfterMarker means geometry cannot be reverted
                # Return success=False to prevent backend from trimming conversation
                error_message = "Timeline.deleteAllAfterMarker is not available; cannot revert geometry."
                logger.error("Revert failed: %s", error_message)
                payload.update({
                    "success": False,
                    "error": error_message,
                    "marker_position": marker_position,
                    "previous_position": current_position,
                    "removed_items": 0,
                    "geometry_reverted": False,
                })
                self._palette_manager.send_log('error', f"✗ {error_message}", doc_id=doc_id, scope='global')

            # Only send success log if operation succeeded
            if payload.get("success"):
                self._palette_manager.send_log('success', f"✓ {success_message}", doc_id=doc_id, scope='global')

        except ValueError as exc:
            error_message = str(exc)
            logger.error("Invalid revert request: %s", error_message)
            payload.update({
                "success": False,
                "error": error_message,
            })
            self._palette_manager.send_log('error', f"Revert failed: {error_message}", doc_id=doc_id, scope='global')

        except Exception as exc:
            error_message = f"Failed to revert timeline: {exc}"
            logger.exception("Unexpected error during timeline revert.")
            payload.update({
                "success": False,
                "error": error_message,
            })
            self._palette_manager.send_log('error', error_message, doc_id=doc_id, scope='global')

        finally:
            # Send result back to backend
            if client:
                client.send_json(payload)
            else:
                logger.warning("Revert timeline result not sent; no active WebSocket client.")

    def handle_plan_approval_on_main_thread(self, plan_text: str, doc_id: Optional[str]) -> None:
        """
        Handle plan approval dialog on the main UI thread.
        Called via CustomEvent to ensure dialog shows properly.
        """
        logger.info("Plan approval dialog - showing on main thread")

        # Show approval dialog (blocking call on main thread)
        dialog_result = self._ui.messageBox(
            f"{'='*60}\nCADAgent Plan\n{'='*60}\n\n{plan_text}\n\n{'='*60}\n\nDo you want to execute this plan?",
            "CADAgent - Plan Approval Required",
            adsk.core.MessageBoxButtonTypes.YesNoButtonType,
            adsk.core.MessageBoxIconTypes.QuestionIconType
        )

        approved = dialog_result == adsk.core.DialogResults.DialogYes

        # Log the decision
        if approved:
            self._palette_manager.send_log('success', '✓ Plan approved by user - beginning execution', doc_id=doc_id)
        else:
            self._palette_manager.send_log('warning', '✗ Plan rejected by user', doc_id=doc_id)

        if not doc_id:
            doc_id = self._active_doc_id

        session_info = self._get_session_info_by_doc(doc_id)
        if not session_info:
            logger.warning("Plan approval received but no session info available; using active session")
            session_info = self._get_session_info_by_doc(self._active_doc_id)

        session_id = session_info.get("session_id") if session_info else None
        client = session_info.get("ws_client") if session_info else None

        plan_text_backend = ""
        if approved:
            full_plan = self._pending_plan_full.get(doc_id or "", "")
            plan_text_backend = full_plan or plan_text

        payload = {
            "type": "plan_approval",
            "session_id": session_id,
            "approved": approved,
            "plan_text": plan_text_backend if approved else "",
            "message": "" if approved else "Plan rejected by user",
        }

        logger.info(f"Plan {'approved' if approved else 'rejected'} by user")
        if client:
            client.send_json(payload)

        # Reset cached plan data after decision
        if doc_id:
            self._pending_plan_full.pop(doc_id, None)


    # ------------------------------------------------------------------ Helpers
    def _capture_visual_context_snapshot(self) -> Optional[Dict[str, Any]]:
        """Capture the active viewport and return it as a base64-encoded PNG bundle."""
        viewport = self._app.activeViewport if self._app else None
        if not viewport:
            logger.warning("Visual context capture skipped: no active viewport is available.")
            return None

        raw_width = getattr(viewport, 'width', 0) or 0
        raw_height = getattr(viewport, 'height', 0) or 0

        try:
            viewport_width = int(raw_width)
        except (TypeError, ValueError):
            viewport_width = 0

        try:
            viewport_height = int(raw_height)
        except (TypeError, ValueError):
            viewport_height = 0

        target_width = max(1, min(1280, viewport_width or 1280))
        if viewport_width > 0 and viewport_height > 0:
            aspect_ratio = viewport_height / max(viewport_width, 1)
            target_height = max(1, int(round(target_width * aspect_ratio)))
        else:
            fallback_height = viewport_height if viewport_height > 0 else 720
            target_height = max(1, min(720, fallback_height))

        tmp_file_path: Optional[Path] = None
        encoded_image: Optional[str] = None

        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_file_path = Path(temp_file.name)
            temp_file.close()

            success = viewport.saveAsImageFile(str(tmp_file_path), target_width, target_height)
            if not success:
                raise RuntimeError("saveAsImageFile returned False")

            image_bytes = tmp_file_path.read_bytes()
            encoded_image = base64.b64encode(image_bytes).decode("ascii")
        except Exception as exc:
            logger.exception("Failed to capture visual context snapshot: %s", exc)
            return None
        finally:
            if tmp_file_path and tmp_file_path.exists():
                try:
                    tmp_file_path.unlink()
                except Exception as cleanup_error:
                    logger.debug("Unable to delete temporary snapshot file %s: %s", tmp_file_path, cleanup_error)

        if not encoded_image:
            return None

        return {
            "media_type": "image/png",
            "data": encoded_image,
            "label": "Visual state of the 3D model inside Fusion 360",
            "width": target_width,
            "height": target_height,
        }

    def _extract_entity_context(
        self,
        design: Optional[adsk.fusion.Design] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract all bodies, faces, and edges from the active design.

        This provides the LLM with complete entity awareness without requiring
        tool calls. The entity tokens allow subsequent select_* operations.

        Returns:
            Dictionary with bodies, faces, edges arrays and unit information,
            or None if extraction fails or no design is active.
        """
        if not self._app:
            return None

        try:
            context: Dict[str, Any] = {}

            if design is None or not getattr(design, "isValid", True):
                design = adsk.fusion.Design.cast(self._app.activeProduct)
            if not design or not getattr(design, "isValid", True):
                return None

            # Extract bodies
            try:
                bodies, body_units = body_tools.list_bodies(self._app, design)
                context["bodies"] = bodies
                context["body_units"] = body_units
            except Exception as exc:
                logger.warning("Body extraction failed: %s", exc)
                context["bodies"] = []
                context["body_units"] = {}

            # Extract faces
            try:
                faces, face_units = face_tools.list_faces(self._app, design)
                context["faces"] = faces
                context["face_units"] = face_units
            except Exception as exc:
                logger.warning("Face extraction failed: %s", exc)
                context["faces"] = []
                context["face_units"] = {}

            # Extract edges
            try:
                edges, edge_units = edge_tools.list_edges(self._app, design)
                context["edges"] = edges
                context["edge_units"] = edge_units
            except Exception as exc:
                logger.warning("Edge extraction failed: %s", exc)
                context["edges"] = []
                context["edge_units"] = ""

            # Build nested spatial_context structure per SPATIAL_CONTEXT_OVERHAUL.md
            bodies_list = context.get("bodies", [])
            faces_list = context.get("faces", [])
            edges_list = context.get("edges", [])

            spatial_bodies = []
            all_vertices = []  # Flat list for backward compatibility

            for body in bodies_list:
                body_idx = body.get("body_index")
                body_vertices = body.get("vertices", [])

                # Add vertices to flat list with body context
                for v in body_vertices:
                    v_copy = dict(v)
                    v_copy["body_index"] = body_idx
                    all_vertices.append(v_copy)

                # Filter faces and edges belonging to this body
                body_faces = [f for f in faces_list if f.get("body_index") == body_idx]
                body_edges = [e for e in edges_list if e.get("body_index") == body_idx]

                spatial_bodies.append({
                    "id": body.get("id"),
                    "name": body.get("name"),
                    "token": body.get("token"),
                    "body_index": body_idx,
                    "occurrence_path": body.get("occurrence_path"),
                    "bbox": body.get("bbox"),
                    "vertices": body_vertices,
                    "faces": body_faces,
                    "edges": body_edges,
                })

            context["spatial_context"] = {
                "units": "mm",
                "bodies": spatial_bodies,
            }
            context["vertices"] = all_vertices

            # Only return context if we have at least some entities
            total_entities = len(context.get("bodies", [])) + len(context.get("faces", [])) + len(context.get("edges", []))
            if total_entities == 0:
                logger.debug("No entities found in design for entity context")
                return None

            logger.info(
                "Extracted entity context: %d bodies, %d faces, %d edges",
                len(context.get("bodies", [])),
                len(context.get("faces", [])),
                len(context.get("edges", []))
            )
            return context

        except Exception as exc:
            logger.warning("Entity context extraction failed: %s", exc)
            return None

    def _collect_timeline_state(self) -> Dict[str, Any]:
        """Collect the current state of the Fusion timeline for context, including marker position."""
        design = self._design
        if not design:
            return {"items": [], "marker_position": 0, "count": 0}

        timeline = design.timeline
        items: List[Dict[str, Any]] = []

        for index in range(timeline.count):
            item = timeline.item(index)
            entity = item.entity
            entity_name = ""

            if entity:
                entity_name = getattr(entity, "name", "") or entity.objectType.split("::")[-1]

            items.append({
                "index": index,
                "type": entity_name or "Unknown",
                "name": item.name,
            })

        marker_position = timeline.markerPosition
        count = timeline.count

        logger.debug(f"Collected timeline state with {count} items, marker at position {marker_position}")
        return {
            "items": items,
            "marker_position": marker_position,
            "count": count
        }

    # ---------------------------- Per-document sessions and events ----------------------------
    def _doc_identity(self, doc: adsk.core.Document) -> Tuple[str, str]:
        """Return (doc_id, doc_name) using stable identifiers when possible.

        Strategy:
        - If the document is saved, use DataFile.id/objectId (stable across sessions).
        - Otherwise, store a UUID attribute on the Document itself so we get a
          stable runtime identifier even if Python proxy objects change.
        - Only as a final fallback, use a synthetic id mapped to the doc's memory id.
        """
        name = getattr(doc, 'name', 'Untitled')
        doc_id: Optional[str] = None

        # First, prefer CADAgent-specific attribute if present (stable across save events)
        try:
            attrs = getattr(doc, 'attributes', None)
            if attrs is not None:
                existing = attrs.itemByName('CADAgent', 'doc_uuid')
                if existing:
                    doc_id = existing.value
        except Exception:
            pass

        # If no attribute yet, prefer DataFile id if available (saved documents)
        if not doc_id:
            try:
                data_file = getattr(doc, 'dataFile', None)
                if data_file:
                    doc_id = getattr(data_file, 'id', None) or getattr(data_file, 'objectId', None)
            except Exception:
                pass

        # If still no id (unsaved or attribute API available but no value), create and store attribute
        if not doc_id:
            try:
                attrs = getattr(doc, 'attributes', None)
                if attrs is not None:
                    new_id = f"unsaved-{uuid.uuid4()}"
                    attrs.add('CADAgent', 'doc_uuid', new_id)
                    doc_id = new_id
            except Exception:
                # Attribute API not available on this object; fall back below
                pass

        # Fall back to Document.id or objectId if available
        if not doc_id:
            for attr in ('id', 'objectId'):
                try:
                    val = getattr(doc, attr, None)
                    if val:
                        doc_id = str(val)
                        break
                except Exception:
                    continue

        # Last resort: synthetic id tied to this document instance for add-in lifetime
        if not doc_id:
            key = id(doc)
            if key not in self._synthetic_ids:
                self._synthetic_ids[key] = f"unsaved-{uuid.uuid4()}"
            doc_id = self._synthetic_ids[key]

        return doc_id, name

    def _get_active_ws_client(self) -> Optional[FusionWebSocketClient]:
        # Refresh/propagate token on every access to keep backend auth current
        self._get_user_token()
        if self._active_doc_id and self._active_doc_id in self._sessions:
            return self._sessions[self._active_doc_id].get('ws_client')
        return None

    def _ensure_session_for_doc(self, doc: adsk.core.Document) -> Dict[str, Any]:
        doc_id, name = self._doc_identity(doc)
        token_required = bool(self._auth_client) and not self._auth_bypass

        if doc_id in self._sessions:
            info = self._sessions[doc_id]
            info['doc_name'] = name
            info['document'] = doc
            design = self._resolve_design_reference(doc)
            if design:
                info['design'] = design
            # If the client exists but isn't connected (e.g., backend restart), recreate it using the same session_id
            try:
                client = info.get('ws_client')
                if not client or not client.is_connected():
                    session_id = info.get('session_id') or str(uuid.uuid4())
                    ws_url = config.build_ws_url(session_id)
                    # Get user token for usage tracking
                    user_token = self._get_user_token()
                    if token_required and not user_token:
                        logger.info(
                            "Deferring reconnect for '%s' until authentication is available",
                            name,
                        )
                        return info
                    if not user_token:
                        logger.info("Reconnecting '%s' in auth-bypass mode (no auth token)", name)
                    new_client = FusionWebSocketClient(ws_url, user_token=user_token)
                    # Set API keys for BYOK
                    api_keys = self.get_api_keys_for_backend()
                    if api_keys:
                        new_client.set_api_keys(api_keys)
                        logger.info(
                            "[api_keys] Attaching keys on reconnect (doc=%s) anthropic=%s openai=%s google=%s",
                            doc_id,
                            bool(api_keys.get("anthropic_api_key")),
                            bool(api_keys.get("openai_api_key")),
                            bool(api_keys.get("google_api_key")),
                        )
                    new_client.add_message_handler(self._make_ws_handler(doc_id))
                    new_client.add_state_handler(lambda connected, d=doc_id: self._on_ws_state(d, connected))
                    new_client.start()
                    info['ws_client'] = new_client
                    if 'session_id' not in info:
                        info['session_id'] = session_id
                    logger.info(
                        "✓ Reconnected WebSocket for doc '%s' (%s) at %s (authenticated=%s)",
                        name,
                        doc_id,
                        ws_url,
                        bool(user_token),
                    )
                    # Proactively push keys again even though they were sent in authenticate payload
                    self._send_api_keys_to_backend(doc_id, reason="post_reconnect")
            except Exception as e:
                logger.error(f"Failed to (re)connect WebSocket for document {name}: {e}")
            return info

        # Create a new session for this document
        session_id = str(uuid.uuid4())
        ws_url = config.build_ws_url(session_id)
        # Get user token for usage tracking
        user_token = self._get_user_token()
        if token_required and not user_token:
            logger.info(
                "Deferring WebSocket creation for '%s' until authentication is available",
                name,
            )
            info = {
                'session_id': session_id,
                'ws_client': None,
                'created_at': time.time(),
                'last_active': time.time(),
                'doc_name': name,
                'document': doc,
            }
            design = self._resolve_design_reference(doc)
            if design:
                info['design'] = design
            self._sessions[doc_id] = info
            return info
        if not user_token:
            logger.info("Creating '%s' in auth-bypass mode (no auth token)", name)
        client = FusionWebSocketClient(ws_url, user_token=user_token)
        # Set API keys for BYOK
        api_keys = self.get_api_keys_for_backend()
        if api_keys:
            client.set_api_keys(api_keys)
            logger.info(
                "[api_keys] Attaching keys on new client (doc=%s) anthropic=%s openai=%s google=%s",
                doc_id,
                bool(api_keys.get("anthropic_api_key")),
                bool(api_keys.get("openai_api_key")),
                bool(api_keys.get("google_api_key")),
            )
        client.add_message_handler(self._make_ws_handler(doc_id))
        client.add_state_handler(lambda connected, d=doc_id: self._on_ws_state(d, connected))
        try:
            client.start()
            logger.info(
                "✓ WebSocket started for doc '%s' (%s) at %s (authenticated=%s)",
                name,
                doc_id,
                ws_url,
                bool(user_token),
            )
            # Proactively push keys again even though they were sent in authenticate payload
            self._send_api_keys_to_backend(doc_id, reason="post_start")
        except Exception as e:
            logger.error(f"Failed to start WebSocket for document {name}: {e}")
            raise

        info = {
            'session_id': session_id,
            'ws_client': client,
            'created_at': time.time(),
            'last_active': time.time(),
            'doc_name': name,
        }
        info['document'] = doc
        design = self._resolve_design_reference(doc)
        if design:
            info['design'] = design
        self._sessions[doc_id] = info
        return info

    # ------------------------------------------------------------------ Auth Methods
    def send_magic_link(self, email: str) -> Dict[str, Any]:
        """
        Send a magic link to the user's email for passwordless login.

        Args:
            email: User's email address

        Returns:
            Response dictionary with success status

        Raises:
            Exception: If auth client not initialized or sending fails
        """
        if not self._auth_client:
            raise Exception("Supabase auth not configured")

        logger.info(f"[auth] send_magic_link called (email={email})")
        return self._auth_client.send_magic_link(email)

    def send_otp_code(self, email: str) -> Dict[str, Any]:
        """Send a one-time code to the user's email."""
        if not self._auth_client:
            raise Exception("Supabase auth not configured")
        logger.info(f"[auth] send_otp_code called (email={email})")
        return self._auth_client.send_otp_code(email)

    def check_and_handle_signup(self, email: str) -> Dict[str, Any]:
        """
        Smart signup/login handler:
        - For NEW users: Creates account instantly without OTP (frictionless)
        - For EXISTING users: Sends OTP code
        """
        if not self._auth_client:
            raise Exception("Supabase auth not configured")
        logger.info(f"[auth] check_and_handle_signup called (email={email})")
        return self._auth_client.check_and_handle_signup(email)

    def login_with_password(self, email: str, password: str) -> Dict[str, Any]:
        """
        Testing-only password login for allowlisted email.
        """
        if not self._auth_client:
            raise Exception("Supabase auth not configured")
        if not config.TESTING_MODE:
            raise Exception("Password login is disabled")
        if config.TEST_EMAIL and email.strip().lower() != config.TEST_EMAIL:
            raise Exception("Password login not allowed for this account")
        if not password:
            raise Exception("Password is required")

        logger.info(f"[auth] login_with_password called (email={email})")
        return self._auth_client.login_with_password(email, password)

    def verify_otp_code(self, email: str, code: str) -> Dict[str, Any]:
        """Verify OTP code and establish session."""
        if not self._auth_client:
            raise Exception("Supabase auth not configured")
        logger.info(f"[auth] verify_otp_code called (email={email})")
        return self._auth_client.verify_otp_code(email, code)

    def handle_auth_callback(self, access_token: str, refresh_token: str) -> Dict[str, Any]:
        """
        Handle auth callback after user clicks magic link.

        Args:
            access_token: JWT access token from callback
            refresh_token: Refresh token from callback

        Returns:
            Session data dictionary

        Raises:
            Exception: If auth client not initialized or callback fails
        """
        if not self._auth_client:
            raise Exception("Supabase auth not configured")

        logger.info("[auth] handle_auth_callback called")
        return self._auth_client.set_session_from_callback(access_token, refresh_token)

    def logout(self) -> None:
        """
        Logout the current user and clear session.

        Raises:
            Exception: If auth client not initialized
        """
        if not self._auth_client:
            raise Exception("Supabase auth not configured")

        logger.info("[auth] logout called")
        self._auth_client.clear_session()

        # Close all WebSocket sessions
        for doc_id, info in list(self._sessions.items()):
            if info.get('ws_client'):
                try:
                    info['ws_client'].stop()
                except Exception:
                    pass
            self._code_executor.reset_context(doc_id)

        # Clear sessions and reset state
        self._sessions.clear()
        self._active_doc_id = None

        logger.info("[auth] User logged out, all sessions closed")

    def get_user_profile(self) -> Optional[Dict[str, Any]]:
        """
        Get the current user's profile from /me endpoint.

        Returns:
            User profile dictionary or None if not authenticated

        Raises:
            Exception: If auth client not initialized or fetch fails
        """
        if not self._auth_client:
            raise Exception("Supabase auth not configured")

        logger.info("[auth] get_user_profile called")
        return self._auth_client.get_profile()

    def is_authenticated(self) -> bool:
        """
        Check if user is currently authenticated.

        Returns:
            True if authenticated, False otherwise
        """
        if not self._auth_client:
            return False
        return self._auth_client.is_authenticated()

    def is_auth_bypass(self) -> bool:
        """Return whether auth bypass mode is enabled for this add-in session."""
        return bool(self._auth_bypass)

    def get_user_email(self) -> Optional[str]:
        """
        Get the current user's email.

        Returns:
            User email or None if not authenticated
        """
        if not self._auth_client:
            return None
        return self._auth_client.get_user_email()

    # ------------------------------------------------------------------ API Key Methods
    def get_api_keys_status(self) -> Dict[str, Any]:
        """
        Get the status of configured API keys.
        Requires authentication.

        Returns:
            Dictionary with status of each provider's API key

        Raises:
            PermissionError: If not authenticated
        """
        if not self.is_authenticated():
            raise PermissionError("Authentication required to get API keys status")
        api_key_manager = get_api_key_manager()
        return api_key_manager.get_status()

    def save_api_keys(self, keys: Dict[str, str]) -> bool:
        """
        Save API keys and update active websocket connections.
        Requires authentication.

        Args:
            keys: Dictionary with provider keys (anthropic_api_key, openai_api_key, google_api_key)

        Returns:
            True if keys were saved successfully

        Raises:
            PermissionError: If not authenticated
        """
        if not self.is_authenticated():
            raise PermissionError("Authentication required to save API keys")
        api_key_manager = get_api_key_manager()
        success = api_key_manager.set_keys(keys)

        if success:
            # Update all active websocket clients with new API keys
            backend_keys = api_key_manager.get_keys_for_backend()
            for info in self._sessions.values():
                client: Optional[FusionWebSocketClient] = info.get('ws_client')
                if client and client.is_connected():
                    try:
                        client.set_api_keys(backend_keys)
                        # Re-authenticate with new keys
                        client.send_json({
                            "type": "update_api_keys",
                            "api_keys": backend_keys
                        })
                        logger.info("Updated API keys on active websocket connection")
                    except Exception as e:
                        logger.warning(f"Failed to update API keys on connection: {e}")

        return success

    def get_api_keys_for_backend(self) -> Dict[str, str]:
        """
        Get API keys formatted for sending to the backend.

        Returns:
            Dictionary with keys ready for WebSocket auth message
        """
        api_key_manager = get_api_key_manager()
        return api_key_manager.get_keys_for_backend()

    def has_api_keys(self) -> bool:
        """
        Check if user has at least one API key configured.

        Returns:
            True if at least one key is configured
        """
        api_key_manager = get_api_key_manager()
        has_keys, _ = api_key_manager.has_required_keys()
        return has_keys

    def _send_api_keys_to_backend(self, doc_id: Optional[str], reason: str = "") -> bool:
        """
        Push locally stored API keys to the backend for a specific document/session.

        Args:
            doc_id: Target document id (falls back to active doc)
            reason: Optional log context for why this push is happening

        Returns:
            True if keys were sent to an active websocket client
        """
        keys = self.get_api_keys_for_backend()
        if not any(keys.values()):
            logger.info("[api_keys] Skipping push (%s): no stored keys", reason or "no reason")
            return False

        # Resolve session/client
        target_doc_id = doc_id or self._active_doc_id
        info = self._get_session_info_by_doc(target_doc_id)
        if not info:
            logger.warning("[api_keys] Cannot push keys (%s): no session for doc_id=%s", reason or "no reason", target_doc_id)
            return False

        client: Optional[FusionWebSocketClient] = info.get('ws_client')
        if not client or not client.is_connected():
            logger.warning("[api_keys] Cannot push keys (%s): websocket not connected for doc_id=%s", reason or "no reason", target_doc_id)
            return False

        try:
            client.set_api_keys(keys)
            client.send_json({
                "type": "update_api_keys",
                "api_keys": keys
            })
            logger.info(
                "[api_keys] Pushed keys (%s) for doc_id=%s anthropic=%s openai=%s google=%s",
                reason or "manual",
                target_doc_id,
                bool(keys.get("anthropic_api_key")),
                bool(keys.get("openai_api_key")),
                bool(keys.get("google_api_key")),
            )
            return True
        except Exception as e:
            logger.error(f"[api_keys] Failed to push keys ({reason}) for doc_id={target_doc_id}: {e}")
            return False

    def _on_ws_state(self, doc_id: str, connected: bool) -> None:
        """
        WebSocket state change handler (connected/disconnected).
        Used to push stored API keys as soon as a socket comes up.
        """
        try:
            if connected:
                self._send_api_keys_to_backend(doc_id, reason="ws_state_connected")
        except Exception as e:
            logger.warning(f"[api_keys] Failed to push on ws_state change: {e}")

    def _push_api_keys_all_sessions(self, reason: str = "") -> None:
        """Push stored API keys to all active sessions (best effort)."""
        for doc_id in list(self._sessions.keys()):
            self._send_api_keys_to_backend(doc_id, reason=reason or "bulk")

    # ------------------------------------------------------------------ Private Methods
    def _make_ws_handler(self, doc_id: str):
        def handler(message: Dict[str, Any]) -> None:
            self._enqueue_message(doc_id, message)
        return handler

    def _get_user_token(self) -> Optional[str]:
        """
        Get the user's JWT access token for usage tracking.

        Returns:
            JWT access token or None if not authenticated
        """
        if not self._auth_client:
            return None
        try:
            token = self._auth_client.get_valid_access_token()
            if not token:
                return None

            # If token changed (e.g., refreshed), propagate to active WS clients
            if token != self._last_sent_token:
                self._last_sent_token = token
                for info in self._sessions.values():
                    client: Optional[FusionWebSocketClient] = info.get('ws_client')
                    if client and client.is_connected():
                        try:
                            client.set_user_token(token)
                            logger.info("Updated WebSocket client with refreshed user token")
                        except Exception as e:
                            logger.warning(f"Failed to update token on WS client: {e}")
            return token
        except Exception as e:
            logger.warning(f"Failed to get user token: {e}")
        return None

    def _get_session_info_by_doc(self, doc_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not doc_id:
            return None
        return self._sessions.get(doc_id)

    def _get_ws_client_for_doc(self, doc_id: Optional[str]) -> Optional[FusionWebSocketClient]:
        # Ensure token is fresh before returning client
        self._get_user_token()
        info = self._get_session_info_by_doc(doc_id)
        if info:
            return info.get('ws_client')
        return None

    def _resolve_design_reference(self, doc: Optional[adsk.core.Document]) -> Optional[adsk.fusion.Design]:
        if not doc:
            return None
        try:
            products = getattr(doc, 'products', None)
            if products and hasattr(products, 'itemByProductType'):
                design = adsk.fusion.Design.cast(products.itemByProductType('DesignProductType'))
                if design:
                    return design
        except Exception:
            pass

        try:
            design = adsk.fusion.Design.cast(self._app.activeProduct)
            if design and getattr(design, 'document', None) == doc:
                return design
        except Exception:
            pass

        return None

    def _activate_session_for_current_document(self) -> None:
        doc = self._app.activeDocument if self._app else None
        if not doc:
            logger.info("No active document available; deferring session activation until one opens.")
            return
        self._switch_to_document(doc)

    def _switch_to_document(self, doc: adsk.core.Document) -> None:
        target_id, target_name = self._doc_identity(doc)
        if self._active_doc_id == target_id:
            logger.info(f"Document '{target_name}' already active; no switch required")
            return

        try:
            active_selections = getattr(self._ui, "activeSelections", None)
            if active_selections:
                logger.debug("Clearing active selections before switching documents")
                active_selections.clear()
        except Exception as exc:
            logger.warning("Unable to clear selections during document switch: %s", exc)

        # Ensure target session exists (create once per document) and keep all sessions connected.
        # We avoid stopping previous sessions so chat state and connections remain warm.
        info = self._ensure_session_for_doc(doc)
        self._active_doc_id = target_id
        info['last_active'] = time.time()

        # Refresh cached design reference so timeline and other queries target the active document.
        new_design = info.get('design') or self._resolve_design_reference(doc)
        if not new_design:
            try:
                new_design = adsk.fusion.Design.cast(self._app.activeProduct)
            except Exception:
                new_design = None
        if new_design:
            self._design = new_design

        # Notify palette about new session and switch (do NOT send fusionReady here to avoid UI resets)
        try:
            self._palette_manager.send_connection_status(doc_id=target_id)
            self._palette_manager.send_document_switched(target_id, target_name, info['session_id'])
        except Exception as e:
            logger.warning(f"Failed to notify palette about document switch: {e}")

    # Event relays from adsk handlers
    def on_document_activated(self, args: adsk.core.DocumentEventArgs) -> None:
        try:
            doc = args.document
            if doc:
                logger.info(f"documentActivated → {getattr(doc, 'name', 'Untitled')}")
                self._switch_to_document(doc)
        except Exception as e:
            logger.exception(f"Error handling documentActivated: {e}")

    def on_document_closed(self, args: adsk.core.DocumentEventArgs) -> None:
        try:
            doc = args.document
            doc_id, name = self._doc_identity(doc)
            logger.info(f"documentClosed → {name} ({doc_id})")
            # Clean up session
            info = self._sessions.pop(doc_id, None)
            if info and info.get('ws_client'):
                try:
                    info['ws_client'].stop()
                except Exception:
                    pass
            self._code_executor.reset_context(doc_id)
            if self._active_doc_id == doc_id:
                # Switch to current active document if any
                try:
                    self._activate_session_for_current_document()
                except Exception:
                    logger.info("No remaining active document after close.")
                    self._active_doc_id = None
                    self._palette_manager.send_connection_status()
        except Exception as e:
            logger.exception(f"Error handling documentClosed: {e}")

    def on_document_opened_or_created(self, args: adsk.core.DocumentEventArgs) -> None:
        # Optional: prepare mapping; actual switch occurs on activation
        try:
            doc = args.document
            if doc:
                doc_id, name = self._doc_identity(doc)
                logger.info(f"documentOpened/Created → {name} ({doc_id})")
                # Lazy session creation on activation
        except Exception as e:
            logger.debug(f"Open/create event handling skipped: {e}")

    def on_workspace_activated(self, args: adsk.core.WorkspaceEventArgs) -> None:
        """
        Handle workspace activation - shows palette when Fusion UI is ready.
        This solves the startup timing issue where palettes can't be shown during add-in initialization.
        """
        try:
            workspace = args.workspace
            logger.info(f"workspaceActivated → {getattr(workspace, 'name', 'Unknown')}")

            # Show the palette now that workspace is ready
            self._palette_manager.show_palette()

        except Exception as e:
            logger.exception(f"Error handling workspaceActivated: {e}")


# ---------------------------- adsk Document Event Handlers ----------------------------
class _DocumentActivatedHandler(adsk.core.DocumentEventHandler):
    def __init__(self, controller: "AgentController"):
        super().__init__()
        self._controller = controller

    def notify(self, args: adsk.core.DocumentEventArgs) -> None:
        self._controller.on_document_activated(args)


class _DocumentClosedHandler(adsk.core.DocumentEventHandler):
    def __init__(self, controller: "AgentController"):
        super().__init__()
        self._controller = controller

    def notify(self, args: adsk.core.DocumentEventArgs) -> None:
        self._controller.on_document_closed(args)


class _DocumentOpenedHandler(adsk.core.DocumentEventHandler):
    def __init__(self, controller: "AgentController"):
        super().__init__()
        self._controller = controller

    def notify(self, args: adsk.core.DocumentEventArgs) -> None:
        self._controller.on_document_opened_or_created(args)


class _DocumentCreatedHandler(adsk.core.DocumentEventHandler):
    def __init__(self, controller: "AgentController"):
        super().__init__()
        self._controller = controller

    def notify(self, args: adsk.core.DocumentEventArgs) -> None:
        self._controller.on_document_opened_or_created(args)


class _WorkspaceActivatedHandler(adsk.core.WorkspaceEventHandler):
    """Handles workspace activation to show palette when UI is ready."""
    def __init__(self, controller: "AgentController"):
        super().__init__()
        self._controller = controller

    def notify(self, args: adsk.core.WorkspaceEventArgs) -> None:
        self._controller.on_workspace_activated(args)


# ---------------------------------------------------------------------- Fusion entry points
def run(context: Any) -> None:
    """
    Entry point called by Fusion 360 when the add-in is loaded.

    Args:
        context: Fusion context dictionary
    """
    try:
        global _app, _controller
        _app = adsk.core.Application.get()
        if not _app:
            return

        logger.info(f"Starting CADAgent add-in v{config.COMPANY_NAME}")

        _controller = AgentController(_app)
        _controller.start()

        logger.info("CADAgent add-in started successfully")

    except Exception as exc:
        logger.exception("Failed to start CADAgent add-in")
        if _app and _app.userInterface:
            _app.userInterface.messageBox(
                f"Failed to start CADAgent add-in:\n\n{general_utils.format_exception(exc)}",
                "CADAgent Startup Error",
                adsk.core.MessageBoxButtonTypes.OKButtonType,
                adsk.core.MessageBoxIconTypes.CriticalIconType
            )
        raise


def stop(context: Any) -> None:
    """
    Entry point called by Fusion 360 when the add-in is stopped.

    Args:
        context: Fusion context dictionary
    """
    try:
        global _app, _controller
        logger.info("Stopping CADAgent add-in")

        if _controller:
            _controller.stop()
            _controller = None

        _app = None

        logger.info("CADAgent add-in stopped successfully")

    except Exception as exc:
        logger.exception("Error during add-in shutdown")
