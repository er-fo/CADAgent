"""
Palette UI Manager for CADAgent

Manages the HTML palette UI that provides a persistent, visible interface
for the CADAgent add-in with real-time feedback and activity logging.
"""

import adsk.core
import json
import logging
import os
import threading
import time
import webbrowser
from typing import Any, Dict, Optional, List, Tuple

from . import config

# Auth message types that should never be deferred or dropped
AUTH_MESSAGE_TYPES = frozenset(['auth_success', 'auth_error', 'user_profile'])

# Critical messages that should be dispatched as soon as palette exists (no visibility/handshake required)
# Messages that must reach the palette even if the custom send event isn't available yet.
CRITICAL_MESSAGE_TYPES = frozenset(['connection_status', 'auth_success', 'auth_error', 'user_profile', 'document_switched'])
SUPPORT_CONTACT_LINE = "If issue persists, email erik@cadagent.co"
SUPPORT_CONTACT_MESSAGE_TYPES = frozenset(['error', 'auth_error', 'api_keys_error'])

logger = logging.getLogger(__name__)


def _append_support_contact(message: str) -> str:
    """Append support contact details to user-facing error messages."""
    if not message:
        return SUPPORT_CONTACT_LINE
    if SUPPORT_CONTACT_LINE.lower() in message.lower():
        return message
    separator = "\n\n" if "\n" in message else " "
    return f"{message}{separator}{SUPPORT_CONTACT_LINE}"


class PaletteSendEventHandler(adsk.core.CustomEventHandler):
    """Custom event handler to marshal palette sends onto the UI thread."""

    def __init__(self, palette_manager: "PaletteManager"):
        super().__init__()
        self._palette_manager = palette_manager

    def notify(self, args: adsk.core.CustomEventArgs) -> None:
        try:
            event_args = adsk.core.CustomEventArgs.cast(args)
            info = getattr(event_args, 'additionalInfo', '') or ''
            payload: Dict[str, Any] = {}
            if info:
                try:
                    payload = json.loads(info)
                except Exception:
                    logger.warning(f"Palette send event payload parse failed: {info}")
                    payload = {}

            action = payload.get('action', 'send')

            logger.info(
                f"[palette_send_event] action={action}, "
                f"thread={threading.current_thread().name}, payload_keys={list(payload.keys())}"
            )

            if action == 'flush':
                with self._palette_manager._retry_lock:
                    pending_count = len(self._palette_manager._pending_messages)
                logger.debug(f"[palette_send_event] flush requested (pending={pending_count})")
                logger.info("Palette send event: flush pending messages")
                self._palette_manager._flush_pending_messages_if_ready()
                with self._palette_manager._retry_lock:
                    has_pending = bool(self._palette_manager._pending_messages)
                if has_pending:
                    self._palette_manager._schedule_retry_flush()
                else:
                    self._palette_manager._retry_delay_sec = 0.3
                return

            message_type = payload.get('message_type')
            doc_id = payload.get('doc_id')
            kwargs = payload.get('kwargs') or {}

            if not message_type:
                logger.warning("Palette send event missing message_type")
                return

            # Already on UI thread; call send_message directly
            self._palette_manager.send_message(message_type, doc_id=doc_id, **kwargs)
        except Exception as e:
            logger.error(f"Palette send event handler error: {e}", exc_info=True)


class PaletteManager:
    """Manages the CADAgent HTML palette UI."""

    PALETTE_ID = "CADAgentPalette"
    PALETTE_NAME = "CADAgent"
    PALETTE_WIDTH = 400
    PALETTE_HEIGHT = 600

    def __init__(self, app: adsk.core.Application, controller: "AgentController"):
        self._app = app
        self._ui = app.userInterface
        self._controller = controller
        self._palette: Optional[adsk.core.Palette] = None
        self._handlers = []
        self._is_visible = False  # Track if palette has been shown
        self._pending_messages: List[Tuple[str, Optional[str], Dict[str, Any]]] = []  # Queue messages until palette + handshake ready
        self._handshake_received = False
        self._bootstrap_sent = False
        # Retry mechanism for failed message sends
        self._retry_timer: Optional[threading.Timer] = None
        self._retry_delay_sec = 0.3  # Start at 300ms
        self._retry_lock = threading.Lock()  # Protect pending queue access
        # Main-thread palette send custom event
        self._send_event_id = 'CADAgentPaletteSend'
        self._send_event = None
        self._send_event_handler = None

    def start(self) -> None:
        """Create the palette (but don't show it yet - call show_palette() when workspace is ready)."""
        try:
            logger.info("=" * 60)
            logger.info("STARTING PALETTE MANAGER")
            logger.info("=" * 60)

            # Ensure palette object and handlers exist (handles startup and post-workspace switch recreation)
            self._ensure_palette()

            logger.info("=" * 60)
            logger.info("PALETTE MANAGER CREATED (waiting for workspace activation to show)")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ FAILED TO CREATE PALETTE: {e}", exc_info=True)
            raise

    def _ensure_palette(self) -> None:
        """
        Create the palette if it doesn't exist or was destroyed by a workspace switch.
        Fusion deletes palettes when changing workspaces, leaving stale handles.
        """
        palettes = self._ui.palettes
        try:
            existing = palettes.itemById(self.PALETTE_ID)
        except Exception:
            existing = None

        has_valid_handle = (
            self._palette is not None
            and (not hasattr(self._palette, "isValid") or bool(getattr(self._palette, "isValid", True)))
        )

        if has_valid_handle and existing is not None:
            # Keep reference in sync with collection instance.
            self._palette = existing
            return

        # If a palette with this ID already exists, remove it before add().
        # This happens on add-in restarts when stop() didn't fully clean up.
        if existing is not None:
            try:
                deleted = existing.deleteMe()
                if deleted:
                    logger.info(f"Deleted stale palette before recreation: {self.PALETTE_ID}")
                else:
                    logger.warning(f"deleteMe() returned False for stale palette: {self.PALETTE_ID}")
            except Exception as e:
                logger.warning(f"Failed to delete stale palette {self.PALETTE_ID}: {e}")

            try:
                existing = palettes.itemById(self.PALETTE_ID)
            except Exception:
                existing = None
            if existing is not None:
                raise RuntimeError(f"Palette ID '{self.PALETTE_ID}' is still present; cannot recreate palette.")

        # Reset handshake/bootstrap because new HTML will need to handshake again
        self._is_visible = False
        self._handshake_received = False
        self._bootstrap_sent = False

        logger.info("Creating (or recreating) palette instance")

        # Acquire HTML path
        html_file = os.path.join(
            os.path.dirname(__file__),
            'resources',
            'html',
            'index.html'
        )
        if not os.path.exists(html_file):
            raise FileNotFoundError(f"HTML file not found: {html_file}")

        logger.info(f"HTML file exists: ✓ ({html_file})")

        # Create palette
        self._palette = palettes.add(
            self.PALETTE_ID,
            self.PALETTE_NAME,
            html_file,
            True,   # showCloseButton
            True,   # isResizable
            False,  # isVisible (keep hidden until workspace ready)
            self.PALETTE_WIDTH,
            self.PALETTE_HEIGHT
        )

        logger.info(f"✓ Created palette: {self.PALETTE_NAME}")

        # Docking
        try:
            desired_state = adsk.core.PaletteDockingStates.PaletteDockStateLeft
            if self._palette.dockingState != desired_state:
                logger.info("Setting palette docking state → Left")
                self._palette.dockingState = desired_state
            else:
                logger.info("Palette already docked on the left")
        except AttributeError:
            logger.warning("Palette API does not expose dockingState; skipping docking configuration")
        except RuntimeError as err:
            logger.warning(f"Failed to set palette docking state: {err}")

        # Register HTML event handler (one per palette instance)
        on_html_event = HTMLEventHandler(self._controller, self)
        self._palette.incomingFromHTML.add(on_html_event)
        self._handlers.append(on_html_event)
        logger.info("✓ HTML event handler registered")

        # Register custom event to marshal palette sends onto the UI thread (register once)
        if not self._send_event:
            try:
                # Make registration idempotent across restart/crash scenarios.
                try:
                    self._app.unregisterCustomEvent(self._send_event_id)
                except Exception:
                    pass
                self._send_event = self._app.registerCustomEvent(self._send_event_id)
                self._send_event_handler = PaletteSendEventHandler(self)
                self._send_event.add(self._send_event_handler)
                self._handlers.append(self._send_event_handler)
                logger.info("✓ Registered palette send custom event handler")
            except Exception as e:
                logger.error(f"❌ Failed to register palette send event: {e}")

            # Get or create the palette
        logger.info("Palette ensure complete")

    def show_palette(self) -> None:
        """
        Show the palette and send initial messages.
        Called by WorkspaceActivated event handler when Fusion UI is ready.
        """
        # Recreate palette if Fusion destroyed it during workspace switch
        self._ensure_palette()

        if self._is_visible:
            logger.info("Palette already visible, skipping show_palette()")
            return

        if not self._palette:
            logger.error("Cannot show palette: palette not created yet")
            return

        try:
            logger.info("=" * 60)
            logger.info("SHOWING PALETTE (workspace now ready)")
            logger.info("=" * 60)

            # Show the palette (docking state already set in start())
            logger.info("Making palette visible...")
            self._palette.isVisible = True
            self._is_visible = True
            logger.info(f"✓ Palette visible: {self._is_palette_visible()}")

            # Bootstrap once both palette visibility and handshake are confirmed
            self.send_bootstrap_if_ready()
            self._flush_pending_messages_if_ready()

            logger.info("=" * 60)
            logger.info("PALETTE NOW VISIBLE AND READY")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ FAILED TO SHOW PALETTE: {e}", exc_info=True)

    def stop(self) -> None:
        """Hide and clean up the palette."""
        try:
            # Cancel any pending retry timer
            self._cancel_retry_timer()

            self._is_visible = False
            self._handshake_received = False
            self._bootstrap_sent = False

            if self._palette:
                try:
                    self._palette.isVisible = False
                except Exception:
                    pass

                try:
                    deleted = self._palette.deleteMe()
                    logger.info(f"Palette deleted: {deleted}")
                except Exception as e:
                    logger.warning(f"Failed to delete palette during stop: {e}")
                finally:
                    self._palette = None

            if self._send_event_handler and self._send_event:
                try:
                    self._send_event.remove(self._send_event_handler)
                except Exception:
                    pass

            try:
                self._app.unregisterCustomEvent(self._send_event_id)
            except Exception as e:
                logger.debug(f"Custom event unregister skipped/failed ({self._send_event_id}): {e}")

            self._send_event = None
            self._send_event_handler = None

            with self._retry_lock:
                self._pending_messages.clear()

            self._handlers.clear()

        except Exception as e:
            logger.error(f"Failed to stop palette: {e}")

    def send_message(self, message_type: str, doc_id: Optional[str] = None, **kwargs) -> None:
        """
        Send a message to the HTML palette.

        Args:
            message_type: Type of message (e.g., 'log', 'connection_status')
            **kwargs: Additional message data
        """
        if message_type in SUPPORT_CONTACT_MESSAGE_TYPES and isinstance(kwargs.get("message"), str):
            kwargs = dict(kwargs)
            kwargs["message"] = _append_support_contact(kwargs["message"])

        logger.info(f"→ send_message called: type={message_type}, doc_id={doc_id}, kwargs={kwargs}")

        # Recreate palette if it was destroyed (e.g., workspace change)
        self._ensure_palette()

        # Fast-path when custom send event is unavailable: send directly even off UI thread
        if not self._send_event and self._palette is not None:
            try:
                message = {'type': message_type, **kwargs}
                if doc_id:
                    message['doc_id'] = doc_id
                message_json = json.dumps(message)
                self._palette.sendInfoToHTML('cadagent_message', message_json)
                logger.info(f"✓ Message sent to palette via fast-path (no send_event): {message_type}")
                return
            except Exception as e:
                logger.error(f"❌ Fast-path send failed for {message_type}: {e}; falling back to queue")

        # If we're not on the main/UI thread, queue and schedule a flush on the UI thread
        if threading.current_thread() is not threading.main_thread():
            logger.debug("send_message called off main thread; enqueueing for UI-thread flush")
            self._enqueue_message(message_type, doc_id, kwargs)
            # Try immediate flush via custom event; fall back to timed retry
            if not self._fire_flush_event():
                self._schedule_retry_flush()
            return

        # Critical messages (auth + connection_status) can bypass handshake/visibility checks,
        # but only if the palette is actually functional (not in startup state)
        ready_for_critical = message_type in CRITICAL_MESSAGE_TYPES and self._can_send_critical_to_palette()

        if not self._can_send_to_palette() and not ready_for_critical:
            logger.info(
                "Queuing message until palette is ready "
                f"(visible={self._is_palette_visible()}, handshake={self._handshake_received})"
            )
            self._enqueue_message(message_type, doc_id, kwargs)
            # Critical messages are essential for auth/connection state; try immediate UI-thread flush
            if message_type in CRITICAL_MESSAGE_TYPES:
                sent = self._fire_flush_event()
                if not sent:
                    logger.debug("Critical message flush via event failed; scheduling retry")
                    self._schedule_retry_flush()
            else:
                self._schedule_retry_flush()
            return

        try:
            message = {'type': message_type, **kwargs}
            if doc_id:
                message['doc_id'] = doc_id
            message_json = json.dumps(message)
            logger.info(f"→ Message JSON: {message_json}")

            # Send to HTML - this triggers window.fusionJavaScriptHandler.handle('cadagent_message', ...)
            self._palette.sendInfoToHTML('cadagent_message', message_json)

            logger.info(f"✓ Message sent to palette: {message_type}")

        except RuntimeError as e:
            # RuntimeError typically means palette isn't ready (startup race condition)
            # Queue the message but DON'T immediately retry - let natural timer handle it
            logger.warning(f"❌ Palette not ready for {message_type} (RuntimeError: {e})")
            self._enqueue_message(message_type, doc_id, kwargs)
            # Don't call _schedule_retry_flush() here - prevents infinite loop during startup
            # The retry timer from the initial enqueue will eventually fire
        except Exception as e:
            logger.error(f"❌ Failed to send message to palette: {e}", exc_info=True)
            # Re-queue the message for retry - critical for auth messages
            logger.info(f"Re-queuing failed message: {message_type}")
            self._enqueue_message(message_type, doc_id, kwargs)
            self._schedule_retry_flush()

    def _fire_flush_event(self) -> bool:
        """Fire custom event to flush pending messages on the UI thread."""
        try:
            if not self._send_event:
                logger.warning("Palette send event not registered; cannot flush via event")
                return False
            self._app.fireCustomEvent(self._send_event_id, json.dumps({"action": "flush"}))
            return True
        except Exception as e:
            logger.error(f"Failed to fire palette flush event: {e}", exc_info=True)
            return False

    def send_fusion_ready(self, extra: Optional[Dict[str, Any]] = None) -> None:
        """Send Fusion bridge handshake details to the palette."""
        logger.info("→ send_fusion_ready called")

        if not self._palette:
            logger.warning("❌ Cannot send fusionReady: palette is None")
            return

        if not self._is_palette_visible():
            logger.warning("❌ Cannot send fusionReady: palette not visible")
            return

        payload: Dict[str, Any] = {
            'session_id': self._controller.get_session_id(),
            'backend_connected': self._controller.is_connected(),
            'timestamp': time.time(),
            'testing_mode': config.TESTING_MODE,
            'testing_email': config.TEST_EMAIL
        }

        if extra:
            for key, value in extra.items():
                if value is not None:
                    payload[key] = value

        message_json = json.dumps(payload)
        logger.info(f"→ fusionReady payload: {message_json}")

        try:
            self._palette.sendInfoToHTML('fusionReady', message_json)
            logger.info("✓ fusionReady message sent to palette")
        except Exception as e:
            logger.error(f"❌ Failed to send fusionReady message: {e}", exc_info=True)

    def mark_handshake_received(self) -> None:
        """Record handshake and attempt to bootstrap + flush pending messages."""
        self._handshake_received = True
        # Reset retry delay since handshake succeeded
        self._retry_delay_sec = 0.3
        self.send_bootstrap_if_ready()
        # After handshake, proactively request/push profile so UI reflects auth state
        if not self._controller.is_auth_bypass():
            try:
                profile = self._controller.get_user_profile()
                if profile:
                    self.send_message('user_profile', profile=profile)
            except Exception as e:
                logger.debug(f"Deferred profile fetch after handshake failed: {e}")
        self._flush_pending_messages_if_ready()

    def send_bootstrap_if_ready(self) -> None:
        """Send initial status/logs once palette is visible and handshake complete."""
        if not self._is_palette_visible():
            logger.debug("Bootstrap deferred: palette not visible")
            return
        if not self._handshake_received:
            logger.debug("Bootstrap deferred: handshake not yet received")
            return
        if self._bootstrap_sent:
            return

        self._bootstrap_sent = True

        # Initial log and connection state
        self.send_log('success', 'CADAgent loaded successfully!')
        try:
            self.send_log('info', f'Backend: {config.backend_label()}')
        except Exception:
            self.send_log('info', 'Backend: <unknown>')

        sid = self._controller.get_session_id()
        if sid:
            self.send_log('info', f'Session: {sid[:8]}...')

        active_doc_id = self._controller.get_active_doc_id()
        self.send_connection_status(doc_id=active_doc_id)

        # In bypass mode, immediately surface a synthetic auth_success + user_profile
        if self._controller.is_auth_bypass():
            self.send_message('auth_success', message='Auth bypass enabled (dev)', user={'email': 'dev-bypass@cadagent.local'})
            self.send_message('user_profile', profile={'email': 'dev-bypass@cadagent.local'})

        if active_doc_id and sid:
            try:
                doc_name = None
                session_info = self._controller._sessions.get(active_doc_id, {})
                doc_name = session_info.get("doc_name")
                if not doc_name:
                    try:
                        doc_name = getattr(self._controller._app.activeDocument, "name", None)
                    except Exception:
                        doc_name = None
                self.send_document_switched(active_doc_id, doc_name or "Active Document", sid)
                logger.info(f"✓ Sent document_switched for {doc_name or active_doc_id}")
            except Exception as e:
                logger.warning(f"Failed to send initial document_switched: {e}")

        # Drain any queued messages now that the palette is fully ready
        self._flush_pending_messages_if_ready()

    def _is_palette_functional(self) -> bool:
        """
        Check if the palette is functional (can query isVisible without RuntimeError).
        During startup, the palette object exists but isn't fully initialized - any
        attempt to access properties like isVisible throws RuntimeError: pArea.
        """
        if not self._palette:
            return False
        try:
            # Just accessing isVisible is enough to test functionality
            _ = self._palette.isVisible
            return True
        except Exception:
            # Palette not fully initialized during startup
            return False

    def _is_palette_visible(self) -> bool:
        """Safely check if palette is visible, handling startup race conditions."""
        if not self._palette:
            return False
        try:
            return bool(getattr(self._palette, 'isVisible', False))
        except Exception:
            # Palette not fully initialized during startup - isVisible access fails
            # Fall back to our internal visibility flag to avoid hard startup failures.
            return bool(self._is_visible)

    def _can_send_to_palette(self) -> bool:
        return self._palette is not None and self._handshake_received and self._is_palette_visible()

    def _can_send_critical_to_palette(self) -> bool:
        """
        Critical messages (auth + connection_status) can be sent even if handshake/visibility
        flags haven't been set, BUT only if the palette is actually functional.
        During startup, the palette exists but isn't ready - we must wait.
        """
        # Palette must exist AND be functional (isVisible doesn't throw RuntimeError)
        return self._is_palette_functional()

    def _flush_pending_messages_if_ready(self) -> None:
        """Send any queued messages once palette and handshake are ready."""
        # Allow critical messages (auth + connection_status) to flush as soon as palette exists,
        # even if handshake/visibility not set
        critical_pending = False
        with self._retry_lock:
            if self._pending_messages:
                critical_pending = any(mt in CRITICAL_MESSAGE_TYPES for (mt, _, _) in self._pending_messages)
            if not self._pending_messages:
                return

            # For non-critical messages we require handshake before checking palette visibility.
            # This avoids early isVisible probes during Fusion startup/reopen races.
            if not self._handshake_received and not critical_pending:
                return

            can_send_normal = self._can_send_to_palette()
            can_send_critical = critical_pending and self._can_send_critical_to_palette()
            if not can_send_normal and not can_send_critical:
                return

            logger.info(f"Flushing {len(self._pending_messages)} pending palette messages")
            # Use a copy to avoid mutation issues if send_message queues again (it can now on failure)
            pending = list(self._pending_messages)
            self._pending_messages.clear()

        # Send outside the lock to avoid blocking other operations
        for message_type, doc_id, kwargs in pending:
            self.send_message(message_type, doc_id=doc_id, **kwargs)

    def _enqueue_message(self, message_type: str, doc_id: Optional[str], kwargs: Dict[str, Any]) -> None:
        """
        Add a message to the pending queue with deduplication for critical messages.

        Critical messages (auth_success, auth_error, user_profile, connection_status) are coalesced -
        newer messages of the same type replace older ones to prevent stale state.
        """
        with self._retry_lock:
            # For critical messages, replace any existing message of the same type
            if message_type in CRITICAL_MESSAGE_TYPES:
                # Remove any existing message of this type
                self._pending_messages = [
                    (mt, did, kw) for (mt, did, kw) in self._pending_messages
                    if mt != message_type
                ]
                logger.debug(f"Critical message {message_type} coalesced (replacing older)")

            self._pending_messages.append((message_type, doc_id, kwargs))

    def _schedule_retry_flush(self) -> None:
        """Schedule a retry flush with exponential backoff."""
        with self._retry_lock:
            # Don't schedule if timer already active
            if self._retry_timer is not None:
                return
            
            delay = self._retry_delay_sec
            logger.info(f"Scheduling retry flush in {delay:.2f}s")
            def _post_flush_request():
                with self._retry_lock:
                    self._retry_timer = None
                # Fire custom event to flush on UI thread
                if not self._fire_flush_event():
                    # If firing fails, fall back to immediate flush (may still be off-thread, but better than drop)
                    self._flush_pending_messages_if_ready()
            
            # Increase delay for next retry (exponential backoff, capped at 5s)
            self._retry_delay_sec = min(self._retry_delay_sec * 2, 5.0)

            self._retry_timer = threading.Timer(delay, _post_flush_request)
            self._retry_timer.daemon = True
            self._retry_timer.start()

    def _do_retry_flush(self) -> None:
        """Execute the scheduled retry flush."""
        # Retained for compatibility; not used directly now that we marshal via custom event.
        logger.info("Retry flush invoked")
        self._flush_pending_messages_if_ready()
        with self._retry_lock:
            has_pending = bool(self._pending_messages)
        if has_pending:
            self._schedule_retry_flush()
        else:
            self._retry_delay_sec = 0.3

    def _cancel_retry_timer(self) -> None:
        """Cancel any pending retry timer."""
        with self._retry_lock:
            if self._retry_timer is not None:
                self._retry_timer.cancel()
                self._retry_timer = None
            self._retry_delay_sec = 0.3

    def send_log(
        self,
        level: str,
        message: str,
        doc_id: Optional[str] = None,
        *,
        scope: Optional[str] = None,
        dismiss_hero: Optional[bool] = None,
        message_format: Optional[str] = None,
    ) -> None:
        """Send a log message to the palette."""
        extra: Dict[str, Any] = {}
        if scope is not None:
            extra["scope"] = scope
        if dismiss_hero is not None:
            extra["dismiss_hero"] = dismiss_hero
        if message_format:
            extra["format"] = message_format
        self.send_message('log', doc_id=doc_id, level=level, message=message, **extra)

    def send_connection_status(self, doc_id: Optional[str] = None) -> None:
        """Send current connection status to the palette."""
        logger.info(f"→ send_connection_status called (doc_id={doc_id})")
        connected = self._controller.is_connected(doc_id)
        session_id = self._controller.get_session_id_for_doc(doc_id)

        logger.info(f"→ Connection status: connected={connected}, session_id={session_id}, doc_id={doc_id}")

        self.send_message('connection_status', doc_id=doc_id, connected=connected, session_id=session_id)

    def send_progress(self, message: str, progress: Optional[int] = None, doc_id: Optional[str] = None) -> None:
        """Send progress update to the palette."""
        self.send_message('progress', doc_id=doc_id, message=message, progress=progress)

    def hide_progress(self, doc_id: Optional[str] = None) -> None:
        """Hide progress indicator in the palette."""
        self.send_message('hide_progress', doc_id=doc_id)

    def send_completed(self, message: str = "Request completed", doc_id: Optional[str] = None) -> None:
        """Send completion message to the palette."""
        self.send_message('completed', doc_id=doc_id, message=message)

    def send_error(self, message: str, doc_id: Optional[str] = None) -> None:
        """Send error message to the palette."""
        self.send_message('error', doc_id=doc_id, message=message)

    def send_document_switched(self, doc_id: str, doc_name: str, session_id: str) -> None:
        """Notify HTML that the active document context switched.

        Args:
            doc_id: Stable identifier for the Fusion document
            doc_name: Display name of the document/tab
            session_id: Active backend session id for this document
        """
        # Bypass the custom event queue for this critical message if needed
        if not self._send_event:
            logger.debug("document_switched sent without send_event (critical fast-path)")
            message = {'type': 'document_switched', 'doc_id': doc_id, 'doc_name': doc_name, 'session_id': session_id}
            try:
                self._palette.sendInfoToHTML('cadagent_message', json.dumps(message))
                logger.info("✓ document_switched delivered via fast-path")
            except Exception as e:
                logger.error(f"❌ Failed to fast-path document_switched: {e}")
                # Fall back to normal send_message (will enqueue/retry)
                self.send_message('document_switched', doc_id=doc_id, doc_name=doc_name, session_id=session_id)
            return

        self.send_message('document_switched', doc_id=doc_id, doc_name=doc_name, session_id=session_id)


class HTMLEventHandler(adsk.core.HTMLEventHandler):
    """Handles events from the HTML palette."""

    def __init__(self, controller: "AgentController", palette_manager: PaletteManager):
        super().__init__()
        self._controller = controller
        self._palette_manager = palette_manager

    def notify(self, args: adsk.core.HTMLEventArgs) -> None:
        """Handle incoming messages from HTML."""
        logger.info("=" * 60)
        logger.info("← HTML EVENT RECEIVED")
        logger.info("=" * 60)

        try:
            html_args = adsk.core.HTMLEventArgs.cast(args)
            action = getattr(html_args, 'action', None)
            data_str = getattr(html_args, 'data', None) or ''

            logger.info(f"← Fusion action value: {action}")
            logger.info(f"← Raw data: {data_str}")

            # Validate that this is our expected action
            if action != 'messageFromPalette':
                logger.warning(f"← Unexpected action from palette: {action}")
                return

            payload: Dict[str, Any] = {}

            if data_str:
                try:
                    payload = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("← Failed to parse JSON payload from palette")
                    payload = {}

            if not isinstance(payload, dict):
                payload = {}

            action_name = payload.get('action')
            logger.info(f"← Parsed action: {action_name}")
            logger.info(f"← Parsed data: {payload}")

            if action_name == 'handshake':
                logger.info("← Handshake request received")
                extra = {
                    'app_version': payload.get('app_version'),
                    'client_timestamp': payload.get('timestamp')
                }
                self._palette_manager.send_fusion_ready(extra)
                self._palette_manager.send_connection_status(doc_id=self._controller.get_active_doc_id())
                if not self._controller.get_active_doc_id():
                    # Rebind to the current document if initialization raced the first handshake.
                    # This can fail if user isn't authenticated yet (fresh launch) - that's OK,
                    # session will be created later after auth succeeds.
                    try:
                        self._controller._activate_session_for_current_document()
                    except Exception as session_exc:
                        logger.info(f"← Session activation deferred (expected pre-auth): {session_exc}")
                # CRITICAL: Always mark handshake received so non-critical messages (log, llm_message,
                # completed) can flow to the palette. Without this, _can_send_to_palette() returns False.
                self._palette_manager.mark_handshake_received()

            elif action_name == 'log_to_fusion':
                # Palette log forwarding disabled to reduce noise.
                return

            elif action_name == 'execute_request':
                request = payload.get('request', '').strip()
                planning_mode = payload.get('planning_mode', True)
                include_visual_context = bool(payload.get('include_visual_context', False))
                model_name = payload.get('model_name', 'claude-sonnet-4.5')
                reasoning_effort = payload.get('reasoning_effort')
                request_id = payload.get('request_id')  # Extract request_id for checkpoint correlation

                # Extract image data if present
                image_data = payload.get('image_data')
                image_format = payload.get('image_format', 'png')

                active_doc_id = self._controller.get_active_doc_id()

                # Log with image indicator
                has_image = " + image" if image_data else ""
                logger.info(f"← Execute request: '{request}'{has_image} (planning={planning_mode}, visual_context={include_visual_context}, model={model_name}, reasoning={reasoning_effort}, request_id={request_id})")

                if request or image_data:  # Allow image-only requests
                    log_msg = f'Executing request (Planning: {planning_mode}, Model: {model_name})'
                    if image_data:
                        log_msg += ' [with sketch]'
                    self._palette_manager.send_log('info', log_msg, doc_id=active_doc_id)

                    # Pass image data to controller
                    self._controller.submit_user_request(
                        request,
                        planning_mode,
                        include_visual_context,
                        model_name,
                        request_id,
                        image_data=image_data,
                        image_format=image_format,
                        reasoning_effort=reasoning_effort
                    )
                else:
                    logger.warning("← Empty request received (no text or image)")
                    self._palette_manager.send_error('Request cannot be empty', doc_id=active_doc_id)

            elif action_name == 'get_status':
                logger.info("← Status request received")
                # Auto-heal: each status poll attempts to (re)bind the active document
                # and re-establish the websocket if it dropped (e.g., backend restart).
                try:
                    self._controller._activate_session_for_current_document()
                except Exception as exc:  # defensive; don't break UI on reconnect attempt
                    logger.debug(f"Deferred session activation during get_status failed: {exc}")
                self._palette_manager.send_connection_status(doc_id=self._controller.get_active_doc_id())

            elif action_name == 'reconnect_request':
                logger.info("← Reconnect request received")
                self._controller.handle_reconnect_request()
                self._palette_manager.send_connection_status(doc_id=self._controller.get_active_doc_id())

            elif action_name == 'cancel_request':
                logger.info("← Cancel request received")
                # Forward cancel request to backend via WebSocket
                client = self._controller._get_active_ws_client()
                if client and client.is_connected():
                    client.send_json({"type": "cancel_request"})
                    logger.info("✓ Cancel request forwarded to backend")
                else:
                    logger.warning("Cannot forward cancel request: backend not connected")

            elif action_name == 'revert_request':
                message_id = payload.get('message_id')
                logger.info(f"← Revert request received (message_id={message_id})")
                if not message_id:
                    logger.warning("Revert request missing message_id")
                    self._palette_manager.send_error('Revert request missing message_id', doc_id=self._controller.get_active_doc_id())
                else:
                    # Forward revert request to backend via WebSocket
                    client = self._controller._get_active_ws_client()
                    if client and client.is_connected():
                        client.send_json({
                            "type": "revert_request",
                            "message_id": message_id
                        })
                        logger.info(f"✓ Revert request forwarded to backend (message_id={message_id})")
                    else:
                        logger.warning("Cannot forward revert request: backend not connected")
                        self._palette_manager.send_error('Backend not connected', doc_id=self._controller.get_active_doc_id())

            elif action_name == 'iteration_feedback':
                iteration = payload.get('iteration')
                verdict = payload.get('verdict')
                logger.info(f"← Iteration feedback received (iteration={iteration}, verdict={verdict})")
                client = self._controller._get_active_ws_client()
                if client and client.is_connected():
                    client.send_json({
                        "type": "iteration_feedback",
                        "iteration": iteration,
                        "verdict": verdict,
                    })
                    self._palette_manager.send_log('info', f'Feedback saved: iteration {iteration} → {verdict}', doc_id=self._controller.get_active_doc_id())
                else:
                    logger.warning("Cannot forward iteration feedback: backend not connected")
                    self._palette_manager.send_error('Backend not connected', doc_id=self._controller.get_active_doc_id())

            elif action_name == 'send_to_backend':
                logger.info("← send_to_backend request received")
                message = payload.get('message')
                if not isinstance(message, dict):
                    logger.warning("send_to_backend payload missing or invalid 'message'")
                    self._palette_manager.send_error('Invalid message payload', doc_id=self._controller.get_active_doc_id())
                else:
                    client = self._controller._get_active_ws_client()
                    if client and client.is_connected():
                        client.send_json(message)
                        logger.info(f"✓ Forwarded message to backend (type={message.get('type', 'unknown')})")
                    else:
                        logger.warning("Cannot forward send_to_backend: backend not connected")
                        self._palette_manager.send_error('Backend not connected', doc_id=self._controller.get_active_doc_id())

            elif action_name == 'send_magic_link':
                email = payload.get('email', '').strip()
                logger.info(f"← Send magic link request received (email={email})")
                if not email:
                    logger.warning("Email is required for magic link")
                    self._palette_manager.send_message('auth_error', message='Email is required')
                else:
                    try:
                        result = self._controller.send_magic_link(email)
                        self._palette_manager.send_message('auth_success', message=result.get('message', 'Magic link sent!'))
                    except Exception as e:
                        logger.error(f"Failed to send magic link: {e}")
                        self._palette_manager.send_message('auth_error', message=str(e))

            elif action_name == 'check_and_handle_signup':
                email = payload.get('email', '').strip()
                logger.info(f"← Check and handle signup request received (email={email})")
                if self._controller.is_auth_bypass():
                    logger.info("[auth] bypass enabled; skipping Supabase signup/login flow")
                    self._palette_manager.send_message(
                        'auth_success',
                        message='Auth bypass enabled (dev)',
                        user={'email': 'dev-bypass@cadagent.local'}
                    )
                    self._palette_manager.send_message('user_profile', profile={'email': 'dev-bypass@cadagent.local'})
                    return
                if not email:
                    logger.warning("Email is required for signup/login")
                    self._palette_manager.send_message('auth_error', message='Email is required')
                else:
                    # Run in background thread to avoid blocking palette event handler
                    def _do_check_and_signup():
                        try:
                            result = self._controller.check_and_handle_signup(email)
                            if not result.get('needs_otp'):
                                logger.warning(f"[auth] Unexpected response format: {result}")
                                self._palette_manager.send_message('auth_error', message='Unexpected authentication response')
                                return

                            if result.get('is_new_user'):
                                logger.info(f"[auth] New user created for {email}; OTP required for first login")
                            else:
                                logger.info(f"[auth] Existing user - OTP sent to {email}")

                            self._palette_manager.send_message(
                                'auth_otp_required',
                                message=result.get('message', 'Code sent! Check your email.')
                            )
                        except Exception as e:
                            logger.error(f"Failed to check and handle signup: {e}")
                            self._palette_manager.send_message('auth_error', message=str(e))

                    threading.Thread(target=_do_check_and_signup, daemon=True).start()
                    logger.info(f"[auth] Check and signup started in background thread")

            elif action_name == 'send_otp_code':
                email = payload.get('email', '').strip()
                logger.info(f"← Send OTP code request received (email={email})")
                if self._controller.is_auth_bypass():
                    logger.info("[auth] bypass enabled; ignoring OTP send request")
                    self._palette_manager.send_message(
                        'auth_success',
                        message='Auth bypass enabled (dev)',
                        user={'email': 'dev-bypass@cadagent.local'}
                    )
                    return
                if not email:
                    logger.warning("Email is required for OTP code")
                    self._palette_manager.send_message('auth_error', message='Email is required')
                else:
                    # Run in background thread to avoid blocking palette event handler
                    def _do_send_otp():
                        try:
                            result = self._controller.send_otp_code(email)
                            # Now safe to dispatch directly; send_message will marshal if needed
                            self._palette_manager.send_message('auth_success', message=result.get('message', 'Code sent!'))
                        except Exception as e:
                            logger.error(f"Failed to send OTP code: {e}")
                            self._palette_manager.send_message('auth_error', message=str(e))
                    
                    threading.Thread(target=_do_send_otp, daemon=True).start()
                    logger.info(f"[auth] OTP send started in background thread")

            elif action_name == 'verify_otp_code':
                email = payload.get('email', '').strip()
                code = payload.get('code', '').strip()
                logger.info(f"← Verify OTP code request received (email={email})")
                if self._controller.is_auth_bypass():
                    logger.info("[auth] bypass enabled; skipping OTP verification")
                    self._palette_manager.send_message(
                        'auth_success',
                        message='Auth bypass enabled (dev)',
                        user={'email': 'dev-bypass@cadagent.local'}
                    )
                    self._palette_manager.send_message('user_profile', profile={'email': 'dev-bypass@cadagent.local'})
                    return
                if not email or not code:
                    logger.warning("Email and code are required for OTP verification")
                    self._palette_manager.send_message('auth_error', message='Email and code are required')
                else:
                    # Run verification in background thread to avoid blocking the palette event handler.
                    # Fusion's sendInfoToHTML doesn't work reliably when called during a long-running
                    # synchronous operation within the notify callback.
                    def _do_verify():
                        try:
                            result = self._controller.verify_otp_code(email, code)
                            user_email = result.get('user', {}).get('email', 'Unknown')
                            logger.info(f"[auth] OTP verification succeeded for {user_email}, scheduling auth_success")
                            # Dispatch via thread-safe send_message (main-thread marshalled)
                            self._palette_manager.send_message('auth_success', message=f'Logged in as {user_email}', user=result.get('user'))
                            # Profile fetch is optional - don't let it fail the login
                            try:
                                profile = self._controller.get_user_profile()
                                if profile:
                                    self._palette_manager.send_message('user_profile', profile=profile)
                            except Exception as profile_err:
                                logger.warning(f"Profile fetch failed after successful login (non-fatal): {profile_err}")
                        except Exception as e:
                            logger.error(f"Failed to verify OTP code: {e}")
                            self._palette_manager.send_message('auth_error', message=str(e))
                    
                    # Start verification in background thread
                    verify_thread = threading.Thread(target=_do_verify, daemon=True)
                    verify_thread.start()
                    logger.info(f"[auth] OTP verification started in background thread")

            elif action_name == 'login_with_password':
                logger.info("← Password login request received (disabled)")
                self._palette_manager.send_message(
                    'auth_error',
                    message='Password login is disabled. Use email code verification (OTP).'
                )

            elif action_name == 'auth_callback':
                access_token = payload.get('access_token', '').strip()
                refresh_token = payload.get('refresh_token', '').strip()
                logger.info("← Auth callback received")
                if self._controller.is_auth_bypass():
                    logger.info("[auth] bypass enabled; ignoring auth callback")
                    self._palette_manager.send_message(
                        'auth_success',
                        message='Auth bypass enabled (dev)',
                        user={'email': 'dev-bypass@cadagent.local'}
                    )
                    self._palette_manager.send_message('user_profile', profile={'email': 'dev-bypass@cadagent.local'})
                    return
                if not access_token or not refresh_token:
                    logger.warning("Missing tokens in auth callback")
                    self._palette_manager.send_message('auth_error', message='Missing authentication tokens')
                else:
                    try:
                        result = self._controller.handle_auth_callback(access_token, refresh_token)
                        user_email = result.get('user', {}).get('email', 'Unknown')
                        self._palette_manager.send_message('auth_success', message=f'Logged in as {user_email}', user=result.get('user'))
                        # Fetch and send profile
                        profile = self._controller.get_user_profile()
                        if profile:
                            self._palette_manager.send_message('user_profile', profile=profile)
                    except Exception as e:
                        logger.error(f"Failed to handle auth callback: {e}")
                        self._palette_manager.send_message('auth_error', message=str(e))

            elif action_name == 'logout':
                logger.info("← Logout request received")
                try:
                    self._controller.logout()
                    self._palette_manager.send_message('logout_success', message='Logged out successfully')
                except Exception as e:
                    logger.error(f"Failed to logout: {e}")
                    self._palette_manager.send_message('auth_error', message=str(e))

            elif action_name == 'get_profile':
                logger.info("← Get profile request received")
                try:
                    profile = self._controller.get_user_profile()
                    if profile:
                        self._palette_manager.send_message('user_profile', profile=profile)
                    else:
                        self._palette_manager.send_message('auth_error', message='Not authenticated')
                except Exception as e:
                    logger.error(f"Failed to get profile: {e}")
                    self._palette_manager.send_message('auth_error', message=str(e))

            elif action_name == 'get_api_keys_status':
                logger.info("← Get API keys status request received")
                # Auth guard: require valid session
                if not self._controller.is_authenticated():
                    logger.warning("get_api_keys_status rejected: not authenticated")
                    self._palette_manager.send_message('auth_error', message='Not authenticated')
                else:
                    try:
                        status = self._controller.get_api_keys_status()
                        self._palette_manager.send_message('api_keys_status', status=status)
                    except Exception as e:
                        logger.error(f"Failed to get API keys status: {e}")
                        self._palette_manager.send_message('api_keys_error', message=str(e))

            elif action_name == 'save_api_keys':
                logger.info("← Save API keys request received")
                # Auth guard: require valid session
                if not self._controller.is_authenticated():
                    logger.warning("save_api_keys rejected: not authenticated")
                    self._palette_manager.send_message('auth_error', message='Not authenticated')
                else:
                    keys = payload.get('keys', {})

                    def _do_save_keys():
                        try:
                            success = self._controller.save_api_keys(keys)
                            if success:
                                # Get updated status for UI
                                status = self._controller.get_api_keys_status()
                                self._palette_manager.send_message('api_keys_saved', success=True, status=status)
                                logger.info("API keys saved successfully")
                            else:
                                self._palette_manager.send_message('api_keys_saved', success=False, message='Failed to save API keys')
                        except Exception as e:
                            logger.error(f"Failed to save API keys: {e}")
                            self._palette_manager.send_message('api_keys_error', message=str(e))

                    threading.Thread(target=_do_save_keys, daemon=True).start()
                    logger.info("[api_keys] Save keys started in background thread")

            elif action_name == 'open_external_url':
                url = payload.get('url', '').strip()
                logger.info(f"← Open external URL request received: {url}")

                if not url:
                    logger.warning("open_external_url: URL is empty")
                elif not (url.startswith('http://') or url.startswith('https://')):
                    logger.warning(f"open_external_url: Invalid URL protocol: {url}")
                else:
                    try:
                        logger.info(f"Opening external URL in system browser: {url}")
                        webbrowser.open(url)
                        self._palette_manager.send_log('info', f'Opening {url} in browser', doc_id=self._controller.get_active_doc_id())
                    except Exception as e:
                        logger.error(f"Failed to open external URL: {e}")
                        self._palette_manager.send_error(f'Failed to open URL: {str(e)}', doc_id=self._controller.get_active_doc_id())

            else:
                logger.warning(f"← Unknown action from palette: {action_name}")

            # Send acknowledgment back to JavaScript
            html_args.returnData = 'OK'
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ Failed to handle HTML event: {e}", exc_info=True)
            self._palette_manager.send_error(f"Error: {str(e)}", doc_id=self._controller.get_active_doc_id())
            # Send failure acknowledgment
            try:
                html_args.returnData = 'FAILED'
            except:
                pass
