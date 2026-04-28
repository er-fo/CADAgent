"""
FastAPI Backend for Fusion 360 LLM CAD Agent

This server handles WebSocket connections from Fusion 360 add-ins and orchestrates
LLM-driven CAD operations through sequential tool execution.
"""

import asyncio
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import jwt
import os
import json
import base64
from functools import lru_cache

SUPABASE_URL = os.environ.get("SUPABASE_URL")

# In local/dev, developers often don't configure Supabase. Make bypass automatic when no
# Supabase URL is provided so the WebSocket won't close immediately after "connected".
AUTH_BYPASS = (
    os.environ.get("CADAGENT_AUTH_BYPASS", os.environ.get("AUTH_BYPASS", "false"))
    .lower()
    in ("1", "true", "yes", "on")
) or not SUPABASE_URL


@lru_cache(maxsize=1)
def _get_jwk_client():
    """Return a cached PyJWKClient for Supabase if URL is configured."""
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL not configured")
    # Supabase JWKS is exposed at the well-known path without requiring an apikey
    jwks_url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
    return jwt.PyJWKClient(jwks_url)


def _decode_supabase_jwt(token: str) -> dict:
    """
    Validate a Supabase JWT using HS256 secret when available, otherwise JWKS.

    Supports HS256 (shared secret) and ES256/RS256 (via JWKS endpoint).
    """
    header = jwt.get_unverified_header(token)
    alg = header.get("alg")

    # Try shared secret for HS algorithms if provided
    if alg and alg.startswith("HS"):
        secret = os.environ.get("SUPABASE_JWT_SECRET") or os.environ.get("SUPABASE_JWT_SECRET_KEY")
        if not secret:
            raise RuntimeError("SUPABASE_JWT_SECRET not configured for HS* tokens")
        return jwt.decode(
            token,
            secret,
            algorithms=[alg],
            options={"require": ["exp", "sub"], "verify_aud": False},
        )

    # Fallback to JWKS (ES256/RS256)
    jwk_client = _get_jwk_client()
    signing_key = jwk_client.get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=[alg] if alg else None,
        options={"require": ["exp", "sub"], "verify_aud": False},
    )


def _extract_user_id_from_jwt(token: str) -> Optional[str]:
    """
    Extract user ID (sub claim) from a JWT without verifying signature.

    Used for rate limiting, where we need the user ID before validation.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("utf-8")))
        return payload.get("sub")
    except Exception:
        return None

try:
    from .websocket_manager import ConnectionManager
    from .agent_workflow import handle_execute_request, handle_planning_request, handle_revert_request
    from .session_logger import update_iteration_feedback
    from .rate_limiter import get_rate_limiter
except ImportError:  # pragma: no cover - allows running as a script
    from websocket_manager import ConnectionManager  # type: ignore
    from agent_workflow import handle_execute_request, handle_planning_request, handle_revert_request  # type: ignore
    from session_logger import update_iteration_feedback  # type: ignore
    from rate_limiter import get_rate_limiter  # type: ignore

# Configure logging with rotation to prevent disk exhaustion
# Log files rotate when they reach 10MB, keeping up to 5 backups
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console handler (unbuffered)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# File handler with rotation (10MB per file, 5 backups = 50MB max)
file_handler = RotatingFileHandler(
    'cadagent.log',
    maxBytes=10 * 1024 * 1024,  # 10MB per file
    backupCount=5                # Keep 5 backup files (cadagent.log.1, .log.2, etc)
)
file_handler.setFormatter(log_formatter)

# Reduce noise in cadagent.log while keeping cache metrics and warnings/errors.
class _CadagentFileFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        message = record.getMessage()
        # Keep cache metrics and no-op skip messages for performance monitoring
        return "EntityStore cache" in message or "Skipping entity context refresh" in message

file_handler.addFilter(_CadagentFileFilter())

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Reduce noisy library loggers in file output.
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Initialize WebSocket connection manager
manager = ConnectionManager()
session_tasks: Dict[str, asyncio.Task] = {}

# Locks to prevent race conditions when managing session tasks
# Ensures only one request can modify a session's task at a time
session_locks: Dict[str, asyncio.Lock] = {}


def _get_session_lock(session_id: str) -> asyncio.Lock:
    """
    Get or create a lock for the given session.

    Used to synchronize access to session_tasks dictionary.
    """
    if session_id not in session_locks:
        session_locks[session_id] = asyncio.Lock()
    return session_locks[session_id]


async def _check_rate_limit(session_id: str, message_type: str) -> tuple[bool, Optional[Dict]]:
    """
    Check if a user is within rate limits for a message type.

    Args:
        session_id: Session identifier
        message_type: Type of message being sent

    Returns:
        Tuple of (allowed, error_response) where error_response is a dict
        to send to client if rate limited, or None if allowed.
    """
    user_id = manager.get_user_id(session_id)
    if not user_id:
        # Not authenticated yet or couldn't extract user ID
        return True, None

    rate_limiter = get_rate_limiter()
    allowed, status = await rate_limiter.check_request_limit(user_id, tokens=1)

    if not allowed:
        return False, {
            "type": "rate_limit_error",
            "message": "Rate limit exceeded. Please wait a moment before sending more requests.",
            "details": status
        }

    return True, None


async def _launch_session_task(session_id: str, coro):
    """Run long-lived agent workflows without blocking the WebSocket receiver."""
    async def runner():
        task = asyncio.current_task()
        try:
            await coro
        except asyncio.CancelledError:
            logger.info("Background task for session %s cancelled", session_id)
            raise
        except Exception as exc:
            logger.exception("Background task for session %s failed", session_id)

            # Notify the client about the failure
            try:
                error_str = str(exc).lower()
                if "overloaded" in error_str or "529" in error_str:
                    error_type = "api_overloaded"
                    message = "The AI service is currently experiencing high load. Please try again in a few moments."
                elif "rate_limit" in error_str or "429" in error_str:
                    error_type = "rate_limited"
                    message = "Rate limit exceeded. Please wait a moment before trying again."
                elif "no anthropic api key" in error_str or "no openai api key" in error_str or "no google api key" in error_str:
                    error_type = "api_key_missing"
                    message = "No API key is configured for the selected model. Open Settings -> API Keys and add one, then retry."
                elif "parsed_output" in error_str and "extra inputs are not permitted" in error_str:
                    error_type = "backend_payload_error"
                    message = "A backend message-format issue occurred. Please retry. If this persists, contact support."
                else:
                    error_type = "execution_failed"
                    message = "An error occurred during execution. Please try again or check the logs for details."

                await manager.send_message(session_id, {
                    "type": "error",
                    "error": error_type,
                    "message": message
                })
            except Exception:
                logger.debug("Could not send error notification to session %s", session_id)
        finally:
            # Use lock to safely remove task when it completes
            async with _get_session_lock(session_id):
                if session_tasks.get(session_id) is task:
                    session_tasks.pop(session_id, None)

    # Acquire lock before checking and modifying session_tasks
    # This prevents race condition where multiple requests check at the same time
    async with _get_session_lock(session_id):
        existing = session_tasks.get(session_id)
        if existing and not existing.done():
            logger.warning("Cancelled existing task for session %s (superseded by new request)", session_id)
            await _cancel_session_task(session_id, reason="superseded")

        task = asyncio.create_task(runner())
        session_tasks[session_id] = task


async def _cancel_session_task(session_id: str, reason: str = "cancelled"):
    """
    Cancel any running workflow task for the session and flush stale results.

    Must be called with the session lock already held to prevent race conditions.
    """
    task = session_tasks.pop(session_id, None)
    if task and not task.done():
        try:
            task.cancel(reason)
        except TypeError:
            # Python versions without cancel(message)
            task.cancel()

    # Flush any orphaned queue entries that may have arrived before/during cancellation
    # to prevent them from being consumed by subsequent operations.
    flushed_results = manager.flush_pending_results(session_id)
    flushed_context = manager.flush_pending_entity_context(session_id)
    if flushed_results or flushed_context:
        logger.debug(
            "Flushed %d stale result(s) and %d stale entity-context payload(s) after cancelling task for session %s",
            flushed_results,
            flushed_context,
            session_id,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    logger.info("Starting CAD Agent Backend")
    yield
    logger.info("Shutting down CAD Agent Backend")


# Initialize FastAPI application
app = FastAPI(
    title="Fusion 360 LLM CAD Agent",
    description="Backend server for LLM-driven CAD modeling in Fusion 360",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
# WebSocket connections originate from Fusion 360 desktop application, not browsers.
# CORS wildcard is acceptable because:
# 1. Native apps don't send Origin headers (or send null)
# 2. Authentication via JWT tokens is enforced before any operations
# 3. All sensitive operations require valid user tokens verified against Supabase
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Native app connections (Fusion 360)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Fusion 360 LLM CAD Agent",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint

    Returns the current health status of the backend server.
    Used by monitoring tools and the Fusion 360 add-in to verify connectivity.
    """
    logger.debug("Health check requested")
    connection_stats = manager.get_connection_stats()
    return {
        "status": "healthy",
        "service": "cadagent-backend",
        "message": "Backend server is operational",
        **connection_stats,
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for Fusion 360 add-in connections.

    Handles bidirectional communication with Fusion 360 sessions:
    - Accepts connections and tracks them by session_id
    - Routes incoming messages based on message type
    - Handles disconnections gracefully

    Args:
        websocket: WebSocket connection object
        session_id: Unique identifier for the Fusion 360 session
    """
    accepted = await manager.connect(session_id, websocket)
    if not accepted:
        return

    # In bypass mode, trust the session immediately with a dummy user/token.
    if AUTH_BYPASS:
        manager.set_user_id(session_id, "dev-bypass")
        manager.set_user_token(session_id, "dev-bypass-token")
        manager.mark_authenticated(session_id, True)

    try:
        while True:
            # Receive message from Fusion 360
            data = await websocket.receive_json()
            logger.info(f"Received message from session {session_id}: {data.get('type', 'unknown')}")

            # Route message based on type
            message_type = data.get("type")

            # Enforce authentication first, allow only auth/heartbeat until verified
            allowed_pre_auth = {"authenticate", "update_api_keys"}
            if not AUTH_BYPASS and not manager.is_authenticated(session_id) and message_type not in allowed_pre_auth:
                logger.warning("Rejecting message before auth for session %s: %s", session_id, message_type)
                await manager.send_message(session_id, {
                    "type": "authentication_error",
                    "message": "Authenticate first."
                })
                await websocket.close(code=1008, reason="Authentication required")
                return

            if message_type == "execute_request":
                allowed, error = await _check_rate_limit(session_id, message_type)
                if not allowed:
                    await manager.send_message(session_id, error)
                else:
                    await _launch_session_task(session_id, handle_execute_request(session_id, data, manager))

            elif message_type == "planning_request":
                allowed, error = await _check_rate_limit(session_id, message_type)
                if not allowed:
                    await manager.send_message(session_id, error)
                else:
                    await _launch_session_task(session_id, handle_planning_request(session_id, data, manager))

            elif message_type in {"execution_result", "result"}:
                await manager.store_fusion_result(session_id, data)
                logger.debug(f"Stored execution result for session {session_id}")

            elif message_type in {"execution_error", "error"}:
                await manager.store_fusion_result(session_id, data)
                logger.warning(f"Error received from session {session_id}: {data.get('message', 'unknown')}")

            elif message_type == "plan_approval":
                await manager.store_fusion_result(session_id, data)
                logger.info(f"Plan approval received for session {session_id}: {data.get('approved')}")

            elif message_type == "question_tree_completed":
                # User completed the design exploration question tree
                await manager.store_fusion_result(session_id, data)
                logger.info(f"Question tree completed for session {session_id}")

            elif message_type == "design_selected":
                # User selected a design from the proposals
                await manager.store_fusion_result(session_id, data)
                logger.info(f"Design selected for session {session_id}: {data.get('design_id')}")

            elif message_type == "feature_snapshot":
                manager.set_feature_snapshot(session_id, data)
                await manager.store_fusion_result(session_id, data)
                logger.debug(f"Feature snapshot stored for session {session_id}")

            elif message_type == "entity_context_response":
                entity_context = data.get("entity_context")
                if entity_context:
                    queued_entity_context = dict(entity_context)
                    context_request_id = data.get("context_request_id")
                    message_id = data.get("message_id")
                    if context_request_id is not None:
                        queued_entity_context["context_request_id"] = context_request_id
                    if message_id is not None:
                        queued_entity_context["message_id"] = message_id
                    await manager.store_entity_context(session_id, queued_entity_context)
                    logger.debug(f"Entity context stored for session {session_id}: {len(entity_context.get('bodies', []))} bodies, {len(entity_context.get('faces', []))} faces, {len(entity_context.get('edges', []))} edges")
                else:
                    logger.warning(f"Received entity_context_response without entity_context data for session {session_id}")

            elif message_type == "status":
                # Handle status updates from Fusion 360
                logger.info(f"Status update from session {session_id}: {data.get('message', '')}")

            elif message_type == "cancel_request":
                # Cancel the currently running request
                async with _get_session_lock(session_id):
                    await _cancel_session_task(session_id, reason="user_cancel")
                # Note: cancelled message is sent by agent_workflow.py when CancelledError is caught
                logger.info(f"Request cancelled by user for session {session_id}")

            elif message_type == "revert_request":
                # Handle timeline revert request
                allowed, error = await _check_rate_limit(session_id, message_type)
                if not allowed:
                    await manager.send_message(session_id, error)
                else:
                    message_id = data.get("message_id")
                    logger.info(f"Revert request received for session {session_id} (message_id={message_id})")
                    await _launch_session_task(session_id, handle_revert_request(session_id, data, manager))

            elif message_type == "iteration_feedback":
                iteration = data.get("iteration")
                verdict = data.get("verdict")
                success = update_iteration_feedback(session_id, iteration, verdict)
                # best-effort ack back to Fusion client
                try:
                    await manager.send_message(session_id, {
                        "type": "iteration_feedback_ack",
                        "success": success,
                        "iteration": iteration,
                        "verdict": verdict,
                    })
                except Exception:
                    logger.debug("Could not send feedback ack to session %s", session_id)

            elif message_type == "update_api_keys":
                api_keys_payload = data.get("llm_api_keys") or data.get("api_keys")
                manager.set_llm_api_keys(session_id, api_keys_payload)
                logger.info("Updated API keys for session %s", session_id)
                await manager.send_message(session_id, {"type": "api_keys_updated"})

            elif message_type == "authenticate":
                if AUTH_BYPASS:
                    manager.set_user_id(session_id, "dev-bypass")
                    manager.set_user_token(session_id, "dev-bypass-token")
                    manager.mark_authenticated(session_id, True)
                    await manager.send_message(session_id, {
                        "type": "authentication_ack",
                        "authenticated": True
                    })
                    continue

                # Authentication is optional for command execution. If no token is
                # provided (or it is invalid), continue as anonymous.
                user_token = data.get("token")
                if not user_token:
                    logger.info("Session %s authenticated as anonymous", session_id)
                    manager.set_user_id(session_id, None)
                    manager.set_user_token(session_id, None)
                    manager.mark_authenticated(session_id, False)
                    await manager.send_message(session_id, {
                        "type": "authentication_ack",
                        "authenticated": False
                    })
                    continue

                # Validate JWT token immediately (signature + exp)
                try:
                    _decode_supabase_jwt(user_token)
                except jwt.ExpiredSignatureError:
                    logger.warning("Session %s sent expired auth token; continuing as anonymous", session_id)
                    manager.set_user_id(session_id, None)
                    manager.set_user_token(session_id, None)
                    manager.mark_authenticated(session_id, False)
                    await manager.send_message(session_id, {
                        "type": "authentication_error",
                        "message": "Your session has expired. Please sign in again."
                    })
                    await manager.send_message(session_id, {
                        "type": "authentication_ack",
                        "authenticated": False
                    })
                    continue
                except Exception as e:
                    logger.error("Session %s token validation failed; continuing as anonymous - %s", session_id, e)
                    manager.set_user_id(session_id, None)
                    manager.set_user_token(session_id, None)
                    manager.mark_authenticated(session_id, False)
                    await manager.send_message(session_id, {
                        "type": "authentication_error",
                        "message": "Authentication failed. Please sign in again."
                    })
                    await manager.send_message(session_id, {
                        "type": "authentication_ack",
                        "authenticated": False
                    })
                    continue

                # Extract user ID from token for rate limiting
                user_id = _extract_user_id_from_jwt(user_token)
                if user_id:
                    manager.set_user_id(session_id, user_id)
                    logger.debug(f"Extracted user ID for rate limiting: {user_id[:8]}...")

                # Store user's JWT token for usage tracking
                manager.set_user_token(session_id, user_token)
                manager.set_llm_api_keys(session_id, data.get("llm_api_keys") or data.get("api_keys"))
                manager.mark_authenticated(session_id, True)
                logger.info(f"Authentication received for session {session_id}: authenticated")
                await manager.send_message(session_id, {
                    "type": "authentication_ack",
                    "authenticated": True
                })

            else:
                logger.warning(f"Unknown message type from session {session_id}: {message_type}")

    except WebSocketDisconnect as exc:
        logger.info(f"WebSocket disconnected for session {session_id} (code={getattr(exc, 'code', 'unknown')})")
        manager.disconnect(session_id)
        async with _get_session_lock(session_id):
            await _cancel_session_task(session_id)
        # Clean up the lock to prevent memory leak
        session_locks.pop(session_id, None)
    except Exception as e:
        logger.error(f"Error in WebSocket connection for session {session_id}: {str(e)}")
        manager.disconnect(session_id)
        async with _get_session_lock(session_id):
            await _cancel_session_task(session_id)
        # Clean up the lock to prevent memory leak
        session_locks.pop(session_id, None)


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server with uvicorn")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
