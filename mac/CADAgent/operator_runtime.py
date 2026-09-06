"""
Local operator runtime for the CADAgent Fusion add-in.

This module intentionally stays independent from Fusion and the existing
controller implementation.  The runtime accepts a controller object plus a
main-thread dispatcher callback, exposes a small local HTTP API, and records
operator artifacts under runs/<run_id>/.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


JsonDict = Dict[str, Any]
Dispatcher = Callable[[Callable[[], Any]], Any]


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 0
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_REQUEST_BYTES = 10 * 1024 * 1024
MAX_EVENT_MEMORY = 1000
MAX_ARTIFACT_STRING = 4000

SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "jwt",
    "key",
    "password",
    "refresh_token",
    "secret",
    "session_token",
    "supabase_key",
    "token",
)

BLOB_KEY_PARTS = (
    "base64",
    "data_url",
    "image",
    "image_data",
    "screenshot",
)


def utc_timestamp() -> str:
    """Return an ISO-like UTC timestamp without relying on datetime timezone APIs."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_dumps(value: Any, pretty: bool = False) -> str:
    """Serialize JSON with deterministic, Fusion-safe defaults."""
    if pretty:
        return json.dumps(to_jsonable(value), indent=2, sort_keys=True)
    return json.dumps(to_jsonable(value), separators=(",", ":"), sort_keys=True)


def json_loads(data: bytes) -> Any:
    """Parse UTF-8 JSON request bytes."""
    if not data:
        return {}
    return json.loads(data.decode("utf-8"))


def to_jsonable(value: Any) -> Any:
    """Convert arbitrary controller values into JSON-compatible structures."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return "<%d bytes>" % len(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_jsonable(to_dict())
        except Exception:
            pass

    as_dict = getattr(value, "__dict__", None)
    if isinstance(as_dict, dict):
        public = {}
        for key, item in as_dict.items():
            if not str(key).startswith("_"):
                public[str(key)] = to_jsonable(item)
        if public:
            return public

    return str(value)


def _key_matches(key: str, parts: Tuple[str, ...]) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in parts)


def redact(value: Any, include_blobs: bool = True, max_string: int = MAX_ARTIFACT_STRING) -> Any:
    """Return a JSON-safe copy with secrets removed and optional blob summarization."""
    value = to_jsonable(value)
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            key_text = str(key)
            if _key_matches(key_text, SECRET_KEY_PARTS):
                redacted[key_text] = "<redacted>"
            elif include_blobs and _key_matches(key_text, BLOB_KEY_PARTS):
                redacted[key_text] = summarize_blob(item)
            else:
                redacted[key_text] = redact(item, include_blobs=include_blobs, max_string=max_string)
        return redacted
    if isinstance(value, list):
        return [redact(item, include_blobs=include_blobs, max_string=max_string) for item in value]
    if isinstance(value, str):
        if len(value) > max_string:
            return value[:max_string] + "...<truncated %d chars>" % (len(value) - max_string)
        if looks_like_secret(value):
            return "<redacted>"
        return value
    return value


def summarize_blob(value: Any) -> Any:
    """Summarize large image/blob payloads for artifacts without breaking responses."""
    value = to_jsonable(value)
    if isinstance(value, str):
        return "<blob %d chars>" % len(value)
    if isinstance(value, bytes):
        return "<blob %d bytes>" % len(value)
    if isinstance(value, dict):
        summary = {}
        for key, item in value.items():
            if _key_matches(str(key), SECRET_KEY_PARTS):
                summary[str(key)] = "<redacted>"
            elif isinstance(item, (str, bytes)) or _key_matches(str(key), BLOB_KEY_PARTS):
                summary[str(key)] = summarize_blob(item)
            else:
                summary[str(key)] = redact(item, include_blobs=True)
        return summary
    if isinstance(value, list):
        return "<blob list length=%d>" % len(value)
    return value


def looks_like_secret(value: str) -> bool:
    """Catch common token shapes even when a caller used a vague key name."""
    if len(value) < 24:
        return False
    prefixes = ("sk-", "sk_", "eyJ", "sb_secret_", "sb_publishable_")
    return value.startswith(prefixes)


class DispatchError(Exception):
    """Raised when a main-thread action fails or times out."""


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class OperatorRuntime:
    """Controller-agnostic local HTTP runtime for operator workflows."""

    def __init__(
        self,
        controller: Any,
        dispatcher: Optional[Dispatcher],
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        runs_dir: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.controller = controller
        self.dispatcher = dispatcher
        self.host = host or DEFAULT_HOST
        self.port = int(port or DEFAULT_PORT)
        self.timeout_seconds = float(timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
        self.runs_dir = Path(runs_dir) if runs_dir else Path(__file__).resolve().parent / "runs"

        self._server: Optional[_ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._events: List[JsonDict] = []
        self._event_seq = 0
        self._started_at: Optional[str] = None
        self._active_run_id: Optional[str] = None
        self._active_run_dir: Optional[Path] = None
        self._run_started_at: Optional[str] = None
        self._last_error: Optional[str] = None
        self._questions: List[Any] = []
        self._designs: List[Any] = []
        self._final_summary: JsonDict = {}
        self._verdict: JsonDict = {}

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> JsonDict:
        """Start the local HTTP server."""
        with self._lock:
            if self._server:
                return self.status_snapshot()

            self.runs_dir.mkdir(parents=True, exist_ok=True)
            handler_cls = self._make_handler_class()
            try:
                self._server = _ThreadingHTTPServer((self.host, self.port), handler_cls)
            except OSError:
                if self.port == 0:
                    raise
                self._server = _ThreadingHTTPServer((self.host, 0), handler_cls)
            self.host, self.port = self._server.server_address[:2]
            self._started_at = utc_timestamp()

            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                name="CADAgentOperatorRuntime",
            )
            self._server_thread.daemon = True
            self._server_thread.start()

        self.record_event("runtime_started", {"url": self.url})
        return self.status_snapshot()

    def stop(self) -> JsonDict:
        """Stop the local HTTP server."""
        server = None
        thread = None
        with self._lock:
            server = self._server
            thread = self._server_thread
            self._server = None
            self._server_thread = None

        if server:
            try:
                server.shutdown()
                server.server_close()
            except Exception as exc:
                self._last_error = str(exc)

        if thread and thread.is_alive():
            thread.join(2.0)

        self.record_event("runtime_stopped", {})
        return self.status_snapshot()

    @property
    def url(self) -> str:
        """Return the base URL for the running server."""
        return "http://%s:%s" % (self.host, self.port)

    def get_url(self) -> str:
        """Compatibility helper for integration code."""
        return self.url

    def get_active_run_dir(self) -> Optional[Path]:
        """Return the active run directory, if a run is active."""
        return self._ensure_run_dir()

    def is_running(self) -> bool:
        """Return whether the HTTP server is currently active."""
        return self._server is not None

    # ------------------------------------------------------------------ public records
    def record_backend_message(self, doc_id: Optional[str], message: Any) -> JsonDict:
        """Record a backend message and update operator-facing cached state."""
        safe_message = redact(message, include_blobs=True)
        if isinstance(message, dict):
            message_type = message.get("type")
            data = message.get("data")
            if message_type == "question_tree_generated":
                self.set_questions(data or message.get("questions") or [], doc_id=doc_id)
            elif message_type == "designs_proposed":
                self.set_designs(data or message.get("designs") or [], doc_id=doc_id)
            elif message_type in ("completed", "build_plan_completed"):
                self._final_summary = {
                    "doc_id": doc_id,
                    "message": safe_message,
                    "recorded_at": utc_timestamp(),
                }
                self._write_artifact("final_summary.json", self._final_summary)
            elif message_type == "error":
                self._verdict = {
                    "ok": False,
                    "doc_id": doc_id,
                    "error": safe_message,
                    "recorded_at": utc_timestamp(),
                }
                self._write_artifact("verdict.json", self._verdict)

        return self.record_event("backend_message", {"message": safe_message}, doc_id=doc_id)

    def record_execution_result(self, doc_id: Optional[str], result: Any) -> JsonDict:
        """Record an execution result emitted by controller/tool execution."""
        safe_result = redact(result, include_blobs=True)
        if isinstance(result, dict):
            ok = bool(result.get("success", result.get("ok", False)))
            self._verdict = {
                "ok": ok,
                "doc_id": doc_id,
                "result": safe_result,
                "recorded_at": utc_timestamp(),
            }
            self._write_artifact("verdict.json", self._verdict)
        return self.record_event("execution_result", {"result": safe_result}, doc_id=doc_id)

    def record_event(
        self,
        event_type: str,
        payload: Optional[Any] = None,
        doc_id: Optional[str] = None,
    ) -> JsonDict:
        """Append an event to memory and the active run's events.jsonl."""
        with self._lock:
            self._event_seq += 1
            event = {
                "seq": self._event_seq,
                "timestamp": utc_timestamp(),
                "type": event_type,
                "doc_id": doc_id,
                "payload": redact(payload or {}, include_blobs=True),
            }
            self._events.append(event)
            if len(self._events) > MAX_EVENT_MEMORY:
                self._events = self._events[-MAX_EVENT_MEMORY:]
            self._append_jsonl("events.jsonl", event)
            self._write_status_locked()
            return event

    def record_command(
        self,
        command: str,
        payload: Optional[Any] = None,
        result: Optional[Any] = None,
        ok: bool = True,
    ) -> JsonDict:
        """Append a command record to commands.jsonl."""
        record = {
            "timestamp": utc_timestamp(),
            "command": command,
            "ok": bool(ok),
            "payload": redact(payload or {}, include_blobs=True),
            "result": redact(result or {}, include_blobs=True),
        }
        self._append_jsonl("commands.jsonl", record)
        return record

    def status_snapshot(self) -> JsonDict:
        """Return a redacted runtime and controller status snapshot."""
        controller_status = self._dispatch_action(
            "status_snapshot",
            {},
            timeout_seconds=5.0,
            allow_missing=True,
        )
        if not isinstance(controller_status, dict):
            controller_status = {}

        with self._lock:
            snapshot = {
                "ok": True,
                "runtime": {
                    "running": self.is_running(),
                    "host": self.host,
                    "port": self.port,
                    "url": self.url,
                    "started_at": self._started_at,
                    "last_error": self._last_error,
                },
                "run": {
                    "id": self._active_run_id,
                    "started_at": self._run_started_at,
                    "dir": str(self._active_run_dir) if self._active_run_dir else None,
                },
                "controller": controller_status,
                "events": {
                    "count": len(self._events),
                    "last_seq": self._event_seq,
                },
                "questions": {
                    "count": len(self._questions),
                },
                "designs": {
                    "count": len(self._designs),
                },
            }
            return redact(snapshot, include_blobs=False)

    def events_since(self, since_seq: int = 0) -> List[JsonDict]:
        """Return in-memory events after a sequence number."""
        with self._lock:
            return [event for event in self._events if int(event.get("seq", 0)) > since_seq]

    def set_questions(self, questions: Any, doc_id: Optional[str] = None) -> JsonDict:
        """Set cached operator questions for GET /questions."""
        with self._lock:
            if isinstance(questions, dict):
                self._questions = [questions]
            elif isinstance(questions, list):
                self._questions = questions
            else:
                self._questions = [questions] if questions else []
        return self.record_event("questions_updated", {"questions": self._questions}, doc_id=doc_id)

    def set_designs(self, designs: Any, doc_id: Optional[str] = None) -> JsonDict:
        """Set cached operator design options for GET /designs."""
        with self._lock:
            if isinstance(designs, dict):
                self._designs = [designs]
            elif isinstance(designs, list):
                self._designs = designs
            else:
                self._designs = [designs] if designs else []
        return self.record_event("designs_updated", {"designs": self._designs}, doc_id=doc_id)

    def start_run(self, payload: Optional[Any] = None) -> JsonDict:
        """Create a new run directory and initialize artifacts."""
        run_id = self._new_run_id(payload)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            self._active_run_id = run_id
            self._active_run_dir = run_dir
            self._run_started_at = utc_timestamp()
            self._questions = []
            self._designs = []
            self._final_summary = {}
            self._verdict = {"ok": None, "recorded_at": self._run_started_at}

            self._touch_run_file("commands.jsonl")
            self._touch_run_file("events.jsonl")
            self._write_artifact("final_summary.json", {"status": "pending", "run_id": run_id})
            self._write_artifact("verdict.json", self._verdict)
            self._write_status_locked()

        self.record_event("run_started", {"run_id": run_id, "payload": payload or {}})
        return {"run_id": run_id, "run_dir": str(run_dir), "started_at": self._run_started_at}

    def end_run(self, payload: Optional[Any] = None, controller_result: Optional[Any] = None) -> JsonDict:
        """Finalize the active run artifacts."""
        payload = payload or {}
        safe_payload = redact(payload, include_blobs=True)
        safe_result = redact(controller_result or {}, include_blobs=True)
        ended_at = utc_timestamp()

        summary = {
            "run_id": self._active_run_id,
            "started_at": self._run_started_at,
            "ended_at": ended_at,
            "payload": safe_payload,
            "controller_result": safe_result,
        }
        verdict = {
            "ok": bool(payload.get("ok", payload.get("success", True))),
            "run_id": self._active_run_id,
            "ended_at": ended_at,
            "payload": safe_payload,
        }

        self._final_summary = summary
        self._verdict = verdict
        self._write_artifact("final_summary.json", summary)
        self._write_artifact("verdict.json", verdict)
        self.record_event("run_ended", {"summary": summary, "verdict": verdict})
        return {"summary": summary, "verdict": verdict}

    # ------------------------------------------------------------------ HTTP routing
    def _make_handler_class(self) -> Any:
        runtime = self

        class OperatorRequestHandler(BaseHTTPRequestHandler):
            server_version = "CADAgentOperatorRuntime/1.0"

            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_OPTIONS(self) -> None:
                self._send_json(200, {"ok": True})

            def do_GET(self) -> None:
                runtime._handle_http(self, "GET")

            def do_POST(self) -> None:
                runtime._handle_http(self, "POST")

            def _send_json(self, status_code: int, payload: Any) -> None:
                data = (json_dumps(redact(payload, include_blobs=False)) + "\n").encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()
                self.wfile.write(data)

        return OperatorRequestHandler

    def _handle_http(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        parsed = urlparse(handler.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query or "")

        try:
            if method == "GET":
                payload = {}
            else:
                payload = self._read_request_json(handler)

            status_code, result = self._route(method, path, query, payload)
            handler._send_json(status_code, result)  # type: ignore[attr-defined]
        except DispatchError as exc:
            self._last_error = str(exc)
            handler._send_json(504, {"ok": False, "error": str(exc)})  # type: ignore[attr-defined]
        except ValueError as exc:
            handler._send_json(400, {"ok": False, "error": str(exc)})  # type: ignore[attr-defined]
        except Exception as exc:
            self._last_error = str(exc)
            self.record_event(
                "operator_error",
                {"error": str(exc), "traceback": traceback.format_exc(), "path": path},
            )
            handler._send_json(500, {"ok": False, "error": str(exc)})  # type: ignore[attr-defined]

    def _read_request_json(self, handler: BaseHTTPRequestHandler) -> JsonDict:
        length_header = handler.headers.get("Content-Length", "0")
        try:
            length = int(length_header)
        except Exception:
            raise ValueError("Invalid Content-Length")
        if length > MAX_REQUEST_BYTES:
            raise ValueError("Request body exceeds %d bytes" % MAX_REQUEST_BYTES)
        data = handler.rfile.read(length) if length else b""
        parsed = json_loads(data)
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise ValueError("JSON request body must be an object")
        return parsed

    def _route(
        self,
        method: str,
        path: str,
        query: Dict[str, List[str]],
        payload: JsonDict,
    ) -> Tuple[int, JsonDict]:
        if method == "GET" and path == "/health":
            snapshot = self.status_snapshot()
            return 200, {
                "ok": True,
                "status": "healthy",
                "url": self.url,
                "runtime": snapshot.get("runtime", {}),
                "controller": snapshot.get("controller", {}),
            }

        if method == "GET" and path == "/status":
            return 200, self.status_snapshot()

        if method == "GET" and path == "/events":
            since = self._query_int(query, "since", self._query_int(query, "since_event_id", 0))
            events = self.events_since(since)
            next_seq = self._event_seq
            return 200, {"ok": True, "events": events, "next_seq": next_seq}

        if method == "POST" and path == "/run/start":
            run = self.start_run(payload)
            controller_result = self._dispatch_action("run_start", payload, allow_missing=True)
            result = {"ok": True, "run": run, "controller": controller_result}
            self.record_command("run_start", payload, result, ok=True)
            return 200, result

        if method == "POST" and path == "/run/end":
            controller_result = self._dispatch_action("run_end", payload, allow_missing=True)
            ended = self.end_run(payload, controller_result)
            result = {"ok": True, "run": ended, "controller": controller_result}
            self.record_command("run_end", payload, result, ok=True)
            return 200, result

        if method == "POST" and path == "/prompt":
            result = self._dispatch_action("prompt", payload)
            self.record_event("prompt_submitted", {"payload": payload, "result": result}, doc_id=self._payload_doc_id(payload))
            self.record_command("prompt", payload, result, ok=True)
            return 200, {"ok": True, "result": result}

        if method == "POST" and path == "/planning/prompt":
            result = self._dispatch_action("planning_prompt", payload)
            self.record_event("planning_prompt_submitted", {"payload": payload, "result": result}, doc_id=self._payload_doc_id(payload))
            self.record_command("planning_prompt", payload, result, ok=True)
            return 200, {"ok": True, "result": result}

        if method == "POST" and path == "/planning/approve":
            result = self._dispatch_action("planning_approve", payload)
            self.record_event("planning_approved", {"payload": payload, "result": result}, doc_id=self._payload_doc_id(payload))
            self.record_command("planning_approve", payload, result, ok=True)
            return 200, {"ok": True, "result": result}

        if method == "GET" and path == "/questions":
            controller_result = self._dispatch_action("questions", {}, allow_missing=True)
            questions = self._extract_items(controller_result, "questions", self._questions)
            return 200, {"ok": True, "questions": questions, "controller": controller_result}

        if method == "POST" and path == "/questions/answer":
            result = self._dispatch_action("question_answer", payload)
            self.record_event("question_answered", {"payload": payload, "result": result}, doc_id=self._payload_doc_id(payload))
            self.record_command("question_answer", payload, result, ok=True)
            return 200, {"ok": True, "result": result}

        if method == "GET" and path == "/designs":
            controller_result = self._dispatch_action("designs", {}, allow_missing=True)
            designs = self._extract_items(controller_result, "designs", self._designs)
            return 200, {"ok": True, "designs": designs, "controller": controller_result}

        if method == "POST" and path == "/documents/new":
            result = self._dispatch_action("document_new", payload)
            self.record_event("document_created", {"payload": payload, "result": result}, doc_id=self._payload_doc_id(payload))
            self.record_command("document_new", payload, result, ok=True)
            return 200, {"ok": True, "result": result}

        if method == "POST" and path == "/designs/select":
            result = self._dispatch_action("design_select", payload)
            self.record_event("design_selected", {"payload": payload, "result": result}, doc_id=self._payload_doc_id(payload))
            self.record_command("design_select", payload, result, ok=True)
            return 200, {"ok": True, "result": result}

        if method == "POST" and path == "/camera/view":
            result = self._dispatch_action("camera_view", payload)
            self.record_event("camera_view_requested", {"payload": payload, "result": result}, doc_id=self._payload_doc_id(payload))
            self.record_command("camera_view", payload, result, ok=True)
            return 200, {"ok": True, "result": result}

        if method == "POST" and path == "/screenshot":
            result = self._dispatch_action("screenshot", payload)
            self.record_event("screenshot_requested", {"payload": payload, "result": result}, doc_id=self._payload_doc_id(payload))
            self.record_command("screenshot", payload, result, ok=True)
            return 200, {"ok": True, "result": result}

        return 404, {"ok": False, "error": "Not found", "path": path}

    # ------------------------------------------------------------------ dispatch
    def _dispatch_action(
        self,
        action: str,
        payload: JsonDict,
        timeout_seconds: Optional[float] = None,
        allow_missing: bool = False,
    ) -> Any:
        """Run a controller action through the injected main-thread dispatcher."""
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        done = threading.Event()
        result_queue: "queue.Queue[Tuple[bool, Any]]" = queue.Queue()

        def work() -> Any:
            try:
                result = self._invoke_controller_action(action, payload, allow_missing=allow_missing)
                result_queue.put((True, result))
                return result
            except Exception as exc:
                result_queue.put((False, exc))
                return None
            finally:
                done.set()

        if self.dispatcher is None:
            work()
        else:
            try:
                self.dispatcher(work)
            except TypeError:
                try:
                    self.dispatcher(action, payload, work)  # type: ignore[misc]
                except Exception as exc:
                    result_queue.put((False, exc))
                    done.set()
            except Exception as exc:
                result_queue.put((False, exc))
                done.set()

        if not done.wait(timeout):
            raise DispatchError("Timed out waiting for controller action '%s'" % action)

        ok, value = result_queue.get()
        if not ok:
            raise DispatchError("Controller action '%s' failed: %s" % (action, value))
        return redact(value, include_blobs=False)

    def _invoke_controller_action(self, action: str, payload: JsonDict, allow_missing: bool = False) -> Any:
        if self.controller is None:
            if allow_missing:
                return {"handled": False, "reason": "controller unavailable"}
            raise RuntimeError("controller unavailable")

        for method_name in ("handle_operator_action", "dispatch_operator_action", "operator_action"):
            method = getattr(self.controller, method_name, None)
            if callable(method):
                return method(action, payload)

        handled, result = self._fallback_controller_action(action, payload)
        if handled:
            return result
        if allow_missing:
            return {"handled": False, "action": action}
        raise RuntimeError("controller does not support operator action '%s'" % action)

    def _fallback_controller_action(self, action: str, payload: JsonDict) -> Tuple[bool, Any]:
        if action == "status_snapshot":
            return True, self._controller_status()

        if action == "prompt":
            return self._submit_prompt(payload, planning_mode=False)

        if action == "planning_prompt":
            return self._submit_prompt(payload, planning_mode=True)

        if action == "planning_approve":
            return self._approve_plan(payload)

        method_map = {
            "document_new": ("create_operator_document", "create_document"),
            "run_start": ("start_operator_run", "operator_run_start"),
            "run_end": ("end_operator_run", "operator_run_end"),
            "questions": ("get_operator_questions", "get_questions"),
            "question_answer": ("answer_operator_question", "answer_question"),
            "designs": ("get_operator_designs", "get_designs"),
            "design_select": ("select_operator_design", "select_design"),
            "camera_view": ("set_operator_camera_view", "set_camera_view", "get_camera_view"),
            "screenshot": ("capture_operator_screenshot", "capture_screenshot"),
        }

        for method_name in method_map.get(action, ()):
            method = getattr(self.controller, method_name, None)
            if callable(method):
                return True, self._call_controller_method(method, payload)

        return False, None

    def _submit_prompt(self, payload: JsonDict, planning_mode: bool) -> Tuple[bool, Any]:
        method = getattr(self.controller, "submit_user_request", None)
        if not callable(method):
            return False, None

        text = self._payload_text(payload)
        if not text:
            raise ValueError("Missing prompt text")

        kwargs = {
            "include_visual_context": bool(payload.get("include_visual_context", payload.get("visual_context", False))),
            "model_name": payload.get("model_name", payload.get("model", "claude-sonnet-4.6")),
            "request_id": payload.get("request_id"),
            "image_data": payload.get("image_data"),
            "image_format": payload.get("image_format", "png"),
            "reasoning_effort": payload.get("reasoning_effort"),
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        try:
            result = method(text, planning_mode, **kwargs)
        except TypeError:
            result = method(text, planning_mode)

        return True, {
            "accepted": True,
            "planning_mode": planning_mode,
            "result": result,
            "doc_id": self._safe_call("get_active_doc_id"),
            "session_id": self._safe_call("get_session_id"),
        }

    def _approve_plan(self, payload: JsonDict) -> Tuple[bool, Any]:
        approved = bool(payload.get("approved", True))
        if not approved:
            return True, {"accepted": True, "approved": False}

        plan_text = str(payload.get("plan_text") or payload.get("plan") or "")
        doc_id = payload.get("doc_id") or self._safe_call("get_active_doc_id")

        method = getattr(self.controller, "handle_plan_approval_on_main_thread", None)
        if callable(method):
            result = method(plan_text, doc_id)
            return True, {"accepted": True, "approved": True, "result": result}

        for method_name in ("approve_operator_plan", "approve_plan"):
            method = getattr(self.controller, method_name, None)
            if callable(method):
                return True, self._call_controller_method(method, payload)

        return False, None

    def _call_controller_method(self, method: Callable[..., Any], payload: JsonDict) -> Any:
        try:
            return method(payload)
        except TypeError:
            try:
                return method(**payload)
            except TypeError:
                return method()

    def _controller_status(self) -> JsonDict:
        status = {
            "active_doc_id": self._safe_call("get_active_doc_id"),
            "session_id": self._safe_call("get_session_id"),
            "connected": self._safe_call("is_connected"),
        }
        get_backend_target = getattr(self.controller, "get_active_backend_target", None)
        if callable(get_backend_target):
            try:
                status["backend_target"] = redact(get_backend_target(), include_blobs=False)
            except Exception:
                pass
        api_status = self._safe_call("get_api_keys_status")
        if isinstance(api_status, dict):
            status["api_keys"] = redact(api_status, include_blobs=False)
        return status

    def _safe_call(self, method_name: str) -> Any:
        method = getattr(self.controller, method_name, None)
        if not callable(method):
            return None
        try:
            return method()
        except Exception:
            return None

    # ------------------------------------------------------------------ artifacts
    def _new_run_id(self, payload: Optional[Any]) -> str:
        payload = payload if isinstance(payload, dict) else {}
        requested = str(payload.get("run_id") or payload.get("id") or "").strip()
        if requested:
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in requested)
            return safe[:80] or uuid.uuid4().hex
        return "%s-%s" % (time.strftime("%Y%m%d-%H%M%S", time.gmtime()), uuid.uuid4().hex[:8])

    def _ensure_run_dir(self) -> Optional[Path]:
        with self._lock:
            if self._active_run_dir:
                return self._active_run_dir
        return None

    def _touch_run_file(self, name: str) -> None:
        run_dir = self._ensure_run_dir()
        if not run_dir:
            return
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8"):
            pass

    def _append_jsonl(self, name: str, record: Any) -> None:
        run_dir = self._ensure_run_dir()
        if not run_dir:
            return
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json_dumps(record) + "\n")

    def _write_artifact(self, name: str, payload: Any) -> None:
        run_dir = self._ensure_run_dir()
        if not run_dir:
            return
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json_dumps(redact(payload, include_blobs=True), pretty=True) + "\n", encoding="utf-8")
        os.replace(str(tmp_path), str(path))

    def _write_status_locked(self) -> None:
        run_dir = self._active_run_dir
        if not run_dir:
            return
        status = {
            "run_id": self._active_run_id,
            "started_at": self._run_started_at,
            "runtime_started_at": self._started_at,
            "event_seq": self._event_seq,
            "event_count": len(self._events),
            "question_count": len(self._questions),
            "design_count": len(self._designs),
            "updated_at": utc_timestamp(),
        }
        path = run_dir / "status.json"
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json_dumps(redact(status, include_blobs=True), pretty=True) + "\n", encoding="utf-8")
        os.replace(str(tmp_path), str(path))

    # ------------------------------------------------------------------ small helpers
    @staticmethod
    def _query_int(query: Dict[str, List[str]], name: str, default: int) -> int:
        values = query.get(name)
        if not values:
            return default
        try:
            return int(values[0])
        except Exception:
            return default

    @staticmethod
    def _payload_text(payload: JsonDict) -> str:
        return str(payload.get("prompt") or payload.get("text") or payload.get("request_text") or "").strip()

    @staticmethod
    def _payload_doc_id(payload: JsonDict) -> Optional[str]:
        doc_id = payload.get("doc_id") or payload.get("document_id")
        return str(doc_id) if doc_id else None

    @staticmethod
    def _extract_items(controller_result: Any, key: str, fallback: List[Any]) -> Any:
        if isinstance(controller_result, dict):
            if key in controller_result:
                return controller_result.get(key)
            data = controller_result.get("data")
            if isinstance(data, dict) and key in data:
                return data.get(key)
        return fallback


__all__ = [
    "OperatorRuntime",
    "DispatchError",
    "json_dumps",
    "json_loads",
    "redact",
    "to_jsonable",
    "utc_timestamp",
]
