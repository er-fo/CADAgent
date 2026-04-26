"""
Claude API Client for Fusion 360 LLM CAD Agent

Handles communication with Claude 4.5 Sonnet for agentic CAD operations.
Implements tool-based execution with sequential operation flow.
"""

import os
import base64
import logging
import asyncio
import copy
import json
import re
import random
from datetime import datetime, timezone
from html import unescape
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
import markdown

# Optional: Google Gemini (google-genai)
try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
except Exception:  # pragma: no cover - handled gracefully when SDK missing
    genai = None
    genai_types = None
    genai_errors = None

from .prompt_builder import build_full_prompt, PROMPT_VERSION
from .session_logger import _extract_usage_stats

logger = logging.getLogger(__name__)

# Dedicated logger for raw LLM responses (tool-level debugging)
_tool_logger = logging.getLogger("cadagent_tool")
if not _tool_logger.handlers:
    _tool_logger.setLevel(logging.INFO)


def _decode_jwt_exp(token: str) -> Optional[int]:
    """Extract exp claim from a JWT without verifying signature."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("utf-8")))
        return int(payload.get("exp")) if "exp" in payload else None
    except Exception:
        return None
    tool_handler = logging.FileHandler("cadagent_tool.log")
    tool_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    _tool_logger.addHandler(tool_handler)
    _tool_logger.propagate = False

# Dedicated logger for LLM reasoning summaries (chain of thought)
_reasoning_logger = logging.getLogger("cadagent_reasoning")
if not _reasoning_logger.handlers:
    _reasoning_logger.setLevel(logging.INFO)
    reasoning_handler = logging.FileHandler("cadagent_reasoning.log")
    reasoning_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    _reasoning_logger.addHandler(reasoning_handler)
    _reasoning_logger.propagate = False

# Lightweight client caches keyed by API key to support per-session BYOK.
_anthropic_clients: Dict[str, AsyncAnthropic] = {}
_openai_clients: Dict[str, AsyncOpenAI] = {}
_gemini_clients: Dict[str, Any] = {}


def _resolve_api_key(
    api_keys: Optional[Dict[str, str]],
    preferred_keys: List[str],
    env_key: str,
) -> Optional[str]:
    if api_keys:
        for key in preferred_keys:
            value = api_keys.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    value = os.environ.get(env_key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _get_anthropic_client(api_keys: Optional[Dict[str, str]] = None) -> AsyncAnthropic:
    api_key = _resolve_api_key(api_keys, ["anthropic_api_key", "ANTHROPIC_API_KEY"], "ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No Anthropic API key available for this session")
    client = _anthropic_clients.get(api_key)
    if client is None:
        client = AsyncAnthropic(api_key=api_key)
        _anthropic_clients[api_key] = client
    return client


def _get_openai_client(api_keys: Optional[Dict[str, str]] = None) -> AsyncOpenAI:
    api_key = _resolve_api_key(api_keys, ["openai_api_key", "OPENAI_API_KEY"], "OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key available for this session")
    client = _openai_clients.get(api_key)
    if client is None:
        client = AsyncOpenAI(api_key=api_key)
        _openai_clients[api_key] = client
    return client


def _get_gemini_async_client(api_keys: Optional[Dict[str, str]] = None):
    if genai is None:
        return None
    api_key = _resolve_api_key(api_keys, ["google_api_key", "GOOGLE_API_KEY"], "GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("No Google API key available for this session")
    client = _gemini_clients.get(api_key)
    if client is None:
        client = genai.Client(api_key=api_key)
        _gemini_clients[api_key] = client
    return client.aio

# Model identifiers
MODEL_CLAUDE_SONNET_45 = "claude-sonnet-4-5-20250929"
MODEL_CLAUDE_HAIKU_45 = "claude-haiku-4-5-20250929"
MODEL_CLAUDE_OPUS_45 = "claude-opus-4-5-20251101"
MODEL_GPT_5 = "gpt-5"
MODEL_GPT_52 = "gpt-5.2"
MODEL_GPT_5_MINI = "gpt-5-mini"
MODEL_DEFAULT = MODEL_CLAUDE_SONNET_45
MODEL_GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"

# Map friendly names to API model identifiers
MODEL_MAP = {
    "claude-sonnet-4.5": MODEL_CLAUDE_SONNET_45,
    "claude-sonnet-45": MODEL_CLAUDE_SONNET_45,
    "claude": MODEL_CLAUDE_SONNET_45,
    "claude-haiku-4.5": MODEL_CLAUDE_HAIKU_45,
    "claude-haiku-45": MODEL_CLAUDE_HAIKU_45,
    "haiku": MODEL_CLAUDE_HAIKU_45,
    "claude-opus-4.5": MODEL_CLAUDE_OPUS_45,
    "claude-opus-45": MODEL_CLAUDE_OPUS_45,
    "opus": MODEL_CLAUDE_OPUS_45,
    "gpt-5": MODEL_GPT_5,
    "gpt5": MODEL_GPT_5,
    "gpt-5.2": MODEL_GPT_52,
    "gpt5.2": MODEL_GPT_52,
    "gpt-5-mini": MODEL_GPT_5_MINI,
    "gpt5-mini": MODEL_GPT_5_MINI,
    "gemini-3-pro-preview": MODEL_GEMINI_3_PRO_PREVIEW,
    "gemini-3-pro": MODEL_GEMINI_3_PRO_PREVIEW,
    "gemini-3": MODEL_GEMINI_3_PRO_PREVIEW,
    "gemini3": MODEL_GEMINI_3_PRO_PREVIEW,
    "gemini": MODEL_GEMINI_3_PRO_PREVIEW,
}

# OpenAI Responses API max_output_tokens based on reasoning effort
# Reasoning models (GPT-5, o1, o3, etc.) consume output tokens for BOTH reasoning AND tool calls.
# With max_output_tokens=4096 and complex tasks, the model may exhaust the budget on reasoning alone,
# leaving no tokens for the actual tool call - causing "incomplete" responses.
# Higher reasoning efforts need larger budgets to accommodate extended chain-of-thought.
# Doubled from previous values (2025-12-19) to reduce max_tokens errors during complex operations.
OPENAI_MAX_OUTPUT_TOKENS = {
    "xhigh": 131072,  # 128K - Maximum reasoning (Extra High)
    "high": 131072,   # 128K - Complex multi-step reasoning tasks
    "medium": 65536,  # 64K - Moderate reasoning depth
    "low": 32768,     # 32K - Basic reasoning, quick tool calls
}
OPENAI_MAX_OUTPUT_TOKENS_DEFAULT = 32768  # Fallback for unknown effort levels

# Claude Extended Thinking budget_tokens based on reasoning effort
# Extended thinking enables Claude to engage in deeper analysis before responding.
# Higher budgets allow for more comprehensive chain-of-thought reasoning.
# Minimum budget is 1,024 tokens per Anthropic's API requirements.
CLAUDE_THINKING_BUDGET_TOKENS = {
    "high": 10000,    # Complex multi-step reasoning tasks
    "medium": 5000,   # Moderate reasoning depth
    "low": 2000,      # Basic reasoning, quick analysis
}
CLAUDE_THINKING_BUDGET_DEFAULT = 5000  # Fallback for unknown effort levels
CLAUDE_THINKING_BUDGET_MIN = 1024  # Anthropic API minimum
# Keep headroom for a real answer after thinking (max_tokens must be > budget_tokens)
CLAUDE_THINKING_OUTPUT_HEADROOM = 512


def _get_openai_max_output_tokens(reasoning_effort: str) -> int:
    """
    Get the appropriate max_output_tokens for OpenAI Responses API based on reasoning effort.

    Reasoning models consume output tokens for both internal reasoning AND the actual response/tool call.
    Higher reasoning efforts need larger token budgets to avoid truncation during complex reasoning.

    Args:
        reasoning_effort: One of "high", "medium", "low"

    Returns:
        Appropriate max_output_tokens value
    """
    return OPENAI_MAX_OUTPUT_TOKENS.get(reasoning_effort, OPENAI_MAX_OUTPUT_TOKENS_DEFAULT)


def _get_claude_thinking_budget(reasoning_effort: str) -> int:
    """
    Get the appropriate budget_tokens for Claude Extended Thinking based on reasoning effort.

    Extended thinking enables Claude to engage in deeper analysis before responding.
    Higher budgets allow for more comprehensive chain-of-thought reasoning.

    Args:
        reasoning_effort: One of "high", "medium", "low"

    Returns:
        Appropriate budget_tokens value (minimum 1,024 per API requirements)
    """
    requested = CLAUDE_THINKING_BUDGET_TOKENS.get(reasoning_effort, CLAUDE_THINKING_BUDGET_DEFAULT)
    return max(requested, CLAUDE_THINKING_BUDGET_MIN)


def _get_base_claude_model(model: str) -> str:
    """
    Get the base Claude model identifier for API calls.

    Args:
        model: Model identifier string

    Returns:
        Base model identifier for API calls
    """
    return model


def markdown_to_html(text: str) -> str:
    """
    Convert markdown text to HTML for Fusion palette display.

    Args:
        text: Markdown formatted text

    Returns:
        HTML string suitable for palette rendering
    """
    if not text or not isinstance(text, str):
        return ""

    try:
        # Convert markdown to HTML with safe defaults
        html = markdown.markdown(
            text,
            extensions=['fenced_code', 'tables', 'nl2br'],
            output_format='html'
        )
        return html
    except Exception as e:
        logger.warning(f"Failed to convert markdown to HTML: {e}")
        # Fallback: return plain text
        return text


def html_to_plain_text(html: str) -> str:
    """
    Convert simple HTML fragments into readable plain text.

    Args:
        html: HTML string to normalize.

    Returns:
        Plain text representation with basic newlines and bullet markers.
    """
    if not html or not isinstance(html, str):
        return ""

    text = html
    text = re.sub(r"<\s*br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*(p|div|h[1-6]|tr)\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*li[^>]*>", "\n• ", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*li\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _get_openai_responses_client(openai_client: AsyncOpenAI):
    """
    Return the OpenAI Responses API client, supporting both GA and beta namespaces.
    """
    responses_client = getattr(openai_client, "responses", None)
    if responses_client is not None:
        return responses_client

    beta_client = getattr(openai_client, "beta", None)
    if beta_client is not None:
        responses_client = getattr(beta_client, "responses", None)
        if responses_client is not None:
            return responses_client

    raise AttributeError(
        "OpenAI Responses API is not available on the configured SDK. "
        "Please upgrade the openai Python package to a version that includes responses streaming."
    )

def normalize_model_name(model_name: Optional[str]) -> str:
    """
    Normalize model name to API identifier.
    Returns default model if name is None or unrecognized.
    """
    if not model_name:
        return MODEL_DEFAULT

    normalized = model_name.lower().strip()
    return MODEL_MAP.get(normalized, MODEL_DEFAULT)

# Tool schema guardrails to avoid provider 400s
_FORBIDDEN_TOP_LEVEL_KEYS = {"oneOf", "anyOf", "allOf", "enum", "not"}

# Strict mode: raise errors instead of silently stripping forbidden constructs
_STRICT_SCHEMA_MODE = os.environ.get("CADAGENT_STRICT_SCHEMA", "").lower() in ("1", "true", "yes")


def _sanitize_tool_schema(schema: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    """
    Remove top-level JSON Schema constructs that OpenAI/Anthropic reject for functions.

    Providers require the root schema to be a plain object without oneOf/anyOf/allOf/enum/not.
    We strip those keys and coerce type='object' so we fail fast on our side instead
    of triggering HTTP 400s during live requests.

    Set CADAGENT_STRICT_SCHEMA=1 to raise errors instead of silently stripping.
    """
    if not isinstance(schema, dict):
        logger.warning(
            "Tool %s has non-dict input_schema (%s); replacing with empty object schema",
            tool_name,
            type(schema).__name__,
        )
        return {"type": "object", "properties": {}, "additionalProperties": True}

    cleaned = copy.deepcopy(schema)

    removed = [key for key in _FORBIDDEN_TOP_LEVEL_KEYS if key in cleaned]
    for key in removed:
        cleaned.pop(key, None)

    if removed:
        if _STRICT_SCHEMA_MODE:
            raise ValueError(
                f"Tool '{tool_name}' uses forbidden top-level constructs {removed}. "
                "Fix the schema or disable CADAGENT_STRICT_SCHEMA."
            )
        logger.warning(
            "Tool %s: stripped top-level %s for API compatibility (function-call schemas forbid them)",
            tool_name,
            removed,
        )

    if cleaned.get("type") != "object":
        logger.warning(
            "Tool %s: forcing schema type to 'object' (was %r) to satisfy provider requirements",
            tool_name,
            cleaned.get("type"),
        )
        cleaned["type"] = "object"

    return cleaned


def _prepare_tools_for_api(tools: List[dict]) -> List[dict]:
    """
    Return a sanitized copy of the tool list ready for API submission.
    """
    prepared: List[dict] = []
    for tool in tools or []:
        schema = tool.get("input_schema") or tool.get("parameters")
        sanitized_schema = _sanitize_tool_schema(schema, tool.get("name", "<unknown>"))
        prepared.append({**tool, "input_schema": sanitized_schema})
    return prepared

# Tool schema definition for Fusion 360 CAD operations
TOOLS = [
    {
        "name": "create_construction_plane",
        "description": "Create a construction plane with specific orientation and location. All sketches reference planes by plane_id. Orientation ALWAYS belongs to the plane, never the sketch. Use this when you need angled planes, offset planes, or planes aligned with existing faces.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plane_id": {
                    "type": "string",
                    "description": "Unique identifier for this plane (e.g., 'base_plane', 'mount_plane_30deg', 'offset_plane_5cm'). This ID will be used by create_sketch to reference the plane."
                },
                "mode": {
                    "type": "string",
                    "enum": ["datum", "offset_from_datum", "angle_to_edge", "face_normal"],
                    "description": "How to define the plane. 'datum' for built-in XY/XZ/YZ (rarely needed, can use these directly in create_sketch), 'offset_from_datum' for parallel offset planes, 'angle_to_edge' for tilted mounting surfaces and angled brackets, 'face_normal' to sketch on existing faces."
                },
                "datum_axis_plane": {
                    "type": "string",
                    "enum": ["XY", "XZ", "YZ"],
                    "description": "For mode='datum': which base datum plane to use. XY is horizontal, XZ is vertical front-back, YZ is vertical left-right."
                },
                "base_datum_plane": {
                    "type": "string",
                    "enum": ["XY", "XZ", "YZ"],
                    "description": "For mode='offset_from_datum': base datum plane to offset from."
                },
                "offset_cm": {
                    "type": "number",
                    "description": "For mode='offset_from_datum': signed offset distance from the base datum plane in cm. Positive moves in the positive axis direction (e.g., +5 on XY moves up in +Z)."
                },
                "reference_face_token": {
                    "type": "string",
                    "description": "For mode='angle_to_edge': MUST be one of 'XY', 'XZ', or 'YZ' datum planes only. Chained angled planes (referencing previously created planes) are NOT supported due to Fusion 360 API limitations."
                },
                "reference_edge_token": {
                    "type": "string",
                    "description": "For mode='angle_to_edge': edge/axis token used as the rotation axis. Use 'X', 'Y', or 'Z' for world axes, OR use an edge ref (e0, e1, ...) from Design Entities context for existing solid edges. Local plane axes (e.g., 'plane_id:U') are NOT supported."
                },
                "angle_deg": {
                    "type": "number",
                    "description": "For mode='angle_to_edge': rotation angle in degrees between the new plane and the reference face. Use positive values for counterclockwise rotation around the edge."
                },
                "face_token": {
                    "type": "string",
                    "description": "For mode='face_normal': the plane will lie coincident with this face (same orientation). Use plane_id from a previously created construction plane (enables plane copying), OR use a face ref (face_0, face_1, ...) from Design Entities context. Use spatial properties (normal, centroid) in the entity context to pick the right face. Use this to sketch directly on existing geometry or duplicate custom planes."
                },
                "point_x": {
                    "type": "number",
                    "description": "Optional for mode='face_normal': world X coordinate of a point on the face to place the plane's origin (cm). If omitted, uses face centroid."
                },
                "point_y": {
                    "type": "number",
                    "description": "Optional for mode='face_normal': world Y coordinate of origin point (cm)."
                },
                "point_z": {
                    "type": "number",
                    "description": "Optional for mode='face_normal': world Z coordinate of origin point (cm)."
                },
                "description": {
                    "type": "string",
                    "description": "Brief explanation of what you're doing and why (e.g., 'Creating angled plane 30° off right face for mounting bracket', 'Offset XY plane up by 5cm for second level')."
                }
            },
            "required": ["plane_id", "mode", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "create_sketch",
        "description": "Create a new sketch on a construction plane or directly on an existing planar face. Must be called before adding any 2D geometry. Returns a sketch_id for reference.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plane_id": {
                    "type": "string",
                    "description": "ID of the plane/face to sketch on. Use 'XY', 'XZ', or 'YZ' for datum planes, a custom plane_id from create_construction_plane, OR a face ref (face_0, face_1, ...) from Design Entities to sketch directly on existing geometry. Use spatial properties (normal, centroid) to select the correct face. When using a face ref, a construction plane is automatically created."
                },
                "sketch_id": {
                    "type": "string",
                    "description": "Unique identifier for this sketch (e.g., 'sketch_1', 'base_circle')"
                },
                "sketch_name": {
                    "type": "string",
                    "description": "Descriptive name for this sketch in the timeline (e.g., 'Base Profile', 'Mounting Holes Sketch'). If not provided, Fusion will use default names like 'Sketch1'."
                },
                "description": {
                    "type": "string",
                    "description": "Brief explanation of what you're doing and why (e.g., 'Creating sketch for the base plate')"
                }
            },
            "required": ["plane_id", "sketch_id", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "add_circle",
        "description": "Add a circle to an existing sketch. The sketch must already exist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_id": {
                    "type": "string",
                    "description": "ID of the sketch to add the circle to"
                },
                "center_u": {
                    "type": "number",
                    "description": "U coordinate of circle center in the sketch's 2D coordinate system (cm). U is the first axis in the sketch plane - think of this as 'X on the paper'."
                },
                "center_v": {
                    "type": "number",
                    "description": "V coordinate of circle center in the sketch's 2D coordinate system (cm). V is the second axis in the sketch plane - think of this as 'Y on the paper'. The plane determines how this 2D drawing sits in 3D space."
                },
                "radius": {
                    "type": "number",
                    "description": "Circle radius in cm"
                },
                "description": {
                    "type": "string",
                    "description": "Brief explanation of what you're doing and why (e.g., 'Adding circular profile for the shaft')"
                }
            },
            "required": ["sketch_id", "center_u", "center_v", "radius", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "add_line",
        "description": "Add a line to an existing sketch. The sketch must already exist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_id": {
                    "type": "string",
                    "description": "ID of the sketch to add the line to"
                },
                "start_u": {"type": "number", "description": "Start point U in sketch's 2D system (cm). U is horizontal in the sketch plane."},
                "start_v": {"type": "number", "description": "Start point V in sketch's 2D system (cm). V is vertical in the sketch plane."},
                "end_u": {"type": "number", "description": "End point U in sketch's 2D system (cm)."},
                "end_v": {"type": "number", "description": "End point V in sketch's 2D system (cm)."},
                "description": {
                    "type": "string",
                    "description": "Brief explanation of what you're doing and why (e.g., 'Drawing edge of the mounting bracket')"
                }
            },
            "required": ["sketch_id", "start_u", "start_v", "end_u", "end_v", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "add_arc",
        "description": "Add an arc to an existing sketch using center-start-end points in sketch coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_id": {
                    "type": "string",
                    "description": "ID of the sketch to add the arc to"
                },
                "center_u": {
                    "type": "number",
                    "description": "U coordinate of arc center in the sketch's 2D coordinate system (cm)."
                },
                "center_v": {
                    "type": "number",
                    "description": "V coordinate of arc center in the sketch's 2D coordinate system (cm)."
                },
                "start_u": {
                    "type": "number",
                    "description": "U coordinate of arc start point in sketch coordinates (cm)."
                },
                "start_v": {
                    "type": "number",
                    "description": "V coordinate of arc start point in sketch coordinates (cm)."
                },
                "end_u": {
                    "type": "number",
                    "description": "U coordinate of arc end point in sketch coordinates (cm)."
                },
                "end_v": {
                    "type": "number",
                    "description": "V coordinate of arc end point in sketch coordinates (cm)."
                },
                "arc_id": {
                    "type": "string",
                    "description": "Optional stable alias for later references (e.g., 'arc_0')."
                },
                "description": {
                    "type": "string",
                    "description": "Brief explanation of what you're doing and why (e.g., 'Adding rounded slot end')."
                }
            },
            "required": ["sketch_id", "center_u", "center_v", "start_u", "start_v", "end_u", "end_v", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "add_rectangle",
        "description": "Add a rectangle to an existing sketch by specifying two opposite corners.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_id": {
                    "type": "string",
                    "description": "ID of the sketch to add the rectangle to."
                },
                "corner1_u": {"type": "number", "description": "First corner U coordinate in sketch's 2D system (cm)."},
                "corner1_v": {"type": "number", "description": "First corner V coordinate in sketch's 2D system (cm)."},
                "corner2_u": {"type": "number", "description": "Opposite corner U coordinate in sketch's 2D system (cm)."},
                "corner2_v": {"type": "number", "description": "Opposite corner V coordinate in sketch's 2D system (cm)."},
                "description": {
                    "type": "string",
                    "description": "Brief explanation of what you're doing and why (e.g., 'Creating base plate outline')"
                }
            },
            "required": ["sketch_id", "corner1_u", "corner1_v", "corner2_u", "corner2_v", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "list_sketch_profiles",
        "description": "List available closed profiles/regions in a sketch (Fusion 360 'profiles') so you can choose the correct profile_index/profile_indices for extrude/revolve.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_id": {
                    "type": "string",
                    "description": "ID of the sketch to inspect"
                },
                "description": {
                    "type": "string",
                    "description": "Brief explanation of what you're doing and why (e.g., 'Listing sketch profiles to choose the correct regions for extrusion')"
                }
            },
            "required": ["sketch_id"],
            "additionalProperties": False
        }
    },
    {
        "name": "extrude_profile",
        "description": "Extrude a closed profile from a sketch to create 3D geometry. The profile must be a closed loop. DIRECTION: Positive distance extrudes in the sketch plane's normal direction (typically +Z for XY plane). For Cut operations on a face, use NEGATIVE distance to cut INTO the body below the sketch.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_id": {
                    "type": "string",
                    "description": "ID of the sketch containing the profile"
                },
                "profile_index": {
                    "type": "integer",
                    "description": "Index of a SINGLE profile to extrude (0-based). Only use this when you want exactly ONE profile. If you need multiple profiles (overlapping shapes), use profile_indices instead and OMIT this field entirely."
                },
                "profile_indices": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "description": "PREFERRED for overlapping shapes: List of profile indices to extrude as one unified feature. Example: profile_indices=[0,1,2] extrudes all 3 profiles together. When using this, OMIT profile_index entirely from the JSON. Mutually exclusive with profile_index."
                },
                "distance": {
                    "type": "number",
                    "description": "Extrusion distance in cm. CRITICAL: Use NEGATIVE distance for Cut operations when the sketch is on top of an existing body (e.g., -1 to cut 1cm downward into material). Positive extrudes in the sketch normal direction."
                },
                "operation": {
                    "type": "string",
                    "enum": ["NewBody", "Join", "Cut"],
                    "description": "How to combine with existing geometry",
                    "default": "NewBody"
                },
                "feature_name": {
                    "type": "string",
                    "description": "Descriptive name for this feature in the timeline (e.g., 'Base Mounting Plate', 'Support Column'). If not provided, Fusion will use default names like 'Extrude1'."
                },
                "description": {
                    "type": "string",
                    "description": "Brief explanation of what you're doing and why (e.g., 'Extruding to create the main body')"
                }
            },
            "required": ["sketch_id", "distance", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "revolve_profile",
        "description": "Revolve a profile around an axis to create solids or surfaces. Supports construction/sketch/edge/face axes, angle or to-entity extents, solid or surface output, and all feature operations including Intersect.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_id": {
                    "type": "string",
                    "description": "ID of the sketch containing the profile to revolve. Profile must be closed when is_solid=true; open profiles are allowed when is_solid=false."
                },
                "profile_index": {
                    "type": "integer",
                    "description": "Index of the profile in the sketch (0-based). CRITICAL: Each closed loop in a sketch is a SEPARATE profile. If you have multiple closed loops, specify which one to revolve by index. Required for multi-profile sketches.",
                    "default": 0,
                    "minimum": 0
                },
                "axis": {
                    "type": "object",
                    "description": "Axis of revolution. Set 'type' to select axis source, then provide the corresponding fields: construction→axis, sketch_line→sketch_id+line_index, edge→edge_token, face→face_token.",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["construction", "sketch_line", "edge", "face"],
                            "description": "Axis source type: 'construction' for world X/Y/Z axes, 'sketch_line' for a line in a sketch, 'edge' for an existing edge, 'face' for a cylindrical face axis."
                        },
                        "axis": {
                            "type": "string",
                            "enum": ["x", "y", "z"],
                            "description": "For type='construction': which world axis to revolve around."
                        },
                        "sketch_id": {
                            "type": "string",
                            "description": "For type='sketch_line': ID of the sketch containing the axis line."
                        },
                        "line_index": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                            "description": "For type='sketch_line': index of the line in the sketch (0-based)."
                        },
                        "edge_token": {
                            "type": "string",
                            "description": "For type='edge': entity token of an existing linear edge to use as axis."
                        },
                        "face_token": {
                            "type": "string",
                            "description": "For type='face': entity token of a cylindrical face whose axis to use."
                        }
                    },
                    "required": ["type"],
                    "additionalProperties": False
                },
                "extent": {
                    "type": "object",
                    "description": "Extent configuration. Set 'mode' to select extent type, then provide corresponding fields: full→(none), angle→symmetric+angle_degrees, two_sides_angle→angle1_degrees+angle2_degrees, to→to_entity_token, two_sides_to→to_entity1_token+to_entity2_token.",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["full", "angle", "two_sides_angle", "to", "two_sides_to"],
                            "description": "Extent type: 'full' for 360°, 'angle' for single-direction or symmetric, 'two_sides_angle' for asymmetric angles, 'to' for revolve-to-entity, 'two_sides_to' for two target entities."
                        },
                        "symmetric": {
                            "type": "boolean",
                            "description": "For mode='angle': if true, angle_degrees is PER-SIDE (total = 2×angle)."
                        },
                        "angle_degrees": {
                            "type": "number",
                            "minimum": -360,
                            "maximum": 360,
                            "description": "For mode='angle': angle in degrees. Negative reverses direction. Must be non-zero."
                        },
                        "angle1_degrees": {
                            "type": "number",
                            "minimum": -360,
                            "maximum": 360,
                            "description": "For mode='two_sides_angle': angle on side 1 (degrees, non-zero)."
                        },
                        "angle2_degrees": {
                            "type": "number",
                            "minimum": -360,
                            "maximum": 360,
                            "description": "For mode='two_sides_angle': angle on side 2 (degrees, non-zero)."
                        },
                        "to_entity_token": {
                            "type": "string",
                            "description": "For mode='to': face or body token to revolve until (one direction)."
                        },
                        "to_entity1_token": {
                            "type": "string",
                            "description": "For mode='two_sides_to': first target entity token."
                        },
                        "to_entity2_token": {
                            "type": "string",
                            "description": "For mode='two_sides_to': second target entity token."
                        }
                    },
                    "required": ["mode"],
                    "additionalProperties": False
                },
                "operation": {
                    "type": "string",
                    "enum": ["NewBody", "Join", "Cut", "Intersect"],
                    "description": "How to combine with existing geometry. Join/Cut/Intersect require existing bodies with spatial overlap.",
                    "default": "NewBody"
                },
                "is_solid": {
                    "type": "boolean",
                    "description": "True for solid revolve (requires closed profile), false for surface revolve (allows open profiles).",
                    "default": True
                },
                "creation_occurrence_token": {
                    "type": "string",
                    "description": "Optional occurrence token when the profile and axis are in different components. Sets revolveInput.creationOccurrence."
                },
                "feature_name": {
                    "type": "string",
                    "description": "Descriptive name for the timeline feature."
                },
                "description": {
                    "type": "string",
                    "description": "Brief explanation of what you're doing and why."
                }
            },
            "required": ["sketch_id", "axis", "extent", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "jump_to_timeline_position",
        "description": "Jump back to a specific point in the timeline by repositioning the user's view (marker) and deleting all operations after that point. The marker position represents the state of the project that the user currently sees. Use this when you want to undo recent operations and continue from an earlier state, or when explicitly requested by the user to continue from a specific timeline position. This is like git reset --hard - operations after the target position will be permanently deleted.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_index": {
                    "type": "integer",
                    "description": "The timeline index to jump back to (0-based). The user's view (marker) will be positioned here and everything after will be deleted. Must be between 0 and the current timeline count."
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you're jumping back (for logging and user context)"
                },
                "description": {
                    "type": "string",
                    "description": "Brief explanation of what you're doing and why (e.g., 'Reverting to try a different approach')"
                }
            },
            "required": ["target_index", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "delete_feature",
        "description": "Delete a feature (extrude, fillet, revolve, etc.) from the timeline by its entity_token. Use this to remove problematic features (faulty fillets, incorrect extrusions, failed operations) without reverting the entire timeline. Get the entity_token from list_features output. NOTE: Sketches cannot be deleted with this tool - only timeline features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_token": {
                    "type": "string",
                    "description": "The entity_token of the feature to delete (from list_features output, NOT a short ref like f_0)"
                },
                "description": {
                    "type": "string",
                    "description": "Brief explanation of why this feature is being deleted"
                },
                "expected_name": {
                    "type": "string",
                    "description": "Optional: expected feature name for safety verification (use 'name' or 'timeline_name' from list_features output). If provided and neither field matches, deletion fails."
                },
                "expected_timeline_index": {
                    "type": "integer",
                    "description": "Optional: expected timeline index for safety verification. If provided and doesn't match, deletion fails."
                }
            },
            "required": ["feature_token", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "select_edges",
        "description": "Select edges in active design. Clears existing selection by default.",
        "input_schema": {
            "type": "object",
            "properties": {
                "edge_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Edge refs (e0, e1, e2, ...) from the spatial context."
                },
                "clear_existing": {
                    "type": "boolean",
                    "description": "Clear selection first.",
                    "default": True
                },
                "description": {
                    "type": "string",
                    "description": "What and why."
                }
            },
            "required": ["edge_refs", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "clear_edge_selection",
        "description": "Clear active edge selection.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What and why."
                }
            },
            "required": ["description"],
            "additionalProperties": False
        }
    },
    {
        "name": "select_faces",
        "description": "Select faces in active design. Clears existing selection by default.",
        "input_schema": {
            "type": "object",
            "properties": {
                "face_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Sequential face refs (face_0, face_1, ...) from Design Entities. Use spatial properties (normal, centroid, surface_type) to identify the correct face for selection."
                },
                "clear_existing": {
                    "type": "boolean",
                    "description": "Clear selection first.",
                    "default": True
                },
                "description": {
                    "type": "string",
                    "description": "What and why."
                }
            },
            "required": ["face_refs", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "clear_face_selection",
        "description": "Clear active face selection.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What and why."
                }
            },
            "required": ["description"],
            "additionalProperties": False
        }
    },
    {
        "name": "select_bodies",
        "description": "Select bodies in active design. Clears existing selection by default.",
        "input_schema": {
            "type": "object",
            "properties": {
                "body_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Body refs (body_0, body_1, ...) from the spatial context."
                },
                "clear_existing": {
                    "type": "boolean",
                    "description": "Clear selection first.",
                    "default": True
                },
                "description": {
                    "type": "string",
                    "description": "What and why."
                }
            },
            "required": ["body_refs", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "clear_body_selection",
        "description": "Clear active body selection.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What and why."
                }
            },
            "required": ["description"],
            "additionalProperties": False
        }
    },
    {
        "name": "list_features",
        "description": "List recently created timeline features with their entity tokens, names, types, and associated bodies. Uses the cached feature snapshot so you can quickly reference the latest operations without querying Fusion again.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Brief explanation of why you need the feature list (e.g., 'Checking the last fillet before patterning')."
                }
            },
            "required": ["description"],
            "additionalProperties": False
        }
    },
    {
        "name": "create_pattern_feature",
        "description": "Create a rectangular or circular pattern from existing timeline features. Supply only the pattern type, the features to duplicate (use ['auto_last'] to reference the most recent feature), minimal counts/spacing, and an optional orientation hint. NOTE: Circular patterns currently rotate about the component X/Y/Z construction axes THROUGH THE GLOBAL ORIGIN (axis location is not configurable). If you need a bolt-circle around an off-origin center, place instances explicitly (e.g., multiple create_simple_hole calls) instead of circular patterning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern_type": {
                    "type": "string",
                    "enum": ["rectangular", "circular"],
                    "description": "Type of pattern to create. Rectangular patterns duplicate features along one or two linear directions. Circular patterns rotate features about a global component axis (X/Y/Z) through the origin; only the axis direction is configurable, not the rotation center."
                },
                "feature_tokens": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Entity tokens of the features to pattern. Include ['auto_last'] to automatically use the most recently created feature when appropriate."
                },
                "count_x": {
                    "type": "integer",
                    "minimum": 2,
                    "description": "Instance count along the primary direction for rectangular patterns. Defaults to 2 when omitted."
                },
                "spacing_x_cm": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Spacing between instances along the primary direction in centimeters. If omitted, a heuristic based on feature size is used."
                },
                "count_y": {
                    "type": "integer",
                    "minimum": 2,
                    "description": "Instance count for the secondary direction of a rectangular pattern. Provide only when a second axis is needed."
                },
                "spacing_y_cm": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Spacing between instances along the secondary direction in centimeters."
                },
                "rotation_count": {
                    "type": "integer",
                    "minimum": 2,
                    "description": "Total number of instances for a circular pattern. Defaults to 3 when omitted."
                },
                "rotation_angle_deg": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Total sweep angle for a circular pattern in degrees. Defaults to 360° when omitted."
                },
                "orientation_hint": {
                    "description": "Optional axis hint (direction only): string ('x', 'y', 'z', '-x', '-y', '-z') OR 3-element array [x, y, z] direction vector in world coordinates. This chooses which global axis to rotate about; it does NOT set the axis location (which is through the global origin)."
                },
                "feature_name": {
                    "type": "string",
                    "description": "Name to assign to the resulting pattern feature in the timeline."
                },
                "description": {
                    "type": "string",
                    "description": "Short sentence stating what you're doing and why (e.g., 'Pattern the mounting holes along X')."
                }
            },
            "required": ["pattern_type", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "apply_fillet",
        "description": "Apply constant-radius fillet to edges. Uses edge refs from Design Entities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "edge_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Edge refs (e0, e1, e2, ...) from the spatial context."
                },
                "radius": {
                    "type": "number",
                    "description": "Fillet radius (default: mm)."
                },
                "radius_unit": {
                    "type": "string",
                    "enum": ["mm", "cm", "m", "in", "ft"],
                    "description": "Radius unit.",
                    "default": "mm"
                },
                "include_tangent_edges": {
                    "type": "boolean",
                    "description": "Auto-include tangent edges for chain filleting.",
                    "default": True
                },
                "feature_name": {
                    "type": "string",
                    "description": "Timeline name."
                },
                "description": {
                    "type": "string",
                    "description": "What and why."
                }
            },
            "required": ["edge_refs", "radius", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "apply_chamfer",
        "description": "Apply equal-distance chamfer to edges. Uses edge refs from Design Entities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "edge_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Edge refs (e0, e1, e2, ...) from the spatial context."
                },
                "distance": {
                    "type": "number",
                    "description": "Chamfer distance (default: mm)."
                },
                "distance_unit": {
                    "type": "string",
                    "enum": ["mm", "cm", "m", "in", "ft"],
                    "description": "Distance unit.",
                    "default": "mm"
                },
                "include_tangent_edges": {
                    "type": "boolean",
                    "description": "Auto-include tangent edges.",
                    "default": True
                },
                "feature_name": {
                    "type": "string",
                    "description": "Timeline name."
                },
                "description": {
                    "type": "string",
                    "description": "What and why."
                }
            },
            "required": ["edge_refs", "distance", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "create_simple_hole",
        "description": "Create drilled hole perpendicular to face. Uses face ref from Design Entities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "face_ref": {
                    "type": "string",
                    "description": "Face ref from Design Entities (face_0, face_1, ...). Use spatial properties (normal, centroid) to select the correct face."
                },
                "center_x": {
                    "type": "number",
                    "description": "X coordinate (mm)."
                },
                "center_y": {
                    "type": "number",
                    "description": "Y coordinate (mm)."
                },
                "center_z": {
                    "type": "number",
                    "description": "Z coordinate (mm)."
                },
                "diameter": {
                    "type": "number",
                    "description": "Hole diameter (default: mm)."
                },
                "diameter_unit": {
                    "type": "string",
                    "enum": ["mm", "cm", "m", "in"],
                    "description": "Diameter unit.",
                    "default": "mm"
                },
                "extent_type": {
                    "type": "string",
                    "enum": ["through_all", "distance"],
                    "description": "'through_all' or 'distance' (requires depth)."
                },
                "depth": {
                    "type": "number",
                    "description": "Hole depth (mm). Required when extent_type='distance'."
                },
                "feature_name": {
                    "type": "string",
                    "description": "Timeline name."
                },
                "description": {
                    "type": "string",
                    "description": "What and why."
                }
            },
            "required": ["face_ref", "center_x", "center_y", "center_z", "diameter", "extent_type", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "create_counterbore_hole",
        "description": "Create counterbore hole for recessing bolt heads. Two diameters: counterbore for head, smaller hole for shank.",
        "input_schema": {
            "type": "object",
            "properties": {
                "face_ref": {
                    "type": "string",
                    "description": "Face ref from Design Entities (face_0, face_1, ...). Use spatial properties (normal, centroid) to select the correct face."
                },
                "center_x": {
                    "type": "number",
                    "description": "X coordinate (mm)."
                },
                "center_y": {
                    "type": "number",
                    "description": "Y coordinate (mm)."
                },
                "center_z": {
                    "type": "number",
                    "description": "Z coordinate (mm)."
                },
                "hole_diameter": {
                    "type": "number",
                    "description": "Main hole diameter for shank (default: mm)."
                },
                "hole_depth": {
                    "type": "number",
                    "description": "Main hole depth (mm)."
                },
                "counterbore_diameter": {
                    "type": "number",
                    "description": "Counterbore diameter for head (default: mm)."
                },
                "counterbore_depth": {
                    "type": "number",
                    "description": "Counterbore depth (mm)."
                },
                "diameter_unit": {
                    "type": "string",
                    "enum": ["mm", "cm", "m", "in"],
                    "description": "Diameter unit.",
                    "default": "mm"
                },
                "feature_name": {
                    "type": "string",
                    "description": "Timeline name."
                },
                "description": {
                    "type": "string",
                    "description": "What and why."
                }
            },
            "required": [
                "face_ref", "center_x", "center_y", "center_z",
                "hole_diameter", "hole_depth", "counterbore_diameter", "counterbore_depth", "description"
            ],
            "additionalProperties": False
        }
    },
    {
        "name": "create_tapped_hole",
        "description": "Create threaded hole for screw insertion. See THREAD SIZE REFERENCE for available sizes. Position hole center ≥1.5× diameter from any edge.",
        "input_schema": {
            "type": "object",
            "properties": {
                "face_ref": {
                    "type": "string",
                    "description": "Face ref from Design Entities (face_0, face_1, ...). Use spatial properties (normal, centroid) to select the correct face."
                },
                "center_x": {
                    "type": "number",
                    "description": "X coordinate (mm)."
                },
                "center_y": {
                    "type": "number",
                    "description": "Y coordinate (mm)."
                },
                "center_z": {
                    "type": "number",
                    "description": "Z coordinate (mm)."
                },
                "thread_type": {
                    "type": "string",
                    "enum": ["metric", "unc", "unf"],
                    "description": "Thread standard."
                },
                "thread_size": {
                    "type": "string",
                    "description": "Size from THREAD SIZE REFERENCE (e.g., 'M6', '1/4-20')."
                },
                "thread_depth": {
                    "type": "number",
                    "description": "Thread depth (mm). Min: 1.5× diameter, recommended: 2.5×."
                },
                "pilot_hole_depth": {
                    "type": "number",
                    "description": "Optional tap-drill depth (mm). Default: thread_depth + 2× pitch."
                },
                "diameter_unit": {
                    "type": "string",
                    "enum": ["mm", "cm", "m", "in"],
                    "description": "Unit for diameter reporting.",
                    "default": "mm"
                },
                "feature_name": {
                    "type": "string",
                    "description": "Timeline name (e.g., 'M6 Tapped Hole')."
                },
                "description": {
                    "type": "string",
                    "description": "What and why."
                }
            },
            "required": [
                "face_ref", "center_x", "center_y", "center_z",
                "thread_type", "thread_size", "thread_depth", "description"
            ],
            "additionalProperties": False
        }
    },
    {
        "name": "create_external_thread",
        "description": "Add external threads to cylindrical face. See THREAD SIZE REFERENCE. If unavailable size requested, choose nearest.",
        "input_schema": {
            "type": "object",
            "properties": {
                "face_ref": {
                    "type": "string",
                    "description": "Cylindrical face ref from Design Entities (face_0, face_1, ...). Filter by surface_type='cylindrical' in entity context to find the right face."
                },
                "thread_type": {
                    "type": "string",
                    "enum": ["metric", "unc", "unf"],
                    "description": "Thread standard."
                },
                "thread_size": {
                    "type": "string",
                    "description": "Size from THREAD SIZE REFERENCE."
                },
                "thread_length": {
                    "type": "number",
                    "description": "Length (cm). Required when is_full_length=false."
                },
                "thread_offset": {
                    "type": "number",
                    "description": "Offset from edge (cm). Default: 0.",
                    "default": 0.0
                },
                "is_full_length": {
                    "type": "boolean",
                    "description": "True for full-length threading.",
                    "default": True
                },
                "diameter_unit": {
                    "type": "string",
                    "enum": ["mm", "cm", "m", "in"],
                    "description": "Unit for diameter reporting.",
                    "default": "mm"
                },
                "feature_name": {
                    "type": "string",
                    "description": "Timeline name."
                },
                "description": {
                    "type": "string",
                    "description": "What and why."
                }
            },
            "required": [
                "face_ref", "thread_type", "thread_size", "description"
            ],
            "additionalProperties": False
        }
    },
    {
        "name": "create_shell",
        "description": "Hollow out a solid body by adding wall thickness. Two modes: 'open' removes specified faces (like opening a box lid), 'closed' hollows the entire body without removing faces (like making a hollow sphere). Set mode first, then provide face_refs OR body_refs accordingly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["open", "closed"],
                    "description": "Shell mode: 'open' removes faces to create openings (requires face_refs), 'closed' hollows body uniformly without openings (requires body_refs). Default: 'open'."
                },
                "face_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For mode='open': Face refs to remove (face_0, face_1, ...). Use spatial properties (normal, centroid) to identify which faces to open. These faces become openings in the shell. Omit this field entirely when using mode='closed'."
                },
                "body_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For mode='closed': Body refs to hollow (body_0, body_1, ...). Creates uniform wall thickness without any openings. Omit this field entirely when using mode='open'."
                },
                "inside_thickness": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Inward thickness (default: mm). Set to 0 to disable inward growth. At least one of inside_thickness or outside_thickness must be >0."
                },
                "outside_thickness": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Outward thickness (default: mm). Set to 0 to disable outward growth. At least one of inside_thickness or outside_thickness must be >0."
                },
                "thickness_unit": {
                    "type": "string",
                    "enum": ["mm", "cm", "m", "in"],
                    "default": "mm",
                    "description": "Thickness unit."
                },
                "is_tangent_chain": {
                    "type": "boolean",
                    "default": True,
                    "description": "Auto-include tangent faces (mode='open' only)."
                },
                "shell_type": {
                    "type": "string",
                    "enum": ["sharp", "rounded"],
                    "default": "sharp",
                    "description": "Corner treatment."
                },
                "feature_name": {
                    "type": "string",
                    "description": "Timeline name."
                },
                "description": {
                    "type": "string",
                    "description": "What and why."
                }
            },
            "required": ["description"],
            "additionalProperties": False
        }
    },
    {
        "name": "create_loft",
        "description": "Create a smooth 3D transition between 2 or more sketch profiles. The loft will blend between the profiles in order, creating a smooth organic shape. Use this for transitioning between different cross-sections (e.g., square to circle) or creating bottle/handle shapes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "profile_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "Ordered list of sketch IDs to loft between (minimum 2 profiles). Profiles will be connected in the order provided."
                },
                "operation": {
                    "type": "string",
                    "enum": ["NewBody", "Join", "Cut"],
                    "description": "How to combine with existing geometry",
                    "default": "NewBody"
                },
                "feature_name": {
                    "type": "string",
                    "description": "Descriptive name for this loft feature in the timeline (e.g., 'Bottle Transition', 'Handle Shape'). If not provided, Fusion will use default names like 'Loft1'."
                },
                "description": {
                    "type": "string",
                    "description": "Brief explanation of what you're doing and why (e.g., 'Creating smooth transition from base to neck')"
                }
            },
            "required": ["profile_ids", "description"],
            "additionalProperties": False
        }
    },
    {
        "name": "respond_to_user",
        "description": "Send a message to the user to ask for clarification or provide a status update. Do not send identical text more than once; wait for the user's response before repeating.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to display to the user"
                }
            },
            "required": ["message"],
            "additionalProperties": False
        }
    },
    {
        "name": "generate_question_tree",
        "description": "Generate a tree of clarifying questions to understand the user's requirements. Use this when the request is ambiguous or missing critical constraints. Do not use for simple, well-specified requests.",
        "input_schema": {
            "type": "object",
            "properties": {
                "problem_summary": {
                    "type": "string",
                    "description": "One-sentence summary of what the user is trying to solve"
                },
                "known_constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Constraints already stated by the user (do not ask about these)"
                },
                "questions": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Question"},
                    "description": "Root-level questions (top of the tree)"
                }
            },
            "$defs": {
                "Question": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique identifier for this question (snake_case)"
                        },
                        "question": {
                            "type": "string",
                            "description": "The question text"
                        },
                        "hint": {
                            "type": "string",
                            "description": "Tooltip explaining why this question matters and how it affects the design"
                        },
                        "options": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Option"},
                            "minItems": 2,
                            "description": "Available answer options. Final option should allow freeform input."
                        }
                    },
                    "required": ["id", "question", "options"]
                },
                "Option": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "string",
                            "description": "Machine-readable value for this option"
                        },
                        "label": {
                            "type": "string",
                            "description": "Human-readable label displayed to user"
                        },
                        "allows_text_input": {
                            "type": "boolean",
                            "description": "If true, user can provide freeform text (use for 'Other' options)"
                        },
                        "follow_up_questions": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Question"},
                            "description": "Questions that appear only if this option is selected"
                        }
                    },
                    "required": ["value", "label"]
                }
            },
            "required": ["problem_summary", "known_constraints", "questions"],
            "additionalProperties": False
        }
    },
    {
        "name": "propose_designs",
        "description": "Propose one or more design solutions for the user to choose from. This tool sends design cards directly to the Fusion palette (user-facing). Call it without additional narration and wait for the user's selection. Use after requirements are clear and include tradeoffs for each option.",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_summary": {
                    "type": "string",
                    "description": "Brief summary of the problem and gathered requirements"
                },
                "designs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique identifier for this design (snake_case)"
                            },
                            "name": {
                                "type": "string",
                                "description": "Short descriptive name (e.g., 'Domed Strainer')"
                            },
                            "description": {
                                "type": "string",
                                "description": "2-3 sentence description of the design approach"
                            },
                            "key_features": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Bullet points of main features"
                            },
                            "tradeoffs": {
                                "type": "object",
                                "properties": {
                                    "pros": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "cons": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            },
                            "best_for": {
                                "type": "string",
                                "description": "One-line summary of ideal use case"
                            },
                            "specifications": {
                                "type": "object",
                                "description": "Key dimensions and parameters for this design",
                                "additionalProperties": True
                            }
                        },
                        "required": ["id", "name", "description", "key_features", "tradeoffs"]
                    },
                    "minItems": 1,
                    "maxItems": 4,
                    "description": "Proposed design options (1-4)"
                },
                "recommendation": {
                    "type": "string",
                    "description": "ID of the recommended design, if one is clearly superior"
                }
            },
            "required": ["context_summary", "designs"],
            "additionalProperties": False
        }
    },
    {
        "name": "output_build_plan",
        "description": "Output a step-by-step build plan before constructing a design. Use this after a design is selected to show the user what steps will be taken, then execute the plan step by step. Each step should map roughly to one CAD operation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "design_name": {
                    "type": "string",
                    "description": "Name of the design being built"
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_number": {
                                "type": "integer",
                                "description": "Sequential step number (1, 2, 3, ...)"
                            },
                            "operation": {
                                "type": "string",
                                "description": "The CAD operation type (e.g., 'create_sketch', 'extrude', 'fillet')"
                            },
                            "description": {
                                "type": "string",
                                "description": "Human-readable description of what this step does"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Key parameters for this operation (dimensions, positions, etc.)",
                                "additionalProperties": True
                            }
                        },
                        "required": ["step_number", "operation", "description"]
                    },
                    "minItems": 1,
                    "description": "Ordered list of build steps"
                }
            },
            "required": ["design_name", "steps"],
            "additionalProperties": False
        }
    }
]


def _should_enable_caching(model: str) -> bool:
    """
    Determine if prompt caching should be enabled for the given model.

    Only Anthropic models support prompt caching. OpenAI models use a different
    caching mechanism that is handled automatically by their API.

    Args:
        model: Model identifier

    Returns:
        True if caching should be enabled, False otherwise
    """
    # Only enable caching for Anthropic/Claude models
    return not (model.startswith("gpt-") or model == MODEL_GPT_5)


def _build_cacheable_system(
    base_prompt: str,
    enable_caching: bool = True,
    ttl: str = "5m",
    split_at_version: bool = True
) -> Any:
    """
    Convert system prompt string to Anthropic list format with optional caching.

    Anthropic's prompt caching requires the system parameter to be a list of
    content blocks, with cache_control markers on blocks that should be cached.

    For prompts with 1,024+ tokens (Sonnet/Opus) or 4,096+ tokens (Haiku),
    caching can reduce costs by 90% and latency by 85% on cache hits.

    Cache Strategy (when split_at_version=True):
    - If the prompt contains "PROMPT VERSION:", split into two blocks:
      1. Static core (before version) - uses 1h TTL for maximum reuse
      2. Dynamic section (version onwards) - uses specified TTL
    - This ensures minor version bumps don't invalidate the entire cache

    Args:
        base_prompt: System prompt text
        enable_caching: Whether to add cache_control markers
        ttl: Cache TTL - "5m" (1.25x write cost) or "1h" (2x write cost)
        split_at_version: If True, split at "PROMPT VERSION:" for optimal caching

    Returns:
        For Anthropic with caching: List[Dict] with cache_control markers
        For Anthropic without caching: List[Dict] without markers
        For other models or fallback: Original string
    """
    if not enable_caching:
        # Return list format without caching (Anthropic accepts both formats)
        return [{"type": "text", "text": base_prompt}]

    # Try to split at version marker for optimal caching
    version_marker = "PROMPT VERSION:"
    if split_at_version and version_marker in base_prompt:
        split_index = base_prompt.find(version_marker)
        stable_part = base_prompt[:split_index].rstrip()
        dynamic_part = base_prompt[split_index:]

        # Only split if stable part is substantial (>500 chars)
        if len(stable_part) > 500:
            return [
                # Static core with 1-hour TTL (rarely changes)
                {
                    "type": "text",
                    "text": stable_part,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                },
                # Dynamic section with specified TTL
                {
                    "type": "text",
                    "text": dynamic_part,
                    "cache_control": {"type": "ephemeral"} if ttl == "5m" else {"type": "ephemeral", "ttl": ttl}
                }
            ]

    # Fallback: single block with specified TTL
    system_block = {
        "type": "text",
        "text": base_prompt,
        "cache_control": {"type": "ephemeral"}
    }

    # Add TTL if 1-hour caching is requested
    if ttl == "1h":
        system_block["cache_control"]["ttl"] = "1h"

    return [system_block]


def _add_tool_cache_markers(tools: List[dict], enable_caching: bool = True, ttl: str = "5m") -> List[dict]:
    """
    Add cache_control markers to tool definitions for Anthropic prompt caching.

    According to Anthropic's caching best practices, we should place cache
    breakpoints at the end of static content. Tools are static and come after
    the system prompt, so we mark the last tool for caching.

    Args:
        tools: List of tool definitions in Anthropic format
        enable_caching: Whether to add cache_control markers
        ttl: Cache TTL - "5m" (1.25x write cost) or "1h" (2x write cost)

    Returns:
        Modified copy of tools with cache markers (if enabled)
    """
    if not enable_caching or not tools:
        return tools

    # Create a deep copy to avoid mutating the original
    cached_tools = copy.deepcopy(tools)

    # Add cache marker to the last tool (end of static content)
    if cached_tools:
        cache_control = {"type": "ephemeral"}
        if ttl == "1h":
            cache_control["ttl"] = "1h"
        cached_tools[-1]["cache_control"] = cache_control

    return cached_tools


def _add_conversation_cache_marker(
    messages: List[Dict[str, Any]],
    enable_caching: bool = True,
    ttl: str = "5m"
) -> List[Dict[str, Any]]:
    """
    Add cache_control marker to the last user message for incremental conversation caching.

    This enables Anthropic's incremental caching strategy where each turn's conversation
    history gets cached, reducing costs by 90% on cache hits. The cache marker is placed
    on the last content block of the most recent user message.

    Args:
        messages: List of conversation messages in Anthropic format
        enable_caching: Whether to add cache_control markers (only for Anthropic models)
        ttl: Cache TTL - "5m" (1.25x write cost) or "1h" (2x write cost)

    Returns:
        Modified copy of messages with cache marker on last user message

    Note:
        - Only modifies Anthropic API calls (OpenAI handles caching automatically)
        - Creates a cache breakpoint that includes all conversation history up to this point
        - Requires minimum 1,024 tokens for Claude Sonnet 4.5 to cache
    """
    if not enable_caching or not messages:
        return messages

    # Deep copy to avoid mutating the original
    messages_copy = copy.deepcopy(messages)

    # Create cache_control object with appropriate TTL
    cache_control = {"type": "ephemeral"}
    if ttl == "1h":
        cache_control["ttl"] = "1h"

    # Find the last user message (work backwards)
    for i in range(len(messages_copy) - 1, -1, -1):
        msg = messages_copy[i]
        if msg.get("role") != "user":
            continue

        content = msg.get("content")

        # Convert string content to list format
        if isinstance(content, str):
            msg["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": cache_control
            }]
            break

        # Add cache_control to last block in list content
        elif isinstance(content, list) and content:
            last_block = content[-1]

            # Only add to text/image blocks (not tool_result blocks)
            if isinstance(last_block, dict):
                block_type = last_block.get("type")
                if block_type in ("text", "image"):
                    last_block["cache_control"] = cache_control
                    break
                # If last block is tool_result, find the last text/image block
                else:
                    for j in range(len(content) - 1, -1, -1):
                        block = content[j]
                        if isinstance(block, dict) and block.get("type") in ("text", "image"):
                            block["cache_control"] = cache_control
                            break
                    break

        # Found a user message, exit loop
        break

    return messages_copy



def _safe_model_dump(value: Any) -> Any:
    """Safely convert SDK objects to dicts when possible."""
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            return value
    return value


def _safe_text_value(value: Any) -> str:
    """Normalize text payloads to plain strings for provider compatibility."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "value", "content"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    if value is None:
        return ""
    return str(value)


def _sanitize_content_block_for_anthropic(block: Any, role: str) -> Optional[Dict[str, Any]]:
    """Return a provider-safe content block or None to drop unsupported blocks."""
    block = _safe_model_dump(block)

    if isinstance(block, str):
        text_value = block.strip()
        return {"type": "text", "text": text_value} if text_value else None

    if not isinstance(block, dict):
        return None

    block_type = block.get("type")

    if block_type == "text":
        clean: Dict[str, Any] = {
            "type": "text",
            "text": _safe_text_value(block.get("text", "")),
        }
        cache_control = block.get("cache_control")
        if isinstance(cache_control, dict):
            clean["cache_control"] = cache_control
        return clean

    if block_type == "image" and role == "user":
        source = block.get("source")
        if not isinstance(source, dict):
            return None
        image_data = source.get("data")
        if not isinstance(image_data, str) or not image_data:
            return None
        clean = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": source.get("media_type", "image/png"),
                "data": image_data,
            },
        }
        cache_control = block.get("cache_control")
        if isinstance(cache_control, dict):
            clean["cache_control"] = cache_control
        return clean

    if block_type == "tool_use" and role == "assistant":
        tool_name = block.get("name")
        tool_id = block.get("id")
        if not tool_name or not tool_id:
            return None
        tool_input = block.get("input")
        if not isinstance(tool_input, dict):
            tool_input = {}
        return {
            "type": "tool_use",
            "id": str(tool_id),
            "name": str(tool_name),
            "input": tool_input,
        }

    if block_type == "tool_result" and role == "user":
        tool_use_id = block.get("tool_use_id")
        if not tool_use_id:
            return None

        raw_content = block.get("content")
        content_blocks: List[Dict[str, Any]] = []

        if isinstance(raw_content, str):
            text_value = raw_content.strip()
            if text_value:
                content_blocks.append({"type": "text", "text": text_value})
        elif isinstance(raw_content, list):
            for item in raw_content:
                cleaned_item = _sanitize_content_block_for_anthropic(item, role="user")
                if cleaned_item and cleaned_item.get("type") in ("text", "image"):
                    content_blocks.append(cleaned_item)
        elif raw_content is not None:
            content_blocks.append({"type": "text", "text": _safe_text_value(raw_content)})

        if not content_blocks:
            content_blocks = [{"type": "text", "text": ""}]

        clean: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": str(tool_use_id),
            "content": content_blocks,
        }
        if block.get("is_error"):
            clean["is_error"] = True
        return clean

    # Drop unsupported/ephemeral block types (e.g., thinking).
    return None


def _sanitize_anthropic_messages_for_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize conversation history into Anthropic-compatible message blocks."""
    sanitized_messages: List[Dict[str, Any]] = []

    for message in messages or []:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        if role not in ("user", "assistant"):
            continue

        content = message.get("content")
        sanitized_content: List[Dict[str, Any]] = []

        if isinstance(content, str):
            text_value = content.strip()
            if text_value:
                sanitized_content.append({"type": "text", "text": text_value})
        elif isinstance(content, list):
            for block in content:
                clean_block = _sanitize_content_block_for_anthropic(block, role=role)
                if clean_block:
                    sanitized_content.append(clean_block)
        elif isinstance(content, dict):
            clean_block = _sanitize_content_block_for_anthropic(content, role=role)
            if clean_block:
                sanitized_content.append(clean_block)

        if sanitized_content:
            sanitized_messages.append({"role": role, "content": sanitized_content})

    return sanitized_messages


def _sanitize_anthropic_assistant_content(raw_blocks: List[Any]) -> List[Dict[str, Any]]:
    """Keep only assistant blocks that can be safely replayed in later turns."""
    sanitized: List[Dict[str, Any]] = []
    for block in raw_blocks or []:
        clean_block = _sanitize_content_block_for_anthropic(block, role="assistant")
        if clean_block:
            sanitized.append(clean_block)
    return sanitized


def _redact_messages_for_logging(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a copy of messages with large binary payloads redacted for logging."""
    redacted: List[Dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            redacted.append(message)
            continue

        message_copy: Dict[str, Any] = dict(message)
        content = message_copy.get("content")

        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict):
                    block_copy = dict(block)
                    if block_copy.get("type") == "image":
                        source = dict(block_copy.get("source", {}))
                        if "data" in source:
                            source["data"] = "<base64 omitted>"
                        block_copy["source"] = source
                    new_content.append(block_copy)
                else:
                    new_content.append(block)
            message_copy["content"] = new_content
        elif isinstance(content, dict):
            message_copy["content"] = {
                key: ("<base64 omitted>" if key == "data" else value)
                for key, value in content.items()
            }

        redacted.append(message_copy)

    return redacted


def _emit_llm_request_payload(
    *,
    provider: str,
    model: str,
    system_prompt: Optional[str],
    messages: Any,
    max_tokens: Optional[int],
    reasoning_effort: Optional[str],
    tools: Optional[Any] = None,
) -> None:
    """Log high-level request metadata without dumping full prompt content."""
    try:
        message_count = len(messages) if isinstance(messages, list) else "n/a"
        tool_count = len(tools) if isinstance(tools, list) else "n/a"
        roles = []
        if isinstance(messages, list):
            for msg in messages:
                role = msg.get("role") if isinstance(msg, dict) else None
                if role:
                    roles.append(role)

        summary = {
            "provider": provider,
            "model": model,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "message_count": message_count,
            "roles": roles,
            "tool_count": tool_count,
        }
        # print(f"[LLM REQUEST META] {json.dumps(summary, ensure_ascii=False)}")
    except Exception as exc:
        logger.warning("Failed to print LLM request metadata: %s", exc, exc_info=True)


def _log_stream_payload(source: str, payload: Any, model: Optional[str] = None) -> None:
    """
    Log raw LLM responses for debugging (written to cadagent_tool.log).

    The content is the exact provider payload (no COT stripping), pretty-printed
    to make inspection easy. Each call is wrapped with start/end separators to
    keep iterations distinct.

    Gated behind DEBUG level to avoid log/perf overhead in production.
    """
    if not _tool_logger.isEnabledFor(logging.DEBUG):
        return
    def convert_bytes_to_base64(obj):
        """Recursively convert bytes to base64 strings for JSON serialization."""
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        elif isinstance(obj, dict):
            return {k: convert_bytes_to_base64(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_bytes_to_base64(item) for item in obj]
        return obj

    try:
        if hasattr(payload, "model_dump"):
            data = payload.model_dump()
        elif isinstance(payload, dict):
            data = payload
        else:
            data = json.loads(json.dumps(payload, default=str))

        # Convert bytes to base64 for JSON serialization (e.g., thought_signature)
        data = convert_bytes_to_base64(data)

        # Keep logs focused: only persist minimal fields that show the LLM decision.
        compact: Dict[str, Any] = {}
        if isinstance(data, dict):
            for key in ("id", "model", "usage", "output"):
                if key in data:
                    compact[key] = data[key]
        data_to_log = compact or data
    except TypeError:
        data_to_log = str(payload)
    except Exception as exc:
        logger.warning("Failed to serialise stream payload for logging: %s", exc, exc_info=True)
        data_to_log = repr(payload)

    try:
        header = f"===== LLM RAW RESPONSE START [{source}] model={model or 'unknown'} ====="
        footer = f"===== LLM RAW RESPONSE END   [{source}] ====="
        pretty = json.dumps(data_to_log, ensure_ascii=False, indent=2)
        _tool_logger.debug(header)
        _tool_logger.debug(pretty)
        _tool_logger.debug(footer)
    except Exception as exc:
        logger.warning("Failed to write raw LLM response to cadagent_tool.log: %s", exc, exc_info=True)

    # Extract and log reasoning summaries separately for OpenAI responses
    if source == "openai_responses":
        _extract_and_log_reasoning(payload, model)


def _extract_and_log_reasoning(payload: Any, model: Optional[str] = None) -> None:
    """
    Extract and log reasoning summaries from OpenAI Responses API payloads.

    This function extracts the Chain of Thought (COT) reasoning summaries from
    OpenAI's reasoning models (GPT-5, o3, o1, etc.) and logs them to a dedicated
    file for analysis of the model's internal reasoning process.

    According to OpenAI's Responses API documentation:
    - Reasoning summaries are found in output items with type "reasoning"
    - The summary field contains an array of summary objects
    - Each summary object has a "text" field with the reasoning content

    Note: OpenAI's API inconsistently returns summaries (known issue as of 2025).
    Even when reasoning_tokens > 0, summary arrays are often empty.

    Args:
        payload: The response payload from OpenAI Responses API
        model: The model name for context in logs
    """
    try:
        # Convert payload to dict if needed
        if hasattr(payload, "model_dump"):
            data = payload.model_dump()
        elif isinstance(payload, dict):
            data = payload
        else:
            return  # Can't extract from unknown format

        # Check if this is a streaming event or full response
        event_type = data.get("type", None)

        # Handle streaming reasoning summary deltas
        if event_type == "response.reasoning_summary_text.delta":
            delta_text = data.get("delta", "")
            if delta_text and delta_text.strip():
                _reasoning_logger.info(f"[STREAMING DELTA] {delta_text}")
                return

        # Handle full responses - extract from output array
        output_items = data.get("output", [])
        if not output_items:
            return

        reasoning_summaries = []
        response_id = data.get("id", "unknown")
        reasoning_tokens = 0

        # Get reasoning token count from usage
        usage = data.get("usage", {})
        if usage:
            output_tokens_details = usage.get("output_tokens_details", {})
            reasoning_tokens = output_tokens_details.get("reasoning_tokens", 0)

        # Iterate through output items to find reasoning items
        for item in output_items:
            if isinstance(item, dict) and item.get("type") == "reasoning":
                summary_array = item.get("summary", [])

                # Extract text from summary objects
                for summary_obj in summary_array:
                    if isinstance(summary_obj, dict):
                        text = summary_obj.get("text", "")
                        if text and text.strip():
                            reasoning_summaries.append(text)

        # Log extracted reasoning summaries
        if reasoning_summaries:
            header = f"===== REASONING SUMMARY START [response_id={response_id}] model={model or 'unknown'} reasoning_tokens={reasoning_tokens} ====="
            footer = f"===== REASONING SUMMARY END   [response_id={response_id}] ====="

            _reasoning_logger.info(header)
            for idx, summary_text in enumerate(reasoning_summaries, 1):
                if len(reasoning_summaries) > 1:
                    _reasoning_logger.info(f"--- Summary Part {idx}/{len(reasoning_summaries)} ---")
                _reasoning_logger.info(summary_text)
            _reasoning_logger.info(footer)

            logger.debug(f"Extracted {len(reasoning_summaries)} reasoning summary part(s) from response {response_id}")

        # Log when reasoning tokens exist but no summary was provided (OpenAI API limitation)
        elif reasoning_tokens > 0:
            _reasoning_logger.info(f"[NO SUMMARY PROVIDED] response_id={response_id} model={model or 'unknown'} reasoning_tokens={reasoning_tokens} - OpenAI API did not return summary text despite reasoning occurring")
            logger.debug(f"Reasoning tokens generated ({reasoning_tokens}) but no summary provided for response {response_id}")

    except Exception as exc:
        logger.warning("Failed to extract reasoning summary from OpenAI response: %s", exc, exc_info=True)


# System prompt for CAD Agent (fallback) built from shared prompt builder to keep
# routed and fallback paths aligned. Tools list remains defined below.
SYSTEM_PROMPT, _SYSTEM_PROMPT_TOOLS = build_full_prompt()

# Planning mode system prompt
# NOTE: This prompt must exceed 1,024 tokens to be eligible for Anthropic prompt caching.
# Current size: ~1,300 tokens (verified). Do not reduce without checking token count.
PLANNING_PROMPT = """You are CADAgent, an expert at planning 3D CAD models in Fusion 360.

TASK:
Generate a detailed, step-by-step plan for creating the requested CAD model. This plan will be shown to the user for approval before execution.

AVAILABLE OPERATIONS (feature families):
- Sketch & Solids: create_construction_plane, create_sketch, add_circle, add_line, add_rectangle, extrude_profile, revolve_profile, create_loft
- Holes & Threads: create_simple_hole, create_counterbore_hole, create_tapped_hole, create_external_thread
- Edge Mods: apply_fillet, apply_chamfer
- Shell & Pattern: create_shell, create_pattern_feature
- Selection: select_edges, select_faces, select_bodies, clear_edge_selection, clear_face_selection, clear_body_selection, list_features

NOTE: Entity information (edges, faces, bodies) is provided automatically in the "Design Entities" context section. Use the short refs (edge_1, face_1, body_1, ...) directly with select_* tools—no need to list entities first.

PLANNING GUIDELINES:
1. Units: Sketch coordinates are in cm; hole center coordinates and hole/thread depths are in mm; diameters/radii default to mm unless stated—convert and restate user inputs.
2. Break down the model into logical steps; state operation intent (NewBody/Join/Cut/Intersect for features).
3. Choose planes/datum explicitly; call out offsets/angles for custom planes when needed.
4. Order of operations: base solid → shell (if any) → edge treatments → holes/threads → patterns; refresh list_features before patterning.
5. Axis is required for revolutions—pick construction axis normal to sketch plane when unspecified.
6. For holes/threads, specify face_token source, world coordinates in mm, diameters/depths, and fastener clearance.
7. For fillets/chamfers, identify target edges (by selection source) and purpose (safety, clearance, aesthetics).
8. Be specific and executable; if a dimension is unknown, mark it as TODO with a short label instead of guessing.

COMMON PLANNING MISTAKES TO AVOID:
- Forgetting to create a sketch before adding geometry (circles, lines, rectangles require an active sketch)
- Using world coordinates (x,y,z) in sketch geometry instead of sketch-local coordinates (u,v)
- Placing revolve profile on the revolution axis (causes self-intersection; offset by at least the profile radius)
- Using positive extrude distance for Cut operations when sketch is on top face (use negative to cut INTO the body)
- Attempting shell after fillet (shell first, then fillet the edges)
- Specifying hole diameters without considering fastener clearance (M6 bolt needs 6.5mm clearance hole)
- Creating patterns before refreshing feature list (pattern needs current entity tokens)

FORMAT YOUR PLAN:
Use clear numbered steps with specific parameters. Use markdown formatting for structure.

## CAD Model Plan: [Object Name]

**Step 1:** Create base sketch
- Operation: create_sketch
- Plane: XY
- Purpose: Foundation for the base geometry

**Step 2:** Add circle to base sketch
- Operation: add_circle
- Center (u,v): (0, 0) cm
- Radius: 3 cm
- Purpose: Circular profile for cylinder base

**Step 3:** Extrude the circle
- Operation: extrude_profile
- Distance: 5 cm
- Operation type: NewBody
- Purpose: Create main cylinder body

**Step 4:** Drill mounting holes
- Operation: create_simple_hole
- Face: use face ref from Design Entities context (face_1, ...)
- Centers: four holes at (±20, ±20, Z_top) mm
- Diameter: 6.5 mm (clearance for M6)
- Extent: through_all
- Purpose: Mounting pattern

**Step 5:** Fillet edges
- Operation: apply_fillet
- Targets: use edge refs from Design Entities context (edge_1, ...)
- Radius: 1.5 mm
- Purpose: Safety/chamfer alternative

## Result
Short description of the final model and any TODO dimensions.

ADDITIONAL EXAMPLES:

Example A - Enclosure with mounting bosses:
1. Sketch rectangle on XY → extrude 4cm (NewBody) for base
2. Shell with 2mm wall thickness, removing top face
3. Create offset plane at Z=3.5cm for boss sketch
4. Add 4 circles at corners → extrude 0.5cm (Join) for bosses
5. Create tapped holes M3 through bosses for lid screws
6. Fillet outer edges 1mm for safety

Example B - Shaft with keyway:
1. Sketch circle on XY, radius 1cm → extrude 10cm for shaft body
2. Create construction plane offset 9cm from XY
3. Sketch rectangle 3mm x 3mm at top of shaft → extrude -5cm (Cut) for keyway
4. Chamfer shaft ends 1mm x 45°

Example C - Flanged pipe fitting:
1. Sketch circle on XY, radius 2.5cm → extrude 1cm for flange
2. Sketch concentric circle radius 1.5cm → extrude 5cm (Join) for pipe
3. Shell with 3mm thickness, removing both circular end faces
4. Create bolt hole pattern: 6x M8 clearance holes on flange at radius 2cm

FORMATTING RULES:
- NEVER use emojis in any responses
- Use markdown for structure (headers, bold, lists)
- Use clear, professional technical language

Be thorough and precise. The user will review this plan before execution."""


PLAN_SUMMARIZER_PROMPT = """You rewrite long Fusion 360 CAD plans into a compact, human-friendly checklist.

Requirements:
- Keep the plan numbered with no more than 6 main steps.
- Merge micro-actions where possible; each step should be one short sentence.
- Only keep dimensions or parameters that materially affect the design.
- Drop repeated setup notes, coordinate origins, and secondary variations unless absolutely essential.
- If a useful variation must be mentioned, add a single bullet prefixed with "Optional:" after the steps.
- End with one "Result:" line summarizing the final outcome in <20 words.
- Entire response must stay under 80 words.
- Use natural formatting with markdown for structure (bold for step labels, etc).
- NEVER use emojis - use plain text only.

Write clearly so a CAD operator can scan it quickly."""


async def generate_plan(
    user_request: str,
    max_tokens: int = 4096,
    model_name: Optional[str] = None,
    reasoning_effort: str = "high",
    api_keys: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate a CAD model plan using LLM API with streaming.

    This function is used for Planning Mode, where the LLM generates a detailed
    step-by-step plan without executing any operations. The plan is streamed
    to the frontend for user review and approval.

    For GPT-5, this uses the Responses API to enable streaming of reasoning summaries
    (chain of thought) alongside the plan text.

    Args:
        user_request: User's natural language CAD request
        max_tokens: Maximum tokens for response
        model_name: Model to use (defaults to Claude Sonnet 4.5)
        reasoning_effort: GPT-5 reasoning level - "minimal", "low", "medium", "high" (default: "high" for complex planning)

    Yields:
        dict: Event dictionaries with structure:
            {"type": "reasoning", "content": str} - Reasoning summary chunk (GPT-5 only)
            {"type": "text", "content": str} - Plan text chunk

    Raises:
        Exception: On API errors
    """
    model = normalize_model_name(model_name)
    logger.info(f"Generating plan for user request using {model} (streaming mode, reasoning_effort={reasoning_effort})")
    logger.debug(f"User request: {user_request}")

    messages = [
        {
            "role": "user",
            "content": f"Please create a detailed step-by-step plan for this CAD model:\n\n{user_request}"
        }
    ]

    try:
        if model == MODEL_GPT_5 or model.startswith("gpt-"):
            # OpenAI Responses API streaming - enables reasoning summary streaming
            input_prompt = f"{PLANNING_PROMPT}\n\nPlease create a detailed step-by-step plan for this CAD model:\n\n{user_request}"

            _emit_llm_request_payload(
                provider="openai_responses",
                model=model,
                system_prompt=PLANNING_PROMPT,
                messages=[{"role": "user", "content": user_request}],
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )

            # Use Responses API for reasoning summary streaming
            streamed_any_content = False

            try:
                openai_client = _get_openai_client(api_keys)
                responses_client = _get_openai_responses_client(openai_client)
            except AttributeError as exc:
                logger.error("OpenAI Responses API client not available: %s", exc)
                raise

            async with responses_client.stream(
                model=model,
                input=input_prompt,
                reasoning={"effort": reasoning_effort, "summary": "detailed"},
                max_output_tokens=max_tokens,
            ) as stream:
                async for event in stream:
                    _log_stream_payload("openai_responses", event, model)
                    event_type = getattr(event, "type", None)

                    # Log event types to debug
                    if event_type and not event_type.startswith("response.done"):
                        logger.debug(f"Plan streaming event type: {event_type}")

                    # Handle reasoning summary chunks
                    if event_type == "response.reasoning_summary_text.delta":
                        delta = getattr(event, "delta", None)
                        if delta and delta.strip():
                            streamed_any_content = True
                            yield {
                                "type": "reasoning",
                                "content": delta,
                                "raw_text": delta,
                            }

                    # Handle output text chunks - try multiple event types
                    elif event_type in ("response.output_text.delta", "response.text.delta"):
                        delta = getattr(event, "delta", None)
                        if delta and delta.strip():
                            streamed_any_content = True
                            yield {
                                "type": "text",
                                "content": delta,
                                "raw_text": delta,
                            }

                    # Handle output items that might contain text
                    elif event_type == "response.output_item.done":
                        item = getattr(event, "item", None)
                        if item:
                            item_type = getattr(item, "type", None)
                            if item_type == "message":
                                # Extract text from completed message
                                content = getattr(item, "content", [])
                                for block in content:
                                    if getattr(block, "type", None) == "output_text":
                                        text = getattr(block, "text", "")
                                        if text and text.strip():
                                            streamed_any_content = True
                                            yield {
                                                "type": "text",
                                                "content": text,
                                                "raw_text": text,
                                            }

                # Get final response and extract plan text
                final_response = None
                try:
                    final_response = await stream.get_final_response()
                    _log_stream_payload("openai_responses", final_response, model)
                except RuntimeError as exc:
                    logger.warning("OpenAI Responses stream did not return completion event: %s", exc)
                    # final_response will be None, so we'll skip extraction below

                # Extract plan text from final response (fallback if streaming didn't work)
                # This runs regardless of whether get_final_response() succeeded
                if final_response:
                    logger.info("Extracting plan from final response (streamed_any_content=%s)", streamed_any_content)
                    if hasattr(final_response, 'output') and final_response.output:
                        for item in final_response.output:
                            item_type = getattr(item, "type", None)
                            logger.debug(f"Final response output item type: {item_type}")
                            if item_type == "message":
                                content = getattr(item, "content", [])
                                logger.debug(f"Message content blocks: {len(content)}")
                                for block in content:
                                    block_type = getattr(block, "type", None)
                                    logger.debug(f"Message block type: {block_type}")
                                    if block_type == "output_text":
                                        text = getattr(block, "text", "")
                                        if text and text.strip():
                                            logger.info("Found plan text in final response (length: %d chars)", len(text))
                                            if not streamed_any_content:
                                                # Only yield if we didn't already stream this content
                                                yield {
                                                    "type": "text",
                                                    "content": text,
                                                    "raw_text": text,
                                                }
                                            streamed_any_content = True
                else:
                    logger.warning("No final response available - cannot extract plan text")

            if not streamed_any_content:
                logger.warning("Plan generation stream returned no content from OpenAI Responses API; plan output may be empty.")

            logger.info("Plan generation complete (OpenAI Responses API)")

        elif model.startswith("gemini"):
            gemini_async_client = _get_gemini_async_client(api_keys)
            if not gemini_async_client:
                raise RuntimeError("Gemini client is not initialized. Install google-genai before using Gemini models.")

            _emit_llm_request_payload(
                provider="google_gemini",
                model=model,
                system_prompt=PLANNING_PROMPT,
                messages=messages,
                max_tokens=max_tokens,
                reasoning_effort=None,
            )

            # Configure Gemini with thinking mode enabled
            gemini_config = genai_types.GenerateContentConfig(
                system_instruction=PLANNING_PROMPT,
                max_output_tokens=max_tokens,
                thinking_config=genai_types.ThinkingConfig(
                    include_thoughts=True,  # Enable streaming of thought summaries
                ),
            )

            # Stream Gemini response with thinking content
            streamed_any_content = False
            stream = await gemini_async_client.models.generate_content_stream(
                model=model,
                contents=_convert_messages_to_gemini_contents(messages),
                config=gemini_config,
            )

            async for chunk in stream:
                _log_stream_payload("google_gemini", chunk, model)

                # Extract parts from the streaming chunk
                try:
                    candidate = chunk.candidates[0] if chunk.candidates else None
                    if not candidate:
                        continue

                    content = getattr(candidate, "content", None)
                    if not content:
                        continue

                    parts = getattr(content, "parts", None)
                    if not parts:
                        continue

                    # Process each part in the streaming chunk
                    for part in parts:
                        # Check if this part is thinking content
                        is_thought = getattr(part, "thought", False)
                        text = getattr(part, "text", None)

                        if text and text.strip():
                            streamed_any_content = True
                            if is_thought:
                                # Yield thinking/reasoning content
                                yield {
                                    "type": "reasoning",
                                    "content": text,
                                    "raw_text": text,
                                }
                            else:
                                # Yield regular output text
                                yield {
                                    "type": "text",
                                    "content": text,
                                    "raw_text": text,
                                }
                except (AttributeError, IndexError) as e:
                    logger.warning(f"Error processing Gemini streaming chunk: {e}")
                    continue

            if not streamed_any_content:
                logger.warning("Gemini plan generation returned no text content.")

            logger.info("Plan generation complete (Gemini with thinking mode)")

        else:
            # Anthropic/Claude streaming (no reasoning summaries available)
            # Enable prompt caching for planning prompt
            # Using 1-hour TTL since planning sessions often take >5 min for user review
            enable_caching = _should_enable_caching(model)
            system_param = _build_cacheable_system(
                PLANNING_PROMPT,
                enable_caching=enable_caching,
                ttl="1h",  # Planning sessions often exceed 5-min TTL
                split_at_version=False  # Planning prompt doesn't have version marker
            )

            _emit_llm_request_payload(
                provider="anthropic",
                model=model,
                system_prompt=PLANNING_PROMPT,
                messages=messages,
                max_tokens=max_tokens,
                reasoning_effort=None,
            )

            anthropic_client = _get_anthropic_client(api_keys)
            async with anthropic_client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system_param,  # Cacheable system blocks
                messages=messages
            ) as stream:
                streamed_any_text = False
                async for event in stream:
                    _log_stream_payload("anthropic", event, model)
                    event_type = getattr(event, "type", None)

                    if event_type == "text":
                        text = getattr(event, "text", "")
                        if text:
                            streamed_any_text = True
                            yield {
                                "type": "text",
                                "content": text,
                                "raw_text": text,
                            }

                if not streamed_any_text:
                    logger.warning("Plan generation stream returned no text chunks from Anthropic; plan output may be empty.")

            # Get final message for logging
            final_message = await stream.get_final_message()
            _log_stream_payload("anthropic", final_message, model)

            # Log token usage with cache metrics
            usage = final_message.usage
            cache_creation = getattr(usage, 'cache_creation_input_tokens', 0)
            cache_read = getattr(usage, 'cache_read_input_tokens', 0)

            if cache_creation > 0 or cache_read > 0:
                logger.info(
                    f"Plan generation complete. Token usage: input={usage.input_tokens}, "
                    f"output={usage.output_tokens}, cache_write={cache_creation}, cache_hit={cache_read}"
                )
            else:
                logger.info(f"Plan generation complete. Token usage: {final_message.usage}")

    except Exception as e:
        logger.error(f"Error generating plan: {str(e)}")
        raise


async def summarize_plan_for_user(
    plan_text: str,
    max_tokens: int = 1024,
    api_keys: Optional[Dict[str, str]] = None,
) -> str:
    """
    Create a concise, user-friendly plan summary using Claude Haiku.

    Args:
        plan_text: The detailed plan generated for execution.
        max_tokens: Maximum tokens to allocate for the summary response.

    Returns:
        A shortened version of the plan suitable for user review.
        Falls back to the original plan if summarization fails.
    """
    if not plan_text.strip():
        return plan_text

    logger.info("Summarizing plan for user display using claude-haiku-4-5-20251001")

    try:
        summarizer_model = "claude-haiku-4-5-20251001"
        request_messages = [
            {
                "role": "user",
                "content": (
                    "Original plan:\n\n"
                    f"{plan_text}\n\n"
                    "Provide ONLY the condensed plan that meets the system requirements."
                )
            }
        ]
        _emit_llm_request_payload(
            provider="anthropic",
            model=summarizer_model,
            system_prompt=PLAN_SUMMARIZER_PROMPT,
            messages=request_messages,
            max_tokens=max_tokens,
            reasoning_effort=None,
        )

        anthropic_client = _get_anthropic_client(api_keys)
        response = await anthropic_client.messages.create(
            model=summarizer_model,
            max_tokens=max_tokens,
            system=PLAN_SUMMARIZER_PROMPT,
            messages=request_messages
        )
        _log_stream_payload("anthropic", response, summarizer_model)

        summary_parts: List[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                summary_parts.append(getattr(block, "text", ""))

        summary = "".join(summary_parts).strip()
        if not summary:
            logger.warning("Plan summarizer returned empty content; reverting to full plan.")
            return markdown_to_html(plan_text)

        logger.info("Plan summarization completed successfully.")
        return markdown_to_html(summary)

    except Exception as exc:
        logger.exception("Failed to summarize plan with claude-haiku: %s", exc)
        return markdown_to_html(plan_text)


def _convert_tools_to_openai_format(tools: List[dict]) -> List[dict]:
    """
    Convert Anthropic tool schema to OpenAI Responses API function format.

    Responses API uses internally-tagged function definitions (no nested 'function' object)
    and strict mode is enabled by default.
    """
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"]
        })
    return openai_tools


def _convert_openai_response_to_anthropic_format(response: Any) -> dict:
    """
    Convert OpenAI Responses API response to Anthropic-compatible format.

    Responses API returns an 'output' array of typed items (message, function_call, reasoning, etc.)
    instead of a single 'message' object. We iterate through output items to extract content.
    """
    content = []
    stop_reason = "end_turn"

    # Iterate through output items (reasoning, message, function_call, function_call_output)
    for item in response.output:
        item_type = getattr(item, "type", None)

        if item_type == "message":
            # Extract text from message content
            message_content = getattr(item, "content", [])
            for block in message_content:
                block_type = getattr(block, "type", None)
                if block_type == "output_text":
                    text = getattr(block, "text", "")
                    if text:
                        content.append({"type": "text", "text": text})

        elif item_type == "function_call":
            # Tool calls become tool_use blocks
            stop_reason = "tool_use"

            # Extract function call details
            tool_name = getattr(item, "name", "")
            call_id = getattr(item, "call_id", "")
            arguments_str = getattr(item, "arguments", "")

            # Parse arguments - OpenAI returns them as a JSON string, not a dict
            arguments = {}
            if arguments_str:
                if isinstance(arguments_str, str):
                    try:
                        arguments = json.loads(arguments_str)
                        if not isinstance(arguments, dict):
                            logger.error(f"Function arguments parsed to non-dict type: {type(arguments)}")
                            arguments = {}
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Failed to parse function arguments as JSON: {arguments_str}, error: {e}")
                        arguments = {}
                elif isinstance(arguments_str, dict):
                    # Already a dict (shouldn't happen with OpenAI Responses API)
                    arguments = arguments_str
                else:
                    logger.warning(f"Unexpected arguments type: {type(arguments_str)}")
                    arguments = {}

            # Log the tool call for debugging
            logger.debug(f"Converted function_call to tool_use: name={tool_name}, call_id={call_id}, arguments={arguments}")

            content.append({
                "type": "tool_use",
                "id": call_id,
                "name": tool_name,
                "input": arguments
            })

    return {
        "stop_reason": stop_reason,
        "content": content,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
    }


def _convert_tools_to_gemini_format(tools: List[dict]) -> List[Any]:
    """
    Convert internal tool schema to Google Gemini FunctionDeclarations.
    """
    if not genai_types:
        raise RuntimeError("google-genai SDK is not installed; cannot use Gemini models.")

    declarations: List[genai_types.FunctionDeclaration] = []
    for tool in tools or []:
        schema_dict = tool.get("input_schema") or tool.get("parameters") or {}
        try:
            # Convert dict to JSONSchema object first
            json_schema_obj = genai_types.JSONSchema(**schema_dict)
            parameters_schema = genai_types.Schema.from_json_schema(json_schema=json_schema_obj)
        except Exception as exc:  # pragma: no cover - defensive conversion
            logger.warning("Gemini schema conversion failed for %s: %s; using loose object schema", tool.get("name"), exc)
            parameters_schema = genai_types.Schema(type="object", properties={})

        declarations.append(
            genai_types.FunctionDeclaration(
                name=tool.get("name"),
                description=tool.get("description"),
                parameters=parameters_schema,
            )
        )

    if not declarations:
        return []

    return [genai_types.Tool(functionDeclarations=declarations)]


def _convert_messages_to_gemini_contents(messages: List[dict]) -> List[Any]:
    """
    Transform Anthropic-style message list into Gemini Content objects.
    """
    if not genai_types:
        raise RuntimeError("google-genai SDK is not installed; cannot use Gemini models.")

    contents: List[genai_types.Content] = []
    tool_id_to_name: Dict[str, Optional[str]] = {}

    for message in messages or []:
        role = message.get("role") or "user"
        raw_content = message.get("content", "")
        parts: List[genai_types.Part] = []

        if isinstance(raw_content, str):
            if raw_content.strip():
                parts.append(genai_types.Part.from_text(text=raw_content))
        elif isinstance(raw_content, list):
            for block in raw_content:
                if not isinstance(block, dict):
                    parts.append(genai_types.Part.from_text(text=str(block)))
                    continue

                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(genai_types.Part.from_text(text=text))
                elif block_type == "image":
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        try:
                            parts.append(genai_types.Part.from_bytes(data=base64.b64decode(data), mime_type=media_type))
                        except Exception as exc:  # pragma: no cover - conversion guard
                            logger.warning("Failed to decode image for Gemini payload: %s", exc)
                elif block_type == "tool_use":
                    tool_name = block.get("name")
                    call_id = block.get("id") or block.get("tool_use_id")
                    if call_id:
                        tool_id_to_name[call_id] = tool_name
                    args = block.get("input") or {}
                    function_call = genai_types.FunctionCall(
                        name=tool_name,
                        args=args,
                        id=call_id,
                    )
                    # Restore thought_signature if present (needed for Gemini 3 Pro)
                    thought_sig_b64 = block.get("thought_signature")
                    if thought_sig_b64:
                        import base64
                        thought_sig_bytes = base64.b64decode(thought_sig_b64)
                        parts.append(genai_types.Part(function_call=function_call, thoughtSignature=thought_sig_bytes))
                    else:
                        parts.append(genai_types.Part(function_call=function_call))
                elif block_type == "tool_result":
                    call_id = block.get("tool_use_id")
                    is_error = block.get("is_error", False)
                    result_content = block.get("content", [])

                    text_chunks: List[str] = []
                    image_blobs: List[tuple[str, str]] = []
                    if isinstance(result_content, list):
                        for item in result_content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    text_chunks.append(item.get("text", ""))
                                elif item.get("type") == "image":
                                    src = item.get("source", {})
                                    if src.get("type") == "base64":
                                        image_blobs.append((src.get("media_type", "image/png"), src.get("data", "")))
                            else:
                                text_chunks.append(str(item))
                    else:
                        text_chunks.append(str(result_content))

                    response_payload: Dict[str, Any] = {
                        "result": "\n".join([t for t in text_chunks if t]).strip()
                    }
                    if is_error:
                        response_payload["is_error"] = True

                    function_response = genai_types.FunctionResponse(
                        name=tool_id_to_name.get(call_id),
                        response=response_payload,
                        id=call_id,
                    )
                    parts.append(genai_types.Part(function_response=function_response))

                    for media_type, data in image_blobs:
                        try:
                            parts.append(genai_types.Part.from_bytes(data=base64.b64decode(data), mime_type=media_type))
                        except Exception as exc:  # pragma: no cover - conversion guard
                            logger.warning("Failed to decode image from tool_result for Gemini payload: %s", exc)
                else:
                    # Fallback: preserve unknown blocks as text
                    parts.append(genai_types.Part.from_text(text=json.dumps(block)))

        if parts:
            contents.append(genai_types.Content(role=role, parts=parts))

    return contents


def _convert_gemini_response_to_anthropic_format(response: Any) -> dict:
    """
    Convert Gemini GenerateContentResponse into Anthropic-compatible structure.

    Note: This function filters out thought parts (where part.thought == True) since
    thinking content is streamed separately via the reasoning_callback during execution.
    Only regular text and tool calls are included in the final response.
    """
    content: List[dict] = []
    stop_reason = "end_turn"

    if response is None:
        raise RuntimeError("Gemini API returned no response object")

    candidate = None
    try:
        candidate = response.candidates[0] if getattr(response, "candidates", None) else None
    except Exception:
        candidate = None

    parts = getattr(candidate, "content", None)
    if parts and getattr(parts, "parts", None):
        for idx, part in enumerate(parts.parts):
            # Skip thinking content - it's already streamed via reasoning_callback
            is_thought = getattr(part, "thought", False)
            if is_thought:
                continue

            if getattr(part, "function_call", None):
                fc = part.function_call
                stop_reason = "tool_use"
                tool_use_block = {
                    "type": "tool_use",
                    "id": getattr(fc, "id", None) or f"toolu_{idx+1}",
                    "name": getattr(fc, "name", None),
                    "input": getattr(fc, "args", None) or {},
                }
                # Preserve thought_signature if present (needed for Gemini 3 Pro)
                thought_sig = getattr(part, "thoughtSignature", None) or getattr(part, "thought_signature", None)
                if thought_sig:
                    # Store as base64 string for JSON serialization
                    import base64
                    tool_use_block["thought_signature"] = base64.b64encode(thought_sig).decode('utf-8')
                content.append(tool_use_block)
            elif getattr(part, "function_response", None):
                fr = part.function_response
                resp_data = getattr(fr, "response", None)
                text = ""
                if isinstance(resp_data, dict):
                    text = resp_data.get("result") or resp_data.get("text") or json.dumps(resp_data)
                elif resp_data is not None:
                    text = str(resp_data)
                if text:
                    content.append({"type": "text", "text": text})
            elif getattr(part, "text", None):
                text = part.text
                if text:
                    content.append({"type": "text", "text": text})
    else:
        # Fallback to response.text if no structured parts were returned
        fallback_text = getattr(response, "text", None)
        if fallback_text:
            content.append({"type": "text", "text": fallback_text})

    usage_meta = getattr(response, "usage_metadata", None)
    usage: Dict[str, int] = {}
    if usage_meta:
        try:
            usage["input_tokens"] = getattr(usage_meta, "prompt_token_count", 0) or 0
            usage["output_tokens"] = getattr(usage_meta, "response_token_count", 0) or 0
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to extract Gemini usage metadata", exc_info=True)

    return {
        "stop_reason": stop_reason,
        "content": content,
        "usage": usage,
    }


async def call_claude_with_tools(
    messages: List[dict],
    tools: List[dict] = None,
    system_prompt: str = None,
    max_tokens: int = 4096,
    model_name: Optional[str] = None,
    reasoning_effort: str = "low",
    session_id: Optional[str] = None,
    iteration: Optional[int] = None,
    session_path: Optional[str] = None,
    api_call_counter: Optional[Dict[str, int]] = None,
    session_context: Optional[Dict[str, Any]] = None,
    reasoning_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    user_token: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Call LLM API with tool support for agentic CAD operations.

    Args:
        messages: List of message dictionaries in Anthropic format
        tools: List of tool definitions (defaults to TOOLS)
        system_prompt: System prompt (defaults to SYSTEM_PROMPT)
        max_tokens: Maximum tokens for response
        model_name: Model to use (defaults to Claude Sonnet 4.5)
        reasoning_effort: GPT-5 reasoning level - "minimal", "low", "medium", "high" (default: "low" for tool execution)
        session_id: Session identifier for logging (optional)
        iteration: Current iteration number for logging (optional)
        session_path: Path to session log directory (optional)
        api_call_counter: Counter dict for API calls {'count': int} (optional)
        session_context: Session-specific context for logging (optional)

    Returns:
        dict: Response with structure:
        {
            "stop_reason": "tool_use" | "end_turn",
            "content": [...],  # Contains text and/or tool_use blocks
            "usage": {...}
        }

    Raises:
        Exception: On API errors after retries
    """
    if tools is None:
        tools = TOOLS
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    # Sanitize tool schemas to avoid provider-side 400s (e.g., top-level allOf/oneOf)
    safe_tools = _prepare_tools_for_api(tools)

    model = normalize_model_name(model_name)
    requested_model = model

    # Route through Supabase API Gateway for usage tracking if user_token is provided
    # Allow bypass via env flag when quota tracking is not needed
    bypass_gateway = os.environ.get("BYPASS_SUPABASE_GATEWAY", "").lower() in ("1", "true", "yes", "on")
    if user_token is not None and bypass_gateway:
        logger.info("BYPASS_SUPABASE_GATEWAY enabled – sending Anthropics calls directly (no quota tracking)")

    if user_token is not None and not bypass_gateway:
        logger.info(f"Routing LLM call through Supabase API Gateway with usage tracking (model={model})")
        try:
            from .supabase_client import SupabaseAPIGateway
            exp_ts = _decode_jwt_exp(user_token)
            if exp_ts:
                exp_dt = datetime.fromtimestamp(exp_ts, tz=timezone.utc)
                now_dt = datetime.now(tz=timezone.utc)
                seconds_left = int((exp_dt - now_dt).total_seconds())
                logger.info(f"[auth-debug] user_token exp={exp_dt.isoformat()} (in {seconds_left}s)")

            # Determine provider based on model
            if model.startswith("gpt-") or model == MODEL_GPT_5:
                provider = "openai"
            else:
                provider = "anthropic"

            # Create gateway client
            gateway = SupabaseAPIGateway(user_token=user_token, timeout=120.0)

            # Call through gateway
            response_data = await gateway.call_llm(
                provider=provider,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                # Always forward the system prompt; the edge function will place it
                # into the Chat Completions message array for OpenAI and as the
                # dedicated system field for Anthropic.
                system=system_prompt,
                tools=safe_tools,
            )

            # Extract result from gateway response
            result = response_data.get("result")
            usage_data = response_data.get("usage")

            if result is None:
                raise ValueError("No result returned from Supabase API Gateway")

            logger.info(f"Supabase API Gateway call successful: cost_cents={usage_data.get('cost_cents')}, "
                       f"remaining_cents={usage_data.get('remaining_cents')}")

            # Log usage info if session logging enabled
            if session_path and api_call_counter and session_context:
                try:
                    from .session_logger import log_api_call
                    # Create a mock response object for compatibility with existing logging
                    class MockResponse:
                        def __init__(self, result_data, usage_info):
                            self.stop_reason = result_data.get("stop_reason")
                            self.content = result_data.get("content", [])
                            class MockUsage:
                                def __init__(self, usage_info):
                                    self.input_tokens = usage_info.get("input_tokens", 0)
                                    self.output_tokens = usage_info.get("output_tokens", 0)
                            self.usage = MockUsage(result_data.get("usage", {}))

                    mock_response = MockResponse(result, usage_data)
                    log_api_call(
                        session_path=session_path,
                        iteration=iteration or 0,
                        api_call_num=api_call_counter['count'],
                        llm_response=mock_response,
                        context=session_context
                    )
                    api_call_counter['count'] += 1
                except Exception as e:
                    logger.warning(f"Failed to log API call: {e}")

            return result

        except ValueError as e:
            # Quota exceeded or authentication errors
            error_msg = str(e)
            if "quota exceeded" in error_msg.lower():
                logger.error(f"Quota exceeded for user: {error_msg}")
            elif "authentication failed" in error_msg.lower():
                logger.error(f"Authentication failed: {error_msg}")
            raise
        except Exception as e:
            logger.error(f"Supabase API Gateway error: {e}")
            raise

    max_retries = 5
    base_retry_delay = 2  # seconds - increased for better handling of overloaded APIs

    for attempt in range(max_retries):
        try:
            logger.info(f"Calling LLM API with {model} (attempt {attempt + 1}/{max_retries}, reasoning_effort={reasoning_effort})")

            if model == MODEL_GPT_5 or model.startswith("gpt-"):
                openai_client = _get_openai_client(api_keys)
                # OpenAI Responses API call
                openai_tools = _convert_tools_to_openai_format(safe_tools)
                openai_messages = []

                # Note: system_prompt will be passed as 'instructions' parameter
                # (Responses API separates instructions from input messages)

                # Convert messages to OpenAI Responses API format
                # In Responses API, items are separate (not embedded like Chat Completions)
                for idx, msg in enumerate(messages):
                    role = msg["role"]
                    content = msg["content"]

                    if isinstance(content, str):
                        openai_messages.append({"role": role, "content": content})
                    elif isinstance(content, list):
                        # Handle complex content (text, image, tool_use, tool_result)
                        # In Responses API, these become separate items
                        text_parts = []

                        for block in content:
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "image":
                                # Images are supported in Responses API as input_image type
                                source = block.get("source", {})
                                media_type = source.get("media_type", "image/png")
                                image_data = source.get("data", "")

                                # Add image as a separate message (no "type": "message" wrapper!)
                                openai_messages.append({
                                    "role": role,
                                    "content": [{
                                        "type": "input_image",
                                        "image_url": f"data:{media_type};base64,{image_data}",
                                        "detail": "auto"
                                    }]
                                })
                            elif block.get("type") == "tool_use":
                                # Tool calls become separate function_call items in Responses API
                                # Arguments must be a JSON string, not a dict
                                tool_input = block.get("input", {})
                                arguments_str = json.dumps(tool_input) if tool_input else "{}"
                                openai_messages.append({
                                    "type": "function_call",
                                    "call_id": block.get("id"),
                                    "name": block.get("name"),
                                    "arguments": arguments_str
                                })
                            elif block.get("type") == "tool_result":
                                # Tool results become function_call_output items in Responses API
                                # IMPORTANT: Tool results can contain both text AND images
                                result_content = block.get("content", [])
                                result_text_parts = []
                                result_images = []
                                
                                if isinstance(result_content, list):
                                    for item in result_content:
                                        if isinstance(item, dict):
                                            if item.get("type") == "text":
                                                result_text_parts.append(item.get("text", ""))
                                            elif item.get("type") == "image":
                                                # Extract image data from tool result
                                                source = item.get("source", {})
                                                media_type = source.get("media_type", "image/png")
                                                image_data = source.get("data", "")
                                                result_images.append({
                                                    "type": "input_image",
                                                    "image_url": f"data:{media_type};base64,{image_data}",
                                                    "detail": "auto"
                                                })
                                        else:
                                            result_text_parts.append(str(item))
                                else:
                                    result_text_parts.append(str(result_content))
                                
                                result_text = "\n".join(result_text_parts) if result_text_parts else ""

                                # Add function_call_output with text
                                openai_messages.append({
                                    "type": "function_call_output",
                                    "call_id": block.get("tool_use_id"),
                                    "output": result_text
                                })
                                
                                # If there are images in the tool result, add them as a separate user message
                                # (OpenAI Responses API doesn't support images directly in function_call_output)
                                if result_images:
                                    openai_messages.append({
                                        "role": "user",
                                        "content": result_images
                                    })

                        # If there's text content, add it as a message item
                        if text_parts:
                            openai_messages.append({"role": role, "content": "\n".join(text_parts)})

                # Responses API uses instructions/input split and max_output_tokens
                _emit_llm_request_payload(
                    provider="openai_responses",
                    model=model,
                    system_prompt=system_prompt,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    tools=openai_tools,
                )

                # Calculate appropriate max_output_tokens based on reasoning effort
                # Reasoning models consume output tokens for BOTH reasoning AND tool calls
                effective_max_output_tokens = _get_openai_max_output_tokens(reasoning_effort or "low")
                # OpenAI Responses API may not accept "xhigh"; map it to "high" while keeping the larger token budget.
                openai_effort = "high" if reasoning_effort == "xhigh" else reasoning_effort
                reasoning_param = {"effort": openai_effort, "summary": "detailed"} if openai_effort else None
                logger.info(f"Using max_output_tokens={effective_max_output_tokens} for reasoning_effort={reasoning_effort}")

                use_streaming = reasoning_callback is not None

                streamed_reasoning = False

                if use_streaming:
                    try:
                        responses_client = _get_openai_responses_client(openai_client)
                    except AttributeError as exc:
                        logger.error("OpenAI Responses API client not available for streaming: %s", exc)
                        # Fall back to non-streaming call below
                        use_streaming = False

                if use_streaming:
                    streamed_response = None
                    try:
                        stream_kwargs = {
                            "model": model,
                            "instructions": system_prompt,
                            "input": openai_messages,
                            "tools": openai_tools,
                            "max_output_tokens": effective_max_output_tokens,
                            "store": False,
                        }
                        if reasoning_param:
                            stream_kwargs["reasoning"] = reasoning_param
                        async with responses_client.stream(**stream_kwargs) as stream:
                            async for event in stream:
                                _log_stream_payload("openai_responses", event, model)
                                event_type = getattr(event, "type", None)
                                if (
                                    event_type == "response.reasoning_summary_text.delta"
                                    and reasoning_callback
                                ):
                                    delta = getattr(event, "delta", None)
                                    if delta and delta.strip():
                                        try:
                                            await reasoning_callback(delta)
                                            streamed_reasoning = True
                                        except Exception as cb_exc:  # pragma: no cover - UI failures shouldn't crash execution
                                            logger.warning("Reasoning callback failed: %s", cb_exc, exc_info=True)

                            try:
                                streamed_response = await stream.get_final_response()
                            except RuntimeError as exc:
                                logger.warning("OpenAI Responses stream did not return completion event: %s", exc)
                                streamed_response = None

                        response = streamed_response
                    except Exception as exc:
                        logger.error("Streaming call to OpenAI Responses failed, retrying with non-streaming: %s", exc, exc_info=True)
                        response = None
                        use_streaming = False

                if not use_streaming:
                    create_kwargs = {
                        "model": model,
                        "instructions": system_prompt,
                        "input": openai_messages,
                        "tools": openai_tools,
                        "max_output_tokens": effective_max_output_tokens,
                        "store": False,  # We manage conversation state locally
                    }
                    if reasoning_param:
                        create_kwargs["reasoning"] = reasoning_param
                    response = await openai_client.responses.create(**create_kwargs)

                logger.info(f"OpenAI Responses API response received")
                if response is not None:
                    _log_stream_payload("openai_responses", response, model)
                    logger.info(f"Token usage: {response.usage}")

                    # Emit reasoning summaries if none arrived as streaming deltas
                    if reasoning_callback and not streamed_reasoning:
                        try:
                            summaries: List[str] = []
                            for item in getattr(response, "output", []) or []:
                                if getattr(item, "type", None) == "reasoning":
                                    for summary_obj in getattr(item, "summary", []) or []:
                                        text = getattr(summary_obj, "text", "") if hasattr(summary_obj, "text") else summary_obj.get("text") if isinstance(summary_obj, dict) else ""
                                        if text and str(text).strip():
                                            summaries.append(str(text))
                            for summary_text in summaries:
                                await reasoning_callback(summary_text)
                                streamed_reasoning = True
                        except Exception as cb_exc:  # pragma: no cover
                            logger.warning("Reasoning callback (final summaries) failed: %s", cb_exc, exc_info=True)

                    # Debug log the output items to understand the structure
                    if hasattr(response, 'output') and response.output:
                        logger.debug(f"Response output items count: {len(response.output)}")
                        for idx, item in enumerate(response.output):
                            item_type = getattr(item, "type", None)
                            logger.debug(f"Output item {idx}: type={item_type}")
                            if item_type == "function_call":
                                name = getattr(item, "name", None)
                                call_id = getattr(item, "call_id", None)
                                arguments_raw = getattr(item, "arguments", None)
                                logger.debug(f"  Function call: name={name}, call_id={call_id}, arguments_type={type(arguments_raw)}")
                                logger.debug(f"  Arguments (raw): {arguments_raw!r}")

                if response is None:
                    raise RuntimeError("OpenAI Responses API returned no response object")

                # Normalize usage for cache-aware metrics (OpenAI Reports cached_tokens in input_tokens_details)
                usage_stats = _extract_usage_stats(response)

                result = _convert_openai_response_to_anthropic_format(response)

                # Log API call if session logging is enabled
                if session_path and api_call_counter and session_context:
                    try:
                        from .session_logger import log_api_call, log_cache_metrics
                        log_api_call(
                            session_path=session_path,
                            iteration=iteration or 0,
                            api_call_num=api_call_counter['count'],
                            llm_response=response,
                            context=session_context
                        )
                        log_cache_metrics(
                            session_path=session_path,
                            iteration=iteration or 0,
                            api_call_num=api_call_counter['count'],
                            cache_creation_tokens=usage_stats.get("cache_creation_tokens", 0),
                            cache_read_tokens=usage_stats.get("cache_read_tokens", 0),
                            input_tokens=0 if usage_stats.get("_input_tokens_from_cacheable_prompt") else usage_stats.get("input_tokens", 0),
                            model_name=model,
                        )
                        api_call_counter['count'] += 1
                    except Exception as e:
                        logger.warning(f"Failed to log API call: {e}")

                return result

            elif model.startswith("gemini"):
                gemini_async_client = _get_gemini_async_client(api_keys)
                if not gemini_async_client:
                    raise RuntimeError("Gemini client is not initialized. Install google-genai before using Gemini models.")

                gemini_tools = _convert_tools_to_gemini_format(safe_tools) if safe_tools else []
                gemini_contents = _convert_messages_to_gemini_contents(messages)

                tool_config = None
                if gemini_tools:
                    tool_config = genai_types.ToolConfig(
                        function_calling_config=genai_types.FunctionCallingConfig(
                            mode=genai_types.FunctionCallingConfigMode.ANY
                        )
                    )

                # Enable thinking mode if reasoning_callback is provided
                thinking_config = None
                if reasoning_callback:
                    thinking_config = genai_types.ThinkingConfig(
                        include_thoughts=True,  # Enable streaming of thought summaries
                    )

                # Gemini 3 Pro with thinking needs larger token budget to prevent empty responses
                # (thinking tokens can consume the entire output budget, leaving nothing for actual output)
                gemini_max_tokens = max_tokens
                if model.startswith("gemini-3"):
                    gemini_max_tokens = max(max_tokens * 2, 8192)
                    logger.debug(f"Increased max_output_tokens for {model} from {max_tokens} to {gemini_max_tokens}")

                gemini_config = genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=gemini_tools or None,
                    tool_config=tool_config,
                    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
                    max_output_tokens=gemini_max_tokens,
                    thinking_config=thinking_config,
                )

                _emit_llm_request_payload(
                    provider="google_gemini",
                    model=model,
                    system_prompt=system_prompt,
                    messages=messages,
                    max_tokens=max_tokens,
                    reasoning_effort=None,
                    tools=safe_tools,
                )

                # Use streaming if reasoning callback is provided
                use_streaming = reasoning_callback is not None
                streamed_reasoning = False
                response = None

                try:
                    if use_streaming:
                        # Stream with thinking mode to capture reasoning
                        stream = await gemini_async_client.models.generate_content_stream(
                            model=model,
                            contents=gemini_contents,
                            config=gemini_config,
                        )

                        # Collect all parts for final response reconstruction
                        all_parts = []

                        async for chunk in stream:
                            _log_stream_payload("google_gemini", chunk, model)

                            # Extract parts from the streaming chunk
                            try:
                                candidate = chunk.candidates[0] if chunk.candidates else None
                                if not candidate:
                                    continue

                                content = getattr(candidate, "content", None)
                                if not content:
                                    continue

                                parts = getattr(content, "parts", None)
                                if not parts:
                                    continue

                                # Process each part in the streaming chunk
                                for part in parts:
                                    all_parts.append(part)

                                    # Check if this part is thinking content
                                    is_thought = getattr(part, "thought", False)
                                    text = getattr(part, "text", None)

                                    if text and text.strip() and is_thought:
                                        # Send thinking content to reasoning callback
                                        try:
                                            await reasoning_callback(text)
                                            streamed_reasoning = True
                                        except Exception as cb_exc:  # pragma: no cover
                                            logger.warning("Reasoning callback failed: %s", cb_exc, exc_info=True)

                            except (AttributeError, IndexError) as e:
                                logger.warning(f"Error processing Gemini streaming chunk: {e}")
                                continue

                        # Reconstruct final response from all parts
                        # Get the last chunk as the base response
                        response = chunk if 'chunk' in locals() else None

                    else:
                        # Non-streaming execution
                        response = await gemini_async_client.models.generate_content(
                            model=model,
                            contents=gemini_contents,
                            config=gemini_config,
                        )

                        _log_stream_payload("google_gemini", response, model)
                except Exception as gemini_exc:
                    # Log detailed error information for debugging
                    status_code = getattr(gemini_exc, "status_code", None)
                    error_message = str(gemini_exc)

                    # Provide helpful context for common Gemini errors
                    if status_code and int(status_code) >= 500:
                        logger.error(
                            "Gemini API server error (status %s) for model %s. "
                            "This is typically a transient infrastructure issue. "
                            "Please retry your request. Error: %s",
                            status_code,
                            model,
                            error_message
                        )
                    elif "thought_signature" in error_message.lower():
                        logger.error(
                            "Gemini API validation error for model %s: %s. "
                            "This may indicate conversation history incompatibility. "
                            "Try starting a new session.",
                            model,
                            error_message
                        )
                    else:
                        logger.error("Gemini API error for model %s: %s", model, error_message)

                    # Re-raise the exception - no fallback to other models
                    raise

                if response is None:
                    raise RuntimeError("Gemini API returned no response object")

                result = _convert_gemini_response_to_anthropic_format(response)

                # Log API call if session logging is enabled
                if session_path and api_call_counter and session_context:
                    try:
                        from .session_logger import log_api_call
                        log_api_call(
                            session_path=session_path,
                            iteration=iteration or 0,
                            api_call_num=api_call_counter['count'],
                            llm_response=response,
                            context=session_context
                        )
                        api_call_counter['count'] += 1
                    except Exception as e:
                        logger.warning(f"Failed to log Gemini API call: {e}")

                return result

            else:
                # Anthropic/Claude API call
                # Enable prompt caching for Anthropic models (reduces cost by 90% and latency by 85%)
                enable_caching = _should_enable_caching(model)

                # Convert system prompt to cacheable format
                system_param = _build_cacheable_system(
                    system_prompt,
                    enable_caching=enable_caching,
                    ttl="5m"  # Use 5-minute cache (1.25x write cost, 0.1x read cost)
                )

                # Add cache markers to tools (last tool gets the breakpoint)
                tools_param = _add_tool_cache_markers(safe_tools, enable_caching=enable_caching, ttl="5m")

                # Add cache marker to conversation history (last user message)
                messages_param = _add_conversation_cache_marker(messages, enable_caching=enable_caching, ttl="5m")
                messages_param = _sanitize_anthropic_messages_for_api(messages_param)

                # Enable extended thinking when user has set reasoning effort (not None/off)
                enable_thinking = reasoning_effort is not None and reasoning_effort not in ("off", "none")
                base_model = _get_base_claude_model(model)  # Get actual model ID for API

                # Configure extended thinking if user enabled it
                thinking_param = None
                if enable_thinking:
                    requested_budget = _get_claude_thinking_budget(reasoning_effort or "medium")
                    max_budget = max_tokens - CLAUDE_THINKING_OUTPUT_HEADROOM

                    if max_budget < CLAUDE_THINKING_BUDGET_MIN:
                        logger.warning(
                            "Disabling Claude extended thinking: max_tokens=%s leaves < %s tokens for minimum budget.",
                            max_tokens,
                            CLAUDE_THINKING_BUDGET_MIN,
                        )
                        enable_thinking = False
                    else:
                        budget_tokens = min(requested_budget, max_budget)
                        if budget_tokens < requested_budget:
                            logger.warning(
                                "Clamped Claude thinking budget from %s to %s to satisfy max_tokens=%s and headroom=%s.",
                                requested_budget,
                                budget_tokens,
                                max_tokens,
                                CLAUDE_THINKING_OUTPUT_HEADROOM,
                            )
                        thinking_param = {
                            "type": "enabled",
                            "budget_tokens": budget_tokens
                        }
                        logger.info(f"Enabled Claude extended thinking with budget_tokens={budget_tokens}")

                _emit_llm_request_payload(
                    provider="anthropic",
                    model=base_model,
                    system_prompt=system_prompt,
                    messages=messages,
                    max_tokens=max_tokens,
                    reasoning_effort=None,
                    tools=safe_tools,
                )

                # Use streaming if thinking enabled with reasoning_callback to capture thinking blocks
                use_streaming = enable_thinking and reasoning_callback is not None
                streamed_thinking = False

                if use_streaming:
                    # Stream with extended thinking to capture reasoning
                    logger.info("Using streaming for Claude COT to capture thinking blocks")
                    anthropic_client = _get_anthropic_client(api_keys)
                    async with anthropic_client.messages.stream(
                        model=base_model,
                        max_tokens=max_tokens,
                        system=system_param,
                        tools=tools_param,
                        messages=messages_param,
                        thinking=thinking_param
                    ) as stream:
                        async for event in stream:
                            _log_stream_payload("anthropic", event, model)
                            event_type = getattr(event, "type", None)

                            # Capture thinking content blocks
                            if event_type == "content_block_start":
                                content_block = getattr(event, "content_block", None)
                                if content_block and getattr(content_block, "type", None) == "thinking":
                                    # Thinking block started
                                    pass
                            elif event_type == "content_block_delta":
                                delta = getattr(event, "delta", None)
                                if delta and getattr(delta, "type", None) == "thinking_delta":
                                    thinking_text = getattr(delta, "thinking", None)
                                    if thinking_text and reasoning_callback:
                                        try:
                                            await reasoning_callback(thinking_text)
                                            streamed_thinking = True
                                        except Exception as cb_exc:  # pragma: no cover
                                            logger.warning("Reasoning callback failed: %s", cb_exc, exc_info=True)

                        # Get final response after streaming completes
                        response = await stream.get_final_message()
                else:
                    # Non-streaming API call
                    anthropic_client = _get_anthropic_client(api_keys)
                    api_kwargs = {
                        "model": base_model,
                        "max_tokens": max_tokens,
                        "system": system_param,
                        "tools": tools_param,
                        "messages": messages_param
                    }
                    if thinking_param:
                        api_kwargs["thinking"] = thinking_param

                    response = await anthropic_client.messages.create(**api_kwargs)

                    # If thinking enabled but no streaming, emit thinking blocks to callback
                    if enable_thinking and reasoning_callback:
                        try:
                            for content_block in response.content:
                                if getattr(content_block, "type", None) == "thinking":
                                    thinking_text = getattr(content_block, "thinking", None)
                                    if thinking_text:
                                        await reasoning_callback(thinking_text)
                                        streamed_thinking = True
                        except Exception as cb_exc:  # pragma: no cover
                            logger.warning("Reasoning callback (non-streaming) failed: %s", cb_exc, exc_info=True)

                logger.info(f"Claude API response: stop_reason={response.stop_reason}")
                _log_stream_payload("anthropic", response, model)

                # Log token usage with cache metrics
                usage = response.usage
                cache_creation = getattr(usage, 'cache_creation_input_tokens', 0)
                cache_read = getattr(usage, 'cache_read_input_tokens', 0)

                if cache_creation > 0 or cache_read > 0:
                    # Calculate cost savings (cache reads are 90% cheaper)
                    savings_tokens = int(cache_read * 0.9)
                    savings_percent = int((cache_read / max(usage.input_tokens + cache_read, 1)) * 90) if cache_read > 0 else 0

                    logger.info(
                        f"Token usage: input={usage.input_tokens}, output={usage.output_tokens}, "
                        f"cache_write={cache_creation}, cache_hit={cache_read} "
                        f"(Savings: ~{savings_tokens} equivalent tokens, ~{savings_percent}% cost reduction)"
                    )
                else:
                    logger.info(f"Token usage: input={usage.input_tokens}, output={usage.output_tokens}")

                # Parse response into expected format
                result = {
                    "stop_reason": response.stop_reason,
                    "content": _sanitize_anthropic_assistant_content([block.model_dump() for block in response.content]),
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    }
                }

                # Log API call if session logging is enabled
                if session_path and api_call_counter and session_context:
                    try:
                        from .session_logger import log_api_call, log_cache_metrics
                        log_api_call(
                            session_path=session_path,
                            iteration=iteration or 0,
                            api_call_num=api_call_counter['count'],
                            llm_response=response,
                            context=session_context
                        )
                        # Log cache metrics for cost tracking
                        if enable_caching:
                            log_cache_metrics(
                                session_path=session_path,
                                iteration=iteration or 0,
                                api_call_num=api_call_counter['count'],
                                cache_creation_tokens=cache_creation,
                                cache_read_tokens=cache_read,
                                input_tokens=usage.input_tokens,
                                model_name=model,
                            )
                        api_call_counter['count'] += 1
                    except Exception as e:
                        logger.warning(f"Failed to log API call: {e}")

                return result

        except Exception as e:
            logger.error(f"LLM API error (attempt {attempt + 1}/{max_retries}): {str(e)}")

            if attempt < max_retries - 1:
                # Exponential backoff with jitter to prevent thundering herd
                exponential_delay = base_retry_delay * (2 ** attempt)
                # Add jitter: random value between 0% and 25% of the delay
                jitter = random.uniform(0, exponential_delay * 0.25)
                wait_time = exponential_delay + jitter
                logger.info(f"Retrying in {wait_time:.1f} seconds... (backoff: {exponential_delay}s + jitter: {jitter:.1f}s)")
                await asyncio.sleep(wait_time)
            else:
                logger.error("Max retries reached, raising exception")
                raise


def extract_tool_calls(response: dict) -> List[dict]:
    """
    Extract tool calls from Claude API response.

    Args:
        response: Response from call_claude_with_tools()

    Returns:
        List of tool call dictionaries with structure:
        [
            {
                "id": "toolu_...",
                "name": "create_sketch",
                "input": {...}
            },
            ...
        ]
    """
    tool_calls = []

    for block in response["content"]:
        if block.get("type") == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "name": block["name"],
                "input": block["input"]
            })

    logger.debug(f"Extracted {len(tool_calls)} tool call(s): {[tc['name'] for tc in tool_calls]}")
    return tool_calls


def extract_text_content(response: dict) -> str:
    """
    Extract text content from Claude API response.

    Args:
        response: Response from call_claude_with_tools()

    Returns:
        Concatenated text from all text blocks
    """
    text_parts = []

    for block in response["content"]:
        if block.get("type") == "text":
            text_parts.append(block["text"])

    return "".join(text_parts)
