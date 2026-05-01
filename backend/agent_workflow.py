"""
Agentic workflow for orchestrating Claude tool calls with Fusion 360 execution.

This module handles both the sequential execution loop and the planning mode
workflow described in the implementation plan. It bridges the LLM, template
code generator, and WebSocket manager so operations run deterministically while
surfacing actionable feedback to the model and UI.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple
from uuid import uuid4

try:
    from .code_generator import CodeGenerationError, format_error_for_llm, translate_tool_call
    from .reasoning_context import (
        ReasoningEntry,
        extract_decisions,
        summarize_reasoning,
        assess_outcome,
        mask_old_observations,
    )
    from .prompt_structure import REASONING_CONTEXT_SECTION
    from .llm_client import (
        call_claude_with_tools,
        extract_text_content,
        extract_tool_calls,
        generate_plan,
        markdown_to_html,
        summarize_plan_for_user,
        html_to_plain_text,
    )
    from .websocket_manager import ConnectionManager
    from .prompt_router import route_request, get_routing_summary
    from .prompt_builder import build_prompt, estimate_token_savings, get_build_summary
    from .entity_store import EntityStore
    from .sketch_entity_store import SketchEntityStore
    from .session_logger import initialize_session, _extract_session_context
    from .ir import IRDocument, IRDocumentState, map_tool_call_to_ir, validate_ir_candidate
    from .ir.mapper import UnsupportedToolMappingError
    from .ir.types import (
        AddCircleParams,
        AddRectangleParams,
        CreateSketchParams,
        ExtrudeParams,
        IROperation,
    )
    from .backends.build123d import Build123dTargetExecutor
    from .backends.fusion import FusionTargetExecutor
    from .attachments import attachment_debug_summary, normalize_request_attachments
except ImportError:  # pragma: no cover - script execution fallback
    from code_generator import CodeGenerationError, format_error_for_llm, translate_tool_call  # type: ignore
    from reasoning_context import (  # type: ignore
        ReasoningEntry,
        extract_decisions,
        summarize_reasoning,
        assess_outcome,
        mask_old_observations,
    )
    from prompt_structure import REASONING_CONTEXT_SECTION  # type: ignore
    from llm_client import (  # type: ignore
        call_claude_with_tools,
        extract_text_content,
        extract_tool_calls,
        generate_plan,
        markdown_to_html,
        summarize_plan_for_user,
        html_to_plain_text,
    )
    from websocket_manager import ConnectionManager  # type: ignore
    from prompt_router import route_request, get_routing_summary  # type: ignore
    from prompt_builder import build_prompt, estimate_token_savings, get_build_summary  # type: ignore
    from entity_store import EntityStore  # type: ignore
    from sketch_entity_store import SketchEntityStore  # type: ignore
    from session_logger import initialize_session, _extract_session_context  # type: ignore
    from ir import IRDocument, IRDocumentState, map_tool_call_to_ir, validate_ir_candidate  # type: ignore
    from ir.mapper import UnsupportedToolMappingError  # type: ignore
    from ir.types import (  # type: ignore
        AddCircleParams,
        AddRectangleParams,
        CreateSketchParams,
        ExtrudeParams,
        IROperation,
    )
    from backends.build123d import Build123dTargetExecutor  # type: ignore
    from backends.fusion import FusionTargetExecutor  # type: ignore
    from attachments import attachment_debug_summary, normalize_request_attachments  # type: ignore

logger = logging.getLogger(__name__)
AUTH_BYPASS = (
    os.environ.get("CADAGENT_AUTH_BYPASS", os.environ.get("AUTH_BYPASS", "false"))
    .lower()
    in ("1", "true", "yes", "on")
)

DEFAULT_MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "30"))
EXECUTION_TIMEOUT = int(os.environ.get("EXECUTION_TIMEOUT", "30"))
PLAN_APPROVAL_TIMEOUT = int(os.environ.get("PLAN_APPROVAL_TIMEOUT", "120"))
EDGE_OPERATION_TOOLS = {"select_edges", "clear_edge_selection"}
FACE_SKETCH_SEQUENCING_BLOCK_TOOLS = {
    "add_circle",
    "add_line",
    "add_arc",
    "add_rectangle",
    "list_sketch_profiles",
    "extrude_profile",
    "extrude",
    "revolve_profile",
}
FACE_SKETCH_UV_BOUNDS_MARGIN_CM = 0.05  # 0.5mm tolerance

# Feature flag for intelligent prompt routing
USE_PROMPT_ROUTING = os.environ.get("USE_PROMPT_ROUTING", "true").lower() == "true"
if USE_PROMPT_ROUTING:
    logger.info("✓ Intelligent prompt routing ENABLED - tool clusters will be selected dynamically")
else:
    logger.info("✗ Intelligent prompt routing DISABLED - all tools will be loaded")
FACE_OPERATION_TOOLS = {"select_faces", "clear_face_selection"}
BODY_OPERATION_TOOLS = {"select_bodies", "clear_body_selection"}
def _extract_user_request_for_routing(message: Dict[str, Any]) -> Optional[str]:
    """
    Extract user request text from message for routing.

    Args:
        message: User message dictionary with 'content' field

    Returns:
        User request text, or None if not extractable
    """
    content = message.get("content", [])

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Extract text from content blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif "user_request" in block:
                    return block["user_request"]

        if text_parts:
            return " ".join(text_parts)

    # Try to extract from nested structure
    if isinstance(content, dict) and "user_request" in content:
        return content["user_request"]

    return None


FEATURE_OPERATION_TOOLS = {
    "apply_fillet",
    "apply_chamfer",
    "create_shell",
    "create_simple_hole",
    "create_counterbore_hole",
    "create_tapped_hole",
    "create_external_thread",
    "list_features",
    "create_pattern_feature",
    "adjust_feature_parameters",
}
GEOMETRY_OPERATION_TOOLS = EDGE_OPERATION_TOOLS | FACE_OPERATION_TOOLS | BODY_OPERATION_TOOLS | FEATURE_OPERATION_TOOLS

# Tools that create or modify geometry and should trigger entity context refresh.
# This ensures the LLM always has fresh entity refs (face_N, edge_N, body_N) after
# any operation that adds new geometry to the design.
# NOTE: create_construction_plane is NOT included - construction planes are infrastructure
# for sketches, not geometry entities. They don't modify bodies/faces/edges and are
# tracked separately by PlaneManager.
GEOMETRY_MODIFYING_TOOLS = {
    # Feature creation tools
    "extrude_profile",
    "revolve_profile",
    "create_loft",
    "apply_fillet",
    "apply_chamfer",
    "create_shell",
    "create_simple_hole",
    "create_counterbore_hole",
    "create_tapped_hole",
    "create_external_thread",
    "create_pattern_feature",
    "adjust_feature_parameters",
}

# Tools that modify the timeline and invalidate existing entity refs.
# These require clearing the EntityStore and regenerating fresh refs.
TIMELINE_MODIFYING_TOOLS = {
    "jump_to_timeline_position",
    "delete_feature",
}

# Tools that can invalidate topology refs and should run one-at-a-time per LLM turn.
TOPOLOGY_MUTATING_TOOLS = GEOMETRY_MODIFYING_TOOLS | TIMELINE_MODIFYING_TOOLS

OPERATION_CHECKPOINT_TOOLS = GEOMETRY_MODIFYING_TOOLS | TIMELINE_MODIFYING_TOOLS | {
    "create_sketch",
    "add_line",
    "add_arc",
    "add_circle",
    "add_rectangle",
    "draw_lines",
    "draw_rectangle",
    "draw_circle",
    "draw_arc",
    "draw_spline",
    "draw_polygon",
    "draw_slot",
    "draw_point",
    "draw_ellipse",
    "close_sketch",
    "create_construction_plane",
    "revolve_profile",
    "create_loft",
}

# Guardrail against repeated identical hole/thread operations in a single request loop.
DUPLICATE_INTENT_GUARD_TOOLS = {
    "create_simple_hole",
    "create_counterbore_hole",
    "create_tapped_hole",
    "create_external_thread",
}

INTENT_NOISE_FIELDS = {"description", "feature_name"}

# Tools that don't emit body/face/edge entities in the standard signature
# Reserved for future use. Currently empty.
# NOTE: create_construction_plane is not here - it's excluded from GEOMETRY_MODIFYING_TOOLS
# entirely since construction planes are infrastructure, not geometry entities.
TOOLS_WITHOUT_ENTITY_EMISSION = set()  # type: set[str]

# All tools that require entity context refresh with signature validation
# (geometry changes + timeline jumps)
REFRESH_ON_SUCCESS_TOOLS = GEOMETRY_MODIFYING_TOOLS | TIMELINE_MODIFYING_TOOLS

# Tools that require face/edge entity references to function correctly.
# These tools should NOT proceed if design_entities is empty - the model
# would be operating blind and likely to make spatial errors.
TOOLS_REQUIRING_FACE_REFS = {
    "create_simple_hole",
    "create_counterbore_hole",
    "create_tapped_hole",
    "create_external_thread",
    "select_faces",
    "apply_fillet",
    "apply_chamfer",
    "create_shell",
}

TOOLS_REQUIRING_EDGE_REFS = {
    "select_edges",
    "apply_fillet",
    "apply_chamfer",
}

TOOLS_REQUIRING_BODY_REFS = {
    "select_bodies",
    "create_shell",
}

# Combined set of tools requiring entity context
TOOLS_REQUIRING_ENTITY_CONTEXT = (
    TOOLS_REQUIRING_FACE_REFS |
    TOOLS_REQUIRING_EDGE_REFS |
    TOOLS_REQUIRING_BODY_REFS
)

# Tools that count as build plan steps when completed successfully
# These are operations that create or modify geometry - the "real" CAD work
BUILD_PLAN_STEP_TOOLS = GEOMETRY_MODIFYING_TOOLS | {
    "create_sketch",
    "draw_lines",
    "draw_rectangle",
    "draw_circle",
    "draw_arc",
    "add_arc",
    "draw_spline",
    "draw_polygon",
    "draw_slot",
    "draw_point",
    "draw_ellipse",
    "close_sketch",
    "create_construction_plane",
}

FEATURE_SNAPSHOT_LIMIT = 25
WORLD_AXIS_VECTORS: Dict[str, List[float]] = {
    "x": [1.0, 0.0, 0.0],
    "-x": [-1.0, 0.0, 0.0],
    "y": [0.0, 1.0, 0.0],
    "-y": [0.0, -1.0, 0.0],
    "z": [0.0, 0.0, 1.0],
    "-z": [0.0, 0.0, -1.0],
}

# =============================================================================
# Backward-compatible parameter name mapping for direct execution path
# =============================================================================
# Mirrors PARAM_ALIASES from code_generator.py. Maps new ref-based names
# to legacy token-based names so handlers can accept either form.

_PARAM_ALIASES: Dict[str, Dict[str, str]] = {
    # Tool name -> {new_name: legacy_name}
    # Selection/feature tools
    "apply_fillet": {"edge_refs": "entity_tokens"},
    "apply_chamfer": {"edge_refs": "entity_tokens"},
    "select_edges": {"edge_refs": "entity_tokens"},
    "select_faces": {"face_refs": "entity_tokens"},
    "select_bodies": {"body_refs": "entity_tokens"},
    # NOTE: create_shell intentionally omitted - entity_tokens is ambiguous
    # (could be faces OR bodies). Handler accepts entity_tokens directly.
    
    # Hole/thread tools
    "create_simple_hole": {"face_ref": "face_token"},
    "create_counterbore_hole": {"face_ref": "face_token"},
    "create_tapped_hole": {"face_ref": "face_token"},
    "create_external_thread": {"face_ref": "face_token"},
    
    # Construction plane
    "create_construction_plane": {
        "reference_face_ref": "reference_face_token",
        "reference_edge_ref": "reference_edge_token",
        "face_ref": "face_token",
        "datum_plane": "base_datum_plane",
        "reference_plane": "base_datum_plane",
        "offset": "offset_cm",
        "offset_distance": "offset_cm",
    },
    
    # Pattern feature
    "create_pattern_feature": {"feature_refs": "feature_tokens"},
}


def _normalize_tool_params(tool_name: str, tool_input: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalize parameter names to support both new and legacy field names.

    New names (edge_refs, face_refs, body_refs, face_ref) are preferred.
    Legacy names (entity_tokens, face_token) are accepted for backward compatibility.

    Returns a new dict with canonical field names (legacy form for handlers).
    """
    result = dict(tool_input)

    aliases = _PARAM_ALIASES.get(tool_name, {})
    for new_name, legacy_name in aliases.items():
        # If new name present, convert to legacy name for handler compatibility
        if new_name in result and legacy_name not in result:
            result[legacy_name] = result.pop(new_name)
            logger.debug(f"Normalized {tool_name}: {new_name} -> {legacy_name}")

    return result


def _canonicalize_intent_value(value: Any) -> Any:
    """Canonicalize JSON-ish values for stable duplicate-intent keys."""
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for key in sorted(value.keys()):
            key_str = str(key)
            if key_str in INTENT_NOISE_FIELDS:
                continue
            normalized[key_str] = _canonicalize_intent_value(value[key])
        return normalized

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize_intent_value(item) for item in value]

    if isinstance(value, float):
        # Trim float noise from repeated LLM retries.
        return round(value, 6)

    return value


def _build_tool_intent_key(tool_name: str, tool_input: Mapping[str, Any]) -> str:
    """Build a stable intent key used for retry tracking and duplicate suppression."""
    normalized_input = (
        _normalize_tool_params(tool_name, tool_input)
        if tool_name in _PARAM_ALIASES
        else dict(tool_input)
    )
    canonical_payload = _canonicalize_intent_value(normalized_input)
    payload_json = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{tool_name}:{payload_json}"


_NO_INTERSECTION_FAILURE_MARKERS = (
    "no target body found to cut or intersect",
    "no_intersection_after_direction_retry",
)


def _normalize_extrude_operation_name(operation_value: Any) -> str:
    """Normalize extrude operation variants for intent comparison."""
    if operation_value is None:
        return "newbody"
    operation = str(operation_value).strip().lower()
    if operation.endswith("featureoperation"):
        operation = operation[: -len("featureoperation")]
    return operation


def _is_no_intersection_failure_text(failure_detail: Any) -> bool:
    if not isinstance(failure_detail, str):
        return False
    detail = failure_detail.lower()
    return any(marker in detail for marker in _NO_INTERSECTION_FAILURE_MARKERS)


def _build_failure_intent_key(tool_name: str, tool_input: Mapping[str, Any], failure_detail: Any) -> str:
    """
    Build a failure-intent key for retry guards.

    For Cut/Intersect extrude no-intersection failures, treat direction sign flips
    as the same logical intent so repeated failed retries are detected reliably.
    """
    base_key = _build_tool_intent_key(tool_name, tool_input)
    if tool_name != "extrude_profile":
        return base_key
    if not _is_no_intersection_failure_text(failure_detail):
        return base_key

    normalized_input = dict(tool_input)
    operation = _normalize_extrude_operation_name(normalized_input.get("operation"))
    if operation not in {"cut", "intersect"}:
        return base_key

    canonical_payload = _canonicalize_intent_value(normalized_input)
    if not isinstance(canonical_payload, Mapping):
        return base_key

    retry_payload = dict(canonical_payload)
    retry_payload["operation"] = operation

    distance_value = retry_payload.get("distance")
    try:
        retry_payload["distance"] = abs(float(distance_value))
    except (TypeError, ValueError):
        pass

    retry_payload["failure_mode"] = "no_intersection_after_direction_retry"
    payload_json = json.dumps(retry_payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{tool_name}:retry:{payload_json}"


WORLD_AXIS_ORDER = ["x", "y", "z"]

IR_MVP_TOOLS = {"create_sketch", "add_rectangle", "add_circle", "extrude_profile", "extrude"}
BUILD123D_TARGET_NAMES = {"build123d", "studio"}
_BUILD123D_EXECUTOR = Build123dTargetExecutor()


class UnsupportedExecutionTargetError(ValueError):
    """Raised when execution_target is not a supported backend target."""


def _resolve_execution_target(request: Optional[Mapping[str, Any]]) -> str:
    """Resolve execution target while preserving Fusion as default for missing values."""
    raw_target = (request or {}).get("execution_target")
    if raw_target is None or (isinstance(raw_target, str) and not raw_target.strip()):
        return "fusion"

    target = str(raw_target).strip().lower()
    if target in BUILD123D_TARGET_NAMES:
        return "build123d"
    if target == "fusion":
        return "fusion"
    raise UnsupportedExecutionTargetError(
        f"Unsupported execution_target '{raw_target}'. Expected one of: fusion, build123d, studio."
    )


def _is_missing_provider_key_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "api key" in text and ("no " in text or "missing" in text)


def _extract_latest_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, Mapping):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    elif "user_request" in block:
                        parts.append(str(block.get("user_request", "")))
            if parts:
                return " ".join(parts)
    return ""


def _deterministic_mvp_tool_calls(user_request: str) -> Optional[List[Dict[str, Any]]]:
    """Fallback-only deterministic planner for MVP cube/cylinder coverage."""
    normalized = (user_request or "").lower()
    number_pattern = r"[-+]?\d+(?:\.\d+)?"

    cube_match = re.search(rf"({number_pattern})\s*mm?\s*(cube|box)", normalized)
    if cube_match:
        size = float(cube_match.group(1))
        half = size / 2.0
        return [
            {
                "id": "toolu_det_1",
                "name": "create_sketch",
                "input": {"plane_id": "XY", "sketch_id": "sketch_0", "description": "Create sketch on XY"},
            },
            {
                "id": "toolu_det_2",
                "name": "add_rectangle",
                "input": {
                    "sketch_id": "sketch_0",
                    "corner1_u": -half,
                    "corner1_v": -half,
                    "corner2_u": half,
                    "corner2_v": half,
                    "description": f"Add {size}mm square profile",
                },
            },
            {
                "id": "toolu_det_3",
                "name": "extrude_profile",
                "input": {
                    "sketch_id": "sketch_0",
                    "profile_index": 0,
                    "distance": size,
                    "operation": "NewBody",
                    "description": f"Extrude {size}mm for cube",
                },
            },
        ]

    radius_match = re.search(rf"radius\s*({number_pattern})", normalized)
    height_match = re.search(rf"height\s*({number_pattern})", normalized)
    if "cylinder" in normalized and radius_match and height_match:
        radius = float(radius_match.group(1))
        height = float(height_match.group(1))
        return [
            {
                "id": "toolu_det_1",
                "name": "create_sketch",
                "input": {"plane_id": "XY", "sketch_id": "sketch_0", "description": "Create sketch on XY"},
            },
            {
                "id": "toolu_det_2",
                "name": "add_circle",
                "input": {
                    "sketch_id": "sketch_0",
                    "center_u": 0.0,
                    "center_v": 0.0,
                    "radius": radius,
                    "description": f"Add circle radius {radius}",
                },
            },
            {
                "id": "toolu_det_3",
                "name": "extrude_profile",
                "input": {
                    "sketch_id": "sketch_0",
                    "profile_index": 0,
                    "distance": height,
                    "operation": "NewBody",
                    "description": f"Extrude {height}mm for cylinder",
                },
            },
        ]

    return None

# Global counter for generating unique message IDs
_message_id_counter = 0


def _generate_message_id() -> str:
    """Generate a unique message ID for checkpoint tracking."""
    global _message_id_counter
    _message_id_counter += 1
    timestamp = int(time.time() * 1000)  # milliseconds
    return f"msg_{timestamp}_{_message_id_counter}"


async def _handle_build_plan_step(
    session_id: str,
    manager: ConnectionManager,
    tool_name: str,
    success: bool
) -> Optional[str]:
    """
    Handle build plan progress tracking for a completed tool operation.

    If there's an active build plan and the operation was successful,
    this increments the step counter, sends a UI update, and returns
    the plan context to append to the tool result.

    Args:
        session_id: Session identifier
        manager: WebSocket connection manager
        tool_name: Name of the tool that just executed
        success: Whether the tool execution was successful

    Returns:
        Plan context string to append to tool result, or None if no active plan
    """
    # Only track progress for build-plan-relevant tools that succeeded
    if not success or tool_name not in BUILD_PLAN_STEP_TOOLS:
        return None

    plan = manager.get_active_build_plan(session_id)
    if not plan:
        return None

    # Increment step counter
    completed = manager.increment_build_plan_step(session_id)
    if completed is None:
        return None

    total = len(plan.get("steps", []))
    design_name = plan.get("design_name", "Design")

    # Send UI update for step completion
    step_update = {
        "type": "build_step_completed",
        "data": {
            "completed_steps": completed,
            "total_steps": total,
            "design_name": design_name
        }
    }
    try:
        await manager.send_json(session_id, step_update)
        logger.debug("Session %s sent build step update: %d/%d", session_id, completed, total)
    except Exception as e:
        logger.warning("Failed to send build step update: %s", e)

    # Check if build plan is complete
    if completed >= total:
        # Send completion message to UI
        completion_msg = {
            "type": "build_plan_completed",
            "data": {
                "design_name": design_name,
                "total_steps": total
            }
        }
        try:
            await manager.send_json(session_id, completion_msg)
            logger.info("Session %s build plan completed: %s (%d steps)", session_id, design_name, total)
        except Exception as e:
            logger.warning("Failed to send build plan completion: %s", e)

        # Clear the active plan
        manager.set_active_build_plan(session_id, None)

        return f"\n\n[BUILD PLAN COMPLETE: All {total} steps finished for '{design_name}']"

    # Return plan context for injection into tool result
    return manager.format_build_plan_context(session_id)


def _get_entity_store(session_id: str, manager: ConnectionManager) -> EntityStore:
    """Convenience accessor for the per-session entity reference store."""
    return manager.get_entity_store(session_id)


def _get_sketch_entity_store(session_id: str, manager: ConnectionManager) -> SketchEntityStore:
    """Convenience accessor for per-session sketch entity/constraint refs."""
    getter = getattr(manager, "get_sketch_entity_store", None)
    if callable(getter):
        return getter(session_id)

    # Lightweight fallback for tests/stubs that do not implement the manager API.
    stores = getattr(manager, "_fallback_sketch_entity_stores", None)
    if not isinstance(stores, dict):
        stores = {}
        setattr(manager, "_fallback_sketch_entity_stores", stores)
    if session_id not in stores or not isinstance(stores[session_id], SketchEntityStore):
        stores[session_id] = SketchEntityStore()
    return stores[session_id]


def _extract_entity_token(entity: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Extract canonical entity token from either token or entity_token fields."""
    if not entity or not isinstance(entity, Mapping):
        return None
    token = entity.get("entity_token") or entity.get("token")
    if not isinstance(token, str):
        return None
    stripped = token.strip()
    return stripped or None


def _collect_context_entities(entity_context: Mapping[str, Any]) -> Dict[str, List[Mapping[str, Any]]]:
    """
    Collect entities from context in a uniform shape.

    Prefers nested spatial_context when it actually contains bodies, otherwise
    falls back to flat top-level entities.
    """

    def _as_mapping_list(value: Any) -> List[Mapping[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, Mapping)]

    spatial_context = entity_context.get("spatial_context")
    spatial_bodies: List[Mapping[str, Any]] = []
    if isinstance(spatial_context, dict):
        spatial_bodies = _as_mapping_list(spatial_context.get("bodies", []))

    if spatial_bodies:
        bodies: List[Mapping[str, Any]] = []
        faces: List[Mapping[str, Any]] = []
        edges: List[Mapping[str, Any]] = []
        vertices: List[Mapping[str, Any]] = []
        for body in spatial_bodies:
            bodies.append(body)
            faces.extend(_as_mapping_list(body.get("faces", [])))
            edges.extend(_as_mapping_list(body.get("edges", [])))
            vertices.extend(_as_mapping_list(body.get("vertices", [])))
        return {
            "bodies": bodies,
            "faces": faces,
            "edges": edges,
            "vertices": vertices,
        }

    return {
        "bodies": _as_mapping_list(entity_context.get("bodies", [])),
        "faces": _as_mapping_list(entity_context.get("faces", [])),
        "edges": _as_mapping_list(entity_context.get("edges", [])),
        "vertices": _as_mapping_list(entity_context.get("vertices", [])),
    }


def _normalise_entities_for_registration(entities: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert mixed nested/flat entities into register_entities-compatible payloads.

    register_entities requires `entity_token`. Nested payloads often provide `token`.
    """
    normalised: List[Dict[str, Any]] = []
    for entity in entities:
        token = _extract_entity_token(entity)
        if not token:
            continue
        if isinstance(entity, dict) and isinstance(entity.get("entity_token"), str) and entity.get("entity_token").strip():
            normalised.append(entity)
            continue
        enriched = dict(entity)
        enriched["entity_token"] = token
        normalised.append(enriched)
    return normalised


def _extract_tokens_from_context(entity_context: Mapping[str, Any]) -> List[str]:
    """Extract all entity tokens from an entity context dict."""
    tokens: List[str] = []
    seen_tokens: Set[str] = set()
    entities = _collect_context_entities(entity_context)

    for kind in ("bodies", "faces", "edges", "vertices"):
        for entity in entities[kind]:
            token = _extract_entity_token(entity)
            if not token or token in seen_tokens:
                continue
            seen_tokens.add(token)
            tokens.append(token)

    return tokens


async def _prepopulate_entity_store(
    session_id: str,
    manager: ConnectionManager,
    entity_context: Mapping[str, Any]
) -> None:
    """
    Pre-populate the entity store with entities from the frontend-provided context.

    This allows the LLM to use entity refs immediately without needing to call
    list_* tools first.

    New ID scheme (SPATIAL_CONTEXT_OVERHAUL.md):
    - Bodies: body_0, body_1, ...
    - Faces: face_0, face_1, face_2, ... (sequential)
    - Edges: e0, e1, e2, ...
    - Vertices: v0, v1, v2, ...

    The function also builds cross-references for the new spatial context structure:
    - edge→face adjacency
    - edge→vertex connections
    - face→edge via loops

    Note: Caller should clear the store before calling this function to ensure
    stale refs from previous requests are removed.

    Args:
        session_id: Session identifier
        manager: Connection manager for entity store access
        entity_context: Dictionary containing bodies, faces, edges, vertices arrays
    """
    store = _get_entity_store(session_id, manager)

    # Support both flat structure and new nested per-body structure.
    # Only treat as nested when bodies are present; otherwise fall back to flat.
    spatial_context = entity_context.get("spatial_context")
    bodies_data: List[Mapping[str, Any]] = []
    if isinstance(spatial_context, dict):
        maybe_bodies = spatial_context.get("bodies", [])
        if isinstance(maybe_bodies, list):
            bodies_data = maybe_bodies

    if bodies_data:
        # Process nested per-body structure
        await _prepopulate_from_spatial_context(store, bodies_data, entity_context)
        return

    # Fall back to flat structure (backward compatibility)
    bodies = entity_context.get("bodies", [])
    faces = entity_context.get("faces", [])
    edges = entity_context.get("edges", [])
    vertices = entity_context.get("vertices", [])

    # Register entities and attach ref IDs back to the context for formatting.
    # We iterate by index to preserve alignment when some entities are skipped
    # (e.g., missing entity_token). Using zip would misalign refs when register_entities
    # skips invalid items.

    async def _register_and_attach_refs(kind: str, entities: List[Mapping[str, Any]]) -> int:
        """Register entities and attach refs. Returns count of successfully registered."""
        if not entities:
            return 0

        # Build a map from token -> EntityRef for successfully registered items
        registered = await store.register_entities(kind, entities)
        token_to_ref = {entry.token: entry.ref_id for entry in registered}
        
        # Attach ref IDs back to original entities by matching their token
        for entity in entities:
            token = entity.get("entity_token")
            if token and token in token_to_ref:
                entity["entity_ref"] = token_to_ref[token]
        
        return len(registered)

    body_count = await _register_and_attach_refs("body", bodies)
    face_count = await _register_and_attach_refs("face", faces)
    edge_count = await _register_and_attach_refs("edge", edges)
    vertex_count = await _register_and_attach_refs("vertex", vertices)

    # Entity store cache stats are logged by EntityStore itself


async def _prepopulate_from_spatial_context(
    store: "EntityStore",
    bodies_data: List[Mapping[str, Any]],
    entity_context: Mapping[str, Any]
) -> None:
    """
    Populate entity store from the new nested per-body spatial_context structure.
    
    Args:
        store: EntityStore instance to populate
        bodies_data: List of body dicts from spatial_context.bodies
        entity_context: Original entity_context for attaching refs
    """
    all_bodies: List[Mapping[str, Any]] = []
    all_faces: List[Mapping[str, Any]] = []
    all_edges: List[Mapping[str, Any]] = []
    all_vertices: List[Mapping[str, Any]] = []
    
    def _get_token(obj: Optional[Mapping[str, Any]]) -> Optional[str]:
        if not obj or not isinstance(obj, Mapping):
            return None
        return obj.get("token") or obj.get("entity_token")

    for body in bodies_data:
        # Extract body-level entity
        body_entity = {
            "entity_token": _get_token(body),
            "id": body.get("id"),
            "name": body.get("name"),
            "bounding_box": body.get("bbox"),
            "body_index": body.get("body_index"),
            "occurrence_path": body.get("occurrence_path"),
        }
        all_bodies.append(body_entity)
        
        # Extract vertices
        for vertex in body.get("vertices", []):
            vertex_entity = {
                "entity_token": _get_token(vertex),
                "id": vertex.get("id"),
                "p": vertex.get("p"),
                "position": vertex.get("p"),
                "body_name": body.get("name"),
                "body_index": body.get("body_index"),
                "occurrence_path": body.get("occurrence_path"),
            }
            all_vertices.append(vertex_entity)
        
        # Extract faces
        for face in body.get("faces", []):
            face_entity = {
                "entity_token": _get_token(face),
                "id": face.get("id"),
                "surface_type": face.get("surface_type"),
                "centroid": face.get("centroid"),
                "normal": face.get("normal"),
                "frame": face.get("frame"),
                "loops": face.get("loops"),
                "body_name": body.get("name"),
                "body_index": body.get("body_index"),
                "occurrence_path": body.get("occurrence_path"),
            }
            all_faces.append(face_entity)
        
        # Extract edges
        for edge in body.get("edges", []):
            edge_entity = {
                "entity_token": _get_token(edge),
                "id": edge.get("id"),
                "edge_type": edge.get("edge_type"),
                "v0": edge.get("v0"),
                "v1": edge.get("v1"),
                "p0": edge.get("p0"),
                "p1": edge.get("p1"),
                "midpoint": edge.get("midpoint"),
                "tangent": edge.get("tangent"),
                "length": edge.get("length"),
                "adjacent_faces": edge.get("adjacent_faces", []),
                "body_name": body.get("name"),
                "body_index": body.get("body_index"),
                "occurrence_path": body.get("occurrence_path"),
            }
            all_edges.append(edge_entity)
    
    # Register in order: bodies, vertices, faces, edges
    await store.register_entities("body", all_bodies)
    await store.register_entities("vertex", all_vertices)
    await store.register_entities("face", all_faces)
    await store.register_entities("edge", all_edges)
    
    # Map token references to ref IDs in entity cross-references
    _map_tokens_to_refs(bodies_data, store)


def _map_tokens_to_refs(
    bodies_data: List[Mapping[str, Any]],
    store: "EntityStore"
) -> None:
    """
    Post-process entities to replace raw tokens with ref IDs.
    
    After registration, entities still contain raw tokens in cross-references:
    - edge.adjacent_faces: list of face tokens → list of face ref IDs
    - edge.v0/v1: vertex tokens → vertex ref ID strings
    - face.loops.outer/inner[].edge: edge tokens → edge ref IDs
    
    This function maps those tokens to their registered ref IDs in-place.
    """
    def _get_token(obj: Optional[Mapping[str, Any]]) -> Optional[str]:
        if not obj or not isinstance(obj, Mapping):
            return None
        return obj.get("token") or obj.get("entity_token")

    # Access the token→ref mapping from the store
    token_to_ref = store._token_to_ref
    
    for body in bodies_data:
        body_token = _get_token(body)
        if body_token and body_token in token_to_ref:
            body["entity_ref"] = token_to_ref[body_token]

        # Map vertex refs
        for vertex in body.get("vertices", []):
            v_token = _get_token(vertex)
            if v_token and v_token in token_to_ref:
                vertex["entity_ref"] = token_to_ref[v_token]

        # Map face refs
        for face in body.get("faces", []):
            f_token = _get_token(face)
            if f_token and f_token in token_to_ref:
                face["entity_ref"] = token_to_ref[f_token]

        # Map edge refs
        for edge in body.get("edges", []):
            e_token = _get_token(edge)
            if e_token and e_token in token_to_ref:
                edge["entity_ref"] = token_to_ref[e_token]

        # Map edge cross-references
        for edge in body.get("edges", []):
            # Map adjacent_faces tokens to face ref IDs
            adj_faces = edge.get("adjacent_faces", [])
            if adj_faces:
                edge["adjacent_faces"] = [
                    token_to_ref.get(token, token) if isinstance(token, str) else token
                    for token in adj_faces
                ]
            
            # Map v0/v1 tokens to vertex ref IDs
            v0 = edge.get("v0")
            v0_token = None
            if isinstance(v0, str):
                v0_token = v0
            elif isinstance(v0, dict):
                v0_token = v0.get("token") or v0.get("entity_token")
            if not v0_token:
                v0_token = edge.get("v0_token")
            if v0_token and v0_token in token_to_ref:
                edge["v0_ref"] = token_to_ref[v0_token]
                edge["v0"] = token_to_ref[v0_token]
            
            v1 = edge.get("v1")
            v1_token = None
            if isinstance(v1, str):
                v1_token = v1
            elif isinstance(v1, dict):
                v1_token = v1.get("token") or v1.get("entity_token")
            if not v1_token:
                v1_token = edge.get("v1_token")
            if v1_token and v1_token in token_to_ref:
                edge["v1_ref"] = token_to_ref[v1_token]
                edge["v1"] = token_to_ref[v1_token]
        
        # Map face loop edge references
        for face in body.get("faces", []):
            loops = face.get("loops")
            if not loops:
                continue
            
            # Process outer loop
            outer = loops.get("outer", [])
            for loop_entry in outer:
                if isinstance(loop_entry, dict):
                    edge_token = loop_entry.get("edge") or loop_entry.get("edge_token")
                    if edge_token and edge_token in token_to_ref:
                        loop_entry["edge"] = token_to_ref[edge_token]
            
            # Process inner loops
            for inner_loop in loops.get("inner", []):
                for loop_entry in inner_loop:
                    if isinstance(loop_entry, dict):
                        edge_token = loop_entry.get("edge") or loop_entry.get("edge_token")
                        if edge_token and edge_token in token_to_ref:
                            loop_entry["edge"] = token_to_ref[edge_token]


def _capture_checkpoint(
    session_id: str,
    request: Mapping[str, Any],
    manager: ConnectionManager,
    conversation_index: int
) -> str:
    """
    Capture a checkpoint of the current timeline state before executing a user request.

    Args:
        session_id: Session identifier
        request: Execution request containing timeline_state
        manager: Connection manager for storing the checkpoint
        conversation_index: Current position in conversation history

    Returns:
        Generated message_id for this checkpoint
    """
    timeline_state = request.get("timeline_state")
    user_request = request.get("user_request", "")

    # Extract timeline marker position (this is the key state we need to restore)
    marker_position = 0
    timeline_count = 0

    if timeline_state and isinstance(timeline_state, dict):
        marker_position = timeline_state.get("marker_position", 0)
        timeline_count = timeline_state.get("count", 0)

    message_id = _generate_message_id()

    checkpoint_data = {
        "message_id": message_id,
        "marker_position": marker_position,
        "timeline_count": timeline_count,
        "message_text": str(user_request)[:200],  # Truncate for storage efficiency
        "timestamp": time.time(),
        "conversation_index": conversation_index,
    }
    checkpoint_data.update(_message_checkpoint_runtime_state(session_id, manager))

    manager.save_checkpoint(session_id, checkpoint_data)
    return message_id


def _operation_display_label(tool_name: str, description: str = "") -> str:
    """Create a compact label for a completed operation checkpoint."""
    clean_description = " ".join(str(description or "").split())
    if clean_description:
        return clean_description[:120]
    return tool_name.replace("_", " ").strip().title()


def _checkpoint_public_metadata(checkpoint_data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return UI-safe checkpoint metadata without conversation/entity payloads."""
    public_keys = {
        "checkpoint_id",
        "request_id",
        "tool_name",
        "tool_use_id",
        "display_label",
        "description",
        "marker_position",
        "timeline_count",
        "conversation_index",
        "timestamp",
    }
    return {key: checkpoint_data.get(key) for key in public_keys if key in checkpoint_data}


def _operation_checkpoint_conversation_snapshot(
    messages: Sequence[Mapping[str, Any]],
    tool_use_id: str,
) -> List[Dict[str, Any]]:
    """
    Return a provider-valid conversation prefix for an operation checkpoint.

    Claude can return multiple tool_use blocks in one assistant message. The
    backend executes those tools one at a time, so a checkpoint after the first
    tool must not preserve later sibling tool_use blocks that do not yet have
    matching tool_result blocks. Anthropic rejects that shape on resume.
    """
    snapshot = json.loads(json.dumps(list(messages), default=str))
    target_tool_use_id = str(tool_use_id or "").strip()
    if not target_tool_use_id:
        return snapshot

    assistant_index: Optional[int] = None
    target_block_index: Optional[int] = None
    allowed_tool_use_ids: List[str] = []

    for message_index in range(len(snapshot) - 1, -1, -1):
        message = snapshot[message_index]
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        sibling_tool_ids: List[str] = []
        for block_index, block in enumerate(content):
            if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                continue
            sibling_tool_id = str(block.get("id") or "").strip()
            if not sibling_tool_id:
                continue
            sibling_tool_ids.append(sibling_tool_id)
            if sibling_tool_id == target_tool_use_id:
                assistant_index = message_index
                target_block_index = block_index
                allowed_tool_use_ids = list(sibling_tool_ids)
                break
        if assistant_index is not None:
            break

    if assistant_index is None or target_block_index is None or not allowed_tool_use_ids:
        logger.warning(
            "Unable to locate tool_use_id %s while snapshotting operation checkpoint; using full history",
            target_tool_use_id,
        )
        return snapshot

    allowed_tool_use_id_set = set(allowed_tool_use_ids)
    assistant_message = dict(snapshot[assistant_index])
    assistant_content = assistant_message.get("content")
    if isinstance(assistant_content, list):
        assistant_message["content"] = assistant_content[: target_block_index + 1]

    trimmed_snapshot: List[Dict[str, Any]] = [
        dict(message) if isinstance(message, Mapping) else message
        for message in snapshot[:assistant_index]
    ]
    trimmed_snapshot.append(assistant_message)

    target_result_seen = False
    for message in snapshot[assistant_index + 1 :]:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            break

        filtered_content: List[Any] = []
        allowed_result_seen_in_message = False
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "tool_result":
                result_tool_use_id = str(block.get("tool_use_id") or "").strip()
                if result_tool_use_id not in allowed_tool_use_id_set:
                    continue
                filtered_content.append(block)
                allowed_result_seen_in_message = True
                if result_tool_use_id == target_tool_use_id:
                    target_result_seen = True
            elif allowed_result_seen_in_message:
                filtered_content.append(block)

        if filtered_content:
            filtered_message = dict(message)
            filtered_message["content"] = filtered_content
            trimmed_snapshot.append(filtered_message)

        if target_result_seen:
            break

    if not target_result_seen:
        logger.warning(
            "Unable to locate tool_result for tool_use_id %s while snapshotting operation checkpoint; using full history",
            target_tool_use_id,
        )
        return snapshot

    return trimmed_snapshot


def _serialize_ir_operation(operation: IROperation) -> Dict[str, Any]:
    """Serialize one committed IR operation into checkpoint-safe plain data."""
    return {
        "id": operation.id,
        "type": operation.type,
        "params": asdict(operation.params),
        "dependencies": list(operation.dependencies or []),
        "metadata": dict(operation.metadata) if operation.metadata else None,
    }


def _serialize_ir_document_state(ir_doc_state: IRDocumentState) -> Dict[str, Any]:
    """Serialize committed IR document state for cross-request/resume continuity."""
    return {
        "version": ir_doc_state.version,
        "units": ir_doc_state.units,
        "counter": int(getattr(ir_doc_state, "_counter", 0) or 0),
        "operations": [_serialize_ir_operation(operation) for operation in ir_doc_state.operations],
        "metadata": dict(ir_doc_state.metadata) if ir_doc_state.metadata else None,
    }


def _deserialize_ir_operation(payload: Mapping[str, Any]) -> IROperation:
    """Deserialize a checkpointed IR operation."""
    operation_type = str(payload.get("type") or "").strip()
    raw_params = payload.get("params")
    params_payload = dict(raw_params) if isinstance(raw_params, Mapping) else {}

    if operation_type == "create_sketch":
        params = CreateSketchParams(
            plane=str(params_payload.get("plane") or "XY").strip() or "XY",
            sketch=str(params_payload.get("sketch") or "").strip(),
        )
    elif operation_type == "add_rectangle":
        params = AddRectangleParams(
            sketch=str(params_payload.get("sketch") or "").strip(),
            center=[float(value) for value in (params_payload.get("center") or [0.0, 0.0])[:2]],
            width=float(params_payload.get("width") or 0.0),
            height=float(params_payload.get("height") or 0.0),
        )
    elif operation_type == "add_circle":
        params = AddCircleParams(
            sketch=str(params_payload.get("sketch") or "").strip(),
            center=[float(value) for value in (params_payload.get("center") or [0.0, 0.0])[:2]],
            radius=float(params_payload.get("radius") or 0.0),
        )
    elif operation_type == "extrude":
        raw_profile_indices = params_payload.get("profile_indices")
        profile_indices = None
        if isinstance(raw_profile_indices, list):
            profile_indices = [int(value) for value in raw_profile_indices]
        raw_profile_index = params_payload.get("profile_index")
        profile_index = int(raw_profile_index) if raw_profile_index is not None else None
        raw_sketch = params_payload.get("sketch")
        params = ExtrudeParams(
            profile=str(params_payload.get("profile") or "").strip(),
            distance=float(params_payload.get("distance") or 0.0),
            direction=str(params_payload.get("direction") or "positive").strip() or "positive",  # type: ignore[arg-type]
            operation=str(params_payload.get("operation") or "new").strip() or "new",  # type: ignore[arg-type]
            sketch=str(raw_sketch).strip() if raw_sketch is not None else None,
            profile_index=profile_index,
            profile_indices=profile_indices,
        )
    else:
        raise ValueError(f"Unsupported serialized IR operation type: {operation_type}")

    raw_dependencies = payload.get("dependencies")
    dependencies = [str(dep) for dep in raw_dependencies] if isinstance(raw_dependencies, list) else []
    raw_metadata = payload.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else None

    return IROperation(
        id=str(payload.get("id") or "").strip(),
        type=operation_type,  # type: ignore[arg-type]
        params=params,
        dependencies=dependencies,
        metadata=metadata,  # type: ignore[arg-type]
    )


def _counter_floor_from_ir_operations(operations: Sequence[IROperation]) -> int:
    """Infer a safe minimum operation counter from serialized operation ids."""
    max_counter = len(operations)
    for operation in operations:
        op_id = str(operation.id or "")
        match = re.match(r"^op_(\d+)$", op_id)
        if match:
            max_counter = max(max_counter, int(match.group(1)))
    return max_counter


def _deserialize_ir_document_state(ir_state: Mapping[str, Any]) -> IRDocumentState:
    """Deserialize committed IR document state from manager/checkpoint storage."""
    raw_metadata = ir_state.get("metadata")
    state = IRDocumentState(
        version=str(ir_state.get("version") or "1.0"),
        units=str(ir_state.get("units") or "mm"),
        metadata=dict(raw_metadata) if isinstance(raw_metadata, Mapping) else None,
    )

    raw_operations = ir_state.get("operations")
    if isinstance(raw_operations, list):
        for operation_payload in raw_operations:
            if not isinstance(operation_payload, Mapping):
                continue
            state.append(_deserialize_ir_operation(operation_payload))

    try:
        stored_counter = int(ir_state.get("counter") or 0)
    except (TypeError, ValueError):
        stored_counter = 0
    state._counter = max(stored_counter, _counter_floor_from_ir_operations(state.operations))
    return state


def _manager_get_ir_document_state(manager: ConnectionManager, session_id: str) -> Optional[Dict[str, Any]]:
    """Best-effort accessor for serialized IR state."""
    getter = getattr(manager, "get_ir_document_state", None)
    if not callable(getter):
        return None
    ir_state = getter(session_id)
    return ir_state if isinstance(ir_state, dict) and ir_state else None


def _manager_save_ir_document_state(
    manager: ConnectionManager,
    session_id: str,
    ir_doc_state: IRDocumentState,
) -> None:
    """Best-effort persistence for serialized IR state."""
    saver = getattr(manager, "save_ir_document_state", None)
    if callable(saver):
        saver(session_id, _serialize_ir_document_state(ir_doc_state))


def _manager_clear_ir_document_state(manager: ConnectionManager, session_id: str) -> None:
    """Best-effort clearing for serialized IR state."""
    clearer = getattr(manager, "clear_ir_document_state", None)
    if callable(clearer):
        clearer(session_id)


def _restore_or_create_ir_document_state(
    manager: ConnectionManager,
    session_id: str,
    *,
    execution_target: str,
) -> IRDocumentState:
    """Initialize IR state from persisted session data when it matches the target."""
    expected_source = "studio" if execution_target == "build123d" else "fusion"
    default_metadata = {"source": expected_source, "session_id": session_id}
    stored_state = _manager_get_ir_document_state(manager, session_id)
    if stored_state:
        try:
            restored = _deserialize_ir_document_state(stored_state)
            restored_metadata = dict(restored.metadata) if restored.metadata else {}
            stored_source = str(restored_metadata.get("source") or "").strip()
            if stored_source and stored_source != expected_source:
                logger.info(
                    "Ignoring IR state for session %s because source %s does not match target %s",
                    session_id,
                    stored_source,
                    expected_source,
                )
            else:
                restored_metadata.update(default_metadata)
                restored.metadata = restored_metadata
                logger.debug(
                    "Restored IR state for session %s with %d committed operation(s)",
                    session_id,
                    len(restored.operations),
                )
                return restored
        except Exception as exc:
            logger.warning("Unable to restore IR state for session %s: %s", session_id, exc)
            _manager_clear_ir_document_state(manager, session_id)

    return IRDocumentState(metadata=default_metadata)


def _message_checkpoint_runtime_state(
    session_id: str,
    manager: ConnectionManager,
) -> Dict[str, Any]:
    """Capture backend runtime state that must match a message-level timeline checkpoint."""
    state: Dict[str, Any] = {}

    ir_state = _manager_get_ir_document_state(manager, session_id)
    if ir_state:
        state["ir_state"] = ir_state

    feature_snapshot = _manager_get_feature_snapshot(manager, session_id)
    if feature_snapshot:
        state["feature_snapshot"] = feature_snapshot

    latest_entity_context = _manager_get_latest_entity_context(manager, session_id)
    if latest_entity_context:
        state["latest_entity_context"] = latest_entity_context

    try:
        reasoning_context = manager.get_reasoning_context(session_id)
        reasoning_summary = reasoning_context.get_injection_text()
        if reasoning_summary:
            state["reasoning_summary"] = reasoning_summary
    except Exception:
        pass

    return state


def _latest_operation_checkpoint_at_or_before(
    session_id: str,
    manager: ConnectionManager,
    conversation_index: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Find the newest operation checkpoint at or before a message checkpoint boundary."""
    getter = getattr(manager, "get_operation_checkpoints", None)
    if not callable(getter):
        return None

    try:
        max_index = int(conversation_index) if conversation_index is not None else 0
    except (TypeError, ValueError):
        max_index = 0

    best_checkpoint: Optional[Dict[str, Any]] = None
    best_sort_key: Tuple[int, float] = (-1, -1.0)
    for checkpoint in getter(session_id):
        if not isinstance(checkpoint, Mapping):
            continue
        try:
            checkpoint_index = int(checkpoint.get("conversation_index", 0) or 0)
        except (TypeError, ValueError):
            checkpoint_index = 0
        if checkpoint_index > max_index:
            continue
        try:
            checkpoint_time = float(checkpoint.get("timestamp", 0.0) or 0.0)
        except (TypeError, ValueError):
            checkpoint_time = 0.0
        sort_key = (checkpoint_index, checkpoint_time)
        if sort_key >= best_sort_key:
            best_checkpoint = dict(checkpoint)
            best_sort_key = sort_key

    return best_checkpoint


def _manager_set_feature_snapshot(manager: ConnectionManager, session_id: str, snapshot: Mapping[str, Any]) -> None:
    setter = getattr(manager, "set_feature_snapshot", None)
    if callable(setter):
        setter(session_id, dict(snapshot))


def _manager_clear_feature_snapshot(manager: ConnectionManager, session_id: str) -> None:
    clearer = getattr(manager, "clear_feature_snapshot", None)
    if callable(clearer):
        clearer(session_id)


async def _restore_message_checkpoint_runtime_state(
    session_id: str,
    manager: ConnectionManager,
    checkpoint: Mapping[str, Any],
    conversation_index: Optional[int],
) -> None:
    """
    Restore backend runtime state for a message-level revert.

    New message checkpoints carry their own runtime state. For checkpoints
    created before that field existed, fall back to the latest operation
    checkpoint at or before the same conversation boundary.
    """
    fallback_checkpoint = _latest_operation_checkpoint_at_or_before(session_id, manager, conversation_index)

    def _state_value(key: str) -> Any:
        value = checkpoint.get(key)
        if value:
            return value
        if fallback_checkpoint:
            return fallback_checkpoint.get(key)
        return None

    ir_state = _state_value("ir_state")
    if isinstance(ir_state, Mapping) and ir_state:
        try:
            restored_ir_state = _deserialize_ir_document_state(ir_state)
            _manager_save_ir_document_state(manager, session_id, restored_ir_state)
        except Exception as exc:
            logger.warning("Unable to restore IR state from message checkpoint for session %s: %s", session_id, exc)
            _manager_clear_ir_document_state(manager, session_id)
    else:
        _manager_clear_ir_document_state(manager, session_id)

    feature_snapshot = _state_value("feature_snapshot")
    if isinstance(feature_snapshot, Mapping) and feature_snapshot:
        _manager_set_feature_snapshot(manager, session_id, feature_snapshot)
    else:
        _manager_clear_feature_snapshot(manager, session_id)

    latest_entity_context = _state_value("latest_entity_context")
    if isinstance(latest_entity_context, Mapping) and latest_entity_context:
        store = _get_entity_store(session_id, manager)
        store.clear()
        try:
            await _prepopulate_entity_store(session_id, manager, latest_entity_context)
        except Exception as exc:
            logger.warning("Unable to repopulate entity store from message checkpoint for session %s: %s", session_id, exc)
        _manager_set_latest_entity_context(manager, session_id, latest_entity_context)
    else:
        clearer = getattr(manager, "clear_entity_store", None)
        if callable(clearer):
            clearer(session_id)
        else:
            store = _get_entity_store(session_id, manager)
            store.clear()
            _manager_clear_latest_entity_context(manager, session_id)

    reasoning_context = manager.get_reasoning_context(session_id)
    reasoning_context.clear()
    reasoning_summary = _state_value("reasoning_summary")
    if isinstance(reasoning_summary, str) and reasoning_summary.strip():
        reasoning_context.compacted_summary = reasoning_summary.strip()[:4000]


async def _capture_operation_checkpoint(
    session_id: str,
    manager: ConnectionManager,
    *,
    request: Optional[Mapping[str, Any]],
    tool_name: str,
    tool_use_id: str,
    description: str,
    messages: List[Dict[str, Any]],
    ir_doc_state: Optional[IRDocumentState] = None,
    force_snapshot_refresh: bool = False,
) -> Optional[Dict[str, Any]]:
    """Persist and announce a resumable checkpoint after a successful operation."""
    if tool_name not in OPERATION_CHECKPOINT_TOOLS:
        return None
    if not callable(getattr(manager, "save_operation_checkpoint", None)):
        return None

    feature_snapshot = _manager_get_feature_snapshot(manager, session_id)
    if force_snapshot_refresh or not (feature_snapshot and feature_snapshot.get("success", True)):
        try:
            refreshed = await _request_feature_snapshot(
                session_id,
                manager,
                reason=f"operation_checkpoint_{tool_name}",
                max_features=FEATURE_SNAPSHOT_LIMIT,
            )
            if refreshed:
                feature_snapshot = refreshed
        except Exception as exc:
            logger.warning("Unable to refresh feature snapshot for operation checkpoint %s: %s", tool_name, exc)

    marker_position = 0
    timeline_count = 0
    if isinstance(feature_snapshot, Mapping):
        try:
            marker_position = int(feature_snapshot.get("marker_position") or 0)
        except (TypeError, ValueError):
            marker_position = 0
        try:
            timeline_count = int(feature_snapshot.get("timeline_count") or 0)
        except (TypeError, ValueError):
            timeline_count = 0

    latest_entity_context = _manager_get_latest_entity_context(manager, session_id) or {}
    reasoning_summary = ""
    try:
        reasoning_context = manager.get_reasoning_context(session_id)
        reasoning_summary = reasoning_context.get_injection_text()
    except Exception:
        reasoning_summary = ""

    checkpoint_id = f"opchk_{uuid4().hex}"
    conversation_snapshot = _operation_checkpoint_conversation_snapshot(messages, tool_use_id)
    serialized_ir_state = (
        _serialize_ir_document_state(ir_doc_state)
        if ir_doc_state is not None
        else _manager_get_ir_document_state(manager, session_id)
    )
    checkpoint_data: Dict[str, Any] = {
        "checkpoint_id": checkpoint_id,
        "request_id": str((request or {}).get("request_id") or ""),
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "display_label": _operation_display_label(tool_name, description),
        "description": description,
        "marker_position": marker_position,
        "timeline_count": timeline_count,
        "conversation_index": len(conversation_snapshot),
        "conversation_snapshot": conversation_snapshot,
        "ir_state": serialized_ir_state or {},
        "latest_entity_context": latest_entity_context,
        "feature_snapshot": feature_snapshot or {},
        "reasoning_summary": reasoning_summary,
        "timestamp": time.time(),
    }

    manager.save_operation_checkpoint(session_id, checkpoint_data)
    public_checkpoint = _checkpoint_public_metadata(checkpoint_data)

    await _send_message_safe(
        manager,
        session_id,
        {
            "type": "operation_checkpoint_created",
            "checkpoint": public_checkpoint,
        },
    )
    logger.info(
        "Session %s captured operation checkpoint %s for %s at marker %s",
        session_id,
        checkpoint_id,
        tool_name,
        marker_position,
    )
    return public_checkpoint


class SelectionToolCallError(Exception):
    """Raised when a geometry selection tool call cannot be processed."""


class EntityRefreshError(Exception):
    """Raised when entity context refresh fails after geometry modification."""
    pass


class PatternPreparation(NamedTuple):
    """Data returned when preparing a pattern feature payload."""

    parameters: Dict[str, Any]
    diagnostics: List[str]


def _timeline_count_from_state(timeline_state: Any) -> Optional[int]:
    """Extract the timeline item count from the request payload."""
    if timeline_state is None:
        return None

    if isinstance(timeline_state, Mapping):
        count_value = timeline_state.get("count")
        try:
            return int(count_value) if count_value is not None else None
        except (TypeError, ValueError):
            return None

    if isinstance(timeline_state, Sequence) and not isinstance(timeline_state, (str, bytes, bytearray)):
        return len(timeline_state)

    return None


async def _ensure_feature_snapshot(
    session_id: str,
    manager: ConnectionManager,
    timeline_state: Any,
) -> Optional[Dict[str, Any]]:
    """Fetch a fresh feature snapshot from Fusion when the cache is stale."""
    expected_count = _timeline_count_from_state(timeline_state)
    cached_snapshot = _manager_get_feature_snapshot(manager, session_id)

    if cached_snapshot is not None and expected_count is not None:
        cached_count = cached_snapshot.get("timeline_count")
        try:
            if cached_count is not None and int(cached_count) == expected_count:
                return cached_snapshot
        except (TypeError, ValueError):
            pass

    request_payload: Dict[str, Any] = {
        "type": "feature_snapshot_request",
        "reason": "refresh_before_prompt",
        "max_features": FEATURE_SNAPSHOT_LIMIT,
        "message_id": f"feature_snapshot_{uuid4().hex}",
    }

    if expected_count is not None:
        request_payload["timeline_count"] = expected_count

    if isinstance(timeline_state, Mapping):
        marker_position = timeline_state.get("marker_position")
        if marker_position is not None:
            request_payload["marker_position"] = marker_position

    await _send_message_safe(manager, session_id, request_payload)

    try:
        result = await _wait_for_matching_fusion_message_id(
            session_id,
            manager,
            expected_message_id=str(request_payload["message_id"]),
            timeout=EXECUTION_TIMEOUT,
            wait_context="feature_snapshot_refresh_before_prompt",
        )
    except asyncio.TimeoutError:
        return cached_snapshot

    result_type = result.get("type")
    if result_type != "feature_snapshot":
        # Put the message back on the queue so the main workflow can handle it appropriately
        await manager.store_fusion_result(session_id, result)
        return cached_snapshot

    if not result.get("success", True):
        return cached_snapshot if cached_snapshot is not None else result

    manager.set_feature_snapshot(session_id, result)
    return result


def _validate_entity_store_for_tool(
    session_id: str,
    manager: ConnectionManager,
    tool_name: str,
) -> Tuple[bool, str]:
    """
    Validate that the entity store has sufficient entities for the given tool.

    This implements the "mandatory entity refresh gate" from the spatial validation block:
    - If design_entities is empty, tool execution should HALT and request refresh
    - Returns a warning message if validation fails

    Args:
        session_id: Session identifier
        manager: WebSocket connection manager
        tool_name: Name of the tool about to be executed

    Returns:
        (is_valid, warning_message) - is_valid=True if OK to proceed,
        warning_message contains guidance if not valid
    """
    if tool_name not in TOOLS_REQUIRING_ENTITY_CONTEXT:
        return True, ""

    store = _get_entity_store(session_id, manager)

    if store.is_empty():
        return False, (
            f"ENTITY CONTEXT MISSING: Cannot execute '{tool_name}' - design_entities is empty. "
            f"No faces/edges/bodies are registered. This usually means:\n"
            f"1. No geometry has been created yet, OR\n"
            f"2. Entity context was not refreshed after the last operation.\n"
            f"ACTION: Either create geometry first, or wait for refreshed Design Entities after a successful geometry operation."
        )

    counts = store.get_entity_counts()

    # Check specific requirements
    if tool_name in TOOLS_REQUIRING_FACE_REFS and counts["face"] == 0:
        return False, (
            f"NO FACES AVAILABLE: Cannot execute '{tool_name}' - no faces in entity context. "
            f"Current entities: {counts['body']} bodies, {counts['face']} faces, {counts['edge']} edges. "
            f"ACTION: Ensure geometry has been created and entity context is refreshed."
        )

    if tool_name in TOOLS_REQUIRING_EDGE_REFS and counts["edge"] == 0:
        return False, (
            f"NO EDGES AVAILABLE: Cannot execute '{tool_name}' - no edges in entity context. "
            f"Current entities: {counts['body']} bodies, {counts['face']} faces, {counts['edge']} edges. "
            f"ACTION: Ensure geometry has been created and entity context is refreshed."
        )

    if tool_name in TOOLS_REQUIRING_BODY_REFS and counts["body"] == 0:
        return False, (
            f"NO BODIES AVAILABLE: Cannot execute '{tool_name}' - no bodies in entity context. "
            f"Current entities: {counts['body']} bodies, {counts['face']} faces, {counts['edge']} edges. "
            f"ACTION: Create solid geometry first (extrude a profile, etc.)."
        )

    return True, ""


def _validate_entity_context(context: Optional[Dict[str, Any]]) -> bool:
    """
    Validate that entity context is non-empty and contains valid tokens.

    Args:
        context: Entity context dictionary from Fusion

    Returns:
        True if context is valid, False otherwise
    """
    if not context or not isinstance(context, dict):
        return False

    entities = _collect_context_entities(context)

    # Require at least one body/face/edge to avoid silently accepting empty contexts.
    total_entities = (
        len(entities["bodies"]) +
        len(entities["faces"]) +
        len(entities["edges"])
    )
    if total_entities == 0:
        return False

    # Validate that entity tokens are valid strings in either token field.
    for entity_list in (entities["bodies"], entities["faces"], entities["edges"]):
        for entity in entity_list:
            token = _extract_entity_token(entity)
            if not token:
                return False

    return True


def _count_entities_in_context(context: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    """Return body/face/edge counts for a raw entity context payload."""
    if not isinstance(context, Mapping):
        return {"body": 0, "face": 0, "edge": 0}
    entities = _collect_context_entities(context)
    return {
        "body": len(entities["bodies"]),
        "face": len(entities["faces"]),
        "edge": len(entities["edges"]),
    }


def _manager_get_feature_snapshot(manager: Any, session_id: str) -> Optional[Dict[str, Any]]:
    getter = getattr(manager, "get_feature_snapshot", None)
    if callable(getter):
        snapshot = getter(session_id)
        return snapshot if isinstance(snapshot, dict) else None
    return None


def _manager_set_latest_entity_context(manager: Any, session_id: str, entity_context: Mapping[str, Any]) -> None:
    setter = getattr(manager, "set_latest_entity_context", None)
    if callable(setter):
        setter(session_id, dict(entity_context))


def _manager_get_latest_entity_context(manager: Any, session_id: str) -> Optional[Dict[str, Any]]:
    getter = getattr(manager, "get_latest_entity_context", None)
    if callable(getter):
        context = getter(session_id)
        return context if isinstance(context, dict) else None
    return None


def _manager_clear_latest_entity_context(manager: Any, session_id: str) -> None:
    clearer = getattr(manager, "clear_latest_entity_context", None)
    if callable(clearer):
        clearer(session_id)


async def _refresh_entity_context_with_retry(
    session_id: str,
    manager: ConnectionManager,
    tool_name: str,
    prev_signature: str = "",
    max_attempts: int = 4,  # 1 initial try + 3 retries
    operation_was_noop: bool = False  # True if Fusion reported no geometry change
) -> Dict[str, Any]:
    """
    Request and validate fresh entity context with retry logic and signature change detection.

    The function ensures that:
    1. Context is valid (non-empty, valid tokens)
    2. Entity refs or spatial data have actually changed (signature differs from prev_signature)
       UNLESS operation_was_noop=True or tool is in TOOLS_WITHOUT_ENTITY_EMISSION

    Retry schedule:
    - Attempt 1: 5 second timeout
    - Attempt 2: 5 second timeout (1 second after attempt 1)
    - Attempt 3: 5 second timeout (3 seconds after attempt 2)
    - Attempt 4: 5 second timeout (6 seconds after attempt 3)

    Args:
        session_id: Session identifier
        manager: WebSocket connection manager
        tool_name: Name of the tool that triggered the refresh
        prev_signature: Previous entity signature to compare against
        max_attempts: Maximum number of attempts (default: 4)
        operation_was_noop: If True, Fusion explicitly reported no geometry modification.
                           Unchanged signature will be accepted in this case.

    Returns:
        Valid entity context dictionary with changed signature (or unchanged if no-op)

    Raises:
        EntityRefreshError: If all attempts fail to get valid entity context
                           or if signature never changes for geometry-modifying operations
    """
    retry_delays = [0, 1, 3, 6]  # Delays BETWEEN attempts (0 for first attempt)
    timeout = 5.0  # Timeout for each individual attempt

    # Track failure modes for better error messaging
    timeout_count = 0
    validation_fail_count = 0
    unchanged_signature_count = 0
    last_valid_context = None
    last_signature = None
    last_entity_counts = None  # Track (bodies, faces, edges) tuple for invariant checks

    # Track timing for CAD API round-trips
    attempt_timings_ms = []
    total_refresh_start = time.perf_counter()

    for attempt in range(max_attempts):
        # Wait before retry (skip on first attempt)
        delay = retry_delays[min(attempt, len(retry_delays) - 1)]
        if delay > 0:
            await asyncio.sleep(delay)

        attempt_num = attempt + 1
        attempt_start = time.perf_counter()

        # Send request to Fusion
        context_request_id = f"ctx-{uuid4().hex}"
        await _send_message_safe(manager, session_id, {
            "type": "request_entity_context",
            "context_request_id": context_request_id,
        })

        # Wait for response
        try:
            fresh_context = await manager.wait_for_entity_context(
                session_id,
                timeout=timeout,
                expected_context_request_id=context_request_id,
            )

            # Explicit check for timeout (wait_for_entity_context returns None on timeout)
            if fresh_context is None:
                attempt_elapsed_ms = (time.perf_counter() - attempt_start) * 1000.0
                attempt_timings_ms.append(attempt_elapsed_ms)
                logger.debug(
                    "CAD API round-trip attempt %d/%d: TIMEOUT after %.1fms",
                    attempt_num, max_attempts, attempt_elapsed_ms
                )
                timeout_count += 1
                continue  # Skip to next retry

            # Record successful response time (before validation/signature checks)
            attempt_elapsed_ms = (time.perf_counter() - attempt_start) * 1000.0
            attempt_timings_ms.append(attempt_elapsed_ms)

            # Validate the context structure
            if not _validate_entity_context(fresh_context):
                logger.debug(
                    "CAD API round-trip attempt %d/%d: received in %.1fms but VALIDATION FAILED",
                    attempt_num, max_attempts, attempt_elapsed_ms
                )
                validation_fail_count += 1
                continue

            # Compute signature of fresh context to detect change.
            # Use normalised nested-or-flat entities so signature generation is consistent
            # with the prepopulation path and does not discard spatial_context payloads.
            fresh_entities = _collect_context_entities(fresh_context)
            bodies_for_signature = _normalise_entities_for_registration(fresh_entities["bodies"])
            faces_for_signature = _normalise_entities_for_registration(fresh_entities["faces"])
            edges_for_signature = _normalise_entities_for_registration(fresh_entities["edges"])

            # Create temporary store to compute signature without polluting session store.
            temp_store = EntityStore()
            await temp_store.register_entities("body", bodies_for_signature)
            await temp_store.register_entities("face", faces_for_signature)
            await temp_store.register_entities("edge", edges_for_signature)
            fresh_signature = temp_store.get_signature()

            # Extract entity counts for invariant checking (detect recycled topology IDs)
            fresh_counts = (
                len(bodies_for_signature),
                len(faces_for_signature),
                len(edges_for_signature),
            )

            # Check if signature changed from previous state
            if fresh_signature == prev_signature:
                logger.debug(
                    "CAD API round-trip attempt %d/%d: received in %.1fms but SIGNATURE UNCHANGED",
                    attempt_num, max_attempts, attempt_elapsed_ms
                )
                unchanged_signature_count += 1
                # Update tracking variables for next iteration
                last_valid_context = fresh_context
                last_signature = fresh_signature
                last_entity_counts = fresh_counts
                continue  # Retry - wait for geometry change to propagate

            # Success: valid context and signature changed
            total_refresh_ms = (time.perf_counter() - total_refresh_start) * 1000.0
            entity_count = fresh_counts[0] + fresh_counts[1] + fresh_counts[2]
            logger.info(
                "CAD API refresh SUCCESS: tool=%s attempt=%d/%d round_trip=%.1fms "
                "total_refresh=%.1fms entities=%d (bodies=%d faces=%d edges=%d)",
                tool_name, attempt_num, max_attempts, attempt_elapsed_ms,
                total_refresh_ms, entity_count, fresh_counts[0], fresh_counts[1], fresh_counts[2]
            )
            _manager_set_latest_entity_context(manager, session_id, fresh_context)
            return fresh_context

        except Exception as exc:
            attempt_elapsed_ms = (time.perf_counter() - attempt_start) * 1000.0
            attempt_timings_ms.append(attempt_elapsed_ms)
            logger.debug(
                "CAD API round-trip attempt %d/%d: EXCEPTION after %.1fms: %s",
                attempt_num, max_attempts, attempt_elapsed_ms, exc
            )
            pass  # Retry on next attempt

    # All attempts exhausted - determine failure mode and whether to fail hard or accept unchanged
    total_refresh_ms = (time.perf_counter() - total_refresh_start) * 1000.0
    avg_round_trip_ms = sum(attempt_timings_ms) / len(attempt_timings_ms) if attempt_timings_ms else 0

    # Accept unchanged signature ONLY when Fusion explicitly reported no-op
    # REMOVED: TOOLS_WITHOUT_ENTITY_EMISSION acceptance (was allowing silent failures)
    # Construction planes now must report no_op=true for duplicate creation
    # Otherwise, for all tools, unchanged signature = failure.
    if last_valid_context and unchanged_signature_count > 0:
        if tool_name in TOOLS_WITHOUT_ENTITY_EMISSION:
            logger.info(
                "CAD API refresh ACCEPTED (no-emission tool): tool=%s attempts=%d "
                "total_refresh=%.1fms avg_round_trip=%.1fms",
                tool_name, len(attempt_timings_ms), total_refresh_ms, avg_round_trip_ms
            )
            _manager_set_latest_entity_context(manager, session_id, last_valid_context)
            return last_valid_context
        elif operation_was_noop:
            logger.info(
                "CAD API refresh ACCEPTED (no-op): tool=%s attempts=%d "
                "total_refresh=%.1fms avg_round_trip=%.1fms",
                tool_name, len(attempt_timings_ms), total_refresh_ms, avg_round_trip_ms
            )
            _manager_set_latest_entity_context(manager, session_id, last_valid_context)
            return last_valid_context
        else:
            # CRITICAL: Geometry-modifying tool should have changed the signature
            # Unchanged signature indicates Fusion didn't process the operation or failed silently
            logger.warning(
                "CAD API refresh FAILED (signature unchanged): tool=%s attempts=%d "
                "total_refresh=%.1fms avg_round_trip=%.1fms",
                tool_name, len(attempt_timings_ms), total_refresh_ms, avg_round_trip_ms
            )
            raise EntityRefreshError(
                f"Entity signature unchanged after {tool_name} "
                f"({unchanged_signature_count}/{max_attempts} attempts with exponential backoff). "
                f"This indicates Fusion did not process the geometry modification or failed silently. "
                f"Cannot proceed with potentially stale entity references. "
                f"Tool: {tool_name}"
            )

    # No valid context - construct detailed error message
    logger.warning(
        "CAD API refresh FAILED (no valid context): tool=%s attempts=%d "
        "total_refresh=%.1fms avg_round_trip=%.1fms timeouts=%d validation_fails=%d",
        tool_name, len(attempt_timings_ms), total_refresh_ms, avg_round_trip_ms,
        timeout_count, validation_fail_count
    )
    error_parts = [
        f"Failed to refresh entity context after {tool_name}",
        f"(tried {max_attempts} times with exponential backoff)."
    ]

    if timeout_count > 0:
        error_parts.append(f"{timeout_count} timeout(s).")
    if validation_fail_count > 0:
        error_parts.append(f"{validation_fail_count} validation failure(s).")
    if unchanged_signature_count > 0 and not last_valid_context:
        error_parts.append(f"{unchanged_signature_count} unchanged signature(s) with invalid context.")

    error_parts.append("Cannot proceed without valid spatial data.")

    raise EntityRefreshError(" ".join(error_parts))


async def _ensure_runtime_entity_context_synced(
    session_id: str,
    manager: ConnectionManager,
    *,
    reason: str,
) -> Optional[Dict[str, Any]]:
    """
    Ensure runtime entity context is present whenever the store has active refs.

    Fail-closed behavior:
    - If the entity store is non-empty but the latest runtime entity context is
      missing/empty, force a refresh.
    - If forced refresh cannot produce valid non-empty context, raise
      EntityRefreshError and stop execution.
    """
    if not callable(getattr(manager, "get_entity_store", None)):
        return None

    store = _get_entity_store(session_id, manager)
    store_counts = {
        "body": len(store.get_all_bodies()),
        "face": len(store.get_all_faces()),
        "edge": len(store.get_all_edges()),
    }
    total_store_entities = sum(store_counts.values())
    if total_store_entities == 0:
        return None

    latest_context = _manager_get_latest_entity_context(manager, session_id)
    if _validate_entity_context(dict(latest_context) if isinstance(latest_context, Mapping) else None):
        return dict(latest_context)  # type: ignore[arg-type]

    logger.warning(
        "Session %s runtime context missing while store has refs (reason=%s, store: %d bodies, %d faces, %d edges). "
        "Forcing context refresh.",
        session_id,
        reason,
        store_counts["body"],
        store_counts["face"],
        store_counts["edge"],
    )

    refreshed_context = await _refresh_entity_context_with_retry(
        session_id,
        manager,
        tool_name="runtime_state_sync_guard",
        prev_signature=store.get_signature(),
        operation_was_noop=True,
    )

    refreshed_counts = _count_entities_in_context(refreshed_context)
    if sum(refreshed_counts.values()) == 0:
        raise EntityRefreshError(
            "Runtime state sync guard failed: entity store has active refs, but refreshed entity context was empty. "
            f"(store bodies={store_counts['body']} faces={store_counts['face']} edges={store_counts['edge']})"
        )

    store.soft_clear()
    await _prepopulate_entity_store(session_id, manager, refreshed_context)
    store.prune_stale_tokens(_extract_tokens_from_context(refreshed_context))
    store.store_signature()
    _manager_set_latest_entity_context(manager, session_id, refreshed_context)
    return refreshed_context


def _replace_last_tool_result_text(messages: List[Dict[str, Any]], new_text: str) -> None:
    """Replace the text payload of the last tool_result message in-place (if present)."""
    if not messages:
        return
    last_msg = messages[-1]
    if last_msg.get("role") != "user":
        return
    content = last_msg.get("content")
    if not isinstance(content, list):
        return
    for content_block in content:
        if not isinstance(content_block, dict) or content_block.get("type") != "tool_result":
            continue
        block_content = content_block.get("content")
        if not isinstance(block_content, list):
            continue
        for text_block in block_content:
            if isinstance(text_block, dict) and text_block.get("type") == "text":
                text_block["text"] = new_text
                return


def _append_to_last_tool_result_text(messages: List[Dict[str, Any]], suffix: str) -> None:
    """Append text to the last tool_result message in-place (if present)."""
    if not suffix:
        return
    if not messages:
        return
    last_msg = messages[-1]
    if last_msg.get("role") != "user":
        return
    content = last_msg.get("content")
    if not isinstance(content, list):
        return
    for content_block in content:
        if not isinstance(content_block, dict) or content_block.get("type") != "tool_result":
            continue
        block_content = content_block.get("content")
        if not isinstance(block_content, list):
            continue
        for text_block in block_content:
            if isinstance(text_block, dict) and text_block.get("type") == "text":
                text_block["text"] += suffix
                return


async def _refresh_and_enrich_after_success(
    session_id: str,
    manager: ConnectionManager,
    tool_name: str,
    result: Mapping[str, Any],
    messages: List[Dict[str, Any]],
) -> None:
    """Refresh entity context (option A no-op behavior), update store, and append unified context."""
    if tool_name not in REFRESH_ON_SUCCESS_TOOLS:
        return

    # Option A: always refresh for geometry/timeline tools, even if Fusion reports no-op.
    operation_was_noop = (result.get("no_op") is True)

    store = _get_entity_store(session_id, manager)
    prev_signature = store.get_signature()
    full_refresh_start = time.perf_counter()

    fresh_context = await _refresh_entity_context_with_retry(
        session_id,
        manager,
        tool_name,
        prev_signature=prev_signature,
        operation_was_noop=operation_was_noop,
    )

    store_update_start = time.perf_counter()
    if tool_name in TIMELINE_MODIFYING_TOOLS:
        store.clear()
    else:
        store.soft_clear()
    await _prepopulate_entity_store(session_id, manager, fresh_context)
    current_tokens = _extract_tokens_from_context(fresh_context)
    store.prune_stale_tokens(current_tokens)
    _manager_set_latest_entity_context(manager, session_id, fresh_context)
    store_update_ms = (time.perf_counter() - store_update_start) * 1000.0
    full_refresh_ms = (time.perf_counter() - full_refresh_start) * 1000.0
    logger.info(
        "Entity refresh complete: tool=%s full_cycle=%.1fms store_update=%.1fms (CAD API time captured separately above)",
        tool_name,
        full_refresh_ms,
        store_update_ms,
    )
    store.store_signature()

    # Keep feature snapshot aligned with refreshed runtime state.
    try:
        refreshed_snapshot = await _request_feature_snapshot(
            session_id,
            manager,
            reason=f"post_{tool_name}",
            max_features=FEATURE_SNAPSHOT_LIMIT,
        )
        if refreshed_snapshot and refreshed_snapshot.get("success", True):
            manager.set_feature_snapshot(session_id, refreshed_snapshot)
    except Exception as exc:
        logger.warning("Failed to refresh feature snapshot after %s: %s", tool_name, exc)

    # Re-enrich the tool result (tool-specific summary) using fresh entity context.
    if tool_name in _TOOL_RESULT_ENRICHERS:
        try:
            _, enriched_text = _summarise_execution_result(tool_name, result, fresh_context)
            _replace_last_tool_result_text(messages, enriched_text)
        except Exception as exc:
            logger.warning("Failed to re-enrich tool result: %s", exc)

    # Append unified entity context to the last tool result.
    try:
        entity_text = _format_unified_context(fresh_context)
        if entity_text:
            _append_to_last_tool_result_text(messages, f"\n\n{entity_text}")
    except Exception as exc:
        logger.warning(
            "Failed to format unified entity context after %s: %s. Entity data still available in store.",
            tool_name,
            exc,
        )

    # Announce newly created refs only after cross-checking with refreshed context.
    created_entities = result.get("created_entities")
    if isinstance(created_entities, Mapping):
        announced_refs: List[str] = []
        for kind in ("bodies", "faces", "edges"):
            created_tokens = created_entities.get(kind, []) or []
            pending_tokens = {str(token) for token in created_tokens if str(token).strip()}
            if not pending_tokens:
                continue
            context_entities = fresh_context.get(kind, []) or []
            for entity in context_entities:
                if not isinstance(entity, Mapping):
                    continue
                token = entity.get("entity_token")
                if token in pending_tokens:
                    entity_ref = entity.get("entity_ref")
                    if entity_ref:
                        announced_refs.append(str(entity_ref))
                    pending_tokens.discard(token)
            if pending_tokens:
                logger.warning(
                    "Session %s: %d created %s not found in refreshed context after %s",
                    session_id,
                    len(pending_tokens),
                    kind,
                    tool_name,
                )
        if announced_refs:
            _append_to_last_tool_result_text(
                messages,
                f"\n\nNew entities available: {', '.join(announced_refs)}",
            )


async def _request_feature_snapshot(
    session_id: str,
    manager: ConnectionManager,
    *,
    reason: str,
    max_features: int = FEATURE_SNAPSHOT_LIMIT,
) -> Optional[Dict[str, Any]]:
    """Request a fresh feature snapshot from Fusion without timeline context."""

    payload: Dict[str, Any] = {
        "type": "feature_snapshot_request",
        "reason": reason,
        "max_features": max_features,
        "message_id": f"feature_snapshot_{uuid4().hex}",
    }

    await _send_message_safe(manager, session_id, payload)

    try:
        result = await _wait_for_matching_fusion_message_id(
            session_id,
            manager,
            expected_message_id=str(payload["message_id"]),
            timeout=EXECUTION_TIMEOUT,
            wait_context=f"feature_snapshot_{reason}",
        )
    except asyncio.TimeoutError:
        logger.warning("Timed out waiting for feature snapshot (%s) for session %s", reason, session_id)
        return None

    if result.get("type") != "feature_snapshot":
        await manager.store_fusion_result(session_id, result)
        logger.warning(
            "Unexpected message type '%s' received while requesting feature snapshot (%s) for session %s",
            result.get("type"),
            reason,
            session_id,
        )
        return None

    if not result.get("success", True):
        logger.warning(
            "Feature snapshot request (%s) failed for session %s: %s",
            reason,
            session_id,
            result.get("error") or result.get("message") or "unknown error",
        )
        return result

    manager.set_feature_snapshot(session_id, result)
    return result


async def _wait_for_matching_fusion_message_id(
    session_id: str,
    manager: ConnectionManager,
    *,
    expected_message_id: str,
    timeout: int = EXECUTION_TIMEOUT,
    wait_context: str = "message_id_wait",
) -> Mapping[str, Any]:
    """
    Wait for a Fusion payload with an exact matching message_id.

    Non-matching payloads are deferred and re-queued so unrelated operations are
    not accidentally consumed by the wrong wait path.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0, int(timeout))
    deferred_results: List[Mapping[str, Any]] = []

    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError

            result = await manager.wait_for_fusion_result(session_id, timeout=remaining)
            if not isinstance(result, Mapping):
                continue

            result_message_id = result.get("message_id")
            result_message_id_str = str(result_message_id).strip() if result_message_id is not None else ""
            if result_message_id_str != expected_message_id:
                logger.warning(
                    "Session %s deferred %s result for message_id %s while waiting for %s",
                    session_id,
                    wait_context,
                    result_message_id_str or "<missing>",
                    expected_message_id,
                )
                deferred_results.append(result)
                continue

            return result
    finally:
        for deferred in deferred_results:
            try:
                await manager.store_fusion_result(session_id, dict(deferred))
            except Exception:
                logger.exception(
                    "Session %s failed to re-queue deferred %s result while waiting for message_id %s",
                    session_id,
                    wait_context,
                    expected_message_id,
                )


async def _wait_for_matching_tool_result(
    session_id: str,
    manager: ConnectionManager,
    *,
    tool_name: str,
    tool_use_id: str,
    timeout: int = EXECUTION_TIMEOUT,
    wait_context: str = "tool",
) -> Mapping[str, Any]:
    """
    Wait for a Fusion result that matches the expected tool_use_id.

    Mismatched results are held temporarily and re-queued before returning so
    unrelated operations are not consumed by the wrong wait path.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0, int(timeout))
    deferred_results: List[Mapping[str, Any]] = []

    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError

            result = await manager.wait_for_fusion_result(session_id, timeout=remaining)
            if not isinstance(result, Mapping):
                continue

            result_tool_use_id = result.get("tool_use_id")
            result_tool_use_id_str = str(result_tool_use_id).strip() if result_tool_use_id is not None else ""
            if result_tool_use_id_str != tool_use_id:
                logger.warning(
                    "Session %s deferred %s result for tool_use_id %s while waiting for %s (%s)",
                    session_id,
                    wait_context,
                    result_tool_use_id_str or "<missing>",
                    tool_use_id,
                    tool_name,
                )
                deferred_results.append(result)
                continue

            return result
    finally:
        for deferred in deferred_results:
            try:
                await manager.store_fusion_result(session_id, dict(deferred))
            except Exception:
                logger.exception(
                    "Session %s failed to re-queue deferred %s result while waiting for %s (%s)",
                    session_id,
                    wait_context,
                    tool_use_id,
                    tool_name,
                )


async def _handle_list_features(
    session_id: str,
    manager: ConnectionManager,
    tool_input: Mapping[str, Any],
) -> Tuple[bool, str]:
    """Render the most recent feature snapshot for the LLM."""

    allowed_keys = {"description"}
    extra = set(tool_input.keys()) - allowed_keys
    if extra:
        raise SelectionToolCallError(f"list_features received unexpected parameter(s): {sorted(extra)}")

    description = tool_input.get("description")
    if description is None or not str(description).strip():
        raise SelectionToolCallError("list_features requires a non-empty 'description' field.")

    diagnostics: List[str] = []

    snapshot = await _request_feature_snapshot(session_id, manager, reason="list_features")

    if snapshot is None:
        cached = _manager_get_feature_snapshot(manager, session_id)
        if cached is None:
            return False, (
                "Feature snapshot unavailable. Ask the user to create or modify a feature so the cache can be populated."
            )
        snapshot = cached
        diagnostics.append("using cached snapshot (refresh timed out)")
    elif not snapshot.get("success", True):
        message = _format_feature_snapshot(snapshot) or (
            snapshot.get("error") or snapshot.get("message") or "Feature snapshot unavailable."
        )
        return False, message
    else:
        diagnostics.append("captured fresh snapshot")

    message_body = _format_feature_snapshot(snapshot)
    if not message_body:
        return True, "No recent features were captured in the snapshot."

    if diagnostics:
        diag_text = "; ".join(diagnostics)
        message_body = f"diagnostics={diag_text}\n{message_body}"

    return True, message_body


def _axis_index_from_key(axis_key: str) -> int:
    key = axis_key.lower()
    if key.startswith("-"):
        key = key[1:]
    try:
        return WORLD_AXIS_ORDER.index(key)
    except ValueError:  # pragma: no cover - defensive guard
        return 0


def _normalize_axis_key(axis_key: str) -> str:
    axis_key = axis_key.lower()
    if axis_key in WORLD_AXIS_VECTORS:
        return axis_key
    if axis_key.startswith("-") and axis_key[1:] in WORLD_AXIS_VECTORS:
        return axis_key
    if axis_key in {"x", "y", "z"}:
        return axis_key
    if axis_key in {"+x", "+y", "+z"}:
        return axis_key[1:]
    # Default fallback
    return "x"


def _axis_key_from_hint(hint: Any) -> Optional[str]:
    if hint is None:
        return None

    if isinstance(hint, str):
        key = hint.strip().lower()
        if key.startswith("+"):
            key = key[1:]
        if key in WORLD_AXIS_VECTORS:
            return key
        if key in {"x", "y", "z"}:
            return key
        return None

    if isinstance(hint, Sequence) and not isinstance(hint, (str, bytes, bytearray)):
        values: List[float] = []
        for item in hint:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                return None
        if len(values) != 3:
            return None
        dominant_index = max(range(3), key=lambda idx: abs(values[idx]))
        dominant_value = values[dominant_index]
        if math.isclose(dominant_value, 0.0, abs_tol=1e-6):
            return None
        base_axis = WORLD_AXIS_ORDER[dominant_index]
        return base_axis if dominant_value >= 0 else f"-{base_axis}"

    return None


def _extract_feature_bbox(feature: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    if not isinstance(feature, Mapping):
        return None

    bbox = feature.get("bounding_box")
    if isinstance(bbox, Mapping):
        return bbox

    bodies = feature.get("bodies")
    if isinstance(bodies, Sequence):
        for body in bodies:
            if isinstance(body, Mapping):
                candidate = body.get("bounding_box")
                if isinstance(candidate, Mapping):
                    return candidate
    return None


def _bbox_extents(bbox: Mapping[str, Any]) -> Optional[Tuple[List[float], List[float]]]:
    min_point = bbox.get("min") or bbox.get("min_point")
    max_point = bbox.get("max") or bbox.get("max_point")
    if not (isinstance(min_point, Mapping) and isinstance(max_point, Mapping)):
        return None

    try:
        min_coords = [float(min_point.get(axis, 0.0)) for axis in WORLD_AXIS_ORDER]
        max_coords = [float(max_point.get(axis, 0.0)) for axis in WORLD_AXIS_ORDER]
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return None

    return min_coords, max_coords


def _aggregate_bounds(features: Sequence[Mapping[str, Any]]) -> Optional[Tuple[List[float], List[float]]]:
    min_coords = [math.inf, math.inf, math.inf]
    max_coords = [-math.inf, -math.inf, -math.inf]
    any_valid = False

    for feature in features:
        bbox = _extract_feature_bbox(feature)
        if not bbox:
            continue
        extents = _bbox_extents(bbox)
        if not extents:
            continue
        mins, maxs = extents
        for idx in range(3):
            if mins[idx] < min_coords[idx]:
                min_coords[idx] = mins[idx]
            if maxs[idx] > max_coords[idx]:
                max_coords[idx] = maxs[idx]
        any_valid = True

    if not any_valid:
        return None

    return min_coords, max_coords


def _extent_along_axis(bounds: Tuple[List[float], List[float]], axis_index: int) -> Optional[float]:
    try:
        extent = bounds[1][axis_index] - bounds[0][axis_index]
    except (IndexError, TypeError):  # pragma: no cover - defensive guard
        return None
    if not math.isfinite(extent):
        return None
    return max(extent, 0.0)


def _infer_anchor_point(bounds: Optional[Tuple[List[float], List[float]]]) -> List[float]:
    if not bounds:
        return [0.0, 0.0, 0.0]
    center = []
    for idx in range(3):
        min_val = bounds[0][idx]
        max_val = bounds[1][idx]
        if not (math.isfinite(min_val) and math.isfinite(max_val)):
            center.append(0.0)
        else:
            center.append((min_val + max_val) / 2.0)
    return center


def _sync_payload_parameters(payload: Dict[str, Any]) -> None:
    """
    Mirror resolved top-level fields into the nested 'parameters' dict that
    the Fusion add-in consumes. This ensures we do not send stale logical
    refs (e.g., face_5) after converting them to real entity tokens.
    """
    if not isinstance(payload, dict):
        return

    reserved = {"type", "operation", "tool_use_id", "description", "parameters"}

    # Start with any existing parameters (e.g., pattern_type prepared upstream),
    # then overlay canonical, resolved top-level fields. This preserves nested
    # fields that have no top-level counterpart while still favoring the
    # resolved values we place at the top level (entity_tokens, numeric types).
    params: Dict[str, Any] = {}
    if isinstance(payload.get("parameters"), Mapping):
        params.update(payload["parameters"])

    for key, value in payload.items():
        if key in reserved:
            continue
        params[key] = value

    payload["parameters"] = params


def _infer_axis_key(
    features: Sequence[Mapping[str, Any]],
    *,
    preferred_hint: Optional[str] = None,
    exclude_axis: Optional[str] = None,
) -> str:
    if preferred_hint:
        normalized = _normalize_axis_key(preferred_hint)
        if normalized in WORLD_AXIS_VECTORS:
            return normalized

    bounds = _aggregate_bounds(features)
    if bounds:
        extents = [
            _extent_along_axis(bounds, idx) or 0.0
            for idx in range(3)
        ]

        if exclude_axis:
            exclusion_index = _axis_index_from_key(exclude_axis)
        else:
            exclusion_index = -1

        candidate_indices = [idx for idx in range(3) if idx != exclusion_index]
        candidate_indices.sort(key=lambda idx: extents[idx], reverse=True)

        if candidate_indices:
            best_idx = candidate_indices[0]
            if extents[best_idx] > 0:
                axis_label = WORLD_AXIS_ORDER[best_idx]
                return axis_label

    # Fallback axis selection avoiding exclusion if possible
    for axis_label in WORLD_AXIS_ORDER:
        if exclude_axis and axis_label == exclude_axis.lstrip("-"):
            continue
        return axis_label

    return "x"


def _infer_spacing_cm(
    features: Sequence[Mapping[str, Any]],
    axis_key: str,
    *,
    minimum_spacing: float = 0.5,
) -> float:
    bounds = _aggregate_bounds(features)
    extent = _extent_along_axis(bounds, _axis_index_from_key(axis_key)) if bounds else None

    if extent is None or extent <= 0:
        return max(minimum_spacing, 1.0)

    spacing = max(extent * 1.5, minimum_spacing)
    return round(spacing, 4)


async def _prepare_pattern_feature(
    session_id: str,
    manager: ConnectionManager,
    tool_input: Mapping[str, Any],
) -> PatternPreparation:
    allowed_keys = {
        "pattern_type",
        "feature_tokens",
        "count_x",
        "spacing_x_cm",
        "count_y",
        "spacing_y_cm",
        "rotation_count",
        "rotation_angle_deg",
        "orientation_hint",
        "feature_name",
        "description",
    }
    extra = set(tool_input.keys()) - allowed_keys
    if extra:
        raise SelectionToolCallError(
            f"create_pattern_feature received unexpected parameter(s): {sorted(extra)}"
        )

    pattern_type = str(tool_input.get("pattern_type", "")).strip().lower()
    if pattern_type not in {"rectangular", "circular"}:
        raise SelectionToolCallError("create_pattern_feature requires pattern_type to be 'rectangular' or 'circular'.")

    raw_tokens = tool_input.get("feature_tokens")
    if raw_tokens is None:
        raw_tokens = ["auto_last"]
    if not isinstance(raw_tokens, Sequence) or isinstance(raw_tokens, (str, bytes, bytearray)):
        raise SelectionToolCallError("create_pattern_feature expects 'feature_tokens' to be an array of strings.")

    description = tool_input.get("description")
    if description is None or not str(description).strip():
        raise SelectionToolCallError("create_pattern_feature requires a non-empty 'description' field.")

    feature_name = tool_input.get("feature_name")
    if feature_name is not None:
        feature_name = str(feature_name).strip()
        if not feature_name:
            feature_name = None

    snapshot = await _request_feature_snapshot(session_id, manager, reason="create_pattern_feature")
    if snapshot is None:
        snapshot = _manager_get_feature_snapshot(manager, session_id)
        if snapshot is None:
            raise SelectionToolCallError(
                "No feature snapshot available. Patterning requires at least one recent feature; run a feature operation first."
            )
        diagnostics = ["using cached snapshot (refresh timed out)"]
    elif not snapshot.get("success", True):
        message = snapshot.get("error") or snapshot.get("message") or "Feature snapshot unavailable."
        raise SelectionToolCallError(message)
    else:
        diagnostics = ["captured fresh snapshot"]

    features = snapshot.get("features")
    if not isinstance(features, list) or not features:
        raise SelectionToolCallError("Feature snapshot does not contain any features to pattern.")

    feature_lookup: Dict[str, Mapping[str, Any]] = {}
    for feature in features:
        token = feature.get("entity_token")
        if isinstance(token, str) and token.strip():
            feature_lookup[token] = feature

    latest_feature = None
    for feature in features:
        token = feature.get("entity_token")
        if isinstance(token, str) and token.strip():
            latest_feature = token
            break

    resolved_tokens: List[str] = []
    missing_tokens: List[str] = []
    auto_last_used = False

    for token in raw_tokens:
        token_str = str(token).strip()
        if token_str.lower() == "auto_last":
            if latest_feature is None:
                missing_tokens.append("auto_last")
                continue
            if latest_feature not in resolved_tokens:
                resolved_tokens.append(latest_feature)
                auto_last_used = True
            continue
        if token_str in feature_lookup:
            if token_str not in resolved_tokens:
                resolved_tokens.append(token_str)
        else:
            missing_tokens.append(token_str)

    if missing_tokens:
        raise SelectionToolCallError(
            f"create_pattern_feature could not find feature token(s): {missing_tokens}. Use list_features to inspect available tokens."
        )

    if not resolved_tokens:
        raise SelectionToolCallError("create_pattern_feature requires at least one resolvable feature token.")

    selected_features = [feature_lookup[token] for token in resolved_tokens]

    bounds = _aggregate_bounds(selected_features)
    anchor_point = _infer_anchor_point(bounds)

    orientation_hint = tool_input.get("orientation_hint")
    hinted_axis = _axis_key_from_hint(orientation_hint)

    diagnostics.append(f"resolved_tokens={resolved_tokens}")
    diagnostics.append(f"anchor_point={tuple(round(v, 4) for v in anchor_point)}")
    diagnostics.append(f"auto_last_used={auto_last_used}")

    parameters: Dict[str, Any] = {
        "pattern_type": pattern_type,
        "feature_tokens": resolved_tokens,
        "anchor_point_cm": [round(v, 6) for v in anchor_point],
    }
    if feature_name:
        parameters["feature_name"] = feature_name

    if pattern_type == "rectangular":
        count_x = tool_input.get("count_x")
        if count_x is None:
            count_x = 2
        if not isinstance(count_x, int) or count_x < 2:
            raise SelectionToolCallError("count_x must be an integer greater than or equal to 2 for rectangular patterns.")

        primary_axis = _infer_axis_key(selected_features, preferred_hint=hinted_axis)
        primary_axis = _normalize_axis_key(primary_axis)
        primary_vector = WORLD_AXIS_VECTORS.get(primary_axis, WORLD_AXIS_VECTORS["x"])

        spacing_x = tool_input.get("spacing_x_cm")
        if spacing_x is None:
            spacing_x = _infer_spacing_cm(selected_features, primary_axis)
            diagnostics.append(f"inferred_spacing_x={spacing_x}")
        try:
            spacing_x_val = float(spacing_x)
        except (TypeError, ValueError):
            raise SelectionToolCallError("spacing_x_cm must be a number if provided.")
        if spacing_x_val <= 0:
            raise SelectionToolCallError("spacing_x_cm must be positive.")

        count_y = tool_input.get("count_y")
        spacing_y_val: Optional[float] = None
        secondary_axis: Optional[str] = None
        secondary_vector: Optional[List[float]] = None

        if count_y is not None:
            if not isinstance(count_y, int) or count_y < 2:
                raise SelectionToolCallError("count_y must be an integer greater than or equal to 2 when provided.")

            secondary_axis = _infer_axis_key(
                selected_features,
                preferred_hint=None,
                exclude_axis=primary_axis,
            )
            secondary_axis = _normalize_axis_key(secondary_axis)
            secondary_vector = WORLD_AXIS_VECTORS.get(secondary_axis, WORLD_AXIS_VECTORS["y"])

            spacing_y = tool_input.get("spacing_y_cm")
            if spacing_y is None:
                spacing_y = _infer_spacing_cm(selected_features, secondary_axis)
                diagnostics.append(f"inferred_spacing_y={spacing_y}")
            try:
                spacing_y_val = float(spacing_y)
            except (TypeError, ValueError):
                raise SelectionToolCallError("spacing_y_cm must be a number if provided.")
            if spacing_y_val <= 0:
                raise SelectionToolCallError("spacing_y_cm must be positive when count_y is provided.")

        parameters.update(
            {
                "count_x": int(count_x),
                "spacing_x_cm": float(spacing_x_val),
                "primary_axis": primary_axis,
                "primary_direction": primary_vector,
                "auto_last_used": auto_last_used,
            }
        )

        if count_y is not None and spacing_y_val is not None and secondary_axis and secondary_vector:
            parameters.update(
                {
                    "count_y": int(count_y),
                    "spacing_y_cm": float(spacing_y_val),
                    "secondary_axis": secondary_axis,
                    "secondary_direction": secondary_vector,
                }
            )

        diagnostics.append(f"primary_axis={primary_axis}")
        if secondary_axis:
            diagnostics.append(f"secondary_axis={secondary_axis}")
        diagnostics.append(f"count_x={count_x}")
        if count_y is not None:
            diagnostics.append(f"count_y={count_y}")

    else:  # circular pattern
        rotation_count = tool_input.get("rotation_count")
        if rotation_count is None:
            rotation_count = 3
        if not isinstance(rotation_count, int) or rotation_count < 2:
            raise SelectionToolCallError("rotation_count must be an integer greater than or equal to 2 for circular patterns.")

        rotation_angle = tool_input.get("rotation_angle_deg")
        if rotation_angle is None:
            rotation_angle = 360.0
        try:
            rotation_angle_val = float(rotation_angle)
        except (TypeError, ValueError):
            raise SelectionToolCallError("rotation_angle_deg must be a number if provided.")
        if rotation_angle_val <= 0:
            raise SelectionToolCallError("rotation_angle_deg must be positive.")

        axis_key = _infer_axis_key(selected_features, preferred_hint=hinted_axis)
        axis_key = _normalize_axis_key(axis_key)
        axis_vector = WORLD_AXIS_VECTORS.get(axis_key, WORLD_AXIS_VECTORS["z"])

        parameters.update(
            {
                "rotation_count": int(rotation_count),
                "rotation_angle_deg": rotation_angle_val,
                "axis": axis_key,
                "axis_direction": axis_vector,
                "auto_last_used": auto_last_used,
            }
        )

        diagnostics.append(f"axis={axis_key}")
        diagnostics.append(f"rotation_count={rotation_count}")
        diagnostics.append(f"rotation_angle_deg={rotation_angle_val}")

    return PatternPreparation(parameters=parameters, diagnostics=diagnostics)

async def handle_execute_request(
    session_id: str,
    request: Mapping[str, Any],
    manager: ConnectionManager,
) -> None:
    """
    Main execution loop that repeatedly calls LLM for a single tool call,
    executes it in Fusion, and feeds the result back into the conversation.
    """
    request = dict(request)
    normalized_attachments = normalize_request_attachments(request)
    if normalized_attachments.prompt_context:
        request["attachments_context"] = normalized_attachments.prompt_context
    if normalized_attachments.normalized:
        request["normalized_attachments"] = normalized_attachments.normalized
        logger.info(
            "Normalized attachments for session %s: %s",
            session_id,
            attachment_debug_summary(normalized_attachments),
        )
    if normalized_attachments.first_image_data and not request.get("image_data"):
        request["image_data"] = normalized_attachments.first_image_data
        request["image_format"] = normalized_attachments.first_image_format or "png"

    has_attachments = bool(request.get("attachments")) or bool(request.get("normalized_attachments"))
    if not request.get("user_request") and not request.get("image_data") and not has_attachments:
        raise ValueError("Execution request must include 'user_request', 'image_data', or 'attachments'.")

    max_iterations = int(request.get("max_iterations", DEFAULT_MAX_ITERATIONS))
    model_name = request.get("model_name")

    # ========== VISION TRANSLATION ==========
    # If image_data is present, translate sketch to text before processing
    image_data = request.get("image_data")
    if image_data:
        try:
            from .vision_translator import get_vision_translator
            
            logger.info(f"Vision data detected in request for session {session_id}")
            
            # Send activity log to UI
            await _send_message_safe(manager, session_id, {
                "type": "activity_log",
                "message": "🔍 Analyzing your sketch with Gemini vision...",
                "level": "info"
            })
            
            # Get vision translator and process image
            translator = get_vision_translator()
            image_format = request.get("image_format", "png")
            user_text = request.get("user_request", "")

            # Extract and format entity context for spatial awareness
            entity_context = request.get("entity_context")
            spatial_info = ""
            if entity_context:
                try:
                    spatial_info = _format_unified_context(entity_context)
                except Exception as format_exc:
                    logger.warning(f"Failed to format entity context for vision translator: {format_exc}")
                    spatial_info = ""  # Continue without spatial info

            # Combine user text with spatial context
            combined_context = user_text
            if spatial_info:
                combined_context = f"{user_text}\n\n{spatial_info}" if user_text else spatial_info

            logger.info(f"Vision translator context: user_text={len(user_text)} chars, spatial_info={len(spatial_info)} chars")

            # Translate sketch to prompt with full spatial awareness
            vision_prompt, metadata = await translator.translate_sketch_to_prompt(
                image_data=image_data,
                image_format=image_format,
                user_context=combined_context if combined_context else None
            )
            
            logger.info(f"Vision translation complete: {metadata}")
            
            # Merge vision output with user text
            if user_text:
                combined_request = f"{user_text}\n\n[From sketch analysis]:\n{vision_prompt}"
            else:
                combined_request = vision_prompt
            
            # Replace user_request with combined prompt
            request = dict(request)  # Make mutable copy
            request["user_request"] = combined_request
            
            # Send success log to UI
            await _send_message_safe(manager, session_id, {
                "type": "activity_log",
                "message": f"✅ Sketch analyzed! Extracted: {metadata.get('prompt_length', 0)} chars of CAD instructions",
                "level": "success"
            })
            
            logger.debug(f"Combined request preview: {combined_request[:300]}...")
            
        except Exception as e:
            logger.error(f"Vision translation failed: {e}", exc_info=True)
            await _send_message_safe(manager, session_id, {
                "type": "activity_log",
                "message": f"⚠️ Failed to analyze sketch: {str(e)}. Continuing with text only.",
                "level": "warning"
            })
            # Continue execution with original text (don't fail the whole request)
    # ========== END VISION TRANSLATION ==========

    # Initialize session logging
    session_path = initialize_session(session_id)
    if session_path:
        logger.info(f"Session logging enabled for session {session_id}: {session_path}")
    else:
        logger.warning(f"Session logging disabled for session {session_id}")

    # Get existing conversation history or start new
    history = manager.get_conversation_history(session_id)
    messages: List[Dict[str, Any]] = _prune_empty_messages(history)
    if len(messages) != len(history):
        logger.debug(
            "Session %s pruned %d empty message(s) from conversation history before calling Claude",
            session_id,
            len(history) - len(messages),
        )

    # Capture checkpoint BEFORE appending the new user message
    # This captures the timeline state before this request is executed
    message_id = _capture_checkpoint(session_id, request, manager, len(messages))

    # Notify frontend of the checkpoint message_id so it can associate it with the UI message
    # Include request_id if provided for exact request-checkpoint correlation
    checkpoint_msg = {
        "type": "checkpoint_created",
        "message_id": message_id,
    }
    if "request_id" in request:
        checkpoint_msg["request_id"] = request["request_id"]

    await _send_message_safe(manager, session_id, checkpoint_msg)

    # Append new user request with current timeline state
    message_context = dict(request)

    # Request-start store guard:
    # - With entity_context: clear active refs and repopulate from request payload.
    # - Without entity_context: preserve existing refs if already populated to avoid
    #   wiping usable context on short follow-up turns.
    store = _get_entity_store(session_id, manager)
    entity_context = request.get("entity_context")
    if entity_context:
        store.soft_clear()
        await _prepopulate_entity_store(session_id, manager, entity_context)
        _manager_set_latest_entity_context(manager, session_id, dict(entity_context))
        # Prune stale tokens from persistent cache (entities no longer in model)
        current_tokens = _extract_tokens_from_context(entity_context)
        store.prune_stale_tokens(current_tokens)
    elif store.is_empty():
        _manager_clear_latest_entity_context(manager, session_id)
        logger.debug("Session %s has no entity_context and empty store at request start.", session_id)
    else:
        logger.debug(
            "Session %s has no entity_context at request start; preserving %d active refs.",
            session_id,
            len(store.get_all_faces()) + len(store.get_all_edges()) + len(store.get_all_bodies()),
        )

    timeline_state = message_context.get("timeline_state")
    feature_snapshot = await _ensure_feature_snapshot(session_id, manager, timeline_state)
    if feature_snapshot is not None:
        message_context["feature_snapshot"] = feature_snapshot

    new_message = _build_user_message(message_context)
    messages.append(new_message)

    last_user_message_sent = _extract_last_user_notification(messages)

    logger.info("Starting execution loop for session %s with max %d iterations (history: %d messages)",
                session_id, max_iterations, len(messages))

    try:
        await _execute_workflow_loop(
            session_id,
            messages,
            max_iterations,
            model_name,
            manager,
            last_user_message_sent,
            session_path=session_path,
            request=request,
            feature_snapshot=feature_snapshot
        )
    except asyncio.CancelledError as e:
        reason = ""
        if e.args:
            reason = str(e.args[0])
        if "superseded" in reason:
            cancel_msg = "Request superseded by new request"
        elif "user_cancel" in reason:
            cancel_msg = "Request cancelled by user"
        else:
            cancel_msg = "Request cancelled"
        logger.info("%s for session %s", cancel_msg, session_id)
        await _send_message_safe(manager, session_id, {"type": "cancelled", "message": cancel_msg})
        # DO NOT save conversation history when cancelled - the revert handler has already
        # managed the conversation state. Saving here would overwrite the trimmed history.
        raise


async def handle_studio_export_request(
    session_id: str,
    request: Mapping[str, Any],
    manager: ConnectionManager,
) -> None:
    """Export the latest build123d session model through the unified backend flow."""
    export_format = str(request.get("format") or "step").strip().lower()
    if export_format != "step":
        await _send_error(
            manager,
            session_id,
            "Unsupported export format",
            "Studio export currently supports STEP only.",
        )
        return

    suggested_path = request.get("output_path")
    if suggested_path:
        output_path = str(suggested_path)
    else:
        export_dir = Path(__file__).parent.parent / "runs" / "studio_exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = str(export_dir / f"{session_id}_{timestamp}.step")

    try:
        exported_path = _BUILD123D_EXECUTOR.export_step(session_id, output_path)
    except Exception as exc:
        await _send_error(manager, session_id, "Studio export failed", str(exc))
        return

    await _send_message_safe(
        manager,
        session_id,
        {
            "type": "studio_export_complete",
            "format": "step",
            "path": exported_path,
        },
    )


async def _execute_workflow_loop(
    session_id: str,
    messages: List[Dict[str, Any]],
    max_iterations: int,
    model_name: Optional[str],
    manager: ConnectionManager,
    last_user_message_sent: Optional[str],
    session_path: Optional[str] = None,
    request: Optional[Mapping[str, Any]] = None,
    feature_snapshot: Optional[Dict[str, Any]] = None
) -> None:
    """Execute the main workflow loop with proper cancellation support."""

    # Get user token for usage tracking
    user_token = manager.get_user_token(session_id)
    llm_api_keys = manager.get_llm_api_keys(session_id)
    try:
        execution_target = _resolve_execution_target(request)
    except UnsupportedExecutionTargetError as exc:
        await _send_error(manager, session_id, "Invalid execution target", str(exc))
        return
    fusion_ir_executor = FusionTargetExecutor(manager, timeout_seconds=EXECUTION_TIMEOUT)
    ir_doc_state = _restore_or_create_ir_document_state(
        manager,
        session_id,
        execution_target=execution_target,
    )
    # Track all attempted IR operations (success + failure) so dependency mapping
    # can fail closed when an operation chain breaks mid-turn.
    ir_attempt_history: List[Any] = []

    # Hard fail if unauthenticated to prevent provider calls without quota enforcement
    if not user_token and not AUTH_BYPASS:
        raise ValueError("Unauthenticated session: user token is required before executing workflow")

    # Initialize API call counter for session logging
    api_call_counter = {'count': 1}

    # Perform routing on first iteration if enabled
    routed_tools = None
    routed_system_prompt = None
    routing_result = None

    if USE_PROMPT_ROUTING and messages:
        try:
            # Extract user request from last user message for routing
            last_user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if last_user_msg:
                user_request = _extract_user_request_for_routing(last_user_msg)
                if user_request:
                    logger.info(f"Routing request for session {session_id}...")
                    routing_result = await route_request(
                        user_request,
                        messages,
                        build_plan=manager.get_active_build_plan(session_id),
                        api_keys=llm_api_keys,
                    )

                    # Log routing decision
                    logger.info(get_routing_summary(routing_result))

                    # Build prompt from routing
                    routed_system_prompt, routed_tools = build_prompt(routing_result)

                    # Log build summary
                    savings = estimate_token_savings(routing_result)
                    logger.info(
                        f"Prompt optimized: {len(routed_tools)} tools loaded "
                        f"({savings['percent_saved']}% reduction, ~{savings['estimated_tokens_saved']} tokens saved)"
                    )
        except Exception as e:
            logger.error(f"Routing failed, falling back to full prompt: {e}", exc_info=True)
            # Fall back to default (all tools) if routing fails
            routed_tools = None
            routed_system_prompt = None

    # Track consecutive failures for the same intent across iterations.
    consecutive_tool_failures = 0
    last_failed_intent_key: Optional[str] = None

    # Track duplicate operation intents for this request loop.
    seen_guarded_intents: Set[str] = set()

    # Reasoning context management - accumulate reasoning across iterations
    reasoning_buffer: List[str] = []
    iteration_tool_calls: List[str] = []
    iteration_tool_results: List[Dict[str, Any]] = []
    deterministic_fallback_used = False

    for iteration in range(max_iterations):
        logger.debug("Iteration %d for session %s", iteration + 1, session_id)

        # Track if any tool fails in this iteration
        iteration_had_failure = False
        iteration_had_success = False
        iteration_first_failure_intent: Optional[str] = None
        iteration_force_stop = False
        topology_mutation_executed = False
        face_sketches_created_this_turn: Set[str] = set()

        # Clear per-iteration reasoning accumulator
        reasoning_buffer.clear()
        iteration_tool_calls.clear()
        iteration_tool_results.clear()

        # Use user-provided reasoning effort from UI selector
        # Anthropic "on" → map to "medium" budget; "off" → disable reasoning
        _re_raw = request.get("reasoning_effort") if request else None
        if _re_raw == "on":
            reasoning_effort = "medium"
        elif _re_raw in (None, "off"):
            reasoning_effort = None
        else:
            reasoning_effort = _re_raw  # pass through: low/medium/high/xhigh

        feature_snapshot = _manager_get_feature_snapshot(manager, session_id) or feature_snapshot

        runtime_entity_context: Optional[Dict[str, Any]] = _manager_get_latest_entity_context(manager, session_id)
        try:
            synced_context = await _ensure_runtime_entity_context_synced(
                session_id,
                manager,
                reason=f"iteration_{iteration + 1}_pre_llm",
            )
            if synced_context is not None:
                runtime_entity_context = synced_context
        except EntityRefreshError as exc:
            error_text = str(exc)
            await _send_error(manager, session_id, "Runtime state sync failed", error_text)
            messages.append(_user_text_message(f"Runtime state sync failed: {error_text}"))
            manager.set_conversation_history(session_id, messages)
            logger.error(
                "Session %s: CRITICAL runtime sync failure before LLM call at iteration %d: %s",
                session_id,
                iteration + 1,
                error_text,
            )
            return

        # Build session context for logging
        session_context = None
        if session_path and request:
            # Extract entity store data using proper accessor methods
            entity_store = manager.get_entity_store(session_id)
            entity_store_data = {
                "selected_edges": entity_store.get_all_edges(),
                "selected_faces": entity_store.get_all_faces(),
                "selected_bodies": entity_store.get_all_bodies(),
            }

            # Get tool names from routed_tools or all tools
            loaded_tools = None
            if routed_tools:
                loaded_tools = [tool.get("name") for tool in routed_tools if "name" in tool]

            # Use latest runtime context for logging, falling back to request context.
            entity_context_for_logging = runtime_entity_context or request.get("entity_context") or {}
            spatial_context_data = entity_context_for_logging.get("spatial_context")

            # Build context using session_logger helper
            session_context = _extract_session_context(
                user_request=request.get("user_request"),
                timeline_state=request.get("timeline_state"),
                messages=messages,
                feature_snapshot=feature_snapshot,
                entity_store_data=entity_store_data,
                routing_result=routing_result,
                loaded_tools=loaded_tools,
                model_name=model_name,
                spatial_context=spatial_context_data,
                entity_context=entity_context_for_logging,
                reasoning_effort=reasoning_effort,
                iteration=iteration + 1,
                max_iterations=max_iterations,
                capture_phase="pre_llm",
                system_prompt=routed_system_prompt,  # Actual prompt sent to LLM
                tools=routed_tools,  # Actual tools sent to LLM
            )

        async def _on_reasoning_delta(delta: str) -> None:
            """Forward streamed reasoning text to the UI and accumulate for context."""
            if not delta or not delta.strip():
                return
            # Accumulate reasoning for context injection into next iteration
            reasoning_buffer.append(delta)
            await _send_message_safe(
                manager,
                session_id,
                {"type": "reasoning_chunk", "content": delta},
            )

        # Inject reasoning context from prior iterations into system prompt
        effective_system_prompt = routed_system_prompt
        reasoning_ctx = manager.get_reasoning_context(session_id)
        if reasoning_ctx and reasoning_ctx.entries:
            reasoning_injection = reasoning_ctx.get_injection_text()
            if reasoning_injection and effective_system_prompt:
                # Append reasoning context section to system prompt
                effective_system_prompt = (
                    effective_system_prompt + "\n\n" +
                    REASONING_CONTEXT_SECTION.format(reasoning_history=reasoning_injection)
                )
                logger.debug(
                    "Injected reasoning context (%d chars) for session %s iteration %d",
                    len(reasoning_injection), session_id, iteration + 1
                )

        # Apply observation masking to reduce context size for older tool outputs
        # This preserves reasoning and actions while compressing verbose results.
        effective_messages = mask_old_observations(messages, keep_recent=6, max_result_chars=800)
        current_ref_table = _format_current_ref_table(runtime_entity_context or {})
        if current_ref_table:
            # Keep the latest valid refs in the immediate LLM context even after older
            # verbose design_entities blocks are masked out of the conversation.
            effective_messages = [
                *effective_messages,
                _user_text_message(current_ref_table),
            ]

        try:
            response = await call_claude_with_tools(
                effective_messages,
                tools=routed_tools,  # Use routed tools if available
                system_prompt=effective_system_prompt,  # Use routed prompt with reasoning context
                model_name=model_name,
                reasoning_effort=reasoning_effort,
                session_id=session_id,
                iteration=iteration + 1,
                session_path=session_path,
                api_call_counter=api_call_counter,
                session_context=session_context,
                reasoning_callback=_on_reasoning_delta,
                user_token=user_token,  # Enable usage tracking via Supabase
                api_keys=llm_api_keys,
            )
        except ValueError as e:
            # Auth failures from the gateway (e.g., invalid/expired token) should surface
            # to the UI as auth_error so the palette can prompt re-login immediately.
            err_text = str(e).lower()
            if "auth" in err_text and "fail" in err_text:
                logger.warning("Authentication to usage gateway failed for session %s: %s", session_id, e)
                await _send_message_safe(
                    manager,
                    session_id,
                    {"type": "auth_error", "message": "Authentication failed - please sign in again"},
                )
                return
            if _is_missing_provider_key_error(e):
                fallback_calls = _deterministic_mvp_tool_calls(_extract_latest_user_text(messages))
                if fallback_calls:
                    if deterministic_fallback_used:
                        response = {
                            "stop_reason": "end_turn",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Deterministic fallback execution complete.",
                                }
                            ],
                        }
                    else:
                        deterministic_fallback_used = True
                        logger.warning(
                            "Session %s using deterministic MVP fallback due to missing provider key: %s",
                            session_id,
                            e,
                        )
                        await _send_message_safe(
                            manager,
                            session_id,
                            {
                                "type": "runtime_fallback",
                                "message": "Provider key missing. Using deterministic MVP fallback planner for this request.",
                            },
                        )
                        response = {
                            "stop_reason": "tool_use",
                            "content": [
                                {"type": "text", "text": "Deterministic fallback plan generated for MVP request."},
                                *[
                                    {
                                        "type": "tool_use",
                                        "id": tc["id"],
                                        "name": tc["name"],
                                        "input": tc["input"],
                                    }
                                    for tc in fallback_calls
                                ],
                            ],
                        }
                else:
                    raise
            else:
                raise
        assistant_content = response.get("content") or []
        if not _message_has_content({"role": "assistant", "content": assistant_content}):
            fallback_text = _final_response_text(response) or "Execution complete."
            assistant_content = [{"type": "text", "text": fallback_text}]

        assistant_message = {"role": "assistant", "content": assistant_content}
        messages.append(assistant_message)

        assistant_text_blocks = [
            str(block.get("text", "")).strip()
            for block in assistant_content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        assistant_text = "\n".join(filter(None, assistant_text_blocks))

        # Capture reasoning from this iteration into persistent context
        # This happens after each LLM call, regardless of stop_reason
        if reasoning_buffer:
            full_reasoning = "".join(reasoning_buffer)
            # Extract tool call names from response for this iteration
            tool_calls_response = extract_tool_calls(response)
            current_tool_names = [tc.get("name", "") for tc in tool_calls_response] if tool_calls_response else []

            try:
                reasoning_entry = ReasoningEntry(
                    iteration=iteration + 1,
                    timestamp=datetime.now(timezone.utc),
                    raw_thinking=full_reasoning,
                    summary=summarize_reasoning(full_reasoning, max_chars=400),
                    decisions=extract_decisions(full_reasoning, max_decisions=3),
                    tool_calls=current_tool_names,
                    outcome=None,  # Will be updated after tool execution
                )
                reasoning_ctx = manager.get_reasoning_context(session_id)
                reasoning_ctx.add_entry(reasoning_entry)
                logger.debug(
                    "Captured reasoning for session %s iteration %d: %d chars, %d decisions",
                    session_id, iteration + 1, len(full_reasoning), len(reasoning_entry.decisions)
                )
            except Exception as e:
                logger.warning("Failed to capture reasoning context: %s", e)

        stop_reason = response.get("stop_reason")
        if stop_reason == "end_turn":
            # Send any text content on end_turn (completion message, final status, etc.)
            # This includes both first-iteration responses AND final completion messages after tool execution
            if assistant_text:
                await _send_message_safe(
                    manager,
                    session_id,
                    {"type": "llm_message", "message": markdown_to_html(assistant_text), "format": "html"},
                )
                logger.info("Session %s sent direct text response (%d chars) on end_turn", session_id, len(assistant_text))

            # Save conversation history for next request
            manager.set_conversation_history(session_id, messages)

            # Finalize cache metrics summary for this session
            if session_path:
                try:
                    from .session_logger import finalize_session_cache_summary
                    cache_summary = finalize_session_cache_summary(session_path)
                    if cache_summary:
                        logger.info(
                            "Session %s cache summary: %d calls, hit_rate=%.1f%%, savings=$%.4f (%.1f%%)",
                            session_id,
                            cache_summary.get("total_api_calls", 0),
                            cache_summary.get("overall_cache_hit_rate_pct", 0),
                            cache_summary.get("total_savings_usd", 0),
                            cache_summary.get("overall_savings_pct", 0),
                        )
                except Exception as e:
                    logger.warning("Failed to finalize cache summary: %s", e)

            # Notify frontend that execution has completed successfully
            await _send_message_safe(manager, session_id, {
                "type": "completed",
                "message": "Request completed successfully"
            })

            logger.info("Session %s completed execution loop in %d iterations", session_id, iteration + 1)
            return  # Exit the loop successfully

        if stop_reason != "tool_use":
            error_text = f"Unexpected stop_reason '{stop_reason}'. Expected 'tool_use' or 'end_turn'."
            await _send_error(manager, session_id, "Invalid stop reason from Claude", error_text)
            messages.append(_user_text_message(error_text))
            logger.warning("Session %s received unexpected stop_reason: %s", session_id, stop_reason)
            continue

        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            error_text = "Claude returned stop_reason='tool_use' but no tool calls were found."
            await _send_error(manager, session_id, "Invalid tool usage", error_text)
            messages.append(_user_text_message(error_text))
            logger.warning("Session %s received tool_use stop with zero tool calls", session_id)
            continue

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_use_id = tool_call["id"]

            raw_tool_input = tool_call.get("input") or {}
            if isinstance(raw_tool_input, Mapping):
                tool_input = dict(raw_tool_input)
            else:
                tool_input = {}

            description = str(tool_input.get("description", "") or "").strip()
            if not description and assistant_text:
                description = assistant_text
            tool_intent_key = _build_tool_intent_key(tool_name, tool_input)
            logger.info("Session %s executing tool '%s' (id=%s)", session_id, tool_name, tool_use_id)

            defer_text = _maybe_defer_face_sketch_followup(
                tool_name,
                tool_input,
                face_sketches_created_this_turn,
            )
            if defer_text:
                messages.append(_tool_result_message(tool_use_id, defer_text))
                logger.warning(
                    "Session %s deferred face-sketch follow-up '%s' in iteration %d",
                    session_id,
                    tool_name,
                    iteration + 1,
                )
                continue

            if tool_name in IR_MVP_TOOLS:
                ir_tool_call: Mapping[str, Any] = tool_call
                resolved_ir_input: Optional[Dict[str, Any]] = None
                if (
                    execution_target == "fusion"
                    and tool_name == "create_sketch"
                    and callable(getattr(manager, "get_entity_store", None))
                ):
                    try:
                        resolved_ir_input = _resolve_codegen_entity_refs(
                            session_id,
                            manager,
                            tool_name,
                            dict(tool_input),
                        )
                    except SelectionToolCallError as exc:
                        error_text = f"IR reference resolution failed: {exc}"
                        await _send_error(manager, session_id, "IR validation failed", error_text)
                        messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                        iteration_had_failure = True
                        if iteration_first_failure_intent is None:
                            iteration_first_failure_intent = tool_intent_key
                        continue

                    original_plane = str(tool_input.get("plane_id", "") or "").strip()
                    resolved_plane = str(resolved_ir_input.get("plane_id", "") or "").strip()
                    if original_plane and resolved_plane and original_plane != resolved_plane:
                        logger.info(
                            "Session %s resolved IR create_sketch plane '%s' -> token for Fusion execution",
                            session_id,
                            original_plane,
                        )
                    ir_tool_call = dict(tool_call)
                    ir_tool_call["input"] = resolved_ir_input

                if tool_name in {"add_rectangle", "add_circle"}:
                    sketch_preflight_error = _preflight_face_sketch_uv_bounds(
                        session_id,
                        manager,
                        tool_name=tool_name,
                        tool_input=tool_input,
                    )
                    if sketch_preflight_error:
                        await _send_error(manager, session_id, "Face sketch bounds preflight failed", sketch_preflight_error)
                        messages.append(_tool_result_message(tool_use_id, sketch_preflight_error, is_error=True))
                        iteration_had_failure = True
                        if iteration_first_failure_intent is None:
                            iteration_first_failure_intent = _build_failure_intent_key(
                                tool_name,
                                tool_input,
                                sketch_preflight_error,
                            )
                        logger.warning(
                            "Session %s blocked '%s' by face-sketch bounds preflight: %s",
                            session_id,
                            tool_name,
                            sketch_preflight_error,
                        )
                        continue

                ir_metadata = {
                    "source": "studio" if execution_target == "build123d" else "fusion",
                    "request_id": str((request or {}).get("request_id") or ""),
                    "iteration": iteration + 1,
                }
                try:
                    ir_op = map_tool_call_to_ir(
                        ir_tool_call,
                        ir_doc_state,
                        metadata=ir_metadata,
                        dependency_operations=[*ir_doc_state.operations, *ir_attempt_history],
                    )
                except UnsupportedToolMappingError as exc:
                    error_text = str(exc)
                    await _send_error(manager, session_id, "IR mapping failed", error_text)
                    messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                    iteration_had_failure = True
                    if iteration_first_failure_intent is None:
                        iteration_first_failure_intent = tool_intent_key
                    continue

                ir_attempt_history.append(ir_op)
                validation_errors = validate_ir_candidate(ir_op, ir_doc_state.operations)
                if validation_errors:
                    has_dependency_block = any(
                        "depends on uncommitted operation" in err for err in validation_errors
                    )
                    title = "IR dependency blocked" if has_dependency_block else "IR validation failed"
                    prefix = "IR dependency blocked: " if has_dependency_block else "IR validation failed: "
                    error_text = prefix + "; ".join(validation_errors)
                    await _send_error(manager, session_id, title, error_text)
                    messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                    iteration_had_failure = True
                    if iteration_first_failure_intent is None:
                        iteration_first_failure_intent = tool_intent_key
                    continue

                await _send_message_safe(
                    manager,
                    session_id,
                    {
                        "type": "ir_operation",
                        "target": execution_target,
                        "operation": asdict(ir_op),
                    },
                )

                if execution_target == "build123d":
                    candidate_document = IRDocument(
                        version=ir_doc_state.version,
                        units="mm",
                        operations=[*ir_doc_state.operations, ir_op],
                        metadata=dict(ir_doc_state.metadata) if ir_doc_state.metadata else None,
                    )
                    target_result = await _BUILD123D_EXECUTOR.execute_document(
                        session_id,
                        candidate_document,
                        request_id=str((request or {}).get("request_id") or f"iter_{iteration+1}"),
                    )
                else:
                    target_result = await fusion_ir_executor.execute_operation(
                        session_id,
                        ir_op,
                        tool_use_id=tool_use_id,
                        description=description,
                    )
                raw_result_payload: Dict[str, Any] = {}
                if isinstance(target_result.raw_result, Mapping):
                    raw_result_payload = dict(target_result.raw_result)

                if target_result.success and tool_name in {"create_sketch", "add_line", "add_arc", "add_circle", "add_rectangle"}:
                    executed_input = target_result.data.get("tool_input") if isinstance(target_result.data, Mapping) else {}
                    plane_fallback = ""
                    if tool_name == "create_sketch":
                        plane_fallback = str(
                            (
                                (executed_input or {}).get("plane_id")
                                or (resolved_ir_input or {}).get("plane_id")
                                or tool_input.get("plane_id")
                                or ""
                            )
                        ).strip()
                    sketch_fallback = str(
                        (
                            (executed_input or {}).get("sketch_id")
                            or tool_input.get("sketch_id")
                            or ""
                        )
                    ).strip()
                    if sketch_fallback and "sketch_id" not in raw_result_payload:
                        raw_result_payload["sketch_id"] = sketch_fallback
                    if plane_fallback and "plane_id_input" not in raw_result_payload:
                        raw_result_payload["plane_id_input"] = plane_fallback

                    if raw_result_payload:
                        try:
                            _register_sketch_result_entities(
                                session_id,
                                manager,
                                tool_name,
                                raw_result_payload,
                                fallback_plane_id=plane_fallback or None,
                            )
                        except Exception as exc:
                            logger.warning("Sketch metadata registration failed for IR %s: %s", tool_name, exc)

                        if tool_name == "create_sketch":
                            sketch_id = str(raw_result_payload.get("sketch_id") or "").strip()
                            if sketch_id:
                                sketch_metadata = _get_sketch_entity_store(session_id, manager).get_sketch_metadata(sketch_id)
                                if sketch_metadata.get("plane_kind") == "face":
                                    face_sketches_created_this_turn.add(sketch_id)

                result_text = target_result.message
                if raw_result_payload:
                    _summary_success, _summary_text = _summarise_execution_result(
                        tool_name,
                        raw_result_payload,
                    )
                    if _summary_success == target_result.success or target_result.success:
                        result_text = _summary_text

                messages.append(_tool_result_message(tool_use_id, result_text, is_error=not target_result.success))

                if not target_result.success:
                    iteration_had_failure = True
                    if iteration_first_failure_intent is None:
                        failure_detail = result_text
                        if raw_result_payload:
                            raw_error = raw_result_payload.get("error") or raw_result_payload.get("message")
                            if raw_error:
                                failure_detail = f"{failure_detail} | {raw_error}"
                        iteration_first_failure_intent = _build_failure_intent_key(
                            tool_name,
                            tool_input,
                            failure_detail,
                        )
                    await _send_error(manager, session_id, f"{execution_target} execution failed", result_text)
                    logger.error(
                        "Session %s IR tool '%s' failed on target %s: %s",
                        session_id,
                        tool_name,
                        execution_target,
                        result_text,
                    )
                else:
                    # Commit operation only after successful target execution.
                    ir_doc_state.append(ir_op)
                    _manager_save_ir_document_state(manager, session_id, ir_doc_state)
                    logger.info(
                        "Session %s IR tool '%s' executed on target %s",
                        session_id,
                        tool_name,
                        execution_target,
                    )
                    if execution_target == "build123d":
                        iteration_had_success = True
                        await _send_message_safe(
                            manager,
                            session_id,
                            {
                                "type": "studio_geometry_update",
                                "target": "build123d",
                                "result": target_result.data,
                            },
                        )
                    else:
                        refresh_result: Dict[str, Any] = {}
                        if raw_result_payload:
                            refresh_result = dict(raw_result_payload)
                        if "success" not in refresh_result:
                            refresh_result["success"] = target_result.success
                        if "message" not in refresh_result and result_text:
                            refresh_result["message"] = result_text
                        if tool_name in {"create_sketch", "add_rectangle", "add_circle"}:
                            refresh_result.setdefault("no_op", True)

                        try:
                            await _refresh_and_enrich_after_success(
                                session_id,
                                manager,
                                tool_name,
                                refresh_result,
                                messages,
                            )
                        except EntityRefreshError as exc:
                            error_text = str(exc)
                            await _send_error(
                                manager,
                                session_id,
                                "Entity refresh failed",
                                error_text,
                            )
                            messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                            iteration_had_failure = True
                            if iteration_first_failure_intent is None:
                                iteration_first_failure_intent = tool_intent_key
                            manager.set_conversation_history(session_id, messages)
                            logger.error(
                                "Session %s: CRITICAL - Entity refresh failed after IR %s, stopping workflow: %s",
                                session_id,
                                tool_name,
                                error_text,
                            )
                            iteration_force_stop = True
                            break
                        iteration_had_success = True
                        await _capture_operation_checkpoint(
                            session_id,
                            manager,
                            request=request,
                            tool_name=tool_name,
                            tool_use_id=tool_use_id,
                            description=description,
                            messages=messages,
                            ir_doc_state=ir_doc_state,
                            force_snapshot_refresh=tool_name not in REFRESH_ON_SUCCESS_TOOLS,
                        )
                continue

            if tool_name in TOPOLOGY_MUTATING_TOOLS:
                if topology_mutation_executed:
                    defer_text = (
                        f"Deferred '{tool_name}' in this turn. "
                        "Run at most one topology-changing operation per turn, then wait for refreshed entities."
                    )
                    messages.append(_tool_result_message(tool_use_id, defer_text))
                    logger.warning(
                        "Session %s deferred extra topology mutation '%s' in iteration %d",
                        session_id,
                        tool_name,
                        iteration + 1,
                    )
                    continue
                topology_mutation_executed = True

            if tool_name in DUPLICATE_INTENT_GUARD_TOOLS:
                if tool_intent_key in seen_guarded_intents:
                    duplicate_text = (
                        f"Skipped duplicate {tool_name} call with identical parameters. "
                        "Change placement/ref parameters before retrying."
                    )
                    messages.append(_tool_result_message(tool_use_id, duplicate_text))
                    logger.warning("Session %s suppressed duplicate intent for '%s'", session_id, tool_name)
                    continue

            if tool_name == "respond_to_user":
                message_text = str(tool_input.get("message", "")).strip()
                if not message_text:
                    message_text = "Tool 'respond_to_user' invoked without message content."

                normalized_message = _normalise_user_notification(message_text)
                if last_user_message_sent and normalized_message == last_user_message_sent:
                    duplicate_notice = (
                        "Duplicate respond_to_user output suppressed. The user already received this exact wording. "
                        "Revise the message or wait for the user to respond."
                    )
                    messages.append(_tool_result_message(tool_use_id, duplicate_notice, is_error=True))
                    iteration_had_failure = True
                    if iteration_first_failure_intent is None:
                        iteration_first_failure_intent = tool_intent_key
                    logger.warning(
                        "Session %s suppressed duplicate respond_to_user message: %s",
                        session_id,
                        message_text,
                    )
                    continue

                await _send_message_safe(
                    manager,
                    session_id,
                    {"type": "llm_message", "message": markdown_to_html(message_text), "format": "html"},
                )
                confirmation_text = f"Delivered message to user: {message_text}"
                messages.append(_tool_result_message(tool_use_id, confirmation_text))
                last_user_message_sent = normalized_message
                iteration_had_success = True
                logger.debug("Session %s forwarded respond_to_user message", session_id)
                continue

            # Handle design exploration tools
            if tool_name == "generate_question_tree":
                # Send question tree to UI for user interaction
                question_tree_data = {
                    "type": "question_tree_generated",
                    "data": {
                        "problem_summary": tool_input.get("problem_summary", ""),
                        "known_constraints": tool_input.get("known_constraints", []),
                        "questions": tool_input.get("questions", [])
                    }
                }
                await _send_message_safe(manager, session_id, question_tree_data)
                logger.info("Session %s sent question tree to UI", session_id)

                # Wait for user to complete the question tree
                try:
                    result = await manager.wait_for_fusion_result(session_id, timeout=300)  # 5 minute timeout
                    if result.get("type") == "question_tree_completed":
                        answers = result.get("answers", {})
                        # Format answers for the LLM
                        answer_summary = []
                        for q_id, answer in answers.items():
                            value = answer.get("value", "")
                            text_input = answer.get("text_input", "")
                            if text_input:
                                answer_summary.append(f"- {q_id}: {value} (user specified: {text_input})")
                            else:
                                answer_summary.append(f"- {q_id}: {value}")
                        result_text = f"User completed the question tree. Answers:\n" + "\n".join(answer_summary)
                        messages.append(_tool_result_message(tool_use_id, result_text))
                        iteration_had_success = True
                        logger.info("Session %s received question tree answers", session_id)
                    else:
                        error_text = "User did not complete the question tree."
                        messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                        iteration_had_failure = True
                        if iteration_first_failure_intent is None:
                            iteration_first_failure_intent = tool_intent_key
                except asyncio.TimeoutError:
                    error_text = "Timed out waiting for user to complete the question tree."
                    messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                    iteration_had_failure = True
                    if iteration_first_failure_intent is None:
                        iteration_first_failure_intent = tool_intent_key
                    logger.warning("Session %s timed out waiting for question tree completion", session_id)
                continue

            if tool_name == "propose_designs":
                # Send design proposals to UI for user selection
                designs_data = {
                    "type": "designs_proposed",
                    "data": {
                        "context_summary": tool_input.get("context_summary", ""),
                        "designs": tool_input.get("designs", []),
                        "recommendation": tool_input.get("recommendation")
                    }
                }
                await _send_message_safe(manager, session_id, designs_data)
                logger.info("Session %s sent design proposals to UI", session_id)

                # Wait for user to select a design
                try:
                    result = await manager.wait_for_fusion_result(session_id, timeout=300)  # 5 minute timeout
                    if result.get("type") == "design_selected":
                        selected_id = result.get("design_id", "")
                        # Find the selected design's specifications
                        selected_design = None
                        for design in tool_input.get("designs", []):
                            if design.get("id") == selected_id:
                                selected_design = design
                                break

                        if selected_design:
                            specs = selected_design.get("specifications", {})
                            specs_text = "\n".join([f"  - {k}: {v}" for k, v in specs.items()])
                            result_text = (
                                f"User selected design: {selected_design.get('name', selected_id)}\n"
                                f"Description: {selected_design.get('description', '')}\n"
                                f"Specifications:\n{specs_text}\n\n"
                                f"IMPORTANT: Before building, use the `output_build_plan` tool to:\n"
                                f"1. Generate a step-by-step build plan based on these specifications\n"
                                f"2. Show the user what operations you will perform\n"
                                f"3. Then execute each step in order\n\n"
                                f"The build plan should list each CAD operation with its key parameters."
                            )
                        else:
                            result_text = (
                                f"User selected design with ID: {selected_id}.\n\n"
                                f"Use the `output_build_plan` tool to generate a step-by-step build plan, "
                                f"then execute it."
                            )

                        messages.append(_tool_result_message(tool_use_id, result_text))
                        iteration_had_success = True
                        logger.info("Session %s user selected design: %s", session_id, selected_id)
                    else:
                        error_text = "User did not select a design."
                        messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                        iteration_had_failure = True
                        if iteration_first_failure_intent is None:
                            iteration_first_failure_intent = tool_intent_key
                except asyncio.TimeoutError:
                    error_text = "Timed out waiting for user to select a design."
                    messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                    iteration_had_failure = True
                    if iteration_first_failure_intent is None:
                        iteration_first_failure_intent = tool_intent_key
                    logger.warning("Session %s timed out waiting for design selection", session_id)
                continue

            if tool_name == "output_build_plan":
                # Store the build plan in session state and send to UI
                design_name = tool_input.get("design_name", "Design")
                steps = tool_input.get("steps", [])

                # Store plan in manager for context injection
                build_plan = {
                    "design_name": design_name,
                    "steps": steps,
                    "completed_steps": 0
                }
                manager.set_active_build_plan(session_id, build_plan)

                # Send build plan to UI for display
                build_plan_data = {
                    "type": "build_plan_generated",
                    "data": {
                        "design_name": design_name,
                        "steps": steps,
                        "total_steps": len(steps)
                    }
                }
                await _send_message_safe(manager, session_id, build_plan_data)
                logger.info("Session %s sent build plan to UI: %s with %d steps", session_id, design_name, len(steps))

                # Re-route with build plan context to load required tools
                if USE_PROMPT_ROUTING:
                    try:
                        logger.info(f"Re-routing with build plan context: {design_name} ({len(steps)} steps)")

                        # Create a request summary from the build plan
                        operations = set(s.get("operation", "") for s in steps if s.get("operation"))
                        plan_summary = f"Build plan: {design_name} with operations including: {', '.join(operations)}"

                        # Re-route with build plan information
                        routing_result = await route_request(
                            user_request=plan_summary,
                            conversation_history=messages,
                            build_plan=build_plan,
                            api_keys=llm_api_keys,
                        )

                        # Rebuild prompt with new routing
                        routed_system_prompt, routed_tools = build_prompt(routing_result)

                        # Log the re-routing decision
                        logger.info(get_routing_summary(routing_result))
                        logger.info(
                            f"Re-routing complete: {len(routed_tools)} tools now loaded "
                            f"(added missing tools for build plan execution)"
                        )
                    except Exception as e:
                        logger.error(f"Re-routing after build plan failed: {e}", exc_info=True)
                        # Continue with existing tools if re-routing fails

                # Format step summary for LLM confirmation
                step_summary = "\n".join([
                    f"  {s.get('step_number', i+1)}. {s.get('description', s.get('operation', 'Step'))}"
                    for i, s in enumerate(steps)
                ])
                result_text = (
                    f"Build plan for '{design_name}' is now active and displayed to the user.\n\n"
                    f"Steps:\n{step_summary}\n\n"
                    f"Execute these steps in order. After each CAD operation completes, "
                    f"you will see the plan progress in the tool result."
                )
                messages.append(_tool_result_message(tool_use_id, result_text))
                iteration_had_success = True
                continue

            if tool_name in GEOMETRY_OPERATION_TOOLS:
                # =============================================================
                # ENTITY CONTEXT VALIDATION GATE (Spatial Validation Block - Rule 26)
                # Before executing tools that require face/edge/body refs, validate
                # that the entity store is populated. This prevents "blind" operations.
                # =============================================================
                if tool_name in TOOLS_REQUIRING_ENTITY_CONTEXT:
                    is_valid, validation_warning = _validate_entity_store_for_tool(
                        session_id, manager, tool_name
                    )
                    if not is_valid:
                        logger.warning(
                            "Session %s entity validation failed for '%s'; attempting one-time context recovery: %s",
                            session_id,
                            tool_name,
                            validation_warning,
                        )
                        try:
                            store = _get_entity_store(session_id, manager)
                            fresh_context = await _refresh_entity_context_with_retry(
                                session_id,
                                manager,
                                tool_name,
                                prev_signature=store.get_signature(),
                                max_attempts=1,
                                operation_was_noop=True,
                            )
                            store.soft_clear()
                            await _prepopulate_entity_store(session_id, manager, fresh_context)
                            current_tokens = _extract_tokens_from_context(fresh_context)
                            store.prune_stale_tokens(current_tokens)
                            store.store_signature()

                            is_valid, validation_warning = _validate_entity_store_for_tool(
                                session_id, manager, tool_name
                            )
                            if is_valid:
                                logger.info(
                                    "Session %s entity context recovery succeeded for '%s'; continuing execution.",
                                    session_id,
                                    tool_name,
                                )
                            else:
                                logger.warning(
                                    "Session %s entity context recovery completed but validation still failed for '%s': %s",
                                    session_id,
                                    tool_name,
                                    validation_warning,
                                )
                        except Exception as exc:
                            logger.warning(
                                "Session %s entity context recovery failed for '%s': %s",
                                session_id,
                                tool_name,
                                exc,
                            )

                    if not is_valid:
                        # Entity context is missing - return error to LLM with guidance
                        error_text = (
                            f"SPATIAL VALIDATION FAILED:\n{validation_warning}\n\n"
                            f"The tool '{tool_name}' requires entity references but none are available. "
                            f"This is a critical error that must be resolved before proceeding."
                        )
                        await _send_error(manager, session_id, "Entity context missing", error_text)
                        messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                        iteration_had_failure = True
                        if iteration_first_failure_intent is None:
                            iteration_first_failure_intent = tool_intent_key
                        logger.warning(
                            "Session %s entity validation failed for '%s': %s",
                            session_id, tool_name, validation_warning
                        )
                        continue

                if tool_name in FEATURE_OPERATION_TOOLS:
                    # Handle feature operations (fillet, chamfer, holes, patterns) - pass full tool_input with description
                    try:
                        success, result_text, raw_result = await _execute_feature_tool_call(
                            session_id, manager, tool_name, tool_use_id, tool_input, description
                        )
                    except SelectionToolCallError as exc:
                        error_text = str(exc)
                        await _send_error(manager, session_id, "Feature tool execution failed", error_text)
                        messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                        iteration_had_failure = True
                        if iteration_first_failure_intent is None:
                            iteration_first_failure_intent = tool_intent_key
                        logger.error(
                            "Session %s feature tool '%s' failed: %s",
                            session_id,
                            tool_name,
                            error_text,
                        )
                        continue

                    # Inject build plan context if active
                    plan_context = await _handle_build_plan_step(session_id, manager, tool_name, success)
                    if plan_context:
                        result_text += plan_context

                    messages.append(_tool_result_message(tool_use_id, result_text, is_error=not success))
                    if not success:
                        iteration_had_failure = True
                        if iteration_first_failure_intent is None:
                            iteration_first_failure_intent = tool_intent_key
                        await _send_error(
                            manager,
                            session_id,
                            "Feature tool execution failed",
                            result_text,
                        )
                        logger.error(
                            "Session %s feature tool '%s' reported failure: %s",
                            session_id,
                            tool_name,
                            result_text,
                        )
                    else:
                        logger.info(
                            "Session %s feature tool '%s' completed successfully: %s",
                            session_id,
                            tool_name,
                            result_text,
                        )
                        try:
                            await _refresh_and_enrich_after_success(
                                session_id,
                                manager,
                                tool_name,
                                raw_result,
                                messages,
                            )
                        except EntityRefreshError as exc:
                            error_text = str(exc)
                            await _send_error(
                                manager,
                                session_id,
                                "Entity refresh failed",
                                error_text,
                            )
                            messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                            iteration_had_failure = True
                            if iteration_first_failure_intent is None:
                                iteration_first_failure_intent = tool_intent_key
                            manager.set_conversation_history(session_id, messages)
                            logger.error(
                                "Session %s: CRITICAL - Entity refresh failed after %s, stopping workflow: %s",
                                session_id,
                                tool_name,
                                error_text,
                            )
                            iteration_force_stop = True
                            break
                        if tool_name in DUPLICATE_INTENT_GUARD_TOOLS:
                            seen_guarded_intents.add(tool_intent_key)
                        iteration_had_success = True
                        await _capture_operation_checkpoint(
                            session_id,
                            manager,
                            request=request,
                            tool_name=tool_name,
                            tool_use_id=tool_use_id,
                            description=description,
                            messages=messages,
                            ir_doc_state=ir_doc_state,
                            force_snapshot_refresh=False,
                        )
                    continue

                # Handle edge/face/body operations - remove description from tool_input
                # (it's already extracted and passed as a separate parameter)
                geometry_tool_input = {k: v for k, v in tool_input.items() if k != "description"}
                
                if tool_name in EDGE_OPERATION_TOOLS:
                    geometry_kind = "edge"
                elif tool_name in FACE_OPERATION_TOOLS:
                    geometry_kind = "face"
                elif tool_name in BODY_OPERATION_TOOLS:
                    geometry_kind = "body"
                else:  # pragma: no cover - guarded by GEOMETRY_OPERATION_TOOLS
                    raise SelectionToolCallError(f"Unsupported geometry tool '{tool_name}'.")

                try:
                    success, result_text = await _execute_geometry_tool_call(
                        session_id, manager, tool_name, tool_use_id, geometry_tool_input, geometry_kind, description
                    )
                except SelectionToolCallError as exc:
                    error_text = str(exc)
                    await _send_error(manager, session_id, f"{geometry_kind.title()} tool execution failed", error_text)
                    messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                    iteration_had_failure = True
                    if iteration_first_failure_intent is None:
                        iteration_first_failure_intent = tool_intent_key
                    logger.error(
                        "Session %s %s tool '%s' failed: %s",
                        session_id,
                        geometry_kind,
                        tool_name,
                        error_text,
                    )
                    continue

                # Inject build plan context if active
                plan_context = await _handle_build_plan_step(session_id, manager, tool_name, success)
                if plan_context:
                    result_text += plan_context

                messages.append(_tool_result_message(tool_use_id, result_text, is_error=not success))
                if not success:
                    iteration_had_failure = True
                    if iteration_first_failure_intent is None:
                        iteration_first_failure_intent = tool_intent_key
                    await _send_error(
                        manager,
                        session_id,
                        f"{geometry_kind.title()} tool execution failed",
                        result_text,
                    )
                    logger.error(
                        "Session %s %s tool '%s' reported failure: %s",
                        session_id,
                        geometry_kind,
                        tool_name,
                        result_text,
                    )
                else:
                    iteration_had_success = True
                    logger.info(
                        "Session %s %s tool '%s' completed successfully: %s",
                        session_id,
                        geometry_kind,
                        tool_name,
                        result_text,
                    )
                continue

            if execution_target == "build123d":
                unsupported_text = (
                    f"Tool '{tool_name}' is not supported by the build123d target yet. "
                    "Supported MVP tools: create_sketch, add_rectangle, add_circle, extrude_profile."
                )
                await _send_error(manager, session_id, "Unsupported build123d tool", unsupported_text)
                messages.append(_tool_result_message(tool_use_id, unsupported_text, is_error=True))
                iteration_had_failure = True
                if iteration_first_failure_intent is None:
                    iteration_first_failure_intent = tool_intent_key
                continue

            # Remove narrative-only fields before translation to keep tool schema strict.
            codegen_input = dict(tool_input)

            if tool_name in {"add_rectangle", "add_circle"}:
                sketch_preflight_error = _preflight_face_sketch_uv_bounds(
                    session_id,
                    manager,
                    tool_name=tool_name,
                    tool_input=codegen_input,
                )
                if sketch_preflight_error:
                    await _send_error(manager, session_id, "Face sketch bounds preflight failed", sketch_preflight_error)
                    messages.append(_tool_result_message(tool_use_id, sketch_preflight_error, is_error=True))
                    iteration_had_failure = True
                    if iteration_first_failure_intent is None:
                        iteration_first_failure_intent = _build_failure_intent_key(
                            tool_name,
                            codegen_input,
                            sketch_preflight_error,
                        )
                    logger.warning(
                        "Session %s blocked '%s' by face-sketch bounds preflight in direct path: %s",
                        session_id,
                        tool_name,
                        sketch_preflight_error,
                    )
                    continue

            # If create_sketch targets a face_N ref but faces are not loaded, try one
            # context recovery before reference resolution to avoid stale/empty-context misses.
            if tool_name == "create_sketch":
                plane_id = str(codegen_input.get("plane_id", "")).strip()
                if re.match(r"^face_\d+$", plane_id):
                    store = _get_entity_store(session_id, manager)
                    if not store.get_refs_by_kind("face"):
                        logger.warning(
                            "Session %s create_sketch on '%s' has no face refs loaded; attempting one-time context recovery.",
                            session_id,
                            plane_id,
                        )
                        try:
                            fresh_context = await _refresh_entity_context_with_retry(
                                session_id,
                                manager,
                                tool_name,
                                prev_signature=store.get_signature(),
                                max_attempts=1,
                                operation_was_noop=True,
                            )
                            store.soft_clear()
                            await _prepopulate_entity_store(session_id, manager, fresh_context)
                            current_tokens = _extract_tokens_from_context(fresh_context)
                            store.prune_stale_tokens(current_tokens)
                            store.store_signature()
                            logger.info(
                                "Session %s create_sketch context recovery completed before resolving '%s'.",
                                session_id,
                                plane_id,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Session %s create_sketch context recovery failed for '%s': %s",
                                session_id,
                                plane_id,
                                exc,
                            )

            try:
                codegen_input = _resolve_codegen_entity_refs(session_id, manager, tool_name, codegen_input)
                code = translate_tool_call(tool_name, codegen_input)
            except SelectionToolCallError as exc:
                error_text = str(exc)
                await _send_error(manager, session_id, "Reference resolution failed", error_text)
                messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                iteration_had_failure = True
                if iteration_first_failure_intent is None:
                    iteration_first_failure_intent = tool_intent_key
                logger.warning("Session %s failed to resolve refs for '%s': %s", session_id, tool_name, error_text)
                continue
            except CodeGenerationError as exc:
                error_text = format_error_for_llm(exc)
                await _send_error(manager, session_id, "Code generation failed", error_text)
                messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                iteration_had_failure = True
                if iteration_first_failure_intent is None:
                    iteration_first_failure_intent = tool_intent_key
                logger.exception("Session %s failed to translate tool call '%s'", session_id, tool_name)
                continue

            execute_payload = {
                "type": "execute_code",
                "code": code,
                "operation": tool_name,
                "tool_use_id": tool_use_id,
                "description": description,
            }

            await _send_message_safe(manager, session_id, execute_payload)

            try:
                result = await _wait_for_matching_tool_result(
                    session_id,
                    manager,
                    tool_name=tool_name,
                    tool_use_id=tool_use_id,
                    timeout=EXECUTION_TIMEOUT,
                    wait_context="execute_code",
                )
            except asyncio.TimeoutError:
                error_text = (
                    f"Timed out waiting for Fusion to execute '{tool_name}' "
                    f"(tool_use_id={tool_use_id})."
                )
                await _send_error(manager, session_id, "Fusion execution timeout", error_text)
                messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                iteration_had_failure = True
                if iteration_first_failure_intent is None:
                    iteration_first_failure_intent = tool_intent_key
                logger.error("Session %s timed out waiting for tool result", session_id)
                continue

            # Register sketch entities BEFORE summarization so enrichers
            # can reference registered refs (critical for compound tools like add_rectangle).
            _pre_success = bool(result.get("success", result.get("type") != "error"))
            if _pre_success:
                try:
                    fallback_plane_id = None
                    if tool_name == "create_sketch":
                        fallback_plane_id = str(codegen_input.get("plane_id") or "").strip() or None
                    _register_sketch_result_entities(
                        session_id,
                        manager,
                        tool_name,
                        result,
                        fallback_plane_id=fallback_plane_id,
                    )
                except Exception as exc:
                    logger.warning("Sketch entity registration failed for %s: %s", tool_name, exc)
                if tool_name == "create_sketch":
                    sketch_id = str(result.get("sketch_id") or "").strip()
                    if sketch_id:
                        sketch_metadata = _get_sketch_entity_store(session_id, manager).get_sketch_metadata(sketch_id)
                        if sketch_metadata.get("plane_kind") == "face":
                            face_sketches_created_this_turn.add(sketch_id)

            success, result_text = _summarise_execution_result(tool_name, result)

            # Note: created_entities announcement moved to AFTER entity context refresh
            # to ensure we only announce refs that actually exist in the refreshed context

            messages.append(_tool_result_message(tool_use_id, result_text, is_error=not success))
            if not success:
                iteration_had_failure = True
                if iteration_first_failure_intent is None:
                    failure_detail = (
                        result.get("error")
                        or result.get("message")
                        or result.get("details")
                        or result_text
                    )
                    iteration_first_failure_intent = _build_failure_intent_key(
                        tool_name,
                        tool_input,
                        failure_detail,
                    )
                await _send_error(manager, session_id, "Fusion execution failed", result_text)
                logger.error("Session %s reported execution failure for '%s': %s", session_id, tool_name, result_text)
            else:
                iteration_had_success = True
                logger.info("Session %s completed tool '%s' successfully", session_id, tool_name)

                try:
                    await _refresh_and_enrich_after_success(
                        session_id,
                        manager,
                        tool_name,
                        result,
                        messages,
                    )
                except EntityRefreshError as exc:
                    error_text = str(exc)
                    await _send_error(
                        manager,
                        session_id,
                        "Entity refresh failed",
                        error_text,
                    )
                    messages.append(_tool_result_message(tool_use_id, error_text, is_error=True))
                    iteration_had_failure = True
                    if iteration_first_failure_intent is None:
                        iteration_first_failure_intent = tool_intent_key
                    manager.set_conversation_history(session_id, messages)
                    logger.error(
                        "Session %s: CRITICAL - Entity refresh failed after %s, stopping workflow: %s",
                        session_id,
                        tool_name,
                        error_text,
                    )
                    iteration_force_stop = True
                    break

                # Inject build plan context if active (for all successful tools)
                plan_context = await _handle_build_plan_step(session_id, manager, tool_name, success)
                if plan_context and messages and messages[-1].get("role") == "user":
                    last_msg = messages[-1]
                    if "content" in last_msg and isinstance(last_msg["content"], list):
                        for content_block in last_msg["content"]:
                            if isinstance(content_block, dict) and content_block.get("type") == "tool_result":
                                if isinstance(content_block.get("content"), list):
                                    for text_block in content_block["content"]:
                                        if isinstance(text_block, dict) and text_block.get("type") == "text":
                                            text_block["text"] += plan_context
                                            break
                await _capture_operation_checkpoint(
                    session_id,
                    manager,
                    request=request,
                    tool_name=tool_name,
                    tool_use_id=tool_use_id,
                    description=description,
                    messages=messages,
                    ir_doc_state=ir_doc_state,
                    force_snapshot_refresh=tool_name not in REFRESH_ON_SUCCESS_TOOLS,
                )

        if iteration_force_stop:
            break

        # Track retries against the same failing intent. Mixed success/failure turns reset the streak.
        if iteration_had_failure:
            if iteration_had_success:
                consecutive_tool_failures = 0
                last_failed_intent_key = None
            else:
                failure_intent = iteration_first_failure_intent or "unknown_intent"
                if failure_intent == last_failed_intent_key:
                    consecutive_tool_failures += 1
                else:
                    consecutive_tool_failures = 1
                    last_failed_intent_key = failure_intent

                if consecutive_tool_failures >= 2:
                    error_text = (
                        "Execution stopped after consecutive tool failures for the same operation intent. "
                        "The LLM attempted a retry but it failed again. "
                        "Please review the errors above and submit a corrected request."
                    )
                    await _send_error(manager, session_id, "Consecutive failures limit reached", error_text)

                    # Save conversation state before exiting
                    manager.set_conversation_history(session_id, messages)

                    logger.error(
                        "Session %s: Stopping workflow after %d consecutive failures for intent %s in iteration %d",
                        session_id,
                        consecutive_tool_failures,
                        failure_intent,
                        iteration + 1,
                    )
                    break

                logger.warning(
                    "Session %s: Tool failure in iteration %d for intent %s, allowing one retry attempt",
                    session_id,
                    iteration + 1,
                    failure_intent,
                )
        else:
            consecutive_tool_failures = 0
            last_failed_intent_key = None

    else:
        error_text = f"Reached the iteration limit ({max_iterations}) without 'end_turn'."
        await _send_error(manager, session_id, "Iteration limit reached", error_text)

        # Save conversation even on iteration limit
        manager.set_conversation_history(session_id, messages)

        logger.error("Session %s exceeded iteration limit", session_id)


async def handle_planning_request(
    session_id: str,
    request: Mapping[str, Any],
    manager: ConnectionManager,
) -> None:
    """
    Planning mode workflow: stream plan chunks to the UI, await approval, and
    execute the approved plan via the standard execution loop.

    Note: Planning mode clears conversation history and starts fresh with the plan.
    """
    user_request = request.get("user_request")
    if not user_request:
        raise ValueError("Planning request must include 'user_request'.")

    model_name = request.get("model_name")
    logger.info("Starting planning mode for session %s", session_id)

    # Clear conversation history for planning mode (fresh start with plan)
    manager.clear_conversation(session_id)
    manager.clear_checkpoints(session_id)
    manager.clear_operation_checkpoints(session_id)
    _manager_clear_ir_document_state(manager, session_id)

    try:
        await _execute_planning_workflow(session_id, user_request, model_name, request, manager)
    except asyncio.CancelledError:
        logger.info("Planning cancelled by user for session %s", session_id)
        await _send_message_safe(manager, session_id, {"type": "cancelled", "message": "Planning cancelled"})
        raise


async def handle_revert_request(
    session_id: str,
    request: Mapping[str, Any],
    manager: ConnectionManager,
) -> None:
    """
    Handle a request to revert the timeline to a previous checkpoint.

    Args:
        session_id: Session identifier
        request: Revert request containing message_id
        manager: Connection manager
    """
    message_id = request.get("message_id")
    if not message_id:
        error_text = "Revert request must include 'message_id'."
        await _send_error(manager, session_id, "Invalid revert request", error_text)
        logger.error("Session %s: revert request missing message_id", session_id)
        return

    # Look up the checkpoint
    checkpoint = manager.get_checkpoint_by_message_id(session_id, message_id)
    if not checkpoint:
        error_text = f"Checkpoint not found for message_id: {message_id}"
        await _send_error(manager, session_id, "Checkpoint not found", error_text)
        logger.error("Session %s: checkpoint not found for message_id %s", session_id, message_id)
        return

    marker_position = checkpoint.get("marker_position", 0)
    timeline_count = checkpoint.get("timeline_count", 0)
    message_text = checkpoint.get("message_text", "")
    conversation_index = checkpoint.get("conversation_index")

    logger.info(
        "Session %s: reverting to checkpoint %s (marker_position=%d, timeline_count=%d)",
        session_id, message_id, marker_position, timeline_count
    )

    # Send revert command to Fusion 360
    revert_payload = {
        "type": "revert_timeline",
        "message_id": message_id,
        "marker_position": marker_position,
        "timeline_count": timeline_count,
    }
    await _send_message_safe(manager, session_id, revert_payload)

    try:
        # Wait for confirmation from Fusion, filtering by message_id to avoid
        # consuming stale results from cancelled/superseded operations
        result = await manager.wait_for_fusion_result(
            session_id,
            timeout=EXECUTION_TIMEOUT,
            expected_message_id=message_id,
        )
        success = result.get("success", False)

        if success:
            trimmed_length = manager.trim_conversation_to_index(
                session_id,
                conversation_index,
                include_current_message=False,
            )
            manager.prune_checkpoints_after(session_id, conversation_index)
            manager.prune_operation_checkpoints_after(session_id, conversation_index)
            await _restore_message_checkpoint_runtime_state(session_id, manager, checkpoint, conversation_index)
            logger.debug("Restored runtime state after revert for session %s", session_id)

            await _send_message_safe(manager, session_id, {
                "type": "revert_applied",
                "message_id": message_id,
                "conversation_index": conversation_index,
                "conversation_length": trimmed_length,
                "include_message": False,
            })

            # Notify user of successful revert
            success_message = f"Timeline reverted to: \"{message_text}\""
            await _send_message_safe(manager, session_id, {
                "type": "log",
                "level": "success",
                "message": success_message,
                "scope": "global",
            })
            logger.info("Session %s: successfully reverted to checkpoint %s", session_id, message_id)
        else:
            error_detail = result.get("error") or result.get("message") or "Unknown error"
            timeline_unavailable = "Timeline not available" in error_detail

            if timeline_unavailable:
                trimmed_length = manager.trim_conversation_to_index(
                    session_id,
                    conversation_index,
                    include_current_message=False,
                )
                manager.prune_checkpoints_after(session_id, conversation_index)
                manager.prune_operation_checkpoints_after(session_id, conversation_index)
                await _restore_message_checkpoint_runtime_state(session_id, manager, checkpoint, conversation_index)
                logger.debug("Restored runtime state after timeline-unavailable revert for session %s", session_id)

                await _send_message_safe(manager, session_id, {
                    "type": "revert_applied",
                    "message_id": message_id,
                    "conversation_index": conversation_index,
                    "conversation_length": trimmed_length,
                    "include_message": False,
                    "timeline_reverted": False,
                })

                warning_message = (
                    "Timeline could not be rewound because this design does not expose a timeline. "
                    "Conversation history has been rolled back to keep the agent in sync, "
                    "but any geometry changes remain in the model."
                )
                await _send_message_safe(manager, session_id, {
                    "type": "log",
                    "level": "warning",
                    "message": warning_message,
                    "scope": "global",
                })

                logger.warning(
                    "Session %s: timeline unavailable during revert for checkpoint %s; "
                    "conversation trimmed but geometry unchanged.",
                    session_id,
                    message_id,
                )
            else:
                error_text = f"Failed to revert timeline: {error_detail}"
                await _send_error(manager, session_id, "Revert failed", error_text)
                logger.error("Session %s: revert failed: %s", session_id, error_text)

    except asyncio.TimeoutError:
        error_text = "Timed out waiting for Fusion to revert timeline."
        await _send_error(manager, session_id, "Revert timeout", error_text)
        logger.error("Session %s: timeout waiting for revert confirmation", session_id)


async def handle_resume_operation_request(
    session_id: str,
    request: Mapping[str, Any],
    manager: ConnectionManager,
) -> None:
    """Resume the session from a successful operation checkpoint."""
    checkpoint_id = request.get("checkpoint_id") or request.get("operation_checkpoint_id")
    if not checkpoint_id:
        error_text = "Resume request must include 'checkpoint_id'."
        await _send_error(manager, session_id, "Invalid resume request", error_text)
        logger.error("Session %s: operation resume request missing checkpoint_id", session_id)
        return

    checkpoint = manager.get_operation_checkpoint(session_id, str(checkpoint_id))
    if not checkpoint:
        error_text = f"Operation checkpoint not found: {checkpoint_id}"
        await _send_error(manager, session_id, "Operation checkpoint not found", error_text)
        logger.error("Session %s: operation checkpoint not found: %s", session_id, checkpoint_id)
        return

    marker_position = int(checkpoint.get("marker_position", 0) or 0)
    timeline_count = int(checkpoint.get("timeline_count", 0) or 0)

    logger.info(
        "Session %s: resuming from operation checkpoint %s (tool=%s marker_position=%d)",
        session_id,
        checkpoint_id,
        checkpoint.get("tool_name"),
        marker_position,
    )

    await _send_message_safe(
        manager,
        session_id,
        {
            "type": "revert_timeline",
            "message_id": checkpoint_id,
            "operation_checkpoint_id": checkpoint_id,
            "marker_position": marker_position,
            "timeline_count": timeline_count,
        },
    )

    try:
        result = await manager.wait_for_fusion_result(
            session_id,
            timeout=EXECUTION_TIMEOUT,
            expected_message_id=str(checkpoint_id),
        )
    except asyncio.TimeoutError:
        error_text = "Timed out waiting for Fusion to resume from the operation checkpoint."
        await _send_error(manager, session_id, "Resume timeout", error_text)
        logger.error("Session %s: timeout waiting for operation resume checkpoint %s", session_id, checkpoint_id)
        return

    if not result.get("success", False):
        error_detail = result.get("error") or result.get("message") or "Unknown error"
        error_text = f"Failed to restore operation checkpoint: {error_detail}"
        await _send_error(manager, session_id, "Resume failed", error_text)
        logger.error("Session %s: operation resume failed: %s", session_id, error_text)
        return

    manager.clear_entity_store(session_id)
    latest_entity_context = checkpoint.get("latest_entity_context")
    if isinstance(latest_entity_context, Mapping) and latest_entity_context:
        await _prepopulate_entity_store(session_id, manager, latest_entity_context)
        _manager_set_latest_entity_context(manager, session_id, dict(latest_entity_context))

    trimmed_length = manager.restore_operation_checkpoint_state(session_id, checkpoint)
    manager.prune_checkpoints_after(session_id, checkpoint.get("conversation_index"))
    manager.prune_operation_checkpoints_after(session_id, checkpoint.get("conversation_index"))

    await _send_message_safe(
        manager,
        session_id,
        {
            "type": "operation_resume_applied",
            "checkpoint_id": checkpoint_id,
            "operation_checkpoint_id": checkpoint_id,
            "conversation_index": checkpoint.get("conversation_index"),
            "conversation_length": trimmed_length,
            "tool_name": checkpoint.get("tool_name"),
            "display_label": checkpoint.get("display_label"),
        },
    )

    await _send_message_safe(
        manager,
        session_id,
        {
            "type": "log",
            "level": "success",
            "message": f"Resumed from operation: {checkpoint.get('display_label') or checkpoint.get('tool_name')}",
            "scope": "global",
        },
    )
    logger.info("Session %s: resumed from operation checkpoint %s", session_id, checkpoint_id)


async def _execute_planning_workflow(
    session_id: str,
    user_request: str,
    model_name: Optional[str],
    request: Mapping[str, Any],
    manager: ConnectionManager
) -> None:
    """Execute the planning workflow with proper cancellation support."""
    plan_chunks_text: List[str] = []

    # Soft clear entity store at start of planning request (preserves persistent cache for stable refs),
    llm_api_keys = manager.get_llm_api_keys(session_id)
    user_token = manager.get_user_token(session_id)

    # then repopulate if entity_context is available
    store = _get_entity_store(session_id, manager)
    store.soft_clear()

    entity_context = request.get("entity_context")
    if entity_context:
        await _prepopulate_entity_store(session_id, manager, entity_context)
        _manager_set_latest_entity_context(manager, session_id, dict(entity_context))
        # Prune stale tokens from persistent cache
        current_tokens = _extract_tokens_from_context(entity_context)
        store.prune_stale_tokens(current_tokens)
    else:
        _manager_clear_latest_entity_context(manager, session_id)

    # Build user request with entity context appended for planner awareness
    planning_request = user_request
    if entity_context:
        try:
            entity_text = _format_unified_context(entity_context)
        except Exception as exc:
            logger.warning("Failed to format entity context for planning: %s", exc)
            entity_text = ""
        if entity_text:
            planning_request = f"{user_request}\n\n{entity_text}"

    try:
        # Use "high" reasoning for complex multi-step planning
        async for event in generate_plan(
            planning_request,
            model_name=model_name,
            reasoning_effort="high",
            api_keys=llm_api_keys,
            user_token=user_token,
        ):
            event_type = event.get("type")
            content = event.get("content", "")

            if not content.strip():
                continue

            if event_type == "reasoning":
                # Stream reasoning chunks (GPT-5 chain of thought)
                await _send_message_safe(manager, session_id, {"type": "reasoning_chunk", "content": content})
            elif event_type == "text":
                # Stream plan text chunks for UI while storing raw markdown for execution context
                raw_segment = event.get("raw_text")
                if isinstance(raw_segment, str) and raw_segment.strip():
                    plan_chunks_text.append(raw_segment)
                    chunk_text = raw_segment
                else:
                    fallback = html_to_plain_text(content)
                    plan_chunks_text.append(fallback)
                    chunk_text = fallback

                await _send_message_safe(
                    manager,
                    session_id,
                    {"type": "plan_chunk", "content": chunk_text, "format": "markdown"},
                )
    except asyncio.CancelledError:
        # Re-raise cancellation to be handled by the outer handler
        raise
    except Exception as exc:
        error_text = f"Failed to generate plan: {exc}"
        await _send_error(manager, session_id, "Plan generation failed", error_text)
        logger.exception("Plan generation failed for session %s", session_id)
        return

    full_plan_text = "".join(plan_chunks_text).strip()

    full_plan_html = markdown_to_html(full_plan_text) if full_plan_text else ""

    logger.debug("Full plan generated (length %d characters)", len(full_plan_text))

    summary_source = full_plan_text or full_plan_html
    summary_source_is_plain = bool(full_plan_text)

    if summary_source:
        try:
            display_plan = await summarize_plan_for_user(summary_source, api_keys=llm_api_keys)
        except Exception:  # summarize_plan_for_user already logs, but guard against unexpected failures
            display_plan = markdown_to_html(summary_source) if summary_source_is_plain else summary_source
    else:
        display_plan = ""

    display_plan_plain = html_to_plain_text(display_plan) if display_plan else ""

    await _send_message_safe(
        manager,
        session_id,
        {
            "type": "plan_complete",
            "full_plan": full_plan_text,
            "display_plan": display_plan,
            "display_plan_plain": display_plan_plain,
        },
    )
    logger.info("Plan generation completed for session %s, awaiting approval", session_id)

    try:
        approval = await _wait_for_plan_decision(session_id, manager)
    except asyncio.TimeoutError:
        error_text = "Timed out waiting for plan approval."
        await _send_error(manager, session_id, "Plan approval timeout", error_text)
        logger.error("Session %s timed out waiting for plan approval", session_id)
        return

    if not approval.get("approved"):
        message = approval.get("message", "Plan was rejected by the user.")
        await _send_message_safe(
            manager,
            session_id,
            {"type": "llm_message", "message": markdown_to_html(message), "format": "html"},
        )
        logger.info("Plan rejected for session %s: %s", session_id, message)
        return

    plan_text = full_plan_text
    updated_request = dict(request)
    updated_request["plan_text"] = plan_text

    await _send_message_safe(
        manager,
        session_id,
        {"type": "llm_message", "message": markdown_to_html("Plan approved. Beginning execution."), "format": "html"},
    )
    logger.info("Plan approved for session %s, starting execution workflow", session_id)
    await handle_execute_request(session_id, updated_request, manager)


def _build_user_message(request: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Compose a user message with request and current timeline state.

    For follow-up requests, includes only the new request and updated timeline.
    For initial requests with a plan, includes the plan text.
    """
    user_request = str(request.get("user_request", "")).strip()
    timeline_state = request.get("timeline_state") or []
    plan_text = request.get("plan_text")
    visual_context = request.get("visual_context") or {}
    visual_data = visual_context.get("data")
    visual_label = str(visual_context.get("label") or "Visual state of the 3D model inside Fusion 360").strip()
    selection_context = request.get("selection_context")
    feature_snapshot = request.get("feature_snapshot")
    entity_context = request.get("entity_context")
    attachments_context = str(request.get("attachments_context") or "").strip()

    sections = [f"User Request:\n{user_request}"]

    if attachments_context:
        sections.append(attachments_context)

    # Include entity context (bodies, faces, edges) for LLM awareness
    if entity_context:
        try:
            entity_text = _format_unified_context(entity_context)
        except Exception as exc:
            logger.warning("Failed to format entity context for user message: %s", exc)
            entity_text = ""
        if entity_text:
            sections.append(entity_text)

    # Include selection context if entities were selected
    if selection_context:
        selection_text = _format_selection_context(selection_context)
        if selection_text:
            sections.append(f"Selected Entities:\n{selection_text}")

    # Include recent feature context when available
    if feature_snapshot:
        snapshot_text = _format_feature_snapshot(feature_snapshot)
        sketch_text = _format_sketch_context(feature_snapshot)

        combined_context = []
        if snapshot_text:
            combined_context.append(f"Recent Timeline Features:\n{snapshot_text}")
        if sketch_text:
            combined_context.append(sketch_text)

        if combined_context:
            sections.append("\n\n".join(combined_context))

    # Always include current timeline state for context
    if timeline_state:
        sections.append("Current Timeline:\n" + _format_timeline_state(timeline_state))

    # Only include plan text if provided (first request after planning)
    if plan_text:
        sections.append("Approved Plan:\n" + str(plan_text).strip())

    if visual_data:
        sections.append(f"Visual Context Snapshot:\n{visual_label}")

    text_content = "\n\n".join(sections)

    if not visual_data:
        return {"role": "user", "content": text_content}

    media_type = str(visual_context.get("media_type") or "image/png").strip() or "image/png"
    image_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": visual_data,
        },
    }

    return {"role": "user", "content": [
        {"type": "text", "text": text_content},
        image_block,
    ]}


def _extract_last_user_notification(messages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Retrieve the most recent respond_to_user message that was delivered to the user.
    """
    marker = "Delivered message to user:"

    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for block in reversed(content):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue

            block_content = block.get("content") or []
            if not isinstance(block_content, list):
                block_content = [block_content]

            for item in block_content:
                text_value: Optional[str] = None
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_value = str(item.get("text", ""))
                elif isinstance(item, str):
                    text_value = item

                if not text_value:
                    continue

                if text_value.startswith(marker):
                    delivered_text = text_value[len(marker):].strip()
                    if delivered_text:
                        return _normalise_user_notification(delivered_text)

    return None


def _normalise_user_notification(text: str) -> str:
    """Collapse whitespace for consistent duplicate detection."""
    return " ".join(str(text).split())


def _prune_empty_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove messages that have empty content blocks."""
    return [message for message in messages if _message_has_content(message)]


def _message_has_content(message: Mapping[str, Any]) -> bool:
    """Determine whether a message contains non-empty content."""
    content = message.get("content")
    if content is None:
        return False

    if isinstance(content, str):
        return bool(content.strip())

    if isinstance(content, list):
        if not content:
            return False
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text_value = str(block.get("text", ""))
                    if text_value.strip():
                        return True
                else:
                    # Treat tool_use, tool_result, image, etc. as meaningful content
                    return True
            elif isinstance(block, str):
                if block.strip():
                    return True
            elif block:
                return True
        return False

    if isinstance(content, dict):
        if content.get("type") == "text":
            return bool(str(content.get("text", "")).strip())
        return bool(content)

    return bool(content)


def _build_initial_user_message(request: Mapping[str, Any]) -> Dict[str, Any]:
    """Compose the first user message with request details, timeline, and plan (deprecated - use _build_user_message)."""
    return _build_user_message(request)


def _format_feature_snapshot(feature_snapshot: Mapping[str, Any]) -> str:
    """Render feature snapshot metadata into a concise prompt section."""
    if not isinstance(feature_snapshot, Mapping):
        return ""

    if not feature_snapshot.get("success", True):
        detail = feature_snapshot.get("error") or feature_snapshot.get("message")
        if detail:
            return f"Feature snapshot unavailable (reason: {detail})"
        return "Feature snapshot unavailable."

    features = feature_snapshot.get("features")
    if not isinstance(features, list):
        return ""

    total_features = len(features)
    timeline_count = feature_snapshot.get("timeline_count")
    marker_position = feature_snapshot.get("marker_position")
    captured_label = feature_snapshot.get("captured_at_label") or feature_snapshot.get("captured_at")
    length_unit = feature_snapshot.get("length_unit")
    area_unit = feature_snapshot.get("area_unit")
    volume_unit = feature_snapshot.get("volume_unit")
    mass_unit = feature_snapshot.get("mass_unit")

    header_parts: List[str] = [f"count={total_features}"]
    if timeline_count is not None:
        header_parts.append(f"timeline_count={timeline_count}")
    if marker_position is not None:
        header_parts.append(f"marker_position={marker_position}")
    if captured_label:
        header_parts.append(f"captured_at={captured_label}")
    if length_unit:
        header_parts.append(f"length_unit={length_unit}")
    if area_unit:
        header_parts.append(f"area_unit={area_unit}")
    if volume_unit:
        header_parts.append(f"volume_unit={volume_unit}")
    if mass_unit:
        header_parts.append(f"mass_unit={mass_unit}")

    lines = ["; ".join(header_parts)]

    max_feature_lines = 10
    trimmed_features = features[:max_feature_lines]

    serialized_payload: List[Dict[str, Any]] = []

    for feature in trimmed_features:
        if not isinstance(feature, Mapping):
            continue

        timeline_index = feature.get("timeline_index")
        feature_type = feature.get("feature_type") or feature.get("object_type") or "Feature"
        name = feature.get("name") or ""
        token = feature.get("entity_token")
        suppressed = feature.get("is_suppressed")
        instance_count = feature.get("instance_count")
        hole_details = feature.get("hole") if isinstance(feature.get("hole"), Mapping) else None
        editable = feature.get("editable_parameters") if isinstance(feature.get("editable_parameters"), Mapping) else None
        bodies = feature.get("bodies") if isinstance(feature.get("bodies"), list) else []

        body_names: List[str] = []
        body_tokens: List[str] = []
        body_summaries: List[str] = []
        any_bbox: Optional[Mapping[str, Any]] = None

        for body in bodies:
            if not isinstance(body, Mapping):
                continue
            body_name = body.get("name") or ""
            if body_name:
                body_names.append(str(body_name))
            body_token = body.get("entity_token")
            if isinstance(body_token, str) and body_token.strip():
                body_tokens.append(body_token.strip())
            component = body.get("component")
            summary_parts: List[str] = []
            if body_name:
                summary_parts.append(body_name)
            if component:
                summary_parts.append(f"component={component}")
            if summary_parts:
                body_summaries.append("/".join(summary_parts))
            if any_bbox is None:
                maybe_bbox = body.get("bounding_box")
                if isinstance(maybe_bbox, Mapping):
                    any_bbox = maybe_bbox

        bbox_text = ""
        if any_bbox:
            min_point = any_bbox.get("min") or any_bbox.get("min_point")
            max_point = any_bbox.get("max") or any_bbox.get("max_point")
            if isinstance(min_point, Mapping) and isinstance(max_point, Mapping):
                bbox_text = (
                    f"bbox_min=({min_point.get('x')}, {min_point.get('y')}, {min_point.get('z')}), "
                    f"bbox_max=({max_point.get('x')}, {max_point.get('y')}, {max_point.get('z')})"
                )

        line_parts: List[str] = []
        if timeline_index is not None:
            line_parts.append(f"index={timeline_index}")
        line_parts.append(f"type={feature_type}")
        if name:
            line_parts.append(f"name={name}")
        if token:
            line_parts.append(f"token={token}")
        if suppressed is not None:
            line_parts.append(f"suppressed={bool(suppressed)}")
        if instance_count is not None:
            line_parts.append(f"instances={instance_count}")
        if editable and editable.get("supported"):
            supported_params = editable.get("supported_parameters")
            if isinstance(supported_params, list) and supported_params:
                line_parts.append(f"editable={supported_params}")
        if body_summaries:
            preview = ", ".join(body_summaries[:2])
            if len(body_summaries) > 2:
                preview += f" (+{len(body_summaries) - 2} more)"
            line_parts.append(f"bodies={preview}")
        if body_tokens:
            token_preview = body_tokens[:3]
            if len(body_tokens) > 3:
                token_preview = token_preview + [f"+{len(body_tokens) - 3} more"]
            line_parts.append(f"body_tokens={token_preview}")
        if bbox_text:
            line_parts.append(bbox_text)

        # HoleFeature details (if provided by Fusion snapshot).
        if hole_details:
            dia_mm = hole_details.get("diameter_mm")
            if isinstance(dia_mm, (int, float)):
                line_parts.append(f"hole_dia_mm={dia_mm}")
            centers = hole_details.get("centers_world_mm")
            center_label = "mm"
            if not isinstance(centers, list) or not centers:
                centers = hole_details.get("centers_world_cm")
                center_label = "cm"
            if isinstance(centers, list) and centers:
                # Keep compact: show up to 2 centers.
                preview = []
                for c in centers[:2]:
                    if not isinstance(c, Mapping):
                        continue
                    preview.append(f"({c.get('x')},{c.get('y')},{c.get('z')})")
                if preview:
                    suffix = "" if len(centers) <= 2 else f" (+{len(centers) - 2} more)"
                    line_parts.append(f"hole_centers_{center_label}=[{', '.join(preview)}]{suffix}")

        if line_parts:
            lines.append(" ".join(line_parts))

        serialized_payload.append({
            "timeline_index": timeline_index,
            "feature_type": feature_type,
            "name": name,
            "entity_token": token,
            "is_suppressed": bool(suppressed) if suppressed is not None else None,
            "instance_count": instance_count,
            "body_tokens": body_tokens,
            "body_names": body_names,
            "bounding_box": any_bbox,
            "hole": hole_details,
            "editable_parameters": editable,
        })

    max_serialized = min(len(serialized_payload), 15)
    if max_serialized:
        try:
            lines.append(
                "features_json=" +
                json.dumps(serialized_payload[:max_serialized], ensure_ascii=False, separators=(",", ":"))
            )
        except (TypeError, ValueError):  # pragma: no cover - defensive serialization guard
            logger.debug("Failed to serialize feature snapshot payload for prompt context.")

    return "\n".join(lines)


def _format_sketch_context(feature_snapshot: Mapping[str, Any]) -> str:
    """Render available sketches into a concise prompt section."""
    sketches = feature_snapshot.get("sketches", [])
    if not sketches:
        return ""

    lines = [f"Available Sketches (count={len(sketches)}):"]

    for sketch in sketches:
        name = sketch.get("name", "Unnamed")
        plane = sketch.get("plane", "Unknown")
        profile_count = sketch.get("profile_count", 0)

        sketch_info = f"  - {name} (plane: {plane}, profiles: {profile_count})"
        lines.append(sketch_info)

    lines.append(
        "\nIMPORTANT: When calling revolve_profile, extrude_profile, or other tools, "
        "use the identifier-safe sketch_id (snake_case variable) you created earlier. "
        "The names listed above are human-readable labels only."
    )

    return "\n".join(lines)


def _format_timeline_state(timeline: Any) -> str:
    """Render timeline entries into a human-readable list with marker position."""
    try:
        # Handle new dict format with marker position
        if isinstance(timeline, dict):
            items = timeline.get("items", [])
            marker_position = timeline.get("marker_position", 0)
            count = timeline.get("count", len(items))

            if not items:
                return f"Timeline is empty (user placed marker at position {marker_position})"

            header = f"Timeline: {count} items, user placed marker at position {marker_position}"
            entries = [
                f"  {item.get('index', '?')}: {item.get('type', 'Unknown')} - {item.get('name', 'Unnamed')}"
                for item in items
            ]

            # Insert marker indicator
            formatted_entries = []
            for i, entry in enumerate(entries):
                if i == marker_position:
                    formatted_entries.append("  << USER'S CURRENT VIEW (MARKER POSITION) >>")
                formatted_entries.append(entry)

            # Handle marker at the end
            if marker_position >= len(entries):
                formatted_entries.append("  << USER'S CURRENT VIEW (MARKER POSITION) >>")

            return header + "\n" + "\n".join(formatted_entries)

        # Handle legacy list format (backwards compatibility)
        entries = [
            f"- #{item.get('index', '?')}: {item.get('type', 'Unknown')} - {item.get('name', 'Unnamed')}"
            for item in timeline
        ]
    except Exception:  # pragma: no cover - defensive path
        return "Unable to parse timeline."
    return "\n".join(entries) if entries else "Timeline is empty."


def _format_selection_context(selection_context: Mapping[str, Any]) -> str:
    """
    Format selection context into a concise, human-readable description for the LLM.

    Args:
        selection_context: Selection context dictionary with entities

    Returns:
        Formatted string describing the selected entities
    """
    if not selection_context:
        return ""

    entities = selection_context.get("entities", [])
    if not entities:
        return ""

    count = selection_context.get("count", len(entities))
    lines = [f"The user has selected {count} entity/entities:"]

    for idx, entity in enumerate(entities, 1):
        entity_type = entity.get("type", "unknown")

        if entity_type == "face":
            desc = _format_face_description(idx, entity)
        elif entity_type == "edge":
            desc = _format_edge_description(idx, entity)
        elif entity_type == "body":
            desc = _format_body_description(idx, entity)
        else:
            desc = f"  {idx}. {entity_type.capitalize()}: {entity.get('object_type', 'Unknown')}"

        lines.append(desc)

    return "\n".join(lines)


def _safe_vec3_extract(value: Any) -> Tuple[float, float, float]:
    """
    Safely extract (x, y, z) from a value that could be a dict, list, or None.

    Fusion 360 sends geometry data in inconsistent formats:
    - Sometimes as dict: {"x": 1.0, "y": 2.0, "z": 3.0}
    - Sometimes as list: [1.0, 2.0, 3.0]
    - Sometimes missing/None

    This helper ensures we can always extract coordinates safely.

    Args:
        value: Point/vector data from Fusion (dict, list, or None)

    Returns:
        Tuple of (x, y, z) as floats, defaulting to (0.0, 0.0, 0.0) if invalid
    """
    if value is None:
        return (0.0, 0.0, 0.0)

    # Handle dict format: {"x": val, "y": val, "z": val}
    if isinstance(value, dict):
        x = float(value.get("x", 0.0))
        y = float(value.get("y", 0.0))
        z = float(value.get("z", 0.0))
        return (x, y, z)

    # Handle list/tuple format: [val, val, val]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (ValueError, TypeError, IndexError):
            return (0.0, 0.0, 0.0)

    # Fallback for invalid data
    return (0.0, 0.0, 0.0)


def _compute_face_topology(entity_context: Mapping[str, Any], cap: int = 6) -> Dict[str, List[str]]:
    """
    Build face adjacency from edge.adjacent_faces. Cap output per face.

    Args:
        entity_context: Entity context with edges (flat or spatial_context format)
        cap: Maximum adjacent faces to list per face

    Returns:
        Dict mapping face ref -> list of adjacent face refs
    """
    from collections import defaultdict
    topology: Dict[str, set] = defaultdict(set)

    # Handle spatial_context (nested) format only when bodies are present.
    spatial_context = entity_context.get("spatial_context")
    spatial_bodies: List[Mapping[str, Any]] = []
    if isinstance(spatial_context, dict):
        maybe_bodies = spatial_context.get("bodies", [])
        if isinstance(maybe_bodies, list):
            spatial_bodies = maybe_bodies

    if spatial_bodies:
        for body in spatial_bodies:
            for edge in body.get("edges", []):
                adj = edge.get("adjacent_faces", [])
                if len(adj) == 2:
                    topology[adj[0]].add(adj[1])
                    topology[adj[1]].add(adj[0])
    else:
        # Handle flat format
        for edge in entity_context.get("edges", []):
            adj = edge.get("adjacent_faces", [])
            if len(adj) == 2:
                topology[adj[0]].add(adj[1])
                topology[adj[1]].add(adj[0])

    return {f: sorted(list(neighbors))[:cap] for f, neighbors in topology.items()}


def _compute_parallel_face_pairs(entity_context: Mapping[str, Any]) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Find opposing planar axis-aligned face pairs within each body.

    Returns:
        Dict mapping body_ref/name -> list of (face1_ref, face2_ref, distance) tuples
    """
    MAX_PARALLEL_PAIRS_PER_BODY = 20  # cap to prevent token blow-up
    pairs_by_body: Dict[str, List[Tuple[str, str, float]]] = {}

    def _process_faces(faces: List[Mapping[str, Any]], body_key: str):
        planar_axis = []
        for f in faces:
            ref = f.get("entity_ref", f.get("id", ""))
            normal = f.get("normal")
            centroid = f.get("centroid")
            surface_type = (f.get("surface_type") or f.get("geometry_type") or "").lower()

            if not normal or not centroid:
                continue

            nx, ny, nz = _safe_vec3_extract(normal)
            cx, cy, cz = _safe_vec3_extract(centroid)

            # Only planar faces. Fusion can emit non-planar faces after fillets/rounds
            # (e.g., toroidal). Those must not participate in parallel-pair computation.
            if surface_type and surface_type not in ("planar", "plane"):
                continue
            if max(abs(nx), abs(ny), abs(nz)) < 0.99:
                continue

            planar_axis.append({
                "ref": ref, "normal": (nx, ny, nz), "centroid": (cx, cy, cz)
            })

        body_pairs = []
        pair_keys: set = set()

        def _add_pair(ref1: str, ref2: str, dist_val: float) -> None:
            if not ref1 or not ref2 or ref1 == ref2:
                return
            key = tuple(sorted((ref1, ref2)))
            if key in pair_keys:
                return
            pair_keys.add(key)
            body_pairs.append((ref1, ref2, round(dist_val, 1)))
        for i, f1 in enumerate(planar_axis):
            for f2 in planar_axis[i+1:]:
                dot = sum(a*b for a, b in zip(f1["normal"], f2["normal"]))
                if dot < -0.99:  # Opposing normals
                    dist = abs(sum(
                        (c2 - c1) * n
                        for c1, c2, n in zip(f1["centroid"], f2["centroid"], f1["normal"])
                    ))
                    _add_pair(f1["ref"], f2["ref"], dist)

        # Fallback: if normals are inconsistent, pair extreme faces per axis
        axis_groups: Dict[str, List[Dict[str, Any]]] = {"x": [], "y": [], "z": []}
        for f in planar_axis:
            nx, ny, nz = f["normal"]
            abs_nx, abs_ny, abs_nz = abs(nx), abs(ny), abs(nz)
            if abs_nz >= abs_ny and abs_nz >= abs_nx:
                axis_groups["z"].append(f)
            elif abs_ny >= abs_nx:
                axis_groups["y"].append(f)
            else:
                axis_groups["x"].append(f)

        axis_idx_map = {"x": 0, "y": 1, "z": 2}
        for axis_key, faces_axis in axis_groups.items():
            if len(faces_axis) < 2:
                continue
            axis_idx = axis_idx_map[axis_key]
            min_face = min(faces_axis, key=lambda f: f["centroid"][axis_idx])
            max_face = max(faces_axis, key=lambda f: f["centroid"][axis_idx])
            if min_face["ref"] == max_face["ref"]:
                continue
            dist = abs(max_face["centroid"][axis_idx] - min_face["centroid"][axis_idx])
            _add_pair(max_face["ref"], min_face["ref"], dist)

        if body_pairs:
            pairs_by_body[body_key] = body_pairs[:MAX_PARALLEL_PAIRS_PER_BODY]

    spatial_context = entity_context.get("spatial_context")
    spatial_bodies: List[Mapping[str, Any]] = []
    if isinstance(spatial_context, dict):
        maybe_bodies = spatial_context.get("bodies", [])
        if isinstance(maybe_bodies, list):
            spatial_bodies = maybe_bodies

    if spatial_bodies:
        for body in spatial_bodies:
            body_key = body.get("entity_ref", body.get("id", "body_?"))
            _process_faces(body.get("faces", []), body_key)
    else:
        # Group faces by body name for flat format
        faces_by_body: Dict[str, List[Mapping[str, Any]]] = {}
        for face in entity_context.get("faces", []):
            body_name = face.get("body_name", "unknown")
            if body_name not in faces_by_body:
                faces_by_body[body_name] = []
            faces_by_body[body_name].append(face)

        for body_name, faces in faces_by_body.items():
            _process_faces(faces, body_name)

    return pairs_by_body


def _format_unified_context(entity_context: Mapping[str, Any]) -> str:
    """
    Format entity context as a unified compact block for LLM consumption.

    Replaces both _format_design_entities_xml and _format_entity_context
    with a single YAML-like format that includes:
    - Notation legend (explains !eN)
    - Bodies with bounding boxes
    - Faces with normals, centroids, adjacency, and loops
    - Edges with length, vertices, and adjacent faces
    - Vertices with positions
    - Spatial data: parallel face pairs per body

    This format is used for both initial user messages and tool result injection,
    eliminating the dual text+XML format redundancy.
    """
    if not entity_context:
        return ""

    # Determine format (spatial_context nested vs flat). Some clients include
    # an empty spatial_context.bodies alongside populated flat entities.
    spatial_context = entity_context.get("spatial_context")
    spatial_bodies: List[Mapping[str, Any]] = []
    if isinstance(spatial_context, dict):
        maybe_bodies = spatial_context.get("bodies", [])
        if isinstance(maybe_bodies, list):
            spatial_bodies = maybe_bodies
    is_nested = bool(spatial_bodies)

    lines = ["design_entities:"]

    # Units
    units = "mm"
    if is_nested:
        units = spatial_context.get("units", "mm")
    else:
        units = entity_context.get("units")
        if units is None and isinstance(spatial_context, dict):
            units = spatial_context.get("units")
        if units is None:
            units = "mm"
    lines.append(f"  units: {units}")

    # Notation legend
    lines.append('  notation: "!eN means reversed co-edge orientation for edge eN"')

    # Compute topology and parallel pairs
    try:
        topology = _compute_face_topology(entity_context)
    except Exception:
        topology = {}

    try:
        parallel_pairs = _compute_parallel_face_pairs(entity_context)
    except Exception:
        parallel_pairs = {}

    # Format bodies, faces, edges, vertices
    if is_nested:
        _format_unified_nested(spatial_context, lines, topology, parallel_pairs)
    else:
        _format_unified_flat(entity_context, lines, topology, parallel_pairs)

    return "\n".join(lines)


def _ref_sort_key(ref_id: str) -> Tuple[str, int, str]:
    """Sort refs naturally: face_2 before face_10, e2 before e10."""
    text = str(ref_id)
    match = re.match(r"^([A-Za-z_]+?)(\d+)$", text)
    if not match:
        match = re.match(r"^(e)(\d+)$", text)
    if match:
        return match.group(1), int(match.group(2)), text
    return text, -1, text


def _short_vec(value: Any, *, digits: int = 1) -> str:
    try:
        x, y, z = _safe_vec3_extract(value)
    except Exception:
        return "?"
    return f"[{x:.{digits}f},{y:.{digits}f},{z:.{digits}f}]"


def _entity_ref_id(entity: Mapping[str, Any], fallback: str = "?") -> str:
    ref = entity.get("entity_ref") or entity.get("id") or fallback
    return str(ref).strip() or fallback


def _format_current_ref_table(entity_context: Mapping[str, Any]) -> str:
    """Return a compact authoritative ref table for the current LLM turn."""
    if not entity_context:
        return ""

    entities = _collect_context_entities(entity_context)
    bodies = entities.get("bodies", [])
    faces = entities.get("faces", [])
    edges = entities.get("edges", [])
    if not bodies and not faces and not edges:
        return ""

    lines = [
        "current_design_refs:",
        "  authority: latest runtime entity refs; do not use refs absent from this table",
        "  note: list_features shows timeline features only; it does not refresh entity refs",
    ]

    if bodies:
        body_bits = []
        for body in sorted(bodies, key=lambda item: _ref_sort_key(_entity_ref_id(item))):
            ref_id = _entity_ref_id(body, "body_?")
            name = str(body.get("name") or "Unnamed")
            bbox = body.get("bbox") or body.get("bounding_box") or {}
            bbox_text = ""
            if isinstance(bbox, Mapping):
                mn = bbox.get("min")
                mx = bbox.get("max")
                if mn is not None and mx is not None:
                    bbox_text = f" bbox={_short_vec(mn)}..{_short_vec(mx)}"
            body_bits.append(f"{ref_id}({name}{bbox_text})")
        lines.append(f"  bodies: {', '.join(body_bits[:20])}")

    if faces:
        lines.append(f"  faces_count: {len(faces)}")
        lines.append("  faces:")
        for face in sorted(faces, key=lambda item: _ref_sort_key(_entity_ref_id(item)))[:80]:
            ref_id = _entity_ref_id(face, "face_?")
            body_ref = face.get("body_ref") or face.get("body") or face.get("body_name") or "?"
            surface_type = face.get("surface_type") or face.get("geometry_type") or ""
            normal = face.get("normal")
            centroid = face.get("centroid")
            bits = [f"body={body_ref}"]
            if surface_type:
                bits.append(f"type={surface_type}")
            if normal is not None:
                bits.append(f"n={_short_vec(normal, digits=2)}")
            if centroid is not None:
                bits.append(f"c={_short_vec(centroid)}")
            lines.append(f"    {ref_id}: {', '.join(bits)}")
        if len(faces) > 80:
            lines.append(f"    ... {len(faces) - 80} more faces omitted")

    if edges:
        edge_refs = [
            _entity_ref_id(edge, "e?")
            for edge in sorted(edges, key=lambda item: _ref_sort_key(_entity_ref_id(item)))
        ]
        lines.append(f"  edges_count: {len(edge_refs)}")
        for start in range(0, len(edge_refs), 32):
            chunk = edge_refs[start:start + 32]
            lines.append(f"  edges_{start + 1}_{start + len(chunk)}: {', '.join(chunk)}")
            if start >= 128:
                remaining = len(edge_refs) - (start + len(chunk))
                if remaining > 0:
                    lines.append(f"  edges_omitted: {remaining}")
                break

    return "\n".join(lines)


def _fmt_loop_entry_unified(entry: Any) -> str:
    """Format a single loop co-edge entry with !eN notation."""
    if not isinstance(entry, dict):
        return ""
    edge_id = entry.get("edge") or entry.get("edge_token") or ""
    if not edge_id:
        return ""
    return f"!{edge_id}" if entry.get("isOpposedToEdge") else edge_id


def _format_unified_nested(spatial_context: Mapping[str, Any], lines: List[str],
                            topology: Dict[str, List[str]],
                            parallel_pairs: Dict[str, List[Tuple[str, str, float]]]) -> None:
    """Format nested spatial_context structure in unified format."""
    bodies = spatial_context.get("bodies", [])

    if not bodies:
        return

    # Bodies section
    lines.append("")
    lines.append("  bodies:")
    for body in bodies:
        body_id = body.get("entity_ref", body.get("id", "body_?"))
        body_name = body.get("name", "Unnamed")
        bbox = body.get("bbox", {})
        bbox_str = ""
        if bbox:
            mn_x, mn_y, mn_z = _safe_vec3_extract(bbox.get("min"))
            mx_x, mx_y, mx_z = _safe_vec3_extract(bbox.get("max"))
            bbox_str = f"  bbox: [[{mn_x:.1f},{mn_y:.1f},{mn_z:.1f}],[{mx_x:.1f},{mx_y:.1f},{mx_z:.1f}]]"
        lines.append(f"    {body_id}: {{name: {body_name}{', ' + bbox_str.strip() if bbox_str else ''}}}")

    # Faces section — build a (face, body_ref) list without mutating input dicts
    all_faces_with_body: List[Tuple[Mapping[str, Any], str]] = []
    for body in bodies:
        body_id = body.get("entity_ref", body.get("id", "body_?"))
        for face in body.get("faces", []):
            all_faces_with_body.append((face, body_id))

    if all_faces_with_body:
        lines.append("")
        lines.append("  faces:")
        for face, body_ref in all_faces_with_body:
            f_id = face.get("entity_ref", face.get("id", "f_?"))
            surface_type = face.get("surface_type", "")
            normal = face.get("normal", [0, 0, 1])
            centroid = face.get("centroid", [0, 0, 0])

            nx, ny, nz = _safe_vec3_extract(normal)
            cx, cy, cz = _safe_vec3_extract(centroid)

            parts = [f"body: {body_ref}"]
            if surface_type:
                parts.append(f"type: {surface_type}")
            parts.append(f"n: [{nx:.2f},{ny:.2f},{nz:.2f}]")
            parts.append(f"c: [{cx:.1f},{cy:.1f},{cz:.1f}]")

            # Adjacency from topology
            adj = topology.get(f_id)
            if adj:
                parts.append(f"adjacent: [{', '.join(adj)}]")

            # Loops
            loops = face.get("loops")
            if loops and isinstance(loops, dict):
                outer = loops.get("outer", [])
                if outer:
                    edge_ids = [_fmt_loop_entry_unified(e) for e in outer]
                    edge_ids = [e for e in edge_ids if e]
                    if edge_ids:
                        parts.append(f"loop: [{', '.join(edge_ids)}]")

                inner = loops.get("inner", [])
                if inner:
                    inner_strs = []
                    for inner_loop in inner:
                        entries = [_fmt_loop_entry_unified(e) for e in inner_loop if isinstance(e, dict)]
                        entries = [e for e in entries if e]
                        if entries:
                            inner_strs.append(f"[{', '.join(entries)}]")
                    if inner_strs:
                        parts.append(f"inner: [{', '.join(inner_strs)}]")

            lines.append(f"    {f_id}: {{{', '.join(parts)}}}")

    # Edges section
    all_edges = []
    for body in bodies:
        for edge in body.get("edges", []):
            all_edges.append(edge)

    if all_edges:
        lines.append("")
        lines.append("  edges:")
        for edge in all_edges:
            e_id = edge.get("entity_ref", edge.get("id", "e?"))
            edge_type = edge.get("edge_type", "")
            length = edge.get("length", 0)
            v0 = edge.get("v0_ref", edge.get("v0", ""))
            v1 = edge.get("v1_ref", edge.get("v1", ""))
            adj_faces = edge.get("adjacent_faces", [])

            parts = []
            if edge_type:
                parts.append(f"type: {edge_type}")
            try:
                parts.append(f"len: {float(length):.2f}")
            except (TypeError, ValueError):
                pass
            if v0 and v1:
                parts.append(f"v: [{v0}, {v1}]")
            if adj_faces:
                parts.append(f"f: [{', '.join(adj_faces)}]")

            lines.append(f"    {e_id}: {{{', '.join(parts)}}}")

    # Vertices section
    all_vertices = []
    for body in bodies:
        for v in body.get("vertices", []):
            all_vertices.append(v)

    if all_vertices:
        lines.append("")
        lines.append("  vertices:")
        for v in all_vertices:
            v_id = v.get("entity_ref", v.get("id", "v?"))
            p = v.get("p", [0, 0, 0])
            px, py, pz = _safe_vec3_extract(p)
            lines.append(f"    {v_id}: [{px:.2f}, {py:.2f}, {pz:.2f}]")

    # Spatial section (parallel pairs)
    if parallel_pairs:
        lines.append("")
        lines.append("  spatial:")
        for body_key, pairs in parallel_pairs.items():
            lines.append(f"    parallel_{body_key}:")
            for f1, f2, dist in pairs:
                lines.append(f"      - [{f1}, {f2}, {dist}]")


def _format_unified_flat(entity_context: Mapping[str, Any], lines: List[str],
                          topology: Dict[str, List[str]],
                          parallel_pairs: Dict[str, List[Tuple[str, str, float]]]) -> None:
    """Format flat entity context structure in unified format."""
    bodies = entity_context.get("bodies", [])
    faces = entity_context.get("faces", [])
    edges = entity_context.get("edges", [])
    vertices = entity_context.get("vertices", [])

    if not bodies and not faces and not edges and not vertices:
        return

    # Bodies
    if bodies:
        lines.append("")
        lines.append("  bodies:")
        for body in bodies:
            ref_id = body.get("entity_ref", body.get("id", "?"))
            name = body.get("name", "Unnamed")
            bbox = body.get("bounding_box", {})
            bbox_str = ""
            if bbox:
                min_x, min_y, min_z = _safe_vec3_extract(bbox.get("min"))
                max_x, max_y, max_z = _safe_vec3_extract(bbox.get("max"))
                bbox_str = f", bbox: [[{min_x:.1f},{min_y:.1f},{min_z:.1f}],[{max_x:.1f},{max_y:.1f},{max_z:.1f}]]"
            lines.append(f"    {ref_id}: {{name: {name}{bbox_str}}}")

    # Faces
    if faces:
        lines.append("")
        lines.append("  faces:")
        for face in faces:
            ref_id = face.get("entity_ref", face.get("id", "?"))
            body_name = face.get("body_name", "Unknown")
            surface_type = face.get("surface_type", "")
            normal = face.get("normal")
            centroid = face.get("centroid")

            parts = [f"body: {body_name}"]
            if surface_type:
                parts.append(f"type: {surface_type}")
            if normal:
                nx, ny, nz = _safe_vec3_extract(normal)
                parts.append(f"n: [{nx:.2f},{ny:.2f},{nz:.2f}]")
            if centroid:
                cx, cy, cz = _safe_vec3_extract(centroid)
                parts.append(f"c: [{cx:.1f},{cy:.1f},{cz:.1f}]")

            # Adjacency
            adj = topology.get(ref_id)
            if adj:
                parts.append(f"adjacent: [{', '.join(adj)}]")

            # Loops
            loops = face.get("loops")
            if loops and isinstance(loops, dict):
                outer = loops.get("outer", [])
                if outer:
                    edge_ids = [_fmt_loop_entry_unified(e) for e in outer]
                    edge_ids = [e for e in edge_ids if e]
                    if edge_ids:
                        parts.append(f"loop: [{', '.join(edge_ids)}]")

                inner = loops.get("inner", [])
                if inner:
                    inner_strs = []
                    for inner_loop in inner:
                        entries = [_fmt_loop_entry_unified(e) for e in inner_loop if isinstance(e, dict)]
                        entries = [e for e in entries if e]
                        if entries:
                            inner_strs.append(f"[{', '.join(entries)}]")
                    if inner_strs:
                        parts.append(f"inner: [{', '.join(inner_strs)}]")

            lines.append(f"    {ref_id}: {{{', '.join(parts)}}}")

    # Edges
    if edges:
        lines.append("")
        lines.append("  edges:")
        for edge in edges:
            ref_id = edge.get("entity_ref", edge.get("id", "?"))
            edge_type = edge.get("edge_type", "")
            length = edge.get("length", 0)
            v0 = edge.get("v0", "")
            v1 = edge.get("v1", "")
            adj_faces = edge.get("adjacent_faces", [])

            parts = []
            if edge_type:
                parts.append(f"type: {edge_type}")
            try:
                parts.append(f"len: {float(length):.2f}")
            except (TypeError, ValueError):
                pass
            if v0 and v1:
                parts.append(f"v: [{v0}, {v1}]")
            if adj_faces:
                parts.append(f"f: [{', '.join(str(f) for f in adj_faces)}]")

            lines.append(f"    {ref_id}: {{{', '.join(parts)}}}")

    # Vertices
    if vertices:
        lines.append("")
        lines.append("  vertices:")
        for v in vertices:
            ref_id = v.get("entity_ref", v.get("id", "?"))
            pos = v.get("p") or v.get("position")
            if pos:
                px, py, pz = _safe_vec3_extract(pos)
                lines.append(f"    {ref_id}: [{px:.2f}, {py:.2f}, {pz:.2f}]")
            else:
                lines.append(f"    {ref_id}: []")

    # Parallel pairs
    if parallel_pairs:
        lines.append("")
        lines.append("  spatial:")
        for body_key, pairs in parallel_pairs.items():
            lines.append(f"    parallel_{body_key}:")
            for f1, f2, dist in pairs:
                lines.append(f"      - [{f1}, {f2}, {dist}]")


def _format_design_entities_xml(entity_context: Mapping[str, Any]) -> str:
    """
    Format entity context as <design_entities> XML block for LLM consumption.

    This fulfills the Entity Data Contract promised in the prompt:
    - Bodies with ref, name, bounding box
    - Vertices with ref, position
    - Faces with ref, body, normal, centroid, surface_type, loops
    - Edges with ref, associated faces, length, tangent, vertices

    New ID scheme (SPATIAL_CONTEXT_OVERHAUL.md):
    - Bodies: body_0, body_1, ...
    - Vertices: v0, v1, v2, ...
    - Faces: face_0, face_1, face_2, ... (sequential)
    - Edges: e0, e1, e2, ...

    Args:
        entity_context: Dictionary with bodies, faces, edges, vertices arrays
                       or spatial_context with nested per-body structure

    Returns:
        XML-formatted string with all entity information
    """
    if not entity_context:
        return ""

    # Check for new nested spatial_context structure
    spatial_context = entity_context.get("spatial_context")
    spatial_bodies: List[Mapping[str, Any]] = []
    if isinstance(spatial_context, dict):
        maybe_bodies = spatial_context.get("bodies", [])
        if isinstance(maybe_bodies, list):
            spatial_bodies = maybe_bodies
    if spatial_bodies:
        return _format_spatial_context_xml(spatial_context)

    # Fall back to flat structure
    bodies = entity_context.get("bodies", [])
    faces = entity_context.get("faces", [])
    edges = entity_context.get("edges", [])
    vertices = entity_context.get("vertices", [])

    if not bodies and not faces and not edges and not vertices:
        return ""

    lines = ["<design_entities>"]

    # Format bodies
    if bodies:
        lines.append("  <bodies>")
        for body in bodies:
            ref_id = body.get("entity_ref", body.get("id", "?"))
            name = body.get("name", "Unnamed")
            bbox = body.get("bounding_box", {})
            bbox_str = ""
            if bbox:
                # Safe extraction handles both dict and list formats from Fusion
                min_x, min_y, min_z = _safe_vec3_extract(bbox.get("min"))
                max_x, max_y, max_z = _safe_vec3_extract(bbox.get("max"))
                bbox_str = f' bounding="[[{min_x:.1f},{min_y:.1f},{min_z:.1f}],[{max_x:.1f},{max_y:.1f},{max_z:.1f}]]"'
            lines.append(f'    <body ref="{ref_id}" name="{name}"{bbox_str} />')
        lines.append("  </bodies>")

    # Format vertices (new)
    if vertices:
        lines.append("  <vertices>")
        for vertex in vertices:
            ref_id = vertex.get("entity_ref", vertex.get("id", "?"))
            pos = vertex.get("p") or vertex.get("position")
            pos_str = ""
            if pos:
                px, py, pz = _safe_vec3_extract(pos)
                pos_str = f' p="[{px:.2f},{py:.2f},{pz:.2f}]"'
            lines.append(f'    <vertex ref="{ref_id}"{pos_str} />')
        lines.append("  </vertices>")

    # Format faces
    if faces:
        lines.append("  <faces>")
        for face in faces:
            ref_id = face.get("entity_ref", face.get("id", "?"))
            body_name = face.get("body_name", "Unknown")
            normal = face.get("normal")
            centroid = face.get("centroid")
            surface_type = face.get("surface_type", "")

            attrs = [f'ref="{ref_id}"', f'body="{body_name}"']
            if surface_type:
                attrs.append(f'type="{surface_type}"')
            if normal:
                # Safe extraction handles both dict and list formats from Fusion
                nx, ny, nz = _safe_vec3_extract(normal)
                attrs.append(f'normal="[{nx:.2f},{ny:.2f},{nz:.2f}]"')
            if centroid:
                # Safe extraction handles both dict and list formats from Fusion
                cx, cy, cz = _safe_vec3_extract(centroid)
                attrs.append(f'centroid="[{cx:.1f},{cy:.1f},{cz:.1f}]"')

            # Add frame (u/v/n vectors) if present
            frame = face.get("frame")
            if frame and isinstance(frame, dict):
                u = frame.get("u", [0, 0, 0])
                v = frame.get("v", [0, 0, 0])
                n = frame.get("n", [0, 0, 0])
                frame_str = f'u=[{u[0]:.2f},{u[1]:.2f},{u[2]:.2f}] v=[{v[0]:.2f},{v[1]:.2f},{v[2]:.2f}] n=[{n[0]:.2f},{n[1]:.2f},{n[2]:.2f}]'
                attrs.append(f'frame="{frame_str}"')

            # Add loops (outer loop edge IDs) if present
            loops = face.get("loops")
            if loops and isinstance(loops, dict):
                outer = loops.get("outer", [])
                if outer:
                    edge_ids = [entry.get("edge", "") for entry in outer if isinstance(entry, dict)]
                    edge_ids = [eid for eid in edge_ids if eid]  # Filter empty
                    if edge_ids:
                        attrs.append(f'loops="{",".join(edge_ids)}"')

            lines.append(f'    <face {" ".join(attrs)} />')
        lines.append("  </faces>")

    # Format edges
    if edges:
        lines.append("  <edges>")
        for edge in edges:
            ref_id = edge.get("entity_ref", edge.get("id", "?"))
            length = edge.get("length", 0)
            edge_type = edge.get("edge_type", "")
            # Adjacent faces (using ref IDs)
            adjacent_faces = edge.get("adjacent_faces", [])
            faces_str = ",".join(str(f) for f in adjacent_faces) if adjacent_faces else ""
            # Vertex references (new)
            v0_ref = edge.get("v0", "")
            v1_ref = edge.get("v1", "")
            # Tangent at midpoint (new)
            tangent = edge.get("tangent")
            
            attrs = [f'ref="{ref_id}"']
            if edge_type:
                attrs.append(f'type="{edge_type}"')
            if faces_str:
                attrs.append(f'faces="{faces_str}"')
            if v0_ref and v1_ref:
                attrs.append(f'vertices="{v0_ref},{v1_ref}"')
            attrs.append(f'length="{length:.2f}"')
            if tangent:
                tx, ty, tz = _safe_vec3_extract(tangent)
                attrs.append(f'tangent="[{tx:.2f},{ty:.2f},{tz:.2f}]"')
            
            lines.append(f'    <edge {" ".join(attrs)} />')
        lines.append("  </edges>")

    lines.append("</design_entities>")
    return "\n".join(lines)


def _format_spatial_context_xml(spatial_context: Mapping[str, Any]) -> str:
    """
    Format the new nested spatial_context structure as XML.
    
    Structure: { "units": "mm", "bodies": [{ "id", "vertices", "faces", "edges" }] }
    """
    lines = ["<design_entities>"]
    
    units = spatial_context.get("units", "mm")
    lines.append(f'  <units>{units}</units>')
    
    bodies = spatial_context.get("bodies", [])
    
    for body in bodies:
        body_id = body.get("entity_ref", body.get("id", "body_?"))
        body_name = body.get("name", "Unnamed")
        bbox = body.get("bbox", {})
        
        bbox_attrs = ""
        if bbox:
            min_x, min_y, min_z = _safe_vec3_extract(bbox.get("min"))
            max_x, max_y, max_z = _safe_vec3_extract(bbox.get("max"))
            bbox_attrs = (
                f' bbox_min="[{min_x:.1f},{min_y:.1f},{min_z:.1f}]"'
                f' bbox_max="[{max_x:.1f},{max_y:.1f},{max_z:.1f}]"'
            )
        
        lines.append(f'  <body ref="{body_id}" name="{body_name}"{bbox_attrs}>')
        
        # Vertices
        vertices = body.get("vertices", [])
        if vertices:
            lines.append("    <vertices>")
            for v in vertices:
                v_id = v.get("entity_ref", v.get("id", "v?"))
                px, py, pz = _safe_vec3_extract(v.get("p"))
                lines.append(f'      <vertex ref="{v_id}" p="[{px:.2f},{py:.2f},{pz:.2f}]" />')
            lines.append("    </vertices>")
        
        # Faces
        faces = body.get("faces", [])
        if faces:
            lines.append("    <faces>")
            for f in faces:
                f_id = f.get("entity_ref", f.get("id", "f_?"))
                surface_type = f.get("surface_type", "")
                cx, cy, cz = _safe_vec3_extract(f.get("centroid"))
                nx, ny, nz = _safe_vec3_extract(f.get("normal"))
                
                attrs = [f'ref="{f_id}"']
                if surface_type:
                    attrs.append(f'type="{surface_type}"')
                attrs.append(f'normal="[{nx:.2f},{ny:.2f},{nz:.2f}]"')
                attrs.append(f'centroid="[{cx:.1f},{cy:.1f},{cz:.1f}]"')

                # Add frame (u/v/n vectors) if present
                frame = f.get("frame")
                if frame and isinstance(frame, dict):
                    ux, uy, uz = _safe_vec3_extract(frame.get("u"))
                    vx, vy, vz = _safe_vec3_extract(frame.get("v"))
                    fnx, fny, fnz = _safe_vec3_extract(frame.get("n"))
                    frame_str = (
                        f'u=[{ux:.2f},{uy:.2f},{uz:.2f}] '
                        f'v=[{vx:.2f},{vy:.2f},{vz:.2f}] '
                        f'n=[{fnx:.2f},{fny:.2f},{fnz:.2f}]'
                    )
                    attrs.append(f'frame="{frame_str}"')

                # Add loops (outer/inner) if present
                loops = f.get("loops")
                if loops and isinstance(loops, dict):
                    outer = loops.get("outer", [])
                    if outer:
                        def _fmt_loop_entry(entry: Any) -> str:
                            if not isinstance(entry, dict):
                                return ""
                            edge_id = entry.get("edge") or entry.get("edge_token") or ""
                            if not edge_id:
                                return ""
                            return f"!{edge_id}" if entry.get("isOpposedToEdge") else edge_id

                        edge_ids = [_fmt_loop_entry(entry) for entry in outer]
                        edge_ids = [eid for eid in edge_ids if eid]  # Filter empty
                        if edge_ids:
                            attrs.append(f'outer_loop="{",".join(edge_ids)}"')

                    inner = loops.get("inner", [])
                    if inner:
                        inner_loops = []
                        for inner_loop in inner:
                            entries = [_fmt_loop_entry(entry) for entry in inner_loop if isinstance(entry, dict)]
                            entries = [e for e in entries if e]
                            if entries:
                                inner_loops.append(",".join(entries))
                        if inner_loops:
                            attrs.append(f'inner_loops="{"|".join(inner_loops)}"')

                lines.append(f'      <face {" ".join(attrs)} />')
            lines.append("    </faces>")
        
        # Edges
        edges = body.get("edges", [])
        if edges:
            lines.append("    <edges>")
            for e in edges:
                e_id = e.get("entity_ref", e.get("id", "e?"))
                edge_type = e.get("edge_type", "")
                length = e.get("length", 0)
                v0 = e.get("v0_ref", e.get("v0", ""))
                v1 = e.get("v1_ref", e.get("v1", ""))
                adj_faces = e.get("adjacent_faces", [])
                
                attrs = [f'ref="{e_id}"']
                if edge_type:
                    attrs.append(f'type="{edge_type}"')
                if v0 and v1:
                    attrs.append(f'vertices="{v0},{v1}"')
                if adj_faces:
                    attrs.append(f'faces="{",".join(str(face) for face in adj_faces)}"')
                try:
                    attrs.append(f'length="{float(length):.2f}"')
                except (TypeError, ValueError):
                    pass
                
                lines.append(f'      <edge {" ".join(attrs)} />')
            lines.append("    </edges>")
        
        lines.append("  </body>")
    
    lines.append("</design_entities>")
    return "\n".join(lines)


def _format_entity_context(entity_context: Mapping[str, Any]) -> str:
    """
    Format entity context (bodies, faces, edges, vertices) for inclusion in user message.

    This replaces the need for list_edges/list_faces/list_bodies tool calls by
    providing the LLM with complete entity awareness up front.

    New ID scheme (SPATIAL_CONTEXT_OVERHAUL.md):
    - Bodies: body_0, body_1, ...
    - Vertices: v0, v1, v2, ...
    - Faces: face_0, face_1, face_2, ... (sequential)
    - Edges: e0, e1, e2, ...

    Args:
        entity_context: Dictionary with bodies, faces, edges, vertices arrays and units
                       or spatial_context with nested per-body structure

    Returns:
        Formatted string with all entity information
    """
    if not entity_context:
        return ""

    # Check for new nested spatial_context structure
    spatial_context = entity_context.get("spatial_context")
    spatial_bodies: List[Mapping[str, Any]] = []
    if isinstance(spatial_context, dict):
        maybe_bodies = spatial_context.get("bodies", [])
        if isinstance(maybe_bodies, list):
            spatial_bodies = maybe_bodies
    if spatial_bodies:
        return _format_spatial_context_text(spatial_context)

    # Fall back to flat structure
    bodies = entity_context.get("bodies", [])
    faces = entity_context.get("faces", [])
    edges = entity_context.get("edges", [])
    vertices = entity_context.get("vertices", [])

    if not bodies and not faces and not edges and not vertices:
        return ""

    sections = ["Design Entities:"]

    # Format bodies section
    if bodies:
        body_units = entity_context.get("body_units", {})
        length_unit = body_units.get("length", "mm") if isinstance(body_units, dict) else "mm"
        volume_unit = body_units.get("volume", "mm^3") if isinstance(body_units, dict) else "mm^3"

        sections.append(f"\nBodies ({len(bodies)} total, length: {length_unit}, volume: {volume_unit}):")
        for body in bodies:
            ref_id = body.get("entity_ref", body.get("id", "?"))
            name = body.get("name", "Unnamed")
            volume = body.get("volume", 0)
            face_count = body.get("face_count", 0)
            edge_count = body.get("edge_count", 0)
            is_solid = body.get("is_solid", False)
            bbox = body.get("bounding_box", {})

            body_line = f"  {ref_id}: '{name}' - {volume} {volume_unit}"
            if is_solid:
                body_line += " (solid)"
            body_line += f", {face_count} faces, {edge_count} edges"

            if bbox:
                min_pt = bbox.get("min")
                max_pt = bbox.get("max")
                if min_pt and max_pt:
                    # Safe extraction handles both dict and list formats from Fusion
                    min_x, min_y, min_z = _safe_vec3_extract(min_pt)
                    max_x, max_y, max_z = _safe_vec3_extract(max_pt)
                    body_line += f", bbox: [{min_x:.1f},{min_y:.1f},{min_z:.1f}] to [{max_x:.1f},{max_y:.1f},{max_z:.1f}]"

            sections.append(body_line)

    # Format vertices section (new)
    if vertices:
        sections.append(f"\nVertices ({len(vertices)} total):")
        # Group vertices by body
        vertices_by_body: Dict[str, List[Mapping[str, Any]]] = {}
        for vertex in vertices:
            body_name = vertex.get("body_name", "Unknown")
            if body_name not in vertices_by_body:
                vertices_by_body[body_name] = []
            vertices_by_body[body_name].append(vertex)

        for body_name, body_vertices in vertices_by_body.items():
            sections.append(f"  Body '{body_name}':")
            # Limit display to avoid overwhelming the LLM
            display_count = min(len(body_vertices), 20)
            for vertex in body_vertices[:display_count]:
                ref_id = vertex.get("entity_ref", vertex.get("id", "?"))
                pos = vertex.get("p") or vertex.get("position")
                if pos:
                    px, py, pz = _safe_vec3_extract(pos)
                    sections.append(f"    {ref_id}: ({px:.2f},{py:.2f},{pz:.2f})")
                else:
                    sections.append(f"    {ref_id}")
            if len(body_vertices) > display_count:
                sections.append(f"    ... ({len(body_vertices) - display_count} more)")

    # Format faces section (grouped by body for readability)
    if faces:
        face_units = entity_context.get("face_units", {})
        area_unit = face_units.get("area", "mm^2") if isinstance(face_units, dict) else "mm^2"

        sections.append(f"\nFaces ({len(faces)} total, area: {area_unit}):")

        # Group faces by body
        faces_by_body: Dict[str, List[Mapping[str, Any]]] = {}
        for face in faces:
            body_name = face.get("body_name", "Unknown")
            if body_name not in faces_by_body:
                faces_by_body[body_name] = []
            faces_by_body[body_name].append(face)

        for body_name, body_faces in faces_by_body.items():
            sections.append(f"  Body '{body_name}':")
            for face in body_faces:
                ref_id = face.get("entity_ref", face.get("id", "?"))
                geo_type = face.get("surface_type") or face.get("geometry_type", "Unknown")
                area = face.get("area", 0)
                centroid = face.get("centroid")
                normal = face.get("normal")
                orientation = face.get("orientation")

                face_line = f"    {ref_id}: {geo_type}"
                if area:
                    face_line += f", area={area:.2f}"

                if centroid:
                    # Safe extraction handles both dict and list formats from Fusion
                    cx, cy, cz = _safe_vec3_extract(centroid)
                    face_line += f", centroid=({cx:.1f},{cy:.1f},{cz:.1f})"

                if normal:
                    # Safe extraction handles both dict and list formats from Fusion
                    nx, ny, nz = _safe_vec3_extract(normal)
                    face_line += f", normal=({nx:.2f},{ny:.2f},{nz:.2f})"

                if orientation:
                    face_line += f" [{orientation}]"

                sections.append(face_line)

    # Format edges section (grouped by body for readability)
    if edges:
        edge_units = entity_context.get("edge_units", "mm")
        if isinstance(edge_units, dict):
            edge_units = edge_units.get("length", "mm")

        sections.append(f"\nEdges ({len(edges)} total, length: {edge_units}):")

        # Group edges by body
        edges_by_body: Dict[str, List[Mapping[str, Any]]] = {}
        for edge in edges:
            body_name = edge.get("body_name", "Unknown")
            if body_name not in edges_by_body:
                edges_by_body[body_name] = []
            edges_by_body[body_name].append(edge)

        for body_name, body_edges in edges_by_body.items():
            sections.append(f"  Body '{body_name}':")
            for edge in body_edges:
                ref_id = edge.get("entity_ref", edge.get("id", "?"))
                geo_type = edge.get("edge_type") or edge.get("geometry_type", "Unknown")
                length = edge.get("length", 0)
                start = edge.get("start_coords") or edge.get("p0")
                end = edge.get("end_coords") or edge.get("p1")
                adj_faces = edge.get("adjacent_faces", [])
                v0 = edge.get("v0", "")
                v1 = edge.get("v1", "")

                edge_line = f"    {ref_id}: {geo_type}, length={length:.2f}"

                if v0 and v1:
                    edge_line += f", vertices={v0}→{v1}"

                if start and end:
                    sx, sy, sz = _safe_vec3_extract(start)
                    ex, ey, ez = _safe_vec3_extract(end)
                    edge_line += f", from=({sx:.1f},{sy:.1f},{sz:.1f}) to=({ex:.1f},{ey:.1f},{ez:.1f})"

                if adj_faces:
                    edge_line += f", adjacent_faces=[{','.join(str(f) for f in adj_faces)}]"

                sections.append(edge_line)

    # Add spatial summary section for quick LLM lookups
    spatial_summary = _compute_spatial_summary_from_context(entity_context)
    if spatial_summary:
        sections.append("\nSpatial Summary (USE THIS FIRST for top/bottom/horizontal/vertical classification):")
        
        # Top/bottom faces
        top_faces = spatial_summary.get("top_faces", [])
        bottom_faces = spatial_summary.get("bottom_faces", [])
        if top_faces:
            sections.append(f"  Top faces (horizontal, facing +Z, highest): {', '.join(top_faces)}")
        if bottom_faces:
            sections.append(f"  Bottom faces (horizontal, facing -Z, lowest): {', '.join(bottom_faces)}")
        
        # Horizontal/vertical faces
        horizontal_faces = spatial_summary.get("horizontal_faces", [])
        vertical_faces = spatial_summary.get("vertical_faces", [])
        if horizontal_faces:
            sections.append(f"  Horizontal faces (±Z normal): {', '.join(horizontal_faces)}")
        if vertical_faces:
            # Limit display to avoid overwhelming the LLM
            if len(vertical_faces) > 10:
                sections.append(f"  Vertical faces (XY normal): {', '.join(vertical_faces[:10])}... ({len(vertical_faces)} total)")
            else:
                sections.append(f"  Vertical faces (XY normal): {', '.join(vertical_faces)}")
        
        # Top/bottom edges
        top_edges = spatial_summary.get("top_edges", [])
        bottom_edges = spatial_summary.get("bottom_edges", [])
        if top_edges:
            if len(top_edges) > 10:
                sections.append(f"  Top edges (at max Z): {', '.join(top_edges[:10])}... ({len(top_edges)} total)")
            else:
                sections.append(f"  Top edges (at max Z): {', '.join(top_edges)}")
        if bottom_edges:
            if len(bottom_edges) > 10:
                sections.append(f"  Bottom edges (at min Z): {', '.join(bottom_edges[:10])}... ({len(bottom_edges)} total)")
            else:
                sections.append(f"  Bottom edges (at min Z): {', '.join(bottom_edges)}")
        
        # Z bounds
        z_bounds = spatial_summary.get("z_bounds")
        if z_bounds:
            sections.append(f"  Z bounds: min={z_bounds[0]:.2f}, max={z_bounds[1]:.2f}")

    sections.append("\nUse entity_ref IDs (e.g., body_0, face_0, e0, v0) when calling select_* tools.")

    return "\n".join(sections)


def _format_spatial_context_text(spatial_context: Mapping[str, Any]) -> str:
    """
    Format the new nested spatial_context structure as text.
    
    Structure: { "units": "mm", "bodies": [{ "id", "vertices", "faces", "edges" }] }
    """
    sections = ["Design Entities:"]
    
    units = spatial_context.get("units", "mm")
    sections.append(f"Units: {units}")
    
    bodies = spatial_context.get("bodies", [])
    
    for body in bodies:
        body_id = body.get("entity_ref", body.get("id", "body_?"))
        body_name = body.get("name", "Unnamed")
        bbox = body.get("bbox", {})
        
        body_line = f"\nBody {body_id}: '{body_name}'"
        if bbox:
            min_pt = bbox.get("min", [0, 0, 0])
            max_pt = bbox.get("max", [0, 0, 0])
            body_line += f", bbox: [{min_pt[0]:.1f},{min_pt[1]:.1f},{min_pt[2]:.1f}] to [{max_pt[0]:.1f},{max_pt[1]:.1f},{max_pt[2]:.1f}]"
        sections.append(body_line)
        
        # Vertices (summarized)
        vertices = body.get("vertices", [])
        if vertices:
            sections.append(f"  Vertices: {len(vertices)} total")
            # Show first few
            for v in vertices[:5]:
                v_id = v.get("entity_ref", v.get("id", "v?"))
                p = v.get("p", [0, 0, 0])
                sections.append(f"    {v_id}: ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})")
            if len(vertices) > 5:
                sections.append(f"    ... ({len(vertices) - 5} more)")
        
        # Faces
        faces = body.get("faces", [])
        if faces:
            sections.append(f"  Faces: {len(faces)} total")
            for f in faces:
                f_id = f.get("entity_ref", f.get("id", "f_?"))
                surface_type = f.get("surface_type", "")
                normal = f.get("normal", [0, 0, 1])
                centroid = f.get("centroid", [0, 0, 0])
                
                face_line = f"    {f_id}: {surface_type}"
                face_line += f", normal=({normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f})"
                face_line += f", centroid=({centroid[0]:.1f},{centroid[1]:.1f},{centroid[2]:.1f})"

                # Frame (u/v/n) if present
                frame = f.get("frame")
                if frame and isinstance(frame, dict):
                    u = frame.get("u", [0, 0, 0])
                    v = frame.get("v", [0, 0, 0])
                    n = frame.get("n", [0, 0, 0])
                    face_line += f", frame.u=({u[0]:.2f},{u[1]:.2f},{u[2]:.2f})"
                    face_line += f", frame.v=({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})"
                    face_line += f", frame.n=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f})"

                # Loops (outer/inner) if present
                loops = f.get("loops")
                if loops and isinstance(loops, dict):
                    def _fmt_loop_entry(entry: Any) -> str:
                        if not isinstance(entry, dict):
                            return ""
                        edge_id = entry.get("edge") or entry.get("edge_token") or ""
                        if not edge_id:
                            return ""
                        return f"!{edge_id}" if entry.get("isOpposedToEdge") else edge_id

                    outer = loops.get("outer", [])
                    if outer:
                        outer_edges = [_fmt_loop_entry(e) for e in outer]
                        outer_edges = [e for e in outer_edges if e]
                        if outer_edges:
                            face_line += f", outer=[{','.join(outer_edges)}]"
                    inner = loops.get("inner", [])
                    if inner:
                        inner_loops = []
                        for inner_loop in inner:
                            entries = [_fmt_loop_entry(e) for e in inner_loop if isinstance(e, dict)]
                            entries = [e for e in entries if e]
                            if entries:
                                inner_loops.append(",".join(entries))
                        if inner_loops:
                            face_line += f", inner=[{'|'.join(inner_loops)}]"
                sections.append(face_line)
        
        # Edges (summarized)
        edges = body.get("edges", [])
        if edges:
            sections.append(f"  Edges: {len(edges)} total")
            # Show first few with adjacency
            for e in edges[:10]:
                e_id = e.get("entity_ref", e.get("id", "e?"))
                edge_type = e.get("edge_type", "")
                length = e.get("length", 0)
                adj_faces = e.get("adjacent_faces", [])
                v0 = e.get("v0_ref", e.get("v0", ""))
                v1 = e.get("v1_ref", e.get("v1", ""))
                
                edge_line = f"    {e_id}: {edge_type}, length={length:.2f}"
                if v0 and v1:
                    edge_line += f", {v0}→{v1}"
                if adj_faces:
                    edge_line += f", borders [{','.join(adj_faces)}]"
                sections.append(edge_line)
            if len(edges) > 10:
                sections.append(f"    ... ({len(edges) - 10} more)")
    
    sections.append("\nUse entity_ref IDs (e.g., body_0, face_0, e0, v0) when calling select_* tools.")
    
    return "\n".join(sections)


def _compute_spatial_summary_from_context(entity_context: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Compute spatial summary directly from entity context data.
    
    This mirrors EntityStore.compute_spatial_summary() but works on raw context
    without requiring access to the EntityStore.
    """
    faces = entity_context.get("faces", [])
    edges = entity_context.get("edges", [])
    bodies = entity_context.get("bodies", [])
    
    if not faces and not edges:
        return {}
    
    # Find Z bounds from bodies
    max_z = float('-inf')
    min_z = float('inf')
    for body in bodies:
        bbox = body.get("bounding_box", {})
        if bbox:
            min_pt = bbox.get("min")
            max_pt = bbox.get("max")
            if min_pt is not None:
                _, _, z = _safe_vec3_extract(min_pt)
                min_z = min(min_z, z)
            if max_pt is not None:
                _, _, z = _safe_vec3_extract(max_pt)
                max_z = max(max_z, z)
    
    # If no bodies, use faces/edges for Z bounds
    if max_z == float('-inf') or min_z == float('inf'):
        for face in faces:
            centroid = face.get("centroid")
            if centroid:
                _, _, cz = _safe_vec3_extract(centroid)
                max_z = max(max_z, cz)
                min_z = min(min_z, cz)
        for edge in edges:
            midpoint = edge.get("midpoint")
            if midpoint:
                _, _, mz = _safe_vec3_extract(midpoint)
                max_z = max(max_z, mz)
                min_z = min(min_z, mz)
    
    # Handle case where no Z data found
    if max_z == float('-inf'):
        max_z = 0.0
    if min_z == float('inf'):
        min_z = 0.0
    
    # Classify faces
    top_faces: List[Tuple[str, float]] = []
    bottom_faces: List[Tuple[str, float]] = []
    horizontal_faces: List[str] = []
    vertical_faces: List[str] = []
    
    for face in faces:
        ref_id = face.get("entity_ref", face.get("id", "?"))
        normal = face.get("normal")
        centroid = face.get("centroid")
        
        if not normal:
            continue
        
        nx, ny, nz = _safe_vec3_extract(normal)
        z_normal = nz
        is_horizontal = abs(z_normal) > 0.9
        is_vertical = abs(z_normal) < 0.1
        
        if is_horizontal:
            horizontal_faces.append(ref_id)
            z_level = _safe_vec3_extract(centroid)[2] if centroid else None
            if z_level is not None:
                if z_normal > 0.9:  # Facing up
                    top_faces.append((ref_id, z_level))
                elif z_normal < -0.9:  # Facing down
                    bottom_faces.append((ref_id, z_level))
        elif is_vertical:
            vertical_faces.append(ref_id)
    
    # Sort and extract just refs for top/bottom faces
    top_faces.sort(key=lambda x: x[1], reverse=True)
    bottom_faces.sort(key=lambda x: x[1])
    
    # Filter to faces near the actual top/bottom (within tolerance)
    tolerance = 0.01
    top_face_refs = [ref for ref, z in top_faces if abs(z - max_z) <= tolerance] if top_faces else []
    bottom_face_refs = [ref for ref, z in bottom_faces if abs(z - min_z) <= tolerance] if bottom_faces else []
    
    # Find top/bottom edges
    top_edges: List[str] = []
    bottom_edges: List[str] = []
    
    for edge in edges:
        ref_id = edge.get("entity_ref", edge.get("id", "?"))
        midpoint = edge.get("midpoint")
        
        if not midpoint:
            continue
        
        _, _, z_mid = _safe_vec3_extract(midpoint)
        if abs(z_mid - max_z) <= tolerance:
            top_edges.append(ref_id)
        elif abs(z_mid - min_z) <= tolerance:
            bottom_edges.append(ref_id)
    
    return {
        "top_faces": top_face_refs,
        "bottom_faces": bottom_face_refs,
        "horizontal_faces": horizontal_faces,
        "vertical_faces": vertical_faces,
        "z_bounds": (min_z, max_z),
        "top_edges": top_edges,
        "bottom_edges": bottom_edges,
    }


def _format_face_description(idx: int, face_info: Mapping[str, Any]) -> str:
    """Format face information for LLM display."""
    geo_type = face_info.get("geometry_type", "Unknown")
    area = face_info.get("area", 0)

    parts = [f"  {idx}. Face ({geo_type}): {area} cm²"]

    centroid = face_info.get("centroid")
    if centroid:
        cx, cy, cz = _safe_vec3_extract(centroid)
        parts.append(f", centroid at ({cx:.2f}, {cy:.2f}, {cz:.2f})")

    normal = face_info.get("normal")
    if normal:
        direction = _describe_normal_direction(normal)
        if direction:
            parts.append(f", normal pointing {direction}")

    return "".join(parts)


def _format_edge_description(idx: int, edge_info: Mapping[str, Any]) -> str:
    """Format edge information for LLM display."""
    geo_type = edge_info.get("geometry_type", "Unknown")
    length = edge_info.get("length", 0)

    parts = [f"  {idx}. Edge ({geo_type}): {length} cm long"]

    start = edge_info.get("start")
    end = edge_info.get("end")
    if start and end:
        sx, sy, sz = _safe_vec3_extract(start)
        ex, ey, ez = _safe_vec3_extract(end)
        parts.append(f" from ({sx:.2f}, {sy:.2f}, {sz:.2f}) to ({ex:.2f}, {ey:.2f}, {ez:.2f})")

    return "".join(parts)


def _format_body_description(idx: int, body_info: Mapping[str, Any]) -> str:
    """Format body information for LLM display."""
    name = body_info.get("name", "Unnamed")
    volume = body_info.get("volume", 0)
    is_solid = body_info.get("is_solid", False)

    parts = [f"  {idx}. Body '{name}': {volume} cm³"]

    if is_solid:
        parts.append(" (solid)")
    else:
        parts.append(" (surface/open)")

    face_count = body_info.get("face_count")
    if face_count:
        parts.append(f", {face_count} faces")

    return "".join(parts)


def _describe_normal_direction(normal: Any) -> str:
    """
    Describe a surface normal vector in simple directional terms.
    
    Returns description like "+Z", "-X", "+X+Y", etc.
    """
    if normal is None:
        return ""

    x, y, z = _safe_vec3_extract(normal)

    # Threshold for considering a component significant
    threshold = 0.5

    parts = []

    if abs(x) > threshold:
        parts.append("+" if x > 0 else "-")
        parts.append("X")

    if abs(y) > threshold:
        parts.append("+" if y > 0 else "-")
        parts.append("Y")

    if abs(z) > threshold:
        parts.append("+" if z > 0 else "-")
        parts.append("Z")

    return "".join(parts) if parts else f"({x:.2f}, {y:.2f}, {z:.2f})"


def _final_response_text(response: Mapping[str, Any]) -> Optional[str]:
    """Extract the final assistant text for completion notifications."""
    text = extract_text_content(response)
    return text.strip() if text else None


def _tool_result_message(tool_use_id: str, message: str, *, is_error: bool = False, image_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a tool_result message block for the Anthropic conversation."""
    content_blocks = [{"type": "text", "text": message}]
    
    # Add image if provided by the tool result
    if image_data:
        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_data.get("media_type", "image/png"),
                "data": image_data.get("data", ""),
            },
        }
        content_blocks.append(image_block)
    
    block: Dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content_blocks,
    }
    if is_error:
        block["is_error"] = True
    return {"role": "user", "content": [block]}


def _user_text_message(text: str) -> Dict[str, Any]:
    """Fallback message shape when no tool result ID is available."""
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _summarise_execution_result(
    tool_name: str,
    result: Mapping[str, Any],
    entity_context: Optional[Mapping[str, Any]] = None
) -> Tuple[bool, str]:
    """
    Derive a spatially-enriched success/failure summary from Fusion execution payloads.
    
    Args:
        tool_name: Name of the executed tool
        result: Fusion execution result dict
        entity_context: Optional fresh entity context for spatial enrichment
        
    Returns:
        Tuple of (success_bool, enriched_message_string)
    """
    success = bool(result.get("success", result.get("type") != "error"))
    
    if not success:
        error_detail = result.get("error") or result.get("message") or result.get("details") or "Unknown error."
        hint = ""
        if _is_no_intersection_failure_text(error_detail):
            hint = (
                " Hint: This usually means your sketch/profile does not intersect any solid in the chosen "
                "extrusion direction. For Cut/Intersect, try flipping the distance sign, or ensure the sketch "
                "is created on the intended face (e.g., plane_id=face_0) rather than a datum plane."
            )
        return False, f"{tool_name} failed: {error_detail}{hint}"
    
    # Use tool-specific enrichment if available
    enricher = _TOOL_RESULT_ENRICHERS.get(tool_name)
    if enricher:
        # Determine if this enricher requires entity_context
        requires_context = tool_name in _CONTEXT_REQUIRING_ENRICHERS
        
        # Run enricher if we have context OR if the enricher doesn't require it
        if entity_context or not requires_context:
            try:
                return True, enricher(tool_name, result, entity_context or {})
            except Exception as exc:
                logger.warning("Tool result enrichment failed for %s: %s", tool_name, exc)
                # Fall through to basic message
    
    # Basic success message
    details = result.get("message") or result.get("details") or "Operation executed successfully."
    message = f"{tool_name} succeeded: {details}"

    # Mention created sketches for immediate context
    created_sketches = result.get("created_sketches")
    if created_sketches and isinstance(created_sketches, list):
        sketch_names = ", ".join(created_sketches)
        message += f" Created sketch(es): {sketch_names}"

    return True, message


# ------------------------------------------------------------------ #
# Tool Result Enrichment
# ------------------------------------------------------------------ #

def _compute_dimensions_from_bbox(bbox: Optional[Mapping[str, Any]]) -> Optional[List[float]]:
    """Compute [x, y, z] dimensions from bounding box dict."""
    if not bbox:
        return None
    min_pt = bbox.get("min")
    max_pt = bbox.get("max")
    if not min_pt or not max_pt:
        return None
    min_x, min_y, min_z = _safe_vec3_extract(min_pt)
    max_x, max_y, max_z = _safe_vec3_extract(max_pt)
    return [
        max_x - min_x,
        max_y - min_y,
        max_z - min_z,
    ]


def _check_bbox_anomalies(dims: Optional[List[float]], expected_shape: Optional[str] = None) -> List[str]:
    """
    Check bounding box dimensions for potential anomalies that indicate geometry errors.

    Returns a list of warning strings if anomalies are detected.

    Post-Execution Validation (Rule 16):
    - Detects when one dimension is suspiciously thin compared to others
    - Flags potential "flat shape when 3D bracket expected" scenarios
    - Helps catch missed profile extrusions or wrong plane selections
    """
    warnings = []
    if not dims or len(dims) != 3:
        return warnings

    # Filter out near-zero dimensions and sort
    non_zero_dims = [d for d in dims if d > 0.001]  # 0.001 mm threshold
    if len(non_zero_dims) < 2:
        return warnings

    sorted_dims = sorted(non_zero_dims, reverse=True)
    largest = sorted_dims[0]
    smallest = sorted_dims[-1]

    # Check for suspicious aspect ratios
    # If smallest dimension is < 10% of largest, and we have 3D expectations, flag it
    if largest > 0 and smallest / largest < 0.1:
        thin_axis = "X" if dims[0] == smallest else ("Y" if dims[1] == smallest else "Z")
        warnings.append(
            f"⚠ GEOMETRY CHECK: Body has thin {thin_axis} dimension ({smallest:.2f}mm vs {largest:.2f}mm). "
            f"If you expected a 3D bracket or perpendicular features, verify the profile was correct."
        )

    # Check for nearly 2D shapes (one dimension < 1mm when others are > 10mm)
    if len(non_zero_dims) == 3:
        large_dims = [d for d in dims if d > 10.0]  # > 10mm
        tiny_dims = [d for d in dims if d < 0.1]   # < 1mm
        if len(large_dims) >= 2 and len(tiny_dims) >= 1:
            warnings.append(
                f"⚠ GEOMETRY CHECK: Shape appears nearly 2D (thin dimension < 1mm). "
                f"Expected a 3D extrusion? Check sketch plane and extrude direction."
            )

    return warnings


def _enrich_extrude_result(
    tool_name: str,
    result: Mapping[str, Any],
    entity_context: Mapping[str, Any]
) -> str:
    """
    Generate enriched result message for extrude_profile/revolve_profile.
    
    Includes body dimensions, classified faces, and edge groups.
    Scopes face/edge classification to the newly created body to avoid
    mixing in entities from other bodies in multi-body designs.
    """
    lines = [f"{tool_name} succeeded."]

    if result.get("auto_flipped_distance") is True:
        req = result.get("requested_distance")
        actual = result.get("actual_distance")
        if req is not None and actual is not None:
            lines.append(
                f"Note: Auto-flipped distance sign (requested {req} cm, used {actual} cm) after initial Cut/Intersect miss."
            )
        else:
            lines.append("Note: Auto-flipped distance sign after initial Cut/Intersect miss.")
    
    bodies = entity_context.get("bodies", [])
    faces = entity_context.get("faces", [])
    edges = entity_context.get("edges", [])
    
    # Identify created bodies by token (may be multiple when extruding multiple profiles as NewBody)
    created_entities = result.get("created_entities", {})
    created_body_tokens = set(created_entities.get("bodies", []))
    
    created_bodies: List[Mapping[str, Any]] = []
    created_body_name: Optional[str] = None  # Used to scope face/edge summaries to a single body

    if created_body_tokens:
        for body in bodies:
            if body.get("entity_token") in created_body_tokens:
                created_bodies.append(body)
    
    # Fallback: use most recent body if no token match found
    if not created_bodies and bodies:
        created_bodies = [bodies[-1]]

    primary_body: Optional[Mapping[str, Any]] = created_bodies[0] if created_bodies else None
    if primary_body:
        created_body_name = primary_body.get("name")
    
    # Body summary with dimensions
    if primary_body:
        if len(created_bodies) > 1:
            lines.append("\nCreated Bodies:")
        else:
            lines.append("\nCreated Body:")

        for idx, body in enumerate(created_bodies):
            ref = body.get("entity_ref", "body_?")
            name = body.get("name", "Unnamed")
            bbox = body.get("bounding_box", {})
            volume = body.get("volume", 0)

            dims = _compute_dimensions_from_bbox(bbox)
            if dims:
                lines.append(f"  {ref}: {dims[0]:.1f} × {dims[1]:.1f} × {dims[2]:.1f} mm ({name})")
            else:
                lines.append(f"  {ref}: {name}")

            # For multi-body results, keep detailed bbox/volume on the primary body only.
            if idx == 0:
                if bbox.get("min") and bbox.get("max"):
                    min_pt = bbox["min"]
                    max_pt = bbox["max"]
                    min_x, min_y, min_z = _safe_vec3_extract(min_pt)
                    max_x, max_y, max_z = _safe_vec3_extract(max_pt)
                    lines.append(
                        f"  Bounding box: [{min_x:.1f},{min_y:.1f},{min_z:.1f}] to [{max_x:.1f},{max_y:.1f},{max_z:.1f}]"
                    )

                if volume:
                    lines.append(f"  Volume: {volume:.2f} mm³")

                # Post-execution validation: check for bbox anomalies (Rule 16)
                anomaly_warnings = _check_bbox_anomalies(dims)
                for warning in anomaly_warnings:
                    lines.append(f"\n{warning}")

        if len(created_bodies) > 1:
            primary_ref = primary_body.get("entity_ref", "body_?")
            primary_name = primary_body.get("name", "Unnamed")
            lines.append(
                f"\nNote: Face/edge summary below is scoped to the first created body only: {primary_ref} ({primary_name})."
            )

    # Filter faces/edges to only those belonging to the created body
    # This prevents mixing in entities from other bodies in multi-body designs
    def _belongs_to_created_body(entity: Mapping[str, Any]) -> bool:
        if not created_body_name:
            return True  # No filtering if we don't know the body name
        entity_body = entity.get("body") or entity.get("body_name")
        return entity_body == created_body_name
    
    scoped_faces = [f for f in faces if _belongs_to_created_body(f)]
    scoped_edges = [e for e in edges if _belongs_to_created_body(e)]
    
    # Classify faces
    top_faces: List[Tuple[str, float, float]] = []
    bottom_faces: List[Tuple[str, float, float]] = []
    side_faces: List[str] = []
    
    for face in scoped_faces:
        ref = face.get("entity_ref", face.get("id", "?"))
        normal = face.get("normal")
        centroid = face.get("centroid")
        area = face.get("area", 0)
        
        if not normal:
            side_faces.append(ref)
            continue
        
        # Handle normal as dict or list
        nz = normal.get("z", 0) if isinstance(normal, dict) else (normal[2] if len(normal) > 2 else 0)
        
        if nz > 0.9:  # Facing up
            z = centroid.get("z", 0) if isinstance(centroid, dict) else (centroid[2] if centroid and len(centroid) > 2 else 0)
            top_faces.append((ref, z, area))
        elif nz < -0.9:  # Facing down
            z = centroid.get("z", 0) if isinstance(centroid, dict) else (centroid[2] if centroid and len(centroid) > 2 else 0)
            bottom_faces.append((ref, z, area))
        else:
            side_faces.append(ref)
    
    if top_faces or bottom_faces or side_faces:
        lines.append("\nFace Summary:")
        
        if top_faces:
            # Sort by Z descending, take the highest
            top_faces.sort(key=lambda x: x[1], reverse=True)
            primary_top = top_faces[0]
            lines.append(f"  TOP (for sketching): {primary_top[0]} - Z={primary_top[1]:.2f}, area={primary_top[2]:.1f} mm²")
            if len(top_faces) > 1:
                other_tops = [f[0] for f in top_faces[1:]]
                lines.append(f"    Other upward faces: {', '.join(other_tops)}")
        
        if bottom_faces:
            bottom_faces.sort(key=lambda x: x[1])
            primary_bottom = bottom_faces[0]
            lines.append(f"  BOTTOM: {primary_bottom[0]} - Z={primary_bottom[1]:.2f}")
        
        if side_faces:
            if len(side_faces) <= 4:
                lines.append(f"  SIDES (vertical): {', '.join(side_faces)}")
            else:
                lines.append(f"  SIDES (vertical): {', '.join(side_faces[:2])} ... ({len(side_faces)} total)")
    
    # Classify edges by Z level using the created body's bounding box
    if scoped_edges and primary_body:
        body_bbox = primary_body.get("bounding_box", {})
        max_z = body_bbox.get("max", [0, 0, 0])[2] if body_bbox.get("max") else None
        min_z = body_bbox.get("min", [0, 0, 0])[2] if body_bbox.get("min") else None
        
        if max_z is not None and min_z is not None:
            top_edges: List[str] = []
            bottom_edges: List[str] = []
            vertical_edges: List[str] = []
            tolerance = 0.01
            
            for edge in scoped_edges:
                ref = edge.get("entity_ref", edge.get("id", "?"))
                start = edge.get("start_coords")
                end = edge.get("end_coords")
                
                if not start or not end:
                    continue
                
                # Handle coords as dict or list
                start_z = start.get("z", 0) if isinstance(start, dict) else (start[2] if len(start) > 2 else 0)
                end_z = end.get("z", 0) if isinstance(end, dict) else (end[2] if len(end) > 2 else 0)
                
                # Check if edge is at top or bottom Z
                avg_z = (start_z + end_z) / 2
                is_horizontal = abs(start_z - end_z) < tolerance
                
                if is_horizontal and abs(avg_z - max_z) < tolerance:
                    top_edges.append(ref)
                elif is_horizontal and abs(avg_z - min_z) < tolerance:
                    bottom_edges.append(ref)
                elif abs(start_z - end_z) > tolerance:  # Significant Z change = vertical
                    vertical_edges.append(ref)
            
            if top_edges or bottom_edges or vertical_edges:
                lines.append("\nEdge Summary:")
                if top_edges:
                    lines.append(f"  Top perimeter (ideal for fillets): {', '.join(top_edges)}")
                if bottom_edges:
                    lines.append(f"  Bottom perimeter: {', '.join(bottom_edges)}")
                if vertical_edges:
                    lines.append(f"  Vertical edges: {', '.join(vertical_edges)}")
    
    return "\n".join(lines)


def _enrich_fillet_result(
    tool_name: str,
    result: Mapping[str, Any],
    entity_context: Mapping[str, Any]
) -> str:
    """Generate enriched result for apply_fillet/apply_chamfer."""
    lines = [f"{tool_name} succeeded."]
    
    # Extract fillet details from result
    radius = result.get("radius")
    edge_count = result.get("edge_count") or result.get("selected_count")
    
    if radius:
        unit = "mm" if "fillet" in tool_name else "mm"
        lines.append(f"Applied {radius} {unit} radius")
    if edge_count:
        lines.append(f"Affected {edge_count} edge(s)")
    
    # Warn about remaining sharp edges if any
    edges = entity_context.get("edges", [])
    if edges:
        # Count edges that are still linear (potential sharp corners)
        remaining_corners = sum(
            1 for e in edges 
            if e.get("geometry_type") in ("Line3D", "Line")
        )
        if remaining_corners > 0:
            lines.append(f"\nNote: {remaining_corners} linear edges remain in the model")
    
    return "\n".join(lines)


def _enrich_hole_result(
    tool_name: str,
    result: Mapping[str, Any],
    entity_context: Mapping[str, Any]
) -> str:
    """Generate enriched result for hole operations."""
    lines = [f"{tool_name} succeeded."]
    
    # Extract hole details
    diameter = result.get("diameter")
    diameter_unit = result.get("diameter_unit", "mm")
    extent_type = result.get("extent_type")
    
    if diameter:
        lines.append(f"Created hole with {diameter} {diameter_unit} diameter")
        if extent_type:
            lines.append(f"Extent: {extent_type}")
    
    # Note position from result
    center = result.get("center")
    center_unit = result.get("center_unit", "mm")
    cx = cy = cz = None
    if isinstance(center, Mapping):
        try:
            cx = float(center.get("x")) if center.get("x") is not None else None
            cy = float(center.get("y")) if center.get("y") is not None else None
            cz = float(center.get("z")) if center.get("z") is not None else None
        except Exception:
            cx = cy = cz = None
    elif isinstance(center, Sequence) and not isinstance(center, (str, bytes, bytearray)):
        if len(center) >= 3:
            try:
                cx = float(center[0])
                cy = float(center[1])
                cz = float(center[2])
            except Exception:
                cx = cy = cz = None

    if cx is not None and cy is not None and cz is not None:
        lines.append(f"Center: ({cx:.3f}, {cy:.3f}, {cz:.3f}) {center_unit}")
     
    return "\n".join(lines)


def _is_raw_entity_token(value: str) -> bool:
    """Detect whether a string looks like a raw Fusion 360 entity token (base64/opaque).

    Ref IDs are short identifiers like 'face_0', 'XY', 'body_0', 'e0'.
    Raw entity tokens are long opaque strings that must never leak to the LLM.
    Legitimate long plane IDs (e.g. construction plane paths with '/') are exempt.
    """
    if not value or len(value) < 20:
        return False
    # Face refs, edge refs, body refs, vertex refs and datum plane names
    if re.match(r'^face_\d+$', value) or re.match(r'^(body_\d+|e\d+|v\d+)$', value) or value in ('XY', 'XZ', 'YZ'):
        return False
    # Construction plane or sketch plane IDs often contain '/' or ':' path separators
    # and are legitimate identifiers, not opaque base64 tokens.
    if '/' in value or '::' in value:
        return False
    return True


def _enrich_sketch_result(
    tool_name: str,
    result: Mapping[str, Any],
    entity_context: Mapping[str, Any]
) -> str:
    """Generate enriched result for create_sketch."""
    lines = [f"{tool_name} succeeded."]

    created_sketches = result.get("created_sketches", [])
    if created_sketches:
        lines.append(f"Created sketch: {', '.join(created_sketches)}")

    # Note the plane it was created on (best-effort; depends on tool implementation).
    # IMPORTANT: plane_id_input may contain a resolved raw entity token after
    # ref→token resolution.  Never echo raw tokens to the LLM — use plane_name instead.
    plane_id_input = result.get("plane_id_input")
    plane_name = result.get("plane_name")

    # Suppress raw entity tokens — only show semantic refs or plane names
    if plane_id_input and _is_raw_entity_token(str(plane_id_input)):
        plane_id_input = None
    if plane_id_input or plane_name:
        if plane_id_input and plane_name:
            lines.append(f"Plane: {plane_id_input} ({plane_name})")
        elif plane_id_input:
            lines.append(f"Plane: {plane_id_input}")
        else:
            lines.append(f"Plane: {plane_name}")

    plane_origin_world = result.get("plane_origin_world")
    if plane_origin_world:
        lines.append(f"Plane origin (world, cm): {plane_origin_world}")

    # Include orientation feedback for face/custom plane sketches
    orientation = result.get("orientation")
    if orientation:
        u_axis = orientation.get("u_axis_world", [])
        v_axis = orientation.get("v_axis_world", [])
        extrude_dir = orientation.get("extrude_positive_direction", [])
        lines.append(f"\nSketch orientation in world space:")
        lines.append(f"  u_axis_world: {u_axis}")
        lines.append(f"  v_axis_world: {v_axis}")
        lines.append(f"  extrude_positive_direction: {extrude_dir}")

    lines.append("\nReady for geometry (add_rectangle, add_circle, add_line, add_arc)")

    return "\n".join(lines)


def _enrich_geometry_result(
    tool_name: str,
    result: Mapping[str, Any],
    entity_context: Mapping[str, Any]
) -> str:
    """Generate enriched result for sketch geometry operations (add_circle, add_rectangle, etc.)."""
    lines = [f"{tool_name} succeeded."]
    
    # Note profile count if available
    profile_count = result.get("profile_count")
    if profile_count is not None:
        lines.append(f"Sketch now has {profile_count} profile(s)")
        if profile_count > 0:
            lines.append("Ready for extrude_profile")
    
    return "\n".join(lines)


def _enrich_rectangle_result(
    tool_name: str,
    result: Mapping[str, Any],
    entity_context: Mapping[str, Any],
) -> str:
    """Generate enriched result for add_rectangle with full entity listing."""
    output_lines = [f"{tool_name} succeeded."]
    entities = result.get("entities")
    if isinstance(entities, list) and entities:
        rectangle_id = str(result.get("rectangle_id") or "").strip()
        output_lines.append(f"Created {len(entities)} lines forming a closed rectangle.")

        # Report registered refs (set by _register_sketch_result_entities before enrichment)
        registered = result.get("_registered_line_refs")
        if isinstance(registered, list) and registered:
            ref_labels = []
            for entry in registered:
                if not isinstance(entry, dict):
                    continue
                alias = entry.get("alias")
                ref = entry.get("ref")
                if alias:
                    ref_labels.append(f"{alias} ({ref})")
                elif ref:
                    ref_labels.append(str(ref))
            if ref_labels:
                output_lines.append(f"Registered line refs: {', '.join(ref_labels)}")
        elif rectangle_id:
            # Fallback: aliases are deterministic from rectangle_id
            ref_labels = [f"{rectangle_id}_line_{i}" for i in range(len(entities))]
            output_lines.append(f"Registered line refs: {', '.join(ref_labels)}")

        # Show endpoint coordinates so LLM can identify which line is which edge
        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                continue
            start_uv = entity.get("start_uv", [])
            end_uv = entity.get("end_uv", [])
            alias = f"{rectangle_id}_line_{i}" if rectangle_id else f"line_{i}"
            if len(start_uv) >= 2 and len(end_uv) >= 2:
                output_lines.append(
                    f"  {alias}: ({start_uv[0]}, {start_uv[1]}) -> ({end_uv[0]}, {end_uv[1]})"
                )

        output_lines.append("Adjacent lines share endpoints. Use .start/.end point refs for future reference.")

    profile_count = result.get("profile_count")
    if profile_count is not None:
        output_lines.append(f"Sketch now has {profile_count} profile(s)")
        if profile_count > 0:
            output_lines.append("Ready for extrude_profile")

    return "\n".join(output_lines)


def _enrich_list_sketch_profiles_result(
    tool_name: str,
    result: Mapping[str, Any],
    entity_context: Mapping[str, Any],
) -> str:
    """Generate enriched result for list_sketch_profiles."""
    sketch_id = result.get("sketch_id", "?")
    profile_count = result.get("profile_count")
    profiles = result.get("profiles") or []

    lines = [f"{tool_name} succeeded."]

    if profile_count is not None:
        lines.append(f"Sketch: {sketch_id}, profiles: {profile_count}")
    else:
        lines.append(f"Sketch: {sketch_id}")

    if isinstance(profiles, list) and profiles:
        for prof in profiles:
            if not isinstance(prof, Mapping):
                continue
            idx = prof.get("index")
            area = prof.get("area_cm2")
            outer = prof.get("outer_loop_count")
            inner = prof.get("inner_loop_count")
            centroid = prof.get("centroid_world_cm")
            lines.append(f"  idx={idx} area={area} outer={outer} inner={inner} centroid={centroid}")

    # When multiple profiles exist (common with overlapping primitives), suggest
    # using profile_indices to extrude them all in a single feature.
    if isinstance(profiles, list) and len(profiles) > 1:
        all_indices = [p.get("index", i) for i, p in enumerate(profiles) if isinstance(p, Mapping)]
        indices_str = ", ".join(str(i) for i in all_indices)
        lines.append("")
        lines.append(
            f"HINT: This sketch has {len(profiles)} closed regions (common when shapes overlap). "
            f"To extrude ALL regions as one unified shape, use a single extrude_profile call with "
            f'profile_indices=[{indices_str}] (omit profile_index entirely). '
            f"This is much more reliable than extruding one-by-one with Join operations."
        )

    return "\n".join(lines)


# Registry of tool-specific result enrichers
_TOOL_RESULT_ENRICHERS: Dict[str, Any] = {
    "extrude_profile": _enrich_extrude_result,
    "revolve_profile": _enrich_extrude_result,  # Similar structure
    "apply_fillet": _enrich_fillet_result,
    "apply_chamfer": _enrich_fillet_result,  # Similar structure
    "create_simple_hole": _enrich_hole_result,
    "create_counterbore_hole": _enrich_hole_result,
    "create_tapped_hole": _enrich_hole_result,
    "create_sketch": _enrich_sketch_result,
    "list_sketch_profiles": _enrich_list_sketch_profiles_result,
    "add_circle": _enrich_geometry_result,
    "add_rectangle": _enrich_rectangle_result,
    "add_line": _enrich_geometry_result,
    "add_arc": _enrich_geometry_result,
}

# Enrichers that require fresh entity_context (geometry-modifying tools)
# These won't run until entity_context is available after refresh
_CONTEXT_REQUIRING_ENRICHERS: Set[str] = {
    "extrude_profile",
    "revolve_profile",
    "apply_fillet",
    "apply_chamfer",
    "create_simple_hole",
    "create_counterbore_hole",
    "create_tapped_hole",
}


def _register_sketch_result_entities(
    session_id: str,
    manager: ConnectionManager,
    tool_name: str,
    result: Mapping[str, Any],
    *,
    fallback_plane_id: Optional[str] = None,
) -> None:
    """Persist sketch geometry references from Fusion execution results."""
    sketch_id = result.get("sketch_id")
    if not isinstance(sketch_id, str) or not sketch_id.strip():
        return

    sketch_store = _get_sketch_entity_store(session_id, manager)

    # Origin point registration for new sketches
    if tool_name == "create_sketch":
        origin_token = result.get("origin_point_token")
        if isinstance(origin_token, str) and origin_token.strip():
            sketch_store.register_origin(sketch_id, origin_token)
        _register_sketch_plane_metadata(
            session_id,
            manager,
            sketch_id=sketch_id,
            result=result,
            fallback_plane_id=fallback_plane_id,
        )
        return

    # Geometry registration
    if tool_name in {"add_line", "add_circle", "add_arc"}:
        entity_token = result.get("entity_token")
        kind = result.get("kind")
        if not isinstance(entity_token, str) or not entity_token.strip() or not isinstance(kind, str):
            return

        alias_key = {
            "add_line": "line_id",
            "add_circle": "circle_id",
            "add_arc": "arc_id",
        }.get(tool_name, "")
        alias_val = result.get(alias_key) if alias_key else ""
        alias = str(alias_val).strip() if alias_val else None

        metadata = {
            "point_tokens": result.get("point_tokens") or {},
        }

        try:
            sketch_store.register_entity(
                sketch_id=sketch_id,
                kind=str(kind),
                token=entity_token,
                metadata=metadata,
                alias=alias,
            )
        except ValueError as exc:
            logger.warning("Failed to register sketch entity for %s: %s", tool_name, exc)
        return

    # Compound geometry registration (rectangle = 4 lines)
    if tool_name == "add_rectangle":
        entities = result.get("entities")
        if not isinstance(entities, list):
            return
        rectangle_id = str(result.get("rectangle_id") or "").strip()
        registered_refs: list[dict[str, Any]] = []
        for i, entity_data in enumerate(entities):
            if not isinstance(entity_data, Mapping):
                continue
            entity_token = entity_data.get("entity_token")
            kind = entity_data.get("kind", "line")
            if not isinstance(entity_token, str) or not entity_token.strip():
                continue
            alias = f"{rectangle_id}_line_{i}" if rectangle_id else None
            metadata = {"point_tokens": entity_data.get("point_tokens") or {}}
            try:
                ref_id = sketch_store.register_entity(
                    sketch_id=sketch_id,
                    kind=str(kind),
                    token=entity_token,
                    metadata=metadata,
                    alias=alias,
                )
                registered_refs.append({"ref": ref_id, "alias": alias})
            except ValueError as exc:
                logger.warning("Failed to register rectangle line %d for %s: %s", i, tool_name, exc)
        # Annotate result so enricher can access registered refs
        if isinstance(result, dict):
            result["_registered_line_refs"] = registered_refs


async def _execute_geometry_tool_call(
    session_id: str,
    manager: ConnectionManager,
    tool_name: str,
    tool_use_id: str,
    tool_input: Mapping[str, Any],
    geometry_kind: str,
    description: str = "",
) -> Tuple[bool, str]:
    """Send an edge, face, or body tool request to Fusion and return the formatted result."""
    if geometry_kind not in {"edge", "face", "body"}:
        raise SelectionToolCallError(f"Unsupported geometry kind '{geometry_kind}'.")

    store = _get_entity_store(session_id, manager)

    payload_type = {
        "edge": "edge_operation",
        "face": "face_operation",
        "body": "body_operation",
    }[geometry_kind]
    payload: Dict[str, Any] = {
        "type": payload_type,
        "operation": tool_name,
        "tool_use_id": tool_use_id,
        "description": description,
        "parameters": dict(tool_input) if isinstance(tool_input, Mapping) else {},
    }

    if tool_name in {"list_edges", "list_faces", "list_bodies"}:
        if tool_input:
            extra = set(tool_input.keys())
            if extra:
                raise SelectionToolCallError(f"{tool_name} does not accept parameters: unexpected {sorted(extra)}")

    elif tool_name in {"select_edges", "select_faces", "select_bodies"}:
        # Normalize parameter names to support both new (edge_refs, face_refs, body_refs)
        # and legacy (entity_tokens) field names
        tool_input = _normalize_tool_params(tool_name, tool_input)

        if "entity_tokens" not in tool_input:
            raise SelectionToolCallError(f"{tool_name} requires an 'entity_tokens' array.")

        tokens = tool_input.get("entity_tokens")
        if not isinstance(tokens, list):
            raise SelectionToolCallError("'entity_tokens' must be an array of strings.")

        normalized_inputs = [str(token).strip() for token in tokens if str(token).strip()]
        if not normalized_inputs:
            raise SelectionToolCallError(f"{tool_name} received an empty list of entity tokens/refs.")

        resolved_tokens, missing_refs, kind_errors = store.resolve_tokens(
            normalized_inputs, expected_kind=geometry_kind
        )

        if kind_errors:
            raise SelectionToolCallError(
                f"{tool_name} received entity_refs with incorrect kind: {kind_errors}"
            )
        if missing_refs:
            raise SelectionToolCallError(
                f"{tool_name} could not resolve entity_refs: {missing_refs}. "
                f"Verify these refs exist in the Design Entities context."
            )

        # Accept both new names (edge_refs, face_refs, body_refs) and legacy (entity_tokens)
        extra = set(tool_input.keys()) - {"entity_tokens", "edge_refs", "face_refs", "body_refs", "clear_existing"}
        if extra:
            raise SelectionToolCallError(f"{tool_name} received unexpected parameter(s): {sorted(extra)}")

        # Even though the LLM provides ref IDs, Fusion expects tokens. The resolver above
        # converts refs → tokens while leaving raw tokens untouched.
        payload["entity_tokens"] = resolved_tokens
        payload["clear_existing"] = bool(tool_input.get("clear_existing", True))

    elif tool_name in {"clear_edge_selection", "clear_face_selection", "clear_body_selection"}:
        if tool_input:
            extra = set(tool_input.keys())
            if extra:
                raise SelectionToolCallError(
                    f"{tool_name} does not accept parameters: unexpected {sorted(extra)}"
                )
    else:  # pragma: no cover - guarded upstream
        raise SelectionToolCallError(f"Unsupported geometry tool '{tool_name}'.")

    # Ensure resolved fields are mirrored into parameters so Fusion reads the
    # converted entity tokens instead of the original logical refs.
    _sync_payload_parameters(payload)

    await _send_message_safe(manager, session_id, payload)

    try:
        result = await _wait_for_matching_tool_result(
            session_id,
            manager,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            timeout=EXECUTION_TIMEOUT,
            wait_context=f"{geometry_kind}_operation",
        )
    except asyncio.TimeoutError as exc:
        raise SelectionToolCallError(
            f"Timed out waiting for Fusion to finish '{tool_name}' (tool_use_id={tool_use_id})."
        ) from exc

    success = bool(result.get("success"))
    message_text = result.get("message") or f"{tool_name} completed."

    if not success and result.get("error"):
        message_text = f"{message_text}\nError: {result.get('error')}"

    if tool_name == "list_edges" and success:
        await _attach_entity_refs_to_listing(session_id, manager, "edge", result)
        message_text = _format_edge_listing_message(result, message_text)
    elif tool_name == "select_edges":
        message_text = _format_select_edges_message(result, message_text)
    elif tool_name == "clear_edge_selection":
        message_text = _format_clear_edges_message(result, message_text)
    elif tool_name == "list_faces" and success:
        await _attach_entity_refs_to_listing(session_id, manager, "face", result)
        message_text = _format_face_listing_message(result, message_text)
    elif tool_name == "select_faces":
        message_text = _format_select_faces_message(result, message_text)
    elif tool_name == "clear_face_selection":
        message_text = _format_clear_faces_message(result, message_text)
    elif tool_name == "list_bodies" and success:
        await _attach_entity_refs_to_listing(session_id, manager, "body", result)
        message_text = _format_body_listing_message(result, message_text)
    elif tool_name == "select_bodies":
        message_text = _format_select_bodies_message(result, message_text)
    elif tool_name == "clear_body_selection":
        message_text = _format_clear_bodies_message(result, message_text)

    return success, message_text


async def _attach_entity_refs_to_listing(
    session_id: str,
    manager: ConnectionManager,
    kind: str,
    result: Mapping[str, Any],
) -> None:
    """
    Register listing results with the entity store and add entity_ref fields so the
    LLM can reference short IDs (edge_1, face_2, body_3) instead of long tokens.
    """
    store = _get_entity_store(session_id, manager)

    collection_key = {"edge": "edges", "face": "faces", "body": "bodies"}.get(kind)
    if not collection_key:
        return

    entities = result.get(collection_key) or []
    registered = await store.register_entities(kind, entities)
    token_to_ref = {entry.token: entry.ref_id for entry in registered}

    for entity in entities:
        token = entity.get("entity_token")
        if token in token_to_ref:
            entity["entity_ref"] = token_to_ref[token]


def _resolve_entity_tokens_or_refs(
    session_id: str,
    manager: ConnectionManager,
    values: Sequence[Any],
    *,
    expected_kind: Optional[str] = None,
    context: str = "entity",
) -> List[str]:
    """
    Convert a list of ref IDs or tokens into canonical tokens.

    Raises SelectionToolCallError when refs are missing or wrong kind.
    """
    store = _get_entity_store(session_id, manager)
    
    # Handle JSON-stringified arrays from LLMs (defensive fix)
    if isinstance(values, str):
        values_str = values.strip()
        if values_str.startswith("[") and values_str.endswith("]"):
            try:
                parsed = json.loads(values_str)
                if isinstance(parsed, list):
                    logger.info(f"Parsed JSON-stringified array in {context}: {values_str[:100]}")
                    values = parsed
            except json.JSONDecodeError:
                pass  # Let normal validation handle it
    
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise SelectionToolCallError(f"{context} must be an array of strings.")

    cleaned = [str(v).strip() for v in values if str(v).strip()]
    if not cleaned:
        raise SelectionToolCallError(f"{context} cannot be empty.")

    tokens: List[str] = []
    missing: List[str] = []
    kind_errors: List[str] = []
    candidate_hints: Dict[str, List[str]] = {}

    for ref_or_token in cleaned:
        token, error = store.resolve_token(ref_or_token, expected_kind=expected_kind)
        if token and not error:
            tokens.append(token)
            continue

        if expected_kind == "face":
            recovered_token, recovered_ref, candidates = _resolve_stale_face_ref_if_unambiguous(store, ref_or_token)
            if recovered_token:
                tokens.append(recovered_token)
                logger.info(
                    "Recovered stale face ref '%s' -> '%s' in %s",
                    ref_or_token,
                    recovered_ref,
                    context,
                )
                continue
            if candidates:
                candidate_hints[ref_or_token] = candidates
                kind_errors.append(
                    f"{ref_or_token}: stale face ref is ambiguous. Candidate replacements: {candidates}"
                )
                continue

        if error:
            kind_errors.append(f"{ref_or_token}: {error}")
        else:
            missing.append(ref_or_token)

    if kind_errors:
        diagnostics = _format_ref_resolution_diagnostics(
            store,
            expected_kind=expected_kind,
            requested_refs=cleaned,
            candidate_hints=candidate_hints,
        )
        raise SelectionToolCallError(
            f"{context} contained unresolved or wrong-type refs: {kind_errors}{diagnostics}"
        )
    if missing:
        diagnostics = _format_ref_resolution_diagnostics(
            store,
            expected_kind=expected_kind,
            requested_refs=cleaned,
            candidate_hints=candidate_hints,
        )
        raise SelectionToolCallError(
            f"{context} references not found: {missing}. Re-run list_{expected_kind or 'entities'} to refresh."
            f"{diagnostics}"
        )
    return tokens


def _resolve_single_entity_ref(
    session_id: str,
    manager: ConnectionManager,
    value: Any,
    *,
    expected_kind: Optional[str] = None,
    context: str = "entity_ref",
) -> str:
    store = _get_entity_store(session_id, manager)
    cleaned = str(value).strip()
    token, error = store.resolve_token(cleaned, expected_kind=expected_kind)
    if (error or not token) and expected_kind == "face":
        recovered_token, recovered_ref, candidates = _resolve_stale_face_ref_if_unambiguous(store, cleaned)
        if recovered_token:
            logger.info("Recovered stale face ref '%s' -> '%s' in %s", cleaned, recovered_ref, context)
            return recovered_token
        if candidates:
            diagnostics = _format_ref_resolution_diagnostics(
                store,
                expected_kind=expected_kind,
                requested_refs=[cleaned],
                candidate_hints={cleaned: candidates},
            )
            raise SelectionToolCallError(
                f"{context} could not be resolved: {value}. {error or 'Unknown entity ref.'} "
                f"Candidate replacements: {candidates}{diagnostics}"
            )

    if error or not token:
        diagnostics = _format_ref_resolution_diagnostics(
            store,
            expected_kind=expected_kind,
            requested_refs=[cleaned],
        )
        raise SelectionToolCallError(
            f"{context} could not be resolved: {value}. "
            f"{error or 'Provide a valid reference from the latest list call.'}{diagnostics}"
        )
    return token


def _resolve_token_field_inplace(
    store: EntityStore,
    target: Mapping[str, Any],
    key: str,
    *,
    expected_kind: Optional[str] = None,
) -> None:
    if key not in target:
        return
    value = target.get(key)
    # Skip empty/whitespace values - LLMs often fill optional fields with ""
    if not value or not str(value).strip():
        return
    token, error = store.resolve_token(str(value).strip(), expected_kind=expected_kind)
    if error or not token:
        diagnostics = _format_ref_resolution_diagnostics(
            store,
            expected_kind=expected_kind,
            requested_refs=[str(value).strip()],
        )
        raise SelectionToolCallError((error or f"Unable to resolve reference for '{key}'.") + diagnostics)
    target[key] = token


def _format_ref_resolution_diagnostics(
    store: EntityStore,
    *,
    expected_kind: Optional[str],
    requested_refs: Sequence[str],
    candidate_hints: Optional[Mapping[str, Sequence[str]]] = None,
    limit: int = 40,
) -> str:
    """Append actionable current-ref diagnostics to unresolved-ref errors."""
    if not expected_kind:
        return ""

    refs = sorted(store.get_refs_by_kind(expected_kind), key=_ref_sort_key)
    parts: List[str] = []

    requested = [str(ref).strip() for ref in requested_refs if str(ref).strip()]
    if requested:
        parts.append(f" Requested refs: {requested}.")

    if refs:
        preview = refs[:limit]
        suffix = f", ... ({len(refs) - limit} more)" if len(refs) > limit else ""
        parts.append(
            f" Current valid {expected_kind} refs ({len(refs)}): {', '.join(preview)}{suffix}."
        )
    else:
        parts.append(f" Current valid {expected_kind} refs: none loaded.")

    candidate_hints_by_ref: Dict[str, List[str]] = {
        str(stale_ref): [str(candidate) for candidate in candidates]
        for stale_ref, candidates in (candidate_hints or {}).items()
    }
    for requested_ref in requested:
        candidate_list = candidate_hints_by_ref.get(requested_ref)
        if candidate_list is None:
            candidate_list = _candidate_refs_for_stale_ref(
                store,
                requested_ref,
                expected_kind=expected_kind,
                limit=5,
            )
        if candidate_list:
            parts.append(f" Candidate replacements for {requested_ref}: {', '.join(candidate_list)}.")

    parts.append(" Use only refs from current_design_refs/design_entities; list_features does not refresh entity refs.")
    return " " + " ".join(parts)


def _candidate_refs_for_stale_ref(
    store: EntityStore,
    stale_ref: str,
    *,
    expected_kind: Optional[str],
    limit: int = 5,
) -> List[str]:
    if expected_kind == "face":
        return _candidate_faces_for_stale_ref(store, stale_ref, limit=limit)
    if expected_kind == "edge":
        return _candidate_edges_for_stale_ref(store, stale_ref, limit=limit)
    return []


def _candidate_faces_for_stale_ref(
    store: EntityStore,
    stale_ref: str,
    *,
    limit: int = 5,
) -> List[str]:
    """Find likely replacement face refs for a stale face_N reference."""
    if not stale_ref or not re.match(r"^face_\d+$", str(stale_ref).strip()):
        return []

    target_fp = None
    persistent_cache = getattr(store, "_persistent_cache", {})
    if isinstance(persistent_cache, Mapping):
        for cached_mapping in persistent_cache.values():
            if getattr(cached_mapping, "ref_id", None) != stale_ref:
                continue
            fingerprint = getattr(cached_mapping, "fingerprint", None)
            if fingerprint and getattr(fingerprint, "kind", None) == "face":
                target_fp = fingerprint
                break

    if not target_fp:
        return []

    target_surface = (getattr(target_fp, "surface_type", None) or "").lower()
    target_normal = getattr(target_fp, "normal", None)
    target_centroid = getattr(target_fp, "centroid", None)

    candidates: List[Tuple[float, float, float, str]] = []
    for face_ref in store.get_refs_by_kind("face"):
        entry = store.get_entry(face_ref)
        if not entry:
            continue

        surface_type = (entry.metadata.get("surface_type") or "").lower()
        if target_surface and surface_type and surface_type != target_surface:
            continue

        normal = entry.normal
        if target_normal and normal:
            tx, ty, tz = target_normal
            nx, ny, nz = normal
            t_mag = math.sqrt(tx * tx + ty * ty + tz * tz)
            n_mag = math.sqrt(nx * nx + ny * ny + nz * nz)
            if t_mag <= 1e-9 or n_mag <= 1e-9:
                continue
            dot = (tx * nx + ty * ny + tz * nz) / (t_mag * n_mag)
            if dot < 0.95:
                continue
        else:
            dot = 1.0

        centroid = entry.centroid
        if target_centroid and centroid:
            cx, cy, cz = centroid
            txc, tyc, tzc = target_centroid
            dist_mm = math.sqrt((cx - txc) ** 2 + (cy - tyc) ** 2 + (cz - tzc) ** 2)
        else:
            dist_mm = float("inf")

        area = entry.metadata.get("area")
        target_area = getattr(target_fp, "area", None)
        if area is not None and target_area is not None:
            area_delta = abs(float(area) - float(target_area))
        else:
            area_delta = float("inf")

        candidates.append((dist_mm, -dot, area_delta, face_ref))

    if not candidates:
        return []

    candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return [ref for _, _, _, ref in candidates[: max(1, limit)]]


def _candidate_edges_for_stale_ref(
    store: EntityStore,
    stale_ref: str,
    *,
    limit: int = 5,
) -> List[str]:
    """Find likely replacement edge refs for a stale eN reference."""
    cleaned = str(stale_ref).strip()
    if not cleaned or not re.match(r"^e\d+$", cleaned):
        return []

    target_fp = None
    persistent_cache = getattr(store, "_persistent_cache", {})
    if isinstance(persistent_cache, Mapping):
        for cached_mapping in persistent_cache.values():
            if getattr(cached_mapping, "ref_id", None) != cleaned:
                continue
            fingerprint = getattr(cached_mapping, "fingerprint", None)
            if fingerprint and getattr(fingerprint, "kind", None) == "edge":
                target_fp = fingerprint
                break

    if not target_fp:
        return []

    target_type = (getattr(target_fp, "edge_type", None) or "").lower()
    target_length = getattr(target_fp, "length", None)
    target_midpoint = getattr(target_fp, "midpoint", None)

    candidates: List[Tuple[float, float, str]] = []
    for edge_ref in store.get_refs_by_kind("edge"):
        entry = store.get_entry(edge_ref)
        if not entry:
            continue

        edge_type = (entry.metadata.get("edge_type") or "").lower()
        if target_type and edge_type and edge_type != target_type:
            continue

        midpoint = entry.midpoint
        if target_midpoint and midpoint:
            mx, my, mz = midpoint
            tx, ty, tz = target_midpoint
            dist_mm = math.sqrt((mx - tx) ** 2 + (my - ty) ** 2 + (mz - tz) ** 2)
        else:
            dist_mm = float("inf")

        length = entry.metadata.get("length")
        if target_length is not None and length is not None:
            length_delta = abs(float(length) - float(target_length))
        else:
            length_delta = float("inf")

        if math.isinf(dist_mm) and math.isinf(length_delta) and not target_type:
            continue
        candidates.append((dist_mm, length_delta, edge_ref))

    if not candidates:
        return []

    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    return [ref for _, _, ref in candidates[: max(1, limit)]]


def _resolve_stale_face_ref_if_unambiguous(
    store: EntityStore,
    stale_ref: str,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Resolve stale face_N refs when there is exactly one viable replacement."""
    cleaned = str(stale_ref).strip()
    if not cleaned or not re.match(r"^face_\d+$", cleaned):
        return None, None, []

    candidates = _candidate_faces_for_stale_ref(store, cleaned)
    if len(candidates) != 1:
        return None, None, candidates

    replacement_ref = candidates[0]
    token, error = store.resolve_token(replacement_ref, expected_kind="face")
    if error or not token:
        return None, replacement_ref, candidates

    return token, replacement_ref, candidates


def _find_entity_entry_by_token(store: EntityStore, kind: str, token: str) -> Optional[Any]:
    """Best-effort token lookup via public EntityStore APIs."""
    for ref_id in store.get_refs_by_kind(kind):
        entry = store.get_entry(ref_id)
        if entry and entry.token == token:
            return entry
    return None


def _preflight_hole_center_on_face(
    session_id: str,
    manager: ConnectionManager,
    *,
    tool_name: str,
    face_token: str,
    center_x: float,
    center_y: float,
    center_z: float,
) -> Optional[str]:
    """Reject obviously invalid hole/thread coordinates before Fusion execution."""
    store = _get_entity_store(session_id, manager)
    face_entry = _find_entity_entry_by_token(store, "face", face_token)
    if not face_entry:
        return None

    normal = face_entry.normal
    centroid = face_entry.centroid
    if not normal or not centroid:
        return None

    nx, ny, nz = normal
    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag <= 1e-9:
        return None

    nx /= mag
    ny /= mag
    nz /= mag

    dx = center_x - centroid[0]
    dy = center_y - centroid[1]
    dz = center_z - centroid[2]
    plane_offset_mm = abs(dx * nx + dy * ny + dz * nz)

    # Spatial context/tool inputs are in mm for hole center coordinates.
    if plane_offset_mm > 2.0:
        return (
            f"{tool_name} center point is {plane_offset_mm:.2f}mm away from the selected face plane. "
            "This likely misses the target body. Re-evaluate face_ref and center_x/center_y/center_z."
        )
    return None


def _length_units_to_cm(units: str) -> float:
    normalized = str(units or "").strip().lower()
    if normalized in {"cm", "centimeter", "centimeters"}:
        return 1.0
    if normalized in {"m", "meter", "meters"}:
        return 100.0
    if normalized in {"in", "inch", "inches"}:
        return 2.54
    # Default/legacy backend spatial context units are millimeters.
    return 0.1


def _normalise_vec3(value: Any) -> Optional[Tuple[float, float, float]]:
    x, y, z = _safe_vec3_extract(value)
    mag = math.sqrt((x * x) + (y * y) + (z * z))
    if mag <= 1e-9:
        return None
    return (x / mag, y / mag, z / mag)


def _dot_vec3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])


def _resolve_face_token_for_plane_id(
    session_id: str,
    manager: ConnectionManager,
    plane_id: str,
) -> Optional[str]:
    plane = str(plane_id or "").strip()
    if not plane:
        return None
    store = _get_entity_store(session_id, manager)
    token, error = store.resolve_token(plane, expected_kind="face")
    if not token or error:
        return None
    face_entry = _find_entity_entry_by_token(store, "face", token)
    if not face_entry:
        return None
    return token


def _adjacent_faces_include_target(
    adjacent_faces: Any,
    *,
    face_ref: str,
    face_token: str,
) -> bool:
    if not isinstance(adjacent_faces, Sequence) or isinstance(adjacent_faces, (str, bytes, bytearray)):
        return False
    for item in adjacent_faces:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, Mapping):
            candidate = (
                item.get("entity_ref")
                or item.get("face_ref")
                or item.get("face_token")
                or item.get("ref")
                or item.get("token")
                or item.get("id")
            )
            text = str(candidate or "").strip()
        else:
            text = ""
        if not text:
            continue
        if text == face_ref or text == face_token:
            return True
    return False


def _compute_face_sketch_uv_bounds(
    session_id: str,
    manager: ConnectionManager,
    *,
    face_token: str,
    plane_origin_world: Any,
    u_axis_world: Any,
    v_axis_world: Any,
) -> Optional[Dict[str, float]]:
    store = _get_entity_store(session_id, manager)
    face_entry = _find_entity_entry_by_token(store, "face", face_token)
    if not face_entry:
        return None

    u_axis = _normalise_vec3(u_axis_world)
    v_axis = _normalise_vec3(v_axis_world)
    if not u_axis or not v_axis:
        return None

    ox, oy, oz = _safe_vec3_extract(plane_origin_world)
    origin_cm = (ox, oy, oz)

    face_ref = face_entry.ref_id
    latest_context = _manager_get_latest_entity_context(manager, session_id) or {}
    units_value = latest_context.get("units")
    if not units_value and isinstance(latest_context.get("spatial_context"), Mapping):
        units_value = latest_context.get("spatial_context", {}).get("units")
    scale_to_cm = _length_units_to_cm(str(units_value or "mm"))

    uv_points: List[Tuple[float, float]] = []
    for edge_ref in store.get_refs_by_kind("edge"):
        edge_entry = store.get_entry(edge_ref)
        if not edge_entry:
            continue
        adjacent_faces = edge_entry.metadata.get("adjacent_faces") or []
        if not _adjacent_faces_include_target(
            adjacent_faces,
            face_ref=face_ref,
            face_token=face_token,
        ):
            continue

        for point_key in ("start_coords", "end_coords"):
            raw_point = edge_entry.metadata.get(point_key)
            if raw_point is None:
                continue
            px, py, pz = _safe_vec3_extract(raw_point)
            point_cm = (px * scale_to_cm, py * scale_to_cm, pz * scale_to_cm)
            rel = (
                point_cm[0] - origin_cm[0],
                point_cm[1] - origin_cm[1],
                point_cm[2] - origin_cm[2],
            )
            uv_points.append((_dot_vec3(rel, u_axis), _dot_vec3(rel, v_axis)))

    if len(uv_points) < 2:
        return None

    u_vals = [pt[0] for pt in uv_points]
    v_vals = [pt[1] for pt in uv_points]
    return {
        "u_min": min(u_vals),
        "u_max": max(u_vals),
        "v_min": min(v_vals),
        "v_max": max(v_vals),
    }


def _register_sketch_plane_metadata(
    session_id: str,
    manager: ConnectionManager,
    *,
    sketch_id: str,
    result: Mapping[str, Any],
    fallback_plane_id: Optional[str] = None,
) -> None:
    sketch_store = _get_sketch_entity_store(session_id, manager)
    plane_id_input = str(result.get("plane_id_input") or fallback_plane_id or "").strip()
    metadata: Dict[str, Any] = {
        "plane_id_input": plane_id_input,
    }

    face_token = _resolve_face_token_for_plane_id(session_id, manager, plane_id_input)
    if not face_token:
        metadata["plane_kind"] = "datum_or_custom"
        sketch_store.register_sketch_metadata(sketch_id, metadata)
        return

    metadata["plane_kind"] = "face"
    metadata["face_token"] = face_token
    face_entry = _find_entity_entry_by_token(_get_entity_store(session_id, manager), "face", face_token)
    if face_entry:
        metadata["face_ref"] = face_entry.ref_id

    orientation = result.get("orientation")
    plane_origin_world = result.get("plane_origin_world")
    if isinstance(orientation, Mapping) and plane_origin_world is not None:
        uv_bounds = _compute_face_sketch_uv_bounds(
            session_id,
            manager,
            face_token=face_token,
            plane_origin_world=plane_origin_world,
            u_axis_world=orientation.get("u_axis_world"),
            v_axis_world=orientation.get("v_axis_world"),
        )
        if uv_bounds:
            metadata["uv_bounds"] = uv_bounds

    sketch_store.register_sketch_metadata(sketch_id, metadata)


def _preflight_face_sketch_uv_bounds(
    session_id: str,
    manager: ConnectionManager,
    *,
    tool_name: str,
    tool_input: Mapping[str, Any],
) -> Optional[str]:
    if tool_name not in {"add_rectangle", "add_circle"}:
        return None

    sketch_id = str(tool_input.get("sketch_id") or "").strip()
    if not sketch_id:
        return None

    sketch_store = _get_sketch_entity_store(session_id, manager)
    sketch_metadata = sketch_store.get_sketch_metadata(sketch_id)
    if sketch_metadata.get("plane_kind") != "face":
        return None

    bounds = sketch_metadata.get("uv_bounds")
    if not isinstance(bounds, Mapping):
        return (
            f"{tool_name} blocked: sketch '{sketch_id}' is face-based but UV bounds are unavailable. "
            "Recreate the face sketch first, review orientation feedback, then place geometry in a follow-up turn."
        )

    try:
        u_min = float(bounds.get("u_min"))
        u_max = float(bounds.get("u_max"))
        v_min = float(bounds.get("v_min"))
        v_max = float(bounds.get("v_max"))
    except (TypeError, ValueError):
        return (
            f"{tool_name} blocked: sketch '{sketch_id}' has invalid face UV bounds metadata. "
            "Recreate the sketch and retry."
        )

    requested_u_min = requested_u_max = requested_v_min = requested_v_max = 0.0
    try:
        if tool_name == "add_circle":
            cu = float(tool_input.get("center_u"))
            cv = float(tool_input.get("center_v"))
            radius = float(tool_input.get("radius"))
            requested_u_min = cu - radius
            requested_u_max = cu + radius
            requested_v_min = cv - radius
            requested_v_max = cv + radius
        else:
            c1u = float(tool_input.get("corner1_u"))
            c1v = float(tool_input.get("corner1_v"))
            c2u = float(tool_input.get("corner2_u"))
            c2v = float(tool_input.get("corner2_v"))
            requested_u_min = min(c1u, c2u)
            requested_u_max = max(c1u, c2u)
            requested_v_min = min(c1v, c2v)
            requested_v_max = max(c1v, c2v)
    except (TypeError, ValueError):
        # Let normal tool-schema validation handle malformed numeric inputs.
        return None

    margin = FACE_SKETCH_UV_BOUNDS_MARGIN_CM
    outside = (
        requested_u_min < (u_min - margin)
        or requested_u_max > (u_max + margin)
        or requested_v_min < (v_min - margin)
        or requested_v_max > (v_max + margin)
    )
    if not outside:
        return None

    face_ref = str(sketch_metadata.get("face_ref") or "face_?")
    return (
        f"{tool_name} rejected by face-bounds preflight for sketch '{sketch_id}' on {face_ref}. "
        f"Requested UV extents u=[{requested_u_min:.3f}, {requested_u_max:.3f}] v=[{requested_v_min:.3f}, {requested_v_max:.3f}] cm, "
        f"but face bounds are u=[{u_min:.3f}, {u_max:.3f}] v=[{v_min:.3f}, {v_max:.3f}] cm. "
        "Move the geometry onto the selected wall face (or choose a different face/plane) and retry."
    )


def _maybe_defer_face_sketch_followup(
    tool_name: str,
    tool_input: Mapping[str, Any],
    face_sketches_created_this_turn: Set[str],
) -> Optional[str]:
    if not face_sketches_created_this_turn:
        return None
    if tool_name not in FACE_SKETCH_SEQUENCING_BLOCK_TOOLS:
        return None
    sketch_id = str(tool_input.get("sketch_id") or "").strip()
    if not sketch_id or sketch_id not in face_sketches_created_this_turn:
        return None
    return (
        f"Deferred '{tool_name}' for sketch '{sketch_id}'. "
        "This sketch was just created on a model face in the same turn. "
        "Wait for the next turn so orientation/bounds feedback can guide placement before adding geometry or extruding."
    )


def _resolve_codegen_entity_refs(
    session_id: str,
    manager: ConnectionManager,
    tool_name: str,
    tool_input: Dict[str, Any],
) -> Dict[str, Any]:
    """
    For code-generated tools (executed via translate_tool_call), resolve any
    ref-like fields to real entity tokens before code generation.
    """
    store = _get_entity_store(session_id, manager)
    resolved_input = dict(tool_input)

    if tool_name == "revolve_profile":
        axis_spec = resolved_input.get("axis_spec")
        if isinstance(axis_spec, Mapping):
            axis_type = axis_spec.get("type")
            if axis_type == "edge":
                _resolve_token_field_inplace(store, axis_spec, "edge_token", expected_kind="edge")
                _resolve_token_field_inplace(store, axis_spec, "edge_ref", expected_kind="edge")
                # normalize: if edge_ref used, set edge_token
                if "edge_ref" in axis_spec and "edge_token" not in axis_spec:
                    axis_spec["edge_token"] = axis_spec.pop("edge_ref")
            elif axis_type == "face":
                _resolve_token_field_inplace(store, axis_spec, "face_token", expected_kind="face")
                _resolve_token_field_inplace(store, axis_spec, "face_ref", expected_kind="face")
                if "face_ref" in axis_spec and "face_token" not in axis_spec:
                    axis_spec["face_token"] = axis_spec.pop("face_ref")

        extent_spec = resolved_input.get("extent_spec")
        if isinstance(extent_spec, Mapping):
            _resolve_token_field_inplace(store, extent_spec, "to_entity_token")
            _resolve_token_field_inplace(store, extent_spec, "to_entity1_token")
            _resolve_token_field_inplace(store, extent_spec, "to_entity2_token")
            _resolve_token_field_inplace(store, extent_spec, "to_entity_ref")
            _resolve_token_field_inplace(store, extent_spec, "to_entity1_ref")
            _resolve_token_field_inplace(store, extent_spec, "to_entity2_ref")
            # Normalize *_ref → *_token if only ref was provided
            for ref_key, token_key in [
                ("to_entity_ref", "to_entity_token"),
                ("to_entity1_ref", "to_entity1_token"),
                ("to_entity2_ref", "to_entity2_token"),
            ]:
                if ref_key in extent_spec and token_key not in extent_spec:
                    extent_spec[token_key] = extent_spec.pop(ref_key)

        _resolve_token_field_inplace(store, resolved_input, "creation_occurrence_token")
        _resolve_token_field_inplace(store, resolved_input, "creation_occurrence_ref")
        if "creation_occurrence_ref" in resolved_input and "creation_occurrence_token" not in resolved_input:
            resolved_input["creation_occurrence_token"] = resolved_input.pop("creation_occurrence_ref")

    elif tool_name in {"create_construction_plane"}:
        for key in ("reference_edge_token", "reference_edge_ref", "face_token", "face_ref"):
            _resolve_token_field_inplace(store, resolved_input, key, expected_kind="edge" if "edge" in key else "face")
            if key.endswith("_ref"):
                token_key = key.replace("_ref", "_token")
                if key in resolved_input and token_key not in resolved_input:
                    resolved_input[token_key] = resolved_input.pop(key)

    elif tool_name == "create_sketch":
        # Allow plane_id to be a face ref (face_0, face_1, ...)
        # PlaneManager.get_plane() will auto-create a construction plane from the face token.
        plane_id = resolved_input.get("plane_id", "")
        if plane_id:
            token, error = store.resolve_token(str(plane_id).strip(), expected_kind="face")
            if token and not error and token != plane_id:
                resolved_input["plane_id"] = token
                logger.debug(f"Resolved face ref '{plane_id}' to token for create_sketch")
            elif error and re.match(r"^face_\d+$", str(plane_id).strip()):
                # LLM used a face_N ref that can't be resolved — likely no entity context yet.
                face_refs = store.get_refs_by_kind("face")
                body_refs = store.get_refs_by_kind("body")

                if not face_refs and not body_refs:
                        raise SelectionToolCallError(
                            f"create_sketch plane_id could not be resolved: {plane_id}. {error}. "
                            "No design entities are loaded in the current entity context. "
                            "Wait for refreshed Design Entities after a successful geometry operation, then retry "
                            "with plane_id as a real face ref (e.g., face_0) or provide an explicit datum plane (XY/XZ/YZ)."
                        )
                elif not face_refs:
                        raise SelectionToolCallError(
                            f"create_sketch plane_id could not be resolved: {plane_id}. {error}. "
                            "No face refs are loaded in the current entity context. "
                            "Wait for refreshed Design Entities after a successful geometry operation, "
                            "then retry with plane_id as a real face ref (e.g., face_0) or a datum plane (XY/XZ/YZ)."
                        )
                else:
                    face_preview = ", ".join(sorted(face_refs)[:25])
                    if len(face_refs) > 25:
                        face_preview = f"{face_preview}, ..."
                    candidate_refs = _candidate_faces_for_stale_ref(store, str(plane_id).strip())
                    candidate_text = f" Nearest candidates: {', '.join(candidate_refs)}." if candidate_refs else ""
                    raise SelectionToolCallError(
                        f"create_sketch plane_id could not be resolved: {plane_id}. {error}. "
                        f"Available face refs: {face_preview}.{candidate_text}"
                    )

    return resolved_input


def _format_edge_listing_message(result: Mapping[str, Any], base_message: str) -> str:
    edges: List[Mapping[str, Any]] = result.get("edges") or []
    units = result.get("units") or "cm"

    lines = [base_message, f"units={units}", f"edge_count={len(edges)}"]

    structured_edges = []
    for index, edge in enumerate(edges, 1):
        body_name = edge.get("body_name", f"Body {edge.get('body_index', '?')}")
        geometry_type = edge.get("geometry_type", "Unknown")
        length = edge.get("length")
        start_coords = edge.get("start_coords")
        end_coords = edge.get("end_coords")
        entity_ref = edge.get("entity_ref")
        edge_id = edge.get("id", f"edge_{index}")

        lines.append(
            f"{index}. {edge_id} | ref={entity_ref or '-'} | body={body_name} | type={geometry_type} | "
            f"length={length} {units} | start={_format_coords(start_coords)} | "
            f"end={_format_coords(end_coords)}"
        )

        structured_edges.append(
            {
                "id": edge_id,
                "entity_ref": entity_ref,
                "body": body_name,
                "geometry_type": geometry_type,
                "length": length,
                "units": units,
                "start": start_coords,
                "end": end_coords,
            }
        )

    if structured_edges:
        try:
            lines.append("edge_data_json=" + json.dumps(structured_edges, ensure_ascii=False, separators=(",", ":")))
        except (TypeError, ValueError):
            logger.debug("Failed to serialize edge metadata to JSON for tool response.")

    return "\n".join(lines)


def _format_select_edges_message(result: Mapping[str, Any], base_message: str) -> str:
    selected = result.get("selected_count")
    missing = result.get("missing_tokens") or []
    cleared = result.get("cleared_existing")

    lines = [base_message]
    if selected is not None:
        lines.append(f"selected_count={selected}")
    if cleared is not None:
        lines.append(f"cleared_existing={bool(cleared)}")
    if missing:
        lines.append(f"missing_tokens={missing}")

    return "\n".join(lines)


def _format_clear_edges_message(result: Mapping[str, Any], base_message: str) -> str:
    cleared = result.get("cleared_count")
    lines = [base_message]
    if cleared is not None:
        lines.append(f"cleared_count={cleared}")
    return "\n".join(lines)


def _format_face_listing_message(result: Mapping[str, Any], base_message: str) -> str:
    faces: List[Mapping[str, Any]] = result.get("faces") or []
    units: Mapping[str, Any] = result.get("units") or {}
    length_units = units.get("length") or "cm"
    area_units = units.get("area") or "cm^2"

    lines = [
        base_message,
        f"length_units={length_units}",
        f"area_units={area_units}",
        f"face_count={len(faces)}",
    ]

    structured_faces = []
    for index, face in enumerate(faces, 1):
        body_name = face.get("body_name", f"Body {face.get('body_index', '?')}")
        geometry_type = face.get("geometry_type", "Unknown")
        area = face.get("area")
        centroid = face.get("centroid")
        bbox = face.get("bounding_box")
        entity_ref = face.get("entity_ref")
        face_id = face.get("id", f"face_{index}")
        edges = face.get("edges") or []
        orientation = face.get("orientation")

        # Format bounds as compact axis ranges for LLM spatial awareness
        bounds_str = _format_bounds(bbox, length_units)

        lines.append(
            f"{index}. {face_id} | ref={entity_ref or '-'} | body={body_name} | type={geometry_type} | "
            f"area={area} {area_units} | centroid={_format_coords(centroid)} | {bounds_str} | "
            f"edges={len(edges)} | orientation={orientation or '-'}"
        )

        face_struct = {
            "id": face_id,
            "body": body_name,
            "geometry_type": geometry_type,
            "area": area,
            "area_units": area_units,
            "centroid": centroid,
            "bounding_box": bbox,
            "entity_ref": entity_ref,
            "edges": edges,
        }

        # Pass through new spatial-awareness fields when present
        for key in (
            "normal",
            "orientation",
            "u_direction",
            "v_direction",
            "axis",
            "radius",
            "axis_orientation",
            "temp_id",
        ):
            if key in face:
                face_struct[key] = face.get(key)

        structured_faces.append(face_struct)

    if structured_faces:
        try:
            lines.append("face_data_json=" + json.dumps(structured_faces, ensure_ascii=False, separators=(",", ":")))
        except (TypeError, ValueError):
            logger.debug("Failed to serialize face metadata to JSON for tool response.")

    return "\n".join(lines)


def _format_select_faces_message(result: Mapping[str, Any], base_message: str) -> str:
    selected = result.get("selected_count")
    missing = result.get("missing_tokens") or []
    cleared = result.get("cleared_existing")

    lines = [base_message]
    if selected is not None:
        lines.append(f"selected_count={selected}")
    if cleared is not None:
        lines.append(f"cleared_existing={bool(cleared)}")
    if missing:
        lines.append(f"missing_tokens={missing}")

    return "\n".join(lines)


def _format_clear_faces_message(result: Mapping[str, Any], base_message: str) -> str:
    cleared = result.get("cleared_count")
    lines = [base_message]
    if cleared is not None:
        lines.append(f"cleared_count={cleared}")
    return "\n".join(lines)


def _format_body_listing_message(result: Mapping[str, Any], base_message: str) -> str:
    bodies: List[Mapping[str, Any]] = result.get("bodies") or []
    units: Mapping[str, Any] = result.get("units") or {}
    length_units = units.get("length") or "cm"
    area_units = units.get("area") or f"{length_units}^2"
    volume_units = units.get("volume") or f"{length_units}^3"
    mass_units = units.get("mass") or "kg"
    density_units = units.get("density") or f"{mass_units}/{length_units}^3"

    lines = [
        base_message,
        f"length_units={length_units}",
        f"area_units={area_units}",
        f"volume_units={volume_units}",
        f"mass_units={mass_units}",
        f"density_units={density_units}",
        f"body_count={len(bodies)}",
    ]

    structured_bodies = []
    for index, body in enumerate(bodies, 1):
        body_id = body.get("id", f"body_{index}")
        name = body.get("name") or body_id
        component = body.get("component")
        volume = body.get("volume")
        area = body.get("area")
        is_solid = body.get("is_solid")
        face_count = body.get("face_count")
        entity_ref = body.get("entity_ref")
        center_of_mass = body.get("center_of_mass")
        mass = body.get("mass")
        density = body.get("density")
        bbox = body.get("bounding_box") or {}

        lines.append(
            f"{index}. {body_id} | ref={entity_ref or '-'} | name={name} | component={component} | solid={bool(is_solid)} | "
            f"volume={volume} {body.get('volume_units', volume_units)} | area={area} {body.get('area_units', area_units)} | "
            f"faces={face_count} | mass={mass} {body.get('mass_units', mass_units)} | "
            f"com={_format_coords(center_of_mass)}"
        )

        structured_bodies.append(
            {
                "id": body_id,
                "name": name,
                "component": component,
                "is_solid": bool(is_solid),
                "volume": volume,
                "volume_units": body.get("volume_units", volume_units),
                "area": area,
                "area_units": body.get("area_units", area_units),
                "mass": mass,
                "mass_units": body.get("mass_units", mass_units),
                "density": density,
                "density_units": body.get("density_units", density_units),
                "center_of_mass": center_of_mass,
                "bounding_box": bbox,
                "face_count": face_count,
                "edge_count": body.get("edge_count"),
                "vertex_count": body.get("vertex_count"),
                "entity_ref": entity_ref,
            }
        )

    if structured_bodies:
        try:
            lines.append("body_data_json=" + json.dumps(structured_bodies, ensure_ascii=False, separators=(",", ":")))
        except (TypeError, ValueError):
            logger.debug("Failed to serialize body metadata to JSON for tool response.")

    return "\n".join(lines)


def _format_select_bodies_message(result: Mapping[str, Any], base_message: str) -> str:
    selected = result.get("selected_count")
    missing = result.get("missing_tokens") or []
    cleared = result.get("cleared_existing")

    lines = [base_message]
    if selected is not None:
        lines.append(f"selected_count={selected}")
    if cleared is not None:
        lines.append(f"cleared_existing={bool(cleared)}")
    if missing:
        lines.append(f"missing_tokens={missing}")

    return "\n".join(lines)


def _format_clear_bodies_message(result: Mapping[str, Any], base_message: str) -> str:
    cleared = result.get("cleared_count")
    lines = [base_message]
    if cleared is not None:
        lines.append(f"cleared_count={cleared}")
    return "\n".join(lines)


def _validate_adjust_feature_parameters(tool_input: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate narrow feature parameter edits before sending them to Fusion."""
    allowed_top_level = {
        "feature_token",
        "parameters",
        "expected_name",
        "expected_timeline_index",
        "description",
    }
    extra = set(tool_input.keys()) - allowed_top_level
    if extra:
        raise SelectionToolCallError(f"adjust_feature_parameters received unexpected parameter(s): {sorted(extra)}")

    feature_token = str(tool_input.get("feature_token") or "").strip()
    if not feature_token:
        raise SelectionToolCallError("adjust_feature_parameters requires a non-empty 'feature_token'.")

    parameters = tool_input.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise SelectionToolCallError("adjust_feature_parameters requires a non-empty 'parameters' object.")

    allowed_params = {
        "name",
        "distance",
        "distance_unit",
        "diameter",
        "diameter_unit",
        "depth",
        "depth_unit",
    }
    param_extra = set(parameters.keys()) - allowed_params
    if param_extra:
        raise SelectionToolCallError(
            f"adjust_feature_parameters does not support parameter(s): {sorted(param_extra)}. "
            "Supported parameters are name, distance, diameter, and depth with optional units."
        )

    cleaned_params: Dict[str, Any] = {}
    if "name" in parameters:
        name = str(parameters.get("name") or "").strip()
        if not name:
            raise SelectionToolCallError("'parameters.name' must be non-empty when provided.")
        cleaned_params["name"] = name[:120]

    for numeric_key in ("distance", "diameter", "depth"):
        if numeric_key not in parameters:
            continue
        value = parameters.get(numeric_key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            raise SelectionToolCallError(f"'parameters.{numeric_key}' must be a positive number.")
        cleaned_params[numeric_key] = float(value)

    for unit_key in ("distance_unit", "diameter_unit", "depth_unit"):
        if unit_key not in parameters:
            continue
        unit = str(parameters.get(unit_key) or "").strip().lower()
        if unit not in {"mm", "cm", "m", "in"}:
            raise SelectionToolCallError(f"'parameters.{unit_key}' must be one of ['mm', 'cm', 'm', 'in'].")
        cleaned_params[unit_key] = unit

    if not any(key in cleaned_params for key in ("name", "distance", "diameter", "depth")):
        raise SelectionToolCallError("Provide at least one editable parameter: name, distance, diameter, or depth.")

    cleaned: Dict[str, Any] = {
        "feature_token": feature_token,
        "parameters": cleaned_params,
    }

    expected_name = str(tool_input.get("expected_name") or "").strip()
    if expected_name:
        cleaned["expected_name"] = expected_name

    expected_index = tool_input.get("expected_timeline_index")
    if expected_index is not None:
        try:
            cleaned["expected_timeline_index"] = int(expected_index)
        except (TypeError, ValueError):
            raise SelectionToolCallError("'expected_timeline_index' must be an integer when provided.")

    return cleaned


async def _execute_feature_tool_call(
    session_id: str,
    manager: ConnectionManager,
    tool_name: str,
    tool_use_id: str,
    tool_input: Mapping[str, Any],
    description: str = "",
) -> Tuple[bool, str, Mapping[str, Any]]:
    """Send a feature tool request (fillet, chamfer) to Fusion and return the formatted result."""

    if tool_name == "list_features":
        success, text = await _handle_list_features(session_id, manager, tool_input)
        return success, text, {}

    pattern_prep: Optional[PatternPreparation] = None

    payload: Dict[str, Any] = {
        "type": "feature_operation",
        "operation": tool_name,
        "tool_use_id": tool_use_id,
        "description": description,
        "parameters": dict(tool_input) if isinstance(tool_input, Mapping) else {},
    }

    if tool_name == "adjust_feature_parameters":
        cleaned = _validate_adjust_feature_parameters(tool_input)
        payload["feature_token"] = cleaned["feature_token"]
        payload["parameters"] = cleaned["parameters"]
        if "expected_name" in cleaned:
            payload["expected_name"] = cleaned["expected_name"]
        if "expected_timeline_index" in cleaned:
            payload["expected_timeline_index"] = cleaned["expected_timeline_index"]

    elif tool_name == "create_pattern_feature":
        pattern_prep = await _prepare_pattern_feature(session_id, manager, tool_input)
        payload["parameters"] = pattern_prep.parameters

    elif tool_name == "apply_fillet":
        # Normalize: edge_refs -> entity_tokens
        tool_input = _normalize_tool_params(tool_name, tool_input)

        if "entity_tokens" not in tool_input:
            raise SelectionToolCallError("apply_fillet requires an 'entity_tokens' (or 'edge_refs') array.")
        if "radius" not in tool_input:
            raise SelectionToolCallError("apply_fillet requires a 'radius' parameter.")

        resolved_tokens = _resolve_entity_tokens_or_refs(
            session_id,
            manager,
            tool_input.get("entity_tokens"),
            expected_kind="edge",
            context="entity_tokens (edges)",
        )

        radius = tool_input.get("radius")
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise SelectionToolCallError("'radius' must be a positive number.")

        payload["entity_tokens"] = resolved_tokens
        payload["radius"] = float(radius)
        payload["radius_unit"] = str(tool_input.get("radius_unit", "mm"))
        payload["include_tangent_edges"] = bool(tool_input.get("include_tangent_edges", True))
        if "feature_name" in tool_input:
            payload["feature_name"] = str(tool_input.get("feature_name", "")).strip()

        # Accept both new (edge_refs) and legacy (entity_tokens) names
        extra = set(tool_input.keys()) - {"entity_tokens", "edge_refs", "radius", "radius_unit", "include_tangent_edges", "feature_name", "description"}
        if extra:
            raise SelectionToolCallError(f"apply_fillet received unexpected parameter(s): {sorted(extra)}")

    elif tool_name == "apply_chamfer":
        # Normalize: edge_refs -> entity_tokens
        tool_input = _normalize_tool_params(tool_name, tool_input)

        if "entity_tokens" not in tool_input:
            raise SelectionToolCallError("apply_chamfer requires an 'entity_tokens' (or 'edge_refs') array.")
        if "distance" not in tool_input:
            raise SelectionToolCallError("apply_chamfer requires a 'distance' parameter.")

        resolved_tokens = _resolve_entity_tokens_or_refs(
            session_id,
            manager,
            tool_input.get("entity_tokens"),
            expected_kind="edge",
            context="entity_tokens (edges)",
        )

        distance = tool_input.get("distance")
        if not isinstance(distance, (int, float)) or distance <= 0:
            raise SelectionToolCallError("'distance' must be a positive number.")

        payload["entity_tokens"] = resolved_tokens
        payload["distance"] = float(distance)
        payload["distance_unit"] = str(tool_input.get("distance_unit", "mm"))
        payload["include_tangent_edges"] = bool(tool_input.get("include_tangent_edges", True))
        if "feature_name" in tool_input:
            payload["feature_name"] = str(tool_input.get("feature_name", "")).strip()

        # Accept both new (edge_refs) and legacy (entity_tokens) names
        extra = set(tool_input.keys()) - {"entity_tokens", "edge_refs", "distance", "distance_unit", "include_tangent_edges", "feature_name", "description"}
        if extra:
            raise SelectionToolCallError(f"apply_chamfer received unexpected parameter(s): {sorted(extra)}")

    elif tool_name == "create_shell":
        # create_shell uses mode discriminator: 'open' (face_refs) or 'closed' (body_refs)
        # Also supports legacy entity_tokens for backward compatibility
        
        # Get mode (default to 'open' for backward compatibility)
        mode = str(tool_input.get("mode", "open")).strip().lower()
        if mode not in {"open", "closed"}:
            raise SelectionToolCallError(f"create_shell mode must be 'open' or 'closed', got: '{mode}'")
        
        # Extract arrays, treating missing keys as empty
        face_refs = tool_input.get("face_refs") or []
        body_refs = tool_input.get("body_refs") or []
        entity_tokens = tool_input.get("entity_tokens") or []
        
        # Normalize to lists
        if isinstance(face_refs, str):
            face_refs = [face_refs]
        if isinstance(body_refs, str):
            body_refs = [body_refs]
        if isinstance(entity_tokens, str):
            entity_tokens = [entity_tokens]
        
        has_faces = len(face_refs) > 0
        has_bodies = len(body_refs) > 0
        has_legacy = len(entity_tokens) > 0
        
        # Validate mode matches provided refs - reject wrong ref type explicitly
        if mode == "open":
            # mode='open' uses face_refs; reject body_refs if provided
            if has_bodies:
                raise SelectionToolCallError(
                    "create_shell mode='open' does not accept 'body_refs'. "
                    "Remove body_refs and provide face_refs, or use mode='closed' for body hollowing."
                )
            if has_faces:
                tokens_to_resolve = face_refs
                expected_kind = "face"
            elif has_legacy:
                # Legacy entity_tokens must be faces for mode='open'
                tokens_to_resolve = entity_tokens
                expected_kind = "face"
            else:
                raise SelectionToolCallError(
                    "create_shell mode='open' requires 'face_refs' array with faces to remove."
                )
        else:  # mode == "closed"
            # mode='closed' uses body_refs; reject face_refs if provided
            if has_faces:
                raise SelectionToolCallError(
                    "create_shell mode='closed' does not accept 'face_refs'. "
                    "Remove face_refs and provide body_refs, or use mode='open' to remove faces."
                )
            if has_bodies:
                tokens_to_resolve = body_refs
                expected_kind = "body"
            elif has_legacy:
                # Legacy entity_tokens must be bodies for mode='closed'
                tokens_to_resolve = entity_tokens
                expected_kind = "body"
            else:
                raise SelectionToolCallError(
                    "create_shell mode='closed' requires 'body_refs' array with bodies to hollow."
                )

        normalized_tokens = _resolve_entity_tokens_or_refs(
            session_id,
            manager,
            tokens_to_resolve,
            expected_kind=expected_kind,
            context=f"create_shell mode='{mode}' expects {expected_kind} tokens",
        )

        inside = tool_input.get("inside_thickness", 0)
        outside = tool_input.get("outside_thickness", 0)

        if not isinstance(inside, (int, float)) or inside < 0:
            raise SelectionToolCallError("'inside_thickness' must be a number >= 0.")
        if not isinstance(outside, (int, float)) or outside < 0:
            raise SelectionToolCallError("'outside_thickness' must be a number >= 0.")
        if inside == 0 and outside == 0:
            raise SelectionToolCallError("At least one of 'inside_thickness' or 'outside_thickness' must be > 0.")

        thickness_unit = str(tool_input.get("thickness_unit", "mm")).strip().lower()
        if thickness_unit not in {"mm", "cm", "m", "in"}:
            raise SelectionToolCallError(f"Invalid thickness_unit: {thickness_unit}")

        is_tangent_chain_value = tool_input.get("is_tangent_chain", True)
        if not isinstance(is_tangent_chain_value, bool):
            raise SelectionToolCallError("'is_tangent_chain' must be a boolean.")
        is_tangent_chain = is_tangent_chain_value
        shell_type = str(tool_input.get("shell_type", "sharp")).strip().lower()
        if shell_type not in {"sharp", "rounded"}:
            raise SelectionToolCallError(f"Invalid shell_type: {shell_type}")

        payload["entity_tokens"] = normalized_tokens
        payload["inside_thickness"] = float(inside)
        payload["outside_thickness"] = float(outside)
        payload["thickness_unit"] = thickness_unit
        payload["is_tangent_chain"] = is_tangent_chain
        payload["shell_type"] = shell_type
        if "feature_name" in tool_input:
            payload["feature_name"] = str(tool_input.get("feature_name", "")).strip()

        # Accept mode, face_refs, body_refs, and legacy entity_tokens
        allowed = {
            "mode",
            "entity_tokens",
            "face_refs",
            "body_refs",
            "inside_thickness",
            "outside_thickness",
            "thickness_unit",
            "is_tangent_chain",
            "shell_type",
            "feature_name",
            "description",
        }
        extra = set(tool_input.keys()) - allowed
        if extra:
            raise SelectionToolCallError(f"create_shell received unexpected parameter(s): {sorted(extra)}")

    elif tool_name == "create_simple_hole":
        # Normalize: face_ref -> face_token
        tool_input = _normalize_tool_params(tool_name, tool_input)

        # Validate face_token
        if "face_token" not in tool_input:
            raise SelectionToolCallError("create_simple_hole requires a 'face_token' (or 'face_ref') string.")

        face_token = _resolve_single_entity_ref(
            session_id,
            manager,
            tool_input.get("face_token", ""),
            expected_kind="face",
            context="face_token",
        )
        
        # Validate coordinates
        for coord in ["center_x", "center_y", "center_z"]:
            if coord not in tool_input:
                raise SelectionToolCallError(f"create_simple_hole requires '{coord}'.")
            if not isinstance(tool_input.get(coord), (int, float)):
                raise SelectionToolCallError(f"'{coord}' must be a number.")
        
        # Validate diameter
        if "diameter" not in tool_input:
            raise SelectionToolCallError("create_simple_hole requires 'diameter'.")
        diameter = tool_input.get("diameter")
        if not isinstance(diameter, (int, float)) or diameter <= 0:
            raise SelectionToolCallError("'diameter' must be a positive number.")
        
        # Validate extent_type
        if "extent_type" not in tool_input:
            raise SelectionToolCallError("create_simple_hole requires 'extent_type'.")
        extent_type = tool_input.get("extent_type")
        if extent_type not in ["through_all", "distance"]:
            raise SelectionToolCallError("'extent_type' must be 'through_all' or 'distance'.")
        
        # Validate depth (if extent_type is distance)
        if extent_type == "distance":
            if "depth" not in tool_input:
                raise SelectionToolCallError("create_simple_hole with extent_type='distance' requires 'depth'.")
            depth = tool_input.get("depth")
            if not isinstance(depth, (int, float)) or depth <= 0:
                raise SelectionToolCallError("'depth' must be a positive number.")
        
        # Build payload
        payload["face_token"] = face_token
        payload["center_x"] = float(tool_input.get("center_x"))
        payload["center_y"] = float(tool_input.get("center_y"))
        payload["center_z"] = float(tool_input.get("center_z"))
        preflight_error = _preflight_hole_center_on_face(
            session_id,
            manager,
            tool_name=tool_name,
            face_token=face_token,
            center_x=payload["center_x"],
            center_y=payload["center_y"],
            center_z=payload["center_z"],
        )
        if preflight_error:
            raise SelectionToolCallError(preflight_error)
        payload["diameter"] = float(diameter)
        payload["diameter_unit"] = str(tool_input.get("diameter_unit", "mm"))
        payload["extent_type"] = str(extent_type)
        
        if extent_type == "distance":
            payload["depth"] = float(tool_input.get("depth"))
        
        if "feature_name" in tool_input:
            payload["feature_name"] = str(tool_input.get("feature_name", "")).strip()
        
        # Accept both new (face_ref) and legacy (face_token) names
        allowed = {"face_token", "face_ref", "center_x", "center_y", "center_z", "diameter", "extent_type", "depth", "diameter_unit", "feature_name", "description"}
        extra = set(tool_input.keys()) - allowed
        if extra:
            raise SelectionToolCallError(f"create_simple_hole received unexpected parameter(s): {sorted(extra)}")

    elif tool_name == "create_counterbore_hole":
        # Normalize: face_ref -> face_token
        tool_input = _normalize_tool_params(tool_name, tool_input)

        # Validate face_token
        if "face_token" not in tool_input:
            raise SelectionToolCallError("create_counterbore_hole requires 'face_token' (or 'face_ref').")

        face_token = _resolve_single_entity_ref(
            session_id,
            manager,
            tool_input.get("face_token", ""),
            expected_kind="face",
            context="face_token",
        )
        
        # Validate coordinates
        for coord in ["center_x", "center_y", "center_z"]:
            if coord not in tool_input:
                raise SelectionToolCallError(f"create_counterbore_hole requires '{coord}'.")
            if not isinstance(tool_input.get(coord), (int, float)):
                raise SelectionToolCallError(f"'{coord}' must be a number.")
        
        # Validate hole_diameter
        if "hole_diameter" not in tool_input:
            raise SelectionToolCallError("create_counterbore_hole requires 'hole_diameter'.")
        hole_diameter = tool_input.get("hole_diameter")
        if not isinstance(hole_diameter, (int, float)) or hole_diameter <= 0:
            raise SelectionToolCallError("'hole_diameter' must be positive.")
        
        # Validate counterbore_diameter
        if "counterbore_diameter" not in tool_input:
            raise SelectionToolCallError("create_counterbore_hole requires 'counterbore_diameter'.")
        counterbore_diameter = tool_input.get("counterbore_diameter")
        if not isinstance(counterbore_diameter, (int, float)) or counterbore_diameter <= 0:
            raise SelectionToolCallError("'counterbore_diameter' must be positive.")
        
        # Validate counterbore > hole diameter
        if counterbore_diameter <= hole_diameter:
            raise SelectionToolCallError(
                f"'counterbore_diameter' ({counterbore_diameter}) must be larger than 'hole_diameter' ({hole_diameter})."
            )
        
        # Validate hole_depth
        if "hole_depth" not in tool_input:
            raise SelectionToolCallError("create_counterbore_hole requires 'hole_depth'.")
        hole_depth = tool_input.get("hole_depth")
        if not isinstance(hole_depth, (int, float)) or hole_depth <= 0:
            raise SelectionToolCallError("'hole_depth' must be positive.")
        
        # Validate counterbore_depth
        if "counterbore_depth" not in tool_input:
            raise SelectionToolCallError("create_counterbore_hole requires 'counterbore_depth'.")
        counterbore_depth = tool_input.get("counterbore_depth")
        if not isinstance(counterbore_depth, (int, float)) or counterbore_depth <= 0:
            raise SelectionToolCallError("'counterbore_depth' must be positive.")
        
        # Build payload
        payload["face_token"] = face_token
        payload["center_x"] = float(tool_input.get("center_x"))
        payload["center_y"] = float(tool_input.get("center_y"))
        payload["center_z"] = float(tool_input.get("center_z"))
        preflight_error = _preflight_hole_center_on_face(
            session_id,
            manager,
            tool_name=tool_name,
            face_token=face_token,
            center_x=payload["center_x"],
            center_y=payload["center_y"],
            center_z=payload["center_z"],
        )
        if preflight_error:
            raise SelectionToolCallError(preflight_error)
        payload["hole_diameter"] = float(hole_diameter)
        payload["hole_depth"] = float(hole_depth)
        payload["counterbore_diameter"] = float(counterbore_diameter)
        payload["counterbore_depth"] = float(counterbore_depth)
        payload["diameter_unit"] = str(tool_input.get("diameter_unit", "mm"))
        
        if "feature_name" in tool_input:
            payload["feature_name"] = str(tool_input.get("feature_name", "")).strip()
        
        # Accept both new (face_ref) and legacy (face_token) names
        allowed = {
            "face_token", "face_ref", "center_x", "center_y", "center_z",
            "hole_diameter", "hole_depth", "counterbore_diameter", "counterbore_depth",
            "diameter_unit", "feature_name", "description"
        }
        extra = set(tool_input.keys()) - allowed
        if extra:
            raise SelectionToolCallError(f"create_counterbore_hole received unexpected parameter(s): {sorted(extra)}")

    elif tool_name == "create_tapped_hole":
        # Normalize: face_ref -> face_token
        tool_input = _normalize_tool_params(tool_name, tool_input)

        if "face_token" not in tool_input:
            raise SelectionToolCallError("create_tapped_hole requires 'face_token' (or 'face_ref').")
        face_token = _resolve_single_entity_ref(
            session_id,
            manager,
            tool_input.get("face_token", ""),
            expected_kind="face",
            context="face_token",
        )

        for coord in ["center_x", "center_y", "center_z"]:
            if coord not in tool_input:
                raise SelectionToolCallError(f"create_tapped_hole requires '{coord}'.")
            if not isinstance(tool_input.get(coord), (int, float)):
                raise SelectionToolCallError(f"'{coord}' must be a number.")

        if "thread_type" not in tool_input:
            raise SelectionToolCallError("create_tapped_hole requires 'thread_type'.")
        thread_type = str(tool_input.get("thread_type", "")).strip().lower()
        if not thread_type:
            raise SelectionToolCallError("'thread_type' cannot be empty.")
        if thread_type == "iso":
            thread_type = "metric"
        if thread_type not in {"metric", "unc", "unf"}:
            raise SelectionToolCallError(
                f"'thread_type' must be 'metric', 'unc', or 'unf', got '{thread_type}'."
            )

        if "thread_size" not in tool_input:
            raise SelectionToolCallError("create_tapped_hole requires 'thread_size'.")
        thread_size = str(tool_input.get("thread_size", "")).strip()
        if not thread_size:
            raise SelectionToolCallError("'thread_size' cannot be empty.")

        if "thread_depth" not in tool_input:
            raise SelectionToolCallError("create_tapped_hole requires 'thread_depth'.")
        thread_depth = tool_input.get("thread_depth")
        if not isinstance(thread_depth, (int, float)) or thread_depth <= 0:
            raise SelectionToolCallError("'thread_depth' must be a positive number.")

        pilot_hole_depth = tool_input.get("pilot_hole_depth")
        if pilot_hole_depth is not None:
            if not isinstance(pilot_hole_depth, (int, float)) or pilot_hole_depth <= 0:
                raise SelectionToolCallError("'pilot_hole_depth' must be positive if provided.")
            if pilot_hole_depth < thread_depth:
                raise SelectionToolCallError(
                    f"'pilot_hole_depth' ({pilot_hole_depth}) must be >= 'thread_depth' ({thread_depth})."
                )

        diameter_unit = str(tool_input.get("diameter_unit", "mm")).strip().lower()
        if diameter_unit not in {"mm", "cm", "m", "in"}:
            raise SelectionToolCallError(
                f"'diameter_unit' must be one of ['mm', 'cm', 'm', 'in'], got '{diameter_unit}'."
            )

        payload["face_token"] = face_token
        payload["center_x"] = float(tool_input.get("center_x"))
        payload["center_y"] = float(tool_input.get("center_y"))
        payload["center_z"] = float(tool_input.get("center_z"))
        preflight_error = _preflight_hole_center_on_face(
            session_id,
            manager,
            tool_name=tool_name,
            face_token=face_token,
            center_x=payload["center_x"],
            center_y=payload["center_y"],
            center_z=payload["center_z"],
        )
        if preflight_error:
            raise SelectionToolCallError(preflight_error)
        payload["thread_type"] = thread_type
        payload["thread_size"] = thread_size
        payload["thread_depth"] = float(thread_depth)
        payload["diameter_unit"] = diameter_unit

        if pilot_hole_depth is not None:
            payload["pilot_hole_depth"] = float(pilot_hole_depth)

        if "feature_name" in tool_input:
            payload["feature_name"] = str(tool_input.get("feature_name", "")).strip()

        # Accept both new (face_ref) and legacy (face_token) names
        allowed = {
            "face_token",
            "face_ref",
            "center_x",
            "center_y",
            "center_z",
            "thread_type",
            "thread_size",
            "thread_depth",
            "pilot_hole_depth",
            "diameter_unit",
            "feature_name",
            "description",
        }
        extra = set(tool_input.keys()) - allowed
        if extra:
            raise SelectionToolCallError(f"create_tapped_hole received unexpected parameter(s): {sorted(extra)}")

    elif tool_name == "create_external_thread":
        # Normalize: face_ref -> face_token
        tool_input = _normalize_tool_params(tool_name, tool_input)

        if "face_token" not in tool_input:
            raise SelectionToolCallError("create_external_thread requires 'face_token' (or 'face_ref').")
        face_token = _resolve_single_entity_ref(
            session_id,
            manager,
            tool_input.get("face_token", ""),
            expected_kind="face",
            context="face_token",
        )

        if "thread_type" not in tool_input:
            raise SelectionToolCallError("create_external_thread requires 'thread_type'.")
        thread_type = str(tool_input.get("thread_type", "")).strip().lower()
        if not thread_type:
            raise SelectionToolCallError("'thread_type' cannot be empty.")
        if thread_type == "iso":
            thread_type = "metric"
        if thread_type not in {"metric", "unc", "unf"}:
            raise SelectionToolCallError(
                f"'thread_type' must be 'metric', 'unc', or 'unf', got '{thread_type}'."
            )

        if "thread_size" not in tool_input:
            raise SelectionToolCallError("create_external_thread requires 'thread_size'.")
        thread_size = str(tool_input.get("thread_size", "")).strip()
        if not thread_size:
            raise SelectionToolCallError("'thread_size' cannot be empty.")

        is_full_length = tool_input.get("is_full_length", True)
        if not isinstance(is_full_length, bool):
            raise SelectionToolCallError("'is_full_length' must be a boolean.")

        thread_length = tool_input.get("thread_length")
        if not is_full_length:
            if thread_length is None:
                raise SelectionToolCallError("'thread_length' is required when 'is_full_length' is False.")
            if not isinstance(thread_length, (int, float)) or thread_length <= 0:
                raise SelectionToolCallError("'thread_length' must be a positive number.")

        thread_offset = tool_input.get("thread_offset", 0.0)
        if not isinstance(thread_offset, (int, float)) or thread_offset < 0:
            raise SelectionToolCallError("'thread_offset' must be a non-negative number.")

        diameter_unit = str(tool_input.get("diameter_unit", "mm")).strip().lower()
        if diameter_unit not in {"mm", "cm", "m", "in"}:
            raise SelectionToolCallError(
                f"'diameter_unit' must be one of ['mm', 'cm', 'm', 'in'], got '{diameter_unit}'."
            )

        payload["face_token"] = face_token
        payload["thread_type"] = thread_type
        payload["thread_size"] = thread_size
        payload["is_full_length"] = is_full_length
        payload["diameter_unit"] = diameter_unit

        if thread_length is not None:
            payload["thread_length"] = float(thread_length)

        if thread_offset != 0.0:
            payload["thread_offset"] = float(thread_offset)

        if "feature_name" in tool_input:
            payload["feature_name"] = str(tool_input.get("feature_name", "")).strip()

        # Accept both new (face_ref) and legacy (face_token) names
        allowed = {
            "face_token",
            "face_ref",
            "thread_type",
            "thread_size",
            "thread_length",
            "thread_offset",
            "is_full_length",
            "diameter_unit",
            "feature_name",
            "description",
        }
        extra = set(tool_input.keys()) - allowed
        if extra:
            raise SelectionToolCallError(f"create_external_thread received unexpected parameter(s): {sorted(extra)}")

    else:  # pragma: no cover - guarded upstream
        raise SelectionToolCallError(f"Unsupported feature tool '{tool_name}'.")

    # Mirror resolved/validated values into the nested parameters dict so the
    # Fusion add-in consumes the canonical values (entity tokens, numeric types).
    _sync_payload_parameters(payload)

    await _send_message_safe(manager, session_id, payload)

    try:
        result = await _wait_for_matching_tool_result(
            session_id,
            manager,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            timeout=EXECUTION_TIMEOUT,
            wait_context="feature_operation",
        )
    except asyncio.TimeoutError as exc:
        raise SelectionToolCallError(
            f"Timed out waiting for Fusion to finish '{tool_name}' (tool_use_id={tool_use_id})."
        ) from exc

    success = bool(result.get("success"))
    message_text = result.get("message") or f"{tool_name} completed."

    if not success and result.get("error"):
        message_text = f"{message_text}\nError: {result.get('error')}"

    if tool_name == "apply_fillet" and success:
        edge_count = result.get("edge_count", 0)
        radius = result.get("radius", 0)
        radius_unit = result.get("radius_unit", "mm")
        missing = result.get("missing_tokens", [])
        lines = [
            message_text,
            f"edge_count={edge_count}",
            f"radius={radius}",
            f"radius_unit={radius_unit}",
        ]
        if missing:
            lines.append(f"missing_tokens={missing}")
        message_text = "\n".join(lines)

    elif tool_name == "apply_chamfer" and success:
        edge_count = result.get("edge_count", 0)
        distance = result.get("distance", 0)
        distance_unit = result.get("distance_unit", "mm")
        missing = result.get("missing_tokens", [])
        lines = [
            message_text,
            f"edge_count={edge_count}",
            f"distance={distance}",
            f"distance_unit={distance_unit}",
        ]
        if missing:
            lines.append(f"missing_tokens={missing}")
        message_text = "\n".join(lines)

    elif tool_name == "create_shell" and success:
        entity_count = result.get("entity_count", 0)
        entity_type = result.get("entity_type", "unknown")
        inside = result.get("inside_thickness", 0)
        outside = result.get("outside_thickness", 0)
        thickness_unit = result.get("thickness_unit", "mm")
        shell_type = result.get("shell_type")
        lines = [
            message_text,
            f"entity_count={entity_count}",
            f"entity_type={entity_type}",
        ]
        if inside:
            lines.append(f"inside_thickness={inside}{thickness_unit}")
        if outside:
            lines.append(f"outside_thickness={outside}{thickness_unit}")
        if shell_type:
            lines.append(f"shell_type={shell_type}")
        message_text = "\n".join(lines)

    elif tool_name == "create_simple_hole" and success:
        diameter = result.get("diameter", 0)
        diameter_unit = result.get("diameter_unit", "mm")
        extent_type = result.get("extent_type", "unknown")
        lines = [
            message_text,
            f"diameter={diameter}",
            f"diameter_unit={diameter_unit}",
            f"extent_type={extent_type}",
        ]
        if result.get("depth") is not None:
            lines.append(f"depth={result.get('depth')}")
        message_text = "\n".join(lines)

    elif tool_name == "create_counterbore_hole" and success:
        hole_diameter = result.get("hole_diameter", 0)
        counterbore_diameter = result.get("counterbore_diameter", 0)
        hole_depth = result.get("hole_depth", 0)
        counterbore_depth = result.get("counterbore_depth", 0)
        diameter_unit = result.get("diameter_unit", "mm")
        lines = [
            message_text,
            f"hole_diameter={hole_diameter}",
            f"counterbore_diameter={counterbore_diameter}",
            f"hole_depth={hole_depth}",
            f"counterbore_depth={counterbore_depth}",
            f"diameter_unit={diameter_unit}",
        ]
        message_text = "\n".join(lines)

    elif tool_name == "create_tapped_hole" and success:
        thread_designation = result.get("thread_designation", "")
        tap_drill = result.get("tap_drill_diameter", 0)
        thread_depth = result.get("thread_depth", 0)
        pilot_hole_depth = result.get("pilot_hole_depth", 0)
        diameter_unit = result.get("diameter_unit", "mm")
        lines = [
            message_text,
            f"thread={thread_designation}",
            f"tap_drill={tap_drill}{diameter_unit}",
            f"thread_depth={thread_depth}mm",
            f"pilot_hole_depth={pilot_hole_depth}mm",
        ]
        message_text = "\n".join(lines)

    elif tool_name == "create_external_thread" and success:
        thread_designation = result.get("thread_designation", "")
        nominal_diameter = result.get("nominal_diameter", 0)
        thread_length = result.get("thread_length")
        thread_offset = result.get("thread_offset")
        is_full_length = result.get("is_full_length", False)
        diameter_unit = result.get("diameter_unit", "mm")
        lines = [
            message_text,
            f"thread={thread_designation}",
            f"nominal_diameter={nominal_diameter}{diameter_unit}",
        ]
        if is_full_length:
            lines.append("thread_length=full")
        elif thread_length is not None:
            lines.append(f"thread_length={thread_length}cm")
        if thread_offset and thread_offset > 0:
            lines.append(f"thread_offset={thread_offset}cm")
        message_text = "\n".join(lines)

    elif tool_name == "create_pattern_feature":
        lines = [message_text]
        if success:
            pattern_token = result.get("pattern_token")
            if pattern_token:
                lines.append(f"pattern_token={pattern_token}")
            pattern_type = result.get("pattern_type") or (pattern_prep.parameters.get("pattern_type") if pattern_prep else None)
            if pattern_type:
                lines.append(f"pattern_type={pattern_type}")
            instance_count = result.get("instance_count")
            if instance_count is not None:
                lines.append(f"instance_count={instance_count}")
            if result.get("count_x"):
                lines.append(f"count_x={result.get('count_x')}")
            if result.get("count_y"):
                lines.append(f"count_y={result.get('count_y')}")
            if result.get("rotation_count"):
                lines.append(f"rotation_count={result.get('rotation_count')}")
            if result.get("rotation_angle_deg"):
                lines.append(f"rotation_angle_deg={result.get('rotation_angle_deg')}")
            axis_descriptor = result.get("axis_descriptor")
            if axis_descriptor:
                lines.append(f"axis_descriptor={axis_descriptor}")
        if pattern_prep:
            diag_text = "; ".join(pattern_prep.diagnostics)
            lines.append(f"prep_diagnostics={diag_text}")
        message_text = "\n".join(lines)

    elif tool_name == "adjust_feature_parameters":
        lines = [message_text]
        if success:
            feature_type = result.get("feature_type")
            feature_name = result.get("feature_name")
            timeline_index = result.get("timeline_index")
            changed = result.get("changed_parameters")
            if feature_type:
                lines.append(f"feature_type={feature_type}")
            if feature_name:
                lines.append(f"feature_name={feature_name}")
            if timeline_index is not None:
                lines.append(f"timeline_index={timeline_index}")
            if changed:
                lines.append(f"changed_parameters={changed}")
        message_text = "\n".join(lines)

    return success, message_text, result


def _format_coords(coords: Optional[Sequence[Any]]) -> str:
    if not coords:
        return "unknown"
    try:
        x, y, z = _safe_vec3_extract(coords)  # type: ignore[arg-type]
        return f"({x}, {y}, {z})"
    except Exception:
        return str(coords)


def _format_bounds(bbox: Optional[Mapping[str, Any]], length_units: str = "") -> str:
    """
    Format bounding box as compact axis ranges for LLM spatial awareness.

    Returns format like: bounds=X[-60.0,60.0] Y[-40.0,40.0] Z[8.0,8.0] mm
    This gives the LLM explicit spatial constraints for coordinate validation.
    """
    if not bbox:
        return "bounds=unknown"

    min_pt = bbox.get("min")
    max_pt = bbox.get("max")

    if not min_pt or not max_pt:
        return "bounds=unknown"

    min_x, min_y, min_z = _safe_vec3_extract(min_pt)
    max_x, max_y, max_z = _safe_vec3_extract(max_pt)

    x_range = f"X[{min_x:.1f},{max_x:.1f}]"
    y_range = f"Y[{min_y:.1f},{max_y:.1f}]"
    z_range = f"Z[{min_z:.1f},{max_z:.1f}]"

    unit_suffix = f" {length_units}" if length_units else ""
    return f"bounds={x_range} {y_range} {z_range}{unit_suffix}"


async def _wait_for_plan_decision(session_id: str, manager: ConnectionManager) -> Mapping[str, Any]:
    """Wait for a plan approval/cancellation message from the frontend."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + PLAN_APPROVAL_TIMEOUT

    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise asyncio.TimeoutError

        try:
            message = await manager.wait_for_fusion_result(session_id, timeout=max(1, int(remaining)))
        except asyncio.TimeoutError:
            raise

        if message.get("type") == "plan_approval":
            return message

        logger.debug(
            "Session %s received message of type '%s' while waiting for plan approval; ignoring.",
            session_id,
            message.get("type"),
        )


async def _send_message_safe(manager: ConnectionManager, session_id: str, payload: Dict[str, Any]) -> None:
    """Send a message to the frontend and log failures without crashing the loop."""
    try:
        await manager.send_message(session_id, payload)
    except Exception:  # pragma: no cover - network failures are logged and re-raised
        logger.exception("Failed to send message to session %s: %s", session_id, payload.get("type"))
        raise


async def _send_error(manager: ConnectionManager, session_id: str, message: str, details: str) -> None:
    """Helper to emit an error payload to the frontend with session ID for debugging."""
    # Prepend session ID to details for debugging and support purposes
    details_with_session = f"[Session: {session_id}] {details}"
    payload = {"type": "error", "message": message, "details": details_with_session}
    await _send_message_safe(manager, session_id, payload)
