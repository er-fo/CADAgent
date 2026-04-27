"""Map planner/runtime tool calls into shared backend IR operations."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from .document import IRDocumentState
from .types import (
    AddCircleParams,
    AddRectangleParams,
    CreateSketchParams,
    ExtrudeParams,
    IRMetadata,
    IROperation,
)


class UnsupportedToolMappingError(ValueError):
    """Raised when a planner tool call cannot be mapped to shared IR."""


_DATUM_PLANES = {"XY", "XZ", "YZ"}


def _to_float(value: Any, *, field_name: str) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise UnsupportedToolMappingError(f"Missing numeric value for '{field_name}'.")
    if isinstance(value, bool):
        raise UnsupportedToolMappingError(f"Invalid numeric value for '{field_name}': {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise UnsupportedToolMappingError(f"Invalid numeric value for '{field_name}': {value!r}") from None
    if not math.isfinite(parsed):
        raise UnsupportedToolMappingError(f"Invalid numeric value for '{field_name}': {value!r}")
    return parsed


def _to_int(value: Any, *, field_name: str, default: Optional[int] = None) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        if default is not None:
            return default
        raise UnsupportedToolMappingError(f"Missing integer value for '{field_name}'.")
    if isinstance(value, bool):
        raise UnsupportedToolMappingError(f"Invalid integer value for '{field_name}': {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise UnsupportedToolMappingError(f"Invalid integer value for '{field_name}': {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise UnsupportedToolMappingError(f"Invalid integer value for '{field_name}': {value!r}") from None


def _normalize_operation(value: Any) -> str:
    op = str(value or "new").strip().lower()
    alias = {
        "newbody": "new",
        "new_body": "new",
        "join": "join",
        "cut": "cut",
        "intersect": "intersect",
    }
    return alias.get(op, op)


def _extract_tool_input(tool_call: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("input", "arguments", "parameters"):
        value = tool_call.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _normalize_create_sketch_plane(raw_value: Any) -> str:
    raw_plane = str(raw_value or "XY").strip()
    if not raw_plane:
        return "XY"

    datum_plane = raw_plane.upper()
    if datum_plane in _DATUM_PLANES:
        return datum_plane

    # Preserve face refs and custom construction plane IDs for downstream validation.
    return raw_plane


def map_tool_call_to_ir(
    tool_call: Mapping[str, Any],
    doc_state: IRDocumentState,
    *,
    metadata: Optional[IRMetadata] = None,
) -> IROperation:
    """Convert a single tool call into a shared IR operation."""
    name = str(tool_call.get("name") or "").strip()
    params = _extract_tool_input(tool_call)
    operation_id = doc_state.next_operation_id(prefix="op")

    if name == "create_sketch":
        sketch_id = str(params.get("sketch_id") or f"sketch_{len(doc_state.operations)}").strip()
        plane_raw = _normalize_create_sketch_plane(params.get("plane") or params.get("plane_id"))
        return IROperation(
            id=operation_id,
            type="create_sketch",
            params=CreateSketchParams(plane=plane_raw, sketch=sketch_id),
            metadata=metadata,
        )

    if name == "add_rectangle":
        sketch_id = str(params.get("sketch_id") or "").strip()
        c1u = _to_float(params.get("corner1_u"), field_name="corner1_u")
        c1v = _to_float(params.get("corner1_v"), field_name="corner1_v")
        c2u = _to_float(params.get("corner2_u"), field_name="corner2_u")
        c2v = _to_float(params.get("corner2_v"), field_name="corner2_v")
        width = abs(c2u - c1u)
        height = abs(c2v - c1v)
        center = [(c1u + c2u) / 2.0, (c1v + c2v) / 2.0]
        return IROperation(
            id=operation_id,
            type="add_rectangle",
            params=AddRectangleParams(
                sketch=sketch_id,
                center=center,
                width=width,
                height=height,
            ),
            metadata=metadata,
        )

    if name == "add_circle":
        sketch_id = str(params.get("sketch_id") or "").strip()
        center = [
            _to_float(params.get("center_u"), field_name="center_u"),
            _to_float(params.get("center_v"), field_name="center_v"),
        ]
        radius = _to_float(params.get("radius"), field_name="radius")
        return IROperation(
            id=operation_id,
            type="add_circle",
            params=AddCircleParams(sketch=sketch_id, center=center, radius=radius),
            metadata=metadata,
        )

    if name in {"extrude_profile", "extrude"}:
        # Keep profile identity explicit while preserving Fusion-compatible sketch/profile metadata.
        sketch_id = str(params.get("sketch_id") or "").strip()
        profile_index = _to_int(params.get("profile_index"), field_name="profile_index", default=0)
        profile_ref = str(params.get("profile") or f"{sketch_id}:profile_{profile_index}").strip()
        raw_distance = _to_float(params.get("distance"), field_name="distance")
        direction = "positive" if raw_distance >= 0 else "negative"
        distance = abs(raw_distance)
        operation = _normalize_operation(params.get("operation"))

        return IROperation(
            id=operation_id,
            type="extrude",
            params=ExtrudeParams(
                profile=profile_ref,
                distance=distance,
                direction=direction,
                operation=operation,  # type: ignore[arg-type]
                sketch=sketch_id or None,
                profile_index=profile_index,
            ),
            metadata=metadata,
        )

    raise UnsupportedToolMappingError(f"Tool '{name}' is not mapped into shared IR.")
