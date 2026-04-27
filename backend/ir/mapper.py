"""Map planner/runtime tool calls into shared backend IR operations."""

from __future__ import annotations

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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def _normalize_create_sketch_plane(raw_value: Any, *, metadata: Optional[IRMetadata]) -> str:
    raw_plane = str(raw_value or "XY").strip()
    if not raw_plane:
        return "XY"

    datum_plane = raw_plane.upper()
    if datum_plane in _DATUM_PLANES:
        return datum_plane

    source = str((metadata or {}).get("source") or "").strip().lower()
    if source == "studio":
        # build123d currently supports only datum planes.
        return "XY"

    # Fusion path: preserve face refs and custom construction plane IDs.
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
        plane_raw = _normalize_create_sketch_plane(
            params.get("plane") or params.get("plane_id"),
            metadata=metadata,
        )
        return IROperation(
            id=operation_id,
            type="create_sketch",
            params=CreateSketchParams(plane=plane_raw, sketch=sketch_id),
            metadata=metadata,
        )

    if name == "add_rectangle":
        sketch_id = str(params.get("sketch_id") or "").strip()
        c1u = _to_float(params.get("corner1_u"))
        c1v = _to_float(params.get("corner1_v"))
        c2u = _to_float(params.get("corner2_u"))
        c2v = _to_float(params.get("corner2_v"))
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
        center = [_to_float(params.get("center_u")), _to_float(params.get("center_v"))]
        radius = _to_float(params.get("radius"))
        return IROperation(
            id=operation_id,
            type="add_circle",
            params=AddCircleParams(sketch=sketch_id, center=center, radius=radius),
            metadata=metadata,
        )

    if name in {"extrude_profile", "extrude"}:
        # Keep profile identity explicit while preserving Fusion-compatible sketch/profile metadata.
        sketch_id = str(params.get("sketch_id") or "").strip()
        profile_index = _to_int(params.get("profile_index"), default=0)
        profile_ref = str(params.get("profile") or f"{sketch_id}:profile_{profile_index}").strip()
        raw_distance = _to_float(params.get("distance"))
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
