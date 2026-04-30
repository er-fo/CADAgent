"""Map planner/runtime tool calls into shared backend IR operations."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

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


def _to_optional_int_list(value: Any, *, field_name: str) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        raise UnsupportedToolMappingError(f"Invalid integer list for '{field_name}': {value!r}")
    if isinstance(value, bool):
        raise UnsupportedToolMappingError(f"Invalid integer list for '{field_name}': {value!r}")
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise UnsupportedToolMappingError(f"Invalid integer list for '{field_name}': {value!r}")

    parsed: list[int] = []
    for idx, item in enumerate(value):
        parsed_item = _to_int(item, field_name=f"{field_name}[{idx}]")
        if parsed_item < 0:
            raise UnsupportedToolMappingError(
                f"Invalid integer value for '{field_name}[{idx}]': {item!r} (must be >= 0)"
            )
        parsed.append(parsed_item)

    if not parsed:
        raise UnsupportedToolMappingError(f"'{field_name}' cannot be empty when provided.")

    return parsed


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


def _extract_sketch_from_profile_reference(profile_ref: str) -> str:
    """Best-effort extraction of sketch identifier from '<sketch>:profile_<n>' refs."""
    if ":" not in profile_ref:
        return ""
    sketch_id = profile_ref.split(":", 1)[0].strip()
    return sketch_id


def _lookup_operations(
    doc_state: IRDocumentState,
    dependency_operations: Optional[Sequence[IROperation]],
) -> Sequence[IROperation]:
    """Return operation sequence used for dependency inference."""
    if dependency_operations is not None:
        return dependency_operations
    return doc_state.operations


def _operation_sketch_id(operation: IROperation) -> str:
    """Return sketch id referenced/created by an operation, if any."""
    params = operation.params
    if operation.type == "create_sketch" and isinstance(params, CreateSketchParams):
        return params.sketch
    if operation.type == "add_rectangle" and isinstance(params, AddRectangleParams):
        return params.sketch
    if operation.type == "add_circle" and isinstance(params, AddCircleParams):
        return params.sketch
    if operation.type == "extrude" and isinstance(params, ExtrudeParams):
        if params.sketch:
            return params.sketch
        return _extract_sketch_from_profile_reference(params.profile)
    return ""


def _find_latest_operation_id(
    operations: Sequence[IROperation],
    *,
    sketch_id: str,
    op_types: set[str],
) -> Optional[str]:
    """Find the latest operation id matching sketch + operation type."""
    if not sketch_id:
        return None
    for op in reversed(operations):
        if op.type not in op_types:
            continue
        if _operation_sketch_id(op) == sketch_id:
            return op.id
    return None


def _dedupe_dependencies(*dependency_ids: Optional[str]) -> list[str]:
    """Return ordered, unique dependency ids."""
    deps: list[str] = []
    for dep_id in dependency_ids:
        if not dep_id:
            continue
        if dep_id in deps:
            continue
        deps.append(dep_id)
    return deps


def map_tool_call_to_ir(
    tool_call: Mapping[str, Any],
    doc_state: IRDocumentState,
    *,
    metadata: Optional[IRMetadata] = None,
    dependency_operations: Optional[Sequence[IROperation]] = None,
) -> IROperation:
    """Convert a single tool call into a shared IR operation."""
    name = str(tool_call.get("name") or "").strip()
    params = _extract_tool_input(tool_call)
    operation_id = doc_state.next_operation_id(prefix="op")
    ops_for_dependency = _lookup_operations(doc_state, dependency_operations)

    if name == "create_sketch":
        sketch_id = str(params.get("sketch_id") or f"sketch_{len(doc_state.operations)}").strip()
        plane_raw = _normalize_create_sketch_plane(params.get("plane") or params.get("plane_id"))
        return IROperation(
            id=operation_id,
            type="create_sketch",
            params=CreateSketchParams(plane=plane_raw, sketch=sketch_id),
            dependencies=[],
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
            dependencies=_dedupe_dependencies(
                _find_latest_operation_id(
                    ops_for_dependency,
                    sketch_id=sketch_id,
                    op_types={"create_sketch"},
                )
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
            dependencies=_dedupe_dependencies(
                _find_latest_operation_id(
                    ops_for_dependency,
                    sketch_id=sketch_id,
                    op_types={"create_sketch"},
                )
            ),
            metadata=metadata,
        )

    if name in {"extrude_profile", "extrude"}:
        # Keep profile identity explicit while preserving Fusion-compatible sketch/profile metadata.
        sketch_id = str(params.get("sketch_id") or "").strip()
        profile_indices = _to_optional_int_list(params.get("profile_indices"), field_name="profile_indices")
        has_profile_index = params.get("profile_index") is not None and not (
            isinstance(params.get("profile_index"), str) and not str(params.get("profile_index")).strip()
        )
        if has_profile_index and profile_indices is not None:
            raise UnsupportedToolMappingError('Provide either "profile_index" or "profile_indices", not both.')

        if profile_indices is not None:
            profile_index = None
            default_profile_ref = f"{sketch_id}:profile_{profile_indices[0]}" if sketch_id else ""
        else:
            profile_index = _to_int(params.get("profile_index"), field_name="profile_index", default=0)
            default_profile_ref = f"{sketch_id}:profile_{profile_index}" if sketch_id else ""

        profile_ref = str(params.get("profile") or default_profile_ref).strip()
        inferred_sketch_id = sketch_id or _extract_sketch_from_profile_reference(profile_ref)
        raw_distance = _to_float(params.get("distance"), field_name="distance")
        direction = "positive" if raw_distance >= 0 else "negative"
        distance = abs(raw_distance)
        operation = _normalize_operation(params.get("operation"))
        profile_dependency = _find_latest_operation_id(
            ops_for_dependency,
            sketch_id=inferred_sketch_id,
            op_types={"add_rectangle", "add_circle"},
        )
        sketch_dependency = _find_latest_operation_id(
            ops_for_dependency,
            sketch_id=inferred_sketch_id,
            op_types={"create_sketch"},
        )

        return IROperation(
            id=operation_id,
            type="extrude",
            params=ExtrudeParams(
                profile=profile_ref,
                distance=distance,
                direction=direction,
                operation=operation,  # type: ignore[arg-type]
                sketch=inferred_sketch_id or None,
                profile_index=profile_index,
                profile_indices=profile_indices,
            ),
            dependencies=_dedupe_dependencies(profile_dependency, sketch_dependency),
            metadata=metadata,
        )

    raise UnsupportedToolMappingError(f"Tool '{name}' is not mapped into shared IR.")
