"""Map planner/runtime tool calls into shared backend IR operations."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

from .document import IRDocumentState
from .types import (
    AddArcParams,
    AddCircleParams,
    AddLineParams,
    AddRectangleParams,
    ChamferParams,
    ClearSelectionParams,
    CounterboreHoleParams,
    CreateConstructionPlaneParams,
    CreateSketchParams,
    DeleteFeatureParams,
    ExtrudeParams,
    ExternalThreadParams,
    FilletParams,
    IROperationEffects,
    IRMetadata,
    IROperation,
    JumpToTimelinePositionParams,
    ListFeaturesParams,
    ListSketchProfilesParams,
    LoftParams,
    PatternFeatureParams,
    RevolveParams,
    SelectEntitiesParams,
    ShellParams,
    SimpleHoleParams,
    TappedHoleParams,
)


class UnsupportedToolMappingError(ValueError):
    """Raised when a planner tool call cannot be mapped to shared IR."""


_DATUM_PLANES = {"XY", "XZ", "YZ"}
_LENGTH_TO_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "ft": 304.8,
}

_FEATURE_MUTATION_CAPABILITIES = {
    "fillet": ["b_rep_kernel", "parametric_timeline", "fillet_feature"],
    "chamfer": ["b_rep_kernel", "parametric_timeline", "chamfer_feature"],
    "shell": ["b_rep_kernel", "parametric_timeline", "shell_feature"],
    "create_simple_hole": ["b_rep_kernel", "parametric_timeline", "hole_feature"],
    "create_counterbore_hole": ["b_rep_kernel", "parametric_timeline", "hole_feature"],
    "create_tapped_hole": ["b_rep_kernel", "parametric_timeline", "hole_feature", "thread_catalog"],
    "create_external_thread": ["b_rep_kernel", "parametric_timeline", "thread_catalog"],
    "pattern_feature": ["b_rep_kernel", "parametric_timeline", "feature_pattern"],
}


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


def _length_to_mm(value: Any, *, field_name: str, unit: str = "mm") -> float:
    parsed = _to_float(value, field_name=field_name)
    unit_key = str(unit or "mm").strip().lower()
    factor = _LENGTH_TO_MM.get(unit_key)
    if factor is None:
        allowed = ", ".join(sorted(_LENGTH_TO_MM))
        raise UnsupportedToolMappingError(
            f"Invalid length unit for '{field_name}': {unit!r}. Expected one of: {allowed}"
        )
    return parsed * factor


def _optional_length_to_mm(value: Any, *, field_name: str, unit: str = "mm") -> Optional[float]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _length_to_mm(value, field_name=field_name, unit=unit)


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


def _to_string_list(value: Any, *, field_name: str, allow_empty: bool = False) -> list[str]:
    if value is None:
        if allow_empty:
            return []
        raise UnsupportedToolMappingError(f"Missing string list for '{field_name}'.")
    if isinstance(value, str):
        if value.strip():
            return [value.strip()]
        if allow_empty:
            return []
        raise UnsupportedToolMappingError(f"'{field_name}' cannot be empty.")
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise UnsupportedToolMappingError(f"Invalid string list for '{field_name}': {value!r}")
    parsed = [str(item).strip() for item in value if str(item).strip()]
    if not parsed and not allow_empty:
        raise UnsupportedToolMappingError(f"'{field_name}' cannot be empty.")
    return parsed


def _optional_int(value: Any, *, field_name: str) -> Optional[int]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _to_int(value, field_name=field_name)


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


def _normalize_fusion_feature_operation(value: Any) -> str:
    op = _normalize_operation(value)
    alias = {
        "newbody": "new",
        "new_body": "new",
        "new": "new",
        "join": "join",
        "cut": "cut",
        "intersect": "intersect",
    }
    return alias.get(op, op)


def _normalize_loft_operation(value: Any) -> str:
    op = _normalize_fusion_feature_operation(value)
    if op == "intersect":
        raise UnsupportedToolMappingError("Loft IR does not support intersect operation.")
    return op


def _normalize_thread_type(value: Any, *, field_name: str) -> str:
    thread_type = str(value or "").strip().lower()
    if thread_type == "iso":
        thread_type = "metric"
    if thread_type not in {"metric", "unc", "unf"}:
        raise UnsupportedToolMappingError(
            f"Invalid thread type for '{field_name}': {value!r}. Expected metric/unc/unf."
        )
    return thread_type


def _to_bool(value: Any, *, field_name: str, default: Optional[bool] = None) -> bool:
    if value is None:
        if default is not None:
            return default
        raise UnsupportedToolMappingError(f"Missing boolean value for '{field_name}'.")
    if isinstance(value, bool):
        return value
    raise UnsupportedToolMappingError(f"Invalid boolean value for '{field_name}': {value!r}")


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
    if operation.type == "add_line" and isinstance(params, AddLineParams):
        return params.sketch
    if operation.type == "add_arc" and isinstance(params, AddArcParams):
        return params.sketch
    if operation.type == "list_sketch_profiles" and isinstance(params, ListSketchProfilesParams):
        return params.sketch
    if operation.type == "extrude" and isinstance(params, ExtrudeParams):
        if params.sketch:
            return params.sketch
        return _extract_sketch_from_profile_reference(params.profile)
    if operation.type == "revolve" and isinstance(params, RevolveParams):
        return params.sketch
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


def _effects(
    *,
    creates: Optional[dict[str, list[str]]] = None,
    modifies: Optional[dict[str, list[str]]] = None,
    invalidates: Optional[dict[str, list[str]]] = None,
) -> IROperationEffects:
    return IROperationEffects(
        creates=creates or {},
        modifies=modifies or {},
        invalidates=invalidates or {},
    )


def _feature_effects(operation_id: str, feature_kind: str) -> IROperationEffects:
    return _effects(
        creates={
            "features": [f"{operation_id}:{feature_kind}"],
            "bodies": [f"{operation_id}:created_bodies"],
            "faces": [f"{operation_id}:created_faces"],
            "edges": [f"{operation_id}:created_edges"],
        },
        invalidates={"faces": ["*"], "edges": ["*"], "bodies": ["topology_generation"]},
    )


def _profile_ref(sketch_id: str, profile_index: int) -> str:
    return f"{sketch_id}:profile_{profile_index}" if sketch_id else ""


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
        sketch_name = str(params.get("sketch_name") or "").strip() or None
        return IROperation(
            id=operation_id,
            type="create_sketch",
            params=CreateSketchParams(plane=plane_raw, sketch=sketch_id, sketch_name=sketch_name),
            dependencies=[],
            metadata=metadata,
            requires=["sketch"],
            effects=_effects(creates={"sketches": [sketch_id]}),
        )

    if name == "create_construction_plane":
        plane_id = str(params.get("plane_id") or "").strip()
        mode = str(params.get("mode") or "").strip().lower()
        if mode not in {"datum", "offset_from_datum", "angle_to_edge", "face_normal"}:
            raise UnsupportedToolMappingError(
                "create_construction_plane mode must be datum/offset_from_datum/angle_to_edge/face_normal."
            )
        base_datum = str(params.get("base_datum_plane") or params.get("datum_plane") or params.get("reference_plane") or "").strip().upper() or None
        datum_axis = str(params.get("datum_axis_plane") or "").strip().upper() or None
        offset = _optional_length_to_mm(
            params.get("offset_cm", params.get("offset", params.get("offset_distance"))),
            field_name="offset_cm",
            unit="cm",
        )
        point_values = None
        if any(key in params for key in ("point_x", "point_y", "point_z")):
            point_values = [
                _length_to_mm(params.get("point_x"), field_name="point_x", unit="cm"),
                _length_to_mm(params.get("point_y"), field_name="point_y", unit="cm"),
                _length_to_mm(params.get("point_z"), field_name="point_z", unit="cm"),
            ]
        return IROperation(
            id=operation_id,
            type="create_construction_plane",
            params=CreateConstructionPlaneParams(
                plane=plane_id,
                mode=mode,  # type: ignore[arg-type]
                description=str(params.get("description") or ""),
                datum_axis_plane=datum_axis,
                base_datum_plane=base_datum,
                offset=offset,
                reference_face=str(params.get("reference_face_token") or params.get("reference_face_ref") or "").strip() or None,
                reference_edge=str(params.get("reference_edge_token") or params.get("reference_edge_ref") or "").strip() or None,
                angle_degrees=_to_float(params.get("angle_deg"), field_name="angle_deg")
                if params.get("angle_deg") is not None
                else None,
                face=str(params.get("face_token") or params.get("face_ref") or "").strip() or None,
                point=point_values,
            ),
            dependencies=[],
            metadata=metadata,
            requires=["construction_plane"],
            effects=_effects(creates={"planes": [plane_id]}),
        )

    if name == "add_rectangle":
        sketch_id = str(params.get("sketch_id") or "").strip()
        # Fusion sketch tools use centimeters; IR stores canonical millimeters.
        c1u = _length_to_mm(params.get("corner1_u"), field_name="corner1_u", unit="cm")
        c1v = _length_to_mm(params.get("corner1_v"), field_name="corner1_v", unit="cm")
        c2u = _length_to_mm(params.get("corner2_u"), field_name="corner2_u", unit="cm")
        c2v = _length_to_mm(params.get("corner2_v"), field_name="corner2_v", unit="cm")
        width = abs(c2u - c1u)
        height = abs(c2v - c1v)
        center = [(c1u + c2u) / 2.0, (c1v + c2v) / 2.0]
        rectangle_id = str(params.get("rectangle_id") or "").strip() or f"{operation_id}_rectangle"
        return IROperation(
            id=operation_id,
            type="add_rectangle",
            params=AddRectangleParams(
                sketch=sketch_id,
                center=center,
                width=width,
                height=height,
                corner1=[c1u, c1v],
                corner2=[c2u, c2v],
                rectangle_id=rectangle_id,
            ),
            dependencies=_dedupe_dependencies(
                _find_latest_operation_id(
                    ops_for_dependency,
                    sketch_id=sketch_id,
                    op_types={"create_sketch"},
                )
            ),
            metadata=metadata,
            requires=["sketch"],
            effects=_effects(
                creates={
                    "sketch_curves": [rectangle_id or f"{operation_id}:rectangle"],
                    "profiles": [_profile_ref(sketch_id, 0)],
                },
                modifies={"sketches": [sketch_id]},
            ),
        )

    if name == "add_circle":
        sketch_id = str(params.get("sketch_id") or "").strip()
        center = [
            _length_to_mm(params.get("center_u"), field_name="center_u", unit="cm"),
            _length_to_mm(params.get("center_v"), field_name="center_v", unit="cm"),
        ]
        radius = _length_to_mm(params.get("radius"), field_name="radius", unit="cm")
        circle_id = str(params.get("circle_id") or "").strip() or f"{operation_id}_circle"
        return IROperation(
            id=operation_id,
            type="add_circle",
            params=AddCircleParams(sketch=sketch_id, center=center, radius=radius, circle_id=circle_id),
            dependencies=_dedupe_dependencies(
                _find_latest_operation_id(
                    ops_for_dependency,
                    sketch_id=sketch_id,
                    op_types={"create_sketch"},
                )
            ),
            metadata=metadata,
            requires=["sketch"],
            effects=_effects(
                creates={
                    "sketch_curves": [circle_id or f"{operation_id}:circle"],
                    "profiles": [_profile_ref(sketch_id, 0)],
                },
                modifies={"sketches": [sketch_id]},
            ),
        )

    if name == "add_line":
        sketch_id = str(params.get("sketch_id") or "").strip()
        line_id = str(params.get("line_id") or "").strip() or f"{operation_id}_line"
        start = [
            _length_to_mm(params.get("start_u"), field_name="start_u", unit="cm"),
            _length_to_mm(params.get("start_v"), field_name="start_v", unit="cm"),
        ]
        end = [
            _length_to_mm(params.get("end_u"), field_name="end_u", unit="cm"),
            _length_to_mm(params.get("end_v"), field_name="end_v", unit="cm"),
        ]
        return IROperation(
            id=operation_id,
            type="add_line",
            params=AddLineParams(sketch=sketch_id, start=start, end=end, line_id=line_id),
            dependencies=_dedupe_dependencies(
                _find_latest_operation_id(
                    ops_for_dependency,
                    sketch_id=sketch_id,
                    op_types={"create_sketch"},
                )
            ),
            metadata=metadata,
            requires=["sketch"],
            effects=_effects(
                creates={"sketch_curves": [line_id or f"{operation_id}:line"]},
                modifies={"sketches": [sketch_id]},
            ),
        )

    if name == "add_arc":
        sketch_id = str(params.get("sketch_id") or "").strip()
        arc_id = str(params.get("arc_id") or "").strip() or f"{operation_id}_arc"
        center = [
            _length_to_mm(params.get("center_u"), field_name="center_u", unit="cm"),
            _length_to_mm(params.get("center_v"), field_name="center_v", unit="cm"),
        ]
        start = [
            _length_to_mm(params.get("start_u"), field_name="start_u", unit="cm"),
            _length_to_mm(params.get("start_v"), field_name="start_v", unit="cm"),
        ]
        end = [
            _length_to_mm(params.get("end_u"), field_name="end_u", unit="cm"),
            _length_to_mm(params.get("end_v"), field_name="end_v", unit="cm"),
        ]
        return IROperation(
            id=operation_id,
            type="add_arc",
            params=AddArcParams(sketch=sketch_id, center=center, start=start, end=end, arc_id=arc_id),
            dependencies=_dedupe_dependencies(
                _find_latest_operation_id(
                    ops_for_dependency,
                    sketch_id=sketch_id,
                    op_types={"create_sketch"},
                )
            ),
            metadata=metadata,
            requires=["sketch"],
            effects=_effects(
                creates={"sketch_curves": [arc_id or f"{operation_id}:arc"]},
                modifies={"sketches": [sketch_id]},
            ),
        )

    if name == "list_sketch_profiles":
        sketch_id = str(params.get("sketch_id") or "").strip()
        return IROperation(
            id=operation_id,
            type="list_sketch_profiles",
            params=ListSketchProfilesParams(sketch=sketch_id),
            dependencies=_dedupe_dependencies(
                _find_latest_operation_id(
                    ops_for_dependency,
                    sketch_id=sketch_id,
                    op_types={"add_rectangle", "add_circle", "add_line", "add_arc"},
                ),
                _find_latest_operation_id(
                    ops_for_dependency,
                    sketch_id=sketch_id,
                    op_types={"create_sketch"},
                ),
            ),
            metadata=metadata,
            requires=["profile_inspection"],
            effects=_effects(creates={"profiles": [f"{sketch_id}:profiles"]}),
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
        # Fusion extrude distance is centimeters; IR stores millimeters.
        raw_distance = _length_to_mm(params.get("distance"), field_name="distance", unit="cm")
        direction = "positive" if raw_distance >= 0 else "negative"
        distance = abs(raw_distance)
        operation = _normalize_fusion_feature_operation(params.get("operation"))
        profile_dependency = _find_latest_operation_id(
            ops_for_dependency,
            sketch_id=inferred_sketch_id,
            op_types={"add_rectangle", "add_circle", "list_sketch_profiles"},
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
                feature_name=str(params.get("feature_name") or "").strip() or None,
                fallback_policy=["flip_extrude_direction_on_no_target"]
                if operation in {"cut", "intersect"}
                else [],
            ),
            dependencies=_dedupe_dependencies(profile_dependency, sketch_dependency),
            metadata=metadata,
            requires=["b_rep_kernel", "parametric_timeline", "extrude_feature"],
            effects=_feature_effects(operation_id, "extrude"),
            fallback_policy=["flip_extrude_direction_on_no_target"]
            if operation in {"cut", "intersect"}
            else [],
        )

    if name == "revolve_profile":
        sketch_id = str(params.get("sketch_id") or "").strip()
        profile_index = _to_int(params.get("profile_index"), field_name="profile_index", default=0)
        profile_ref = str(params.get("profile") or _profile_ref(sketch_id, profile_index)).strip()
        operation = _normalize_fusion_feature_operation(params.get("operation"))
        extent = dict(params.get("extent") or {"mode": "full"})
        axis = dict(params.get("axis") or {})
        profile_dependency = _find_latest_operation_id(
            ops_for_dependency,
            sketch_id=sketch_id,
            op_types={"add_rectangle", "add_circle", "list_sketch_profiles"},
        )
        sketch_dependency = _find_latest_operation_id(
            ops_for_dependency,
            sketch_id=sketch_id,
            op_types={"create_sketch"},
        )
        return IROperation(
            id=operation_id,
            type="revolve",
            params=RevolveParams(
                profile=profile_ref,
                sketch=sketch_id,
                profile_index=profile_index,
                axis=axis,
                extent=extent,
                operation=operation,  # type: ignore[arg-type]
                is_solid=_to_bool(params.get("is_solid"), field_name="is_solid", default=True),
                creation_occurrence=str(params.get("creation_occurrence_token") or params.get("creation_occurrence_ref") or "").strip() or None,
                feature_name=str(params.get("feature_name") or "").strip() or None,
            ),
            dependencies=_dedupe_dependencies(profile_dependency, sketch_dependency),
            metadata=metadata,
            requires=["b_rep_kernel", "parametric_timeline", "revolve_feature"],
            effects=_feature_effects(operation_id, "revolve"),
        )

    if name == "create_loft":
        profile_ids = _to_string_list(params.get("profile_ids"), field_name="profile_ids")
        dependencies: list[str] = []
        for sketch_id in profile_ids:
            dependencies.extend(
                _dedupe_dependencies(
                    _find_latest_operation_id(
                        ops_for_dependency,
                        sketch_id=sketch_id,
                        op_types={"add_rectangle", "add_circle", "list_sketch_profiles"},
                    ),
                    _find_latest_operation_id(
                        ops_for_dependency,
                        sketch_id=sketch_id,
                        op_types={"create_sketch"},
                    ),
                )
            )
        return IROperation(
            id=operation_id,
            type="loft",
            params=LoftParams(
                profile_ids=profile_ids,
                operation=_normalize_loft_operation(params.get("operation")),  # type: ignore[arg-type]
                feature_name=str(params.get("feature_name") or "").strip() or None,
                fallback_policy=["fallback_to_surface_loft"],
            ),
            dependencies=_dedupe_dependencies(*dependencies),
            metadata=metadata,
            requires=["b_rep_kernel", "parametric_timeline", "loft_feature"],
            effects=_feature_effects(operation_id, "loft"),
            fallback_policy=["fallback_to_surface_loft"],
        )

    if name == "apply_fillet":
        unit = str(params.get("radius_unit") or "mm")
        edge_refs = _to_string_list(params.get("edge_refs", params.get("entity_tokens")), field_name="edge_refs")
        return IROperation(
            id=operation_id,
            type="fillet",
            params=FilletParams(
                edge_refs=edge_refs,
                radius=_length_to_mm(params.get("radius"), field_name="radius", unit=unit),
                include_tangent_edges=_to_bool(
                    params.get("include_tangent_edges"),
                    field_name="include_tangent_edges",
                    default=True,
                ),
                feature_name=str(params.get("feature_name") or "").strip() or None,
            ),
            dependencies=[],
            metadata=metadata,
            requires=_FEATURE_MUTATION_CAPABILITIES["fillet"],
            effects=_feature_effects(operation_id, "fillet"),
            selectors=[{"kind": "edge", "refs": edge_refs, "tangent_chain": bool(params.get("include_tangent_edges", True))}],
        )

    if name == "apply_chamfer":
        unit = str(params.get("distance_unit") or "mm")
        edge_refs = _to_string_list(params.get("edge_refs", params.get("entity_tokens")), field_name="edge_refs")
        return IROperation(
            id=operation_id,
            type="chamfer",
            params=ChamferParams(
                edge_refs=edge_refs,
                distance=_length_to_mm(params.get("distance"), field_name="distance", unit=unit),
                include_tangent_edges=_to_bool(
                    params.get("include_tangent_edges"),
                    field_name="include_tangent_edges",
                    default=True,
                ),
                feature_name=str(params.get("feature_name") or "").strip() or None,
            ),
            dependencies=[],
            metadata=metadata,
            requires=_FEATURE_MUTATION_CAPABILITIES["chamfer"],
            effects=_feature_effects(operation_id, "chamfer"),
            selectors=[{"kind": "edge", "refs": edge_refs, "tangent_chain": bool(params.get("include_tangent_edges", True))}],
        )

    if name == "create_shell":
        mode = str(params.get("mode") or "open").strip().lower()
        if mode not in {"open", "closed"}:
            raise UnsupportedToolMappingError("create_shell mode must be open or closed.")
        thickness_unit = str(params.get("thickness_unit") or "mm")
        face_refs = _to_string_list(params.get("face_refs"), field_name="face_refs", allow_empty=True)
        body_refs = _to_string_list(params.get("body_refs"), field_name="body_refs", allow_empty=True)
        legacy_refs = _to_string_list(params.get("entity_tokens"), field_name="entity_tokens", allow_empty=True)
        if mode == "open" and not face_refs:
            face_refs = legacy_refs
        if mode == "closed" and not body_refs:
            body_refs = legacy_refs
        return IROperation(
            id=operation_id,
            type="shell",
            params=ShellParams(
                mode=mode,  # type: ignore[arg-type]
                face_refs=face_refs,
                body_refs=body_refs,
                inside_thickness=_length_to_mm(params.get("inside_thickness", 0), field_name="inside_thickness", unit=thickness_unit),
                outside_thickness=_length_to_mm(params.get("outside_thickness", 0), field_name="outside_thickness", unit=thickness_unit),
                is_tangent_chain=_to_bool(params.get("is_tangent_chain"), field_name="is_tangent_chain", default=True),
                shell_type=str(params.get("shell_type") or "sharp").strip().lower(),  # type: ignore[arg-type]
                feature_name=str(params.get("feature_name") or "").strip() or None,
            ),
            dependencies=[],
            metadata=metadata,
            requires=_FEATURE_MUTATION_CAPABILITIES["shell"],
            effects=_feature_effects(operation_id, "shell"),
            selectors=[{"kind": "face" if mode == "open" else "body", "refs": face_refs if mode == "open" else body_refs}],
        )

    if name == "create_simple_hole":
        diameter_unit = str(params.get("diameter_unit") or "mm")
        extent_type = str(params.get("extent_type") or "").strip().lower()
        return IROperation(
            id=operation_id,
            type="create_simple_hole",
            params=SimpleHoleParams(
                face_ref=str(params.get("face_ref") or params.get("face_token") or "").strip(),
                center=[
                    _length_to_mm(params.get("center_x"), field_name="center_x", unit="mm"),
                    _length_to_mm(params.get("center_y"), field_name="center_y", unit="mm"),
                    _length_to_mm(params.get("center_z"), field_name="center_z", unit="mm"),
                ],
                diameter=_length_to_mm(params.get("diameter"), field_name="diameter", unit=diameter_unit),
                extent_type=extent_type,  # type: ignore[arg-type]
                depth=_optional_length_to_mm(params.get("depth"), field_name="depth", unit="mm"),
                feature_name=str(params.get("feature_name") or "").strip() or None,
            ),
            dependencies=[],
            metadata=metadata,
            requires=_FEATURE_MUTATION_CAPABILITIES["create_simple_hole"],
            effects=_feature_effects(operation_id, "simple_hole"),
        )

    if name == "create_counterbore_hole":
        diameter_unit = str(params.get("diameter_unit") or "mm")
        return IROperation(
            id=operation_id,
            type="create_counterbore_hole",
            params=CounterboreHoleParams(
                face_ref=str(params.get("face_ref") or params.get("face_token") or "").strip(),
                center=[
                    _length_to_mm(params.get("center_x"), field_name="center_x", unit="mm"),
                    _length_to_mm(params.get("center_y"), field_name="center_y", unit="mm"),
                    _length_to_mm(params.get("center_z"), field_name="center_z", unit="mm"),
                ],
                hole_diameter=_length_to_mm(params.get("hole_diameter"), field_name="hole_diameter", unit=diameter_unit),
                hole_depth=_length_to_mm(params.get("hole_depth"), field_name="hole_depth", unit="mm"),
                counterbore_diameter=_length_to_mm(
                    params.get("counterbore_diameter"),
                    field_name="counterbore_diameter",
                    unit=diameter_unit,
                ),
                counterbore_depth=_length_to_mm(params.get("counterbore_depth"), field_name="counterbore_depth", unit="mm"),
                feature_name=str(params.get("feature_name") or "").strip() or None,
            ),
            dependencies=[],
            metadata=metadata,
            requires=_FEATURE_MUTATION_CAPABILITIES["create_counterbore_hole"],
            effects=_feature_effects(operation_id, "counterbore_hole"),
        )

    if name == "create_tapped_hole":
        thread_type = _normalize_thread_type(params.get("thread_type"), field_name="thread_type")
        return IROperation(
            id=operation_id,
            type="create_tapped_hole",
            params=TappedHoleParams(
                face_ref=str(params.get("face_ref") or params.get("face_token") or "").strip(),
                center=[
                    _length_to_mm(params.get("center_x"), field_name="center_x", unit="mm"),
                    _length_to_mm(params.get("center_y"), field_name="center_y", unit="mm"),
                    _length_to_mm(params.get("center_z"), field_name="center_z", unit="mm"),
                ],
                thread_type=thread_type,  # type: ignore[arg-type]
                thread_size=str(params.get("thread_size") or "").strip(),
                thread_depth=_length_to_mm(params.get("thread_depth"), field_name="thread_depth", unit="mm"),
                pilot_hole_depth=_optional_length_to_mm(
                    params.get("pilot_hole_depth"),
                    field_name="pilot_hole_depth",
                    unit="mm",
                ),
                diameter_unit=str(params.get("diameter_unit") or "mm"),
                feature_name=str(params.get("feature_name") or "").strip() or None,
            ),
            dependencies=[],
            metadata=metadata,
            requires=_FEATURE_MUTATION_CAPABILITIES["create_tapped_hole"],
            effects=_feature_effects(operation_id, "tapped_hole"),
        )

    if name == "create_external_thread":
        thread_type = _normalize_thread_type(params.get("thread_type"), field_name="thread_type")
        is_full_length = _to_bool(params.get("is_full_length"), field_name="is_full_length", default=True)
        return IROperation(
            id=operation_id,
            type="create_external_thread",
            params=ExternalThreadParams(
                face_ref=str(params.get("face_ref") or params.get("face_token") or "").strip(),
                thread_type=thread_type,  # type: ignore[arg-type]
                thread_size=str(params.get("thread_size") or "").strip(),
                is_full_length=is_full_length,
                thread_length=_optional_length_to_mm(
                    params.get("thread_length"),
                    field_name="thread_length",
                    unit="cm",
                ),
                thread_offset=_length_to_mm(params.get("thread_offset", 0.0), field_name="thread_offset", unit="cm"),
                diameter_unit=str(params.get("diameter_unit") or "mm"),
                feature_name=str(params.get("feature_name") or "").strip() or None,
            ),
            dependencies=[],
            metadata=metadata,
            requires=_FEATURE_MUTATION_CAPABILITIES["create_external_thread"],
            effects=_feature_effects(operation_id, "external_thread"),
        )

    if name == "create_pattern_feature":
        pattern_type = str(params.get("pattern_type") or "").strip().lower()
        if pattern_type not in {"rectangular", "circular"}:
            raise UnsupportedToolMappingError("create_pattern_feature pattern_type must be rectangular or circular.")
        return IROperation(
            id=operation_id,
            type="pattern_feature",
            params=PatternFeatureParams(
                pattern_type=pattern_type,  # type: ignore[arg-type]
                feature_refs=_to_string_list(params.get("feature_refs", params.get("feature_tokens", ["auto_last"])), field_name="feature_refs"),
                count_x=_optional_int(params.get("count_x"), field_name="count_x"),
                spacing_x=_optional_length_to_mm(params.get("spacing_x_cm"), field_name="spacing_x_cm", unit="cm"),
                count_y=_optional_int(params.get("count_y"), field_name="count_y"),
                spacing_y=_optional_length_to_mm(params.get("spacing_y_cm"), field_name="spacing_y_cm", unit="cm"),
                rotation_count=_optional_int(params.get("rotation_count"), field_name="rotation_count"),
                rotation_angle_degrees=_to_float(params.get("rotation_angle_deg"), field_name="rotation_angle_deg")
                if params.get("rotation_angle_deg") is not None
                else None,
                orientation_hint=params.get("orientation_hint"),
                feature_name=str(params.get("feature_name") or "").strip() or None,
            ),
            dependencies=[],
            metadata=metadata,
            requires=_FEATURE_MUTATION_CAPABILITIES["pattern_feature"],
            effects=_feature_effects(operation_id, "pattern"),
            fallback_policy=["infer_pattern_axis_or_spacing", "circular_patterns_use_global_origin_axis"],
        )

    if name == "list_features":
        return IROperation(
            id=operation_id,
            type="list_features",
            params=ListFeaturesParams(description=str(params.get("description") or "")),
            dependencies=[],
            metadata=metadata,
            requires=["feature_query"],
            effects=_effects(creates={"features": ["feature_snapshot"]}),
        )

    if name == "delete_feature":
        return IROperation(
            id=operation_id,
            type="delete_feature",
            params=DeleteFeatureParams(
                feature_ref=str(params.get("feature_ref") or params.get("feature_token") or "").strip(),
                description=str(params.get("description") or ""),
                expected_name=str(params.get("expected_name") or "").strip() or None,
                expected_timeline_index=_optional_int(
                    params.get("expected_timeline_index"),
                    field_name="expected_timeline_index",
                ),
            ),
            dependencies=[],
            metadata=metadata,
            requires=["parametric_timeline", "feature_lifecycle"],
            effects=_effects(
                modifies={"features": [str(params.get("feature_ref") or params.get("feature_token") or "").strip()]},
                invalidates={"features": ["downstream"], "faces": ["*"], "edges": ["*"], "bodies": ["topology_generation"]},
            ),
        )

    if name == "jump_to_timeline_position":
        return IROperation(
            id=operation_id,
            type="jump_to_timeline_position",
            params=JumpToTimelinePositionParams(
                target_index=_to_int(params.get("target_index"), field_name="target_index"),
                reason=str(params.get("reason") or ""),
                description=str(params.get("description") or ""),
            ),
            dependencies=[],
            metadata=metadata,
            requires=["parametric_timeline", "document_revision"],
            effects=_effects(invalidates={"operations": ["after_marker"], "features": ["after_marker"], "faces": ["*"], "edges": ["*"]}),
        )

    if name in {"select_edges", "select_faces", "select_bodies"}:
        kind = {"select_edges": "edge", "select_faces": "face", "select_bodies": "body"}[name]
        field_name = {"edge": "edge_refs", "face": "face_refs", "body": "body_refs"}[kind]
        legacy_value = params.get(field_name) or params.get("entity_tokens")
        refs = _to_string_list(legacy_value, field_name=field_name)
        return IROperation(
            id=operation_id,
            type="select_entities",
            params=SelectEntitiesParams(
                kind=kind,  # type: ignore[arg-type]
                refs=refs,
                clear_existing=_to_bool(params.get("clear_existing"), field_name="clear_existing", default=True),
            ),
            dependencies=[],
            metadata=metadata,
            requires=["selection"],
            selectors=[{"kind": kind, "refs": refs}],
        )

    if name in {"clear_edge_selection", "clear_face_selection", "clear_body_selection"}:
        kind = {
            "clear_edge_selection": "edge",
            "clear_face_selection": "face",
            "clear_body_selection": "body",
        }[name]
        return IROperation(
            id=operation_id,
            type="clear_selection",
            params=ClearSelectionParams(kind=kind),  # type: ignore[arg-type]
            dependencies=[],
            metadata=metadata,
            requires=["selection"],
            effects=_effects(invalidates={"selection": [kind]}),
        )

    raise UnsupportedToolMappingError(f"Tool '{name}' is not mapped into shared IR.")
