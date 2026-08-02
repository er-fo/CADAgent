"""Translate shared IR operations into Fusion-compatible tool calls."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from ...ir.types import (
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
    FeatureParameterEditParams,
    FeatureSuppressionParams,
    FilletParams,
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


_OPERATION_TO_FUSION = {
    "new": "NewBody",
    "join": "Join",
    "cut": "Cut",
    "intersect": "Intersect",
}
_FACE_ALIAS_PATTERN = re.compile(r"^face_\d+$")
_FEATURE_PARAMETER_LENGTH_KEYS = {
    "distance",
    "diameter",
    "depth",
    "radius",
    "chamfer_distance",
    "inside_thickness",
    "outside_thickness",
    "rectangular_spacing_one",
    "rectangular_spacing_two",
}


def _fusion_feature_parameter_payload(parameters: Dict[str, Any]) -> Dict[str, Any]:
    if set(parameters.keys()) == {"name"}:
        raise ValueError("adjust_feature_parameters requires at least one geometry parameter with feature renames.")

    payload: Dict[str, Any] = {}
    for key, value in parameters.items():
        payload[key] = value
        if key in _FEATURE_PARAMETER_LENGTH_KEYS:
            payload[f"{key}_unit"] = "mm"
        elif key == "circular_total_angle":
            payload["circular_total_angle_unit"] = "deg"
    return payload


def _mm_to_cm(value: float) -> float:
    return value / 10.0


def _optional_mm_to_cm(value: Optional[float]) -> Optional[float]:
    return None if value is None else _mm_to_cm(value)


def _parse_profile_reference(profile: str) -> Tuple[str, int]:
    """
    Parse profile references like `sketch_0:profile_0`.

    Returns:
        (sketch_id, profile_index)
    """
    if ":" not in profile:
        raise ValueError(
            f"Unsupported profile reference format '{profile}'. Expected '<sketch_id>:profile_<index>'."
        )
    sketch_id, profile_part = profile.split(":", 1)
    if not sketch_id:
        raise ValueError(
            f"Unsupported profile reference format '{profile}'. Sketch identifier must not be empty."
        )
    if not profile_part.startswith("profile_"):
        raise ValueError(
            f"Unsupported profile reference format '{profile}'. Expected profile part to start with 'profile_'."
        )

    try:
        profile_index = int(profile_part.split("_", 1)[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported profile reference format '{profile}'. Expected numeric profile index."
        ) from exc

    return sketch_id, profile_index


def translate_ir_to_fusion_tool_call(operation: IROperation) -> Tuple[str, Dict[str, Any]]:
    """Convert one IR operation into a Fusion tool call."""
    if operation.type == "create_construction_plane":
        params = operation.params
        if not isinstance(params, CreateConstructionPlaneParams):
            raise ValueError("create_construction_plane IR params shape mismatch")
        tool_input: Dict[str, Any] = {
            "plane_id": params.plane,
            "mode": params.mode,
            "description": params.description,
        }
        if params.datum_axis_plane:
            tool_input["datum_axis_plane"] = params.datum_axis_plane
        if params.base_datum_plane:
            tool_input["base_datum_plane"] = params.base_datum_plane
        if params.offset is not None:
            tool_input["offset_cm"] = _mm_to_cm(params.offset)
        if params.reference_face:
            tool_input["reference_face_token"] = params.reference_face
        if params.reference_edge:
            tool_input["reference_edge_token"] = params.reference_edge
        if params.angle_degrees is not None:
            tool_input["angle_deg"] = params.angle_degrees
        if params.face:
            tool_input["face_token"] = params.face
        if params.point is not None:
            if len(params.point) != 3:
                raise ValueError("create_construction_plane point must contain exactly 3 values")
            tool_input["point_x"] = _mm_to_cm(params.point[0])
            tool_input["point_y"] = _mm_to_cm(params.point[1])
            tool_input["point_z"] = _mm_to_cm(params.point[2])
        return "create_construction_plane", tool_input

    if operation.type == "create_sketch":
        params = operation.params
        if not isinstance(params, CreateSketchParams):
            raise ValueError("create_sketch IR params shape mismatch")
        plane_id = str(params.plane).strip()
        if _FACE_ALIAS_PATTERN.match(plane_id):
            raise ValueError(
                "Unresolved face alias for create_sketch plane. "
                f"Expected datum plane or face token, got '{plane_id}'."
            )
        return "create_sketch", {
            "plane_id": plane_id,
            "sketch_id": params.sketch,
            "sketch_name": params.sketch_name or "",
            "description": f"Create sketch {params.sketch} on {plane_id}",
        }

    if operation.type == "add_rectangle":
        params = operation.params
        if not isinstance(params, AddRectangleParams):
            raise ValueError("add_rectangle IR params shape mismatch")

        half_w = params.width / 2.0
        half_h = params.height / 2.0
        c_u, c_v = params.center

        return "add_rectangle", {
            "sketch_id": params.sketch,
            "corner1_u": _mm_to_cm(c_u - half_w),
            "corner1_v": _mm_to_cm(c_v - half_h),
            "corner2_u": _mm_to_cm(c_u + half_w),
            "corner2_v": _mm_to_cm(c_v + half_h),
            "rectangle_id": params.rectangle_id or f"{operation.id}_rectangle",
            "description": f"Add rectangle to {params.sketch}",
        }

    if operation.type == "add_circle":
        params = operation.params
        if not isinstance(params, AddCircleParams):
            raise ValueError("add_circle IR params shape mismatch")

        return "add_circle", {
            "sketch_id": params.sketch,
            "center_u": _mm_to_cm(params.center[0]),
            "center_v": _mm_to_cm(params.center[1]),
            "radius": _mm_to_cm(params.radius),
            "circle_id": params.circle_id or f"{operation.id}_circle",
            "description": f"Add circle to {params.sketch}",
        }

    if operation.type == "add_line":
        params = operation.params
        if not isinstance(params, AddLineParams):
            raise ValueError("add_line IR params shape mismatch")
        return "add_line", {
            "sketch_id": params.sketch,
            "start_u": _mm_to_cm(params.start[0]),
            "start_v": _mm_to_cm(params.start[1]),
            "end_u": _mm_to_cm(params.end[0]),
            "end_v": _mm_to_cm(params.end[1]),
            "line_id": params.line_id or f"{operation.id}_line",
            "description": f"Add line to {params.sketch}",
        }

    if operation.type == "add_arc":
        params = operation.params
        if not isinstance(params, AddArcParams):
            raise ValueError("add_arc IR params shape mismatch")
        return "add_arc", {
            "sketch_id": params.sketch,
            "center_u": _mm_to_cm(params.center[0]),
            "center_v": _mm_to_cm(params.center[1]),
            "start_u": _mm_to_cm(params.start[0]),
            "start_v": _mm_to_cm(params.start[1]),
            "end_u": _mm_to_cm(params.end[0]),
            "end_v": _mm_to_cm(params.end[1]),
            "arc_id": params.arc_id or f"{operation.id}_arc",
            "description": f"Add arc to {params.sketch}",
        }

    if operation.type == "list_sketch_profiles":
        params = operation.params
        if not isinstance(params, ListSketchProfilesParams):
            raise ValueError("list_sketch_profiles IR params shape mismatch")
        return "list_sketch_profiles", {
            "sketch_id": params.sketch,
            "description": f"Inspect profiles for {params.sketch}",
        }

    if operation.type == "extrude":
        params = operation.params
        if not isinstance(params, ExtrudeParams):
            raise ValueError("extrude IR params shape mismatch")

        sketch_id = params.sketch
        profile_index = params.profile_index if params.profile_index is not None else 0
        profile_indices = list(params.profile_indices) if params.profile_indices is not None else None

        if profile_indices is not None and params.profile_index is not None:
            raise ValueError('Extrude IR params cannot set both "profile_index" and "profile_indices".')
        if profile_indices is not None and not profile_indices:
            raise ValueError('Extrude IR params "profile_indices" cannot be empty.')

        if not sketch_id:
            parsed_sketch, parsed_index = _parse_profile_reference(params.profile)
            sketch_id = parsed_sketch
            if profile_indices is None:
                profile_index = parsed_index

        signed_distance_mm = params.distance if params.direction == "positive" else -params.distance
        signed_distance = _mm_to_cm(signed_distance_mm)
        fusion_operation = _OPERATION_TO_FUSION.get(params.operation)
        if fusion_operation is None:
            raise ValueError(f"Unsupported Fusion extrude operation: {params.operation}")
        tool_input: Dict[str, Any] = {
            "sketch_id": sketch_id,
            "distance": signed_distance,
            "operation": fusion_operation,
            "feature_name": params.feature_name or "",
            "description": f"Extrude {params.profile}",
        }
        if profile_indices is not None:
            tool_input["profile_indices"] = profile_indices
        else:
            tool_input["profile_index"] = profile_index
        return "extrude_profile", tool_input

    if operation.type == "revolve":
        params = operation.params
        if not isinstance(params, RevolveParams):
            raise ValueError("revolve IR params shape mismatch")
        fusion_operation = _OPERATION_TO_FUSION.get(params.operation)
        if fusion_operation is None:
            raise ValueError(f"Unsupported Fusion revolve operation: {params.operation}")
        tool_input = {
            "sketch_id": params.sketch,
            "profile_index": params.profile_index,
            "axis": dict(params.axis),
            "extent": dict(params.extent),
            "operation": fusion_operation,
            "is_solid": params.is_solid,
            "creation_occurrence_token": params.creation_occurrence or "",
            "feature_name": params.feature_name or "",
            "description": f"Revolve {params.profile}",
        }
        return "revolve_profile", tool_input

    if operation.type == "loft":
        params = operation.params
        if not isinstance(params, LoftParams):
            raise ValueError("loft IR params shape mismatch")
        fusion_operation = _OPERATION_TO_FUSION.get(params.operation)
        if fusion_operation is None or fusion_operation == "Intersect":
            raise ValueError(f"Unsupported Fusion loft operation: {params.operation}")
        return "create_loft", {
            "profile_ids": list(params.profile_ids),
            "operation": fusion_operation,
            "feature_name": params.feature_name or "",
            "description": "Create loft from IR profiles",
        }

    if operation.type == "jump_to_timeline_position":
        params = operation.params
        if not isinstance(params, JumpToTimelinePositionParams):
            raise ValueError("jump_to_timeline_position IR params shape mismatch")
        return "jump_to_timeline_position", {
            "target_index": params.target_index,
            "reason": params.reason,
            "description": params.description or "Jump to timeline position",
        }

    if operation.type == "delete_feature":
        params = operation.params
        if not isinstance(params, DeleteFeatureParams):
            raise ValueError("delete_feature IR params shape mismatch")
        tool_input = {
            "feature_token": params.feature_ref,
            "description": params.description or "Delete feature",
            "expected_name": params.expected_name or "",
        }
        if params.expected_timeline_index is not None:
            tool_input["expected_timeline_index"] = params.expected_timeline_index
        return "delete_feature", tool_input

    if operation.type == "adjust_feature_parameters":
        params = operation.params
        if not isinstance(params, FeatureParameterEditParams):
            raise ValueError("adjust_feature_parameters IR params shape mismatch")
        tool_input = {
            "feature_token": params.feature_ref,
            "parameters": _fusion_feature_parameter_payload(dict(params.parameters)),
            "description": params.description or "Adjust feature parameters",
            "expected_name": params.expected_name or "",
        }
        if params.expected_timeline_index is not None:
            tool_input["expected_timeline_index"] = params.expected_timeline_index
        return "adjust_feature_parameters", tool_input

    if operation.type == "set_feature_suppression":
        params = operation.params
        if not isinstance(params, FeatureSuppressionParams):
            raise ValueError("set_feature_suppression IR params shape mismatch")
        tool_input = {
            "feature_token": params.feature_ref,
            "description": params.description or ("Suppress feature" if params.suppress else "Unsuppress feature"),
            "expected_name": params.expected_name or "",
        }
        if params.expected_timeline_index is not None:
            tool_input["expected_timeline_index"] = params.expected_timeline_index
        return ("suppress_feature" if params.suppress else "unsuppress_feature"), tool_input

    if operation.type == "fillet":
        params = operation.params
        if not isinstance(params, FilletParams):
            raise ValueError("fillet IR params shape mismatch")
        return "apply_fillet", {
            "edge_refs": list(params.edge_refs),
            "radius": params.radius,
            "radius_unit": "mm",
            "include_tangent_edges": params.include_tangent_edges,
            "feature_name": params.feature_name or "",
            "description": "Apply IR fillet",
        }

    if operation.type == "chamfer":
        params = operation.params
        if not isinstance(params, ChamferParams):
            raise ValueError("chamfer IR params shape mismatch")
        return "apply_chamfer", {
            "edge_refs": list(params.edge_refs),
            "distance": params.distance,
            "distance_unit": "mm",
            "include_tangent_edges": params.include_tangent_edges,
            "feature_name": params.feature_name or "",
            "description": "Apply IR chamfer",
        }

    if operation.type == "shell":
        params = operation.params
        if not isinstance(params, ShellParams):
            raise ValueError("shell IR params shape mismatch")
        return "create_shell", {
            "mode": params.mode,
            "face_refs": list(params.face_refs),
            "body_refs": list(params.body_refs),
            "inside_thickness": params.inside_thickness,
            "outside_thickness": params.outside_thickness,
            "thickness_unit": "mm",
            "is_tangent_chain": params.is_tangent_chain,
            "shell_type": params.shell_type,
            "feature_name": params.feature_name or "",
            "description": "Create IR shell",
        }

    if operation.type == "create_simple_hole":
        params = operation.params
        if not isinstance(params, SimpleHoleParams):
            raise ValueError("create_simple_hole IR params shape mismatch")
        tool_input = {
            "face_ref": params.face_ref,
            "center_x": params.center[0],
            "center_y": params.center[1],
            "center_z": params.center[2],
            "diameter": params.diameter,
            "diameter_unit": "mm",
            "extent_type": params.extent_type,
            "feature_name": params.feature_name or "",
            "description": "Create IR simple hole",
        }
        if params.depth is not None:
            tool_input["depth"] = params.depth
        return "create_simple_hole", tool_input

    if operation.type == "create_counterbore_hole":
        params = operation.params
        if not isinstance(params, CounterboreHoleParams):
            raise ValueError("create_counterbore_hole IR params shape mismatch")
        return "create_counterbore_hole", {
            "face_ref": params.face_ref,
            "center_x": params.center[0],
            "center_y": params.center[1],
            "center_z": params.center[2],
            "hole_diameter": params.hole_diameter,
            "hole_depth": params.hole_depth,
            "counterbore_diameter": params.counterbore_diameter,
            "counterbore_depth": params.counterbore_depth,
            "diameter_unit": "mm",
            "feature_name": params.feature_name or "",
            "description": "Create IR counterbore hole",
        }

    if operation.type == "create_tapped_hole":
        params = operation.params
        if not isinstance(params, TappedHoleParams):
            raise ValueError("create_tapped_hole IR params shape mismatch")
        tool_input = {
            "face_ref": params.face_ref,
            "center_x": params.center[0],
            "center_y": params.center[1],
            "center_z": params.center[2],
            "thread_type": params.thread_type,
            "thread_size": params.thread_size,
            "thread_depth": params.thread_depth,
            "diameter_unit": params.diameter_unit,
            "feature_name": params.feature_name or "",
            "description": "Create IR tapped hole",
        }
        if params.pilot_hole_depth is not None:
            tool_input["pilot_hole_depth"] = params.pilot_hole_depth
        return "create_tapped_hole", tool_input

    if operation.type == "create_external_thread":
        params = operation.params
        if not isinstance(params, ExternalThreadParams):
            raise ValueError("create_external_thread IR params shape mismatch")
        tool_input = {
            "face_ref": params.face_ref,
            "thread_type": params.thread_type,
            "thread_size": params.thread_size,
            "thread_offset": _mm_to_cm(params.thread_offset),
            "is_full_length": params.is_full_length,
            "diameter_unit": params.diameter_unit,
            "feature_name": params.feature_name or "",
            "description": "Create IR external thread",
        }
        if params.thread_length is not None:
            tool_input["thread_length"] = _mm_to_cm(params.thread_length)
        return "create_external_thread", tool_input

    if operation.type == "pattern_feature":
        params = operation.params
        if not isinstance(params, PatternFeatureParams):
            raise ValueError("pattern_feature IR params shape mismatch")
        tool_input = {
            "pattern_type": params.pattern_type,
            "feature_refs": list(params.feature_refs),
            "description": "Create IR feature pattern",
        }
        if params.count_x is not None:
            tool_input["count_x"] = params.count_x
        if params.spacing_x is not None:
            tool_input["spacing_x_cm"] = _mm_to_cm(params.spacing_x)
        if params.count_y is not None:
            tool_input["count_y"] = params.count_y
        if params.spacing_y is not None:
            tool_input["spacing_y_cm"] = _mm_to_cm(params.spacing_y)
        if params.rotation_count is not None:
            tool_input["rotation_count"] = params.rotation_count
        if params.rotation_angle_degrees is not None:
            tool_input["rotation_angle_deg"] = params.rotation_angle_degrees
        if params.orientation_hint is not None:
            tool_input["orientation_hint"] = params.orientation_hint
        if params.feature_name:
            tool_input["feature_name"] = params.feature_name
        return "create_pattern_feature", tool_input

    if operation.type == "list_features":
        params = operation.params
        if not isinstance(params, ListFeaturesParams):
            raise ValueError("list_features IR params shape mismatch")
        return "list_features", {"description": params.description or "List features"}

    if operation.type == "select_entities":
        params = operation.params
        if not isinstance(params, SelectEntitiesParams):
            raise ValueError("select_entities IR params shape mismatch")
        if params.kind == "edge":
            return "select_edges", {"edge_refs": list(params.refs), "clear_existing": params.clear_existing, "description": "Select IR edges"}
        if params.kind == "face":
            return "select_faces", {"face_refs": list(params.refs), "clear_existing": params.clear_existing, "description": "Select IR faces"}
        return "select_bodies", {"body_refs": list(params.refs), "clear_existing": params.clear_existing, "description": "Select IR bodies"}

    if operation.type == "clear_selection":
        params = operation.params
        if not isinstance(params, ClearSelectionParams):
            raise ValueError("clear_selection IR params shape mismatch")
        if params.kind == "edge":
            return "clear_edge_selection", {"description": "Clear IR edge selection"}
        if params.kind == "face":
            return "clear_face_selection", {"description": "Clear IR face selection"}
        return "clear_body_selection", {"description": "Clear IR body selection"}

    raise ValueError(f"Unsupported IR operation for Fusion translator: {operation.type}")
