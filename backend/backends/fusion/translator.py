"""Translate shared IR operations into Fusion-compatible tool calls."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ...ir.types import AddCircleParams, AddRectangleParams, CreateSketchParams, ExtrudeParams, IROperation


_OPERATION_TO_FUSION = {
    "new": "NewBody",
    "join": "Join",
    "cut": "Cut",
    "intersect": "Intersect",
}


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
    if operation.type == "create_sketch":
        params = operation.params
        if not isinstance(params, CreateSketchParams):
            raise ValueError("create_sketch IR params shape mismatch")
        return "create_sketch", {
            "plane_id": params.plane,
            "sketch_id": params.sketch,
            "description": f"Create sketch {params.sketch} on {params.plane}",
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
            "corner1_u": c_u - half_w,
            "corner1_v": c_v - half_h,
            "corner2_u": c_u + half_w,
            "corner2_v": c_v + half_h,
            "description": f"Add rectangle to {params.sketch}",
        }

    if operation.type == "add_circle":
        params = operation.params
        if not isinstance(params, AddCircleParams):
            raise ValueError("add_circle IR params shape mismatch")

        return "add_circle", {
            "sketch_id": params.sketch,
            "center_u": params.center[0],
            "center_v": params.center[1],
            "radius": params.radius,
            "description": f"Add circle to {params.sketch}",
        }

    if operation.type == "extrude":
        params = operation.params
        if not isinstance(params, ExtrudeParams):
            raise ValueError("extrude IR params shape mismatch")

        sketch_id = params.sketch
        profile_index = params.profile_index if params.profile_index is not None else 0

        if not sketch_id:
            parsed_sketch, parsed_index = _parse_profile_reference(params.profile)
            sketch_id = parsed_sketch
            profile_index = parsed_index

        signed_distance = params.distance if params.direction == "positive" else -params.distance
        fusion_operation = _OPERATION_TO_FUSION.get(params.operation)
        if fusion_operation is None:
            raise ValueError(f"Unsupported Fusion extrude operation: {params.operation}")

        return "extrude_profile", {
            "sketch_id": sketch_id,
            "profile_index": profile_index,
            "distance": signed_distance,
            "operation": fusion_operation,
            "description": f"Extrude {params.profile}",
        }

    raise ValueError(f"Unsupported IR operation for Fusion translator: {operation.type}")
