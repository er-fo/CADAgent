"""Translate shared IR documents into executable build123d code."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ...ir.capabilities import get_build123d_capability_for_ir_operation
from ...ir.types import (
    AddArcParams,
    AddCircleParams,
    AddLineParams,
    AddRectangleParams,
    CreateSketchParams,
    ExtrudeParams,
    IRDocument,
    ListSketchProfilesParams,
)


@dataclass(frozen=True)
class Build123dProgram:
    code: str


class Build123dCapabilityError(ValueError):
    """Raised when the build123d target lacks a contracted IR capability."""

    def __init__(self, operation_type: str, reason: str):
        self.operation_type = operation_type
        self.reason = reason
        super().__init__(
            f"Unsupported IR operation for build123d translator: {operation_type}. {reason}"
        )


Point2D = Tuple[float, float]


@dataclass
class _SketchState:
    operations: List[Tuple[str, object]]


def _plane_expr(plane: str) -> str:
    plane_expr = {
        "XY": "Plane.XY",
        "XZ": "Plane.XZ",
        "YZ": "Plane.YZ",
    }.get(plane)
    if plane_expr is None:
        raise ValueError(f"Unsupported sketch plane for build123d translator: {plane}")
    return plane_expr


def _mode_expr(operation: str) -> str:
    mode_expr = {
        "new": "Mode.ADD",
        "join": "Mode.ADD",
        "cut": "Mode.SUBTRACT",
        "intersect": "Mode.INTERSECT",
    }.get(operation)
    if mode_expr is None:
        raise ValueError(f"Unsupported extrude operation for build123d translator: {operation}")
    return mode_expr


def _point_expr(point: Sequence[float]) -> str:
    return f"({point[0]}, {point[1]})"


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _points_close(a: Point2D, b: Point2D, *, tolerance: float = 1e-7) -> bool:
    return _distance(a, b) <= tolerance


def _arc_angles(params: AddArcParams) -> Tuple[float, float, float]:
    radius_start = _distance(params.center, params.start)
    radius_end = _distance(params.center, params.end)
    if not math.isclose(radius_start, radius_end, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError(
            "add_arc start and end must be the same distance from center for deterministic build123d translation"
        )

    start_angle = math.degrees(
        math.atan2(params.start[1] - params.center[1], params.start[0] - params.center[0])
    )
    end_angle = math.degrees(
        math.atan2(params.end[1] - params.center[1], params.end[0] - params.center[0])
    )
    arc_size = (end_angle - start_angle) % 360.0
    if math.isclose(arc_size, 0.0, abs_tol=1e-7):
        raise ValueError("add_arc start and end angles must produce a non-zero arc for build123d translation")
    return radius_start, start_angle, arc_size


def _curve_endpoints(item: Tuple[str, object]) -> Optional[Tuple[Point2D, Point2D]]:
    kind, params = item
    if kind == "line":
        if not isinstance(params, AddLineParams):
            raise ValueError("add_line IR params shape mismatch")
        return (float(params.start[0]), float(params.start[1])), (float(params.end[0]), float(params.end[1]))
    if kind == "arc":
        if not isinstance(params, AddArcParams):
            raise ValueError("add_arc IR params shape mismatch")
        return (float(params.start[0]), float(params.start[1])), (float(params.end[0]), float(params.end[1]))
    return None


def _ordered_curve_items(sketch_id: str, operations: Iterable[Tuple[str, object]]) -> List[Tuple[str, object]]:
    curve_items = [item for item in operations if item[0] in {"line", "arc"}]
    if not curve_items:
        return []

    first = _curve_endpoints(curve_items[0])
    if first is None:
        return []
    loop_start, previous_end = first

    for item in curve_items[1:]:
        endpoints = _curve_endpoints(item)
        if endpoints is None:
            continue
        start, end = endpoints
        if not _points_close(previous_end, start):
            raise ValueError(
                f"Sketch '{sketch_id}' line/arc profile is not an ordered closed loop: "
                f"curve starts at {start} but previous curve ended at {previous_end}."
            )
        previous_end = end

    if not _points_close(previous_end, loop_start):
        raise ValueError(
            f"Sketch '{sketch_id}' line/arc profile does not form a closed loop for build123d extrusion."
        )
    return curve_items


def _append_sketch_plane_guard(lines: List[str], sketch_id: str, operation_name: str) -> None:
    lines.extend(
        [
            f"    if {sketch_id!r} not in _sketch_planes:",
            f"        raise ValueError(\"Sketch plane missing for '{sketch_id}'. Ensure create_sketch succeeded before {operation_name}.\")",
        ]
    )


def _append_sketch_build(lines: List[str], sketch_id: str, state: _SketchState) -> None:
    if not state.operations:
        raise ValueError(f"Sketch '{sketch_id}' has no profile geometry for build123d translation.")

    curve_items = _ordered_curve_items(sketch_id, state.operations)
    lines.extend(
        [
            f"    _plane = _sketch_planes[{sketch_id!r}]",
            "    with BuildSketch(_plane):",
        ]
    )

    if curve_items:
        lines.append("        with BuildLine():")
        for kind, params in curve_items:
            if kind == "line":
                if not isinstance(params, AddLineParams):
                    raise ValueError("add_line IR params shape mismatch")
                lines.append(f"            Line({_point_expr(params.start)}, {_point_expr(params.end)})")
                continue

            if kind == "arc":
                if not isinstance(params, AddArcParams):
                    raise ValueError("add_arc IR params shape mismatch")
                radius, start_angle, arc_size = _arc_angles(params)
                lines.append(
                    f"            CenterArc({_point_expr(params.center)}, {radius}, {start_angle}, {arc_size})"
                )
                continue
        lines.append("        make_face()")

    for kind, params in state.operations:
        if kind == "rectangle":
            if not isinstance(params, AddRectangleParams):
                raise ValueError("add_rectangle IR params shape mismatch")
            lines.extend(
                [
                    f"        with Locations(({params.center[0]}, {params.center[1]})):",
                    f"            Rectangle({params.width}, {params.height})",
                ]
            )
            continue

        if kind == "circle":
            if not isinstance(params, AddCircleParams):
                raise ValueError("add_circle IR params shape mismatch")
            lines.extend(
                [
                    f"        with Locations(({params.center[0]}, {params.center[1]})):",
                    f"            Circle({params.radius})",
                ]
            )
            continue

        if kind in {"line", "arc"}:
            continue

        raise ValueError(f"Unsupported build123d sketch geometry kind: {kind}")


def translate_ir_document_to_build123d(document: IRDocument) -> Build123dProgram:
    """
    Build Python source that executes the IR in build123d.

    The generated code stores sketch primitives in deterministic IR order and
    materializes the referenced sketch immediately before each extrusion.  This
    lets line/arc loops become one build123d face instead of isolated curves.
    """
    lines: List[str] = [
        "from build123d import BuildLine, BuildPart, BuildSketch, CenterArc, Circle, Line, Locations, Mode, Plane, Rectangle, extrude, make_face",
        "",
        "_sketch_planes = {}",
        "_last_sketch_for_profile = {}",
        "",
        "with BuildPart() as _part:",
    ]
    sketches: Dict[str, _SketchState] = {}

    for op in document.operations:
        if op.type == "create_sketch":
            params = op.params
            if not isinstance(params, CreateSketchParams):
                raise ValueError("create_sketch IR params shape mismatch")
            plane_expr = _plane_expr(params.plane)
            sketches[params.sketch] = _SketchState(operations=[])
            lines.extend(
                [
                    f"    # {op.id}: create_sketch",
                    f"    _sketch_planes[{params.sketch!r}] = {plane_expr}",
                ]
            )
            continue

        if op.type == "add_rectangle":
            params = op.params
            if not isinstance(params, AddRectangleParams):
                raise ValueError("add_rectangle IR params shape mismatch")
            sketches.setdefault(params.sketch, _SketchState(operations=[])).operations.append(("rectangle", params))
            lines.extend(
                [
                    f"    # {op.id}: add_rectangle",
                    f"    _last_sketch_for_profile[{params.sketch!r}] = {op.id!r}",
                ]
            )
            _append_sketch_plane_guard(lines, params.sketch, "add_rectangle")
            continue

        if op.type == "add_circle":
            params = op.params
            if not isinstance(params, AddCircleParams):
                raise ValueError("add_circle IR params shape mismatch")
            sketches.setdefault(params.sketch, _SketchState(operations=[])).operations.append(("circle", params))
            lines.extend(
                [
                    f"    # {op.id}: add_circle",
                    f"    _last_sketch_for_profile[{params.sketch!r}] = {op.id!r}",
                ]
            )
            _append_sketch_plane_guard(lines, params.sketch, "add_circle")
            continue

        if op.type == "add_line":
            params = op.params
            if not isinstance(params, AddLineParams):
                raise ValueError("add_line IR params shape mismatch")
            sketches.setdefault(params.sketch, _SketchState(operations=[])).operations.append(("line", params))
            lines.extend(
                [
                    f"    # {op.id}: add_line",
                    f"    _last_sketch_for_profile[{params.sketch!r}] = {op.id!r}",
                ]
            )
            _append_sketch_plane_guard(lines, params.sketch, "add_line")
            continue

        if op.type == "add_arc":
            params = op.params
            if not isinstance(params, AddArcParams):
                raise ValueError("add_arc IR params shape mismatch")
            _arc_angles(params)
            sketches.setdefault(params.sketch, _SketchState(operations=[])).operations.append(("arc", params))
            lines.extend(
                [
                    f"    # {op.id}: add_arc",
                    f"    _last_sketch_for_profile[{params.sketch!r}] = {op.id!r}",
                ]
            )
            _append_sketch_plane_guard(lines, params.sketch, "add_arc")
            continue

        if op.type == "extrude":
            params = op.params
            if not isinstance(params, ExtrudeParams):
                raise ValueError("extrude IR params shape mismatch")
            sketch_id = (params.sketch or "").strip()
            if not sketch_id and ":" in params.profile:
                sketch_id = params.profile.split(":", 1)[0].strip()
            if not sketch_id:
                raise ValueError("extrude IR params must provide a sketch or '<sketch>:profile_<index>' profile reference")
            sketch_state = sketches.get(sketch_id)
            if sketch_state is None:
                raise ValueError(f"Sketch '{sketch_id}' has no build123d sketch state for extrusion.")
            signed_distance = params.distance if params.direction == "positive" else -params.distance
            lines.extend(
                [
                    f"    # {op.id}: extrude",
                ]
            )
            _append_sketch_plane_guard(lines, sketch_id, "extrude")
            _append_sketch_build(lines, sketch_id, sketch_state)
            lines.append(f"    extrude(amount={signed_distance}, mode={_mode_expr(params.operation)})")
            continue

        if op.type == "list_sketch_profiles":
            params = op.params
            if not isinstance(params, ListSketchProfilesParams):
                raise ValueError("list_sketch_profiles IR params shape mismatch")
            lines.extend(
                [
                    f"    # {op.id}: list_sketch_profiles",
                ]
            )
            _append_sketch_plane_guard(lines, params.sketch, "list_sketch_profiles")
            continue

        capability = get_build123d_capability_for_ir_operation(op.type)
        raise Build123dCapabilityError(op.type, capability.build123d_reason)

    lines.extend(
        [
            "",
            "part_result = _part.part",
        ]
    )

    return Build123dProgram(code="\n".join(lines) + "\n")
