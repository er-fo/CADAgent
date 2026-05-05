"""Translate shared IR documents into executable build123d code."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ...ir.capabilities import get_build123d_capability_for_ir_operation
from ...ir.types import (
    AddArcParams,
    AddCircleParams,
    AddLineParams,
    AddRectangleParams,
    ChamferParams,
    CounterboreHoleParams,
    CreateConstructionPlaneParams,
    CreateSketchParams,
    ExtrudeParams,
    ExternalThreadParams,
    FilletParams,
    IRDocument,
    ListSketchProfilesParams,
    LoftParams,
    RevolveParams,
    ShellParams,
    SimpleHoleParams,
    TappedHoleParams,
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


def _datum_plane_expr(plane: str) -> str:
    plane_expr = {
        "XY": "Plane.XY",
        "XZ": "Plane.XZ",
        "YZ": "Plane.YZ",
    }.get(plane)
    if plane_expr is None:
        raise ValueError(f"Unsupported sketch plane for build123d translator: {plane}")
    return plane_expr


def _plane_expr(plane: str, construction_planes: Dict[str, str]) -> str:
    if plane in construction_planes:
        return f"_construction_planes[{plane!r}]"
    return _datum_plane_expr(plane)


def _construction_plane_expr(params: CreateConstructionPlaneParams) -> str:
    if params.mode == "datum":
        datum_plane = params.datum_axis_plane or params.base_datum_plane or params.plane
        return _datum_plane_expr(str(datum_plane).strip().upper())

    if params.mode == "offset_from_datum":
        if params.base_datum_plane is None:
            raise ValueError("create_construction_plane offset_from_datum requires base_datum_plane")
        if params.offset is None:
            raise ValueError("create_construction_plane offset_from_datum requires offset")
        return f"{_datum_plane_expr(params.base_datum_plane)}.offset({params.offset})"

    raise Build123dCapabilityError(
        "create_construction_plane",
        "build123d construction planes currently support only portable datum and offset_from_datum modes; "
        f"{params.mode} depends on target face/edge topology selectors.",
    )


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


def _ref_list_expr(refs: Sequence[str]) -> str:
    return repr([str(ref).strip() for ref in refs if str(ref).strip()])


def _axis_expr(axis: Dict[str, object]) -> str:
    axis_type = str(axis.get("type") or "").strip().lower()
    if axis_type == "construction":
        axis_name = str(axis.get("axis") or "").strip().lower()
        axis_expr = {"x": "Axis.X", "y": "Axis.Y", "z": "Axis.Z"}.get(axis_name)
        if axis_expr is None:
            raise ValueError(f"Unsupported build123d construction revolve axis: {axis_name}")
        return axis_expr

    raise Build123dCapabilityError(
        "revolve",
        "build123d revolve currently supports only portable construction axes; "
        f"axis type {axis_type or '<missing>'} requires non-portable selector resolution.",
    )


def _revolution_arc(extent: Dict[str, object]) -> float:
    mode = str(extent.get("mode") or "full").strip().lower()
    if mode == "full":
        return 360.0
    if mode == "angle":
        angle = extent.get("angle_degrees")
        if angle is None:
            raise ValueError("revolve extent mode angle requires angle_degrees")
        return float(angle)

    raise Build123dCapabilityError(
        "revolve",
        "build123d revolve currently supports only full and angle extents; "
        f"extent mode {mode or '<missing>'} requires non-portable target entity semantics.",
    )


def _point_expr(point: Sequence[float]) -> str:
    return f"({point[0]}, {point[1]})"


def _point3_expr(point: Sequence[float]) -> str:
    return f"({point[0]}, {point[1]}, {point[2]})"


def _shell_amount(params: ShellParams) -> float:
    inside = float(params.inside_thickness or 0.0)
    outside = float(params.outside_thickness or 0.0)
    if inside > 0.0 and outside > 0.0:
        raise Build123dCapabilityError(
            "shell",
            "build123d shell translation supports either inside_thickness or outside_thickness, not both.",
        )
    if inside > 0.0:
        return -inside
    if outside > 0.0:
        return outside
    raise ValueError("create_shell requires a positive inside_thickness or outside_thickness for build123d translation")


def _thread_nominal_diameter(thread_size: str) -> float:
    text = str(thread_size or "").strip()
    metric_match = re.fullmatch(r"M(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if metric_match:
        return float(metric_match.group(1))

    number_match = re.fullmatch(r"#(6|8|10)-\d+", text)
    if number_match:
        return {"6": 3.5052, "8": 4.1656, "10": 4.826}.get(number_match.group(1), 0.0)

    fraction_match = re.fullmatch(r"(\d+)/(\d+)-\d+", text)
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        return (numerator / denominator) * 25.4

    raise Build123dCapabilityError(
        "create_tapped_hole",
        f"build123d tapped-hole geometry cannot infer nominal diameter for thread size {thread_size!r}.",
    )


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
        "from build123d import Axis, BuildLine, BuildPart, BuildSketch, CenterArc, Circle, CounterBoreHole, Hole, Line, Locations, Mode, Plane, Rectangle, extrude, fillet, chamfer, loft, make_face, offset, revolve",
        "",
        "_thread_metadata = []",
        "_construction_planes = {}",
        "_sketch_planes = {}",
        "_last_sketch_for_profile = {}",
        "",
        "def _parse_ref_index(ref, prefix, count):",
        "    ref_text = str(ref or '').strip()",
        "    expected = prefix + '_'",
        "    if not ref_text.startswith(expected):",
        "        raise ValueError(f\"Unsupported build123d {prefix} ref '{ref_text}'. Expected {expected}<index>.\")",
        "    try:",
        "        index = int(ref_text[len(expected):])",
        "    except ValueError as exc:",
        "        raise ValueError(f\"Unsupported build123d {prefix} ref '{ref_text}'. Expected numeric index.\") from exc",
        "    if index < 0 or index >= count:",
        "        raise ValueError(f\"build123d {prefix} ref '{ref_text}' is out of range for {count} available {prefix}s.\")",
        "    return index",
        "",
        "def _edge_refs(refs):",
        "    edges = list(_part.part.edges())",
        "    return [edges[_parse_ref_index(ref, 'edge', len(edges))] for ref in refs]",
        "",
        "def _face_refs(refs):",
        "    faces = list(_part.part.faces())",
        "    return [faces[_parse_ref_index(ref, 'face', len(faces))] for ref in refs]",
        "",
        "def _assert_z_face(face, operation):",
        "    normal_at = getattr(face, 'normal_at', None)",
        "    normal = normal_at(0.5, 0.5) if callable(normal_at) else None",
        "    if normal is None or abs(float(normal.Z)) < 0.999:",
        "        raise ValueError(f\"{operation} currently supports only planar faces normal to global Z in build123d translation.\")",
        "",
        "",
        "with BuildPart() as _part:",
    ]
    sketches: Dict[str, _SketchState] = {}
    construction_planes: Dict[str, str] = {}

    for op in document.operations:
        if op.type == "create_construction_plane":
            params = op.params
            if not isinstance(params, CreateConstructionPlaneParams):
                raise ValueError("create_construction_plane IR params shape mismatch")
            plane_expr = _construction_plane_expr(params)
            construction_planes[params.plane] = plane_expr
            lines.extend(
                [
                    f"    # {op.id}: create_construction_plane",
                    f"    _construction_planes[{params.plane!r}] = {plane_expr}",
                ]
            )
            continue

        if op.type == "create_sketch":
            params = op.params
            if not isinstance(params, CreateSketchParams):
                raise ValueError("create_sketch IR params shape mismatch")
            plane_expr = _plane_expr(params.plane, construction_planes)
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

        if op.type == "revolve":
            params = op.params
            if not isinstance(params, RevolveParams):
                raise ValueError("revolve IR params shape mismatch")
            if not params.is_solid:
                raise Build123dCapabilityError(
                    "revolve",
                    "build123d translator supports solid revolve output only; surface revolve remains non-portable.",
                )
            sketch_id = params.sketch.strip()
            if not sketch_id and ":" in params.profile:
                sketch_id = params.profile.split(":", 1)[0].strip()
            sketch_state = sketches.get(sketch_id)
            if sketch_state is None:
                raise ValueError(f"Sketch '{sketch_id}' has no build123d sketch state for revolve.")
            lines.extend(
                [
                    f"    # {op.id}: revolve",
                ]
            )
            _append_sketch_plane_guard(lines, sketch_id, "revolve")
            _append_sketch_build(lines, sketch_id, sketch_state)
            lines.append(
                f"    revolve(axis={_axis_expr(params.axis)}, revolution_arc={_revolution_arc(params.extent)}, mode={_mode_expr(params.operation)})"
            )
            continue

        if op.type == "loft":
            params = op.params
            if not isinstance(params, LoftParams):
                raise ValueError("loft IR params shape mismatch")
            if not params.prefer_solid:
                raise Build123dCapabilityError(
                    "loft",
                    "build123d translator supports solid loft output only; surface loft remains non-portable.",
                )
            lines.extend(
                [
                    f"    # {op.id}: loft",
                ]
            )
            for profile_id in params.profile_ids:
                sketch_id = profile_id.split(":", 1)[0].strip()
                sketch_state = sketches.get(sketch_id)
                if sketch_state is None:
                    raise ValueError(f"Sketch '{sketch_id}' has no build123d sketch state for loft.")
                _append_sketch_plane_guard(lines, sketch_id, "loft")
                _append_sketch_build(lines, sketch_id, sketch_state)
            lines.append(f"    loft(mode={_mode_expr(params.operation)})")
            continue

        if op.type == "fillet":
            params = op.params
            if not isinstance(params, FilletParams):
                raise ValueError("fillet IR params shape mismatch")
            lines.extend(
                [
                    f"    # {op.id}: fillet",
                    f"    fillet(_edge_refs({_ref_list_expr(params.edge_refs)}), {params.radius})",
                ]
            )
            continue

        if op.type == "chamfer":
            params = op.params
            if not isinstance(params, ChamferParams):
                raise ValueError("chamfer IR params shape mismatch")
            lines.extend(
                [
                    f"    # {op.id}: chamfer",
                    f"    chamfer(_edge_refs({_ref_list_expr(params.edge_refs)}), {params.distance})",
                ]
            )
            continue

        if op.type == "shell":
            params = op.params
            if not isinstance(params, ShellParams):
                raise ValueError("shell IR params shape mismatch")
            if params.mode == "open":
                if not params.face_refs:
                    raise ValueError("create_shell open mode requires face_refs for build123d translation")
                lines.extend(
                    [
                        f"    # {op.id}: shell",
                        f"    _shell_openings = _face_refs({_ref_list_expr(params.face_refs)})",
                        f"    offset(_part.part, amount={_shell_amount(params)}, openings=_shell_openings, mode=Mode.REPLACE)",
                    ]
                )
                continue
            if params.mode == "closed":
                lines.extend(
                    [
                        f"    # {op.id}: shell",
                        f"    offset(_part.part, amount={_shell_amount(params)}, mode=Mode.REPLACE)",
                    ]
                )
                continue
            raise ValueError(f"Unsupported shell mode for build123d translator: {params.mode}")

        if op.type == "create_simple_hole":
            params = op.params
            if not isinstance(params, SimpleHoleParams):
                raise ValueError("create_simple_hole IR params shape mismatch")
            lines.extend(
                [
                    f"    # {op.id}: create_simple_hole",
                    f"    _hole_face = _face_refs([{params.face_ref!r}])[0]",
                    "    _assert_z_face(_hole_face, 'create_simple_hole')",
                    f"    with Locations({_point3_expr(params.center)}):",
                    f"        Hole({params.diameter / 2.0}, depth={params.depth if params.depth is not None else 'None'}, mode=Mode.SUBTRACT)",
                ]
            )
            continue

        if op.type == "create_counterbore_hole":
            params = op.params
            if not isinstance(params, CounterboreHoleParams):
                raise ValueError("create_counterbore_hole IR params shape mismatch")
            lines.extend(
                [
                    f"    # {op.id}: create_counterbore_hole",
                    f"    _hole_face = _face_refs([{params.face_ref!r}])[0]",
                    "    _assert_z_face(_hole_face, 'create_counterbore_hole')",
                    f"    with Locations({_point3_expr(params.center)}):",
                    f"        CounterBoreHole({params.hole_diameter / 2.0}, {params.counterbore_diameter / 2.0}, {params.counterbore_depth}, depth={params.hole_depth}, mode=Mode.SUBTRACT)",
                ]
            )
            continue

        if op.type == "create_tapped_hole":
            params = op.params
            if not isinstance(params, TappedHoleParams):
                raise ValueError("create_tapped_hole IR params shape mismatch")
            pilot_depth = params.pilot_hole_depth if params.pilot_hole_depth is not None else params.thread_depth
            pilot_radius = _thread_nominal_diameter(params.thread_size) / 2.0
            lines.extend(
                [
                    f"    # {op.id}: create_tapped_hole",
                    f"    _hole_face = _face_refs([{params.face_ref!r}])[0]",
                    "    _assert_z_face(_hole_face, 'create_tapped_hole')",
                    f"    with Locations({_point3_expr(params.center)}):",
                    f"        Hole({pilot_radius}, depth={pilot_depth}, mode=Mode.SUBTRACT)",
                    "    _thread_metadata.append({",
                    f"        'operation_id': {op.id!r},",
                    "        'kind': 'tapped_hole',",
                    f"        'face_ref': {params.face_ref!r},",
                    f"        'thread_type': {params.thread_type!r},",
                    f"        'thread_size': {params.thread_size!r},",
                    f"        'thread_depth': {params.thread_depth},",
                    f"        'pilot_hole_depth': {pilot_depth},",
                    "    })",
                ]
            )
            continue

        if op.type == "create_external_thread":
            params = op.params
            if not isinstance(params, ExternalThreadParams):
                raise ValueError("create_external_thread IR params shape mismatch")
            lines.extend(
                [
                    f"    # {op.id}: create_external_thread",
                    f"    _thread_face = _face_refs([{params.face_ref!r}])[0]",
                    "    _thread_metadata.append({",
                    f"        'operation_id': {op.id!r},",
                    "        'kind': 'external_thread',",
                    f"        'face_ref': {params.face_ref!r},",
                    f"        'thread_type': {params.thread_type!r},",
                    f"        'thread_size': {params.thread_size!r},",
                    f"        'is_full_length': {params.is_full_length},",
                    f"        'thread_length': {params.thread_length!r},",
                    f"        'thread_offset': {params.thread_offset},",
                    "    })",
                ]
            )
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
            "thread_metadata = _thread_metadata",
        ]
    )

    return Build123dProgram(code="\n".join(lines) + "\n")
