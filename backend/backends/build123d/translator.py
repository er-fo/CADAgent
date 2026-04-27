"""Translate shared IR documents into executable build123d code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ...ir.types import AddCircleParams, AddRectangleParams, CreateSketchParams, ExtrudeParams, IRDocument


@dataclass(frozen=True)
class Build123dProgram:
    code: str


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


def translate_ir_document_to_build123d(document: IRDocument) -> Build123dProgram:
    """
    Build Python source that executes the IR in build123d.

    The generated code intentionally keeps one shape per sketch operation, which
    is sufficient for MVP cube/cylinder flows and deterministic regression tests.
    """
    lines: List[str] = [
        "from build123d import BuildPart, BuildSketch, Circle, Mode, Plane, Rectangle, Locations, extrude",
        "",
        "_sketch_planes = {}",
        "_last_sketch_for_profile = {}",
        "",
        "with BuildPart() as _part:",
    ]

    for op in document.operations:
        if op.type == "create_sketch":
            params = op.params
            if not isinstance(params, CreateSketchParams):
                raise ValueError("create_sketch IR params shape mismatch")
            lines.extend(
                [
                    f"    # {op.id}: create_sketch",
                    f"    _sketch_planes[{params.sketch!r}] = {_plane_expr(params.plane)}",
                ]
            )
            continue

        if op.type == "add_rectangle":
            params = op.params
            if not isinstance(params, AddRectangleParams):
                raise ValueError("add_rectangle IR params shape mismatch")
            lines.extend(
                [
                    f"    # {op.id}: add_rectangle",
                    f"    _plane = _sketch_planes.get({params.sketch!r}, Plane.XY)",
                    "    with BuildSketch(_plane):",
                    f"        with Locations(({params.center[0]}, {params.center[1]})):",
                    f"            Rectangle({params.width}, {params.height})",
                    f"    _last_sketch_for_profile[{params.sketch!r}] = {op.id!r}",
                ]
            )
            continue

        if op.type == "add_circle":
            params = op.params
            if not isinstance(params, AddCircleParams):
                raise ValueError("add_circle IR params shape mismatch")
            lines.extend(
                [
                    f"    # {op.id}: add_circle",
                    f"    _plane = _sketch_planes.get({params.sketch!r}, Plane.XY)",
                    "    with BuildSketch(_plane):",
                    f"        with Locations(({params.center[0]}, {params.center[1]})):",
                    f"            Circle({params.radius})",
                    f"    _last_sketch_for_profile[{params.sketch!r}] = {op.id!r}",
                ]
            )
            continue

        if op.type == "extrude":
            params = op.params
            if not isinstance(params, ExtrudeParams):
                raise ValueError("extrude IR params shape mismatch")
            signed_distance = params.distance if params.direction == "positive" else -params.distance
            lines.extend(
                [
                    f"    # {op.id}: extrude",
                    f"    extrude(amount={signed_distance}, mode={_mode_expr(params.operation)})",
                ]
            )
            continue

        raise ValueError(f"Unsupported IR operation for build123d translator: {op.type}")

    lines.extend(
        [
            "",
            "part_result = _part.part",
        ]
    )

    return Build123dProgram(code="\n".join(lines) + "\n")
