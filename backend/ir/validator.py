"""Validation for shared IR operations before target execution."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .types import (
    AddCircleParams,
    AddRectangleParams,
    CreateSketchParams,
    ExtrudeParams,
    IROperation,
)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_operation(operation: IROperation) -> List[str]:
    """Validate a single IR operation."""
    errors: List[str] = []

    if operation.type == "create_sketch":
        params = operation.params
        if not isinstance(params, CreateSketchParams):
            errors.append("create_sketch params must be CreateSketchParams")
            return errors
        if params.plane not in {"XY", "XZ", "YZ"}:
            errors.append("create_sketch plane must be one of XY/XZ/YZ")
        if not _is_nonempty_string(params.sketch):
            errors.append("create_sketch sketch must be a non-empty identifier")
        return errors

    if operation.type == "add_rectangle":
        params = operation.params
        if not isinstance(params, AddRectangleParams):
            errors.append("add_rectangle params must be AddRectangleParams")
            return errors
        if not _is_nonempty_string(params.sketch):
            errors.append("add_rectangle sketch must be provided")
        if len(params.center) != 2:
            errors.append("add_rectangle center must contain exactly 2 values")
        if params.width <= 0:
            errors.append("add_rectangle width must be > 0")
        if params.height <= 0:
            errors.append("add_rectangle height must be > 0")
        return errors

    if operation.type == "add_circle":
        params = operation.params
        if not isinstance(params, AddCircleParams):
            errors.append("add_circle params must be AddCircleParams")
            return errors
        if not _is_nonempty_string(params.sketch):
            errors.append("add_circle sketch must be provided")
        if len(params.center) != 2:
            errors.append("add_circle center must contain exactly 2 values")
        if params.radius <= 0:
            errors.append("add_circle radius must be > 0")
        return errors

    if operation.type == "extrude":
        params = operation.params
        if not isinstance(params, ExtrudeParams):
            errors.append("extrude params must be ExtrudeParams")
            return errors
        if not _is_nonempty_string(params.profile):
            errors.append("extrude profile must be provided")
        if params.distance <= 0:
            errors.append("extrude distance must be > 0")
        if params.direction not in {"positive", "negative"}:
            errors.append("extrude direction must be positive or negative")
        if params.operation not in {"new", "join", "cut", "intersect"}:
            errors.append("extrude operation must be one of new/join/cut/intersect")
        return errors

    return [f"Unsupported IR operation type: {operation.type}"]


def validate_ir_sequence(operations: Sequence[IROperation]) -> Dict[str, List[str]]:
    """
    Validate a sequence and cross-operation dependencies.

    Returns a map keyed by operation id with validation errors.
    """
    errors_by_op: Dict[str, List[str]] = {}
    known_sketches: set[str] = set()

    for op in operations:
        op_errors = validate_operation(op)

        if op.type == "create_sketch" and not op_errors:
            params = op.params
            if isinstance(params, CreateSketchParams):
                known_sketches.add(params.sketch)

        if op.type in {"add_rectangle", "add_circle"}:
            params = op.params
            sketch = getattr(params, "sketch", "")
            if sketch and sketch not in known_sketches:
                op_errors.append(f"Referenced sketch '{sketch}' does not exist yet")

        if op.type == "extrude":
            params = op.params
            if isinstance(params, ExtrudeParams) and params.sketch and params.sketch not in known_sketches:
                op_errors.append(f"Referenced sketch '{params.sketch}' does not exist yet")

        if op_errors:
            errors_by_op[op.id] = op_errors

    return errors_by_op
