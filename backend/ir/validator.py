"""Validation for shared IR operations before target execution."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from .types import (
    AddCircleParams,
    AddRectangleParams,
    CreateSketchParams,
    ExtrudeParams,
    IROperation,
)

_DATUM_PLANES = {"XY", "XZ", "YZ"}
_FACE_ALIAS_PATTERN = re.compile(r"^face_\d+$")


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sketch_ids_from_committed(operations: Sequence[IROperation]) -> set[str]:
    sketches: set[str] = set()
    for op in operations:
        if op.type != "create_sketch":
            continue
        if isinstance(op.params, CreateSketchParams) and _is_nonempty_string(op.params.sketch):
            sketches.add(op.params.sketch.strip())
    return sketches


def _has_committed_profile_for_sketch(
    operations: Sequence[IROperation],
    sketch_id: str,
) -> bool:
    for op in operations:
        if op.type == "add_rectangle" and isinstance(op.params, AddRectangleParams):
            if op.params.sketch.strip() == sketch_id:
                return True
        if op.type == "add_circle" and isinstance(op.params, AddCircleParams):
            if op.params.sketch.strip() == sketch_id:
                return True
    return False


def validate_operation(operation: IROperation) -> List[str]:
    """Validate a single IR operation."""
    errors: List[str] = []

    if operation.type == "create_sketch":
        params = operation.params
        if not isinstance(params, CreateSketchParams):
            errors.append("create_sketch params must be CreateSketchParams")
            return errors
        if not _is_nonempty_string(params.plane):
            errors.append("create_sketch plane must be provided")
            return errors
        source = str((operation.metadata or {}).get("source") or "").strip().lower()
        if source == "studio" and str(params.plane).strip().upper() not in _DATUM_PLANES:
            errors.append("create_sketch plane for studio target must be one of XY/XZ/YZ")
        if source == "fusion" and _FACE_ALIAS_PATTERN.match(str(params.plane).strip()):
            errors.append(
                "create_sketch plane for fusion target must be a datum plane (XY/XZ/YZ), "
                "construction plane ID, or resolved face token; alias refs like face_N are not allowed"
            )
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


def validate_ir_candidate(
    operation: IROperation,
    committed_operations: Sequence[IROperation],
) -> List[str]:
    """
    Validate one candidate operation against already committed operations.

    This enforces fail-closed semantics: dependencies must point to successful
    committed operations, never to speculative or failed operations.
    """
    errors = validate_operation(operation)
    committed_ids = {op.id for op in committed_operations}
    committed_sketches = _sketch_ids_from_committed(committed_operations)

    missing_dependencies = [dep for dep in operation.dependencies if dep not in committed_ids]
    if missing_dependencies:
        errors.append(
            "Operation depends on uncommitted operation(s): " + ", ".join(missing_dependencies)
        )

    if operation.type in {"add_rectangle", "add_circle"}:
        sketch = getattr(operation.params, "sketch", "")
        if sketch and sketch not in committed_sketches:
            errors.append(f"Referenced sketch '{sketch}' does not exist yet")

    if operation.type == "extrude" and isinstance(operation.params, ExtrudeParams):
        sketch = (operation.params.sketch or "").strip()
        if sketch and sketch not in committed_sketches:
            errors.append(f"Referenced sketch '{sketch}' does not exist yet")
        elif sketch and not _has_committed_profile_for_sketch(committed_operations, sketch):
            errors.append(
                f"Extrude for sketch '{sketch}' requires at least one committed profile operation "
                "(add_rectangle/add_circle) before extrusion"
            )

    return errors


def validate_ir_sequence(operations: Sequence[IROperation]) -> Dict[str, List[str]]:
    """
    Validate a sequence and cross-operation dependencies.

    Returns a map keyed by operation id with validation errors.
    """
    errors_by_op: Dict[str, List[str]] = {}
    committed_valid_ops: List[IROperation] = []

    for op in operations:
        op_errors = validate_ir_candidate(op, committed_valid_ops)
        if op_errors:
            errors_by_op[op.id] = op_errors
            continue
        committed_valid_ops.append(op)

    return errors_by_op
