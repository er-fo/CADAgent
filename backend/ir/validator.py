"""Validation for shared IR operations before target execution."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Mapping, Sequence

try:
    from ..thread_specs import validate_thread_spec
except ImportError:  # pragma: no cover - fallback for top-level imports
    from thread_specs import validate_thread_spec  # type: ignore

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
        if op.type == "list_sketch_profiles" and isinstance(op.params, ListSketchProfilesParams):
            if op.params.sketch.strip() == sketch_id and _committed_profile_inspection_has_profiles(op):
                return True
    return False


def _has_committed_sketch_geometry_for_profile_inspection(
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
        if op.type == "add_line" and isinstance(op.params, AddLineParams):
            if op.params.sketch.strip() == sketch_id:
                return True
        if op.type == "add_arc" and isinstance(op.params, AddArcParams):
            if op.params.sketch.strip() == sketch_id:
                return True
    return False


def _committed_profile_inspection_has_profiles(operation: IROperation) -> bool:
    for result in operation.target_results:
        if not isinstance(result, Mapping):
            continue
        for payload in (result, result.get("raw_result"), result.get("data")):
            if not isinstance(payload, Mapping):
                continue
            profile_count = payload.get("profile_count")
            if isinstance(profile_count, int) and profile_count > 0:
                return True
            if isinstance(profile_count, str):
                try:
                    if int(profile_count) > 0:
                        return True
                except ValueError:
                    pass
            profiles = payload.get("profiles")
            if isinstance(profiles, list) and len(profiles) > 0:
                return True
    return False


def _check_finite_number(value: float, field_name: str, errors: List[str]) -> bool:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        errors.append(f"{field_name} must be a finite number")
        return False
    if not math.isfinite(numeric_value):
        errors.append(f"{field_name} must be finite")
        return False
    return True


def _positive(value: float, field_name: str, errors: List[str]) -> None:
    if not _check_finite_number(value, field_name, errors):
        return
    if float(value) <= 0:
        errors.append(f"{field_name} must be > 0")


def _nonnegative(value: float, field_name: str, errors: List[str]) -> None:
    if not _check_finite_number(value, field_name, errors):
        return
    if float(value) < 0:
        errors.append(f"{field_name} must be >= 0")


def _check_point(values: Sequence[float], field_name: str, expected_len: int, errors: List[str]) -> None:
    if len(values) != expected_len:
        errors.append(f"{field_name} must contain exactly {expected_len} values")
        return
    for index, value in enumerate(values):
        _check_finite_number(value, f"{field_name}[{index}]", errors)


def _has_refs(values: Sequence[str]) -> bool:
    return any(_is_nonempty_string(value) for value in values)


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

    if operation.type == "create_construction_plane":
        params = operation.params
        if not isinstance(params, CreateConstructionPlaneParams):
            errors.append("create_construction_plane params must be CreateConstructionPlaneParams")
            return errors
        if not _is_nonempty_string(params.plane):
            errors.append("create_construction_plane plane must be a non-empty identifier")
        if params.mode not in {"datum", "offset_from_datum", "angle_to_edge", "face_normal"}:
            errors.append("create_construction_plane mode must be datum/offset_from_datum/angle_to_edge/face_normal")
        if params.mode == "datum" and params.datum_axis_plane and params.datum_axis_plane not in _DATUM_PLANES:
            errors.append("create_construction_plane datum_axis_plane must be one of XY/XZ/YZ")
        if params.mode == "offset_from_datum":
            if params.base_datum_plane not in _DATUM_PLANES:
                errors.append("create_construction_plane base_datum_plane must be one of XY/XZ/YZ")
            if params.offset is None:
                errors.append("create_construction_plane offset_from_datum requires offset")
            else:
                _check_finite_number(params.offset, "create_construction_plane offset", errors)
        if params.mode == "angle_to_edge":
            if not _is_nonempty_string(params.reference_face):
                errors.append("create_construction_plane angle_to_edge requires reference_face")
            if not _is_nonempty_string(params.reference_edge):
                errors.append("create_construction_plane angle_to_edge requires reference_edge")
            if params.angle_degrees is None:
                errors.append("create_construction_plane angle_to_edge requires angle_degrees")
            else:
                _check_finite_number(params.angle_degrees, "create_construction_plane angle_degrees", errors)
        if params.mode == "face_normal" and not _is_nonempty_string(params.face):
            errors.append("create_construction_plane face_normal requires face")
        if params.point is not None:
            _check_point(params.point, "create_construction_plane point", 3, errors)
        return errors

    if operation.type == "add_rectangle":
        params = operation.params
        if not isinstance(params, AddRectangleParams):
            errors.append("add_rectangle params must be AddRectangleParams")
            return errors
        if not _is_nonempty_string(params.sketch):
            errors.append("add_rectangle sketch must be provided")
        _check_point(params.center, "add_rectangle center", 2, errors)
        _positive(params.width, "add_rectangle width", errors)
        _positive(params.height, "add_rectangle height", errors)
        return errors

    if operation.type == "add_circle":
        params = operation.params
        if not isinstance(params, AddCircleParams):
            errors.append("add_circle params must be AddCircleParams")
            return errors
        if not _is_nonempty_string(params.sketch):
            errors.append("add_circle sketch must be provided")
        _check_point(params.center, "add_circle center", 2, errors)
        _positive(params.radius, "add_circle radius", errors)
        return errors

    if operation.type == "add_line":
        params = operation.params
        if not isinstance(params, AddLineParams):
            errors.append("add_line params must be AddLineParams")
            return errors
        if not _is_nonempty_string(params.sketch):
            errors.append("add_line sketch must be provided")
        _check_point(params.start, "add_line start", 2, errors)
        _check_point(params.end, "add_line end", 2, errors)
        if len(params.start) == 2 and len(params.end) == 2 and params.start == params.end:
            errors.append("add_line start and end must be distinct")
        return errors

    if operation.type == "add_arc":
        params = operation.params
        if not isinstance(params, AddArcParams):
            errors.append("add_arc params must be AddArcParams")
            return errors
        if not _is_nonempty_string(params.sketch):
            errors.append("add_arc sketch must be provided")
        _check_point(params.center, "add_arc center", 2, errors)
        _check_point(params.start, "add_arc start", 2, errors)
        _check_point(params.end, "add_arc end", 2, errors)
        if len(params.center) == len(params.start) == 2 and params.center == params.start:
            errors.append("add_arc center and start must be distinct")
        if len(params.center) == len(params.end) == 2 and params.center == params.end:
            errors.append("add_arc center and end must be distinct")
        if len(params.start) == len(params.end) == 2 and params.start == params.end:
            errors.append("add_arc start and end must be distinct")
        return errors

    if operation.type == "list_sketch_profiles":
        params = operation.params
        if not isinstance(params, ListSketchProfilesParams):
            errors.append("list_sketch_profiles params must be ListSketchProfilesParams")
            return errors
        if not _is_nonempty_string(params.sketch):
            errors.append("list_sketch_profiles sketch must be provided")
        return errors

    if operation.type == "extrude":
        params = operation.params
        if not isinstance(params, ExtrudeParams):
            errors.append("extrude params must be ExtrudeParams")
            return errors
        if not _is_nonempty_string(params.profile):
            errors.append("extrude profile must be provided")
        _positive(params.distance, "extrude distance", errors)
        if params.direction not in {"positive", "negative"}:
            errors.append("extrude direction must be positive or negative")
        if params.operation not in {"new", "join", "cut", "intersect"}:
            errors.append("extrude operation must be one of new/join/cut/intersect")
        return errors

    if operation.type == "revolve":
        params = operation.params
        if not isinstance(params, RevolveParams):
            errors.append("revolve params must be RevolveParams")
            return errors
        if not _is_nonempty_string(params.sketch):
            errors.append("revolve sketch must be provided")
        if not _is_nonempty_string(params.profile):
            errors.append("revolve profile must be provided")
        _nonnegative(params.profile_index, "revolve profile_index", errors)
        if not isinstance(params.axis, dict) or not params.axis:
            errors.append("revolve axis must be provided")
        if not isinstance(params.extent, dict) or not params.extent:
            errors.append("revolve extent must be provided")
        if params.operation not in {"new", "join", "cut", "intersect"}:
            errors.append("revolve operation must be one of new/join/cut/intersect")
        return errors

    if operation.type == "loft":
        params = operation.params
        if not isinstance(params, LoftParams):
            errors.append("loft params must be LoftParams")
            return errors
        if len(params.profile_ids) < 2:
            errors.append("loft requires at least two profile/sketch ids")
        if not all(_is_nonempty_string(item) for item in params.profile_ids):
            errors.append("loft profile_ids must be non-empty strings")
        if params.operation not in {"new", "join", "cut"}:
            errors.append("loft operation must be one of new/join/cut")
        return errors

    if operation.type == "fillet":
        params = operation.params
        if not isinstance(params, FilletParams):
            errors.append("fillet params must be FilletParams")
            return errors
        if not _has_refs(params.edge_refs):
            errors.append("fillet requires at least one edge ref")
        _positive(params.radius, "fillet radius", errors)
        return errors

    if operation.type == "chamfer":
        params = operation.params
        if not isinstance(params, ChamferParams):
            errors.append("chamfer params must be ChamferParams")
            return errors
        if not _has_refs(params.edge_refs):
            errors.append("chamfer requires at least one edge ref")
        _positive(params.distance, "chamfer distance", errors)
        return errors

    if operation.type == "shell":
        params = operation.params
        if not isinstance(params, ShellParams):
            errors.append("shell params must be ShellParams")
            return errors
        if params.mode not in {"open", "closed"}:
            errors.append("shell mode must be open or closed")
        if params.mode == "open" and not _has_refs(params.face_refs):
            errors.append("shell open mode requires face refs")
        if params.mode == "closed" and not _has_refs(params.body_refs):
            errors.append("shell closed mode requires body refs")
        if params.mode == "open" and params.body_refs:
            errors.append("shell open mode must not include body refs")
        if params.mode == "closed" and params.face_refs:
            errors.append("shell closed mode must not include face refs")
        _nonnegative(params.inside_thickness, "shell inside_thickness", errors)
        _nonnegative(params.outside_thickness, "shell outside_thickness", errors)
        if params.inside_thickness == 0 and params.outside_thickness == 0:
            errors.append("shell requires inside_thickness or outside_thickness > 0")
        if params.shell_type not in {"sharp", "rounded"}:
            errors.append("shell shell_type must be sharp or rounded")
        return errors

    if operation.type == "create_simple_hole":
        params = operation.params
        if not isinstance(params, SimpleHoleParams):
            errors.append("create_simple_hole params must be SimpleHoleParams")
            return errors
        if not _is_nonempty_string(params.face_ref):
            errors.append("create_simple_hole face_ref must be provided")
        _check_point(params.center, "create_simple_hole center", 3, errors)
        _positive(params.diameter, "create_simple_hole diameter", errors)
        if params.extent_type not in {"through_all", "distance"}:
            errors.append("create_simple_hole extent_type must be through_all or distance")
        if params.extent_type == "distance":
            if params.depth is None:
                errors.append("create_simple_hole distance extent requires depth")
            else:
                _positive(params.depth, "create_simple_hole depth", errors)
        return errors

    if operation.type == "create_counterbore_hole":
        params = operation.params
        if not isinstance(params, CounterboreHoleParams):
            errors.append("create_counterbore_hole params must be CounterboreHoleParams")
            return errors
        if not _is_nonempty_string(params.face_ref):
            errors.append("create_counterbore_hole face_ref must be provided")
        _check_point(params.center, "create_counterbore_hole center", 3, errors)
        _positive(params.hole_diameter, "create_counterbore_hole hole_diameter", errors)
        _positive(params.hole_depth, "create_counterbore_hole hole_depth", errors)
        _positive(params.counterbore_diameter, "create_counterbore_hole counterbore_diameter", errors)
        _positive(params.counterbore_depth, "create_counterbore_hole counterbore_depth", errors)
        if (
            _check_finite_number(params.counterbore_diameter, "create_counterbore_hole counterbore_diameter", [])
            and _check_finite_number(params.hole_diameter, "create_counterbore_hole hole_diameter", [])
            and params.counterbore_diameter <= params.hole_diameter
        ):
            errors.append("create_counterbore_hole counterbore_diameter must be larger than hole_diameter")
        return errors

    if operation.type == "create_tapped_hole":
        params = operation.params
        if not isinstance(params, TappedHoleParams):
            errors.append("create_tapped_hole params must be TappedHoleParams")
            return errors
        if not _is_nonempty_string(params.face_ref):
            errors.append("create_tapped_hole face_ref must be provided")
        _check_point(params.center, "create_tapped_hole center", 3, errors)
        if params.thread_type not in {"metric", "unc", "unf"}:
            errors.append("create_tapped_hole thread_type must be metric/unc/unf")
        if not _is_nonempty_string(params.thread_size):
            errors.append("create_tapped_hole thread_size must be provided")
        else:
            try:
                validate_thread_spec(params.thread_type, params.thread_size)
            except ValueError as exc:
                errors.append(f"create_tapped_hole thread spec invalid: {exc}")
        _positive(params.thread_depth, "create_tapped_hole thread_depth", errors)
        if params.pilot_hole_depth is not None:
            pilot_is_finite = _check_finite_number(
                params.pilot_hole_depth,
                "create_tapped_hole pilot_hole_depth",
                errors,
            )
            thread_depth_is_finite = _check_finite_number(
                params.thread_depth,
                "create_tapped_hole thread_depth",
                [],
            )
            if pilot_is_finite and thread_depth_is_finite and params.pilot_hole_depth < params.thread_depth:
                errors.append("create_tapped_hole pilot_hole_depth must be >= thread_depth")
        return errors

    if operation.type == "create_external_thread":
        params = operation.params
        if not isinstance(params, ExternalThreadParams):
            errors.append("create_external_thread params must be ExternalThreadParams")
            return errors
        if not _is_nonempty_string(params.face_ref):
            errors.append("create_external_thread face_ref must be provided")
        if params.thread_type not in {"metric", "unc", "unf"}:
            errors.append("create_external_thread thread_type must be metric/unc/unf")
        if not _is_nonempty_string(params.thread_size):
            errors.append("create_external_thread thread_size must be provided")
        else:
            try:
                validate_thread_spec(params.thread_type, params.thread_size)
            except ValueError as exc:
                errors.append(f"create_external_thread thread spec invalid: {exc}")
        if not params.is_full_length:
            if params.thread_length is None:
                errors.append("create_external_thread partial thread requires thread_length")
            else:
                _positive(params.thread_length, "create_external_thread thread_length", errors)
        _nonnegative(params.thread_offset, "create_external_thread thread_offset", errors)
        return errors

    if operation.type == "pattern_feature":
        params = operation.params
        if not isinstance(params, PatternFeatureParams):
            errors.append("pattern_feature params must be PatternFeatureParams")
            return errors
        if params.pattern_type not in {"rectangular", "circular"}:
            errors.append("pattern_feature pattern_type must be rectangular or circular")
        if not _has_refs(params.feature_refs):
            errors.append("pattern_feature requires at least one feature ref")
        if params.count_x is not None:
            if _check_finite_number(params.count_x, "pattern_feature count_x", errors) and params.count_x < 2:
                errors.append("pattern_feature count_x must be >= 2")
        if params.count_y is not None:
            if _check_finite_number(params.count_y, "pattern_feature count_y", errors) and params.count_y < 2:
                errors.append("pattern_feature count_y must be >= 2")
        if params.spacing_x is not None:
            _positive(params.spacing_x, "pattern_feature spacing_x", errors)
        if params.spacing_y is not None:
            _positive(params.spacing_y, "pattern_feature spacing_y", errors)
        if params.rotation_count is not None:
            if (
                _check_finite_number(params.rotation_count, "pattern_feature rotation_count", errors)
                and params.rotation_count < 2
            ):
                errors.append("pattern_feature rotation_count must be >= 2")
        if params.rotation_angle_degrees is not None:
            _positive(params.rotation_angle_degrees, "pattern_feature rotation_angle_degrees", errors)
        return errors

    if operation.type == "list_features":
        if not isinstance(operation.params, ListFeaturesParams):
            errors.append("list_features params must be ListFeaturesParams")
        return errors

    if operation.type == "delete_feature":
        params = operation.params
        if not isinstance(params, DeleteFeatureParams):
            errors.append("delete_feature params must be DeleteFeatureParams")
            return errors
        if not _is_nonempty_string(params.feature_ref):
            errors.append("delete_feature feature_ref must be provided")
        if params.expected_timeline_index is not None:
            if (
                _check_finite_number(params.expected_timeline_index, "delete_feature expected_timeline_index", errors)
                and params.expected_timeline_index < 0
            ):
                errors.append("delete_feature expected_timeline_index must be >= 0")
        return errors

    if operation.type == "set_feature_suppression":
        params = operation.params
        if not isinstance(params, FeatureSuppressionParams):
            errors.append("set_feature_suppression params must be FeatureSuppressionParams")
            return errors
        if not _is_nonempty_string(params.feature_ref):
            errors.append("set_feature_suppression feature_ref must be provided")
        if not isinstance(params.suppress, bool):
            errors.append("set_feature_suppression suppress must be a boolean")
        if params.expected_timeline_index is not None:
            if (
                _check_finite_number(
                    params.expected_timeline_index,
                    "set_feature_suppression expected_timeline_index",
                    errors,
                )
                and params.expected_timeline_index < 0
            ):
                errors.append("set_feature_suppression expected_timeline_index must be >= 0")
        return errors

    if operation.type == "jump_to_timeline_position":
        params = operation.params
        if not isinstance(params, JumpToTimelinePositionParams):
            errors.append("jump_to_timeline_position params must be JumpToTimelinePositionParams")
            return errors
        _nonnegative(params.target_index, "jump_to_timeline_position target_index", errors)
        return errors

    if operation.type == "select_entities":
        params = operation.params
        if not isinstance(params, SelectEntitiesParams):
            errors.append("select_entities params must be SelectEntitiesParams")
            return errors
        if params.kind not in {"edge", "face", "body"}:
            errors.append("select_entities kind must be edge/face/body")
        if not _has_refs(params.refs):
            errors.append("select_entities requires at least one ref")
        return errors

    if operation.type == "clear_selection":
        params = operation.params
        if not isinstance(params, ClearSelectionParams):
            errors.append("clear_selection params must be ClearSelectionParams")
            return errors
        if params.kind not in {"edge", "face", "body"}:
            errors.append("clear_selection kind must be edge/face/body")
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

    if operation.type in {"add_rectangle", "add_circle", "add_line", "add_arc", "list_sketch_profiles"}:
        sketch = getattr(operation.params, "sketch", "")
        if sketch and sketch not in committed_sketches:
            errors.append(f"Referenced sketch '{sketch}' does not exist yet")
        elif operation.type == "list_sketch_profiles" and sketch and not _has_committed_sketch_geometry_for_profile_inspection(
            committed_operations,
            sketch,
        ):
            errors.append(
                f"list_sketch_profiles for sketch '{sketch}' requires committed sketch geometry "
                "(add_rectangle/add_circle/add_line/add_arc) before profile inspection"
            )

    if operation.type in {"extrude", "revolve"} and isinstance(operation.params, (ExtrudeParams, RevolveParams)):
        sketch = (operation.params.sketch or "").strip()
        if sketch and sketch not in committed_sketches:
            errors.append(f"Referenced sketch '{sketch}' does not exist yet")
        elif sketch and not _has_committed_profile_for_sketch(committed_operations, sketch):
            errors.append(
                f"{operation.type} for sketch '{sketch}' requires at least one committed profile operation "
                "(add_rectangle/add_circle or successful list_sketch_profiles) before feature creation"
            )

    if operation.type == "loft" and isinstance(operation.params, LoftParams):
        for sketch in operation.params.profile_ids:
            if sketch not in committed_sketches:
                errors.append(f"Referenced sketch '{sketch}' does not exist yet")
            elif not _has_committed_profile_for_sketch(committed_operations, sketch):
                errors.append(
                    f"Loft for sketch '{sketch}' requires at least one committed profile operation "
                    "(add_rectangle/add_circle or successful list_sketch_profiles)"
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
