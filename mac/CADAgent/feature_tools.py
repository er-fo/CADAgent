"""
Feature operation utilities for CADAgent.

Provides helpers to apply fillets and chamfers to edges in the active Fusion 360
design using entity tokens. All measurements use the design's default length units
to maintain consistency with the user's working context.
"""

from __future__ import annotations

import datetime
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import adsk.core
import adsk.fusion

from . import thread_database

logger = logging.getLogger(__name__)

_VALID_DIAMETER_UNITS = {"mm", "cm", "m", "in"}
_VALID_THICKNESS_UNITS = {"mm", "cm", "m", "in"}
_MM_PER_UNIT = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
}
_MM_TO_CM = 0.1
_CM_TO_MM = 10.0
_HOLE_PLANE_TOL_CM = 0.02  # 0.2mm
_HOLE_BBOX_TOL_CM = 0.02   # 0.2mm


class FeatureOperationError(RuntimeError):
    """Raised when a feature operation cannot be fulfilled."""


def _mm_to_cm(value_mm: float) -> float:
    return float(value_mm) * _MM_TO_CM


def _cm_to_mm(value_cm: float) -> float:
    return float(value_cm) * _CM_TO_MM


def _point_from_mm(x_mm: float, y_mm: float, z_mm: float) -> adsk.core.Point3D:
    return adsk.core.Point3D.create(_mm_to_cm(x_mm), _mm_to_cm(y_mm), _mm_to_cm(z_mm))


def _point_to_mm_dict(point_cm: adsk.core.Point3D) -> Dict[str, float]:
    return {
        "x": round(_cm_to_mm(float(getattr(point_cm, "x", 0.0))), 6),
        "y": round(_cm_to_mm(float(getattr(point_cm, "y", 0.0))), 6),
        "z": round(_cm_to_mm(float(getattr(point_cm, "z", 0.0))), 6),
    }


def _is_missing_target_body_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "no target body found to cut or intersect" in message


def _is_logical_selection_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "logicalselection" in message


def _diameter_to_cm(diameter: float, unit: str) -> float:
    unit_norm = str(unit or "mm").strip().lower()
    if unit_norm not in _MM_PER_UNIT:
        raise FeatureOperationError(
            f"Invalid diameter unit '{unit}'. Must be one of {sorted(_VALID_DIAMETER_UNITS)}."
        )
    diameter_mm = float(diameter) * _MM_PER_UNIT[unit_norm]
    return _mm_to_cm(diameter_mm)


def _fallback_cut_distance_cm(face: adsk.fusion.BRepFace) -> float:
    body = getattr(face, "body", None)
    bbox = getattr(body, "boundingBox", None) if body else None
    if bbox and getattr(bbox, "minPoint", None) and getattr(bbox, "maxPoint", None):
        min_pt = bbox.minPoint
        max_pt = bbox.maxPoint
        span_x = abs(float(max_pt.x) - float(min_pt.x))
        span_y = abs(float(max_pt.y) - float(min_pt.y))
        span_z = abs(float(max_pt.z) - float(min_pt.z))
        span = max(span_x, span_y, span_z)
        return max(span * 2.0, 1.0)
    # Conservative default in Fusion internal units (cm)
    return 10.0


def _create_hole_cut_fallback(
    component: adsk.fusion.Component,
    sketch: adsk.fusion.Sketch,
    sketch_center: adsk.core.Point3D,
    diameter: float,
    diameter_unit: str,
    face: adsk.fusion.BRepFace,
    preferred_direction: adsk.fusion.ExtentDirections,
    feature_name: str = "",
) -> Tuple[adsk.fusion.ExtrudeFeature, adsk.fusion.ExtentDirections, float]:
    """Fallback for hole API selection failures: sketch circle + cut extrude."""
    radius_cm = _diameter_to_cm(diameter, diameter_unit) * 0.5
    sketch.sketchCurves.sketchCircles.addByCenterRadius(sketch_center, radius_cm)
    if sketch.profiles.count <= 0:
        raise FeatureOperationError("Sketch-cut fallback failed: no profile was created.")

    profile = sketch.profiles.item(sketch.profiles.count - 1)
    extrudes = component.features.extrudeFeatures
    cut_distance_cm = _fallback_cut_distance_cm(face)

    preferred_sign = 1.0 if preferred_direction == adsk.fusion.ExtentDirections.PositiveExtentDirection else -1.0
    attempts = [preferred_sign * cut_distance_cm, -preferred_sign * cut_distance_cm]

    last_exception: Optional[Exception] = None
    for distance_cm in attempts:
        try:
            ext_input = extrudes.createInput(profile, adsk.fusion.FeatureOperations.CutFeatureOperation)
            ext_input.setDistanceExtent(False, adsk.core.ValueInput.createByReal(distance_cm))
            feature = extrudes.add(ext_input)
            if feature_name.strip() and getattr(feature, "timelineObject", None):
                feature.timelineObject.name = f"{feature_name.strip()} (Sketch Cut)"
            direction = (
                adsk.fusion.ExtentDirections.PositiveExtentDirection
                if distance_cm >= 0
                else adsk.fusion.ExtentDirections.NegativeExtentDirection
            )
            return feature, direction, abs(distance_cm)
        except Exception as exc:
            last_exception = exc
            continue

    raise FeatureOperationError(
        "Sketch-cut fallback failed for both cut directions."
        + (f" Last error: {last_exception}" if last_exception else "")
    )


def _face_center_point_cm(face: adsk.fusion.BRepFace) -> Optional[adsk.core.Point3D]:
    try:
        centroid = getattr(face, "centroid", None)
        if centroid:
            return centroid
    except Exception:
        pass

    try:
        point_on_face = getattr(face, "pointOnFace", None)
        if point_on_face:
            return point_on_face
    except Exception:
        pass

    bbox = getattr(face, "boundingBox", None)
    if bbox and getattr(bbox, "minPoint", None) and getattr(bbox, "maxPoint", None):
        min_pt = bbox.minPoint
        max_pt = bbox.maxPoint
        return adsk.core.Point3D.create(
            (float(min_pt.x) + float(max_pt.x)) * 0.5,
            (float(min_pt.y) + float(max_pt.y)) * 0.5,
            (float(min_pt.z) + float(max_pt.z)) * 0.5,
        )
    return None


def _body_center_point_cm(face: adsk.fusion.BRepFace) -> Optional[adsk.core.Point3D]:
    body = getattr(face, "body", None)
    if not body:
        return None
    bbox = getattr(body, "boundingBox", None)
    if bbox and getattr(bbox, "minPoint", None) and getattr(bbox, "maxPoint", None):
        min_pt = bbox.minPoint
        max_pt = bbox.maxPoint
        return adsk.core.Point3D.create(
            (float(min_pt.x) + float(max_pt.x)) * 0.5,
            (float(min_pt.y) + float(max_pt.y)) * 0.5,
            (float(min_pt.z) + float(max_pt.z)) * 0.5,
        )
    return None


def _face_normal_vector(face: adsk.fusion.BRepFace) -> Optional[Tuple[float, float, float]]:
    try:
        evaluator = face.evaluator
        point_on_face = getattr(face, "pointOnFace", None)
        if evaluator and point_on_face:
            ok, normal = evaluator.getNormalAtPoint(point_on_face)
            if ok and normal:
                return (float(normal.x), float(normal.y), float(normal.z))
    except Exception:
        pass

    try:
        plane = adsk.core.Plane.cast(getattr(face, "geometry", None))
        if plane and plane.normal:
            n = plane.normal
            return (float(n.x), float(n.y), float(n.z))
    except Exception:
        pass

    return None


def _validate_hole_center_on_face(
    face: adsk.fusion.BRepFace,
    center_point_cm: adsk.core.Point3D,
    center_xyz_mm: Tuple[float, float, float],
) -> None:
    center_x_mm, center_y_mm, center_z_mm = center_xyz_mm

    bbox = getattr(face, "boundingBox", None)
    if bbox and getattr(bbox, "minPoint", None) and getattr(bbox, "maxPoint", None):
        min_pt = bbox.minPoint
        max_pt = bbox.maxPoint
        outside_axes: List[str] = []
        if center_point_cm.x < (min_pt.x - _HOLE_BBOX_TOL_CM) or center_point_cm.x > (max_pt.x + _HOLE_BBOX_TOL_CM):
            outside_axes.append("x")
        if center_point_cm.y < (min_pt.y - _HOLE_BBOX_TOL_CM) or center_point_cm.y > (max_pt.y + _HOLE_BBOX_TOL_CM):
            outside_axes.append("y")
        if center_point_cm.z < (min_pt.z - _HOLE_BBOX_TOL_CM) or center_point_cm.z > (max_pt.z + _HOLE_BBOX_TOL_CM):
            outside_axes.append("z")

        if outside_axes:
            bbox_min_mm = _point_to_mm_dict(min_pt)
            bbox_max_mm = _point_to_mm_dict(max_pt)
            raise FeatureOperationError(
                "Hole center is outside target face bounds. "
                f"face_bbox_mm={bbox_min_mm}->{bbox_max_mm}, "
                f"center_mm=({center_x_mm:.3f}, {center_y_mm:.3f}, {center_z_mm:.3f}). "
                "Hole center inputs are interpreted as millimeters."
            )

    plane = adsk.core.Plane.cast(getattr(face, "geometry", None))
    if plane and plane.normal and plane.origin:
        nx = float(plane.normal.x)
        ny = float(plane.normal.y)
        nz = float(plane.normal.z)
        normal_len = math.sqrt(nx * nx + ny * ny + nz * nz)
        if normal_len > 1e-9:
            dx = float(center_point_cm.x) - float(plane.origin.x)
            dy = float(center_point_cm.y) - float(plane.origin.y)
            dz = float(center_point_cm.z) - float(plane.origin.z)
            distance_cm = abs((dx * nx + dy * ny + dz * nz) / normal_len)
            if distance_cm > _HOLE_PLANE_TOL_CM:
                raise FeatureOperationError(
                    "Hole center is not on the selected face plane "
                    f"(distance={distance_cm * 10.0:.3f}mm). "
                    f"Provided center_mm=({center_x_mm:.3f}, {center_y_mm:.3f}, {center_z_mm:.3f})."
                )


def _preferred_through_all_direction(face: adsk.fusion.BRepFace) -> adsk.fusion.ExtentDirections:
    face_normal = _face_normal_vector(face)
    face_center = _face_center_point_cm(face)
    body_center = _body_center_point_cm(face)

    if not face_normal or not face_center or not body_center:
        return adsk.fusion.ExtentDirections.PositiveExtentDirection

    vx = float(body_center.x) - float(face_center.x)
    vy = float(body_center.y) - float(face_center.y)
    vz = float(body_center.z) - float(face_center.z)
    dot = vx * face_normal[0] + vy * face_normal[1] + vz * face_normal[2]
    if dot >= 0:
        return adsk.fusion.ExtentDirections.PositiveExtentDirection
    return adsk.fusion.ExtentDirections.NegativeExtentDirection


def _through_direction_label(direction: adsk.fusion.ExtentDirections) -> str:
    if direction == adsk.fusion.ExtentDirections.NegativeExtentDirection:
        return "negative"
    return "positive"


def capture_feature_snapshot(design: Optional[adsk.fusion.Design], max_features: int = 25) -> Dict[str, Any]:
    """Capture recent timeline feature metadata for LLM context."""
    if design is None:
        return {
            "success": False,
            "error": "No active design available for feature snapshot.",
        }

    # Validate design type - timeline operations require Parametric mode
    try:
        if design.designType != adsk.fusion.DesignTypes.ParametricDesignType:
            return {
                "success": False,
                "error": "Timeline operations require Parametric design mode. Current mode is Direct Modeling.",
                "design_type": "DirectModelingDesignType",
            }
    except Exception as exc:
        logger.warning("Unable to check design type: %s", exc)
        # Continue anyway - let the timeline access fail with a more specific error if needed

    try:
        timeline = design.timeline
    except Exception as exc:
        logger.warning("Unable to access design timeline: %s", exc)
        return {
            "success": False,
            "error": f"Unable to access design timeline: {exc}",
        }

    try:
        marker_position = timeline.markerPosition
        timeline_count = timeline.count
    except Exception as exc:
        logger.warning("Failed to read timeline metadata: %s", exc)
        return {
            "success": False,
            "error": f"Unable to read timeline metadata: {exc}",
        }

    units_manager = getattr(design, "unitsManager", None)
    length_unit: Optional[str] = None
    area_unit: Optional[str] = None
    volume_unit: Optional[str] = None
    mass_unit: Optional[str] = None

    if units_manager:
        try:
            length_unit = getattr(units_manager, "defaultLengthUnits", None)
        except Exception:
            length_unit = None
        try:
            mass_unit = getattr(units_manager, "defaultMassUnits", None)
        except Exception:
            mass_unit = None

    if length_unit:
        area_unit = f"{length_unit}^2"
        volume_unit = f"{length_unit}^3"

    utc_now = datetime.datetime.utcnow()

    feature_limit = max(1, min(int(max_features or 25), 100))
    features: List[Dict[str, Any]] = []
    seen_tokens: Set[str] = set()

    for index in range(timeline_count - 1, -1, -1):
        try:
            item = timeline.item(index)
        except Exception as exc:
            logger.debug("Failed to access timeline item %d: %s", index, exc)
            continue

        feature_data = _serialize_feature_from_timeline_item(
            item,
            index,
            seen_tokens,
            length_unit,
            area_unit,
            volume_unit,
            mass_unit,
        )

        if feature_data:
            features.append(feature_data)

        if len(features) >= feature_limit:
            break

    snapshot = {
        "success": True,
        "features": features,
        "timeline_count": timeline_count,
        "marker_position": marker_position,
        "captured_at": utc_now.isoformat(timespec="seconds") + "Z",
        "captured_at_label": utc_now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "length_unit": length_unit,
        "area_unit": area_unit,
        "volume_unit": volume_unit,
        "mass_unit": mass_unit,
    }

    # Include sketch snapshot
    sketch_snapshot = capture_sketch_snapshot(design)
    if sketch_snapshot.get("success"):
        snapshot["sketches"] = sketch_snapshot.get("sketches", [])
        snapshot["sketch_count"] = sketch_snapshot.get("count", 0)

    return snapshot


def capture_sketch_snapshot(design: Optional[adsk.fusion.Design]) -> Dict[str, Any]:
    """Capture all sketches in the design for LLM context."""
    if design is None:
        return {"success": False, "error": "No active design available."}

    try:
        root_comp = design.rootComponent
        sketches_collection = root_comp.sketches
        sketch_count = sketches_collection.count
    except Exception as exc:
        logger.warning("Unable to access sketches: %s", exc)
        return {"success": False, "error": f"Unable to access sketches: {exc}"}

    sketches: List[Dict[str, Any]] = []

    for i in range(sketch_count):
        try:
            sketch = sketches_collection.item(i)
            sketch_name = getattr(sketch, "name", f"Sketch{i}")

            # Get plane information
            ref_plane = getattr(sketch, "referencePlane", None)
            plane_name = "Unknown"
            if ref_plane:
                plane_name = getattr(ref_plane, "name", "Custom Plane")

            # Get profile count
            profiles = getattr(sketch, "profiles", None)
            profile_count = profiles.count if profiles else 0

            sketches.append({
                "name": sketch_name,
                "plane": plane_name,
                "profile_count": profile_count,
                "index": i
            })
        except Exception as exc:
            logger.debug("Failed to access sketch %d: %s", i, exc)
            continue

    return {
        "success": True,
        "sketches": sketches,
        "count": len(sketches)
    }


def _serialize_feature_from_timeline_item(
    timeline_item: Any,
    fallback_index: int,
    seen_tokens: Set[str],
    length_unit: Optional[str],
    area_unit: Optional[str],
    volume_unit: Optional[str],
    mass_unit: Optional[str],
) -> Optional[Dict[str, Any]]:
    entity = getattr(timeline_item, "entity", None)
    if entity is None:
        return None

    feature = adsk.fusion.Feature.cast(entity)
    if not feature:
        return None

    token = getattr(feature, "entityToken", None)
    normalized_token = token.strip() if isinstance(token, str) else None

    if normalized_token:
        if normalized_token in seen_tokens:
            return None
        seen_tokens.add(normalized_token)

    feature_type = _normalize_object_type(getattr(feature, "objectType", None))
    timeline_object = getattr(feature, "timelineObject", None)
    feature_index = fallback_index
    if timeline_object is not None:
        try:
            feature_index = int(getattr(timeline_object, "index", fallback_index))
        except Exception:
            feature_index = fallback_index

    feature_info: Dict[str, Any] = {
        "timeline_index": feature_index,
        "feature_type": feature_type,
        "object_type": getattr(feature, "objectType", None),
        "name": getattr(feature, "name", ""),
        "timeline_name": getattr(timeline_item, "name", ""),
        "entity_token": normalized_token,
        "is_suppressed": getattr(feature, "isSuppressed", None),
    }

    parent_component = getattr(feature, "parentComponent", None)
    feature_info["component"] = _component_path(parent_component)

    bodies = _serialize_feature_bodies(
        feature,
        length_unit=length_unit,
        area_unit=area_unit,
        volume_unit=volume_unit,
        mass_unit=mass_unit,
    )
    feature_info["bodies"] = bodies
    feature_info["body_count"] = len(bodies)

    if bodies:
        feature_info["body_tokens"] = [body.get("entity_token") for body in bodies if body.get("entity_token")]
        feature_info["body_names"] = [body.get("name") for body in bodies if body.get("name")]

    health_state = getattr(feature, "healthState", None)
    if health_state is not None:
        try:
            feature_info["health_state"] = int(health_state)
        except Exception:
            feature_info["health_state"] = str(health_state)

    instance_count = _infer_feature_instance_count(feature)
    if instance_count is not None:
        feature_info["instance_count"] = instance_count

    # Attach type-specific details when cheap and reliable to extract.
    # This is especially important for HoleFeatures: the backend prompt needs hole
    # centers/diameters to allow "copy existing holes" workflows without asking users.
    if feature_type == "HoleFeature":
        hole_details = _serialize_hole_feature_details(feature)
        if hole_details:
            feature_info["hole"] = hole_details

    editable = _serialize_editable_feature_parameters(feature, feature_type)
    if editable:
        feature_info["editable_parameters"] = editable

    return feature_info


def _serialize_editable_feature_parameters(feature: adsk.fusion.Feature, feature_type: str) -> Optional[Dict[str, Any]]:
    """Expose narrow safe edit metadata for supported timeline features."""
    supported: List[str] = ["name"]
    parameters: Dict[str, Any] = {}

    if feature_type == "ExtrudeFeature":
        try:
            extrude = adsk.fusion.ExtrudeFeature.cast(feature)
        except Exception:
            extrude = None
        distance_param = _get_extrude_distance_parameter(extrude) if extrude else None
        if distance_param is not None:
            supported.append("distance")
            parameters["distance"] = _parameter_snapshot(distance_param)

    elif feature_type == "HoleFeature":
        try:
            hole = adsk.fusion.HoleFeature.cast(feature)
        except Exception:
            hole = None
        if hole:
            diameter_param = _get_hole_diameter_parameter(hole)
            depth_param = _get_hole_depth_parameter(hole)
            if diameter_param is not None:
                supported.append("diameter")
                parameters["diameter"] = _parameter_snapshot(diameter_param)
            if depth_param is not None:
                supported.append("depth")
                parameters["depth"] = _parameter_snapshot(depth_param)

    else:
        return None

    return {
        "supported": True,
        "tool": "adjust_feature_parameters",
        "supported_feature_type": feature_type,
        "supported_parameters": supported,
        "parameters": parameters,
        "safety": {
            "feature_token_required": True,
            "expected_name_recommended": True,
            "expected_timeline_index_recommended": True,
        },
    }


def _parameter_snapshot(parameter: Any) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    try:
        expression = getattr(parameter, "expression", None)
        if expression:
            snapshot["expression"] = str(expression)
    except Exception:
        pass
    try:
        value_cm = float(getattr(parameter, "value", None))
        snapshot["value_cm"] = _safe_round(value_cm, 6)
        snapshot["value_mm"] = _safe_round(value_cm * 10.0, 6)
    except Exception:
        pass
    return snapshot


def _get_extrude_distance_parameter(extrude: Any) -> Any:
    if not extrude:
        return None
    for extent_attr in ("extentOne", "extentTwo"):
        try:
            extent = getattr(extrude, extent_attr, None)
        except Exception:
            extent = None
        parameter = _get_distance_parameter_from_extent(extent)
        if parameter is not None:
            return parameter
    return None


def _get_distance_parameter_from_extent(extent: Any) -> Any:
    if extent is None:
        return None
    for attr in ("distance", "distanceOne", "distanceTwo"):
        try:
            parameter = getattr(extent, attr, None)
        except Exception:
            parameter = None
        if parameter is not None and (
            hasattr(parameter, "expression") or hasattr(parameter, "value")
        ):
            return parameter
    return None


def _get_hole_diameter_parameter(hole: Any) -> Any:
    for attr in ("holeDiameter", "diameter"):
        try:
            parameter = getattr(hole, attr, None)
        except Exception:
            parameter = None
        if parameter is not None and (
            hasattr(parameter, "expression") or hasattr(parameter, "value")
        ):
            return parameter
    return None


def _get_hole_depth_parameter(hole: Any) -> Any:
    for attr in ("holeDepth", "depth"):
        try:
            parameter = getattr(hole, attr, None)
        except Exception:
            parameter = None
        if parameter is not None and (
            hasattr(parameter, "expression") or hasattr(parameter, "value")
        ):
            return parameter

    for extent_attr in ("extentDefinition", "extentOne"):
        try:
            extent = getattr(hole, extent_attr, None)
        except Exception:
            extent = None
        parameter = _get_distance_parameter_from_extent(extent)
        if parameter is not None:
            return parameter
    return None


def _unit_to_cm(value: float, unit: str) -> float:
    unit_norm = (unit or "mm").strip().lower()
    if unit_norm not in _MM_PER_UNIT:
        raise FeatureOperationError(f"Unsupported unit '{unit}'.")
    return float(value) * _MM_PER_UNIT[unit_norm] * _MM_TO_CM


def _set_length_parameter(parameter: Any, value: float, unit: str, label: str) -> str:
    expression = f"{float(value):g} {unit or 'mm'}"
    try:
        if hasattr(parameter, "expression"):
            parameter.expression = expression
            return expression
    except Exception as exc:
        logger.debug("Failed to set %s expression: %s", label, exc)

    try:
        if hasattr(parameter, "value"):
            parameter.value = _unit_to_cm(float(value), unit or "mm")
            return expression
    except Exception as exc:
        raise FeatureOperationError(f"Unable to set {label}: {exc}") from exc

    raise FeatureOperationError(f"Feature does not expose an editable {label} parameter.")


def _resolve_single_feature_by_token(design: adsk.fusion.Design, feature_token: str) -> adsk.fusion.Feature:
    entities = design.findEntityByToken(feature_token)
    if not entities:
        raise FeatureOperationError(
            "Feature token not found in design. Call list_features for the latest feature snapshot/tokens."
        )
    if len(entities) > 1:
        raise FeatureOperationError(
            f"Feature token resolved to {len(entities)} entities; refusing to edit an ambiguous target."
        )
    feature = adsk.fusion.Feature.cast(entities[0])
    if not feature:
        obj_type = getattr(entities[0], "objectType", type(entities[0]).__name__)
        raise FeatureOperationError(f"Resolved entity is not a timeline feature: {obj_type}.")
    return feature


def adjust_feature_parameters(
    app: adsk.core.Application,
    feature_token: str,
    parameters: Dict[str, Any],
    expected_name: str = "",
    expected_timeline_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Safely adjust a narrow set of editable feature parameters."""
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise FeatureOperationError("No active Fusion design found.")
    if design.designType != adsk.fusion.DesignTypes.ParametricDesignType:
        raise FeatureOperationError("Feature parameter edits require Parametric design mode.")

    feature = _resolve_single_feature_by_token(design, str(feature_token or "").strip())
    feature_type = _normalize_object_type(getattr(feature, "objectType", None))
    timeline_object = getattr(feature, "timelineObject", None)
    timeline_index = getattr(timeline_object, "index", None) if timeline_object else None
    timeline_name = getattr(timeline_object, "name", "") if timeline_object else ""
    feature_name = getattr(feature, "name", "") or ""

    expected_name = str(expected_name or "").strip()
    if expected_name and expected_name not in {timeline_name, feature_name}:
        raise FeatureOperationError(
            f"Safety check failed: expected feature name '{expected_name}' but found "
            f"timeline_name='{timeline_name}', entity_name='{feature_name}'."
        )

    if expected_timeline_index is not None:
        try:
            expected_index_int = int(expected_timeline_index)
        except (TypeError, ValueError):
            raise FeatureOperationError("expected_timeline_index must be an integer.")
        if timeline_index != expected_index_int:
            raise FeatureOperationError(
                f"Safety check failed: expected timeline index {expected_index_int} but found {timeline_index}."
            )

    if not isinstance(parameters, dict) or not parameters:
        raise FeatureOperationError("No feature parameters were provided to edit.")

    changed: Dict[str, Any] = {}

    if "name" in parameters:
        new_name = str(parameters.get("name") or "").strip()
        if not new_name:
            raise FeatureOperationError("Feature name must be non-empty.")
        try:
            feature.name = new_name
        except Exception:
            pass
        if timeline_object:
            try:
                timeline_object.name = new_name
            except Exception:
                pass
        changed["name"] = new_name
        feature_name = new_name

    if "distance" in parameters:
        if feature_type != "ExtrudeFeature":
            raise FeatureOperationError("distance edits are supported only for ExtrudeFeature.")
        try:
            extrude = adsk.fusion.ExtrudeFeature.cast(feature)
        except Exception:
            extrude = None
        parameter = _get_extrude_distance_parameter(extrude)
        if parameter is None:
            raise FeatureOperationError("This ExtrudeFeature does not expose an editable distance parameter.")
        unit = str(parameters.get("distance_unit") or "mm").strip().lower()
        changed["distance"] = _set_length_parameter(parameter, float(parameters["distance"]), unit, "extrude distance")

    if "diameter" in parameters or "depth" in parameters:
        if feature_type != "HoleFeature":
            raise FeatureOperationError("diameter/depth edits are supported only for HoleFeature.")
        try:
            hole = adsk.fusion.HoleFeature.cast(feature)
        except Exception:
            hole = None
        if hole is None:
            raise FeatureOperationError("Unable to access HoleFeature APIs for this feature.")

        if "diameter" in parameters:
            parameter = _get_hole_diameter_parameter(hole)
            if parameter is None:
                raise FeatureOperationError("This HoleFeature does not expose an editable diameter parameter.")
            unit = str(parameters.get("diameter_unit") or "mm").strip().lower()
            changed["diameter"] = _set_length_parameter(parameter, float(parameters["diameter"]), unit, "hole diameter")

        if "depth" in parameters:
            parameter = _get_hole_depth_parameter(hole)
            if parameter is None:
                raise FeatureOperationError(
                    "This HoleFeature does not expose an editable depth parameter; through-all holes cannot be depth-edited."
                )
            unit = str(parameters.get("depth_unit") or "mm").strip().lower()
            changed["depth"] = _set_length_parameter(parameter, float(parameters["depth"]), unit, "hole depth")

    if not changed:
        raise FeatureOperationError("No supported feature parameters were changed.")

    try:
        design.computeAll()
    except Exception as exc:
        raise FeatureOperationError(f"Feature parameters changed but recompute failed: {exc}") from exc

    return {
        "success": True,
        "message": f"Adjusted {feature_type}: {feature_name or timeline_name or 'feature'}",
        "feature_type": feature_type,
        "feature_name": feature_name or timeline_name,
        "timeline_index": timeline_index,
        "changed_parameters": changed,
    }


def _serialize_hole_feature_details(feature: adsk.fusion.Feature) -> Optional[Dict[str, Any]]:
    """Best-effort HoleFeature details for LLM use (centers, diameter).

    Notes:
      - Fusion internal units are cm for geometry values exposed on Point3D.
      - We return centers in world cm (x,y,z) for direct reuse with hole tools.
      - HoleFeature APIs differ across Fusion builds, so this function is defensive.
    """
    try:
        hole_feature = adsk.fusion.HoleFeature.cast(feature)
    except Exception:
        hole_feature = None

    if not hole_feature:
        return None

    details: Dict[str, Any] = {}

    # Diameter (best-effort)
    diameter_cm = None
    diameter_expression = None
    for attr in ("holeDiameter", "diameter"):
        try:
            diam_obj = getattr(hole_feature, attr, None)
        except Exception:
            diam_obj = None
        if diam_obj is None:
            continue

        try:
            diameter_expression = getattr(diam_obj, "expression", None)
        except Exception:
            diameter_expression = None

        # ValueInput.value is typically in internal cm.
        try:
            diameter_cm = float(getattr(diam_obj, "value", None))
        except Exception:
            try:
                diameter_cm = float(diam_obj)
            except Exception:
                diameter_cm = None

        if diameter_cm is not None or diameter_expression:
            break

    if diameter_expression:
        details["diameter_expression"] = str(diameter_expression)
    if diameter_cm is not None:
        details["diameter_cm"] = _safe_round(diameter_cm, 6)
        details["diameter_mm"] = _safe_round(diameter_cm * 10.0, 6)

    centers = _extract_hole_centers_world_cm(hole_feature)
    if centers:
        details["centers_world_cm"] = centers
        details["centers_world_mm"] = [
            {
                "x": _safe_round((c.get("x") or 0.0) * 10.0, 6),
                "y": _safe_round((c.get("y") or 0.0) * 10.0, 6),
                "z": _safe_round((c.get("z") or 0.0) * 10.0, 6),
            }
            for c in centers
            if isinstance(c, dict)
        ]
        details["center_count"] = len(centers)

    # Only return if we found something useful.
    return details or None


def _extract_hole_centers_world_cm(hole_feature: Any) -> List[Dict[str, float]]:
    """Extract hole center point(s) in world coordinates (cm)."""
    centers: List[Dict[str, float]] = []

    def _iter_collection(collection: Any) -> List[Any]:
        try:
            count = int(getattr(collection, "count", 0))
        except Exception:
            return []
        items: List[Any] = []
        for idx in range(count):
            try:
                items.append(collection.item(idx))
            except Exception:
                continue
        return items

    def _point3d_to_dict(pt: Any) -> Optional[Dict[str, float]]:
        if pt is None:
            return None
        try:
            x = float(getattr(pt, "x", None))
            y = float(getattr(pt, "y", None))
            z = float(getattr(pt, "z", None))
        except Exception:
            return None
        if any(v is None for v in (x, y, z)):
            return None
        return {"x": round(x, 6), "y": round(y, 6), "z": round(z, 6)}

    def _extract_point(obj: Any) -> Optional[Dict[str, float]]:
        # SketchPoint -> prefer worldGeometry when present.
        try:
            sp = adsk.fusion.SketchPoint.cast(obj)
        except Exception:
            sp = None
        if sp:
            pt = None
            try:
                pt = getattr(sp, "worldGeometry", None)
            except Exception:
                pt = None
            if pt is None:
                try:
                    pt = getattr(sp, "geometry", None)
                except Exception:
                    pt = None
            return _point3d_to_dict(pt)

        # Point3D or other object with x/y/z.
        return _point3d_to_dict(obj)

    # Try a few known-ish access paths. APIs can vary; we keep it defensive.
    candidates: List[Any] = []
    for attr in ("holePositionDefinition", "positionDefinition", "position", "holePositions", "positions"):
        try:
            cand = getattr(hole_feature, attr, None)
        except Exception:
            cand = None
        if cand is not None:
            candidates.append(cand)

    for cand in candidates:
        # Hole position definitions often expose sketchPoint(s) or point(s).
        for sub_attr in ("sketchPoint", "point"):
            try:
                sub = getattr(cand, sub_attr, None)
            except Exception:
                sub = None
            if sub is not None:
                extracted = _extract_point(sub)
                if extracted:
                    centers.append(extracted)

        for sub_attr in ("sketchPoints", "points"):
            try:
                sub = getattr(cand, sub_attr, None)
            except Exception:
                sub = None
            if sub is not None:
                for item in _iter_collection(sub):
                    extracted = _extract_point(item)
                    if extracted:
                        centers.append(extracted)

        # If cand itself is a collection of points.
        for item in _iter_collection(cand):
            extracted = _extract_point(item)
            if extracted:
                centers.append(extracted)

        # Or if cand itself is a point-ish object.
        extracted = _extract_point(cand)
        if extracted:
            centers.append(extracted)

        if centers:
            break

    # Deduplicate (stable order).
    seen = set()
    uniq: List[Dict[str, float]] = []
    for c in centers:
        key = (c.get("x"), c.get("y"), c.get("z"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _serialize_feature_bodies(
    feature: adsk.fusion.Feature,
    *,
    length_unit: Optional[str],
    area_unit: Optional[str],
    volume_unit: Optional[str],
    mass_unit: Optional[str],
) -> List[Dict[str, Any]]:
    bodies_data: List[Dict[str, Any]] = []
    collection = getattr(feature, "bodies", None)
    if not collection:
        return bodies_data

    try:
        count = collection.count
    except Exception:
        return bodies_data

    for idx in range(count):
        try:
            body_obj = collection.item(idx)
        except Exception:
            continue

        body = adsk.fusion.BRepBody.cast(body_obj)
        if not body:
            continue

        body_info = _serialize_body(
            body,
            index=idx,
            length_unit=length_unit,
            area_unit=area_unit,
            volume_unit=volume_unit,
            mass_unit=mass_unit,
        )
        bodies_data.append(body_info)

    return bodies_data


def _serialize_body(
    body: adsk.fusion.BRepBody,
    *,
    index: int,
    length_unit: Optional[str],
    area_unit: Optional[str],
    volume_unit: Optional[str],
    mass_unit: Optional[str],
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "id": f"body_{index}",
        "name": getattr(body, "name", ""),
        "entity_token": getattr(body, "entityToken", None),
        "is_solid": getattr(body, "isSolid", None),
    }

    parent_component = getattr(body, "parentComponent", None)
    info["component"] = _component_path(parent_component)

    info["volume"] = _safe_round(getattr(body, "volume", None), 6)
    info["area"] = _safe_round(getattr(body, "area", None), 6)
    info["edge_count"] = getattr(body, "edgeCount", None)
    info["face_count"] = getattr(body, "faceCount", None)

    if area_unit:
        info["area_units"] = area_unit
    if volume_unit:
        info["volume_units"] = volume_unit

    bbox = getattr(body, "boundingBox", None)
    bbox_data = _serialize_bounding_box(bbox)
    if bbox_data:
        info["bounding_box"] = bbox_data

    center_of_mass = _extract_center_of_mass(body)
    if center_of_mass:
        info["center_of_mass"] = center_of_mass

    mass = _extract_body_mass(body)
    if mass is not None:
        info["mass"] = mass
        if mass_unit:
            info["mass_units"] = mass_unit

    return info


def _extract_body_mass(body: adsk.fusion.BRepBody) -> Optional[float]:
    try:
        props = body.physicalProperties
        if props:
            return _safe_round(getattr(props, "mass", None), 6)
    except Exception:
        return None
    return None


def _safe_round(value: Any, digits: int = 4) -> Optional[float]:
    try:
        if value is None:
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _normalize_object_type(object_type: Optional[str]) -> str:
    if not object_type:
        return "Feature"
    if "::" in object_type:
        return object_type.split("::")[-1]
    return object_type


def _component_path(component: Any) -> Optional[str]:
    if component is None:
        return None

    for attr in ("fullPathName", "name"):
        try:
            value = getattr(component, attr, None)
            if value:
                return str(value)
        except Exception:
            continue

    return None


def _serialize_bounding_box(bbox: Any) -> Optional[Dict[str, Any]]:
    if bbox is None:
        return None

    try:
        min_point = bbox.minPoint
        max_point = bbox.maxPoint
    except Exception:
        return None

    if not min_point or not max_point:
        return None

    return {
        "min": {
            "x": _safe_round(getattr(min_point, "x", None), 6),
            "y": _safe_round(getattr(min_point, "y", None), 6),
            "z": _safe_round(getattr(min_point, "z", None), 6),
        },
        "max": {
            "x": _safe_round(getattr(max_point, "x", None), 6),
            "y": _safe_round(getattr(max_point, "y", None), 6),
            "z": _safe_round(getattr(max_point, "z", None), 6),
        },
    }


def _extract_center_of_mass(body: adsk.fusion.BRepBody) -> Optional[Dict[str, float]]:
    try:
        props = body.physicalProperties
        if not props:
            return None
        com = props.centerOfMass
        if not com:
            return None
        return {
            "x": _safe_round(getattr(com, "x", None), 6),
            "y": _safe_round(getattr(com, "y", None), 6),
            "z": _safe_round(getattr(com, "z", None), 6),
        }
    except Exception:
        return None


def _infer_feature_instance_count(feature: adsk.fusion.Feature) -> Optional[int]:
    candidate_attributes = (
        "quantity",
        "quantityOne",
        "quantityTwo",
        "instanceCount",
        "patternCount",
    )

    for attr in candidate_attributes:
        try:
            value = getattr(feature, attr, None)
        except Exception:
            value = None

        if value is None:
            continue

        if isinstance(value, (int, float)):
            try:
                return int(value)
            except (TypeError, ValueError):
                continue

    return None

def apply_fillet(
    app: adsk.core.Application,
    entity_tokens: Sequence[str],
    radius: float,
    radius_unit: str = "mm",
    include_tangent_edges: bool = True,
    feature_name: str = "",
) -> Dict[str, Any]:
    """
    Apply a constant-radius fillet to edges identified by entity tokens.

    Args:
        app: Fusion application instance.
        entity_tokens: Iterable of edge entity tokens.
        radius: Fillet radius value.
        radius_unit: Unit for radius (mm, cm, m, in, ft).
        include_tangent_edges: Include tangentially connected edges.

    Returns:
        Dictionary with success status, message, and edge count.
    """
    tokens = [token for token in entity_tokens if isinstance(token, str) and token.strip()]
    if not tokens:
        raise FeatureOperationError("No entity tokens provided for fillet operation.")

    design = _require_design(app)
    root_component = design.rootComponent

    def _native_if_proxy(entity: Any) -> Any:
        try:
            native = getattr(entity, "nativeObject", None)
            return native or entity
        except Exception:
            return entity

    def _assembly_context(entity: Any) -> Optional[adsk.fusion.Occurrence]:
        # Many BRep proxies expose assemblyContext on the entity or its owning body.
        for obj in (entity, getattr(entity, "body", None)):
            try:
                occ = getattr(obj, "assemblyContext", None)
                if occ:
                    return occ
            except Exception:
                continue
        return None

    # Resolve tokens to edges, and infer the correct feature component context.
    edge_collection = adsk.core.ObjectCollection.create()
    missing_tokens: List[str] = []
    inferred_component: Optional[adsk.fusion.Component] = None
    inferred_occurrence: Optional[adsk.fusion.Occurrence] = None

    for token in tokens:
        try:
            entities = design.findEntityByToken(token)
            if entities and len(entities) > 0:
                edge = adsk.fusion.BRepEdge.cast(entities[0])
                if edge:
                    occ = _assembly_context(edge)
                    # If we're operating on an occurrence proxy, build the feature in the
                    # component definition and use native edges (not proxies).
                    candidate_component = None
                    candidate_occ = None
                    edge_for_feature = edge
                    if occ:
                        candidate_occ = occ
                        candidate_component = occ.component
                        edge_for_feature = _native_if_proxy(edge)
                    else:
                        try:
                            body = getattr(edge, "body", None)
                            candidate_component = getattr(body, "parentComponent", None) or root_component
                        except Exception:
                            candidate_component = root_component

                    if inferred_component is None:
                        inferred_component = candidate_component
                        inferred_occurrence = candidate_occ
                    else:
                        if candidate_component != inferred_component:
                            raise FeatureOperationError(
                                "Edges span multiple components/occurrences; cannot fillet them in one feature. "
                                "Select edges from a single component."
                            )
                        # If we started in occurrence context, require all edges to match that occurrence.
                        if inferred_occurrence is not None and candidate_occ != inferred_occurrence:
                            raise FeatureOperationError(
                                "Edges span multiple occurrences; cannot fillet them in one feature. "
                                "Select edges from a single occurrence."
                            )

                    edge_collection.add(edge_for_feature)
                else:
                    missing_tokens.append(token)
            else:
                missing_tokens.append(token)
        except Exception as exc:
            logger.warning("Failed to resolve token %s: %s", token, exc)
            missing_tokens.append(token)

    if edge_collection.count == 0:
        raise FeatureOperationError(
            f"None of the provided entity tokens ({len(tokens)}) resolved to valid edges."
        )

    # Create fillet feature
    try:
        feature_component = inferred_component or root_component
        fillets = feature_component.features.filletFeatures
        fillet_input = fillets.createInput()
        fillet_input.isRollingBallCorner = True

        radius_value = adsk.core.ValueInput.createByString(f"{radius} {radius_unit}")
        fillet_input.edgeSetInputs.addConstantRadiusEdgeSet(
            edge_collection,
            radius_value,
            include_tangent_edges
        )

        fillet_feature = fillets.add(fillet_input)

        if not fillet_feature:
            raise FeatureOperationError("Fillet feature creation returned None.")

        if fillet_feature and feature_name.strip():
            fillet_feature.name = feature_name.strip()

    except Exception as exc:
        logger.exception("Failed to create fillet feature")
        raise FeatureOperationError(f"Failed to apply fillet: {exc}") from exc

    success_message = f"Applied fillet with {radius} {radius_unit} radius to {edge_collection.count} edge(s)"
    if missing_tokens:
        success_message += f". {len(missing_tokens)} token(s) not found"

    logger.info("Fillet applied successfully: %d edges", edge_collection.count)

    return {
        "success": True,
        "message": success_message,
        "edge_count": edge_collection.count,
        "missing_tokens": missing_tokens,
        "radius": radius,
        "radius_unit": radius_unit,
    }


def apply_chamfer(
    app: adsk.core.Application,
    entity_tokens: Sequence[str],
    distance: float,
    distance_unit: str = "mm",
    include_tangent_edges: bool = True,
    feature_name: str = "",
) -> Dict[str, Any]:
    """
    Apply an equal-distance chamfer to edges identified by entity tokens.

    Args:
        app: Fusion application instance.
        entity_tokens: Iterable of edge entity tokens.
        distance: Chamfer distance value.
        distance_unit: Unit for distance (mm, cm, m, in, ft).
        include_tangent_edges: Include tangentially connected edges.

    Returns:
        Dictionary with success status, message, and edge count.
    """
    tokens = [token for token in entity_tokens if isinstance(token, str) and token.strip()]
    if not tokens:
        raise FeatureOperationError("No entity tokens provided for chamfer operation.")

    design = _require_design(app)
    root_component = design.rootComponent

    def _native_if_proxy(entity: Any) -> Any:
        try:
            native = getattr(entity, "nativeObject", None)
            return native or entity
        except Exception:
            return entity

    def _assembly_context(entity: Any) -> Optional[adsk.fusion.Occurrence]:
        for obj in (entity, getattr(entity, "body", None)):
            try:
                occ = getattr(obj, "assemblyContext", None)
                if occ:
                    return occ
            except Exception:
                continue
        return None

    # Resolve tokens to edges, and infer correct component context.
    edge_collection = adsk.core.ObjectCollection.create()
    missing_tokens: List[str] = []
    inferred_component: Optional[adsk.fusion.Component] = None
    inferred_occurrence: Optional[adsk.fusion.Occurrence] = None

    for token in tokens:
        try:
            entities = design.findEntityByToken(token)
            if entities and len(entities) > 0:
                edge = adsk.fusion.BRepEdge.cast(entities[0])
                if edge:
                    occ = _assembly_context(edge)
                    candidate_component = None
                    candidate_occ = None
                    edge_for_feature = edge
                    if occ:
                        candidate_occ = occ
                        candidate_component = occ.component
                        edge_for_feature = _native_if_proxy(edge)
                    else:
                        try:
                            body = getattr(edge, "body", None)
                            candidate_component = getattr(body, "parentComponent", None) or root_component
                        except Exception:
                            candidate_component = root_component

                    if inferred_component is None:
                        inferred_component = candidate_component
                        inferred_occurrence = candidate_occ
                    else:
                        if candidate_component != inferred_component:
                            raise FeatureOperationError(
                                "Edges span multiple components/occurrences; cannot chamfer them in one feature. "
                                "Select edges from a single component."
                            )
                        if inferred_occurrence is not None and candidate_occ != inferred_occurrence:
                            raise FeatureOperationError(
                                "Edges span multiple occurrences; cannot chamfer them in one feature. "
                                "Select edges from a single occurrence."
                            )

                    edge_collection.add(edge_for_feature)
                else:
                    missing_tokens.append(token)
            else:
                missing_tokens.append(token)
        except Exception as exc:
            logger.warning("Failed to resolve token %s: %s", token, exc)
            missing_tokens.append(token)

    if edge_collection.count == 0:
        raise FeatureOperationError(
            f"None of the provided entity tokens ({len(tokens)}) resolved to valid edges."
        )

    # Create chamfer feature
    try:
        feature_component = inferred_component or root_component
        chamfers = feature_component.features.chamferFeatures
        chamfer_input = chamfers.createInput2()

        distance_value = adsk.core.ValueInput.createByString(f"{distance} {distance_unit}")
        chamfer_input.chamferEdgeSets.addEqualDistanceChamferEdgeSet(
            edge_collection,
            distance_value,
            include_tangent_edges
        )

        chamfer_feature = chamfers.add(chamfer_input)

        if not chamfer_feature:
            raise FeatureOperationError("Chamfer feature creation returned None.")

        if chamfer_feature and feature_name.strip():
            chamfer_feature.name = feature_name.strip()

    except Exception as exc:
        logger.exception("Failed to create chamfer feature")
        raise FeatureOperationError(f"Failed to apply chamfer: {exc}") from exc

    success_message = f"Applied chamfer with {distance} {distance_unit} distance to {edge_collection.count} edge(s)"
    if missing_tokens:
        success_message += f". {len(missing_tokens)} token(s) not found"

    logger.info("Chamfer applied successfully: %d edges", edge_collection.count)

    return {
        "success": True,
        "message": success_message,
        "edge_count": edge_collection.count,
        "missing_tokens": missing_tokens,
        "distance": distance,
        "distance_unit": distance_unit,
    }


def create_shell(
    app: adsk.core.Application,
    entity_tokens: Sequence[str],
    inside_thickness: float = 0.0,
    outside_thickness: float = 0.0,
    thickness_unit: str = "mm",
    is_tangent_chain: bool = True,
    shell_type: str = "sharp",
    feature_name: str = "",
) -> Dict[str, Any]:
    """
    Create a shell feature by removing faces (open shell) or hollowing entire bodies (closed shell).

    Args:
        app: Fusion application instance.
        entity_tokens: Iterable of face or body entity tokens. Cannot include a face and its owning body.
        inside_thickness: Wall thickness added inward (in given unit). Set to 0 to disable.
        outside_thickness: Wall thickness added outward (in given unit). Set to 0 to disable.
        thickness_unit: Unit for thickness values (mm, cm, m, in). Defaults to mm.
        is_tangent_chain: When using faces, include tangentially connected faces (default True).
        shell_type: "sharp" (default) or "rounded".
        feature_name: Optional descriptive name for the shell feature.

    Returns:
        Dictionary summarizing the shell operation.
    """
    tokens = [token for token in entity_tokens if isinstance(token, str) and token.strip()]
    if not tokens:
        raise FeatureOperationError("create_shell requires at least one entity token.")

    try:
        inside_value = float(inside_thickness)
        outside_value = float(outside_thickness)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise FeatureOperationError("Shell thickness values must be numbers.") from exc

    if inside_value < 0 or outside_value < 0:
        raise FeatureOperationError("inside_thickness and outside_thickness must be >= 0.")

    if inside_value <= 0 and outside_value <= 0:
        raise FeatureOperationError("At least one of inside_thickness or outside_thickness must be > 0.")

    thickness_unit_norm = _normalize_thickness_unit(thickness_unit)
    shell_type_norm = (shell_type or "sharp").strip().lower()
    if shell_type_norm not in {"sharp", "rounded"}:
        raise FeatureOperationError("shell_type must be 'sharp' or 'rounded'.")

    design = _require_design(app)
    root_component = design.rootComponent

    def _native_if_proxy(entity: Any) -> Any:
        try:
            native = getattr(entity, "nativeObject", None)
            return native or entity
        except Exception:
            return entity

    def _assembly_context(entity: Any) -> Optional[adsk.fusion.Occurrence]:
        # BRep proxies commonly expose assemblyContext on the entity or its owning body.
        for obj in (entity, getattr(entity, "body", None)):
            try:
                occ = getattr(obj, "assemblyContext", None)
                if occ:
                    return occ
            except Exception:
                continue
        return None

    inferred_component: Optional[adsk.fusion.Component] = None
    inferred_occurrence: Optional[adsk.fusion.Occurrence] = None

    entities_collection = adsk.core.ObjectCollection.create()
    faces: List[adsk.fusion.BRepFace] = []
    bodies: List[adsk.fusion.BRepBody] = []
    face_body_tokens: Set[str] = set()
    body_tokens: Set[str] = set()
    body_face_selection_counts: Dict[str, int] = {}
    body_lookup: Dict[str, adsk.fusion.BRepBody] = {}

    for token in tokens:
        resolved = design.findEntityByToken(token.strip())
        if not resolved or len(resolved) == 0:
            raise FeatureOperationError(f"Entity token '{token}' not found.")

        entity = resolved[0]
        face = adsk.fusion.BRepFace.cast(entity)
        if face:
            occ = _assembly_context(face)
            candidate_occ = None
            candidate_component: adsk.fusion.Component
            face_for_feature = face

            if occ:
                candidate_occ = occ
                candidate_component = occ.component
                face_for_feature = _native_if_proxy(face)
            else:
                try:
                    candidate_component = getattr(face.body, "parentComponent", None) or root_component
                except Exception:
                    candidate_component = root_component

            if inferred_component is None:
                inferred_component = candidate_component
                inferred_occurrence = candidate_occ
            else:
                if candidate_component != inferred_component:
                    raise FeatureOperationError(
                        "Shell entities span multiple components/occurrences; cannot shell in one feature. "
                        "Select faces/bodies from a single component."
                    )
                if inferred_occurrence is not None and candidate_occ != inferred_occurrence:
                    raise FeatureOperationError(
                        "Shell entities span multiple occurrences; cannot shell in one feature. "
                        "Select faces from a single occurrence."
                    )

            body = face_for_feature.body
            if not body or not body.isSolid:
                raise FeatureOperationError("Shell requires faces on solid bodies.")

            faces.append(face_for_feature)
            entities_collection.add(face_for_feature)

            body_token = getattr(body, "entityToken", None)
            if isinstance(body_token, str) and body_token.strip():
                face_body_tokens.add(body_token.strip())
                key = body_token.strip()
            else:
                key = str(id(body))

            body_face_selection_counts[key] = body_face_selection_counts.get(key, 0) + 1
            body_lookup[key] = body
            continue

        body = adsk.fusion.BRepBody.cast(entity)
        if body:
            occ = _assembly_context(body)
            candidate_occ = None
            candidate_component: adsk.fusion.Component
            body_for_feature = body

            if occ:
                candidate_occ = occ
                candidate_component = occ.component
                body_for_feature = _native_if_proxy(body)
            else:
                try:
                    candidate_component = getattr(body, "parentComponent", None) or root_component
                except Exception:
                    candidate_component = root_component

            if inferred_component is None:
                inferred_component = candidate_component
                inferred_occurrence = candidate_occ
            else:
                if candidate_component != inferred_component:
                    raise FeatureOperationError(
                        "Shell entities span multiple components/occurrences; cannot shell in one feature. "
                        "Select faces/bodies from a single component."
                    )
                if inferred_occurrence is not None and candidate_occ != inferred_occurrence:
                    raise FeatureOperationError(
                        "Shell entities span multiple occurrences; cannot shell in one feature. "
                        "Select bodies from a single occurrence."
                    )

            if not body_for_feature.isSolid:
                raise FeatureOperationError("Shell requires solid bodies, not surface bodies.")

            bodies.append(body_for_feature)
            entities_collection.add(body_for_feature)

            body_token = getattr(body_for_feature, "entityToken", None)
            if isinstance(body_token, str) and body_token.strip():
                body_tokens.add(body_token.strip())
            continue

        raise FeatureOperationError(f"Token '{token}' is neither a face nor a body.")

    # Conflict check: faces + their owning bodies cannot be combined
    if faces and bodies:
        conflict = False
        if face_body_tokens and body_tokens:
            conflict = any(bt in face_body_tokens for bt in body_tokens)
        if not conflict:
            # Fallback to object identity comparison
            for face in faces:
                owning_body = face.body
                if owning_body and any(owning_body is body for body in bodies):
                    conflict = True
                    break
        if conflict:
            raise FeatureOperationError("Cannot mix face tokens with their owning bodies in create_shell.")
        entity_type = "mixed"
    elif faces:
        entity_type = "face"
    else:
        entity_type = "body"

    # Prevent removing all faces of a body when using only faces
    if faces and not bodies:
        for key, count in body_face_selection_counts.items():
            body = body_lookup.get(key)
            total_faces = getattr(body, "faces", None)
            total_count = getattr(total_faces, "count", 0) if total_faces else 0
            if total_count and count >= total_count:
                raise FeatureOperationError("Shell requires leaving at least one face per body; reduce selected faces.")

    inside_cm = _convert_length_units(inside_value, thickness_unit_norm, "cm") if inside_value > 0 else 0
    outside_cm = _convert_length_units(outside_value, thickness_unit_norm, "cm") if outside_value > 0 else 0

    try:
        feature_component = inferred_component or root_component
        shell_features = feature_component.features.shellFeatures
        shell_input = shell_features.createInput(entities_collection, bool(is_tangent_chain))

        if inside_cm > 0:
            shell_input.insideThickness = adsk.core.ValueInput.createByReal(inside_cm)
        if outside_cm > 0:
            shell_input.outsideThickness = adsk.core.ValueInput.createByReal(outside_cm)

        shell_type_map = {
            "sharp": adsk.fusion.ShellTypes.SharpOffsetShellType,
            "rounded": adsk.fusion.ShellTypes.RoundedOffsetShellType,
        }
        shell_input.shellType = shell_type_map[shell_type_norm]

        shell_feature = shell_features.add(shell_input)
        if not shell_feature:
            raise FeatureOperationError("Shell feature creation failed.")

        if feature_name and feature_name.strip():
            shell_feature.name = feature_name.strip()

    except FeatureOperationError:
        raise
    except Exception as exc:
        logger.exception("Failed to create shell feature")
        raise FeatureOperationError(f"Failed to create shell: {exc}") from exc

    thickness_parts = []
    if inside_value > 0:
        thickness_parts.append(f"inside {inside_value} {thickness_unit_norm}")
    if outside_value > 0:
        thickness_parts.append(f"outside {outside_value} {thickness_unit_norm}")
    thickness_text = ", ".join(thickness_parts)
    if faces and bodies:
        target_label = "entities"
    else:
        target_label = "faces" if faces else "bodies"
    success_message = f"Created shell ({thickness_text}) on {entities_collection.count} {target_label}"

    return {
        "success": True,
        "message": success_message,
        "operation": "create_shell",
        "entity_count": entities_collection.count,
        "entity_type": entity_type,
        "inside_thickness": inside_value,
        "outside_thickness": outside_value,
        "thickness_unit": thickness_unit_norm,
        "is_tangent_chain": bool(is_tangent_chain),
        "shell_type": shell_type_norm,
        "feature_name": feature_name or getattr(shell_feature, "name", ""),
    }


def create_simple_hole(
    app: adsk.core.Application,
    face_token: str,
    center_x: float,
    center_y: float,
    center_z: float,
    diameter: float,
    diameter_unit: str = "mm",
    extent_type: str = "through_all",
    depth: Optional[float] = None,
    feature_name: str = "",
) -> Dict[str, Any]:
    """
    Create a simple drilled hole in a face.

    Args:
        app: Fusion application instance.
        face_token: Entity token of the face to drill into.
        center_x: X coordinate of hole center in mm.
        center_y: Y coordinate of hole center in mm.
        center_z: Z coordinate of hole center in mm.
        diameter: Hole diameter value.
        diameter_unit: Unit for diameter (mm, cm, m, in).
        extent_type: "through_all" or "distance".
        depth: Hole depth in mm (required if extent_type="distance").
        feature_name: Optional descriptive name for timeline.

    Returns:
        Dictionary with success status and hole details.
    """
    if not face_token or not face_token.strip():
        raise FeatureOperationError("Face token cannot be empty.")

    design = _require_design(app)
    root_component = design.rootComponent

    # Resolve face token
    try:
        entities = design.findEntityByToken(face_token.strip())
        if not entities or len(entities) == 0:
            raise FeatureOperationError(f"Face token '{face_token}' not found in design.")
        
        face = adsk.fusion.BRepFace.cast(entities[0])
        if not face:
            raise FeatureOperationError(f"Token '{face_token}' does not resolve to a face.")
    except FeatureOperationError:
        raise
    except Exception as exc:
        logger.exception("Failed to resolve face token")
        raise FeatureOperationError(f"Failed to resolve face token: {exc}") from exc

    # Hole coordinates are mm at the tool boundary; convert to Fusion internal cm.
    center_point = _point_from_mm(center_x, center_y, center_z)
    _validate_hole_center_on_face(face, center_point, (center_x, center_y, center_z))

    # Determine the component that owns this face
    # If the face is a proxy from an occurrence, we need to work with that occurrence
    face_component = None
    face_occurrence = None

    # Try to get the native entity and its component
    if hasattr(face, 'nativeObject'):
        native_face = face.nativeObject
        if native_face and hasattr(native_face, 'body') and native_face.body:
            if hasattr(native_face.body, 'assemblyContext'):
                face_occurrence = native_face.body.assemblyContext
            if hasattr(native_face.body, 'parentComponent'):
                face_component = native_face.body.parentComponent

    # Fallback to root component if we couldn't determine the component
    if face_component is None:
        face_component = root_component
        face_occurrence = None

    # Create hole feature with proper cleanup
    temp_sketch = None
    created_center_world_mm: Optional[Dict[str, float]] = None
    selected_direction: Optional[adsk.fusion.ExtentDirections] = None
    creation_method = "hole_feature"
    fallback_cut_distance_mm: Optional[float] = None
    try:
        # Get holes collection from the correct component
        if face_occurrence:
            holes = face_occurrence.component.features.holeFeatures
        else:
            holes = face_component.features.holeFeatures

        # Create a temporary sketch on the face in the correct component
        if face_occurrence:
            temp_sketch = face_occurrence.component.sketches.add(face)
        else:
            temp_sketch = face_component.sketches.add(face)

        sketch_point_in_face_space = temp_sketch.modelToSketchSpace(center_point)
        sketch_point = temp_sketch.sketchPoints.add(sketch_point_in_face_space)
        try:
            world_geom = getattr(sketch_point, "worldGeometry", None)
            if world_geom:
                created_center_world_mm = _point_to_mm_dict(world_geom)
        except Exception:
            created_center_world_mm = None

        def _build_hole_input() -> adsk.fusion.HoleFeatureInput:
            input_obj = holes.createSimpleInput(
                adsk.core.ValueInput.createByString(f"{diameter} {diameter_unit}")
            )
            input_obj.setPositionBySketchPoint(sketch_point)
            return input_obj

        if extent_type == "through_all":
            preferred = _preferred_through_all_direction(face)
            opposite = (
                adsk.fusion.ExtentDirections.NegativeExtentDirection
                if preferred == adsk.fusion.ExtentDirections.PositiveExtentDirection
                else adsk.fusion.ExtentDirections.PositiveExtentDirection
            )
            directions = [preferred]
            if opposite != preferred:
                directions.append(opposite)

            hole_feature = None
            last_exception: Optional[Exception] = None

            for idx, direction in enumerate(directions):
                hole_input = _build_hole_input()
                hole_input.setAllExtent(direction)
                try:
                    hole_feature = holes.add(hole_input)
                    selected_direction = direction
                    break
                except Exception as exc:
                    last_exception = exc
                    if idx == 0 and (_is_missing_target_body_error(exc) or _is_logical_selection_error(exc)):
                        logger.info(
                            "Simple hole through_all retry with opposite direction "
                            "(first direction=%s, reason=%s).",
                            _through_direction_label(direction),
                            "missing_target_body" if _is_missing_target_body_error(exc) else "logical_selection",
                        )
                        continue
                    raise

            if not hole_feature:
                if last_exception and _is_logical_selection_error(last_exception):
                    logger.info(
                        "Simple hole through_all falling back to sketch-cut due to logicalSelection failure."
                    )
                    _, selected_direction, fallback_cut_distance_cm = _create_hole_cut_fallback(
                        face_component,
                        temp_sketch,
                        sketch_point_in_face_space,
                        diameter,
                        diameter_unit,
                        face,
                        preferred,
                        feature_name,
                    )
                    creation_method = "sketch_cut_fallback"
                    fallback_cut_distance_mm = round(_cm_to_mm(fallback_cut_distance_cm), 6)
                elif last_exception:
                    raise last_exception
                else:
                    raise FeatureOperationError("Hole feature creation returned None.")
        elif extent_type == "distance":
            if depth is None or depth <= 0:
                raise FeatureOperationError("Depth must be positive when extent_type is 'distance'.")

            depth_value = adsk.core.ValueInput.createByReal(_mm_to_cm(depth))
            hole_input = _build_hole_input()
            hole_input.setDistanceExtent(depth_value)
            hole_feature = holes.add(hole_input)
            selected_direction = None
        else:
            raise FeatureOperationError(f"Invalid extent_type: '{extent_type}'. Must be 'through_all' or 'distance'.")

        if not hole_feature and creation_method == "hole_feature":
            raise FeatureOperationError("Hole feature creation returned None.")

        # Apply feature name if provided
        if hole_feature and feature_name.strip():
            hole_feature.name = feature_name.strip()

        # Make the positioning sketch invisible but don't delete it
        # The hole feature references the sketch point, so deleting the sketch
        # would break the hole's references and prevent patterning
        if temp_sketch:
            try:
                temp_sketch.isVisible = False
            except Exception:
                logger.debug("Could not hide positioning sketch")

    except FeatureOperationError:
        raise
    except Exception as exc:
        logger.exception("Failed to create hole feature")
        raise FeatureOperationError(f"Failed to create hole: {exc}") from exc

    success_message = f"Created hole with {diameter} {diameter_unit} diameter"
    if extent_type == "through_all":
        direction_label = _through_direction_label(
            selected_direction or adsk.fusion.ExtentDirections.PositiveExtentDirection
        )
        success_message += f" (through all, {direction_label} extent)"
        if creation_method == "sketch_cut_fallback":
            success_message += " via sketch-cut fallback"
    else:
        success_message += f" to depth {depth} mm"

    logger.info("Hole created successfully: %s", success_message)

    return {
        "success": True,
        "message": success_message,
        "diameter": diameter,
        "diameter_unit": diameter_unit,
        "extent_type": extent_type,
        "depth": depth if extent_type == "distance" else None,
        "center": created_center_world_mm,
        "center_unit": "mm",
        "through_direction": _through_direction_label(selected_direction) if selected_direction else None,
        "creation_method": creation_method,
        "fallback_cut_distance_mm": fallback_cut_distance_mm,
    }


def create_counterbore_hole(
    app: adsk.core.Application,
    face_token: str,
    center_x: float,
    center_y: float,
    center_z: float,
    hole_diameter: float,
    hole_depth: float,
    counterbore_diameter: float,
    counterbore_depth: float,
    diameter_unit: str = "mm",
    feature_name: str = "",
) -> Dict[str, Any]:
    """
    Create a counterbore hole in a face for recessing bolt/screw heads.

    A counterbore hole has two diameters:
    - A larger shallow counterbore for the bolt head
    - A smaller main hole for the bolt shank

    Args:
        app: Fusion application instance.
        face_token: Entity token of the face to drill into.
        center_x: X coordinate of hole center in mm.
        center_y: Y coordinate of hole center in mm.
        center_z: Z coordinate of hole center in mm.
        hole_diameter: Main hole diameter for bolt shank (must be positive).
        hole_depth: Depth of main hole in mm (can exceed part thickness for through holes).
        counterbore_diameter: Counterbore diameter for bolt head recess (must be > hole_diameter).
        counterbore_depth: Depth of counterbore in mm (typically bolt head height + 0.5mm).
        diameter_unit: Unit for diameter measurements (mm, cm, m, in). Coordinates are in mm.
        feature_name: Optional descriptive name for timeline.

    Returns:
        Dictionary with success status and hole details.

    Raises:
        FeatureOperationError: If parameters are invalid or hole creation fails.
    """
    # Validate parameters
    if not face_token or not face_token.strip():
        raise FeatureOperationError("Face token cannot be empty.")
    
    if hole_diameter <= 0:
        raise FeatureOperationError(f"hole_diameter must be positive, got {hole_diameter}.")
    
    if counterbore_diameter <= 0:
        raise FeatureOperationError(f"counterbore_diameter must be positive, got {counterbore_diameter}.")
    
    if counterbore_diameter <= hole_diameter:
        raise FeatureOperationError(
            f"counterbore_diameter ({counterbore_diameter}) must be larger than hole_diameter ({hole_diameter})."
        )
    
    if hole_depth <= 0:
        raise FeatureOperationError(f"hole_depth must be positive, got {hole_depth}.")
    
    if counterbore_depth <= 0:
        raise FeatureOperationError(f"counterbore_depth must be positive, got {counterbore_depth}.")

    design = _require_design(app)
    root_component = design.rootComponent

    # Resolve face token
    try:
        entities = design.findEntityByToken(face_token.strip())
        if not entities or len(entities) == 0:
            raise FeatureOperationError(f"Face token '{face_token}' not found in design.")
        
        face = adsk.fusion.BRepFace.cast(entities[0])
        if not face:
            raise FeatureOperationError(f"Token '{face_token}' does not resolve to a face.")
    except FeatureOperationError:
        raise
    except Exception as exc:
        logger.exception("Failed to resolve face token")
        raise FeatureOperationError(f"Failed to resolve face token: {exc}") from exc

    # Coordinates/depths are mm at the tool boundary; convert to Fusion internal cm.
    center_point = _point_from_mm(center_x, center_y, center_z)
    _validate_hole_center_on_face(face, center_point, (center_x, center_y, center_z))
    hole_depth_cm = _mm_to_cm(hole_depth)
    counterbore_depth_cm = _mm_to_cm(counterbore_depth)

    # Determine the component that owns this face
    # If the face is a proxy from an occurrence, we need to work with that occurrence
    face_component = None
    face_occurrence = None

    # Try to get the native entity and its component
    if hasattr(face, 'nativeObject'):
        native_face = face.nativeObject
        if native_face and hasattr(native_face, 'body') and native_face.body:
            if hasattr(native_face.body, 'assemblyContext'):
                face_occurrence = native_face.body.assemblyContext
            if hasattr(native_face.body, 'parentComponent'):
                face_component = native_face.body.parentComponent

    # Fallback to root component if we couldn't determine the component
    if face_component is None:
        face_component = root_component
        face_occurrence = None

    # Create counterbore hole feature with proper cleanup
    temp_sketch = None
    try:
        # Get holes collection from the correct component
        if face_occurrence:
            holes = face_occurrence.component.features.holeFeatures
        else:
            holes = face_component.features.holeFeatures

        # Create counterbore input with three parameters:
        # 1. Hole diameter (for bolt shank)
        # 2. Counterbore diameter (for bolt head)
        # 3. Counterbore depth (how deep the counterbore recess is)
        hole_input = holes.createCounterboreInput(
            adsk.core.ValueInput.createByString(f"{hole_diameter} {diameter_unit}"),
            adsk.core.ValueInput.createByString(f"{counterbore_diameter} {diameter_unit}"),
            adsk.core.ValueInput.createByReal(counterbore_depth_cm),
        )

        # Create a temporary sketch on the face in the correct component
        # This establishes proper component context for positioning
        if face_occurrence:
            temp_sketch = face_occurrence.component.sketches.add(face)
        else:
            temp_sketch = face_component.sketches.add(face)

        sketch_point_in_face_space = temp_sketch.modelToSketchSpace(center_point)
        sketch_point = temp_sketch.sketchPoints.add(sketch_point_in_face_space)

        # Use sketch point for positioning (more robust than direct point positioning)
        hole_input.setPositionBySketchPoint(sketch_point)

        # Set the main hole depth extent
        # Note: For counterbore holes, we set the depth of the main (smaller) hole
        # The counterbore depth was already specified in createCounterboreInput
        depth_value = adsk.core.ValueInput.createByReal(hole_depth_cm)
        hole_input.setDistanceExtent(depth_value)

        # Create the counterbore hole
        hole_feature = holes.add(hole_input)

        if not hole_feature:
            raise FeatureOperationError("Counterbore hole feature creation returned None.")

        # Apply feature name if provided
        if hole_feature and feature_name.strip():
            hole_feature.name = feature_name.strip()

        # Make the positioning sketch invisible but don't delete it
        # The hole feature references the sketch point, so deleting the sketch
        # would break the hole's references and prevent patterning
        if temp_sketch:
            try:
                temp_sketch.isVisible = False
            except Exception:
                logger.debug("Could not hide positioning sketch")

    except FeatureOperationError:
        raise
    except Exception as exc:
        logger.exception("Failed to create counterbore hole feature")
        raise FeatureOperationError(f"Failed to create counterbore hole: {exc}") from exc

    success_message = (
        f"Created counterbore hole: {hole_diameter} {diameter_unit} hole × {hole_depth} mm deep, "
        f"{counterbore_diameter} {diameter_unit} counterbore × {counterbore_depth} mm deep"
    )

    logger.info("Counterbore hole created successfully: %s", success_message)

    return {
        "success": True,
        "message": success_message,
        "hole_diameter": hole_diameter,
        "counterbore_diameter": counterbore_diameter,
        "hole_depth": hole_depth,
        "counterbore_depth": counterbore_depth,
        "diameter_unit": diameter_unit,
    }


def create_tapped_hole(
    app: adsk.core.Application,
    face_token: str,
    center_x: float,
    center_y: float,
    center_z: float,
    thread_type: str,
    thread_size: str,
    thread_depth: float,
    pilot_hole_depth: Optional[float] = None,
    diameter_unit: str = "mm",
    feature_name: str = "",
) -> Dict[str, Any]:
    """
    Create a tapped (threaded) hole on the specified face.

    Hole center coordinates and depth inputs are specified in mm.
    """
    if not face_token or not face_token.strip():
        raise FeatureOperationError("Face token cannot be empty.")

    thread_type = (thread_type or "").strip()
    thread_size = (thread_size or "").strip()
    if not thread_type or not thread_size:
        raise FeatureOperationError("Thread type and size are required.")

    diameter_unit = _normalize_diameter_unit(diameter_unit)

    try:
        thread_depth_value_mm = float(thread_depth)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise FeatureOperationError("Thread depth must be a number.") from exc

    if thread_depth_value_mm <= 0:
        raise FeatureOperationError("Thread depth must be positive.")

    pilot_depth_value_mm = None
    if pilot_hole_depth is not None:
        try:
            pilot_depth_value_mm = float(pilot_hole_depth)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise FeatureOperationError("Pilot hole depth must be a number.") from exc
        if pilot_depth_value_mm <= 0:
            raise FeatureOperationError("Pilot hole depth must be positive when provided.")
        if pilot_depth_value_mm < thread_depth_value_mm:
            raise FeatureOperationError("Pilot hole depth must be greater than or equal to thread depth.")

    thread_spec = thread_database.lookup_thread(thread_type, thread_size)
    if not thread_spec:
        available = thread_database.list_available_threads(thread_type)
        available_msg = ", ".join(available) if available else "no known designations"
        raise FeatureOperationError(
            f"Unknown thread specification '{thread_type} {thread_size}'. Available: {available_msg}"
        )

    if pilot_depth_value_mm is None:
        clearance_mm = max(thread_spec.pitch * 2.0, 0.5)
        pilot_depth_value_mm = thread_depth_value_mm + clearance_mm

    thread_depth_value_cm = _mm_to_cm(thread_depth_value_mm)
    pilot_depth_value_cm = _mm_to_cm(pilot_depth_value_mm)

    depth_ok, depth_message = thread_database.validate_thread_depth(thread_spec, thread_depth_value_cm)
    if depth_message:
        log_method = logger.warning if not depth_ok else logger.info
        log_method(depth_message)

    design = _require_design(app)
    root_component = design.rootComponent

    # Resolve face token
    try:
        entities = design.findEntityByToken(face_token.strip())
        if not entities or len(entities) == 0:
            raise FeatureOperationError(f"Face token '{face_token}' not found in design.")

        face = adsk.fusion.BRepFace.cast(entities[0])
        if not face:
            raise FeatureOperationError(f"Token '{face_token}' does not resolve to a face.")
    except FeatureOperationError:
        raise
    except Exception as exc:
        logger.exception("Failed to resolve face token")
        raise FeatureOperationError(f"Failed to resolve face token: {exc}") from exc

    center_point = _point_from_mm(center_x, center_y, center_z)
    _validate_hole_center_on_face(face, center_point, (center_x, center_y, center_z))

    face_component = None
    face_occurrence = None
    if hasattr(face, "nativeObject"):
        native_face = face.nativeObject
        if native_face and hasattr(native_face, "body") and native_face.body:
            body = native_face.body
            if hasattr(body, "assemblyContext"):
                face_occurrence = body.assemblyContext
            if hasattr(body, "parentComponent"):
                face_component = body.parentComponent

    if face_component is None:
        face_component = root_component
        face_occurrence = None

    tap_drill_in_unit = _convert_length_units(thread_spec.tap_drill_diameter, "mm", diameter_unit)
    tap_drill_text = f"{tap_drill_in_unit:.4f}".rstrip("0").rstrip(".")
    if not tap_drill_text:
        tap_drill_text = "0"

    temp_sketch = None
    try:
        thread_info = _find_thread_info(app, thread_spec, is_internal=True)

        if face_occurrence:
            holes = face_occurrence.component.features.holeFeatures
        else:
            holes = face_component.features.holeFeatures

        tap_drill_value = adsk.core.ValueInput.createByString(f"{tap_drill_text} {diameter_unit}")

        # Create simple hole input, then convert to tapped hole
        # setToTappedHole was introduced in Sept 2025
        hole_input = holes.createSimpleInput(tap_drill_value)
        hole_input.setToTappedHole(thread_info)

        if face_occurrence:
            temp_sketch = face_occurrence.component.sketches.add(face)
        else:
            temp_sketch = face_component.sketches.add(face)

        sketch_point_in_face_space = temp_sketch.modelToSketchSpace(center_point)
        sketch_point = temp_sketch.sketchPoints.add(sketch_point_in_face_space)
        hole_input.setPositionBySketchPoint(sketch_point)

        pilot_depth_vi = adsk.core.ValueInput.createByReal(pilot_depth_value_cm)
        hole_input.setDistanceExtent(pilot_depth_vi)

        thread_length_vi = adsk.core.ValueInput.createByReal(thread_depth_value_cm)
        thread_offset_vi = adsk.core.ValueInput.createByReal(0)
        hole_input.setLengthAndOffset(thread_length_vi, thread_offset_vi)

        hole_feature = holes.add(hole_input)
        if not hole_feature:
            raise FeatureOperationError("Tapped hole feature creation returned None.")

        if hole_feature and feature_name.strip():
            hole_feature.name = feature_name.strip()

        # Make the positioning sketch invisible but don't delete it
        # The hole feature references the sketch point, so deleting the sketch
        # would break the hole's references and prevent patterning
        if temp_sketch:
            try:
                temp_sketch.isVisible = False
            except Exception:
                logger.debug("Could not hide positioning sketch")

    except FeatureOperationError:
        raise
    except Exception as exc:
        logger.exception("Failed to create tapped hole")
        raise FeatureOperationError(f"Failed to create tapped hole: {exc}") from exc

    success_message = (
        f"Created tapped hole: {thread_spec.designation}, "
        f"tap drill {tap_drill_text} {diameter_unit}, "
        f"thread depth {thread_depth_value_mm} mm, "
        f"pilot depth {pilot_depth_value_mm} mm"
    )
    logger.info(success_message)

    return {
        "success": True,
        "message": success_message,
        "thread_type": thread_spec.thread_type,
        "thread_designation": thread_spec.designation,
        "tap_drill_diameter": tap_drill_in_unit,
        "thread_depth": thread_depth_value_mm,
        "pilot_hole_depth": pilot_depth_value_mm,
        "diameter_unit": diameter_unit,
    }


def create_external_thread(
    app: adsk.core.Application,
    face_token: str,
    thread_type: str,
    thread_size: str,
    thread_length: float,
    thread_offset: float = 0.0,
    is_full_length: bool = False,
    diameter_unit: str = "mm",
    feature_name: str = "",
) -> Dict[str, Any]:
    """
    Create external threads on a cylindrical face (for bolts, screws, threaded shafts).

    Args:
        app: Fusion application instance.
        face_token: Entity token of the cylindrical face to thread.
        thread_type: Thread standard - 'metric' (ISO), 'unc', or 'unf'.
        thread_size: Thread size designation (e.g., 'M6', '1/4-20').
        thread_length: Length of threaded portion in cm.
        thread_offset: Offset distance from face start in cm (default: 0).
        is_full_length: If True, thread the entire cylinder length (default: False).
        diameter_unit: Unit for reporting nominal diameter (mm, cm, m, in).
        feature_name: Optional descriptive name for timeline.

    Returns:
        Dictionary with success status and thread details.

    Raises:
        FeatureOperationError: If parameters are invalid or thread creation fails.
    """
    if not face_token or not face_token.strip():
        raise FeatureOperationError("Face token cannot be empty.")

    thread_type = (thread_type or "").strip()
    thread_size = (thread_size or "").strip()
    if not thread_type or not thread_size:
        raise FeatureOperationError("Thread type and size are required.")

    diameter_unit = _normalize_diameter_unit(diameter_unit)

    # Validate thread_length based on is_full_length setting
    thread_length_value = 0.0  # Default value, not used when is_full_length=True
    if not is_full_length:
        # When not full-length, thread_length is required and must be positive
        try:
            thread_length_value = float(thread_length)
        except (TypeError, ValueError) as exc:
            raise FeatureOperationError("Thread length must be a number when is_full_length is False.") from exc

        if thread_length_value <= 0:
            raise FeatureOperationError("Thread length must be positive when is_full_length is False.")

    try:
        thread_offset_value = float(thread_offset)
    except (TypeError, ValueError) as exc:
        raise FeatureOperationError("Thread offset must be a number.") from exc

    if thread_offset_value < 0:
        raise FeatureOperationError("Thread offset cannot be negative.")

    # Lookup thread specification
    thread_spec = thread_database.lookup_thread(thread_type, thread_size)
    if not thread_spec:
        available = thread_database.list_available_threads(thread_type)
        available_msg = ", ".join(available) if available else "no known designations"
        raise FeatureOperationError(
            f"Unknown thread specification '{thread_type} {thread_size}'. Available: {available_msg}"
        )

    # Validate thread length (only when not full-length)
    if not is_full_length:
        depth_ok, depth_message = thread_database.validate_thread_depth(thread_spec, thread_length_value)
        if depth_message:
            log_method = logger.warning if not depth_ok else logger.info
            log_method(depth_message)

    design = _require_design(app)
    root_component = design.rootComponent

    # Resolve face token
    try:
        entities = design.findEntityByToken(face_token.strip())
        if not entities or len(entities) == 0:
            raise FeatureOperationError(f"Face token '{face_token}' not found in design.")

        face = adsk.fusion.BRepFace.cast(entities[0])
        if not face:
            raise FeatureOperationError(f"Token '{face_token}' does not resolve to a face.")
    except FeatureOperationError:
        raise
    except Exception as exc:
        logger.exception("Failed to resolve face token")
        raise FeatureOperationError(f"Failed to resolve face token: {exc}") from exc

    # Validate that face is cylindrical
    try:
        geometry = face.geometry
        if not geometry or geometry.surfaceType != adsk.core.SurfaceTypes.CylinderSurfaceType:
            raise FeatureOperationError(
                f"Face must be cylindrical for external threads. Found: {geometry.surfaceType if geometry else 'unknown'}"
            )
    except FeatureOperationError:
        raise
    except Exception as exc:
        logger.exception("Failed to validate cylindrical face")
        raise FeatureOperationError(f"Failed to validate face geometry: {exc}") from exc

    # Determine component context
    face_component = None
    face_occurrence = None
    if hasattr(face, "nativeObject"):
        native_face = face.nativeObject
        if native_face and hasattr(native_face, "body") and native_face.body:
            body = native_face.body
            if hasattr(body, "assemblyContext"):
                face_occurrence = body.assemblyContext
            if hasattr(body, "parentComponent"):
                face_component = body.parentComponent

    if face_component is None:
        face_component = root_component
        face_occurrence = None

    # Convert nominal diameter to target unit for reporting
    nominal_in_unit = _convert_length_units(thread_spec.nominal_diameter, "mm", diameter_unit)

    try:
        # Get ThreadInfo for external threads
        thread_info = _find_thread_info(app, thread_spec, is_internal=False)

        # Get thread features collection
        if face_occurrence:
            thread_features = face_occurrence.component.features.threadFeatures
        else:
            thread_features = face_component.features.threadFeatures

        # Create face collection for the cylindrical face
        face_collection = adsk.core.ObjectCollection.create()
        face_collection.add(face)

        # Create thread input
        thread_input = thread_features.createInput(face_collection, thread_info)

        if not thread_input:
            raise FeatureOperationError("Failed to create thread input.")

        # Set thread extent
        if is_full_length:
            thread_input.isFullLength = True
        else:
            thread_input.isFullLength = False
            thread_length_vi = adsk.core.ValueInput.createByReal(thread_length_value)
            thread_input.threadLength = thread_length_vi

            # Set thread offset if provided
            if thread_offset_value > 0:
                thread_offset_vi = adsk.core.ValueInput.createByReal(thread_offset_value)
                thread_input.threadOffset = thread_offset_vi

        # Create the thread feature
        thread_feature = thread_features.add(thread_input)

        if not thread_feature:
            raise FeatureOperationError("Thread feature creation returned None.")

        # Apply feature name if provided
        if thread_feature and feature_name.strip():
            thread_feature.name = feature_name.strip()

    except FeatureOperationError:
        raise
    except Exception as exc:
        logger.exception("Failed to create external thread")
        raise FeatureOperationError(f"Failed to create external thread: {exc}") from exc

    # Build success message
    if is_full_length:
        success_message = f"Created external thread: {thread_spec.designation} (full length)"
    else:
        success_message = (
            f"Created external thread: {thread_spec.designation}, "
            f"length {thread_length_value} cm"
        )
        if thread_offset_value > 0:
            success_message += f", offset {thread_offset_value} cm"

    logger.info(success_message)

    return {
        "success": True,
        "message": success_message,
        "thread_type": thread_spec.thread_type,
        "thread_designation": thread_spec.designation,
        "nominal_diameter": nominal_in_unit,
        "thread_length": thread_length_value if not is_full_length else None,
        "thread_offset": thread_offset_value if not is_full_length else None,
        "is_full_length": is_full_length,
        "diameter_unit": diameter_unit,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_diameter_unit(unit: str) -> str:
    candidate = (unit or "mm").strip().lower()
    if candidate not in _VALID_DIAMETER_UNITS:
        raise FeatureOperationError(
            f"Unsupported diameter_unit '{unit}'. Expected one of {sorted(_VALID_DIAMETER_UNITS)}."
        )
    return candidate


def _normalize_thickness_unit(unit: str) -> str:
    candidate = (unit or "mm").strip().lower()
    if candidate not in _VALID_THICKNESS_UNITS:
        raise FeatureOperationError(
            f"Unsupported thickness_unit '{unit}'. Expected one of {sorted(_VALID_THICKNESS_UNITS)}."
        )
    return candidate


def _convert_length_units(value: float, from_unit: str, to_unit: str) -> float:
    from_key = (from_unit or "").strip().lower()
    to_key = (to_unit or "").strip().lower()
    try:
        mm_value = value * _MM_PER_UNIT[from_key]
        return mm_value / _MM_PER_UNIT[to_key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise FeatureOperationError(f"Unsupported unit conversion from '{from_unit}' to '{to_unit}'.") from exc


def _find_thread_info(app: adsk.core.Application, thread_spec: thread_database.ThreadSpecification, is_internal: bool = True) -> adsk.fusion.ThreadInfo:
    """
    Create ThreadInfo object for the specified thread using the correct API workflow.

    Args:
        app: Fusion application instance.
        thread_spec: Thread specification from thread database.
        is_internal: True for internal threads (tapped holes), False for external threads (bolts/screws).

    Workflow:
    1. Query allSizes(threadType) to get available size values
    2. Match our thread to a size value (e.g., M6 → "6")
    3. Query allDesignations(threadType, size) to get available designations
    4. Pick the designation matching our pitch
    5. Query allClasses(isInternal, threadType, designation) for valid classes
    6. Create ThreadInfo with the queried values
    """
    design = _require_design(app)
    root_component = design.rootComponent

    try:
        thread_features = root_component.features.threadFeatures
        thread_data_query = thread_features.threadDataQuery
    except AttributeError as exc:
        raise FeatureOperationError(f"Failed to access thread features: {exc}") from exc

    # Determine the thread type
    if thread_spec.is_metric:
        try:
            thread_type = thread_data_query.defaultMetricThreadType
        except AttributeError:
            thread_type = "ISO Metric profile"
    else:
        try:
            thread_type = thread_data_query.defaultImperialThreadType
        except AttributeError:
            thread_type = "ANSI Unified Screw Threads"

    thread_direction = "internal" if is_internal else "external"
    logger.info(f"Looking up {thread_direction} thread {thread_spec.designation} with type '{thread_type}'")

    # Step 1: Query available sizes for this thread type
    try:
        all_sizes = thread_data_query.allSizes(thread_type)
        logger.info(f"Available sizes for '{thread_type}': {list(all_sizes)[:10]}")
    except Exception as exc:
        raise FeatureOperationError(f"Failed to query thread sizes: {exc}") from exc

    # Step 2: Match our thread to a size value
    # For M6, we want to find "6" or "6.0" in the sizes list
    # For 1/4-20, we want to find "1/4" or "0.25"
    target_size = None
    nominal_mm = thread_spec.nominal_diameter

    # Try various formats
    size_candidates = []
    if thread_spec.is_metric:
        # Metric: try "6", "6.0", "6.00"
        size_candidates = [
            str(int(nominal_mm)),  # "6"
            f"{nominal_mm:.1f}",   # "6.0"
            f"{nominal_mm:.2f}",   # "6.00"
        ]
    else:
        # Imperial: try fractional and decimal formats
        # For 1/4-20: try "1/4", "0.25", etc.
        size_candidates = [
            thread_spec.fusion_size.split("-")[0],  # "1/4" from "1/4-20"
            f"{nominal_mm / 25.4:.4f}",  # decimal inches
            f"{nominal_mm / 25.4:.2f}",  # decimal inches (2 decimals)
        ]

    for candidate in size_candidates:
        for size_value in all_sizes:
            if str(size_value).strip() == candidate.strip():
                target_size = size_value
                logger.info(f"Matched size: '{candidate}' → '{size_value}'")
                break
        if target_size:
            break

    if not target_size:
        raise FeatureOperationError(
            f"Cannot find size for {thread_spec.designation}. "
            f"Tried: {size_candidates}. Available: {list(all_sizes)[:10]}"
        )

    # Step 3: Query available designations for this size
    try:
        all_designations = thread_data_query.allDesignations(thread_type, target_size)
        logger.info(f"Available designations for size '{target_size}': {list(all_designations)}")
    except Exception as exc:
        raise FeatureOperationError(
            f"Failed to query thread designations for size '{target_size}': {exc}"
        ) from exc

    # Step 4: Pick the designation that matches our pitch
    target_designation = None
    pitch_str = f"{thread_spec.pitch:.1f}" if thread_spec.is_metric else None

    for designation in all_designations:
        designation_str = str(designation)
        # For metric, look for matching pitch (e.g., "M6x1.0" contains "1.0")
        if thread_spec.is_metric and pitch_str and pitch_str in designation_str:
            target_designation = designation
            logger.info(f"Matched designation by pitch: '{designation}'")
            break
        # For imperial, sizes already include pitch (e.g., "1/4-20")
        elif not thread_spec.is_metric and thread_spec.fusion_size in designation_str:
            target_designation = designation
            logger.info(f"Matched designation: '{designation}'")
            break

    # Fallback: use first designation if no exact match
    if not target_designation and len(all_designations) > 0:
        target_designation = all_designations[0]
        logger.warning(
            f"No exact designation match for pitch {thread_spec.pitch}mm, "
            f"using first available: '{target_designation}'"
        )

    if not target_designation:
        raise FeatureOperationError(
            f"No designations available for {thread_spec.designation} (size: {target_size})"
        )

    # Step 5: Query available classes for this designation
    try:
        all_classes = thread_data_query.allClasses(is_internal, thread_type, target_designation)
        logger.info(f"Available classes for '{target_designation}': {list(all_classes)}")
    except Exception as exc:
        raise FeatureOperationError(
            f"Failed to query thread classes for '{target_designation}': {exc}"
        ) from exc

    # Pick the best class based on thread type and direction
    # Internal (tapped holes): 6H (metric), 2B (imperial)
    # External (bolts/screws): 6g (metric), 2A (imperial)
    target_class = None
    if is_internal:
        preferred_class = "6H" if thread_spec.is_metric else "2B"
    else:
        preferred_class = "6g" if thread_spec.is_metric else "2A"

    for thread_class in all_classes:
        if str(thread_class) == preferred_class:
            target_class = thread_class
            break

    # Fallback: use first class
    if not target_class and len(all_classes) > 0:
        target_class = all_classes[0]
        logger.info(f"Using first available class: '{target_class}'")

    if not target_class:
        raise FeatureOperationError(
            f"No thread classes available for {target_designation}"
        )

    # Step 6: Create ThreadInfo
    logger.info(
        f"Creating ThreadInfo: type='{thread_type}', designation='{target_designation}', "
        f"class='{target_class}'"
    )

    try:
        thread_info = thread_features.createThreadInfo(
            is_internal,
            thread_type,
            target_designation,
            target_class
        )

        if not thread_info:
            raise FeatureOperationError(
                f"createThreadInfo returned None for {target_designation}"
            )

        logger.info(f"Successfully created ThreadInfo for {thread_spec.designation}")
        return thread_info

    except Exception as exc:
        raise FeatureOperationError(
            f"Failed to create thread info for {target_designation} "
            f"(type: {thread_type}, class: {target_class}): {exc}"
        ) from exc


def _require_design(app: Optional[adsk.core.Application]) -> adsk.fusion.Design:
    if not app:
        raise FeatureOperationError("Fusion application is not available.")

    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise FeatureOperationError("No active design is available. Open or create a design and try again.")
    return design


def create_pattern_feature(
    app: Optional[adsk.core.Application],
    pattern_type: str,
    feature_tokens: List[str],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a rectangular or circular pattern from existing timeline features.
    
    Args:
        app: Fusion 360 application instance
        pattern_type: 'rectangular' or 'circular'
        feature_tokens: List of feature entity tokens to pattern
        params: Dictionary containing all pattern parameters from backend preparation:
            - anchor_point_cm: [x, y, z] anchor point in cm
            - feature_name: Optional name for the pattern feature
            For rectangular patterns:
                - count_x: Number of instances along primary direction
                - spacing_x_cm: Spacing in cm along primary direction
                - primary_axis: Axis name ("x", "y", or "z") for primary direction
                - count_y: Optional instances along secondary direction
                - spacing_y_cm: Optional spacing along secondary direction
                - secondary_axis: Optional axis name ("x", "y", or "z") for secondary direction
            For circular patterns:
                - rotation_count: Number of instances
                - rotation_angle_deg: Total sweep angle in degrees
                - axis_direction: [x, y, z] axis direction vector
                
    Returns:
        Dictionary with pattern creation results
    """
    design = _require_design(app)
    root_comp = design.rootComponent
    timeline = design.timeline

    # Resolve feature tokens to timeline objects
    resolved_features = adsk.core.ObjectCollection.create()
    for token in feature_tokens:
        found = False
        for i in range(timeline.count):
            timeline_obj = timeline.item(i)
            if timeline_obj.entity and hasattr(timeline_obj.entity, 'entityToken'):
                if timeline_obj.entity.entityToken == token:
                    resolved_features.add(timeline_obj.entity)
                    found = True
                    break
        if not found:
            raise FeatureOperationError(f"Could not resolve feature token: {token}")

    if resolved_features.count == 0:
        raise FeatureOperationError("No features resolved for patterning.")

    # Determine which component owns these features
    # All features must belong to the same component
    feature_component = None
    for i in range(resolved_features.count):
        feature = resolved_features.item(i)
        comp = getattr(feature, "parentComponent", None)
        if comp is None:
            comp = root_comp

        if feature_component is None:
            feature_component = comp
        elif feature_component != comp:
            raise FeatureOperationError(
                "Cannot pattern features from different components. "
                "All features must belong to the same component."
            )

    if feature_component is None:
        feature_component = root_comp

    # Use the features collection from the correct component
    features = feature_component.features
    
    # Get anchor point
    anchor_point_cm = params.get("anchor_point_cm", [0, 0, 0])
    anchor_point = adsk.core.Point3D.create(
        anchor_point_cm[0],
        anchor_point_cm[1],
        anchor_point_cm[2]
    )
    
    feature_name = params.get("feature_name", "")
    
    try:
        if pattern_type == "rectangular":
            return _create_rectangular_pattern(
                feature_component, features, resolved_features, params, feature_name
            )
        elif pattern_type == "circular":
            return _create_circular_pattern(
                feature_component, features, resolved_features, anchor_point, params, feature_name
            )
        else:
            raise FeatureOperationError(f"Unsupported pattern type: {pattern_type}")

    except Exception as exc:
        raise FeatureOperationError(f"Failed to create {pattern_type} pattern: {exc}") from exc


def _create_rectangular_pattern(
    component: adsk.fusion.Component,
    features: adsk.fusion.Features,
    input_features: adsk.core.ObjectCollection,
    params: Dict[str, Any],
    feature_name: str
) -> Dict[str, Any]:
    """Create a rectangular pattern feature using construction axes.

    Directions are derived from the primary_axis / secondary_axis keys
    in *params* (string values "x", "y", or "z") which map directly to
    the component's built-in construction axes.  This avoids the previous
    approach of projecting sketch lines from an anchor point, which was
    brittle when the anchor came from body bounds rather than the seed
    feature.
    """
    rect_patterns = features.rectangularPatternFeatures

    def _axis_from_vector(vec: Any) -> Optional[str]:
        if not isinstance(vec, (list, tuple)) or len(vec) < 3:
            return None
        try:
            comps = [float(vec[0]), float(vec[1]), float(vec[2])]
        except (TypeError, ValueError):
            return None
        abs_comps = [abs(c) for c in comps]
        if max(abs_comps) <= 0:
            return None
        idx = abs_comps.index(max(abs_comps))
        return ("x", "y", "z")[idx]

    # Map axis name → built-in construction axis (deterministic, no anchor needed)
    axis_map = {
        "x": component.xConstructionAxis,
        "y": component.yConstructionAxis,
        "z": component.zConstructionAxis,
    }

    primary_axis = params.get("primary_axis")
    if not primary_axis:
        primary_axis = _axis_from_vector(params.get("primary_direction")) or "x"
    primary_axis = str(primary_axis).lower()
    if primary_axis.startswith(("+", "-")):
        primary_axis = primary_axis[1:]
    direction_one = axis_map.get(primary_axis, component.xConstructionAxis)

    qty1 = adsk.core.ValueInput.createByString(str(params.get("count_x", 2)))
    step1 = adsk.core.ValueInput.createByString(f"{params.get('spacing_x_cm', 1.0)} cm")

    rect_input = rect_patterns.createInput(
        input_features,
        direction_one,
        qty1,
        step1,
        adsk.fusion.PatternDistanceType.SpacingPatternDistanceType
    )

    # Handle second direction if provided
    count_y = params.get("count_y")
    if count_y and count_y >= 2:
        secondary_axis = params.get("secondary_axis")
        if not secondary_axis:
            secondary_axis = _axis_from_vector(params.get("secondary_direction")) or "y"
        secondary_axis = str(secondary_axis).lower()
        if secondary_axis.startswith(("+", "-")):
            secondary_axis = secondary_axis[1:]
        direction_two = axis_map.get(secondary_axis, component.yConstructionAxis)

        qty2 = adsk.core.ValueInput.createByString(str(count_y))
        step2 = adsk.core.ValueInput.createByString(f"{params.get('spacing_y_cm', 1.0)} cm")
        rect_input.setDirectionTwo(direction_two, qty2, step2)
    
    # Create the pattern
    pattern_feature = rect_patterns.add(rect_input)
    
    if feature_name:
        pattern_feature.name = feature_name
    elif count_y and count_y >= 2:
        pattern_feature.name = f"Rectangular Pattern {params.get('count_x', 2)}×{count_y}"
    else:
        pattern_feature.name = f"Rectangular Pattern ×{params.get('count_x', 2)}"
    
    total_count = params.get('count_x', 2)
    if count_y and count_y >= 2:
        total_count *= count_y
    
    return {
        "message": f"Created rectangular pattern with {total_count} instances",
        "pattern_type": "rectangular",
        "instance_count": total_count,
        "feature_token": pattern_feature.entityToken if hasattr(pattern_feature, 'entityToken') else None,
    }


def _create_circular_pattern(
    component: adsk.fusion.Component,
    features: adsk.fusion.Features,
    input_features: adsk.core.ObjectCollection,
    anchor_point: adsk.core.Point3D,
    params: Dict[str, Any],
    feature_name: str
) -> Dict[str, Any]:
    """Create a circular pattern feature."""
    circ_patterns = features.circularPatternFeatures

    # Get the axis from backend params (e.g., "x", "y", "z")
    axis_key = params.get("axis", "z").lower()

    # Use component's built-in construction axes instead of creating custom ones
    # Pattern API requires standard construction axes, not custom axes through arbitrary points
    if axis_key == "x":
        construction_axis = component.xConstructionAxis
    elif axis_key == "y":
        construction_axis = component.yConstructionAxis
    else:  # "z" or default
        construction_axis = component.zConstructionAxis
    
    # Create circular pattern input using 2-parameter signature
    rotation_count = params.get("rotation_count", 3)
    rotation_angle_deg = params.get("rotation_angle_deg", 360.0)

    # Use the 2-parameter createInput signature, then set properties
    # createInput(entities, axis) - returns input object to configure
    circ_input = circ_patterns.createInput(input_features, construction_axis)

    # Set pattern properties using ValueInputs with string format
    circ_input.quantity = adsk.core.ValueInput.createByString(str(rotation_count))
    circ_input.totalAngle = adsk.core.ValueInput.createByString(f"{rotation_angle_deg} deg")
    circ_input.isSymmetric = False

    # Create the pattern
    pattern_feature = circ_patterns.add(circ_input)
    
    if feature_name:
        pattern_feature.name = feature_name
    else:
        pattern_feature.name = f"Circular Pattern ×{rotation_count}"
    
    return {
        "message": f"Created circular pattern with {rotation_count} instances over {rotation_angle_deg}°",
        "pattern_type": "circular",
        "instance_count": rotation_count,
        "feature_token": pattern_feature.entityToken if hasattr(pattern_feature, 'entityToken') else None,
    }


__all__ = [
    "FeatureOperationError",
    "apply_fillet",
    "apply_chamfer",
    "create_simple_hole",
    "create_counterbore_hole",
    "create_tapped_hole",
    "create_external_thread",
    "create_pattern_feature",
]
