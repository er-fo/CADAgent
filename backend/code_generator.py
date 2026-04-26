"""
Utilities for translating tool calls from the LLM into executable Fusion 360 Python code.

The module exposes templated code snippets for each supported tool, performs strict
parameter validation, and runs a syntax check before returning generated code. Errors are
surfaced with clear, LLM-friendly messages to keep the agent loop resilient.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)


class CodeGenerationError(Exception):
    """Raised when we cannot translate a tool call into valid Python code."""


# =============================================================================
# Backward-compatible parameter name mapping
# =============================================================================
# Maps new ref-based names to legacy token-based names for transition period.
# The normalizer converts new names to legacy names for handler compatibility.

PARAM_ALIASES: Dict[str, Dict[str, str]] = {
    # Tool name -> {new_name: legacy_name}
    # Selection/feature tools
    "apply_fillet": {"edge_refs": "entity_tokens"},
    "apply_chamfer": {"edge_refs": "entity_tokens"},
    "select_edges": {"edge_refs": "entity_tokens"},
    "select_faces": {"face_refs": "entity_tokens"},
    "select_bodies": {"body_refs": "entity_tokens"},
    # NOTE: create_shell intentionally omitted - entity_tokens is ambiguous
    # (could be faces OR bodies). Downstream code must inspect token kinds.
    
    # Hole/thread tools
    "create_simple_hole": {"face_ref": "face_token"},
    "create_counterbore_hole": {"face_ref": "face_token"},
    "create_tapped_hole": {"face_ref": "face_token"},
    "create_external_thread": {"face_ref": "face_token"},
    
    # Construction plane
    "create_construction_plane": {
        "reference_face_ref": "reference_face_token",
        "reference_edge_ref": "reference_edge_token",
        "face_ref": "face_token",
    },
    
    # Pattern feature
    "create_pattern_feature": {"feature_refs": "feature_tokens"},
}


def _normalize_param_names(tool_name: str, parameters: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalize parameter names to support both new and legacy field names.

    New names (e.g., face_ref, edge_refs) are preferred, but legacy names
    (e.g., face_token, entity_tokens) are accepted for backward compatibility.

    Returns a new dict with normalized parameter names.
    """
    result = dict(parameters)

    aliases = PARAM_ALIASES.get(tool_name, {})
    for new_name, legacy_name in aliases.items():
        # If new name is present but legacy name is not, convert new to legacy
        if legacy_name not in result and new_name in result:
            result[legacy_name] = result.pop(new_name)
            logger.debug(f"Normalized {tool_name}: {new_name} -> {legacy_name}")

    return result


VALID_PLANES: Dict[str, str] = {
    "XY": "xYConstructionPlane",
    "XZ": "xZConstructionPlane",
    "YZ": "yZConstructionPlane",
}

VALID_EXTRUDE_OPERATIONS = {"NewBody", "Join", "Cut"}
VALID_REVOLVE_OPERATIONS = {"NewBody", "Join", "Cut", "Intersect"}


def _load_templates() -> Dict[str, str]:
    """Return the templated Python snippets for each tool call."""
    return {
        "create_construction_plane": dedent(
            """\
            # Create construction plane: {plane_id} (mode: {mode})
            # Check if plane already exists (duplicate creation = no-op)
            _plane_id = "{plane_id}"
            _plane_exists = _plane_id in plane_manager.planes

            if not _plane_exists:
                plane_manager.create_plane(
                    plane_id=_plane_id,
                    mode="{mode}",
                    description="{description}",
                    {plane_params}
                )

            # Report whether this was a no-op (duplicate creation)
            _result = {{
                "no_op": _plane_exists,
                "plane_id": _plane_id,
                "mode": "{mode}"
            }}
            """
        ),
        "create_sketch": dedent(
            """\
            # Create sketch: {sketch_id} on plane: {plane_id}
            _target_plane = plane_manager.get_plane("{plane_id}")
            {sketch_id} = rootComp.sketches.add(_target_plane)
            if {sketch_id} and "{sketch_name}":
                {sketch_id}.name = "{sketch_name}"
            
            # Capture sketch orientation and approximate plane location for LLM feedback
            _x_dir = {sketch_id}.xDirection
            _y_dir = {sketch_id}.yDirection
            _normal = {sketch_id}.xDirection.crossProduct({sketch_id}.yDirection)

            _plane_origin_world = None
            try:
                _origin_pt = getattr({sketch_id}, "originPoint", None)
                _wg = getattr(_origin_pt, "worldGeometry", None) if _origin_pt else None
                if _wg:
                    _plane_origin_world = [round(_wg.x, 4), round(_wg.y, 4), round(_wg.z, 4)]
            except Exception:
                _plane_origin_world = None

            _plane_name = ""
            try:
                _plane_name = getattr(_target_plane, "name", "") or ""
            except Exception:
                _plane_name = ""

            _result = {{
                "sketch_id": "{sketch_id}",
                "sketch_name": "{sketch_name}",
                "plane_id_input": "{plane_id}",
                "plane_name": _plane_name,
                "plane_origin_world": _plane_origin_world,
                "orientation": {{
                    "u_axis_world": [round(_x_dir.x, 4), round(_x_dir.y, 4), round(_x_dir.z, 4)],
                    "v_axis_world": [round(_y_dir.x, 4), round(_y_dir.y, 4), round(_y_dir.z, 4)],
                    "extrude_positive_direction": [round(_normal.x, 4), round(_normal.y, 4), round(_normal.z, 4)]
                }}
            }}
            """
        ),
        "add_circle": dedent(
            """\
            # Add circle to {sketch_id} at ({center_u}, {center_v}) in sketch space
            _center = adsk.core.Point3D.create({center_u}, {center_v}, 0)
            _circle = {sketch_id}.sketchCurves.sketchCircles.addByCenterRadius(_center, {radius})
            _result = {{
                "success": True,
                "sketch_id": "{sketch_id}",
                "kind": "circle",
                "entity_token": str(getattr(_circle, "entityToken", "")),
                "circle_id": "{circle_id}",
                "point_tokens": {{
                    "center": str(getattr(_circle.centerSketchPoint, "entityToken", ""))
                }}
            }}
            """
        ),
        "add_line": dedent(
            """\
            # Add line to {sketch_id} in sketch space
            _start = adsk.core.Point3D.create({start_u}, {start_v}, 0)
            _end = adsk.core.Point3D.create({end_u}, {end_v}, 0)
            _line = {sketch_id}.sketchCurves.sketchLines.addByTwoPoints(_start, _end)
            _result = {{
                "success": True,
                "sketch_id": "{sketch_id}",
                "kind": "line",
                "entity_token": str(getattr(_line, "entityToken", "")),
                "line_id": "{line_id}",
                "point_tokens": {{
                    "start": str(getattr(_line.startSketchPoint, "entityToken", "")),
                    "end": str(getattr(_line.endSketchPoint, "entityToken", ""))
                }}
            }}
            """
        ),
        "add_arc": dedent(
            """\
            # Add arc to {sketch_id}: {description}
            import adsk.core

            _center = adsk.core.Point3D.create({center_u}, {center_v}, 0)
            _start = adsk.core.Point3D.create({start_u}, {start_v}, 0)
            _end = adsk.core.Point3D.create({end_u}, {end_v}, 0)
            _arc = {sketch_id}.sketchCurves.sketchArcs.addByCenterStartEnd(_center, _start, _end)

            _result = {{
                "success": True,
                "sketch_id": "{sketch_id}",
                "kind": "arc",
                "entity_token": str(getattr(_arc, "entityToken", "")),
                "arc_id": "{arc_id}",
                "point_tokens": {{
                    "start": str(getattr(_arc.startSketchPoint, "entityToken", "")),
                    "end": str(getattr(_arc.endSketchPoint, "entityToken", "")),
                    "center": str(getattr(_arc.centerSketchPoint, "entityToken", ""))
                }}
            }}
            """
        ),
        "add_rectangle": dedent(
            """\
            # Add rectangle to {sketch_id} in sketch space
            _corner1 = adsk.core.Point3D.create({corner1_u}, {corner1_v}, 0)
            _corner2 = adsk.core.Point3D.create({corner2_u}, {corner2_v}, 0)
            _rect_lines = {sketch_id}.sketchCurves.sketchLines.addTwoPointRectangle(_corner1, _corner2)
            _rect_entities = []
            for _i in range(_rect_lines.count):
                _ln = _rect_lines.item(_i)
                _sp = _ln.startSketchPoint
                _ep = _ln.endSketchPoint
                _rect_entities.append({{
                    "kind": "line",
                    "entity_token": str(getattr(_ln, "entityToken", "")),
                    "point_tokens": {{
                        "start": str(getattr(_sp, "entityToken", "")),
                        "end": str(getattr(_ep, "entityToken", ""))
                    }},
                    "start_uv": [round(_sp.geometry.x, 6), round(_sp.geometry.y, 6)],
                    "end_uv": [round(_ep.geometry.x, 6), round(_ep.geometry.y, 6)]
                }})
            _result = {{
                "success": True,
                "sketch_id": "{sketch_id}",
                "kind": "rectangle",
                "rectangle_id": "{rectangle_id}",
                "entities": _rect_entities
            }}
            """
        ),
        "list_sketch_profiles": dedent(
            """\
            # List sketch profiles for {sketch_id}
            import adsk.core
            import adsk.fusion

            if "{sketch_id}" not in globals():
                raise ValueError("Sketch '{sketch_id}' not found. Create it before listing profiles.")
            _sketch = globals()["{sketch_id}"]
            if not isinstance(_sketch, adsk.fusion.Sketch):
                raise ValueError("'{sketch_id}' is not a sketch (type: %s)." % type(_sketch).__name__)

            _profile_count = _sketch.profiles.count
            if _profile_count == 0:
                raise ValueError(
                    "Sketch '{sketch_id}' has no closed profiles. "
                    "Ensure all intended loops are closed (no gaps/overlaps) and fully define the region(s)."
                )
            _profiles_out = []

            for _i in range(_profile_count):
                _profile = _sketch.profiles.item(_i)

                _loops = _profile.profileLoops
                _loop_count = _loops.count
                _outer_count = 0
                for _j in range(_loops.count):
                    _loop = _loops.item(_j)
                    try:
                        if getattr(_loop, "isOuter", False):
                            _outer_count += 1
                    except Exception:
                        pass

                _inner_count = _loop_count - _outer_count

                _area_cm2 = None
                _centroid_world_cm = None
                try:
                    try:
                        _acc = adsk.core.CalculationAccuracy.MediumCalculationAccuracy
                    except AttributeError:
                        _acc = 1
                    _ap = _profile.areaProperties(_acc)
                    if _ap:
                        _a = getattr(_ap, "area", None)
                        if _a is not None:
                            _area_cm2 = float(_a)
                        _c = getattr(_ap, "centroid", None)
                        if _c:
                            _centroid_world_cm = [round(_c.x, 4), round(_c.y, 4), round(_c.z, 4)]
                except Exception:
                    pass

                _profiles_out.append(
                    {{
                        "index": _i,
                        "area_cm2": _area_cm2,
                        "centroid_world_cm": _centroid_world_cm,
                        "outer_loop_count": _outer_count,
                        "inner_loop_count": _inner_count,
                        "loop_count": _loop_count,
                    }}
                )

            _result = {{
                "sketch_id": "{sketch_id}",
                "profile_count": _profile_count,
                "profiles": _profiles_out,
            }}
            """
        ),
        "extrude_profile": dedent(
            """\
            # Extrude profile(s) from {sketch_id}
            import adsk.core
            import adsk.fusion

            if "{sketch_id}" not in globals():
                raise ValueError("Sketch '{sketch_id}' not found. Create it before extruding.")
            _sketch = globals()["{sketch_id}"]
            if not isinstance(_sketch, adsk.fusion.Sketch):
                raise ValueError("'{sketch_id}' is not a sketch (type: %s)." % type(_sketch).__name__)
            if _sketch.profiles.count == 0:
                raise ValueError("Sketch '{sketch_id}' has no profiles to extrude.")

            _profile_indices = {profile_indices}
            if _profile_indices is None:
                _profile_indices = [{profile_index}]

            _profiles = adsk.core.ObjectCollection.create()
            for _idx in _profile_indices:
                if _idx < 0 or _idx >= _sketch.profiles.count:
                    raise ValueError("profile_index %d is out of range (count=%d)." % (_idx, _sketch.profiles.count))
                _profiles.add(_sketch.profiles.item(_idx))

            _requested_distance = {distance}
            _actual_distance = _requested_distance
            _auto_flipped_distance = False

            def _do_extrude(_dist: float):
                _dist_val = adsk.core.ValueInput.createByReal(_dist)
                _input = extrudes.createInput(
                    _profiles,
                    adsk.fusion.FeatureOperations.{operation}FeatureOperation,
                )
                _input.setDistanceExtent(False, _dist_val)
                return extrudes.add(_input)

            try:
                extrude_feat = _do_extrude(_actual_distance)
            except Exception as _exc:
                _exc_text = str(_exc)
                if "{operation}" in ("Cut", "Intersect") and "No target body found to cut or intersect" in _exc_text:
                    # Common failure mode: sketch plane is correct but distance sign points away from the body.
                    # Retry once with distance sign flipped and report the actual distance used.
                    try:
                        _actual_distance = -_requested_distance
                        _auto_flipped_distance = True
                        extrude_feat = _do_extrude(_actual_distance)
                    except Exception as _exc2:
                        raise RuntimeError(
                            "extrude_profile failed: %s. "
                            "Also failed after flipping distance sign (requested=%s, retried=%s): %s"
                            % (_exc_text, _requested_distance, _actual_distance, _exc2)
                        )
                else:
                    raise

            if extrude_feat and extrude_feat.timelineObject and "{feature_name}":
                extrude_feat.timelineObject.name = "{feature_name}"

            # Capture created entities for registration
            # Note: ExtrudeFeature does not have .faces/.edges directly - must access via .bodies
            _created_bodies = []
            _created_faces = []
            _created_edges = []
            if extrude_feat and extrude_feat.bodies:
                for i in range(extrude_feat.bodies.count):
                    body = extrude_feat.bodies.item(i)
                    if body:
                        if body.entityToken:
                            _created_bodies.append(body.entityToken)
                        # Collect faces from each body
                        if body.faces:
                            for j in range(body.faces.count):
                                face = body.faces.item(j)
                                if face and face.entityToken:
                                    _created_faces.append(face.entityToken)
                        # Collect edges from each body
                        if body.edges:
                            for j in range(body.edges.count):
                                edge = body.edges.item(j)
                                if edge and edge.entityToken:
                                    _created_edges.append(edge.entityToken)

            _result = {{
                "requested_distance": _requested_distance,
                "actual_distance": _actual_distance,
                "auto_flipped_distance": _auto_flipped_distance,
                "created_entities": {{
                    "bodies": list(dict.fromkeys(_created_bodies)),
                    "faces": list(dict.fromkeys(_created_faces)),
                    "edges": list(dict.fromkeys(_created_edges))
                }}
            }}
            """
        ),
        "revolve_profile": dedent(
            """\
            # Revolve profile: {description}
            import math
            import adsk.core
            import adsk.fusion

            # Resolve profile
            if "{sketch_id}" not in globals():
                raise ValueError("Sketch '{sketch_id}' not found. Create it before revolving.")
            _sketch = globals()["{sketch_id}"]
            if not isinstance(_sketch, adsk.fusion.Sketch):
                raise ValueError("'{sketch_id}' is not a sketch (type: %s)." % type(_sketch).__name__)
            if _sketch.profiles.count == 0:
                raise ValueError("Sketch '{sketch_id}' has no profiles to revolve.")
            if {profile_index} < 0 or {profile_index} >= _sketch.profiles.count:
                raise ValueError("profile_index {profile_index} is out of range (count=%d)." % _sketch.profiles.count)
            profile = _sketch.profiles.item({profile_index})

            axis_spec = {axis_spec}
            axis_type = axis_spec.get("type")
            axis_entity = None

            if axis_type == "construction":
                axis_map = {{
                    "x": rootComp.xConstructionAxis,
                    "y": rootComp.yConstructionAxis,
                    "z": rootComp.zConstructionAxis,
                }}
                axis_entity = axis_map.get(axis_spec.get("axis"))
                if axis_entity is None:
                    raise ValueError("Invalid construction axis: %s" % axis_spec.get("axis"))

            elif axis_type == "sketch_line":
                axis_sketch_id = axis_spec.get("sketch_id")
                if axis_sketch_id not in globals():
                    raise ValueError("Axis sketch '%s' not found." % axis_sketch_id)
                _axis_sketch = globals()[axis_sketch_id]
                if not isinstance(_axis_sketch, adsk.fusion.Sketch):
                    raise ValueError(
                        "Axis reference '%s' is not a sketch (type: %s)." % (axis_sketch_id, type(_axis_sketch).__name__)
                    )
                line_index = axis_spec.get("line_index", 0)
                line_count = _axis_sketch.sketchCurves.sketchLines.count
                if line_index < 0 or line_index >= line_count:
                    raise ValueError(
                        "line_index %d is out of range for axis sketch '%s' (lines=%d)." % (line_index, axis_sketch_id, line_count)
                    )
                axis_entity = _axis_sketch.sketchCurves.sketchLines.item(line_index)

            elif axis_type == "edge":
                axis_token = axis_spec.get("edge_token", "")
                entities = design.findEntityByToken(axis_token)
                if not entities:
                    raise ValueError("Edge token '%s' not found for axis reference." % axis_token)
                axis_entity = entities[0]

            elif axis_type == "face":
                axis_token = axis_spec.get("face_token", "")
                entities = design.findEntityByToken(axis_token)
                if not entities:
                    raise ValueError("Face token '%s' not found for axis reference." % axis_token)
                axis_entity = entities[0]

            else:
                raise ValueError("Unsupported axis type: %s" % axis_type)

            if axis_entity is None:
                raise ValueError("Axis resolution failed; no valid axis entity located.")

            # Face axes must define an axis (e.g., cylindrical/conical/torus faces)
            if axis_type == "face":
                geom = getattr(axis_entity, "geometry", None)
                surf_type = getattr(geom, "surfaceType", None) if geom else None
                allowed = {{
                    adsk.core.SurfaceTypes.CylinderSurfaceType,
                    adsk.core.SurfaceTypes.ConeSurfaceType,
                    adsk.core.SurfaceTypes.TorusSurfaceType,
                }}
                if surf_type not in allowed:
                    raise ValueError(
                        "Face axis must be cylindrical/conical/torus. "
                        "Got surfaceType=%s." % surf_type
                    )

            operation_map = {{
                "NewBody": adsk.fusion.FeatureOperations.NewBodyFeatureOperation,
                "Join": adsk.fusion.FeatureOperations.JoinFeatureOperation,
                "Cut": adsk.fusion.FeatureOperations.CutFeatureOperation,
                "Intersect": adsk.fusion.FeatureOperations.IntersectFeatureOperation,
            }}
            operation_str = {operation!r}
            operation_enum = operation_map.get(operation_str)
            if operation_enum is None:
                raise ValueError("Invalid operation: %s" % operation_str)

            revolve_feats = rootComp.features.revolveFeatures
            revolve_input = revolve_feats.createInput(profile, axis_entity, operation_enum)
            revolve_input.isSolid = bool({is_solid})

            extent_spec = {extent_spec}
            mode = extent_spec.get("mode")
            if mode == "full":
                _angle_full = adsk.core.ValueInput.createByReal(2 * math.pi)
                revolve_input.setAngleExtent(False, _angle_full)
            elif mode == "angle":
                angle_deg = extent_spec.get("angle_degrees")
                if angle_deg == 0:
                    raise ValueError("angle_degrees must be non-zero.")
                _angle_val = adsk.core.ValueInput.createByReal(math.radians(angle_deg))
                revolve_input.setAngleExtent(bool(extent_spec.get("symmetric", False)), _angle_val)
            elif mode == "two_sides_angle":
                angle1 = extent_spec.get("angle1_degrees")
                angle2 = extent_spec.get("angle2_degrees")
                if angle1 == 0 or angle2 == 0:
                    raise ValueError("angle1_degrees and angle2_degrees must be non-zero.")
                _angle1_val = adsk.core.ValueInput.createByReal(math.radians(angle1))
                _angle2_val = adsk.core.ValueInput.createByReal(math.radians(angle2))
                revolve_input.setTwoSidesAngleExtent(_angle1_val, _angle2_val)
            elif mode == "to":
                target_token = extent_spec.get("to_entity_token", "")
                target_entities = design.findEntityByToken(target_token)
                if not target_entities:
                    raise ValueError("Target entity token '%s' not found for revolve extent." % target_token)
                revolve_input.setOneSideToExtent(target_entities[0])
            elif mode == "two_sides_to":
                token1 = extent_spec.get("to_entity1_token", "")
                token2 = extent_spec.get("to_entity2_token", "")
                targets1 = design.findEntityByToken(token1)
                targets2 = design.findEntityByToken(token2)
                if not targets1:
                    raise ValueError("Target entity token '%s' not found for first side." % token1)
                if not targets2:
                    raise ValueError("Target entity token '%s' not found for second side." % token2)
                revolve_input.setTwoSidesToExtent(targets1[0], targets2[0])
            else:
                raise ValueError("Unsupported extent mode: %s" % mode)

            creation_token = {creation_occurrence_token!r}
            if creation_token:
                occurrences = design.findEntityByToken(creation_token)
                if not occurrences:
                    raise ValueError("creation_occurrence_token '%s' not found." % creation_token)
                revolve_input.creationOccurrence = occurrences[0]

            revolve_feat = revolve_feats.add(revolve_input)
            if revolve_feat and revolve_feat.timelineObject and "{feature_name}":
                revolve_feat.timelineObject.name = "{feature_name}"

            # Capture created entities for registration
            # Note: RevolveFeature does not have .faces/.edges directly - must access via .bodies
            _created_bodies = []
            _created_faces = []
            _created_edges = []
            if revolve_feat and revolve_feat.bodies:
                for i in range(revolve_feat.bodies.count):
                    body = revolve_feat.bodies.item(i)
                    if body:
                        if body.entityToken:
                            _created_bodies.append(body.entityToken)
                        # Collect faces from each body
                        if body.faces:
                            for j in range(body.faces.count):
                                face = body.faces.item(j)
                                if face and face.entityToken:
                                    _created_faces.append(face.entityToken)
                        # Collect edges from each body
                        if body.edges:
                            for j in range(body.edges.count):
                                edge = body.edges.item(j)
                                if edge and edge.entityToken:
                                    _created_edges.append(edge.entityToken)

            _result = {{
                "created_entities": {{
                    "bodies": list(dict.fromkeys(_created_bodies)),
                    "faces": list(dict.fromkeys(_created_faces)),
                    "edges": list(dict.fromkeys(_created_edges))
                }}
            }}
            """
        ),
        "jump_to_timeline_position": dedent(
            """\
            # Jump to timeline position {target_index}
            # Validate design type - timeline operations require Parametric mode
            if design.designType != adsk.fusion.DesignTypes.ParametricDesignType:
                raise ValueError(
                    "Timeline operations require Parametric design mode. "
                    "Current mode is Direct Modeling which does not expose a timeline. "
                    "Switch to Parametric mode in the Design workspace to enable timeline operations."
                )

            # Check if we're already at the target position (no-op case)
            _current_position = design.timeline.markerPosition
            _target_position = {target_index}
            _is_noop = (_current_position == _target_position)

            if not _is_noop:
                design.timeline.markerPosition = _target_position
                design.timeline.deleteAllAfterMarker()

            # Report whether this was a no-op
            _result = {{
                "no_op": _is_noop,
                "current_position": _current_position,
                "target_position": _target_position
            }}
            """
        ),
        "delete_feature": dedent(
            """\
            # Delete feature: {description}

            # Validate design type - timeline operations require Parametric mode
            if design.designType != adsk.fusion.DesignTypes.ParametricDesignType:
                raise ValueError(
                    "Timeline operations require Parametric design mode. "
                    "Current mode is Direct Modeling which does not expose a timeline. "
                    "Switch to Parametric mode in the Design workspace to enable timeline operations."
                )

            _feature_token = "{feature_token}"
            _expected_name = "{expected_name}"
            _expected_index = {expected_timeline_index}

            # Resolve feature from token
            _entities = design.findEntityByToken(_feature_token)
            if not _entities or len(_entities) == 0:
                raise ValueError(f"Feature token not found in design. Token may be stale - call list_features to get current tokens.")

            # Disambiguate: reject multiple matches to prevent deleting the wrong entity
            if len(_entities) > 1:
                raise ValueError(
                    f"Ambiguous token: findEntityByToken returned {{len(_entities)}} entities. "
                    f"Cannot safely determine which to delete. Call list_features for current tokens."
                )

            # Get the entity and check if it has a timeline object
            _entity = _entities[0]
            _timeline_obj = getattr(_entity, "timelineObject", None)
            if not _timeline_obj:
                raise ValueError(f"Entity does not have a timeline object and cannot be deleted via timeline. Entity type: {{type(_entity).__name__}}")

            # Enforce feature-only deletion: reject sketches and other non-Feature timeline entities.
            # Fusion object types for Features contain 'Feature' (e.g., 'adsk::fusion::ExtrudeFeature').
            # Sketches are 'adsk::fusion::Sketch'.
            _obj_type = getattr(_entity, "objectType", "") or ""
            if "Feature" not in _obj_type:
                raise ValueError(
                    f"Only features can be deleted with this tool. "
                    f"Entity type '{{_obj_type}}' is not a feature. "
                    f"Sketches and other non-feature timeline items are not supported."
                )

            # Extract both name fields for comparison:
            #   timeline_name = timelineObject.name (e.g., "Extrude1")
            #   entity_name   = entity.name         (e.g., "Extrude1")
            # list_features exposes both as 'timeline_name' and 'name'.
            _timeline_name = getattr(_timeline_obj, "name", "") or ""
            _entity_name = getattr(_entity, "name", "") or ""
            _feature_name = _timeline_name or _entity_name or "Unknown"
            _timeline_index = getattr(_timeline_obj, "index", -1)

            # Safety verification (if expected values provided).
            # Accept match against EITHER timeline_name or entity_name to avoid
            # false rejections when the caller used one vs. the other from list_features.
            if _expected_name and _expected_name != _timeline_name and _expected_name != _entity_name:
                raise ValueError(
                    f"Safety check failed: expected feature name '{{_expected_name}}' but found "
                    f"timeline_name='{{_timeline_name}}', entity_name='{{_entity_name}}'. "
                    f"Aborting deletion to prevent accidental removal of wrong feature."
                )

            if _expected_index is not None and _expected_index != _timeline_index:
                raise ValueError(
                    f"Safety check failed: expected timeline index {{_expected_index}} but found {{_timeline_index}}. "
                    f"Timeline may have changed. Call list_features to get current state."
                )

            # Perform deletion.
            # deleteMe() returns True on success, False on failure.
            # Some Fusion API paths return None on success, so treat None as success.
            _delete_result = _timeline_obj.entity.deleteMe()

            if _delete_result is False:
                raise ValueError(f"deleteMe() returned False for feature '{{_feature_name}}'. Feature may have dependencies or be locked.")

            _result = {{
                "success": True,
                "message": f"Deleted feature: {{_feature_name}} (was at timeline index {{_timeline_index}})",
                "deleted_feature_name": _feature_name,
                "deleted_timeline_index": _timeline_index,
            }}
            """
        ),
        "create_loft": dedent(
            """\
            # Create loft feature: {description}
            import adsk.core
            import adsk.fusion
            import logging

            logger = logging.getLogger(__name__)

            # Validate profile IDs
            profile_ids = {profile_ids}
            if not isinstance(profile_ids, list) or len(profile_ids) < 2:
                raise ValueError(f"Loft requires at least 2 profiles (received {{len(profile_ids) if isinstance(profile_ids, list) else 0}})")

            # Access sketches from global namespace (sketches are stored directly as variables)
            profiles = []
            available_sketches = [k for k, v in globals().items() if isinstance(v, adsk.fusion.Sketch)]

            for i, prof_id in enumerate(profile_ids):
                if prof_id not in globals():
                    raise ValueError(f"Profile '{{prof_id}}' not found. Available sketches: {{available_sketches}}")

                sketch = globals()[prof_id]
                if not isinstance(sketch, adsk.fusion.Sketch):
                    raise ValueError(f"'{{prof_id}}' is not a sketch (type: {{type(sketch).__name__}})")

                if sketch.profiles.count == 0:
                    raise ValueError(f"Sketch '{{prof_id}}' has no closed profiles. Create closed loops before lofting.")

                # Use first profile by default (most common case)
                profiles.append(sketch.profiles.item(0))

            # Map schema operation strings to Fusion API enum members
            operation_map = {{
                "NewBody": adsk.fusion.FeatureOperations.NewBodyFeatureOperation,
                "Join": adsk.fusion.FeatureOperations.JoinFeatureOperation,
                "Cut": adsk.fusion.FeatureOperations.CutFeatureOperation,
            }}
            operation_str = {operation!r}
            if operation_str not in operation_map:
                raise ValueError(f"Invalid operation: {{operation_str}}. Must be 'NewBody', 'Join', or 'Cut'.")
            operation_enum = operation_map[operation_str]

            # Create loft input
            loft_feats = rootComp.features.loftFeatures
            loft_input = loft_feats.createInput(operation_enum)

            # Add profiles in order
            for profile in profiles:
                loft_input.loftSections.add(profile)

            # Safe defaults with fallback for complex geometry
            loft_input.isClosed = False
            loft_input.isTangentEdgesMerged = True

            description_text = {description!r}
            try:
                # Attempt 1: Solid loft (preferred)
                loft_input.isSolid = True
                loft_feature = loft_feats.add(loft_input)
            except Exception as solid_error:
                # Attempt 2: Surface loft fallback (for non-capping profiles)
                try:
                    loft_input.isSolid = False
                    loft_feature = loft_feats.add(loft_input)
                    logger.warning(f"Loft '{{description_text}}' fell back to surface mode. Original error: {{solid_error}}")
                except Exception as surface_error:
                    raise RuntimeError(
                        f"Loft failed in both solid and surface modes. "
                        f"Context: {{description_text}}. "
                        f"Solid error: {{solid_error}}. Surface error: {{surface_error}}"
                    )

            # Optional feature naming
            feature_name = {feature_name!r}
            if feature_name:
                loft_feature.timelineObject.name = feature_name

            logger.info(f"Created loft feature: {{loft_feature.timelineObject.name}}. Context: {{description_text}}")

            # Capture created entities for registration
            # Note: LoftFeature does not have .faces/.edges directly - must access via .bodies
            _created_bodies = []
            _created_faces = []
            _created_edges = []
            if loft_feature and loft_feature.bodies:
                for i in range(loft_feature.bodies.count):
                    body = loft_feature.bodies.item(i)
                    if body:
                        if body.entityToken:
                            _created_bodies.append(body.entityToken)
                        # Collect faces from each body
                        if body.faces:
                            for j in range(body.faces.count):
                                face = body.faces.item(j)
                                if face and face.entityToken:
                                    _created_faces.append(face.entityToken)
                        # Collect edges from each body
                        if body.edges:
                            for j in range(body.edges.count):
                                edge = body.edges.item(j)
                                if edge and edge.entityToken:
                                    _created_edges.append(edge.entityToken)

            # Return structured result
            _result = {{
                "feature_name": loft_feature.timelineObject.name,
                "body_count": loft_feature.bodies.count,
                "face_count": len(_created_faces),
                "profile_count": len(profiles),
                "is_solid": loft_feature.isSolid if hasattr(loft_feature, 'isSolid') else True,
                "description": description_text,
                "created_entities": {{
                    "bodies": list(dict.fromkeys(_created_bodies)),
                    "faces": list(dict.fromkeys(_created_faces)),
                    "edges": list(dict.fromkeys(_created_edges))
                }}
            }}
            """
        ),
    }


OPERATION_TEMPLATES = _load_templates()


@dataclass
class ParameterRule:
    """Specification for validating and normalising a parameter."""

    required: bool
    validator: Any
    default: Optional[Any] = None


def _validate_nonempty_string(value: Any, field_name: str) -> str:
    """Validate and escape a required string parameter, rejecting empty values."""
    if not value or not str(value).strip():
        raise CodeGenerationError(f'"{field_name}" must be a non-empty string.')
    return _escape_string_for_python(str(value).strip())


def _validate_identifier(value: Any, field_name: str) -> str:
    """Ensure identifiers used in templates are safe Python variable names."""
    if not isinstance(value, str) or not value.strip():
        raise CodeGenerationError(f'"{field_name}" must be a non-empty string.')

    identifier = value.strip()
    if not identifier.replace("_", "").isalnum() or identifier[0].isdigit():
        raise CodeGenerationError(
            f'"{field_name}" must be a valid identifier (letters, numbers, underscores, '
            "cannot start with a number)."
        )
    return identifier


def _validate_plane(value: Any) -> str:
    if not isinstance(value, str):
        raise CodeGenerationError('"plane" must be provided as a string.')

    upper = value.strip().upper()
    if upper not in VALID_PLANES:
        allowed = ", ".join(sorted(VALID_PLANES.keys()))
        raise CodeGenerationError(f'Unknown plane "{value}". Expected one of: {allowed}.')
    return upper


def _validate_mode(value: Any) -> str:
    """Validate construction plane mode."""
    if not isinstance(value, str):
        raise CodeGenerationError('"mode" must be a string.')

    mode = value.strip().lower()
    valid_modes = {"datum", "offset_from_datum", "angle_to_edge", "face_normal"}
    if mode not in valid_modes:
        raise CodeGenerationError(
            f'Invalid mode "{value}". Expected one of: {", ".join(sorted(valid_modes))}'
        )
    return mode


def _validate_float(value: Any, field_name: str) -> str:
    if isinstance(value, bool):
        raise CodeGenerationError(f'"{field_name}" must be numeric, not boolean.')

    if isinstance(value, (int, float)):
        return _format_numeric(float(value))

    if isinstance(value, str):
        try:
            return _format_numeric(float(value.strip()))
        except ValueError as exc:
            raise CodeGenerationError(f'"{field_name}" must be numeric.') from exc

    raise CodeGenerationError(f'"{field_name}" must be numeric.')


def _validate_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise CodeGenerationError(f'"{field_name}" must be an integer, not boolean.')

    if isinstance(value, int):
        return value

    if isinstance(value, float) and value.is_integer():
        return int(value)

    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError as exc:
            raise CodeGenerationError(f'"{field_name}" must be an integer.') from exc
        return parsed

    raise CodeGenerationError(f'"{field_name}" must be an integer.')


def _validate_int_list(value: Any, field_name: str) -> List[int]:
    """
    Validate a list of non-negative integers.

    Accepts:
    - Python list/tuple of ints
    - JSON-stringified list (defensive)
    """
    # Handle JSON-stringified arrays from LLMs (defensive)
    if isinstance(value, str):
        value_str = value.strip()
        if value_str.startswith("[") and value_str.endswith("]"):
            try:
                parsed = json.loads(value_str)
                value = parsed
            except json.JSONDecodeError:
                # Fall through to normal validation for non-JSON strings
                pass

    if not isinstance(value, (list, tuple)):
        raise CodeGenerationError(f'"{field_name}" must be an array of integers.')

    if not value:
        raise CodeGenerationError(f'"{field_name}" cannot be empty.')

    cleaned: List[int] = []
    for item in value:
        if isinstance(item, bool):
            raise CodeGenerationError(f'"{field_name}" entries must be integers, not boolean.')

        if isinstance(item, int):
            parsed = item
        elif isinstance(item, float) and item.is_integer():
            parsed = int(item)
        elif isinstance(item, str):
            try:
                parsed = int(item.strip())
            except ValueError as exc:
                raise CodeGenerationError(f'"{field_name}" entries must be integers.') from exc
        else:
            raise CodeGenerationError(f'"{field_name}" entries must be integers.')

        if parsed < 0:
            raise CodeGenerationError(f'"{field_name}" entries must be zero or greater.')

        cleaned.append(parsed)

    return cleaned


def _validate_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise CodeGenerationError(f'"{field_name}" must be a boolean.')


def _validate_operation(value: Any) -> str:
    if not isinstance(value, str):
        raise CodeGenerationError('"operation" must be a string.')

    op = value.strip()
    if op not in VALID_EXTRUDE_OPERATIONS:
        allowed = ", ".join(sorted(VALID_EXTRUDE_OPERATIONS))
        raise CodeGenerationError(f'Invalid operation "{value}". Expected: {allowed}.')
    return op


def _validate_revolve_operation(value: Any) -> str:
    if not isinstance(value, str):
        raise CodeGenerationError('"operation" must be a string.')

    op = value.strip()
    if op not in VALID_REVOLVE_OPERATIONS:
        allowed = ", ".join(sorted(VALID_REVOLVE_OPERATIONS))
        raise CodeGenerationError(f'Invalid operation \"{value}\". Expected: {allowed}.')
    return op


def _validate_feature_name(value: Any, field_name: str) -> str:
    """Validate feature names for timeline display."""
    if not value:
        return ""

    if not isinstance(value, str):
        raise CodeGenerationError(f'"{field_name}" must be a string.')

    name = value.strip()

    if len(name) > 100:
        raise CodeGenerationError(f'"{field_name}" must be 100 characters or less.')

    # Allow descriptive names with spaces, but prevent special characters that could break Python strings
    if '"' in name or "\\" in name:
        raise CodeGenerationError(f'"{field_name}" cannot contain double quotes or backslashes.')

    return name


def _validate_arc_geometry(
    center_u: float,
    center_v: float,
    start_u: float,
    start_v: float,
    end_u: float,
    end_v: float,
) -> Optional[str]:
    """Validate center-start-end arc geometry prior to template rendering."""
    import math

    epsilon = 1e-6
    radius_tol = 0.01

    def _points_equal(u1: float, v1: float, u2: float, v2: float) -> bool:
        return abs(u1 - u2) < epsilon and abs(v1 - v2) < epsilon

    if _points_equal(center_u, center_v, start_u, start_v):
        return "Arc center and start point are coincident (within tolerance)."
    if _points_equal(center_u, center_v, end_u, end_v):
        return "Arc center and end point are coincident (within tolerance)."
    if _points_equal(start_u, start_v, end_u, end_v):
        return "Arc start and end points are coincident (within tolerance)."

    r_start = math.sqrt((start_u - center_u) ** 2 + (start_v - center_v) ** 2)
    r_end = math.sqrt((end_u - center_u) ** 2 + (end_v - center_v) ** 2)
    diff = abs(r_start - r_end)
    max_radius = max(r_start, r_end)

    if diff > epsilon and (max_radius < epsilon or diff / max_radius > radius_tol):
        return (
            f"Arc radii inconsistent: start={r_start:.6f}cm, end={r_end:.6f}cm "
            f"(diff={diff:.6f}cm)."
        )

    return None


def _validate_axis_spec(value: Any, field_name: str) -> Dict[str, Any]:
    """
    Validate discriminated axis union for revolve.
    Backward-compatible: accepts legacy axis strings ("x"/"y"/"z") or
    legacy objects that used type='datum_axis' or omitted type but supplied axis/axis_reference.
    """
    # Auto-default when axis is completely missing/None
    if value is None:
        return {"type": "construction", "axis": "z"}

    # Legacy shorthand: axis provided as plain string ("x"/"y"/"z")
    if isinstance(value, str):
        axis_val = value.strip().lower()

        # Check if this is a simple axis string first
        if axis_val in {"x", "y", "z"}:
            return {"type": "construction", "axis": axis_val}

        # LLM may send JSON-stringified object - try parsing it
        # Common when OpenAI API double-serializes nested objects
        if axis_val.startswith("{") and axis_val.endswith("}"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    logger.info(
                        f"Successfully parsed JSON-stringified axis object: {repr(value)}"
                    )
                    # Recursively validate the parsed object
                    return _validate_axis_spec(parsed, field_name)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse axis value as JSON: {repr(value)}, error: {e}"
                )
                # Fall through to error below

        # Invalid string value
        logger.error(
            f"Invalid axis string received: {repr(value)} "
            f"(after strip/lower: {repr(axis_val)}). "
            f"Expected one of: 'x', 'y', 'z'"
        )
        raise CodeGenerationError(
            f'Axis string must be "x", "y", or "z". Received: {repr(value)}. '
            'Alternatively, use object format: {{"type": "construction", "axis": "x"}}.'
        )

    if not isinstance(value, Mapping):
        raise CodeGenerationError(f'"{field_name}" must be an object with a "type" field.')

    # Empty object -> fall back to default construction Z (common when LLM omits axis details)
    if len(value) == 0:
        return {"type": "construction", "axis": "z"}

    # NOTE: We do NOT pre-filter fields here. Instead, we filter by the allowed set
    # for each axis type AFTER we determine the type. This correctly handles cases like:
    # {"type": "construction", "axis": "z", "line_index": 0, "edge_token": "", "face_token": ""}
    # where the LLM sends all fields but only the type-relevant ones should be validated.

    # Normalise legacy aliases
    raw_type_value = value.get("type", None)
    # If the discriminator isn't a string (e.g., {}), treat it as missing so we can fallback cleanly
    if isinstance(raw_type_value, str):
        raw_type = raw_type_value.strip().lower()
    else:
        raw_type = ""

    if raw_type in {"datum_axis", "axis"}:
        raw_type = "construction"
    if raw_type in {"world_axis", "world", "global_axis"}:
        raw_type = "construction"
    if raw_type in {"none", "null"}:
        raw_type = ""

    # Some models may emit "{}" as a string; normalize that to empty so we can fall back
    if raw_type in {"{}", "[]"}:
        raw_type = ""

    axis_type = raw_type
    valid_types = {"construction", "sketch_line", "edge", "face"}

    # Infer construction if type missing but axis-like keys exist
    if axis_type == "" and any(k in value for k in ("axis", "axis_reference", "datum_axis")):
        axis_type = "construction"

    if axis_type not in valid_types:
        # Last-resort fallback: treat unknown/empty as construction Z to keep execution flowing
        if axis_type == "" and not value:
            return {"type": "construction", "axis": "z"}
        if axis_type == "" and any(k in value for k in ("axis", "axis_reference", "datum_axis")):
            axis_type = "construction"
        else:
            raise CodeGenerationError(
                f'Invalid axis type "{axis_type}". Expected one of: construction, sketch_line, edge, face.'
            )

    cleaned: Dict[str, Any] = {"type": axis_type}

    if axis_type == "construction":
        # Filter to only allowed keys for construction axis (ignore extras from LLMs that send all fields)
        allowed = {"type", "axis", "axis_reference", "datum_axis"}
        filtered_value = {k: v for k, v in value.items() if k in allowed}
        axis_val = (
            str(
                filtered_value.get("axis")
                or filtered_value.get("axis_reference")
                or filtered_value.get("datum_axis")
                or ""
            )
            .strip()
            .lower()
        )
        if axis_val not in {"x", "y", "z"}:
            # Log actual invalid value for debugging
            logger.error(
                f"Invalid construction axis value: {repr(axis_val)} "
                f"(from object: {repr(value)}). Expected one of: 'x', 'y', 'z'"
            )
            raise CodeGenerationError(
                f'Construction axis must be "x", "y", or "z". '
                f'Received: {repr(axis_val)} from {repr(value)}'
            )
        cleaned["axis"] = axis_val

    elif axis_type == "sketch_line":
        # Filter to only allowed keys for sketch_line axis (ignore extras from LLMs that send all fields)
        allowed = {"type", "sketch_id", "line_index"}
        filtered_value = {k: v for k, v in value.items() if k in allowed}
        sketch_id = filtered_value.get("sketch_id")
        cleaned["sketch_id"] = _validate_identifier(sketch_id, "axis.sketch_id")
        line_index_raw = filtered_value.get("line_index", 0)
        line_index = _validate_int(line_index_raw, "axis.line_index")
        if line_index < 0:
            raise CodeGenerationError('"axis.line_index" must be zero or greater.')
        cleaned["line_index"] = line_index

    elif axis_type == "edge":
        # Filter to only allowed keys for edge axis (ignore extras from LLMs that send all fields)
        allowed = {"type", "edge_token"}
        filtered_value = {k: v for k, v in value.items() if k in allowed}
        token = str(filtered_value.get("edge_token", "")).strip()
        if not token:
            raise CodeGenerationError('"axis.edge_token" must be provided for edge axes.')
        cleaned["edge_token"] = _escape_string_for_python(token)

    elif axis_type == "face":
        # Filter to only allowed keys for face axis (ignore extras from LLMs that send all fields)
        allowed = {"type", "face_token"}
        filtered_value = {k: v for k, v in value.items() if k in allowed}
        token = str(filtered_value.get("face_token", "")).strip()
        if not token:
            raise CodeGenerationError('"axis.face_token" must be provided for face axes.')
        cleaned["face_token"] = _escape_string_for_python(token)

    return cleaned


def _validate_extent_spec(value: Any, field_name: str) -> Dict[str, Any]:
    """Validate discriminated extent union for revolve."""
    # Default to full revolve when omitted/None
    if value is None:
        value = {"mode": "full"}

    if not isinstance(value, Mapping):
        # LLM may send JSON-stringified object - try parsing it
        # Common when APIs double-serialize nested objects
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    logger.info(
                        f"Successfully parsed JSON-stringified extent object: {repr(value)}"
                    )
                    # Recursively validate the parsed object
                    return _validate_extent_spec(parsed, field_name)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse extent value as JSON: {repr(value)}, error: {e}"
                )
                # Fall through to error below

        raise CodeGenerationError(f'"{field_name}" must be an object with a "mode" field.')

    raw_mode = value.get("mode", "")
    # Discriminator may arrive as non-string (e.g., {}); normalize to empty so default kicks in
    if isinstance(raw_mode, str):
        mode = raw_mode.strip()
    else:
        mode = ""
    mode_lower = mode.lower()

    # Legacy/alias modes
    if mode_lower in {"none", "null", ""}:
        mode = "full"
    elif mode_lower in {"symmetric", "symmetric_angle"}:
        # Map to angle extent with symmetric flag
        mode = "angle"
        # Preserve any provided symmetric flag; default to True
        if "symmetric" not in value:
            value = dict(value)
            value["symmetric"] = True
    elif mode_lower in {"one_side", "one_side_angle"}:
        mode = "angle"
        if "symmetric" not in value:
            value = dict(value)
            value["symmetric"] = False

    valid_modes = {"full", "angle", "two_sides_angle", "to", "two_sides_to"}
    if mode not in valid_modes:
        raise CodeGenerationError(
            f'Invalid extent mode "{mode}". Expected one of: full, angle, two_sides_angle, to, two_sides_to.'
        )

    cleaned: Dict[str, Any] = {"mode": mode}

    def _assert_only(allowed_fields: set):
        extra = set(value.keys()) - allowed_fields
        if extra:
            raise CodeGenerationError(f'Unexpected field(s) for extent mode "{mode}": {", ".join(sorted(extra))}')

    if mode == "full":
        # Accept and ignore any stray fields (e.g., angle_degrees/symmetric sent by the LLM)
        return cleaned

    if mode == "angle":
        _assert_only({"mode", "symmetric", "angle_degrees"})
        symmetric = _validate_bool(value.get("symmetric"), "extent.symmetric")
        angle = float(_validate_float(value.get("angle_degrees"), "extent.angle_degrees"))
        if angle == 0:
            raise CodeGenerationError('"extent.angle_degrees" must be non-zero.')
        cleaned.update({"symmetric": symmetric, "angle_degrees": angle})
        return cleaned

    if mode == "two_sides_angle":
        _assert_only({"mode", "angle1_degrees", "angle2_degrees"})
        angle1 = float(_validate_float(value.get("angle1_degrees"), "extent.angle1_degrees"))
        angle2 = float(_validate_float(value.get("angle2_degrees"), "extent.angle2_degrees"))
        if angle1 == 0 or angle2 == 0:
            raise CodeGenerationError("Both angle1_degrees and angle2_degrees must be non-zero.")
        cleaned.update({"angle1_degrees": angle1, "angle2_degrees": angle2})
        return cleaned

    if mode == "to":
        _assert_only({"mode", "to_entity_token"})
        token = str(value.get("to_entity_token", "")).strip()
        if not token:
            raise CodeGenerationError('"extent.to_entity_token" must be provided for mode "to".')
        cleaned["to_entity_token"] = _escape_string_for_python(token)
        return cleaned

    if mode == "two_sides_to":
        _assert_only({"mode", "to_entity1_token", "to_entity2_token"})
        token1 = str(value.get("to_entity1_token", "")).strip()
        token2 = str(value.get("to_entity2_token", "")).strip()
        if not token1 or not token2:
            raise CodeGenerationError('"extent.to_entity1_token" and "extent.to_entity2_token" must both be provided for mode "two_sides_to".')
        cleaned["to_entity1_token"] = _escape_string_for_python(token1)
        cleaned["to_entity2_token"] = _escape_string_for_python(token2)
        return cleaned

    return cleaned


def _validate_profile_ids(value: Any, field_name: str) -> str:
    """Validate array of profile IDs for loft operations."""
    if not isinstance(value, list):
        raise CodeGenerationError(f'"{field_name}" must be an array of sketch IDs.')

    if len(value) < 2:
        raise CodeGenerationError(f'"{field_name}" must contain at least 2 sketch IDs (received {len(value)}).')

    # Validate each ID is a valid identifier
    validated_ids = []
    for i, prof_id in enumerate(value):
        if not isinstance(prof_id, str) or not prof_id.strip():
            raise CodeGenerationError(f'"{field_name}[{i}]" must be a non-empty string.')

        identifier = prof_id.strip()
        if not identifier.replace("_", "").isalnum() or identifier[0].isdigit():
            raise CodeGenerationError(
                f'"{field_name}[{i}]" must be a valid identifier (letters, numbers, underscores, '
                "cannot start with a number)."
            )
        validated_ids.append(identifier)

    # Return as Python list representation for template
    return repr(validated_ids)


def _escape_string_for_python(value: str) -> str:
    """
    Safely escape a string for embedding in Python code as a string literal.

    Escapes backslashes, single quotes, double quotes, and newlines to prevent
    code injection when the string is inserted into generated code.

    Also filters out null bytes and other control characters that would cause
    ast.parse() to fail with "source code string cannot contain null bytes".

    Returns a properly escaped string suitable for repr() style quoting.
    """
    if not value:
        return ""

    # Filter out null bytes and other problematic control characters
    # Keep only printable characters, tabs, newlines, and carriage returns
    filtered = "".join(
        char for char in value
        if char == '\n' or char == '\r' or char == '\t' or ord(char) >= 32
    )

    # Escape backslashes first (must be done before other escapes)
    escaped = filtered.replace("\\", "\\\\")
    # Escape quotes
    escaped = escaped.replace('"', '\\"')
    escaped = escaped.replace("'", "\\'")
    # Escape newlines
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")

    return escaped


def _format_numeric(value: float) -> str:
    """Format numeric literals so generated code stays concise and readable."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _build_parameter_spec() -> Dict[str, Dict[str, ParameterRule]]:
    """Create parameter validation rules for each tool."""
    return {
        "create_construction_plane": {
            "plane_id": ParameterRule(True, lambda v, _: _validate_identifier(v, "plane_id")),
            "mode": ParameterRule(True, lambda v, _: _validate_mode(v)),
            "description": ParameterRule(True, lambda v, _: _escape_string_for_python(str(v).strip()) if v else ""),
            # Mode-specific parameters (validated dynamically in _normalise_parameters)
            "datum_axis_plane": ParameterRule(False, lambda v, _: str(v).strip().upper(), default=""),
            "base_datum_plane": ParameterRule(False, lambda v, _: str(v).strip().upper(), default=""),
            "offset_cm": ParameterRule(False, lambda v, _: _validate_float(v, "offset_cm") if v is not None else None, default=None),
            "reference_face_token": ParameterRule(False, lambda v, _: _escape_string_for_python(str(v).strip()) if v else "", default=""),
            "reference_edge_token": ParameterRule(False, lambda v, _: _escape_string_for_python(str(v).strip()) if v else "", default=""),
            "angle_deg": ParameterRule(False, lambda v, _: _validate_float(v, "angle_deg") if v is not None else None, default=None),
            "face_token": ParameterRule(False, lambda v, _: _escape_string_for_python(str(v).strip()) if v else "", default=""),
            "point_x": ParameterRule(False, lambda v, _: _validate_float(v, "point_x") if v is not None else None, default=None),
            "point_y": ParameterRule(False, lambda v, _: _validate_float(v, "point_y") if v is not None else None, default=None),
            "point_z": ParameterRule(False, lambda v, _: _validate_float(v, "point_z") if v is not None else None, default=None),
        },
        "create_sketch": {
            "plane_id": ParameterRule(True, lambda v, _: _escape_string_for_python(str(v).strip()) if v else ""),
            "sketch_id": ParameterRule(True, lambda v, _: _validate_identifier(v, "sketch_id")),
            "sketch_name": ParameterRule(False, lambda v, _: _validate_feature_name(v, "sketch_name"), default=""),
            "description": ParameterRule(False, lambda v, _: str(v).strip() if v else "", default=""),
        },
        "add_circle": {
            "sketch_id": ParameterRule(True, lambda v, _: _validate_identifier(v, "sketch_id")),
            "center_u": ParameterRule(True, lambda v, _: _validate_float(v, "center_u")),
            "center_v": ParameterRule(True, lambda v, _: _validate_float(v, "center_v")),
            "radius": ParameterRule(True, lambda v, _: _validate_float(v, "radius")),
            "circle_id": ParameterRule(False, lambda v, _: _validate_identifier(v, "circle_id"), default=""),
            "description": ParameterRule(False, lambda v, _: str(v).strip() if v else "", default=""),
        },
        "add_line": {
            "sketch_id": ParameterRule(True, lambda v, _: _validate_identifier(v, "sketch_id")),
            "start_u": ParameterRule(True, lambda v, _: _validate_float(v, "start_u")),
            "start_v": ParameterRule(True, lambda v, _: _validate_float(v, "start_v")),
            "end_u": ParameterRule(True, lambda v, _: _validate_float(v, "end_u")),
            "end_v": ParameterRule(True, lambda v, _: _validate_float(v, "end_v")),
            "line_id": ParameterRule(False, lambda v, _: _validate_identifier(v, "line_id"), default=""),
            "description": ParameterRule(False, lambda v, _: str(v).strip() if v else "", default=""),
        },
        "add_arc": {
            "sketch_id": ParameterRule(True, lambda v, _: _validate_identifier(v, "sketch_id")),
            "center_u": ParameterRule(True, lambda v, _: _validate_float(v, "center_u")),
            "center_v": ParameterRule(True, lambda v, _: _validate_float(v, "center_v")),
            "start_u": ParameterRule(True, lambda v, _: _validate_float(v, "start_u")),
            "start_v": ParameterRule(True, lambda v, _: _validate_float(v, "start_v")),
            "end_u": ParameterRule(True, lambda v, _: _validate_float(v, "end_u")),
            "end_v": ParameterRule(True, lambda v, _: _validate_float(v, "end_v")),
            "arc_id": ParameterRule(False, lambda v, _: _validate_identifier(v, "arc_id"), default=""),
            "description": ParameterRule(False, lambda v, _: str(v).strip() if v else "", default=""),
        },
        "add_rectangle": {
            "sketch_id": ParameterRule(True, lambda v, _: _validate_identifier(v, "sketch_id")),
            "corner1_u": ParameterRule(True, lambda v, _: _validate_float(v, "corner1_u")),
            "corner1_v": ParameterRule(True, lambda v, _: _validate_float(v, "corner1_v")),
            "corner2_u": ParameterRule(True, lambda v, _: _validate_float(v, "corner2_u")),
            "corner2_v": ParameterRule(True, lambda v, _: _validate_float(v, "corner2_v")),
            "rectangle_id": ParameterRule(False, lambda v, _: _validate_identifier(v, "rectangle_id"), default=""),
            "description": ParameterRule(False, lambda v, _: str(v).strip() if v else "", default=""),
        },
        "list_sketch_profiles": {
            "sketch_id": ParameterRule(True, lambda v, _: _validate_identifier(v, "sketch_id")),
            "description": ParameterRule(False, lambda v, _: str(v).strip() if v else "", default=""),
        },
        "extrude_profile": {
            "sketch_id": ParameterRule(True, lambda v, _: _validate_identifier(v, "sketch_id")),
            "distance": ParameterRule(True, lambda v, _: _validate_float(v, "distance")),
            "profile_index": ParameterRule(False, lambda v, _: _validate_int(v, "profile_index"), default=0),
            # Default must be a template-friendly literal (we want `_profile_indices = None`).
            "profile_indices": ParameterRule(False, lambda v, _: _validate_int_list(v, "profile_indices"), default="None"),
            "operation": ParameterRule(False, lambda v, _: _validate_operation(v), default="NewBody"),
            "feature_name": ParameterRule(False, lambda v, _: _validate_feature_name(v, "feature_name"), default=""),
            "description": ParameterRule(False, lambda v, _: str(v).strip() if v else "", default=""),
        },
        "revolve_profile": {
            "sketch_id": ParameterRule(True, lambda v, _: _validate_identifier(v, "sketch_id")),
            "profile_index": ParameterRule(False, lambda v, _: _validate_int(v, "profile_index"), default=0),
            "axis": ParameterRule(True, lambda v, _: _validate_axis_spec(v, "axis")),
            "extent": ParameterRule(False, lambda v, _: _validate_extent_spec(v, "extent"), default={"mode": "full"}),
            "operation": ParameterRule(False, lambda v, _: _validate_revolve_operation(v), default="NewBody"),
            "is_solid": ParameterRule(False, lambda v, _: _validate_bool(v, "is_solid"), default=True),
            "creation_occurrence_token": ParameterRule(False, lambda v, _: _escape_string_for_python(str(v).strip()) if v else "", default=""),
            "feature_name": ParameterRule(False, lambda v, _: _validate_feature_name(v, "feature_name"), default=""),
            "description": ParameterRule(True, lambda v, _: _escape_string_for_python(str(v).strip()) if v else ""),
        },
        "jump_to_timeline_position": {
            "target_index": ParameterRule(True, lambda v, _: _validate_int(v, "target_index")),
            "reason": ParameterRule(False, lambda v, _: str(v).strip() if v else "", default=""),
            "description": ParameterRule(False, lambda v, _: str(v).strip() if v else "", default=""),
        },
        "delete_feature": {
            "feature_token": ParameterRule(True, lambda v, _: _validate_nonempty_string(v, "feature_token")),
            "description": ParameterRule(True, lambda v, _: _validate_nonempty_string(v, "description")),
            "expected_name": ParameterRule(False, lambda v, _: _escape_string_for_python(str(v).strip()) if v else "", default=""),
            "expected_timeline_index": ParameterRule(False, lambda v, _: _validate_int(v, "expected_timeline_index") if v is not None else "None", default="None"),
        },
        "create_loft": {
            "profile_ids": ParameterRule(True, lambda v, n: _validate_profile_ids(v, n)),
            "operation": ParameterRule(False, lambda v, _: _validate_operation(v), default="NewBody"),
            "feature_name": ParameterRule(False, lambda v, _: _validate_feature_name(v, "feature_name"), default=""),
            "description": ParameterRule(True, lambda v, _: _escape_string_for_python(str(v).strip()) if v else ""),
        },
    }


PARAMETER_SPEC = _build_parameter_spec()


def _normalise_parameters(tool_name: str, parameters: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate incoming parameters and return a cleaned dictionary for templating."""
    if tool_name not in PARAMETER_SPEC:
        raise CodeGenerationError(f'Unsupported tool "{tool_name}".')

    if not isinstance(parameters, Mapping):
        raise CodeGenerationError("Parameters must be provided as a dictionary.")

    # Apply backward-compatible parameter name normalization
    normalized_params = _normalize_param_names(tool_name, parameters)

    # Tool-specific validations that must consider raw, user-provided params (not defaults).
    if tool_name == "extrude_profile":
        # LLMs frequently include optional list fields as empty arrays instead of omitting the key.
        # Treat empty profile_indices as "not provided" (silent normalization).
        if "profile_indices" in normalized_params:
            _pi = normalized_params.get("profile_indices")
            if isinstance(_pi, (list, tuple)) and len(_pi) == 0:
                del normalized_params["profile_indices"]
            elif isinstance(_pi, str):
                # Be tolerant of whitespace: " [ ] " should be treated as empty.
                _compact = "".join(_pi.split())
                if _compact == "[]":
                    del normalized_params["profile_indices"]

        # Resolve mutual exclusivity silently and deterministically:
        # - If profile_indices is provided (and non-empty after normalization), it wins.
        # - Otherwise, profile_index is used (default applies later).
        if "profile_indices" in normalized_params and normalized_params.get("profile_indices") is not None:
            normalized_params.pop("profile_index", None)
    
    rules = PARAMETER_SPEC[tool_name]
    cleaned: Dict[str, Any] = {}

    for key, rule in rules.items():
        if key not in normalized_params or normalized_params[key] is None:
            if rule.required:
                raise CodeGenerationError(f'Missing required parameter "{key}".')
            if rule.default is not None:
                cleaned[key] = rule.default
            continue

        raw_value = normalized_params[key]
        try:
            param_count = len(inspect.signature(rule.validator).parameters)
            if param_count == 1:
                cleaned[key] = rule.validator(raw_value)
            else:
                cleaned[key] = rule.validator(raw_value, key)
        except CodeGenerationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise CodeGenerationError(f'Invalid value for "{key}": {raw_value!r}') from exc

    # Enforce no unexpected parameters (helps catch tool misuse).
    # Use normalized_params to avoid flagging legacy names as unexpected.
    extra_keys = set(normalized_params.keys()) - set(rules.keys())
    if extra_keys:
        extras = ", ".join(sorted(extra_keys))
        raise CodeGenerationError(
            f"Unexpected parameter(s) for {tool_name}: {extras}. Refer to the tool schema."
        )

    if tool_name == "create_construction_plane":
        # Build plane_params based on mode
        mode = cleaned["mode"]
        params_parts = []

        if mode == "datum":
            if cleaned.get("datum_axis_plane"):
                params_parts.append(f'datum_axis_plane="{cleaned["datum_axis_plane"]}"')
        elif mode == "offset_from_datum":
            if cleaned.get("base_datum_plane"):
                params_parts.append(f'base_datum_plane="{cleaned["base_datum_plane"]}"')
            if cleaned.get("offset_cm") is not None:
                params_parts.append(f'offset_cm={cleaned["offset_cm"]}')
        elif mode == "angle_to_edge":
            if cleaned.get("reference_face_token"):
                params_parts.append(f'reference_face_token="{cleaned["reference_face_token"]}"')
            if cleaned.get("reference_edge_token"):
                params_parts.append(f'reference_edge_token="{cleaned["reference_edge_token"]}"')
            if cleaned.get("angle_deg") is not None:
                params_parts.append(f'angle_deg={cleaned["angle_deg"]}')
        elif mode == "face_normal":
            if cleaned.get("face_token"):
                params_parts.append(f'face_token="{cleaned["face_token"]}"')
            if cleaned.get("point_x") is not None:
                params_parts.append(f'point_x={cleaned["point_x"]}')
            if cleaned.get("point_y") is not None:
                params_parts.append(f'point_y={cleaned["point_y"]}')
            if cleaned.get("point_z") is not None:
                params_parts.append(f'point_z={cleaned["point_z"]}')

        cleaned["plane_params"] = ",\n                ".join(params_parts)

    if tool_name == "revolve_profile":
        if cleaned["profile_index"] < 0:
            raise CodeGenerationError('"profile_index" must be zero or greater.')
        cleaned["axis_spec"] = cleaned.pop("axis")
        cleaned["extent_spec"] = cleaned.pop("extent")

    if tool_name == "extrude_profile":
        if cleaned["profile_index"] < 0:
            raise CodeGenerationError('"profile_index" must be zero or greater.')

    if tool_name == "jump_to_timeline_position":
        if cleaned["target_index"] < 0:
            raise CodeGenerationError('"target_index" must be zero or greater.')

    if tool_name == "add_arc":
        arc_error = _validate_arc_geometry(
            float(cleaned["center_u"]),
            float(cleaned["center_v"]),
            float(cleaned["start_u"]),
            float(cleaned["start_v"]),
            float(cleaned["end_u"]),
            float(cleaned["end_v"]),
        )
        if arc_error:
            raise CodeGenerationError(arc_error)

    return cleaned


def translate_tool_call(tool_name: str, parameters: Mapping[str, Any]) -> str:
    """
    Translate a validated tool call into executable Python code for Fusion 360.

    Raises CodeGenerationError when the tool is unsupported, parameters are invalid,
    or the rendered template fails validation.
    """
    if tool_name not in OPERATION_TEMPLATES:
        raise CodeGenerationError(f'Unsupported tool "{tool_name}".')

    cleaned_params = _normalise_parameters(tool_name, parameters)
    template = OPERATION_TEMPLATES[tool_name]

    try:
        code = template.format(**cleaned_params).strip() + "\n"
    except KeyError as exc:
        missing = exc.args[0]
        raise CodeGenerationError(
            f'Failed to render template for "{tool_name}". Missing key: {missing}.'
        ) from exc

    ensure_valid_python(code)
    return code


def translate_tool_call_from_dict(tool_call: Mapping[str, Any]) -> str:
    """
    Convenience wrapper that accepts an Anthropic-style tool call payload:

    {"name": "create_sketch", "arguments": {...}}
    """
    if "name" not in tool_call:
        raise CodeGenerationError("Tool call payload must include a 'name'.")

    params: Optional[Mapping[str, Any]] = None
    for key in ("arguments", "input", "parameters"):
        if key in tool_call and isinstance(tool_call[key], Mapping):
            params = tool_call[key]  # type: ignore[assignment]
            break

    if params is None:
        raise CodeGenerationError("Tool call payload must include an arguments dictionary.")

    return translate_tool_call(tool_call["name"], params)


def ensure_valid_python(code: str) -> None:
    """
    Perform a syntax check to ensure generated code is executable by Fusion 360.

    Raises:
        CodeGenerationError: if AST parsing fails.
    """
    try:
        ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise CodeGenerationError(format_error_for_llm(exc)) from exc


def format_error_for_llm(error: Exception) -> str:
    """
    Convert internal exceptions into concise messages suitable for the LLM.
    """
    if isinstance(error, SyntaxError):
        location = f"line {error.lineno}, column {error.offset}" if error.lineno else "unknown location"
        return f"Syntax error at {location}: {error.msg}"

    return str(error)


__all__ = [
    "CodeGenerationError",
    "OPERATION_TEMPLATES",
    "translate_tool_call",
    "translate_tool_call_from_dict",
    "ensure_valid_python",
    "format_error_for_llm",
]
