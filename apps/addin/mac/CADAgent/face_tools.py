"""
Face metadata utilities for CADAgent.

Provides helpers to extract, select, and clear faces in the active Fusion 360
design. Returns face metadata enriched with boundary edge information so the
LLM can reason about selections without embedding raw Fusion API usage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import adsk.core
import adsk.fusion

logger = logging.getLogger(__name__)

from .entity_iteration import iter_brep_bodies


class FaceOperationError(RuntimeError):
    """Raised when a face operation cannot be fulfilled."""


# Surface type normalization mapping (Fusion -> LLM-friendly)
_SURFACE_TYPE_MAP = {
    adsk.core.SurfaceTypes.PlaneSurfaceType: "planar",
    adsk.core.SurfaceTypes.CylinderSurfaceType: "cylindrical",
    adsk.core.SurfaceTypes.ConeSurfaceType: "conical",
    adsk.core.SurfaceTypes.SphereSurfaceType: "spherical",
    adsk.core.SurfaceTypes.TorusSurfaceType: "toroidal",
    adsk.core.SurfaceTypes.NurbsSurfaceType: "nurbs",
}


def _normalize_surface_type(face: adsk.fusion.BRepFace) -> str:
    """Normalize Fusion surface type to LLM-friendly vocabulary."""
    try:
        geometry = face.geometry
        if geometry:
            surface_type_enum = geometry.surfaceType
            return _SURFACE_TYPE_MAP.get(surface_type_enum, "unknown")
    except Exception:
        pass
    return "unknown"


def _extract_face_frame(face: adsk.fusion.BRepFace) -> Optional[Dict[str, List[float]]]:
    """
    Extract evaluator-based frame at face.pointOnFace.
    
    Returns:
        Dictionary with 'u', 'v', 'n' as normalized unit vectors, or None on failure.
        Note: u and v are tangent directions (not guaranteed orthonormal on curved surfaces).
        n is the surface normal, negated if face.isParamReversed is True.
    """
    try:
        import math
        p = face.pointOnFace
        se = face.evaluator
        
        ok, uv = se.getParameterAtPoint(p)
        if not ok:
            return None
        
        ok, du, dv = se.getFirstDerivative(uv)
        if not ok:
            return None
        
        ok, n = se.getNormalAtPoint(p)
        if not ok:
            return None
        
        # Normalize du to unit vector
        du_len = math.sqrt(du.x**2 + du.y**2 + du.z**2)
        if du_len > 1e-10:
            u_vec = [du.x / du_len, du.y / du_len, du.z / du_len]
        else:
            u_vec = [1.0, 0.0, 0.0]
        
        # Normalize dv to unit vector
        dv_len = math.sqrt(dv.x**2 + dv.y**2 + dv.z**2)
        if dv_len > 1e-10:
            v_vec = [dv.x / dv_len, dv.y / dv_len, dv.z / dv_len]
        else:
            v_vec = [0.0, 1.0, 0.0]
        
        # Handle isParamReversed - negate normal for consistent orientation
        n_sign = -1.0 if face.isParamReversed else 1.0
        n_vec = [n.x * n_sign, n.y * n_sign, n.z * n_sign]
        
        return {
            "u": [_rounded(u_vec[0], 4), _rounded(u_vec[1], 4), _rounded(u_vec[2], 4)],
            "v": [_rounded(v_vec[0], 4), _rounded(v_vec[1], 4), _rounded(v_vec[2], 4)],
            "n": [_rounded(n_vec[0], 4), _rounded(n_vec[1], 4), _rounded(n_vec[2], 4)],
        }
    except Exception as exc:
        logger.debug("Failed to extract face frame: %s", exc)
        return None


def list_faces(
    app: adsk.core.Application,
    design: Optional[adsk.fusion.Design] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Collect metadata for every face in the active design, including boundary edges.

    Returns:
        Tuple where the first value is a list of face dictionaries and the second
        value is a mapping of unit identifiers (`length`, `area`).
    """
    if design is None or not getattr(design, "isValid", True):
        design = _require_design(app)
    root_component = design.rootComponent
    units_manager = design.unitsManager

    # Units in mm (convert from Fusion internal cm)
    length_units = "mm"
    area_units = "mm^2"
    internal_length_units = units_manager.internalUnits  # "cm"
    internal_area_units = _internal_area_units(internal_length_units)

    faces: List[Dict[str, Any]] = []

    try:
        for body_index, (ctx, body) in enumerate(iter_brep_bodies(design)):
            body_name = body.name or f"Body {body_index}"

            for face_index, face in enumerate(body.faces):
                # Normalized surface type (planar, cylindrical, etc.)
                surface_type = _normalize_surface_type(face)
                
                # Area converted to mm^2 (Fusion internal is cm^2, multiply by 100)
                area_val = _rounded(
                    getattr(face, "area", 0.0) * 100.0,  # cm^2 -> mm^2
                    precision=6,
                )

                # Centroid in mm
                centroid_coords = _point_coords_mm(
                    getattr(face, "centroid", None),
                )

                bounding_box = _bounding_box_coords_mm(
                    getattr(face, "boundingBox", None),
                )

                # Collect loops with coEdges structure
                loops_info = _collect_face_loops(
                    face,
                )

                # Extract evaluator-based frame
                frame = _extract_face_frame(face)
                
                # Extract normal from frame or fallback
                normal = None
                if frame:
                    normal = frame["n"]

                face_info: Dict[str, Any] = {
                    "id": f"body{body_index}_face{face_index}",
                    "temp_id": _safe_temp_id(face),
                    "body_name": body_name,
                    "body_index": body_index,
                    "component": ctx.get("component"),
                    "occurrence_path": ctx.get("occurrence_path"),
                    "face_index": face_index,
                    "surface_type": surface_type,  # Renamed from geometry_type
                    "area": area_val,
                    "area_units": area_units,
                    "centroid": centroid_coords,
                    "normal": normal,
                    "frame": frame,
                    "bounding_box": bounding_box,
                    "entity_token": face.entityToken,
                    "loops": loops_info,  # New structured loops with outer/inner
                }

                # Extract geometry-specific spatial information for LLM spatial awareness
                try:
                    geometry = face.geometry
                    if geometry:
                        surface_type_enum = geometry.surfaceType

                        # PLANAR FACES: Add orientation classification
                        if surface_type_enum == adsk.core.SurfaceTypes.PlaneSurfaceType:
                            if normal:
                                # Create a mock vector for classification
                                class MockVec:
                                    def __init__(self, n):
                                        self.x, self.y, self.z = n
                                face_info["orientation"] = _classify_plane_orientation(MockVec(normal))

                        # CYLINDRICAL FACES: Add axis and radius (in mm)
                        elif surface_type_enum == adsk.core.SurfaceTypes.CylinderSurfaceType:
                            cylinder = geometry
                            axis = cylinder.axis
                            face_info["axis"] = [
                                _rounded(axis.x, 3),
                                _rounded(axis.y, 3),
                                _rounded(axis.z, 3)
                            ]
                            # Radius in mm (Fusion internal is cm)
                            face_info["radius"] = _rounded(cylinder.radius * 10.0, 6)
                            face_info["axis_orientation"] = _classify_axis_orientation(axis)
                            
                        # CONICAL FACES: Add axis and half-angle
                        elif surface_type_enum == adsk.core.SurfaceTypes.ConeSurfaceType:
                            cone = geometry
                            axis = cone.axis
                            face_info["axis"] = [
                                _rounded(axis.x, 3),
                                _rounded(axis.y, 3),
                                _rounded(axis.z, 3)
                            ]
                            face_info["half_angle"] = _rounded(cone.halfAngle, 6)
                            
                        # SPHERICAL FACES: Add radius (in mm)
                        elif surface_type_enum == adsk.core.SurfaceTypes.SphereSurfaceType:
                            sphere = geometry
                            face_info["radius"] = _rounded(sphere.radius * 10.0, 6)
                            
                        # TOROIDAL FACES: Add major/minor radii (in mm)
                        elif surface_type_enum == adsk.core.SurfaceTypes.TorusSurfaceType:
                            torus = geometry
                            face_info["major_radius"] = _rounded(torus.majorRadius * 10.0, 6)
                            face_info["minor_radius"] = _rounded(torus.minorRadius * 10.0, 6)
                            
                except Exception as exc:  # pragma: no cover - defensive for geometry extraction
                    logger.warning("Failed to extract geometry-specific data for face %s: %s", face_index, exc)
                    import traceback
                    logger.warning("Traceback: %s", traceback.format_exc())

                faces.append(face_info)

    except Exception as exc:  # pragma: no cover - defensive for Fusion runtime issues
        logger.exception("Failed while extracting faces")
        raise FaceOperationError(f"Failed to extract faces: {exc}") from exc

    logger.info("Extracted %d faces (length units: %s, area units: %s)", len(faces), length_units, area_units)
    return faces, {"length": length_units, "area": area_units, "_code_version": "spatial_aware_v1"}


def select_faces(
    app: adsk.core.Application,
    entity_tokens: Sequence[str],
    *,
    clear_existing: bool = True,
) -> Dict[str, Any]:
    """
    Select faces identified by entity tokens.

    Args:
        app: Fusion application instance.
        entity_tokens: Iterable of face entity tokens.
        clear_existing: When True (default) clears the current selection set first.

    Returns:
        Dictionary with counts and any missing tokens.
    """
    tokens = [token for token in entity_tokens if isinstance(token, str) and token.strip()]
    if not tokens:
        raise FaceOperationError("No entity tokens provided for selection.")

    design = _require_design(app)
    ui = _require_ui(app)

    if clear_existing:
        try:
            ui.activeSelections.clear()
        except Exception as exc:  # pragma: no cover - Fusion runtime nuance
            logger.warning("Failed to clear existing selections: %s", exc)

    selected_count = 0
    missing_tokens: List[str] = []

    for token in tokens:
        try:
            resolved = design.findEntityByToken(token.strip())
            if not resolved:
                missing_tokens.append(token)
                continue
            face = adsk.fusion.BRepFace.cast(resolved[0])
            if not face:
                missing_tokens.append(token)
                continue
            ui.activeSelections.add(face)
            selected_count += 1
        except Exception as exc:  # pragma: no cover - should be rare
            logger.warning("Failed to add face with token %s to selection: %s", token, exc)
            missing_tokens.append(token)

    success = selected_count > 0
    if not success:
        raise FaceOperationError("None of the provided entity tokens matched faces in the design.")

    logger.info(
        "Selected %d face(s); %d token(s) missing",
        selected_count,
        len(missing_tokens),
    )

    return {
        "selected_count": selected_count,
        "missing_tokens": missing_tokens,
        "cleared_existing": clear_existing,
    }


def clear_face_selection(app: adsk.core.Application) -> Dict[str, Any]:
    """
    Clear the current Fusion face selection set.

    Returns:
        Dictionary containing the number of entities removed from the selection.
    """
    ui = _require_ui(app)

    active_selections = ui.activeSelections
    cleared_count = active_selections.count if active_selections else 0

    try:
        active_selections.clear()
    except Exception as exc:  # pragma: no cover - Fusion runtime nuance
        logger.warning("Failed to clear selections: %s", exc)
        raise FaceOperationError(f"Failed to clear selections: {exc}") from exc

    logger.info("Cleared %d selection(s)", cleared_count)
    return {"cleared_count": cleared_count}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_design(app: Optional[adsk.core.Application]) -> adsk.fusion.Design:
    if not app:
        raise FaceOperationError("Fusion application is not available.")

    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise FaceOperationError("No active design is available. Open or create a design and try again.")
    return design


def _require_ui(app: Optional[adsk.core.Application]) -> adsk.core.UserInterface:
    if not app:
        raise FaceOperationError("Fusion application is not available.")

    ui = app.userInterface
    if not ui:
        raise FaceOperationError("Fusion user interface is not available.")
    return ui


def _safe_surface_type(face: adsk.fusion.BRepFace) -> str:
    """Legacy function - use _normalize_surface_type for new code."""
    try:
        geometry = face.geometry
        if geometry and geometry.objectType:
            return geometry.objectType.split("::")[-1]
    except Exception:
        pass
    return "Unknown"


def _point_coords_mm(point: Optional[adsk.core.Point3D]) -> Optional[List[float]]:
    """Convert a Point3D to mm coordinates (Fusion internal is cm)."""
    if not point:
        return None
    return [
        _rounded(point.x * 10.0),  # cm -> mm
        _rounded(point.y * 10.0),
        _rounded(point.z * 10.0),
    ]


def _bounding_box_coords_mm(bbox: Optional[adsk.core.BoundingBox3D]) -> Optional[Dict[str, List[float]]]:
    """Convert a BoundingBox3D to mm coordinates."""
    if not bbox:
        return None
    return {
        "min": _point_coords_mm(bbox.minPoint),
        "max": _point_coords_mm(bbox.maxPoint),
    }


def _safe_edge_type(edge: adsk.fusion.BRepEdge) -> str:
    try:
        geometry = edge.geometry
        if geometry and geometry.objectType:
            return geometry.objectType.split("::")[-1]
    except Exception:
        pass
    return "Unknown"


def _internal_area_units(internal_length_units: str) -> str:
    if internal_length_units.endswith("^2"):
        return internal_length_units
    return f"{internal_length_units}^2"


def _convert_units(
    units_manager: adsk.core.UnitsManager,
    value: Optional[float],
    from_units: str,
    to_units: str,
) -> float:
    if value is None:
        return 0.0
    try:
        return float(units_manager.convert(value, from_units, to_units))
    except Exception:
        logger.debug(
            "Failed to convert value %s from %s to %s; returning raw value",
            value,
            from_units,
            to_units,
        )
        return float(value)


def _point_coords(
    point: Optional[adsk.core.Point3D],
    units_manager: adsk.core.UnitsManager,
    from_units: str,
    to_units: str,
) -> Optional[List[float]]:
    if not point:
        return None

    return [
        _rounded(_convert_units(units_manager, point.x, from_units, to_units)),
        _rounded(_convert_units(units_manager, point.y, from_units, to_units)),
        _rounded(_convert_units(units_manager, point.z, from_units, to_units)),
    ]


def _bounding_box_coords(
    bbox: Optional[adsk.core.BoundingBox3D],
    units_manager: adsk.core.UnitsManager,
    from_units: str,
    to_units: str,
) -> Optional[Dict[str, List[float]]]:
    if not bbox:
        return None

    return {
        "min": _point_coords(bbox.minPoint, units_manager, from_units, to_units),
        "max": _point_coords(bbox.maxPoint, units_manager, from_units, to_units),
    }


def _collect_face_loops(face: adsk.fusion.BRepFace) -> Dict[str, Any]:
    """
    Collect loop information using coEdges for proper orientation.
    
    Returns:
        Dictionary with 'outer' (single loop) and 'inner' (array of inner loops).
        Each loop is an array of coEdge references with edge token and orientation.
    """
    result: Dict[str, Any] = {
        "outer": [],
        "inner": [],
    }
    
    try:
        for loop in face.loops:
            coedge_list: List[Dict[str, Any]] = []
            
            for coedge in loop.coEdges:
                edge = coedge.edge
                coedge_info = {
                    "edge_token": edge.entityToken,
                    "edge": edge.entityToken,  # Alias for backend mapping to "e0" format
                    "isOpposedToEdge": coedge.isOpposedToEdge,
                }
                coedge_list.append(coedge_info)
            
            if loop.isOuter:
                result["outer"] = coedge_list
            else:
                result["inner"].append(coedge_list)
                
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to collect loops for face %s: %s", face, exc)

    return result


def _collect_face_edges(
    face: adsk.fusion.BRepFace,
    units_manager: adsk.core.UnitsManager,
    from_units: str,
    to_units: str,
) -> List[Dict[str, Any]]:
    """Legacy function for backward compatibility - collects flat edge list."""
    edges_info: List[Dict[str, Any]] = []
    try:
        for loop_index, loop in enumerate(face.loops):
            for coedge in loop.coEdges:
                edge = coedge.edge
                length_val = _edge_length(edge)
                # Convert to mm (from cm)
                converted_length = length_val * 10.0

                edges_info.append(
                    {
                        "loop_index": loop_index,
                        "is_outer_loop": loop.isOuter,
                        "edge_token": edge.entityToken,
                        "isOpposedToEdge": coedge.isOpposedToEdge,
                        "geometry_type": _safe_edge_type(edge),
                        "length": _rounded(converted_length),
                        "length_units": "mm",
                        "start_coords": _vertex_coords_mm(edge.startVertex),
                        "end_coords": _vertex_coords_mm(edge.endVertex),
                        "entity_token": edge.entityToken,
                        "temp_id": _safe_temp_id(edge),
                    }
                )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to collect edges for face %s: %s", face, exc)

    return edges_info


def _vertex_coords_mm(vertex: Optional[adsk.fusion.BRepVertex]) -> Optional[List[float]]:
    """Get vertex coordinates in mm."""
    if not vertex or not vertex.geometry:
        return None
    return _point_coords_mm(vertex.geometry)


def _edge_identifier(
    face: adsk.fusion.BRepFace,
    loop_index: int,
    edge_index: int,
    edge: adsk.fusion.BRepEdge,
) -> str:
    body = getattr(face, "body", None)
    try:
        body_name = getattr(body, "name", "") or "Body"
    except Exception:
        body_name = "Body"

    base = f"{body_name}_loop{loop_index}_edge{edge_index}"
    temp_id = _safe_temp_id(edge)
    if temp_id is not None:
        return f"{base}_temp{temp_id}"
    return base


def _edge_length(edge: adsk.fusion.BRepEdge) -> float:
    try:
        length_eval = edge.evaluator.getLength()
        if isinstance(length_eval, Iterable) and length_eval:
            length_tuple = list(length_eval)
            if len(length_tuple) >= 2 and length_tuple[0]:
                return float(length_tuple[1])
    except Exception:
        logger.debug("Falling back to edge.length for %s", edge)
    return float(getattr(edge, "length", 0.0))


def _vertex_coords(
    vertex: Optional[adsk.fusion.BRepVertex],
    units_manager: adsk.core.UnitsManager,
    from_units: str,
    to_units: str,
) -> Optional[List[float]]:
    if not vertex or not vertex.geometry:
        return None
    return _point_coords(vertex.geometry, units_manager, from_units, to_units)


def _rounded(value: float, precision: int = 6) -> float:
    return round(value, precision)


def _build_face_lookup(root_component: adsk.fusion.Component) -> Dict[str, adsk.fusion.BRepFace]:
    lookup: Dict[str, adsk.fusion.BRepFace] = {}
    try:
        for body in root_component.bRepBodies:
            for face in body.faces:
                lookup[face.entityToken] = face
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed while building face lookup: %s", exc)
    return lookup


def _safe_temp_id(entity: Any) -> Optional[int]:
    try:
        temp_id = getattr(entity, "tempId", None)
        if temp_id is not None:
            return int(temp_id)
    except Exception:
        pass
    return None


def _classify_plane_orientation(normal: adsk.core.Vector3D) -> str:
    """
    Classify plane orientation based on normal vector.

    Args:
        normal: Normal vector of the plane

    Returns:
        "horizontal", "vertical", or "angled"
    """
    # Threshold for considering alignment (within ~6 degrees)
    threshold = 0.99

    abs_x = abs(normal.x)
    abs_y = abs(normal.y)
    abs_z = abs(normal.z)

    # Horizontal planes (normal mostly in Z)
    if abs_z > threshold:
        return "horizontal"

    # Vertical planes (normal mostly in X or Y)
    if abs_x > threshold or abs_y > threshold:
        return "vertical"

    return "angled"


def _classify_axis_orientation(axis: adsk.core.Vector3D) -> str:
    """
    Classify cylinder/axis orientation.

    Args:
        axis: Axis direction vector of the cylinder

    Returns:
        "x_aligned", "y_aligned", "z_aligned", or "angled"
    """
    threshold = 0.99

    abs_x = abs(axis.x)
    abs_y = abs(axis.y)
    abs_z = abs(axis.z)

    if abs_x > threshold:
        return "x_aligned"
    if abs_y > threshold:
        return "y_aligned"
    if abs_z > threshold:
        return "z_aligned"

    return "angled"


__all__ = [
    "FaceOperationError",
    "list_faces",
    "select_faces",
    "clear_face_selection",
]
