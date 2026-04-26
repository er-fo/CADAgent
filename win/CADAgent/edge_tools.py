"""
Edge metadata utilities for CADAgent.

Provides high-level helpers to extract, select, and clear edges in the active
Fusion 360 design so backend tools can operate without embedding raw Fusion API
logic. All measurements are returned in millimeters (mm), converted from Fusion's
internal centimeter units.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import adsk.core
import adsk.fusion

logger = logging.getLogger(__name__)

from .entity_iteration import iter_brep_bodies


class EdgeOperationError(RuntimeError):
    """Raised when an edge operation cannot be fulfilled."""


# Edge type normalization mapping (Fusion -> LLM-friendly)
_EDGE_TYPE_MAP = {
    "Line3D": "line",
    "InfiniteLine3D": "line",
    "Arc3D": "arc",
    "Circle3D": "arc",
    "EllipseArc3D": "ellipse",
    "Ellipse3D": "ellipse",
    "NurbsCurve3D": "spline",
}


def _normalize_edge_type(edge: adsk.fusion.BRepEdge) -> str:
    """Normalize Fusion edge type to LLM-friendly vocabulary."""
    try:
        geometry = edge.geometry
        if geometry and geometry.objectType:
            type_name = geometry.objectType.split("::")[-1]
            return _EDGE_TYPE_MAP.get(type_name, "unknown")
    except Exception:
        pass
    return "unknown"


def _get_edge_tangent_at_midpoint(edge: adsk.fusion.BRepEdge) -> Optional[List[float]]:
    """
    Compute the tangent vector at the edge midpoint using the curve evaluator.
    
    Returns:
        Normalized unit tangent vector [x, y, z], or None on failure.
    """
    try:
        import math
        evaluator = edge.evaluator
        
        # Get parameter extents
        ok, t0, t1 = evaluator.getParameterExtents()
        if not ok:
            return None
        
        # Compute midpoint parameter
        t_mid = (t0 + t1) / 2.0
        
        # Get first derivative (tangent) at midpoint
        ok, tangent_vec = evaluator.getFirstDerivative(t_mid)
        if not ok:
            return None
        
        # Normalize to unit vector
        length = math.sqrt(tangent_vec.x**2 + tangent_vec.y**2 + tangent_vec.z**2)
        if length > 1e-10:
            return [
                round(tangent_vec.x / length, 6),
                round(tangent_vec.y / length, 6),
                round(tangent_vec.z / length, 6),
            ]
    except Exception as exc:
        logger.debug("Failed to get edge tangent: %s", exc)
    return None


def _get_edge_midpoint_via_evaluator(edge: adsk.fusion.BRepEdge) -> Optional[List[float]]:
    """
    Compute the edge midpoint using the curve evaluator.
    
    Returns:
        Midpoint coordinates in mm [x, y, z], or None on failure.
    """
    try:
        evaluator = edge.evaluator
        
        # Get parameter extents
        ok, t0, t1 = evaluator.getParameterExtents()
        if not ok:
            return None
        
        # Compute midpoint parameter
        t_mid = (t0 + t1) / 2.0
        
        # Get point at midpoint parameter
        ok, midpoint = evaluator.getPointAtParameter(t_mid)
        if not ok:
            return None
        
        # Convert cm to mm
        return [
            round(midpoint.x * 10.0, 6),
            round(midpoint.y * 10.0, 6),
            round(midpoint.z * 10.0, 6),
        ]
    except Exception as exc:
        logger.debug("Failed to get edge midpoint via evaluator: %s", exc)
    return None


def _get_adjacent_face_tokens(edge: adsk.fusion.BRepEdge) -> List[str]:
    """
    Get the entity tokens of faces adjacent to this edge.
    
    Returns:
        List of face entity tokens (typically 1 or 2 faces).
    """
    try:
        return [face.entityToken for face in edge.faces]
    except Exception as exc:
        logger.debug("Failed to get adjacent faces: %s", exc)
        return []


def _point_coords_mm(point: Optional[adsk.core.Point3D]) -> Optional[List[float]]:
    """Convert a Point3D to mm coordinates (Fusion internal is cm)."""
    if not point:
        return None
    return [
        round(point.x * 10.0, 6),  # cm -> mm
        round(point.y * 10.0, 6),
        round(point.z * 10.0, 6),
    ]


def _vertex_info_mm(vertex: Optional[adsk.fusion.BRepVertex]) -> Optional[Dict[str, Any]]:
    """
    Extract vertex info including token and position in mm.
    
    Returns:
        Dictionary with 'token' and 'position', or None if vertex is None.
    """
    if not vertex:
        return None
    try:
        return {
            "token": vertex.entityToken,
            "position": _point_coords_mm(vertex.geometry),
        }
    except Exception:
        return None


def list_edges(
    app: adsk.core.Application,
    design: Optional[adsk.fusion.Design] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Collect metadata for every edge in the active design.

    Returns:
        Tuple where the first value is a list of edge dictionaries and the second
        value is the length unit identifier (always "mm").
    """
    if design is None or not getattr(design, "isValid", True):
        design = _require_design(app)
    root_component = design.rootComponent
    units_manager = design.unitsManager
    # Units in mm (convert from Fusion internal cm)
    default_units = "mm"
    internal_units = units_manager.internalUnits  # "cm"

    edges: List[Dict[str, Any]] = []

    try:
        for body_index, (ctx, body) in enumerate(iter_brep_bodies(design)):
            body_name = body.name or f"Body {body_index}"

            for edge_index, edge in enumerate(body.edges):
                # Normalized edge type (line, arc, spline, etc.)
                edge_type = _normalize_edge_type(edge)
                legacy_geometry_type = _safe_geometry_type(edge)
                
                # Length in mm (Fusion internal is cm, multiply by 10)
                length_val = _edge_length(edge)
                length_mm = length_val * 10.0

                # Vertex coordinates in mm
                start_vertex = edge.startVertex
                end_vertex = edge.endVertex
                start_coords = _point_coords_mm(start_vertex.geometry) if start_vertex and start_vertex.geometry else None
                end_coords = _point_coords_mm(end_vertex.geometry) if end_vertex and end_vertex.geometry else None

                # Midpoint via evaluator (more accurate for curves)
                midpoint = _get_edge_midpoint_via_evaluator(edge)
                # Fallback to linear interpolation if evaluator fails
                if midpoint is None and start_coords and end_coords:
                    midpoint = [
                        _rounded((start_coords[0] + end_coords[0]) / 2),
                        _rounded((start_coords[1] + end_coords[1]) / 2),
                        _rounded((start_coords[2] + end_coords[2]) / 2),
                    ]

                # Tangent at midpoint via curve evaluator (works for all edge types)
                tangent = _get_edge_tangent_at_midpoint(edge)
                
                # Legacy direction (chord direction) as fallback
                direction = None
                if start_coords and end_coords:
                    import math
                    dx = end_coords[0] - start_coords[0]
                    dy = end_coords[1] - start_coords[1]
                    dz = end_coords[2] - start_coords[2]
                    length_vec = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if length_vec > 1e-10:
                        direction = [
                            _rounded(dx / length_vec),
                            _rounded(dy / length_vec),
                            _rounded(dz / length_vec),
                        ]

                # Determine if edge is primarily linear (vs arc/curve)
                is_linear = edge_type == "line"
                
                # Adjacent faces (typically 1 or 2)
                adjacent_face_tokens = _get_adjacent_face_tokens(edge)
                
                # Vertex info with tokens and positions
                v0_info = _vertex_info_mm(start_vertex)
                v1_info = _vertex_info_mm(end_vertex)

                edge_info: Dict[str, Any] = {
                    "id": f"body{body_index}_edge{edge_index}",
                    "body_name": body_name,
                    "body_index": body_index,
                    "component": ctx.get("component"),
                    "occurrence_path": ctx.get("occurrence_path"),
                    "edge_index": edge_index,
                    "edge_type": edge_type,  # Normalized type (line, arc, spline, etc.)
                    "geometry_type": legacy_geometry_type,  # Legacy for compatibility
                    "length": _rounded(length_mm),
                    "length_units": default_units,
                    "p0": start_coords,  # New name per spec
                    "p1": end_coords,  # New name per spec
                    "start_coords": start_coords,  # Legacy for compatibility
                    "end_coords": end_coords,  # Legacy for compatibility
                    "midpoint": midpoint,
                    "tangent": tangent,  # Tangent at midpoint from evaluator
                    "direction": direction,  # Legacy chord direction
                    "is_linear": is_linear,
                    "entity_token": edge.entityToken,
                    "adjacent_face_tokens": adjacent_face_tokens,  # Face tokens this edge borders
                    "adjacent_faces": adjacent_face_tokens,  # Alias for backend compatibility (will map to IDs)
                    "v0": v0_info,  # Start vertex with token and position
                    "v1": v1_info,  # End vertex with token and position
                    "v0_token": v0_info.get("token") if v0_info else None,  # Raw vertex token for ID mapping
                    "v1_token": v1_info.get("token") if v1_info else None,  # Raw vertex token for ID mapping
                }

                edges.append(edge_info)

    except Exception as exc:  # pragma: no cover - defensive for Fusion runtime issues
        logger.exception("Failed while extracting edges")
        raise EdgeOperationError(f"Failed to extract edges: {exc}") from exc

    logger.info("Extracted %d edges (units: %s)", len(edges), default_units)
    return edges, default_units


def select_edges(
    app: adsk.core.Application,
    entity_tokens: Sequence[str],
    *,
    clear_existing: bool = True,
) -> Dict[str, Any]:
    """
    Select edges identified by entity tokens.

    Args:
        app: Fusion application instance.
        entity_tokens: Iterable of edge entity tokens.
        clear_existing: When True (default) clears the current selection set first.

    Returns:
        Dictionary with counts and any missing tokens.
    """
    tokens = [token for token in entity_tokens if isinstance(token, str) and token.strip()]
    if not tokens:
        raise EdgeOperationError("No entity tokens provided for selection.")

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
            edge = adsk.fusion.BRepEdge.cast(resolved[0])
            if not edge:
                missing_tokens.append(token)
                continue
            ui.activeSelections.add(edge)
            selected_count += 1
        except Exception as exc:  # pragma: no cover - should be rare
            logger.warning("Failed to add edge with token %s to selection: %s", token, exc)
            missing_tokens.append(token)

    success = selected_count > 0
    if not success:
        raise EdgeOperationError("None of the provided entity tokens matched edges in the design.")

    logger.info(
        "Selected %d edge(s); %d token(s) missing",
        selected_count,
        len(missing_tokens),
    )

    return {
        "selected_count": selected_count,
        "missing_tokens": missing_tokens,
        "cleared_existing": clear_existing,
    }


def clear_edge_selection(app: adsk.core.Application) -> Dict[str, Any]:
    """
    Clear the current Fusion edge selection set.

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
        raise EdgeOperationError(f"Failed to clear selections: {exc}") from exc

    logger.info("Cleared %d selection(s)", cleared_count)
    return {"cleared_count": cleared_count}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_design(app: Optional[adsk.core.Application]) -> adsk.fusion.Design:
    if not app:
        raise EdgeOperationError("Fusion application is not available.")

    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise EdgeOperationError("No active design is available. Open or create a design and try again.")
    return design


def _require_ui(app: Optional[adsk.core.Application]) -> adsk.core.UserInterface:
    if not app:
        raise EdgeOperationError("Fusion application is not available.")

    ui = app.userInterface
    if not ui:
        raise EdgeOperationError("Fusion user interface is not available.")
    return ui


def _safe_geometry_type(edge: adsk.fusion.BRepEdge) -> str:
    try:
        geometry = edge.geometry
        if geometry and geometry.objectType:
            return geometry.objectType.split("::")[-1]
    except Exception:
        pass
    return "Unknown"


def _edge_length(edge: adsk.fusion.BRepEdge) -> float:
    try:
        length_eval = edge.evaluator.getLength()
        if isinstance(length_eval, Iterable) and length_eval:
            # Autodesk returns (success, length) tuple
            length_tuple = list(length_eval)
            if len(length_tuple) >= 2 and length_tuple[0]:
                return float(length_tuple[1])
    except Exception:
        logger.debug("Falling back to edge.length for %s", edge)
    return float(getattr(edge, "length", 0.0))


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
        logger.debug("Failed to convert value %s from %s to %s; returning raw value", value, from_units, to_units)
        return float(value)


def _point_coords(
    vertex: Optional[adsk.fusion.BRepVertex],
    units_manager: adsk.core.UnitsManager,
    to_units: str,
) -> Optional[List[float]]:
    if not vertex or not vertex.geometry:
        return None

    point = vertex.geometry
    internal_units = units_manager.internalUnits
    return [
        _rounded(_convert_units(units_manager, point.x, internal_units, to_units)),
        _rounded(_convert_units(units_manager, point.y, internal_units, to_units)),
        _rounded(_convert_units(units_manager, point.z, internal_units, to_units)),
    ]


def _rounded(value: float, precision: int = 6) -> float:
    return round(value, precision)


def _build_edge_lookup(root_component: adsk.fusion.Component) -> Dict[str, adsk.fusion.BRepEdge]:
    lookup: Dict[str, adsk.fusion.BRepEdge] = {}
    try:
        for body in root_component.bRepBodies:
            for edge in body.edges:
                lookup[edge.entityToken] = edge
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed while building edge lookup: %s", exc)
    return lookup


__all__ = [
    "EdgeOperationError",
    "list_edges",
    "select_edges",
    "clear_edge_selection",
]
