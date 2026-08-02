"""
Body metadata utilities for CADAgent.

Provides helpers to extract, select, and clear bodies in the active Fusion 360
design so backend tools can operate without embedding raw Fusion API logic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import adsk.core
import adsk.fusion

logger = logging.getLogger(__name__)

from .entity_iteration import iter_brep_bodies


class BodyOperationError(RuntimeError):
    """Raised when a body operation cannot be fulfilled."""


def _point_coords_mm(point: Optional[adsk.core.Point3D]) -> Optional[List[float]]:
    """Convert a Point3D to mm coordinates (Fusion internal is cm)."""
    if not point:
        return None
    return [
        round(point.x * 10.0, 6),  # cm -> mm
        round(point.y * 10.0, 6),
        round(point.z * 10.0, 6),
    ]


def _bounding_box_coords_mm(bbox: Optional[adsk.core.BoundingBox3D]) -> Optional[Dict[str, List[float]]]:
    """Convert a BoundingBox3D to mm coordinates."""
    if not bbox:
        return None
    return {
        "min": _point_coords_mm(bbox.minPoint),
        "max": _point_coords_mm(bbox.maxPoint),
    }


def _collect_vertices(body: adsk.fusion.BRepBody) -> List[Dict[str, Any]]:
    """
    Collect all vertices from a body.
    
    Returns:
        List of vertex dictionaries with id, token, and position in mm.
    """
    vertices: List[Dict[str, Any]] = []
    try:
        for vertex_index, vertex in enumerate(body.vertices):
            vertex_info = {
                "id": f"v{vertex_index}",
                "token": vertex.entityToken,
                "p": _point_coords_mm(vertex.geometry),
            }
            vertices.append(vertex_info)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug("Failed to collect vertices: %s", exc)
    return vertices


def list_bodies(
    app: adsk.core.Application,
    design: Optional[adsk.fusion.Design] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Collect metadata for every body in the active design, including basic
    physical properties, bounding box coordinates, and vertices.

    Returns:
        Tuple where the first value is a list of body dictionaries and the second
        value is a mapping of unit identifiers (`length`, `area`, `volume`, `mass`, `density`).
    """
    if design is None or not getattr(design, "isValid", True):
        design = _require_design(app)
    root_component = design.rootComponent
    units_manager = design.unitsManager

    # Units in mm (convert from Fusion internal cm)
    length_units = "mm"
    area_units = "mm^2"
    volume_units = "mm^3"
    mass_units = getattr(units_manager, "defaultMassUnits", None) or "kg"
    density_units = f"{mass_units}/mm^3"

    internal_length_units = units_manager.internalUnits  # "cm"
    internal_area_units = _internal_area_units(internal_length_units)
    internal_volume_units = _internal_volume_units(internal_length_units)

    bodies: List[Dict[str, Any]] = []

    try:
        for body_index, (ctx, body) in enumerate(iter_brep_bodies(design)):
            body_name = body.name or f"Body {body_index}"
            component = getattr(body, "parentComponent", None)
            # Prefer occurrence/component context for inserted components.
            component_name = ctx.get("component") or getattr(component, "name", None) or root_component.name
            occurrence_path = ctx.get("occurrence_path")

            # Area in mm^2 (Fusion internal is cm^2, multiply by 100)
            area_val = getattr(body, "area", 0.0) * 100.0
            
            # Volume in mm^3 (Fusion internal is cm^3, multiply by 1000)
            volume_val = getattr(body, "volume", 0.0) * 1000.0

            # Bounding box in mm
            bbox = _bounding_box_coords_mm(getattr(body, "boundingBox", None))

            # Compute dimensions from bounding box (in mm)
            dimensions = None
            if bbox:
                min_pt = bbox.get("min")
                max_pt = bbox.get("max")
                if min_pt and max_pt:
                    dimensions = [
                        _rounded(max_pt[0] - min_pt[0]),
                        _rounded(max_pt[1] - min_pt[1]),
                        _rounded(max_pt[2] - min_pt[2]),
                    ]

            # Collect vertices
            vertices = _collect_vertices(body)

            phys_props = _physical_properties_mm(
                body,
                mass_units,
                density_units,
            )

            body_info: Dict[str, Any] = {
                "id": f"body_{body_index}",  # Updated ID format per spec
                "temp_id": _safe_temp_id(body),
                "name": body_name,
                "token": getattr(body, "entityToken", None),  # Added as 'token' per spec
                "component": component_name,
                "occurrence_path": occurrence_path,
                "body_index": body_index,
                "is_solid": bool(getattr(body, "isSolid", False)),
                "is_visible": bool(getattr(body, "isVisible", False)),
                "volume": _rounded(volume_val),
                "area": _rounded(area_val),
                "volume_units": volume_units,
                "area_units": area_units,
                "bbox": bbox,  # New key name per spec
                "bounding_box": bbox,  # Legacy for compatibility
                "dimensions": dimensions,
                "face_count": getattr(body.faces, "count", 0),
                "edge_count": getattr(body.edges, "count", 0),
                "vertex_count": getattr(body.vertices, "count", 0),
                "entity_token": getattr(body, "entityToken", None),  # Legacy for compatibility
                "vertices": vertices,  # NEW: List of vertices with id, token, position
            }

            body_info.update(phys_props)

            bodies.append(body_info)

    except Exception as exc:  # pragma: no cover - defensive for Fusion runtime issues
        logger.exception("Failed while extracting bodies")
        raise BodyOperationError(f"Failed to extract bodies: {exc}") from exc

    logger.info("Extracted %d bodies (length units: %s, volume units: %s)", len(bodies), length_units, volume_units)
    return bodies, {
        "length": length_units,
        "area": area_units,
        "volume": volume_units,
        "mass": mass_units,
        "density": density_units,
    }


def select_bodies(
    app: adsk.core.Application,
    entity_tokens: Sequence[str],
    *,
    clear_existing: bool = True,
) -> Dict[str, Any]:
    """
    Select bodies identified by entity tokens.

    Args:
        app: Fusion application instance.
        entity_tokens: Iterable of body entity tokens.
        clear_existing: When True (default) clears the current selection set first.

    Returns:
        Dictionary with counts and any missing tokens.
    """
    tokens = [token for token in entity_tokens if isinstance(token, str) and token.strip()]
    if not tokens:
        raise BodyOperationError("No entity tokens provided for selection.")

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
            body = adsk.fusion.BRepBody.cast(resolved[0])
            if not body:
                missing_tokens.append(token)
                continue
            ui.activeSelections.add(body)
            selected_count += 1
        except Exception as exc:  # pragma: no cover - should be rare
            logger.warning("Failed to add body with token %s to selection: %s", token, exc)
            missing_tokens.append(token)

    success = selected_count > 0
    if not success:
        raise BodyOperationError("None of the provided entity tokens matched bodies in the design.")

    logger.info(
        "Selected %d body(s); %d token(s) missing",
        selected_count,
        len(missing_tokens),
    )

    return {
        "selected_count": selected_count,
        "missing_tokens": missing_tokens,
        "cleared_existing": clear_existing,
    }


def clear_body_selection(app: adsk.core.Application) -> Dict[str, Any]:
    """
    Clear the current Fusion body selection set.

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
        raise BodyOperationError(f"Failed to clear selections: {exc}") from exc

    logger.info("Cleared %d selection(s)", cleared_count)
    return {"cleared_count": cleared_count}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_design(app: Optional[adsk.core.Application]) -> adsk.fusion.Design:
    if not app:
        raise BodyOperationError("Fusion application is not available.")

    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise BodyOperationError("No active design is available. Open or create a design and try again.")
    return design


def _require_ui(app: Optional[adsk.core.Application]) -> adsk.core.UserInterface:
    if not app:
        raise BodyOperationError("Fusion application is not available.")

    ui = app.userInterface
    if not ui:
        raise BodyOperationError("Fusion user interface is not available.")
    return ui


def _convert_units(
    units_manager: adsk.core.UnitsManager,
    value: Optional[float],
    from_units: str,
    to_units: str,
) -> float:
    if value is None:
        return 0.0
    try:
        return float(units_manager.convert(float(value), from_units, to_units))
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


def _physical_properties_mm(
    body: adsk.fusion.BRepBody,
    mass_units: str,
    density_units: str,
) -> Dict[str, Any]:
    """
    Extract physical properties with center of mass in mm.
    """
    props: Dict[str, Any] = {
        "center_of_mass": None,
        "mass": None,
        "mass_units": mass_units,
        "density": None,
        "density_units": density_units,
    }

    try:
        phys_props = getattr(body, "physicalProperties", None)
        if not phys_props:
            return props

        center = getattr(phys_props, "centerOfMass", None)
        if center:
            props["center_of_mass"] = _point_coords_mm(center)

        mass_value = getattr(phys_props, "mass", None)
        if mass_value is not None:
            props["mass"] = _rounded(float(mass_value))
            props["mass_units"] = mass_units

        density_value = getattr(phys_props, "density", None)
        if density_value is not None:
            # Density doesn't need conversion - it's mass/volume ratio
            props["density"] = _rounded(float(density_value))

    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to extract physical properties for body %s: %s", body, exc)

    return props


def _physical_properties(
    body: adsk.fusion.BRepBody,
    units_manager: adsk.core.UnitsManager,
    from_units: str,
    to_units: str,
    mass_units: str,
    density_units: str,
) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "center_of_mass": None,
        "mass": None,
        "mass_units": mass_units,
        "density": None,
        "density_units": density_units,
    }

    try:
        phys_props = getattr(body, "physicalProperties", None)
        if not phys_props:
            return props

        center = getattr(phys_props, "centerOfMass", None)
        if center:
            props["center_of_mass"] = _point_coords(center, units_manager, from_units, to_units)

        mass_value = getattr(phys_props, "mass", None)
        if mass_value is not None:
            try:
                phys_mass_units = getattr(phys_props, "massUnit", mass_units)
                converted = units_manager.convert(float(mass_value), phys_mass_units, mass_units)
                props["mass"] = _rounded(float(converted))
                props["mass_units"] = mass_units
            except Exception:
                props["mass"] = _rounded(float(mass_value))

        density_value = getattr(phys_props, "density", None)
        if density_value is not None:
            # Preserve Fusion's density unit if available; otherwise use fallback
            # Note: Fusion internally uses g/cm³ for density, so our cm³-based default is appropriate
            actual_density_unit = getattr(phys_props, "densityUnit", None)
            if actual_density_unit:
                props["density_units"] = str(actual_density_unit)
            # else: keep the density_units passed in (already set in props init)
            props["density"] = _rounded(float(density_value))

    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to extract physical properties for body %s: %s", body, exc)

    return props


def _rounded(value: float, precision: int = 6) -> float:
    return round(value, precision)


def _internal_area_units(internal_length_units: str) -> str:
    if internal_length_units.endswith("^2"):
        return internal_length_units
    return f"{internal_length_units}^2"


def _internal_volume_units(internal_length_units: str) -> str:
    if internal_length_units.endswith("^3"):
        return internal_length_units
    return f"{internal_length_units}^3"


def _build_body_lookup(root_component: adsk.fusion.Component) -> Dict[str, adsk.fusion.BRepBody]:
    lookup: Dict[str, adsk.fusion.BRepBody] = {}
    try:
        for body in root_component.bRepBodies:
            lookup[getattr(body, "entityToken", None)] = body
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed while building body lookup: %s", exc)
    return {token: body for token, body in lookup.items() if token}


def _safe_temp_id(entity: Any) -> Optional[int]:
    try:
        temp_id = getattr(entity, "tempId", None)
        if temp_id is not None:
            return int(temp_id)
    except Exception:
        pass
    return None


__all__ = [
    "BodyOperationError",
    "list_bodies",
    "select_bodies",
    "clear_body_selection",
]
