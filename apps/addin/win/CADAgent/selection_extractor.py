"""
Selection context extraction for Fusion 360 LLM CAD Agent.

This module extracts geometry information from user-selected entities
(faces, edges, bodies) to provide context to the LLM about what the user
has selected in the model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import adsk.core
import adsk.fusion

logger = logging.getLogger(__name__)


def extract_selection_context(app: adsk.core.Application) -> Optional[Dict[str, Any]]:
    """
    Extract geometry information from currently selected entities in Fusion 360.

    This function reads the user's active selections and extracts relevant
    geometric properties to provide context to the LLM. The LLM can use this
    information to understand what entities the user is referring to when
    making requests like "extrude this face 10mm" or "round these edges".

    Args:
        app: Fusion 360 Application instance

    Returns:
        Dictionary with selection context:
        {
            "count": 3,
            "entities": [
                {
                    "type": "face",
                    "geometry_type": "Plane",
                    "area": 25.3,  # cm²
                    "centroid": {"x": 5.0, "y": 0.0, "z": 10.0},
                    "normal": {"x": 0.0, "y": 0.0, "z": 1.0}
                },
                {
                    "type": "edge",
                    "geometry_type": "Line",
                    "length": 10.5,  # cm
                    "start": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "end": {"x": 10.5, "y": 0.0, "z": 0.0}
                },
                {
                    "type": "body",
                    "name": "Body1",
                    "volume": 125.6,  # cm³
                    "is_solid": true,
                    "bounding_box": {
                        "min": {"x": 0, "y": 0, "z": 0},
                        "max": {"x": 10, "y": 5, "z": 5}
                    }
                }
            ]
        }

        Returns None if no selections or extraction fails.
    """
    try:
        if not app:
            logger.warning("No application instance provided for selection extraction")
            return None

        ui = app.userInterface
        if not ui:
            logger.warning("No UI available for selection extraction")
            return None

        active_selections = ui.activeSelections
        if not active_selections or active_selections.count == 0:
            logger.debug("No active selections found")
            return None

        entities: List[Dict[str, Any]] = []

        for i in range(active_selections.count):
            selection = active_selections.item(i)
            entity = selection.entity

            if not entity:
                continue

            # Extract info based on entity type
            entity_info = _extract_entity_info(entity)
            if entity_info:
                entities.append(entity_info)

        if not entities:
            logger.debug("No extractable entity information from selections")
            return None

        logger.info(f"Extracted selection context for {len(entities)} entities")
        return {
            "count": len(entities),
            "entities": entities
        }

    except Exception as e:
        logger.warning(f"Failed to extract selection context: {e}")
        return None


def _extract_entity_info(entity: Any) -> Optional[Dict[str, Any]]:
    """
    Extract relevant information from a single selected entity.

    Args:
        entity: Selected Fusion 360 entity (BRepFace, BRepEdge, BRepBody, etc.)

    Returns:
        Dictionary with entity information, or None if extraction fails
    """
    try:
        object_type = entity.objectType

        # Check for BRepFace
        if _is_type(object_type, "BRepFace"):
            return _extract_face_info(adsk.fusion.BRepFace.cast(entity))

        # Check for BRepEdge
        elif _is_type(object_type, "BRepEdge"):
            return _extract_edge_info(adsk.fusion.BRepEdge.cast(entity))

        # Check for BRepBody
        elif _is_type(object_type, "BRepBody"):
            return _extract_body_info(adsk.fusion.BRepBody.cast(entity))

        else:
            # Other entity types (vertices, sketches, etc.) - basic info only
            logger.debug(f"Unsupported entity type for detailed extraction: {object_type}")
            return {
                "type": "other",
                "object_type": object_type
            }

    except Exception as e:
        logger.warning(f"Failed to extract entity info: {e}")
        return None


def _extract_face_info(face: adsk.fusion.BRepFace) -> Dict[str, Any]:
    """Extract geometry information from a BRepFace."""
    info: Dict[str, Any] = {
        "type": "face"
    }

    try:
        # Get geometry type
        geometry = face.geometry
        if geometry:
            geometry_type = geometry.objectType.split("::")[-1]
            info["geometry_type"] = geometry_type

        # Get area (in cm²)
        info["area"] = round(face.area, 3)

        # Get centroid
        centroid = face.centroid
        if centroid:
            info["centroid"] = {
                "x": round(centroid.x, 3),
                "y": round(centroid.y, 3),
                "z": round(centroid.z, 3)
            }

        # Get surface normal at centroid
        evaluator = face.evaluator
        if evaluator and centroid:
            success, normal = evaluator.getNormalAtPoint(centroid)
            if success and normal:
                info["normal"] = {
                    "x": round(normal.x, 3),
                    "y": round(normal.y, 3),
                    "z": round(normal.z, 3)
                }

        # Get bounding box for spatial context
        bbox = face.boundingBox
        if bbox:
            info["bounding_box"] = {
                "min": {
                    "x": round(bbox.minPoint.x, 3),
                    "y": round(bbox.minPoint.y, 3),
                    "z": round(bbox.minPoint.z, 3)
                },
                "max": {
                    "x": round(bbox.maxPoint.x, 3),
                    "y": round(bbox.maxPoint.y, 3),
                    "z": round(bbox.maxPoint.z, 3)
                }
            }

    except Exception as e:
        logger.debug(f"Error extracting face details: {e}")

    return info


def _extract_edge_info(edge: adsk.fusion.BRepEdge) -> Dict[str, Any]:
    """Extract geometry information from a BRepEdge."""
    info: Dict[str, Any] = {
        "type": "edge"
    }

    try:
        # Get geometry type
        geometry = edge.geometry
        if geometry:
            geometry_type = geometry.objectType.split("::")[-1]
            info["geometry_type"] = geometry_type

        # Get length (in cm)
        info["length"] = round(edge.length, 3)

        # Get start and end vertices
        start_vertex = edge.startVertex
        end_vertex = edge.endVertex

        if start_vertex and start_vertex.geometry:
            info["start"] = {
                "x": round(start_vertex.geometry.x, 3),
                "y": round(start_vertex.geometry.y, 3),
                "z": round(start_vertex.geometry.z, 3)
            }

        if end_vertex and end_vertex.geometry:
            info["end"] = {
                "x": round(end_vertex.geometry.x, 3),
                "y": round(end_vertex.geometry.y, 3),
                "z": round(end_vertex.geometry.z, 3)
            }

        # Get bounding box
        bbox = edge.boundingBox
        if bbox:
            info["bounding_box"] = {
                "min": {
                    "x": round(bbox.minPoint.x, 3),
                    "y": round(bbox.minPoint.y, 3),
                    "z": round(bbox.minPoint.z, 3)
                },
                "max": {
                    "x": round(bbox.maxPoint.x, 3),
                    "y": round(bbox.maxPoint.y, 3),
                    "z": round(bbox.maxPoint.z, 3)
                }
            }

    except Exception as e:
        logger.debug(f"Error extracting edge details: {e}")

    return info


def _extract_body_info(body: adsk.fusion.BRepBody) -> Dict[str, Any]:
    """Extract geometry information from a BRepBody."""
    info: Dict[str, Any] = {
        "type": "body"
    }

    try:
        # Get name
        info["name"] = body.name or "Unnamed"

        # Get volume (in cm³) and area (in cm²)
        info["volume"] = round(body.volume, 3)
        info["area"] = round(body.area, 3)

        # Check if solid
        info["is_solid"] = body.isSolid

        # Get bounding box
        bbox = body.boundingBox
        if bbox:
            info["bounding_box"] = {
                "min": {
                    "x": round(bbox.minPoint.x, 3),
                    "y": round(bbox.minPoint.y, 3),
                    "z": round(bbox.minPoint.z, 3)
                },
                "max": {
                    "x": round(bbox.maxPoint.x, 3),
                    "y": round(bbox.maxPoint.y, 3),
                    "z": round(bbox.maxPoint.z, 3)
                }
            }

        # Get face count for additional context
        if body.faces:
            info["face_count"] = body.faces.count

        if body.edges:
            info["edge_count"] = body.edges.count

    except Exception as e:
        logger.debug(f"Error extracting body details: {e}")

    return info


def _is_type(object_type: str, expected_type: str) -> bool:
    """
    Check if an object type matches the expected type.

    Handles both fully-qualified types (adsk::fusion::BRepFace)
    and simple types (BRepFace).
    """
    return expected_type in object_type or object_type.endswith(f"::{expected_type}")


def format_selection_context_for_llm(selection_context: Optional[Dict[str, Any]]) -> str:
    """
    Format selection context into a concise, human-readable description for the LLM.

    Args:
        selection_context: Selection context dictionary from extract_selection_context()

    Returns:
        Formatted string describing the selected entities, or empty string if no selections
    """
    if not selection_context or not selection_context.get("entities"):
        return ""

    entities = selection_context["entities"]
    count = selection_context.get("count", len(entities))

    lines = [f"The user has selected {count} entity/entities:"]

    for idx, entity in enumerate(entities, 1):
        entity_type = entity.get("type", "unknown")

        if entity_type == "face":
            desc = _format_face_description(idx, entity)
        elif entity_type == "edge":
            desc = _format_edge_description(idx, entity)
        elif entity_type == "body":
            desc = _format_body_description(idx, entity)
        else:
            desc = f"  {idx}. {entity_type.capitalize()}: {entity.get('object_type', 'Unknown')}"

        lines.append(desc)

    return "\n".join(lines)


def _format_face_description(idx: int, face_info: Dict[str, Any]) -> str:
    """Format face information for LLM display."""
    geo_type = face_info.get("geometry_type", "Unknown")
    area = face_info.get("area", 0)

    parts = [f"  {idx}. Face ({geo_type}): {area} cm²"]

    centroid = face_info.get("centroid")
    if centroid:
        parts.append(f", centroid at ({centroid['x']}, {centroid['y']}, {centroid['z']})")

    normal = face_info.get("normal")
    if normal:
        # Describe normal direction in simple terms
        direction = _describe_normal_direction(normal)
        if direction:
            parts.append(f", normal pointing {direction}")

    return "".join(parts)


def _format_edge_description(idx: int, edge_info: Dict[str, Any]) -> str:
    """Format edge information for LLM display."""
    geo_type = edge_info.get("geometry_type", "Unknown")
    length = edge_info.get("length", 0)

    parts = [f"  {idx}. Edge ({geo_type}): {length} cm long"]

    start = edge_info.get("start")
    end = edge_info.get("end")
    if start and end:
        parts.append(f" from ({start['x']}, {start['y']}, {start['z']}) to ({end['x']}, {end['y']}, {end['z']})")

    return "".join(parts)


def _format_body_description(idx: int, body_info: Dict[str, Any]) -> str:
    """Format body information for LLM display."""
    name = body_info.get("name", "Unnamed")
    volume = body_info.get("volume", 0)
    is_solid = body_info.get("is_solid", False)

    parts = [f"  {idx}. Body '{name}': {volume} cm³"]

    if is_solid:
        parts.append(" (solid)")
    else:
        parts.append(" (surface/open)")

    face_count = body_info.get("face_count")
    if face_count:
        parts.append(f", {face_count} faces")

    return "".join(parts)


def _describe_normal_direction(normal: Dict[str, float]) -> str:
    """
    Describe a surface normal vector in simple directional terms.

    Returns description like "+Z", "-X", "+X+Y", etc.
    """
    x = normal.get("x", 0)
    y = normal.get("y", 0)
    z = normal.get("z", 0)

    # Threshold for considering a component significant
    threshold = 0.5

    parts = []

    if abs(x) > threshold:
        parts.append("+" if x > 0 else "-")
        parts.append("X")

    if abs(y) > threshold:
        parts.append("+" if y > 0 else "-")
        parts.append("Y")

    if abs(z) > threshold:
        parts.append("+" if z > 0 else "-")
        parts.append("Z")

    return "".join(parts) if parts else f"({x:.2f}, {y:.2f}, {z:.2f})"
