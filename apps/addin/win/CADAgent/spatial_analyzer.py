"""
Spatial relationship analysis for improving LLM spatial awareness.

This module analyzes geometric relationships between bodies, faces, and features
to provide explicit mathematical descriptions that help LLMs reason about 3D space.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import adsk.core
import adsk.fusion

logger = logging.getLogger(__name__)


def extract_spatial_context(app: adsk.core.Application) -> Optional[Dict[str, Any]]:
    """
    Extract comprehensive spatial relationships from the active design.

    This provides the LLM with explicit spatial awareness about:
    - Parallel faces (useful for offset planes)
    - Body-to-body spatial relationships (distance, direction)
    - Overall workspace bounds

    Args:
        app: Fusion 360 Application instance

    Returns:
        Dictionary with spatial context, or None if extraction fails
    """
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            logger.debug("No active design for spatial context extraction")
            return None

        root_component = design.rootComponent

        context = {
            "units": {
                # Standardized to mm for user-friendly output (Fusion internal units are cm)
                "length": "mm",
                "coordinate_system": {
                    "origin": [0, 0, 0],
                    "x_axis": {"direction": [1, 0, 0], "label": "right (+X)"},
                    "y_axis": {"direction": [0, 1, 0], "label": "forward (+Y)"},
                    "z_axis": {"direction": [0, 0, 1], "label": "up (+Z)"}
                }
            }
        }

        # Extract body count and workspace bounds
        bodies = list(root_component.bRepBodies)
        context["overview"] = {
            "body_count": len(bodies),
            "workspace_bounds": _compute_workspace_bounds(bodies)
        }

        # Find parallel faces (high value for construction plane suggestions)
        context["parallel_faces"] = _find_parallel_face_groups(root_component)

        # Compute body-to-body relationships (if multiple bodies exist)
        if len(bodies) >= 2:
            context["body_relationships"] = _analyze_body_relationships(bodies)

        logger.info(f"Extracted spatial context: {len(bodies)} bodies, {len(context.get('parallel_faces', []))} parallel face groups")
        return context

    except Exception as e:
        logger.warning(f"Failed to extract spatial context: {e}")
        return None


def _compute_workspace_bounds(bodies: List[adsk.fusion.BRepBody]) -> Optional[Dict[str, Any]]:
    """Compute overall bounding box containing all bodies."""
    if not bodies:
        return None

    try:
        # Find min/max across all bodies
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for body in bodies:
            bbox = body.boundingBox
            if bbox:
                min_x = min(min_x, bbox.minPoint.x)
                min_y = min(min_y, bbox.minPoint.y)
                min_z = min(min_z, bbox.minPoint.z)
                max_x = max(max_x, bbox.maxPoint.x)
                max_y = max(max_y, bbox.maxPoint.y)
                max_z = max(max_z, bbox.maxPoint.z)

        if min_x == float('inf'):
            return None

        # Convert from cm (Fusion internal) to mm
        return {
            "min": [round(min_x * 10, 2), round(min_y * 10, 2), round(min_z * 10, 2)],
            "max": [round(max_x * 10, 2), round(max_y * 10, 2), round(max_z * 10, 2)],
            "dimensions": [
                round((max_x - min_x) * 10, 2),
                round((max_y - min_y) * 10, 2),
                round((max_z - min_z) * 10, 2)
            ]
        }
    except Exception as e:
        logger.debug(f"Failed to compute workspace bounds: {e}")
        return None


def _find_parallel_face_groups(component: adsk.fusion.Component) -> List[Dict[str, Any]]:
    """
    Find groups of parallel planar faces.

    Critical for LLM to suggest offset construction planes.
    """
    parallel_groups = []

    try:
        # Collect all planar faces with their normals
        planar_faces = []

        for body in component.bRepBodies:
            for face in body.faces:
                geometry = face.geometry
                if geometry and geometry.surfaceType == adsk.core.SurfaceTypes.PlaneSurfaceType:
                    normal = geometry.normal
                    planar_faces.append({
                        "face": face,
                        "token": face.entityToken,
                        "normal": [normal.x, normal.y, normal.z],
                        "centroid": face.centroid,
                        "area": face.area
                    })

        # Group faces by parallel normals (threshold: within 1 degree)
        angle_threshold = math.cos(math.radians(1.0))  # ~0.9998
        processed = set()

        for i, face_a in enumerate(planar_faces):
            if i in processed:
                continue

            group = {
                "normal_direction": _normalize_vector(face_a["normal"]),
                "faces": [face_a["token"]],
                "count": 1
            }

            # Find all faces parallel to this one
            for j, face_b in enumerate(planar_faces):
                if i == j or j in processed:
                    continue

                # Check if normals are parallel (dot product close to ±1)
                dot = _dot_product(face_a["normal"], face_b["normal"])

                if abs(dot) > angle_threshold:
                    group["faces"].append(face_b["token"])
                    group["count"] += 1
                    processed.add(j)

            processed.add(i)

            # Only include groups with 2+ faces
            if group["count"] >= 2:
                # Add spacing information for most common case (2 parallel faces)
                if group["count"] == 2:
                    face_data_a = planar_faces[i]
                    face_data_b = next(f for f in planar_faces if f["token"] == group["faces"][1])

                    distance = _point_to_plane_distance(
                        face_data_b["centroid"],
                        face_data_a["centroid"],
                        face_a["normal"]
                    )

                    # Convert from cm to mm
                    group["spacing"] = round(abs(distance) * 10, 3)

                parallel_groups.append(group)

        logger.debug(f"Found {len(parallel_groups)} parallel face groups")
        return parallel_groups

    except Exception as e:
        logger.warning(f"Failed to find parallel faces: {e}")
        return []


def _analyze_body_relationships(bodies: List[adsk.fusion.BRepBody]) -> List[Dict[str, Any]]:
    """
    Compute spatial relationships between bodies.

    Provides distance and direction vectors for LLM spatial reasoning.
    """
    relationships = []

    try:
        # Analyze pairwise relationships for first N bodies (avoid O(n²) explosion)
        max_pairs = 10
        pair_count = 0

        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                if pair_count >= max_pairs:
                    break

                body_a = bodies[i]
                body_b = bodies[j]

                # Get centers of mass
                try:
                    com_a = body_a.physicalProperties.centerOfMass
                    com_b = body_b.physicalProperties.centerOfMass
                except:
                    continue

                # Compute direction vector from A to B
                direction = [
                    com_b.x - com_a.x,
                    com_b.y - com_a.y,
                    com_b.z - com_a.z
                ]

                distance = math.sqrt(sum(d*d for d in direction))

                if distance < 0.001:  # Bodies essentially coincident
                    continue

                # Unit direction
                unit_dir = [d / distance for d in direction]

                # Dominant axis
                abs_dir = [abs(d) for d in unit_dir]
                max_component = max(abs_dir)
                dominant_axis_idx = abs_dir.index(max_component)
                dominant_axis = ['X', 'Y', 'Z'][dominant_axis_idx]

                # Convert distances from cm to mm
                distance_mm = distance * 10
                direction_mm = [d * 10 for d in direction]

                relationship = {
                    "body_a": body_a.name or f"Body {i}",
                    "body_b": body_b.name or f"Body {j}",
                    "distance": round(distance_mm, 2),
                    "direction_vector": [round(d, 3) for d in direction_mm],
                    "unit_direction": [round(d, 3) for d in unit_dir],
                    "dominant_axis": dominant_axis,
                    "description": f"{body_b.name or 'Body B'} is {distance_mm:.1f}mm in +{dominant_axis} direction from {body_a.name or 'Body A'}"
                }

                relationships.append(relationship)
                pair_count += 1

        return relationships

    except Exception as e:
        logger.warning(f"Failed to analyze body relationships: {e}")
        return []


# Vector math utilities

def _normalize_vector(v: List[float]) -> List[float]:
    """Normalize a vector to unit length."""
    magnitude = math.sqrt(sum(x*x for x in v))
    if magnitude < 1e-10:
        return [0, 0, 0]
    return [x / magnitude for x in v]


def _dot_product(a: List[float], b: List[float]) -> float:
    """Compute dot product of two 3D vectors."""
    return sum(a[i] * b[i] for i in range(3))


def _point_to_plane_distance(
    point: adsk.core.Point3D,
    plane_point: adsk.core.Point3D,
    plane_normal: List[float]
) -> float:
    """
    Compute signed distance from point to plane.

    Positive if point is on side of normal, negative otherwise.
    """
    # Vector from plane point to query point
    v = [
        point.x - plane_point.x,
        point.y - plane_point.y,
        point.z - plane_point.z
    ]

    # Distance is projection onto normal
    return _dot_product(v, plane_normal)


__all__ = [
    "extract_spatial_context"
]
