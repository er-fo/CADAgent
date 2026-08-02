"""
Construction Plane Manager for Fusion 360 CAD Agent

Manages creation and lifecycle of construction planes used as sketch bases.
Supports 4 creation modes for 93% of typical CAD workflows:
- datum: Built-in XY/XZ/YZ planes
- offset_from_datum: Parallel offset planes
- angle_to_edge: Planes rotated around edges (THE reason for this feature)
- face_normal: Planes coincident with existing faces
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import adsk.core
import adsk.fusion

from .selection_extractor import _is_type

logger = logging.getLogger(__name__)


class PlaneCreationError(Exception):
    """Raised when plane creation fails."""
    pass


class PlaneManager:
    """
    Manages construction planes for the active design.

    Planes are stored by plane_id and persist across multiple tool executions
    within the same design session.
    """

    def __init__(self, root_component: adsk.fusion.Component):
        """
        Initialize the plane manager.

        Args:
            root_component: Root component of the active design
        """
        self.root_component = root_component
        self.planes: Dict[str, adsk.fusion.ConstructionPlane] = {
            "XY": root_component.xYConstructionPlane,
            "XZ": root_component.xZConstructionPlane,
            "YZ": root_component.yZConstructionPlane,
        }
        self.axes: Dict[str, adsk.fusion.ConstructionAxis] = {}
        logger.info("PlaneManager initialized with datum planes: XY, XZ, YZ")

    def get_plane(self, plane_id: str) -> adsk.fusion.ConstructionPlane:
        """
        Retrieve a plane by ID, or auto-create from a face entity token.

        Args:
            plane_id: Unique identifier for the plane, OR a Fusion entity token
                      for a planar face (auto-creates construction plane)

        Returns:
            ConstructionPlane object

        Raises:
            PlaneCreationError: If plane_id not found and not a valid face token
        """
        if plane_id in self.planes:
            return self.planes[plane_id]

        # Check if this looks like a Fusion entity token (face)
        # Tokens typically contain patterns like "BRepFace" or similar Fusion identifiers
        if self._is_face_entity_token(plane_id):
            logger.info(f"Auto-creating construction plane from face token: {plane_id}")
            auto_plane_id = f"_auto_face_plane_{len(self.planes)}"
            plane = self.create_plane(
                plane_id=auto_plane_id,
                mode="face_normal",
                description=f"Auto-created plane for sketching on face",
                face_token=plane_id,
            )
            return plane

        available = ", ".join(sorted(self.planes.keys()))
        raise PlaneCreationError(
            f"Plane '{plane_id}' not found. Available planes: {available}"
        )

    def _is_face_entity_token(self, value: str) -> bool:
        """
        Check if a string looks like a Fusion 360 face entity token.

        Entity tokens are opaque strings but typically contain type indicators.
        """
        if not value or not isinstance(value, str):
            return False
        # Fusion entity tokens for faces often contain these patterns
        # Also check if it can be resolved via findEntityByToken
        try:
            app = adsk.core.Application.get()
            design = adsk.fusion.Design.cast(app.activeProduct)
            if not design:
                return False
            entities = design.findEntityByToken(value)
            if entities and len(entities) > 0:
                entity = entities[0]
                # Check if it's a planar face
                if _is_type(entity.objectType, "BRepFace"):
                    face = adsk.fusion.BRepFace.cast(entity)
                    # Only accept planar faces for sketching
                    if face and face.geometry and _is_type(face.geometry.objectType, "Plane"):
                        return True
        except Exception:
            pass
        return False

    def create_plane(
        self,
        plane_id: str,
        mode: str,
        description: str = "",
        **kwargs
    ) -> adsk.fusion.ConstructionPlane:
        """
        Create a construction plane using one of 4 supported modes.

        Args:
            plane_id: Unique identifier for this plane
            mode: Creation mode ('datum', 'offset_from_datum', 'angle_to_edge', 'face_normal')
            description: Human-readable description for logging
            **kwargs: Mode-specific parameters

        Returns:
            Created ConstructionPlane object

        Raises:
            PlaneCreationError: If plane creation fails
        """
        logger.info(f"Creating plane '{plane_id}' with mode='{mode}': {description}")

        if plane_id in self.planes:
            logger.warning(f"Plane '{plane_id}' already exists; returning existing plane")
            return self.planes[plane_id]

        try:
            if mode == "datum":
                plane = self._create_datum_plane(kwargs)
            elif mode == "offset_from_datum":
                plane = self._create_offset_plane(kwargs)
            elif mode == "angle_to_edge":
                plane = self._create_angled_plane(kwargs)
            elif mode == "face_normal":
                plane = self._create_face_plane(kwargs)
            else:
                raise PlaneCreationError(
                    f"Unknown plane mode '{mode}'. "
                    f"Supported modes: datum, offset_from_datum, angle_to_edge, face_normal"
                )

            # Name the plane in the timeline for user visibility
            if plane and hasattr(plane, 'timelineObject') and plane.timelineObject:
                if description and description.strip():
                    # Use description if provided (e.g., "Offset plane for side sketch")
                    plane.timelineObject.name = description.strip()
                else:
                    # Fall back to plane_id (e.g., "base_plane", "angled_plane_1")
                    plane.timelineObject.name = plane_id

            self.planes[plane_id] = plane
            logger.info(f"✓ Successfully created plane '{plane_id}'")
            return plane

        except PlaneCreationError:
            raise
        except Exception as exc:
            raise PlaneCreationError(f"Failed to create plane '{plane_id}': {exc}") from exc

    def _create_datum_plane(self, params: Dict[str, Any]) -> adsk.fusion.ConstructionPlane:
        """
        Mode 'datum': Return a built-in XY/XZ/YZ plane.

        Required params:
            datum_axis_plane: "XY" | "XZ" | "YZ"
        """
        datum_axis = params.get("datum_axis_plane", "").upper()
        if datum_axis not in ("XY", "XZ", "YZ"):
            raise PlaneCreationError(
                f"Invalid datum_axis_plane '{datum_axis}'. Expected: XY, XZ, or YZ"
            )

        # Return the built-in plane
        if datum_axis == "XY":
            return self.root_component.xYConstructionPlane
        elif datum_axis == "XZ":
            return self.root_component.xZConstructionPlane
        else:  # YZ
            return self.root_component.yZConstructionPlane

    def _create_offset_plane(self, params: Dict[str, Any]) -> adsk.fusion.ConstructionPlane:
        """
        Mode 'offset_from_datum': Create a plane offset from XY/XZ/YZ.

        Required params:
            base_datum_plane: "XY" | "XZ" | "YZ"
            offset_cm: float (signed distance in cm)

        Fusion API: ConstructionPlaneInput.setByOffset(planarEntity, offset)
        """
        base_datum = params.get("base_datum_plane", "").upper()
        offset_cm = params.get("offset_cm")

        if base_datum not in ("XY", "XZ", "YZ"):
            raise PlaneCreationError(f"Invalid base_datum_plane '{base_datum}'")
        if offset_cm is None:
            raise PlaneCreationError("offset_cm parameter is required")

        # Get the base plane
        if base_datum == "XY":
            base_plane = self.root_component.xYConstructionPlane
        elif base_datum == "XZ":
            base_plane = self.root_component.xZConstructionPlane
        else:
            base_plane = self.root_component.yZConstructionPlane

        # Create offset plane
        planes = self.root_component.constructionPlanes
        plane_input = planes.createInput()
        offset_value = adsk.core.ValueInput.createByReal(float(offset_cm))
        plane_input.setByOffset(base_plane, offset_value)

        created_plane = planes.add(plane_input)
        logger.debug(f"Created offset plane: {offset_cm} cm from {base_datum}")
        return created_plane

    def _resolve_builtin_reference(
        self,
        token: str,
        expected_category: str  # "axis" or "plane"
    ) -> Optional[Any]:
        """
        Resolve built-in construction reference tokens.

        Args:
            token: Token string ("X", "Y", "Z", "XY", "XZ", "YZ")
            expected_category: "axis" for edges, "plane" for faces

        Returns:
            Fusion 360 construction object, or None if not a built-in token
        """
        if expected_category == "axis":
            # Built-in construction axes
            if token == "X":
                return self.root_component.xConstructionAxis
            elif token == "Y":
                return self.root_component.yConstructionAxis
            elif token == "Z":
                return self.root_component.zConstructionAxis

        elif expected_category == "plane":
            # Built-in datum planes
            if token == "XY":
                return self.root_component.xYConstructionPlane
            elif token == "XZ":
                return self.root_component.xZConstructionPlane
            elif token == "YZ":
                return self.root_component.yZConstructionPlane

        return None

    def _create_axis_from_plane_direction(
        self,
        plane_id: str,
        axis_name: str
    ) -> adsk.fusion.ConstructionAxis:
        """
        Create a ConstructionAxis from a plane's local direction.

        Args:
            plane_id: ID of the plane whose local axis to extract
            axis_name: 'U', 'V', or 'N' (local U-axis, V-axis, or normal)

        Returns:
            ConstructionAxis object aligned with the specified local direction

        Raises:
            PlaneCreationError: If plane not found or invalid axis name
        """
        # Check if plane exists
        if plane_id not in self.planes:
            available = ", ".join(sorted(self.planes.keys()))
            raise PlaneCreationError(
                f"Plane '{plane_id}' not found for local axis extraction. "
                f"Available planes: {available}"
            )

        # Get the plane's geometry
        plane = self.planes[plane_id]
        plane_geometry = plane.geometry

        # Select the direction vector based on axis name
        if axis_name == 'U':
            direction = plane_geometry.uDirection
        elif axis_name == 'V':
            direction = plane_geometry.vDirection
        elif axis_name == 'N':
            direction = plane_geometry.normal
        else:
            raise PlaneCreationError(
                f"Invalid local axis '{axis_name}' for plane '{plane_id}'. "
                f"Valid axes: U (local U-direction), V (local V-direction), N (normal)"
            )

        # Create two points along the direction vector
        origin = plane_geometry.origin
        # Scale direction vector to ensure points are far enough apart for Fusion validation
        # Direction vectors from plane geometry are unit vectors (length 1.0 cm), scale to 100 cm
        scaled_direction = direction.copy()
        scaled_direction.scaleBy(100.0)
        # Create a second point along the direction (distance is arbitrary, axis is infinite)
        second_point = origin.copy()
        second_point.translateBy(scaled_direction)

        # Create the ConstructionAxis using two points (works in both Parametric and Direct modes)
        construction_axes = self.root_component.constructionAxes
        axis_input = construction_axes.createInput()
        axis_input.setByTwoPoints(origin, second_point)

        created_axis = construction_axes.add(axis_input)
        logger.debug(f"Created local axis '{axis_name}' from plane '{plane_id}'")

        return created_axis

    def _resolve_local_axis_token(
        self,
        token: str
    ) -> Optional[adsk.fusion.ConstructionAxis]:
        """
        Resolve local axis token format 'plane_id:axis'.

        Args:
            token: Token string in format 'plane_id:U', 'plane_id:V', or 'plane_id:N'

        Returns:
            ConstructionAxis object, or None if not a local axis token
        """
        # Check if token contains ':' separator
        if ':' not in token:
            return None

        # Parse token
        parts = token.split(':', 1)
        if len(parts) != 2:
            return None

        plane_id, axis_name = parts
        axis_name = axis_name.upper()

        # Validate axis name
        if axis_name not in ('U', 'V', 'N'):
            return None

        # Check if we've already created this axis (cache)
        cache_key = f"{plane_id}:{axis_name}"
        if cache_key in self.axes:
            return self.axes[cache_key]

        # Create new axis
        try:
            axis = self._create_axis_from_plane_direction(plane_id, axis_name)
            self.axes[cache_key] = axis
            return axis
        except PlaneCreationError:
            # Let the error propagate with helpful context
            raise

    def _create_angled_plane(self, params: Dict[str, Any]) -> adsk.fusion.ConstructionPlane:
        """
        Mode 'angle_to_edge': Create a plane rotated around an edge at an angle.

        Required params:
            reference_face_token: str (plane_id, built-in plane, or entity token from list_faces)
            reference_edge_token: str (world axis X/Y/Z, local axis plane_id:U/V/N, or entity token from list_edges)
            angle_deg: float (rotation angle in degrees)

        Fusion API: ConstructionPlaneInput.setByAngle(linearEntity, angle, planarEntity)
        """
        face_token = params.get("reference_face_token")
        edge_token = params.get("reference_edge_token")
        angle_deg = params.get("angle_deg")

        if not face_token:
            raise PlaneCreationError("reference_face_token is required for mode='angle_to_edge'")
        if not edge_token:
            raise PlaneCreationError("reference_edge_token is required for mode='angle_to_edge'")
        if angle_deg is None:
            raise PlaneCreationError("angle_deg is required for mode='angle_to_edge'")

        normalized_face_token = face_token.upper()
        if normalized_face_token in ("XY", "XZ", "YZ"):
            face_token = normalized_face_token
        else:
            raise PlaneCreationError(
                "reference_face_token must be one of XY, XZ, or YZ. Chained angled planes are not supported."
            )

        if ':' in edge_token:
            raise PlaneCreationError(
                "reference_edge_token cannot reference local plane axes (e.g., plane_id:U). "
                "Angled planes must use global axes or explicit solid edges."
            )

        # Resolve entity tokens to actual Fusion objects
        # Try built-in references first, then custom planes, then entity tokens
        face = self._resolve_builtin_reference(face_token, "plane")
        if face is None and face_token in self.planes:
            face = self.planes[face_token]
        if face is None:
            face = self._resolve_entity_token(face_token, expected_type="BRepFace")

        edge = self._resolve_builtin_reference(edge_token, "axis")
        if edge is None:
            edge = self._resolve_local_axis_token(edge_token)
        if edge is None:
            edge = self._resolve_entity_token(edge_token, expected_type="BRepEdge")

        # Create angled plane
        planes = self.root_component.constructionPlanes
        plane_input = planes.createInput()
        angle_rad = math.radians(float(angle_deg))
        angle_value = adsk.core.ValueInput.createByReal(angle_rad)

        # API: setByAngle(linearEntity, angle, planarEntity)
        plane_input.setByAngle(edge, angle_value, face)

        created_plane = planes.add(plane_input)
        logger.debug(f"Created angled plane: {angle_deg}° around edge relative to face")
        return created_plane

    def _create_face_plane(self, params: Dict[str, Any]) -> adsk.fusion.ConstructionPlane:
        """
        Mode 'face_normal': Create a plane coincident with an existing face.

        Required params:
            face_token: str (entity token from list_faces)

        Optional params:
            point_x, point_y, point_z: Origin point (defaults to face centroid)

        Fusion API: ConstructionPlaneInput.setByOffset(planarEntity, offset)
        Note: setByPlane requires adsk.core.Plane, but setByOffset accepts BRepFace directly
        """
        face_token = params.get("face_token")

        if not face_token:
            raise PlaneCreationError("face_token is required for mode='face_normal'")

        # Resolve entity token (try custom planes first, then entity tokens)
        if face_token in self.planes:
            face = self.planes[face_token]
        else:
            face = self._resolve_entity_token(face_token, expected_type="BRepFace")

        # Create plane coincident with face using setByOffset with 0 offset
        # setByOffset accepts: "A plane, planar face or construction plane"
        # setByPlane requires: "A transient plane object" (adsk.core.Plane)
        planes = self.root_component.constructionPlanes
        plane_input = planes.createInput()
        zero_offset = adsk.core.ValueInput.createByReal(0.0)
        plane_input.setByOffset(face, zero_offset)

        created_plane = planes.add(plane_input)
        logger.debug(f"Created plane coincident with face '{face_token}'")
        return created_plane

    def _resolve_entity_token(
        self,
        entity_token: str,
        expected_type: Optional[str] = None
    ) -> Any:
        """
        Resolve an entity token to a Fusion 360 object.

        Args:
            entity_token: Entity token string (e.g., "BRepFace-T2F3:1:2")
            expected_type: Expected object type for validation (e.g., "BRepFace")

        Returns:
            Fusion 360 entity object

        Raises:
            PlaneCreationError: If entity not found or wrong type
        """
        try:
            # Use Fusion's find method to resolve the token
            app = adsk.core.Application.get()
            design = adsk.fusion.Design.cast(app.activeProduct)
            if not design:
                raise PlaneCreationError("No active design")

            entity = design.findEntityByToken(entity_token)[0]
            if not entity:
                raise PlaneCreationError(
                    f"Entity not found: {entity_token}. "
                    f"For built-in references, use: 'X'/'Y'/'Z' (axes) or 'XY'/'XZ'/'YZ' (planes). "
                    f"For custom planes, use plane_id from create_construction_plane. "
                    f"For local axes, use 'plane_id:U', 'plane_id:V', or 'plane_id:N'"
                )

            # Validate type if specified
            if expected_type and not _is_type(entity.objectType, expected_type):
                raise PlaneCreationError(
                    f"Entity {entity_token} is {entity.objectType}, expected {expected_type}"
                )

            return entity

        except Exception as exc:
            if isinstance(exc, PlaneCreationError):
                raise
            raise PlaneCreationError(f"Failed to resolve entity '{entity_token}': {exc}") from exc


__all__ = ["PlaneManager", "PlaneCreationError"]
