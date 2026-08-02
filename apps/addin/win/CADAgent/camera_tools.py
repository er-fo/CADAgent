"""
Camera Control and Screenshot Capture Tools for Fusion 360

Provides functions for capturing screenshots from different camera angles
to provide visual context to the LLM. The implementation ensures the user's
view is never disrupted by saving and restoring camera state around captures.

Based on CameraExplorer test add-in implementation.
"""

import adsk.core
import adsk.fusion
import logging
import tempfile
import base64
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def save_camera_state(app: adsk.core.Application) -> Optional[adsk.core.Camera]:
    """
    Save the current camera state for later restoration.

    API Used:
    - Application.activeViewport: Gets the currently active viewport
    - Viewport.camera: Gets the Camera object

    Args:
        app: Fusion 360 Application object

    Returns:
        Camera object representing current state, or None if failed
    """
    try:
        viewport = app.activeViewport
        if not viewport:
            logger.warning("No active viewport found")
            return None

        # Get a copy of the current camera
        camera = viewport.camera
        return camera

    except Exception as e:
        logger.error(f"Failed to save camera state: {e}")
        return None


def restore_camera_state(app: adsk.core.Application, saved_camera: adsk.core.Camera) -> bool:
    """
    Restore a previously saved camera state.

    API Used:
    - Application.activeViewport: Gets the currently active viewport
    - Viewport.camera: Sets the Camera object

    Args:
        app: Fusion 360 Application object
        saved_camera: Previously saved Camera object

    Returns:
        True if successful, False otherwise
    """
    try:
        viewport = app.activeViewport
        if not viewport or not saved_camera:
            logger.warning("Cannot restore camera: missing viewport or saved camera")
            return False

        viewport.camera = saved_camera
        return True

    except Exception as e:
        logger.error(f"Failed to restore camera state: {e}")
        return False


def set_camera_from_coordinates(
    app: adsk.core.Application,
    eye_coords: Tuple[float, float, float],
    target_coords: Tuple[float, float, float],
    up_vector: Tuple[float, float, float] = (0, 0, 1),
    fit_view: bool = True
) -> bool:
    """
    Set camera position using explicit coordinates.

    API Used:
    - Application.activeViewport: Gets the currently active viewport
    - Viewport.camera: Gets/sets the Camera object
    - Camera.eye: Sets camera position
    - Camera.target: Sets look-at point
    - Camera.upVector: Sets up direction
    - adsk.core.Point3D.create(): Creates 3D points
    - adsk.core.Vector3D.create(): Creates 3D vectors

    Args:
        app: Fusion 360 Application object
        eye_coords: Camera position as (x, y, z) in cm
        target_coords: Look-at point as (x, y, z) in cm
        up_vector: Up direction as (x, y, z), default (0, 0, 1) for Z-up
        fit_view: Whether to fit view after setting camera (IMPORTANT: For coordinate-based
                  captures this should be False to preserve exact coordinates)

    Returns:
        True if successful, False otherwise
    """
    try:
        viewport = app.activeViewport
        if not viewport:
            logger.warning("No active viewport found")
            return False

        camera = viewport.camera
        if not camera:
            logger.warning("No camera found")
            return False

        # Create Point3D and Vector3D objects
        eye = adsk.core.Point3D.create(eye_coords[0], eye_coords[1], eye_coords[2])
        target = adsk.core.Point3D.create(target_coords[0], target_coords[1], target_coords[2])
        up = adsk.core.Vector3D.create(up_vector[0], up_vector[1], up_vector[2])

        # Set camera properties
        camera.eye = eye
        camera.target = target
        camera.upVector = up
        # CRITICAL: Do not set isFitView = True for coordinate-based captures
        # as it will retarget the camera and undo the explicit coordinates
        camera.isFitView = False

        # Apply the camera
        viewport.camera = camera

        # CRITICAL: Do not call viewport.fit() for coordinate-based captures
        # as it will change the eye/target to fit model bounds, undoing explicit positioning
        # Only fit if explicitly requested (not used for screenshot capture)
        if fit_view:
            viewport.fit()

        logger.info(f"Set camera from coordinates: eye={eye_coords}, target={target_coords}")
        return True

    except Exception as e:
        logger.error(f"Failed to set camera from coordinates: {e}")
        return False


def capture_screenshot_internal(
    app: adsk.core.Application,
    width: int = 1920,
    height: int = 1080
) -> Optional[Dict[str, Any]]:
    """
    Capture a screenshot of the current viewport.

    API Used:
    - Application.activeViewport: Gets the currently active viewport
    - Viewport.saveAsImageFile(filename, width, height): Saves viewport as image

    Args:
        app: Fusion 360 Application object
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Dictionary containing base64 encoded image and metadata, or None on failure
    """
    try:
        viewport = app.activeViewport
        if not viewport:
            logger.warning("No active viewport found")
            return None

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_file_path = Path(temp_file.name)
        temp_file.close()

        # Save viewport as image
        success = viewport.saveAsImageFile(str(tmp_file_path), width, height)

        if not success:
            logger.error("Failed to save viewport as image")
            return None

        # Read image and encode as base64
        image_bytes = tmp_file_path.read_bytes()
        encoded_image = base64.b64encode(image_bytes).decode("ascii")

        # Clean up temp file
        try:
            tmp_file_path.unlink()
        except:
            pass

        # Get current camera info
        camera_info = get_camera_info(app)

        result = {
            "image_base64": encoded_image,
            "width": width,
            "height": height,
            "camera_info": camera_info,
            "format": "png"
        }

        logger.info(f"Captured screenshot: {width}x{height}")
        return result

    except Exception as e:
        logger.error(f"Failed to capture screenshot: {e}")
        return None


def get_camera_info(app: adsk.core.Application) -> Dict[str, Any]:
    """
    Extract current camera position and orientation information.

    API Used:
    - Application.activeViewport: Gets the currently active viewport
    - Viewport.camera: Gets the Camera object
    - Camera.eye: Point3D representing camera position
    - Camera.target: Point3D representing what camera is looking at
    - Camera.upVector: Vector3D representing the up direction

    Args:
        app: Fusion 360 Application object

    Returns:
        Dictionary containing camera information
    """
    try:
        viewport = app.activeViewport
        if not viewport:
            logger.warning("No active viewport found")
            return {"error": "No active viewport"}

        camera = viewport.camera
        if not camera:
            logger.warning("No camera found in viewport")
            return {"error": "No camera in viewport"}

        # Extract camera position (eye)
        eye = camera.eye
        eye_coords = [
            round(eye.x, 2),
            round(eye.y, 2),
            round(eye.z, 2)
        ]

        # Extract camera target (what it's looking at)
        target = camera.target
        target_coords = [
            round(target.x, 2),
            round(target.y, 2),
            round(target.z, 2)
        ]

        # Extract up vector
        up = camera.upVector
        up_coords = [
            round(up.x, 2),
            round(up.y, 2),
            round(up.z, 2)
        ]

        camera_info = {
            "eye": eye_coords,
            "target": target_coords,
            "up_vector": up_coords
        }

        return camera_info

    except Exception as e:
        logger.error(f"Failed to get camera info: {e}")
        return {"error": str(e)}


def capture_screenshot(
    eye_x: float,
    eye_y: float,
    eye_z: float,
    target_x: float,
    target_y: float,
    target_z: float,
    width: int = 1280,
    height: int = 720,
    description: str = ""
) -> Dict[str, Any]:
    """
    Capture a screenshot from a specific camera position without disrupting the user's view.

    This function:
    1. Saves the current camera state
    2. Moves camera to specified coordinates
    3. Captures screenshot
    4. Restores original camera state

    Fusion 360 Coordinate System (Right-handed):
    - X-axis: Red (right)
    - Y-axis: Green (forward)
    - Z-axis: Blue (up)

    Common Viewpoint Examples:
    - Front view: eye=(0, -50, 0), target=(0, 0, 0)
    - Top view: eye=(0, 0, 50), target=(0, 0, 0)
    - Right view: eye=(50, 0, 0), target=(0, 0, 0)
    - Isometric: eye=(30, -30, 25), target=(0, 0, 0)

    Args:
        eye_x: Camera X position in cm
        eye_y: Camera Y position in cm
        eye_z: Camera Z position in cm
        target_x: Look-at point X in cm
        target_y: Look-at point Y in cm
        target_z: Look-at point Z in cm
        width: Image width in pixels (default 1280)
        height: Image height in pixels (default 720)
        description: Brief explanation of what you're capturing (required for tool use)

    Returns:
        Dictionary with:
        - success: bool indicating if capture succeeded
        - image_base64: Base64 encoded PNG image (if successful)
        - width: Image width
        - height: Image height
        - camera_info: Camera position info at capture time
        - error: Error message (if failed)
    """
    try:
        # Get Fusion application
        app = adsk.core.Application.get()

        logger.info(f"Capturing screenshot: eye=({eye_x}, {eye_y}, {eye_z}), target=({target_x}, {target_y}, {target_z})")

        # Save current camera state
        saved_camera = save_camera_state(app)
        if not saved_camera:
            return {
                "success": False,
                "error": "Failed to save current camera state"
            }

        # Move camera to specified coordinates
        eye_coords = (eye_x, eye_y, eye_z)
        target_coords = (target_x, target_y, target_z)
        up_vector = (0, 0, 1)  # Z-up for standard orientation

        # CRITICAL: fit_view=False to preserve exact coordinates
        success = set_camera_from_coordinates(app, eye_coords, target_coords, up_vector, fit_view=False)
        if not success:
            restore_camera_state(app, saved_camera)
            return {
                "success": False,
                "error": "Failed to set camera to target coordinates"
            }

        # Small delay to allow viewport to update
        time.sleep(0.15)

        # Capture screenshot
        result = capture_screenshot_internal(app, width, height)

        # Always restore original camera state
        restore_camera_state(app, saved_camera)

        if result:
            return {
                "success": True,
                "image_base64": result["image_base64"],
                "width": result["width"],
                "height": result["height"],
                "camera_info": result["camera_info"],
                "format": "png"
            }
        else:
            return {
                "success": False,
                "error": "Failed to capture screenshot from viewport"
            }

    except Exception as e:
        logger.error(f"Exception in capture_screenshot: {e}")
        return {
            "success": False,
            "error": f"Exception: {str(e)}"
        }
