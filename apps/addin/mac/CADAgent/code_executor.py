"""
Execute Python snippets received from the backend inside the Fusion 360 context.

The executor provides a scoped namespace with key Fusion objects and keeps a registry
of created sketches so subsequent operations can reference them directly by ID.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, Optional

import adsk.core
import adsk.fusion

from . import camera_tools
from . import plane_manager as plane_manager_module

logger = logging.getLogger(__name__)


class CodeExecutor:
    """Executes backend-supplied Python code scoped per Fusion document."""

    def __init__(self, app: adsk.core.Application):
        self._app = app
        # doc_id -> dict(name->Sketch) registry
        self._sketch_registries: Dict[str, Dict[str, adsk.fusion.Sketch]] = {}
        # doc_id -> PlaneManager registry
        self._plane_managers: Dict[str, plane_manager_module.PlaneManager] = {}

    def reset_context(self, doc_id: str) -> None:
        """Clear cached sketch registry and plane manager for a document (e.g., when it closes)."""
        self._sketch_registries.pop(doc_id, None)
        self._plane_managers.pop(doc_id, None)

    def execute_code(
        self,
        doc_id: str,
        design: Optional[adsk.fusion.Design],
        code: str,
        operation: str
    ) -> Dict[str, Any]:
        """Execute the provided code snippet for the specified design."""
        if not design:
            logger.error("Cannot execute code for doc %s: design reference missing", doc_id)
            return {
                "success": False,
                "operation": operation,
                "error": "Design context unavailable for this document."
            }

        root_comp = design.rootComponent
        extrudes = root_comp.features.extrudeFeatures

        sketch_registry = self._sketch_registries.setdefault(doc_id, {})

        # Get or create plane manager for this document (persists across execution batches)
        if doc_id not in self._plane_managers:
            self._plane_managers[doc_id] = plane_manager_module.PlaneManager(root_comp)
        _plane_manager = self._plane_managers[doc_id]

        exec_globals: Dict[str, Any] = {
            "adsk": adsk,
            "app": self._app,
            "design": design,
            "rootComp": root_comp,
            "extrudes": extrudes,
            "camera_tools": camera_tools,
            "plane_manager": _plane_manager,
        }
        exec_globals.update(sketch_registry)

        try:
            exec(code, exec_globals)  # noqa: S102 - intentional dynamic execution
            # Capture new sketches created during execution.
            new_sketches = []
            for name, value in exec_globals.items():
                if isinstance(value, adsk.fusion.Sketch):
                    if name not in sketch_registry:
                        new_sketches.append(name)
                    sketch_registry[name] = value

            # Check if code set _result (e.g., from camera_tools.capture_screenshot)
            result_data = exec_globals.get("_result")
            
            # DEBUG: Log _result capture
            logger.info("=== DEBUG code_executor ===")
            logger.info("Operation: %s", operation)
            logger.info("_result in exec_globals: %s", "_result" in exec_globals)
            logger.info("result_data value: %s", result_data)
            
            if result_data and isinstance(result_data, dict):
                logger.info("_result is a dict with keys: %s", list(result_data.keys()))
                # Return the result data directly if it contains success info
                if "success" in result_data:
                    result_data["operation"] = operation
                    if new_sketches:
                        result_data["created_sketches"] = new_sketches
                    logger.info("Returning result_data directly: %s", result_data)
                    return result_data
                # Otherwise merge with standard response
                result = {
                    "success": True,
                    "operation": operation,
                    "message": f"{operation} executed successfully.",
                    **result_data
                }
                if new_sketches:
                    result["created_sketches"] = new_sketches
                logger.info("Returning merged result: %s", result)
                return result
            else:
                logger.info("No _result dict found, using standard response")

            result = {
                "success": True,
                "operation": operation,
                "message": f"{operation} executed successfully.",
            }
            if new_sketches:
                result["created_sketches"] = new_sketches
            return result
        except Exception as exc:  # pragma: no cover - Fusion runtime errors are handled at runtime
            # Capture full traceback for debugging context
            tb = traceback.format_exc()
            logger.exception("Failed to execute Fusion code for operation %s", operation)
            return {
                "success": False,
                "operation": operation,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": tb,
            }
