"""Fusion import helpers exposed to backend-generated code."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import adsk.core
import adsk.fusion

logger = logging.getLogger(__name__)

SUPPORTED_STEP_EXTENSIONS = {".step", ".stp"}


class ImportOperationError(RuntimeError):
    """Raised when a CAD file import cannot be completed."""


def import_step_file(
    app: adsk.core.Application,
    file_path: str,
    *,
    component_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Import a STEP/STP file into the active design root component."""
    path = Path(str(file_path or "")).expanduser()
    if not path.is_file():
        raise ImportOperationError(f"STEP file not found: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_STEP_EXTENSIONS:
        raise ImportOperationError("Only .step and .stp files can be imported.")

    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise ImportOperationError("Active Fusion product is not a design.")

    root_comp = design.rootComponent
    import_manager = app.importManager
    if not import_manager:
        raise ImportOperationError("Fusion import manager is unavailable.")

    before_body_count = root_comp.bRepBodies.count
    before_occurrence_count = root_comp.occurrences.count

    options = import_manager.createSTEPImportOptions(str(path))
    success = import_manager.importToTarget(options, root_comp)
    if not success:
        raise ImportOperationError(f"Fusion failed to import STEP file: {path.name}")

    imported_body_count = max(0, root_comp.bRepBodies.count - before_body_count)
    imported_occurrence_count = max(0, root_comp.occurrences.count - before_occurrence_count)

    if component_name:
        _rename_newest_occurrence_or_root_body(root_comp, component_name)

    logger.info(
        "Imported STEP file %s (%d new bodies, %d new occurrences)",
        path,
        imported_body_count,
        imported_occurrence_count,
    )

    return {
        "success": True,
        "file_name": path.name,
        "file_path": str(path),
        "imported_body_count": imported_body_count,
        "imported_occurrence_count": imported_occurrence_count,
        "message": f"Imported {path.name} into the active design.",
    }


def _rename_newest_occurrence_or_root_body(root_comp: adsk.fusion.Component, name: str) -> None:
    safe_name = str(name or "").strip()
    if not safe_name:
        return

    try:
        if root_comp.occurrences.count > 0:
            root_comp.occurrences.item(root_comp.occurrences.count - 1).name = safe_name
            return
    except Exception as exc:
        logger.debug("Unable to rename imported occurrence: %s", exc)

    try:
        if root_comp.bRepBodies.count > 0:
            root_comp.bRepBodies.item(root_comp.bRepBodies.count - 1).name = safe_name
    except Exception as exc:
        logger.debug("Unable to rename imported body: %s", exc)
