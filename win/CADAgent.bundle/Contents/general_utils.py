"""
Lightweight helper utilities used by the CADAgent Fusion 360 add-in.

Only the minimal surface required by CADAgent.py is provided here to avoid
startup failures when the original helper module was removed.
"""

from __future__ import annotations

import logging
import traceback
from typing import Optional

import adsk.core


def show_message_box(
    title: str,
    message: str,
    icon: adsk.core.MessageBoxIconTypes = adsk.core.MessageBoxIconTypes.InformationIconType,
) -> None:
    """
    Display a message box if the Fusion UI is available.

    This helper is intentionally defensive to avoid raising during startup
    when the application or UI may not yet be initialized.
    """
    try:
        app = adsk.core.Application.get()
        ui: Optional[adsk.core.UserInterface] = app.userInterface if app else None
        if ui:
            ui.messageBox(message, title, icon)
    except Exception as exc:  # noqa: BLE001 - best-effort logging only
        logging.getLogger(__name__).debug("Failed to show message box: %s", exc, exc_info=True)


def log_error(context: str, exc: Exception) -> None:
    """Log an exception with stack trace context."""
    logger = logging.getLogger(__name__)
    logger.error("%s: %s", context, exc, exc_info=True)


def format_exception(exc: Exception) -> str:
    """Return a human-readable stack trace string for display."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
