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

SUPPORT_CONTACT_LINE = "If issue persists, email erik@cadagent.co"


def _append_support_contact(message: str) -> str:
    if not message:
        return SUPPORT_CONTACT_LINE
    if SUPPORT_CONTACT_LINE.lower() in message.lower():
        return message
    separator = "\n\n" if "\n" in message else " "
    return f"{message}{separator}{SUPPORT_CONTACT_LINE}"


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
            text = message
            if icon in (
                adsk.core.MessageBoxIconTypes.WarningIconType,
                adsk.core.MessageBoxIconTypes.CriticalIconType,
            ):
                text = _append_support_contact(message)
            ui.messageBox(text, title, icon)
    except Exception as exc:  # noqa: BLE001 - best-effort logging only
        logging.getLogger(__name__).debug("Failed to show message box: %s", exc, exc_info=True)


def log_error(context: str, exc: Exception) -> None:
    """Log an exception with stack trace context."""
    logger = logging.getLogger(__name__)
    logger.error("%s: %s", context, exc, exc_info=True)


def format_exception(exc: Exception) -> str:
    """Return a human-readable stack trace string for display."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
