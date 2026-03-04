"""
General utility functions for Fusion 360 add-in development.

Provides common helper functions for design access, error handling,
and logging.
"""

import adsk.core
import adsk.fusion
import logging
import traceback

logger = logging.getLogger(__name__)


def get_active_design():
    """
    Get the currently active Fusion 360 design.

    Returns:
        adsk.fusion.Design or None: The active design, or None if no design is active

    Raises:
        RuntimeError: If Fusion 360 application cannot be accessed
    """
    try:
        app = adsk.core.Application.get()
        if not app:
            raise RuntimeError("Failed to get Fusion 360 application")

        design = adsk.fusion.Design.cast(app.activeProduct)
        return design
    except Exception as e:
        logger.error(f"Failed to get active design: {str(e)}")
        raise


def get_root_component():
    """
    Get the root component of the active design.

    Returns:
        adsk.fusion.Component or None: The root component

    Raises:
        RuntimeError: If no active design exists
    """
    design = get_active_design()
    if not design:
        raise RuntimeError("No active design found")

    return design.rootComponent


def log_error(message, exception=None):
    """
    Log an error message with optional exception details.

    Args:
        message (str): The error message
        exception (Exception, optional): The exception to log
    """
    if exception:
        logger.error(f"{message}: {str(exception)}")
        logger.debug(traceback.format_exc())
    else:
        logger.error(message)


def format_exception(exception):
    """
    Format an exception with its traceback for display.

    Args:
        exception (Exception): The exception to format

    Returns:
        str: Formatted exception string
    """
    return f"{type(exception).__name__}: {str(exception)}\n{traceback.format_exc()}"


def show_message_box(title, message, icon=adsk.core.MessageBoxIconTypes.InformationIconType):
    """
    Display a message box to the user.

    Args:
        title (str): Dialog title
        message (str): Dialog message
        icon: Message box icon type
    """
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        ui.messageBox(message, title, adsk.core.MessageBoxButtonTypes.OKButtonType, icon)
    except Exception as e:
        logger.error(f"Failed to show message box: {str(e)}")


def get_text_commands():
    """
    Get access to Fusion's text commands for script execution.

    Returns:
        adsk.core.TextCommands: The text commands interface
    """
    try:
        app = adsk.core.Application.get()
        return app.userInterface.commandDefinitions.itemById("TextCommands")
    except Exception as e:
        logger.error(f"Failed to get text commands: {str(e)}")
        return None
