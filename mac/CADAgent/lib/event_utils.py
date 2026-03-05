"""
Event handling utilities for Fusion 360 add-ins.

Provides helper functions for managing event handlers and ensuring they
remain in scope throughout the add-in's lifecycle.
"""

import adsk.core
import logging

logger = logging.getLogger(__name__)


def add_handler(event, callback, handlers_list, event_name=""):
    """
    Add an event handler and keep it alive in the handlers list.

    Args:
        event: The Fusion 360 event object to attach the handler to
        callback: The event handler class instance
        handlers_list: List to store the handler reference
        event_name: Optional name for logging purposes

    Returns:
        The handler that was added
    """
    try:
        event.add(callback)
        handlers_list.append(callback)
        if event_name:
            logger.debug(f"Added handler for event: {event_name}")
        return callback
    except Exception as e:
        logger.error(f"Failed to add handler for {event_name}: {str(e)}")
        raise


def remove_handler(event, callback, handlers_list, event_name=""):
    """
    Remove an event handler and clean it from the handlers list.

    Args:
        event: The Fusion 360 event object
        callback: The event handler to remove
        handlers_list: List containing the handler reference
        event_name: Optional name for logging purposes
    """
    try:
        if callback in handlers_list:
            event.remove(callback)
            handlers_list.remove(callback)
            if event_name:
                logger.debug(f"Removed handler for event: {event_name}")
    except Exception as e:
        logger.error(f"Failed to remove handler for {event_name}: {str(e)}")


def clear_handlers(event, handlers_list, event_name=""):
    """
    Remove all handlers from an event and clear the handlers list.

    Args:
        event: The Fusion 360 event object
        handlers_list: List containing handler references
        event_name: Optional name for logging purposes
    """
    try:
        for handler in list(handlers_list):
            try:
                event.remove(handler)
            except:
                pass
        handlers_list.clear()
        if event_name:
            logger.debug(f"Cleared all handlers for event: {event_name}")
    except Exception as e:
        logger.error(f"Failed to clear handlers for {event_name}: {str(e)}")
