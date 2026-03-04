"""
Configuration settings for the CADAgent Fusion 360 add-in.

This module centralizes all configuration values and global variables used
throughout the add-in, making it easy to customize behavior without modifying
core code.
"""

import os
from pathlib import Path

# Load .env.cadagent file before reading any config values
def _load_env_file(path: Path) -> None:
    """Load environment variables from .env.cadagent file."""
    if not path.exists():
        return
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass  # Silently fail if file can't be read

# Load environment from .env.cadagent file
_env_file_path = Path(__file__).resolve().parent / ".env.cadagent"
_load_env_file(_env_file_path)

# Supabase publishable defaults (safe to embed – these are public/anon keys)
_SUPABASE_URL_DEFAULT = "https://wpgibucctvusoizwhsbz.supabase.co"
_SUPABASE_KEY_DEFAULT = "sb_publishable_VkJwtJWbZn8DwovnrKXIew_W0mr9ix_"

# Ensure Supabase env vars are set even when .env.cadagent is missing (e.g. dotfile
# not copied on Windows or hidden-file-unaware extractors).
if "SUPABASE_URL" not in os.environ:
    os.environ["SUPABASE_URL"] = _SUPABASE_URL_DEFAULT
if "SUPABASE_PUBLISHABLE_KEY" not in os.environ:
    os.environ["SUPABASE_PUBLISHABLE_KEY"] = _SUPABASE_KEY_DEFAULT

# Debug mode - enable verbose logging
DEBUG = os.environ.get("CADAGENT_DEBUG", "False").lower() == "true"

# Testing mode (enables password login for allowlisted email)
TESTING_MODE = os.environ.get("CADAGENT_TESTING_MODE", "False").lower() == "true"
TEST_EMAIL = os.environ.get("CADAGENT_TEST_EMAIL", "").strip().lower()
TEST_PASSWORD = os.environ.get("CADAGENT_TEST_PASSWORD", "")

# Company and add-in identification
COMPANY_NAME = "CADAgent"
ADDIN_NAME = "CADAgent"

# WebSocket backend configuration
# Defaults point to the production backend WebSocket endpoint.
# Override via env vars for local/dev: BACKEND_HOST=localhost BACKEND_PORT=8000 BACKEND_USE_SSL=false
BACKEND_HOST = os.environ.get("BACKEND_HOST", "ws.cadagentpro.com")
# Leave BACKEND_PORT blank by default so standard ports (443 for wss, 80 for ws) are used.
BACKEND_PORT = os.environ.get("BACKEND_PORT", "")
BACKEND_USE_SSL = os.environ.get("BACKEND_USE_SSL", "true").lower() == "true"
# Optional full URL override, e.g. BACKEND_URL="ws://localhost:8000/ws/{session_id}"
BACKEND_URL = os.environ.get("BACKEND_URL")


def build_ws_url(session_id: str) -> str:
    """Return the full WebSocket URL for a given session id."""
    if BACKEND_URL:
        # Allow templated or base URL
        if "{session_id}" in BACKEND_URL:
            return BACKEND_URL.format(session_id=session_id)
        base = BACKEND_URL.rstrip("/")
        return f"{base}/{session_id}"

    scheme = "wss" if BACKEND_USE_SSL else "ws"
    # Only include port if explicitly provided and non-default
    port = ""
    if BACKEND_PORT:
        if not ((scheme == "wss" and BACKEND_PORT in ("443", "")) or (scheme == "ws" and BACKEND_PORT == "80")):
            port = f":{BACKEND_PORT}"

    return f"{scheme}://{BACKEND_HOST}{port}/ws/{session_id}"


def backend_label() -> str:
    """Human-friendly label for palette/status logging."""
    try:
        return build_ws_url("<session>")
    except Exception:
        scheme = "wss" if BACKEND_USE_SSL else "ws"
        port = f":{BACKEND_PORT}" if BACKEND_PORT else ""
        return f"{scheme}://{BACKEND_HOST}{port}"

# UI Configuration
WORKSPACE_ID = "FusionSolidEnvironment"
PANEL_ID = "SolidScriptsAddinsPanel"

# Command identifiers (namespaced to avoid conflicts)
COMMAND_ID_EXECUTE = f"{COMPANY_NAME}_{ADDIN_NAME}_ExecuteCommand"
COMMAND_ID_SETTINGS = f"{COMPANY_NAME}_{ADDIN_NAME}_SettingsCommand"
COMMAND_ID_STATUS = f"{COMPANY_NAME}_{ADDIN_NAME}_StatusCommand"

# Custom event identifiers
EVENT_ID_INBOUND_MESSAGE = f"{COMPANY_NAME}_{ADDIN_NAME}_InboundMessage"

# Logging configuration
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"

# ═══════════════════════════════════════════════════════════════════════════════
# UI Theme Configuration - Fusion 360 Dark/Light Mode
# ═══════════════════════════════════════════════════════════════════════════════
# Enable/disable dark mode (matches Fusion 360's theme options)
FUSION_DARK_MODE = os.environ.get("FUSION_DARK_MODE", "true").lower() == "true"

# Fusion 360 theme colors (official dark greyish-blue and light grey)
# Dark mode: #3D4752 (RGB: 61, 71, 82) - the new Fusion 360 default dark theme
# Light mode: #E8E8E8 (RGB: 232, 232, 232) - Fusion 360 light/bright theme
FUSION_DARK_BG_COLOR = os.environ.get("FUSION_DARK_BG_COLOR", "#3D4752")
FUSION_LIGHT_BG_COLOR = os.environ.get("FUSION_LIGHT_BG_COLOR", "#E8E8E8")


def get_background_color() -> str:
    """
    Get the current background color based on theme setting.
    
    Returns:
        Hex color string (e.g., "#3D4752" for dark mode)
    """
    return FUSION_DARK_BG_COLOR if FUSION_DARK_MODE else FUSION_LIGHT_BG_COLOR


def get_background_rgb() -> tuple:
    """
    Get the current background color as RGB tuple (0-255).
    
    Returns:
        Tuple of (R, G, B) values
    """
    hex_color = get_background_color().lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_background_rgb_normalized() -> tuple:
    """
    Get the current background color as normalized RGB tuple (0.0-1.0).
    
    Returns:
        Tuple of (R, G, B) values normalized to 0.0-1.0 range
    """
    r, g, b = get_background_rgb()
    return (r / 255.0, g / 255.0, b / 255.0)
