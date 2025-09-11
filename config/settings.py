# CADAgent Add-on Configuration
# Following Fusion 360 best practices - keep simple

# Your backend URL
BACKEND_BASE_URL = "https://api.cadagentpro.com"

# Fallback URLs in case of DNS issues in Fusion 360 environment
BACKEND_FALLBACK_URLS = [
    "https://api.cadagentpro.com",  # Primary
]

# Debug mode - set to True to enable verbose logging
DEBUG_MODE = True

# API endpoints (using direct endpoints that accept anthropic_api_key in body)
GENERATE_ENDPOINT = "/api/v1/direct/generate"
ITERATE_ENDPOINT = "/api/v1/direct/iterate/{model_id}"
PARAMETERS_PUT_ENDPOINT = "/api/v1/parameters/{model_id}"
PARAMETERS_GET_ENDPOINT = "/api/v1/parameters/{model_id}"
ONBOARD_ENDPOINT = "/api/v1/status"  # Use status endpoint for health checks

# UI settings for palette - optimized for right-side docking
CHAT_WINDOW_WIDTH = 400  # Wider for better usability when docked to right
CHAT_WINDOW_HEIGHT = 600  # Taller to better utilize vertical space

# File storage settings
SESSION_FOLDER_PREFIX = "CADAgent_Sessions"

# Backend authentication configuration
# For personal use and development, the backend supports a test bypass token.
# Use "test" to match backend auth middleware's Bearer test bypass.
BACKEND_AUTH_TOKEN = "test"

# Users must configure their API key through the UI
# The chat interface provides API key configuration in the dropdown
DEFAULT_ANTHROPIC_API_KEY = None
