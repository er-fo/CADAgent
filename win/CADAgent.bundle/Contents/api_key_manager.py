"""
Local API Key Manager for CADAgent BYOK (Bring Your Own Key).

This module handles secure local storage and retrieval of user's API keys
for LLM providers (Anthropic, OpenAI, Google). Keys are stored in the user's
home directory at ~/.cadagent/api_keys.json.

Security considerations:
- Keys are stored in a file with restrictive permissions (0600)
- Keys are only stored locally, never sent to our servers (only to LLM providers via backend)
- The backend uses these keys directly for API calls
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class APIKeyManager:
    """
    Manages local storage and retrieval of user API keys.
    
    Keys are stored at ~/.cadagent/api_keys.json with the following structure:
    {
        "anthropic_api_key": "sk-ant-...",
        "openai_api_key": "sk-proj-...",
        "google_api_key": "AIza..."
    }
    """
    
    # Supported providers and their key prefixes for validation
    PROVIDER_PREFIXES = {
        "anthropic_api_key": ["sk-ant-"],
        "openai_api_key": ["sk-proj-", "sk-"],
        "google_api_key": ["AIza"],
    }
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the API key manager.
        
        Args:
            config_dir: Override the default config directory (for testing)
        """
        if config_dir is None:
            self.config_dir = Path.home() / ".cadagent"
        else:
            self.config_dir = config_dir
            
        self.keys_file = self.config_dir / "api_keys.json"
        self._ensure_config_dir()
        
        logger.info(f"APIKeyManager initialized with config dir: {self.config_dir}")
    
    def _ensure_config_dir(self) -> None:
        """Ensure the config directory exists with proper permissions."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            # Set restrictive permissions on config directory (owner only)
            if os.name != 'nt':  # Unix-like systems
                os.chmod(self.config_dir, 0o700)
        except Exception as e:
            logger.error(f"Failed to create config directory: {e}")
            raise
    
    def _set_file_permissions(self, file_path: Path) -> None:
        """Set restrictive permissions on a file (owner read/write only)."""
        try:
            if os.name != 'nt':  # Unix-like systems
                os.chmod(file_path, 0o600)
        except Exception as e:
            logger.warning(f"Failed to set file permissions: {e}")
    
    def get_all_keys(self) -> Dict[str, str]:
        """
        Get all stored API keys.
        
        Returns:
            Dictionary mapping key names to values
        """
        try:
            if not self.keys_file.exists():
                logger.debug("No API keys file found")
                return {}
            
            with open(self.keys_file, 'r') as f:
                data = json.load(f)
            
            # Only return recognized key fields
            keys = {}
            for key_name in self.PROVIDER_PREFIXES.keys():
                if key_name in data and data[key_name]:
                    keys[key_name] = data[key_name]
            
            logger.debug(f"Loaded {len(keys)} API key(s)")
            return keys
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API keys file: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            return {}
    
    def get_key(self, key_name: str) -> Optional[str]:
        """
        Get a specific API key.
        
        Args:
            key_name: Name of the key (e.g., 'anthropic_api_key')
            
        Returns:
            The API key value or None if not found
        """
        keys = self.get_all_keys()
        return keys.get(key_name)
    
    def set_key(self, key_name: str, key_value: str) -> bool:
        """
        Store an API key.
        
        Args:
            key_name: Name of the key (e.g., 'anthropic_api_key')
            key_value: The API key value
            
        Returns:
            True if successful, False otherwise
        """
        if key_name not in self.PROVIDER_PREFIXES:
            logger.error(f"Unknown key name: {key_name}")
            return False
        
        try:
            # Load existing keys
            keys = self.get_all_keys()
            
            # Update the key
            if key_value:
                keys[key_name] = key_value.strip()
            elif key_name in keys:
                del keys[key_name]
            
            # Save to file
            with open(self.keys_file, 'w') as f:
                json.dump(keys, f, indent=2)
            
            # Set restrictive permissions
            self._set_file_permissions(self.keys_file)
            
            logger.info(f"Saved API key: {key_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save API key: {e}")
            return False
    
    def set_keys(self, keys: Dict[str, str]) -> bool:
        """
        Store multiple API keys at once.
        
        Args:
            keys: Dictionary mapping key names to values
            
        Returns:
            True if all keys were saved successfully
        """
        success = True
        for key_name, key_value in keys.items():
            if not self.set_key(key_name, key_value):
                success = False
        return success
    
    def delete_key(self, key_name: str) -> bool:
        """
        Delete an API key.
        
        Args:
            key_name: Name of the key to delete
            
        Returns:
            True if successful
        """
        return self.set_key(key_name, "")
    
    def delete_all_keys(self) -> bool:
        """
        Delete all stored API keys.
        
        Returns:
            True if successful
        """
        try:
            if self.keys_file.exists():
                self.keys_file.unlink()
            logger.info("Deleted all API keys")
            return True
        except Exception as e:
            logger.error(f"Failed to delete API keys: {e}")
            return False
    
    def validate_key(self, key_name: str, key_value: str) -> tuple[bool, str]:
        """
        Validate an API key format (not connectivity).
        
        Args:
            key_name: Name of the key
            key_value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if key_name not in self.PROVIDER_PREFIXES:
            return False, f"Unknown key type: {key_name}"
        
        if not key_value or not key_value.strip():
            return False, "API key cannot be empty"
        
        key_value = key_value.strip()
        
        # Check prefix
        prefixes = self.PROVIDER_PREFIXES[key_name]
        if not any(key_value.startswith(p) for p in prefixes):
            expected = " or ".join(prefixes)
            return False, f"Invalid key format. Expected to start with: {expected}"
        
        # Check minimum length
        if len(key_value) < 20:
            return False, "API key appears too short"
        
        return True, ""
    
    def has_required_keys(self) -> tuple[bool, list[str]]:
        """
        Check if user has at least one LLM API key configured.
        
        Returns:
            Tuple of (has_keys, missing_keys)
        """
        keys = self.get_all_keys()
        
        # User needs at least Anthropic OR OpenAI key
        has_anthropic = bool(keys.get("anthropic_api_key"))
        has_openai = bool(keys.get("openai_api_key"))
        
        if has_anthropic or has_openai:
            return True, []
        
        missing = []
        if not has_anthropic:
            missing.append("anthropic_api_key")
        if not has_openai:
            missing.append("openai_api_key")
        
        return False, missing
    
    def get_keys_for_backend(self) -> Dict[str, str]:
        """
        Get API keys formatted for sending to the backend.
        
        Returns:
            Dictionary with keys ready for WebSocket auth message
        """
        keys = self.get_all_keys()
        return {
            "anthropic_api_key": keys.get("anthropic_api_key", ""),
            "openai_api_key": keys.get("openai_api_key", ""),
            "google_api_key": keys.get("google_api_key", ""),
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of all API keys (configured or not, masked + raw values).

        Returns:
            Status dictionary for UI display. Includes `value` so the palette
            can prefill inputs and reveal the stored key when the user toggles
            visibility, while still using `masked` for passive display.
        """
        keys = self.get_all_keys()

        def mask_key(key: str) -> str:
            """Mask API key for display (show first 8 and last 4 chars)."""
            if not key or len(key) < 16:
                return "***"
            return f"{key[:8]}...{key[-4:]}"

        return {
            "anthropic": {
                "configured": bool(keys.get("anthropic_api_key")),
                "masked": mask_key(keys.get("anthropic_api_key", "")) if keys.get("anthropic_api_key") else None,
                "value": keys.get("anthropic_api_key", ""),
            },
            "openai": {
                "configured": bool(keys.get("openai_api_key")),
                "masked": mask_key(keys.get("openai_api_key", "")) if keys.get("openai_api_key") else None,
                "value": keys.get("openai_api_key", ""),
            },
            "google": {
                "configured": bool(keys.get("google_api_key")),
                "masked": mask_key(keys.get("google_api_key", "")) if keys.get("google_api_key") else None,
                "value": keys.get("google_api_key", ""),
            },
        }


# Module-level singleton instance
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get the singleton APIKeyManager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager
