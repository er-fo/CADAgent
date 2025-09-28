"""
CADAgent Add-on API Client
Handles communication with the Fly.io backend following best practices.
Pure Python implementation for compatibility.
"""

import json
import urllib.request
import urllib.parse
import urllib.error
import socket
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime

# Import settings from config with a robust strategy
try:
    # Prefer absolute package import if available (PEP 420 namespace package)
    from Space.config import settings as _settings
    # Force reload to get latest settings
    import importlib
    importlib.reload(_settings)
except Exception:
    # Fallback to path-based import for Fusion runtime contexts
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
    if config_path not in sys.path:
        sys.path.append(config_path)
    import importlib
    _settings = importlib.import_module('settings')
    # Force reload to get latest settings
    importlib.reload(_settings)

# Bind required settings symbols locally
BACKEND_BASE_URL = getattr(_settings, 'BACKEND_BASE_URL')
BACKEND_FALLBACK_URLS = getattr(_settings, 'BACKEND_FALLBACK_URLS', [BACKEND_BASE_URL])
DEBUG_MODE = getattr(_settings, 'DEBUG_MODE', False)
GENERATE_ENDPOINT = getattr(_settings, 'GENERATE_ENDPOINT')
ITERATE_ENDPOINT = getattr(_settings, 'ITERATE_ENDPOINT')
PARAMETERS_PUT_ENDPOINT = getattr(_settings, 'PARAMETERS_PUT_ENDPOINT')
PARAMETERS_GET_ENDPOINT = getattr(_settings, 'PARAMETERS_GET_ENDPOINT')
BACKEND_AUTH_TOKEN = getattr(_settings, 'BACKEND_AUTH_TOKEN', None)

class SpaceAPIClient:
    """
    Simple HTTP client for communicating with the Fly.io backend.
    Uses only standard library to avoid dependencies.
    """

    @staticmethod
    def _write_request_debug(method, url, headers, data_dict, data_bytes):
        """Append sanitized request details to a debug log for diagnostics."""
        try:
            log_path = Path(__file__).with_name('request_debug.log')
            content_length = len(data_bytes) if data_bytes else 0

            # Sanitize headers to avoid leaking secrets
            safe_headers = {}
            if headers:
                for key, value in headers.items():
                    if value is None:
                        safe_headers[key] = value
                        continue
                    lower_key = key.lower()
                    if lower_key in ('authorization', 'x-api-key', 'idempotency-key'):
                        value_str = str(value)
                        masked = f"{value_str[:6]}...len={len(value_str)}"
                        safe_headers[key] = masked
                    else:
                        safe_headers[key] = value

            # Sanitize data dictionary similarly
            safe_body = data_dict
            if isinstance(data_dict, dict):
                safe_body = {}
                for key, value in data_dict.items():
                    if value is None:
                        safe_body[key] = value
                        continue
                    if isinstance(value, str) and key.lower().endswith('key'):
                        masked = f"{value[:6]}...len={len(value)}"
                        safe_body[key] = masked
                    else:
                        safe_body[key] = value

            log_entry = (
                f"{datetime.utcnow().isoformat()}Z | {method} {url}\n"
                f"Headers: {safe_headers}\n"
                f"Body: {safe_body}\n"
                f"Content-Length: {content_length}\n"
                "----\n"
            )

            with log_path.open('a', encoding='utf-8') as log_file:
                log_file.write(log_entry)
        except Exception:
            # Never let diagnostics cause request failure
            pass
    
    def __init__(self, api_key=None):
        # Get DEFAULT_ANTHROPIC_API_KEY using robust import strategy
        DEFAULT_ANTHROPIC_API_KEY = getattr(_settings, 'DEFAULT_ANTHROPIC_API_KEY', None)

        # Try to auto-retrieve cached API key if none provided and no default set
        retrieved_key = None
        if not api_key and not DEFAULT_ANTHROPIC_API_KEY:
            if DEBUG_MODE:
                try:
                    print("SpaceAPIClient: No API key provided, attempting to retrieve cached key...")
                except Exception:
                    pass
            retrieved_key = self._retrieve_cached_api_key()
            if retrieved_key and DEBUG_MODE:
                try:
                    print(f"SpaceAPIClient: Successfully retrieved cached API key, length={len(retrieved_key)}")
                except Exception:
                    pass
            elif DEBUG_MODE:
                try:
                    print("SpaceAPIClient: No cached API key found")
                except Exception:
                    pass

        self.api_key = api_key or retrieved_key or DEFAULT_ANTHROPIC_API_KEY
        self.base_url = BACKEND_BASE_URL
        self.backend_token = BACKEND_AUTH_TOKEN
        
    def set_api_key(self, api_key):
        """Set the Anthropic API key for requests and optionally cache it."""
        # Clean and validate the API key
        if api_key:
            # Strip whitespace and normalize
            cleaned_key = str(api_key).strip()
            if cleaned_key:
                self.api_key = cleaned_key
                # Try to cache the API key for future use
                self._store_cached_api_key(cleaned_key)
                try:
                    print(f"SpaceAPIClient: API key set, length={len(cleaned_key)}, starts_with={cleaned_key[:10]}...")
                except Exception:
                    pass
            else:
                self.api_key = None
        else:
            self.api_key = None

    def _retrieve_cached_api_key(self):
        """
        Attempt to retrieve cached API key from Fusion 360 design attributes.
        Returns None if not available or if running outside Fusion environment.
        """
        try:
            if DEBUG_MODE:
                try:
                    print("SpaceAPIClient: Starting cached API key retrieval...")
                except Exception:
                    pass

            # Import fusion_utils using robust strategy similar to settings import
            fusion_utils_module = None
            try:
                # Try absolute package import first
                from Space.lib import fusion_utils as fusion_utils_module
                if DEBUG_MODE:
                    try:
                        print("SpaceAPIClient: Successfully imported fusion_utils via absolute path")
                    except Exception:
                        pass
            except Exception as e:
                if DEBUG_MODE:
                    try:
                        print(f"SpaceAPIClient: Absolute import failed: {e}, trying fallback...")
                    except Exception:
                        pass
                # Fallback to path-based import for Fusion runtime contexts
                lib_path = os.path.dirname(__file__)
                if lib_path not in sys.path:
                    sys.path.append(lib_path)
                import fusion_utils as fusion_utils_module
                if DEBUG_MODE:
                    try:
                        print("SpaceAPIClient: Successfully imported fusion_utils via fallback path")
                    except Exception:
                        pass

            # Create utils instance and retrieve key
            fusion_utils = fusion_utils_module.SpaceFusionUtils()
            if DEBUG_MODE:
                try:
                    print("SpaceAPIClient: Created SpaceFusionUtils instance, calling retrieve_api_key()...")
                except Exception:
                    pass

            cached_key = fusion_utils.retrieve_api_key()

            if cached_key:
                if DEBUG_MODE:
                    try:
                        print(f"SpaceAPIClient: Retrieved cached API key, length={len(cached_key)}")
                    except Exception:
                        pass
                return cached_key
            else:
                if DEBUG_MODE:
                    try:
                        print("SpaceAPIClient: fusion_utils.retrieve_api_key() returned None")
                    except Exception:
                        pass
                return None

        except Exception as e:
            # This is expected when running outside Fusion 360 environment
            if DEBUG_MODE:
                try:
                    print(f"SpaceAPIClient: Could not retrieve cached API key: {type(e).__name__}: {e}")
                    import traceback
                    print(f"SpaceAPIClient: Traceback: {traceback.format_exc()}")
                except Exception:
                    pass
            return None

    def _store_cached_api_key(self, api_key):
        """
        Attempt to store API key in Fusion 360 design attributes for future use.
        Silently fails if not in Fusion environment or if storage fails.
        """
        try:
            # Import fusion_utils using robust strategy similar to settings import
            fusion_utils_module = None
            try:
                # Try absolute package import first
                from Space.lib import fusion_utils as fusion_utils_module
            except Exception:
                # Fallback to path-based import for Fusion runtime contexts
                lib_path = os.path.dirname(__file__)
                if lib_path not in sys.path:
                    sys.path.append(lib_path)
                import fusion_utils as fusion_utils_module

            # Create utils instance and store key
            fusion_utils = fusion_utils_module.SpaceFusionUtils()
            success = fusion_utils.store_api_key(api_key)

            if success and DEBUG_MODE:
                try:
                    print(f"SpaceAPIClient: Cached API key for future use")
                except Exception:
                    pass

        except Exception as e:
            # This is expected when running outside Fusion 360 environment
            if DEBUG_MODE:
                try:
                    print(f"SpaceAPIClient: Could not cache API key: {e}")
                except Exception:
                    pass

    def clear_cached_api_key(self):
        """
        Clear any cached API key from Fusion 360 design attributes.
        Useful for security or when switching API keys.
        """
        try:
            # Import fusion_utils using robust strategy similar to settings import
            fusion_utils_module = None
            try:
                # Try absolute package import first
                from Space.lib import fusion_utils as fusion_utils_module
            except Exception:
                # Fallback to path-based import for Fusion runtime contexts
                lib_path = os.path.dirname(__file__)
                if lib_path not in sys.path:
                    sys.path.append(lib_path)
                import fusion_utils as fusion_utils_module

            # Create utils instance and clear key
            fusion_utils = fusion_utils_module.SpaceFusionUtils()
            success = fusion_utils.clear_api_key()

            if success and DEBUG_MODE:
                try:
                    print(f"SpaceAPIClient: Cleared cached API key")
                except Exception:
                    pass

        except Exception as e:
            # This is expected when running outside Fusion 360 environment
            if DEBUG_MODE:
                try:
                    print(f"SpaceAPIClient: Could not clear cached API key: {e}")
                except Exception:
                    pass

    def _make_request(self, method, endpoint, data=None, headers=None):
        """
        Make HTTP request to the backend with fallback URLs.
        Returns parsed JSON response.
        """
        # Try primary URL first, then fallbacks
        urls_to_try = [self.base_url] + [url for url in BACKEND_FALLBACK_URLS if url != self.base_url]
        
        last_error = None
        for i, base_url in enumerate(urls_to_try):
            url = base_url + endpoint
            
            if DEBUG_MODE:
                try:
                    print(f"SpaceAPIClient: Attempt {i+1}/{len(urls_to_try)} - trying {url}")
                except Exception:
                    pass
            
            try:
                result = self._make_single_request(method, url, data, headers)

                # If this attempt failed with a server-side error and fallbacks remain, try the next URL.
                if (
                    isinstance(result, dict)
                    and result.get('success') is False
                    and i < len(urls_to_try) - 1
                ):
                    error_text = str(result.get('error', '')).lower()
                    # Retry on common transient/server-side failures.
                    if any(token in error_text for token in ('http 5', ' 5xx', 'bad gateway', 'service unavailable', 'gateway timeout', '502', '503', '504')):
                        last_error = result
                        if DEBUG_MODE:
                            try:
                                print(f"SpaceAPIClient: Attempt {i+1} received server error '{result.get('error')}', trying next fallback")
                            except Exception:
                                pass
                        continue

                return result
            except Exception as e:
                last_error = e
                if DEBUG_MODE:
                    try:
                        print(f"SpaceAPIClient: Attempt {i+1} failed: {e}")
                    except Exception:
                        pass
                continue
        
        # All URLs failed, return the last error
        if isinstance(last_error, dict):
            return last_error
        if isinstance(last_error, urllib.error.URLError):
            return {
                'success': False,
                'error': f'Network error: {str(last_error)}'
            }
        else:
            return {
                'success': False,
                'error': f'Request failed: {str(last_error)}'
            }
    
    def _make_single_request(self, method, url, data=None, headers=None):
        """
        Make a single HTTP request to a specific URL.
        """
        # Prepare headers
        req_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Space-Fusion-Addon/1.0'
        }
        if headers:
            req_headers.update(headers)

        # Always include backend Authorization if configured (JWT or test token)
        if self.backend_token and 'Authorization' not in {k.title(): k for k in req_headers}.keys():
            req_headers['Authorization'] = f'Bearer {self.backend_token}'
        
        # Add idempotency key for POST/PUT requests (required by backend)
        # Use a single canonical header name to avoid proxy/middleware confusion.
        if method in ['POST', 'PUT']:
            req_headers['Idempotency-Key'] = str(uuid.uuid4())
        
        # Prepare request data
        req_data = None
        if data:
            req_data = json.dumps(data).encode('utf-8')
        
        # Persist diagnostic information to a debug log
        try:
            self._write_request_debug(method, url, req_headers, data, req_data)
        except Exception:
            pass

        # Debug: log request summary (mask secrets)
        try:
            log_headers = dict(req_headers)
            if 'Authorization' in log_headers:
                auth_val = log_headers['Authorization']
                if auth_val.startswith('Bearer '):
                    token_part = auth_val[7:]  # Remove "Bearer " prefix
                    if len(token_part) > 10:
                        log_headers['Authorization'] = f'Bearer {token_part[:10]}...'
                    else:
                        log_headers['Authorization'] = 'Bearer ***'
            body_preview = None
            if data:
                body_preview = {k: ('***' if k.lower().endswith('key') else v) for k, v in data.items()}
            print(f"SpaceAPIClient: {method} {url} headers={log_headers} body={body_preview}")
        except Exception:
            pass

        # Create request
        request = urllib.request.Request(
            url, 
            data=req_data, 
            headers=req_headers,
            method=method
        )
        
        try:
            # Make request
            with urllib.request.urlopen(request, timeout=60) as response:
                response_data = response.read()
                
                # Parse JSON response if present
                if response_data:
                    try:
                        parsed = json.loads(response_data.decode('utf-8'))
                    except Exception:
                        parsed = {'raw': response_data[:200].decode('utf-8', errors='ignore')}
                    try:
                        print(f"SpaceAPIClient: Response {getattr(response, 'status', '200')} parsed={str(parsed)[:200]}")
                    except Exception:
                        pass
                    return parsed
                else:
                    return {'success': True}
                    
        except urllib.error.HTTPError as e:
            # Handle HTTP errors
            # Special-case: 413 Payload Too Large with presigned Location header
            if e.code == 413:
                try:
                    location = e.headers.get('Location') if e.headers else None
                    if location:
                        try:
                            print(f"SpaceAPIClient: 413 received; Location={location}")
                        except Exception:
                            pass
                        return {
                            'success': True,
                            'presigned_url': location,
                            'large_file': True
                        }
                except Exception:
                    pass
            
            error_data = {'success': False, 'error': f'HTTP {e.code}: {e.reason}'}
            
            try:
                # Try to parse error response
                raw = e.read()
                error_response = json.loads(raw.decode('utf-8'))
                error_data.update(error_response)
                try:
                    print(f"SpaceAPIClient: HTTPError {e.code} body={str(error_response)[:200]}")
                except Exception:
                    pass
            except:
                try:
                    print(f"SpaceAPIClient: HTTPError {e.code} raw={raw[:200] if 'raw' in locals() else b''}")
                except Exception:
                    pass
                
            return error_data
            
        except urllib.error.URLError as e:
            try:
                print(f"SpaceAPIClient: URLError {str(e)}")
            except Exception:
                pass
            return {
                'success': False,
                'error': f'Network error: {str(e)}'
            }
        except json.JSONDecodeError as e:
            try:
                print(f"SpaceAPIClient: JSON decode error {str(e)}")
            except Exception:
                pass
            return {
                'success': False,
                'error': f'Invalid JSON response: {str(e)}'
            }
        except Exception as e:
            try:
                print(f"SpaceAPIClient: Request failed {str(e)}")
            except Exception:
                pass
            return {
                'success': False,
                'error': f'Request failed: {str(e)}'
            }
    
    def diagnose_network(self):
        """
        Diagnose network connectivity issues.
        Returns detailed diagnostic information.
        """
        diagnostics = []
        
        # Test DNS resolution
        try:
            # Test basic DNS first
            socket.gethostbyname('google.com')
            diagnostics.append("✓ Basic DNS resolution works")
        except Exception as e:
            diagnostics.append(f"✗ Basic DNS resolution failed: {e}")
        
        # Test our backend DNS
        try:
            ip = socket.gethostbyname('space-cad-backend.fly.dev')
            diagnostics.append(f"✓ Backend DNS resolution works: {ip}")
        except Exception as e:
            diagnostics.append(f"✗ Backend DNS resolution failed: {e}")
            
        # Test with different URLs
        for i, test_url in enumerate(BACKEND_FALLBACK_URLS):
            try:
                from urllib.parse import urlparse
                hostname = urlparse(test_url).hostname
                if hostname:
                    ip = socket.gethostbyname(hostname)
                    diagnostics.append(f"✓ Fallback URL {i+1} DNS works: {hostname} -> {ip}")
            except Exception as e:
                diagnostics.append(f"✗ Fallback URL {i+1} DNS failed: {e}")
        
        return {
            'success': len([d for d in diagnostics if d.startswith('✓')]) > 0,
            'diagnostics': diagnostics,
            'environment': {
                'base_url': self.base_url,
                'fallback_urls': BACKEND_FALLBACK_URLS,
                'debug_mode': DEBUG_MODE
            }
        }
    
    def test_connection(self):
        """
        Disabled - API key validation is handled locally only.
        Always returns success to skip server-side validation.
        """
        return {'success': True, 'message': 'API key validation disabled - local caching only'}
    
    def generate(self, prompt):
        """
        Generate new CAD model from natural language prompt.

        Args:
            prompt (str): Natural language description

        Returns:
            dict: Response with model_id, step_file, parameters, etc.
        """
        if not prompt or not prompt.strip():
            return {'success': False, 'error': 'Empty prompt provided'}

        # Prepare request data - include API key as required by backend
        data = {
            'prompt': prompt.strip()
        }
        headers = {}

        # Include Anthropic API key in request body (required by backend)
        if self.api_key:
            data['anthropic_api_key'] = self.api_key
            headers['x-api-key'] = self.api_key  # Also in headers as backup
            print(f"SpaceAPIClient: FINAL DATA TO SEND: {data}")
        else:
            print(f"SpaceAPIClient: NO API KEY - FINAL DATA: {data}")

        try:
            # Debug: log API key format (mask the key)
            if self.api_key and len(self.api_key) > 10:
                masked_key = f'{self.api_key[:10]}...'
            else:
                masked_key = '***'
            print(f"SpaceAPIClient: using API key in x-api-key header: {masked_key}")
        except Exception:
            pass

        result = self._make_request('POST', GENERATE_ENDPOINT, data, headers)
        
        # Handle presigned URL case
        if result.get('large_file'):
            # Download STEP file from presigned URL
            step_data = self._download_step_file(result['presigned_url'])
            if step_data:
                result['step_file'] = step_data
                result['step_file_size'] = len(step_data)
            else:
                result['success'] = False
                result['error'] = 'Failed to download STEP file'
        
        return result
    
    def iterate(self, model_id, prompt):
        """
        Iterate on existing CAD model.
        
        Args:
            model_id (str): ID of the model to iterate
            prompt (str): Description of changes to make
            
        Returns:
            dict: Response with updated step_file, parameters, etc.
        """
        if not model_id:
            return {'success': False, 'error': 'No model ID provided'}
            
        if not prompt or not prompt.strip():
            return {'success': False, 'error': 'Empty iteration prompt provided'}

        # Prepare request data - include API key as required by backend
        data = {
            'prompt': prompt.strip()
        }
        headers = {}

        # Include Anthropic API key in request body (required by backend)
        if self.api_key:
            data['anthropic_api_key'] = self.api_key
            headers['x-api-key'] = self.api_key  # Also in headers as backup

        # Format endpoint with model_id
        endpoint = ITERATE_ENDPOINT.replace('{model_id}', str(model_id))
        
        result = self._make_request('POST', endpoint, data, headers)
        
        # Handle presigned URL case
        if result.get('large_file'):
            # Download STEP file from presigned URL
            step_data = self._download_step_file(result['presigned_url'])
            if step_data:
                result['step_file'] = step_data
                result['step_file_size'] = len(step_data)
            else:
                result['success'] = False
                result['error'] = 'Failed to download STEP file'
        
        return result
    
    def update_parameters(self, model_id, parameter_updates):
        """
        Update model parameters.
        
        Args:
            model_id (str): ID of the model to update
            parameter_updates (list): List of parameter updates
            
        Returns:
            dict: Response with updated step_file
        """
        if not model_id:
            return {'success': False, 'error': 'No model ID provided'}
            
        if not parameter_updates:
            return {'success': False, 'error': 'No parameter updates provided'}

        # Prepare request data - include API key as required by backend
        data = {
            'updates': parameter_updates
        }
        headers = {}

        # Include Anthropic API key in request body (required by backend)
        if self.api_key:
            data['anthropic_api_key'] = self.api_key
            headers['x-api-key'] = self.api_key  # Also in headers as backup

        # Use direct parameters PUT endpoint - hardcoded to avoid caching issues  
        endpoint = f"/api/v1/direct/parameters/{model_id}"
        
        result = self._make_request('PUT', endpoint, data, headers)
        
        # Handle presigned URL case
        if result.get('large_file'):
            # Download STEP file from presigned URL
            step_data = self._download_step_file(result['presigned_url'])
            if step_data:
                result['step_file'] = step_data
                result['step_file_size'] = len(step_data)
            else:
                result['success'] = False
                result['error'] = 'Failed to download STEP file'
        
        return result
    
    def get_parameters(self, model_id):
        """
        Get current model parameters.
        
        Args:
            model_id (str): ID of the model
            
        Returns:
            dict: Response with parameters list
        """
        if not model_id:
            return {'success': False, 'error': 'No model ID provided'}
        
        # Use correct parameters GET endpoint (not direct) - hardcoded to avoid caching issues
        endpoint = f"/api/v1/parameters/{model_id}"
        result = self._make_request('GET', endpoint)
        
        # Backend returns {"parameters": [...]} but we need {"success": true, "parameters": [...]}
        if isinstance(result, dict) and 'parameters' in result:
            return {
                'success': True,
                'parameters': result['parameters']
            }
        elif isinstance(result, dict) and 'error' in result:
            return result  # Already has error format
        else:
            return {'success': False, 'error': 'Invalid response format'}
    
    def _download_step_file(self, presigned_url):
        """
        Download STEP file from presigned URL.
        
        Args:
            presigned_url (str): Presigned URL for the STEP file
            
        Returns:
            str: Base64 encoded STEP file content, or None if failed
        """
        try:
            import base64
            
            request = urllib.request.Request(presigned_url)
            with urllib.request.urlopen(request, timeout=60) as response:
                step_data = response.read()
                # Encode to base64 for consistency with inline responses
                return base64.b64encode(step_data).decode('ascii')
                
        except Exception as e:
            print(f"Failed to download STEP file: {e}")
            return None
