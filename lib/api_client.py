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
ONBOARD_ENDPOINT = getattr(_settings, 'ONBOARD_ENDPOINT')
BACKEND_AUTH_TOKEN = getattr(_settings, 'BACKEND_AUTH_TOKEN', None)

class SpaceAPIClient:
    """
    Simple HTTP client for communicating with the Fly.io backend.
    Uses only standard library to avoid dependencies.
    """
    
    def __init__(self, api_key=None):
        # Use preset API key if none provided
        from settings import DEFAULT_ANTHROPIC_API_KEY
        self.api_key = api_key or DEFAULT_ANTHROPIC_API_KEY
        self.base_url = BACKEND_BASE_URL
        self.backend_token = BACKEND_AUTH_TOKEN
        self._is_onboarded = False
        
    def set_api_key(self, api_key):
        """Set the Anthropic API key for requests."""
        # Clean and validate the API key
        if api_key:
            # Strip whitespace and normalize
            cleaned_key = str(api_key).strip()
            if cleaned_key:
                self.api_key = cleaned_key
                try:
                    print(f"SpaceAPIClient: API key set, length={len(cleaned_key)}, starts_with={cleaned_key[:10]}...")
                except Exception:
                    pass
            else:
                self.api_key = None
        else:
            self.api_key = None
    
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
                return self._make_single_request(method, url, data, headers)
            except Exception as e:
                last_error = e
                if DEBUG_MODE:
                    try:
                        print(f"SpaceAPIClient: Attempt {i+1} failed: {e}")
                    except Exception:
                        pass
                continue
        
        # All URLs failed, return the last error
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
        Test connection to the backend and surface informative errors.
        Returns a dict with 'success' (bool) and either 'message' or 'error'.
        """
        try:
            # First run network diagnostics
            diag_result = self.diagnose_network()
            if not diag_result['success']:
                return {
                    'success': False,
                    'error': 'DNS resolution failed - check network connectivity',
                    'diagnostics': diag_result['diagnostics']
                }
            # Include Authorization if available so backend can validate the key on this check if it wants to.
            # First, if we have an Anthropic key, attempt onboarding to store it server-side.
            if self.api_key:
                onboard_out = self._make_request('POST', ONBOARD_ENDPOINT, data={'anthropic_api_key': self.api_key.strip()})
                try:
                    print(f"SpaceAPIClient: onboard result: {onboard_out}")
                except Exception:
                    pass
                # If onboarding returns an error envelope, surface it but continue status check
                if isinstance(onboard_out, dict) and onboard_out.get('success') is False:
                    # Keep going to status probe, but remember onboarding failed
                    self._is_onboarded = False
                else:
                    self._is_onboarded = True

            headers = None  # Status check is public on backend

            # Try a small set of common health endpoints
            endpoints = ['/api/v1/status', '/status', '/health', '/healthz']
            last_err = None
            for ep in endpoints:
                result = self._make_request('GET', ep, headers=headers)
                try:
                    print(f"SpaceAPIClient: test_connection [{ep}] raw result: {result}")
                except Exception:
                    pass

                # Success path: health ok
                if isinstance(result, dict) and result.get('health') == 'ok':
                    return {'success': True, 'message': 'Backend connection successful', 'health': 'ok', 'endpoint': ep}

                # Standard error envelope
                if isinstance(result, dict) and result.get('success') is False:
                    err = result.get('error') or result.get('message') or 'Backend reported failure'
                    out = {'success': False, 'error': err, 'endpoint': ep}
                    for k in ('code', 'message', 'detail', 'details'):
                        if k in result and k not in out:
                            out[k] = result[k]
                    out['payload'] = result
                    # Return on first definitive failure with details
                    return out

                # Collect last error context for unexpected schemas
                last_err = {
                    'success': False,
                    'error': f'Unexpected response schema from {ep}',
                    'endpoint': ep,
                    'payload': result if isinstance(result, dict) else str(result)[:200]
                }

            # If none succeeded and no definitive failure was returned, surface last observed error
            return last_err or {'success': False, 'error': 'Status check failed for all known endpoints'}

        except Exception as e:
            return {'success': False, 'error': f'Connection test failed: {type(e).__name__}: {str(e)}'}
    
    def generate(self, prompt):
        """
        Generate new CAD model from natural language prompt.
        
        Args:
            prompt (str): Natural language description
            
        Returns:
            dict: Response with model_id, step_file, parameters, etc.
        """
        if not self.api_key:
            return {'success': False, 'error': 'No API key configured'}
        
        if not prompt or not prompt.strip():
            return {'success': False, 'error': 'Empty prompt provided'}
        
        # Always ensure onboarding before generate requests for reliability
        if self.api_key:
            onboard_result = self._make_request('POST', ONBOARD_ENDPOINT, data={'anthropic_api_key': self.api_key.strip()})
            try:
                print(f"SpaceAPIClient: generate onboard result: {onboard_result}")
            except Exception:
                pass
            
            # Check if onboarding failed
            if isinstance(onboard_result, dict) and onboard_result.get('success') is False:
                # Return the onboarding error instead of proceeding
                error_msg = onboard_result.get('error') or onboard_result.get('message') or 'Onboarding failed'
                return {'success': False, 'error': f'API key onboarding failed: {error_msg}'}
            
            self._is_onboarded = True

        # Prepare request data - include API key as backup for backend
        data = {
            'prompt': prompt.strip(),
            'anthropic_api_key': self.api_key.strip()  # Backup auth method
        }
        headers = None  # Use backend Authorization only
        
        try:
            # Debug: log auth header format (mask the key)
            masked_auth = f'Bearer {self.api_key[:10]}...' if len(self.api_key) > 10 else 'Bearer ***'
            print(f"SpaceAPIClient: generate auth header: {masked_auth}")
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
        if not self.api_key:
            return {'success': False, 'error': 'No API key configured'}
        
        if not model_id:
            return {'success': False, 'error': 'No model ID provided'}
            
        if not prompt or not prompt.strip():
            return {'success': False, 'error': 'Empty iteration prompt provided'}
        
        # Always ensure onboarding before iterate requests for reliability
        if self.api_key:
            onboard_result = self._make_request('POST', ONBOARD_ENDPOINT, data={'anthropic_api_key': self.api_key.strip()})
            try:
                print(f"SpaceAPIClient: iterate onboard result: {onboard_result}")
            except Exception:
                pass
            
            # Check if onboarding failed
            if isinstance(onboard_result, dict) and onboard_result.get('success') is False:
                # Return the onboarding error instead of proceeding
                error_msg = onboard_result.get('error') or onboard_result.get('message') or 'Onboarding failed'
                return {'success': False, 'error': f'API key onboarding failed: {error_msg}'}
            
            self._is_onboarded = True

        # Prepare request data - include API key as backup for backend
        data = {
            'prompt': prompt.strip(),
            'anthropic_api_key': self.api_key.strip()  # Backup auth method
        }
        headers = None
        
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
        if not self.api_key:
            return {'success': False, 'error': 'No API key configured'}
        
        if not model_id:
            return {'success': False, 'error': 'No model ID provided'}
            
        if not parameter_updates:
            return {'success': False, 'error': 'No parameter updates provided'}
        
        # Always ensure onboarding before parameter update requests for reliability
        if self.api_key:
            onboard_result = self._make_request('POST', ONBOARD_ENDPOINT, data={'anthropic_api_key': self.api_key.strip()})
            try:
                print(f"SpaceAPIClient: update_parameters onboard result: {onboard_result}")
            except Exception:
                pass
            
            # Check if onboarding failed
            if isinstance(onboard_result, dict) and onboard_result.get('success') is False:
                # Return the onboarding error instead of proceeding
                error_msg = onboard_result.get('error') or onboard_result.get('message') or 'Onboarding failed'
                return {'success': False, 'error': f'API key onboarding failed: {error_msg}'}
            
            self._is_onboarded = True

        # Prepare request data - include API key as backup for backend
        data = {
            'updates': parameter_updates,
            'anthropic_api_key': self.api_key.strip()  # Backup auth method
        }
        headers = None
        
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
