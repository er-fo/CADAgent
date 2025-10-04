"""
Space - Fusion 360 AI CAD Add-on
Updated to follow Fusion 360 best practices

Architecture:
- JavaScript frontend handles UI interactions and user input
- Python backend handles network requests and Fusion API operations  
- Bridge communication for all external API calls and CAD data
"""

import adsk.core
import adsk.fusion
import adsk.cam
import traceback
import sys
import os
from pathlib import Path
import json
import base64
import tempfile
import threading
import shutil
import urllib.request
import urllib.parse
import urllib.error
from collections import deque

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file in Space directory."""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

# Load environment variables at module level
load_env_file()

# Add lib folder to path for our modules
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
if lib_path not in sys.path:
    sys.path.append(lib_path)

# Add config folder to path for settings
config_path = os.path.join(os.path.dirname(__file__), 'config')
if config_path not in sys.path:
    sys.path.append(config_path)

# Import our modules (only ones needed for Fusion API operations)
SpaceFileManager = None
SpaceFusionUtils = None
SpaceAPIClient = None
CHAT_WINDOW_WIDTH = 320
CHAT_WINDOW_HEIGHT = 480

try:
    # Force reload modules to pick up code changes without restarting Fusion
    import importlib
    import sys

    # Reload file_manager if already imported
    if 'file_manager' in sys.modules:
        importlib.reload(sys.modules['file_manager'])
    from file_manager import SpaceFileManager

    # Reload fusion_utils if already imported
    if 'fusion_utils' in sys.modules:
        importlib.reload(sys.modules['fusion_utils'])
    from fusion_utils import SpaceFusionUtils

    # Optional: API client for network calls in background threads
    try:
        if 'api_client' in sys.modules:
            importlib.reload(sys.modules['api_client'])
        from api_client import SpaceAPIClient
    except Exception:
        SpaceAPIClient = None

    if 'settings' in sys.modules:
        importlib.reload(sys.modules['settings'])
    from settings import CHAT_WINDOW_WIDTH, CHAT_WINDOW_HEIGHT
except ImportError as e:
    # Log import error but continue with defaults
    try:
        adsk.core.Application.get().log(f'CADAgent: Import error during initialization: {e}')
    except Exception:
        print('CADAgent: Import error during initialization:', e, file=sys.stderr)

# Global variables for add-in lifecycle
_app = adsk.core.Application.get()
_ui = _app.userInterface
_palette = None
_handlers = []
_backend_url = None
_api_client = None
_op_in_progress = False

# Add-in components (only Fusion API related)
_file_manager = None
_fusion_utils = None

# API key cache (persists until Fusion 360 is restarted)
_cached_api_key = None

# Simplified approach - no complex threading, use Fusion's event-driven pattern
# Background threads signal HTML, HTML sends actions back to main thread

def make_error_user_friendly(error_message):
    """
    Convert technical error messages to user-friendly ones.
    """
    if not error_message:
        return "An error occurred. Please try again."
    
    error_lower = error_message.lower()
    
    # API key related errors
    if 'api key onboarding failed' in error_lower:
        if 'connection refused' in error_lower or 'network error' in error_lower:
            return "Unable to connect to CADAgent servers. Please check your internet connection and try again."
        elif 'unauthorized' in error_lower or '401' in error_lower:
            return "API key is invalid. Please check your Anthropic API key at https://console.anthropic.com/account/keys"
        elif 'forbidden' in error_lower or '403' in error_lower:
            return "API key access denied. Please verify your Anthropic API key has sufficient permissions."
        else:
            return "API key validation failed. Please verify your Anthropic API key at https://console.anthropic.com/account/keys"
    
    # Network related errors
    if 'connection refused' in error_lower or 'network error' in error_lower:
        return "Unable to connect to CADAgent servers. Please check your internet connection and try again."
    
    # Rate limiting
    if 'rate limit' in error_lower or '429' in error_lower:
        return "API rate limit exceeded. Please wait a moment and try again."
    
    # Server errors
    if '500' in error_lower or 'server error' in error_lower:
        return "Server temporarily unavailable. Please try again in a few moments."
    
    # Timeout errors
    if 'timeout' in error_lower:
        return "Request timed out. Please try again with a simpler design or check your connection."
    
    # If no specific pattern matches, return a cleaned up version
    # Remove technical details but keep the core message
    if 'error:' in error_lower:
        # Extract the main error message after "Error:"
        error_parts = error_message.split('Error:', 1)
        if len(error_parts) > 1:
            return f"Error: {error_parts[1].strip()}"
    
    return error_message

def signal_html_from_background(message_type, data):
    """
    Signal the HTML interface from a background thread.
    This is safe and follows Fusion 360 patterns.
    """
    try:
        if _palette:
            _palette.sendInfoToHTML(message_type, json.dumps(data))
            return True
        return False
    except Exception as e:
        try:
            print(f"CADAgent: Failed to signal HTML: {e}", file=sys.stderr)
        except Exception:
            pass
        return False



def run(context):
    """
    Entry point for the add-in following Fusion 360 best practices.
    Only initializes Fusion API components.
    """
    try:
        global _palette, _file_manager, _fusion_utils, _api_client
        
        _app.log('CADAgent: Starting CADAgent AI CAD add-in...')
        
        # Initialize Fusion API components only
        if SpaceFileManager is not None:
            _file_manager = SpaceFileManager()
        if SpaceFusionUtils is not None:
            _fusion_utils = SpaceFusionUtils()
        if SpaceAPIClient is not None:
            try:
                client_module = sys.modules.get(SpaceAPIClient.__module__)
                if client_module and hasattr(client_module, '__file__'):
                    _app.log(f'CADAgent: SpaceAPIClient module path: {client_module.__file__}')
            except Exception as client_path_error:
                _app.log(f'CADAgent: Could not determine SpaceAPIClient module path: {client_path_error}')

            try:
                _api_client = SpaceAPIClient()
            except Exception:
                _api_client = None

        # Auto-retrieve cached API key from Fusion attributes on startup
        global _cached_api_key
        if _fusion_utils and not _cached_api_key:
            try:
                startup_key = _fusion_utils.retrieve_api_key()
                if startup_key:
                    _cached_api_key = startup_key
                    _app.log('CADAgent: Auto-retrieved cached API key from Fusion attributes on startup')
                    # Also set in API client if available
                    if _api_client:
                        try:
                            _api_client.set_api_key(startup_key)
                            _app.log('CADAgent: API client initialized with cached key')
                        except Exception as client_init_e:
                            _app.log(f'CADAgent: Warning - could not initialize API client with cached key: {client_init_e}')
                else:
                    _app.log('CADAgent: No cached API key found in Fusion attributes on startup')
            except Exception as startup_e:
                _app.log(f'CADAgent: Warning - could not retrieve cached API key on startup: {startup_e}')
        
        # Create the palette for HTML/JS frontend
        palette_id = 'SpaceAICadPaletteV23'
        
        # Remove existing palette if present
        for palette in _ui.palettes:
            if palette.id == palette_id:
                palette.deleteMe()

        # Get the path to the HTML file and convert to proper file URI (fixes Windows backslash issues)
        # Using Path.as_uri() avoids invalid URLs like ...%5Cui%5Cchat.html in some environments
        html_path = Path(__file__).parent / 'ui' / 'chat.html'
        html_file = html_path.as_uri() if html_path.exists() else str(html_path)
        try:
            _app.log(f'CADAgent: HTML URL resolved to: {html_file}')
        except Exception:
            pass
        
        # Store backend URL to pass to HTML via initial message
        global _backend_url
        config_path = os.path.join(os.path.dirname(__file__), 'config')
        if config_path not in sys.path:
            sys.path.append(config_path)
        from settings import BACKEND_BASE_URL
        _backend_url = BACKEND_BASE_URL

        # Create new palette
        _palette = _ui.palettes.add(
                id=palette_id,
                name='space v23',
                htmlFileURL=html_file,
                isVisible=True,
                showCloseButton=True,
                isResizable=True,
                width=CHAT_WINDOW_WIDTH,
                height=CHAT_WINDOW_HEIGHT,
                useNewWebBrowser=True  # Use Qt browser following best practices
            )
            
        # Dock the palette to the right side, filling from top to bottom
        # Following Fusion 360 best practices for palette positioning
        try:
            _palette.dockingState = adsk.core.PaletteDockingStates.PaletteDockStateLeft
            # Make the palette fill the available vertical space
            _palette.isResizable = True
            # Set appropriate width for right-docked panel (wider than floating)
            _palette.width = 420  # 5% wider than previous 400px
        except Exception as dock_error:
            _app.log(f'CADAgent: Could not dock palette to right: {dock_error}')
            # Fallback: ensure it's visible even if docking fails
            _palette.isVisible = True
        
        # Main-thread callback system is ready to use with beginInvoke testing

        # Set up communication handler
        onHTMLEvent = HTMLEventHandler()
        _palette.incomingFromHTML.add(onHTMLEvent)
        _handlers.append(onHTMLEvent)

        # System is ready - using event-driven pattern
        _app.log('CADAgent: Event-driven system ready (background thread → HTML → main thread)')
        
        _app.log(f'CADAgent: Add-in started successfully. Palette visible: {_palette.isVisible}')
        _app.log('CADAgent: ===== STARTUP COMPLETE =====')
        
    except Exception as e:
        _app.log(f'CADAgent: Failed to start add-in: {e}')
        _app.log(traceback.format_exc())


def stop(context):
    """Clean up when stopping the add-in."""
    try:
        global _palette, _handlers
        
        # Clean up event handlers
        for handler in _handlers:
            try:
                handler.removeHandler()
            except:
                pass
        _handlers.clear()
        
        # Clean up palette
        if _palette:
            _palette.deleteMe()
            _palette = None
        
        _app.log('CADAgent: Add-in stopped successfully')
        
    except Exception as e:
        _app.log(f'CADAgent: Error during stop: {e}')


def safe_log(message):
    """Safe logging that won't crash if Application is not available."""
    try:
        _app.log(message)
    except Exception:
        try:
            print(f"CADAgent: {message}", file=sys.stderr)
        except Exception:
            pass


class HTMLEventHandler(adsk.core.HTMLEventHandler):
    """
    Handler for communication between HTML/JS and Python.
    Following Fusion 360 best practices: Python only handles Fusion API operations.
    """
    
    def __init__(self):
        super().__init__()
    
    def notify(self, args):
        try:
            html_args = adsk.core.HTMLEventArgs.cast(args)
            action = html_args.action
            data = html_args.data
            
            _app.log(f'CADAgent: Received action "{action}" with data: {data[:200] if data else "None"}...')
            
            # Parse data if present
            request_data = {}
            if data:
                try:
                    request_data = json.loads(data)
                except:
                    request_data = {'raw_data': data}
            
            # Handle different actions - only Fusion API operations
            if action == 'ping':
                # Simple connectivity test
                _app.log('CADAgent: Ping received from HTML; responding with pong')
                html_args.returnData = 'pong'
                _app.log('CADAgent: Ping response sent (pong)')
            
            elif action == 'html_ready':
                # HTML reports it is ready
                _app.log(f"CADAgent: HTML reported ready with data: {request_data}")
                try:
                    # Check if we have a cached API key to inform the UI
                    cached_key_info = self.handle_get_cached_api_key({})
                    has_cached_key = cached_key_info.get('success') and cached_key_info.get('has_cached_key', False)
                    cached_key_source = cached_key_info.get('source', 'none') if has_cached_key else None

                    if has_cached_key:
                        _app.log(f'CADAgent: Informing UI about cached API key from {cached_key_source}')
                    else:
                        _app.log('CADAgent: No cached API key to report to UI')

                    initial_message = json.dumps({
                        'type': 'init',
                        'message': 'CADAgent AI CAD ready - Fusion API mode',
                        'version': 'v2.0.0',
                        'backend': 'python-fusion-api-only',
                        'backend_url': _backend_url,
                        'has_cached_api_key': has_cached_key,
                        'cached_key_source': cached_key_source,
                        'cached_key_display': 'Using cached API key' if has_cached_key else None
                    })
                    ret = _palette.sendInfoToHTML('init', initial_message) if _palette else ''
                    _app.log(f'CADAgent: Sent init to HTML; handler returned: {ret}')
                except Exception as init_e:
                    _app.log(f'CADAgent: Error sending init: {init_e}')
                html_args.returnData = json.dumps({'success': True, 'ack': 'HTML_READY'})
                
            elif action == 'step_import_from_background':
                # Handle STEP import triggered by background thread via HTML
                # This runs on main thread and can safely use Fusion API
                _app.log('CADAgent: Received step_import_from_background action on main thread')
                result = self.handle_step_import(request_data)
                html_args.returnData = json.dumps(result)
                
                # Signal completion back to HTML
                if result.get('success'):
                    _app.log('CADAgent: STEP import completed successfully')
                    if _palette:
                        success_data = {
                            'success': True,
                            'message': 'Model generated and imported successfully',
                            'model_id': result.get('model_id'),
                            'is_iteration': result.get('is_iteration', False)
                        }
                        _palette.sendInfoToHTML('generation_complete', json.dumps(success_data))
                else:
                    _app.log(f'CADAgent: STEP import failed: {result.get("error")}')
                    if _palette:
                        _palette.sendInfoToHTML('error', json.dumps(result))
                
            elif action == 'import_step_file':
                # Import STEP file into Fusion 360 (core Fusion API operation)
                result = self.handle_step_import(request_data)
                html_args.returnData = json.dumps(result)
                
            elif action == 'export_step_file':
                # Export current design as STEP file
                result = self.handle_step_export(request_data)
                html_args.returnData = json.dumps(result)
                
            elif action == 'get_design_info':
                # Get information about current design
                result = self.handle_get_design_info(request_data)
                html_args.returnData = json.dumps(result)
            
            elif action == 'validate_api_key':
                # Store API key locally without server validation
                try:
                    api_key = request_data.get('api_key', '')
                    if not api_key:
                        html_args.returnData = json.dumps({'success': False, 'error': 'No API key provided'})
                    else:
                        # Just store the key locally and return success immediately
                        result = self.handle_store_api_key({'api_key': api_key})
                        if result.get('success'):
                            html_args.returnData = json.dumps({'success': True, 'message': 'API key saved locally'})
                            # Signal success to HTML immediately
                            signal_html_from_background('api_validation_result', {'success': True, 'message': 'API key saved locally'})
                        else:
                            html_args.returnData = json.dumps(result)
                            signal_html_from_background('api_validation_result', result)
                except Exception as e:
                    error_result = {'success': False, 'error': f'Failed to save API key: {str(e)}'}
                    html_args.returnData = json.dumps(error_result)
                    signal_html_from_background('api_validation_result', error_result)
                
            elif action == 'generate_model':
                # Kick off generation in background, ACK immediately
                try:
                    html_args.returnData = json.dumps({'success': True, 'processing': True, 'message': 'generation started'})
                except Exception:
                    html_args.returnData = '{"success": true, "processing": true}'

                def generate_in_background():
                    try:
                        prompt = request_data.get('prompt', '')
                        api_key = request_data.get('anthropic_api_key', '').strip()

                        _app.log(f'CADAgent: DEBUG - request_data keys: {list(request_data.keys())}')
                        _app.log(f'CADAgent: DEBUG - anthropic_api_key from UI: "{api_key}", length: {len(api_key)}')
                        _app.log(f'CADAgent: DEBUG - full request_data: {request_data}')

                        # If no API key provided from UI, retrieve from all cache sources
                        if not api_key:
                            _app.log('CADAgent: No API key from UI, checking all cache sources...')
                            try:
                                cache_result = self.handle_get_cached_api_key({})
                                if cache_result.get('success') and cache_result.get('api_key'):
                                    api_key = cache_result['api_key']
                                    _app.log(f'CADAgent: Using cached API key from {cache_result.get("source", "unknown")}')
                                else:
                                    _app.log('CADAgent: No cached API key found in any source')
                            except Exception as cache_e:
                                _app.log(f'CADAgent: Error retrieving cached API key: {cache_e}')

                        # Final fallback to environment and settings
                        if not api_key:
                            api_key = os.environ.get('ANTHROPIC_API_KEY', '')
                            if api_key:
                                _app.log('CADAgent: Using API key from environment variable')

                        if not api_key:
                            try:
                                config_path = os.path.join(os.path.dirname(__file__), 'config')
                                if config_path not in sys.path:
                                    sys.path.append(config_path)
                                from settings import DEFAULT_ANTHROPIC_API_KEY
                                api_key = DEFAULT_ANTHROPIC_API_KEY
                                if api_key:
                                    _app.log('CADAgent: Using default API key from settings')
                            except ImportError:
                                pass
                        
                        if not prompt or not api_key:
                            # Signal error to HTML
                            signal_html_from_background('error', {
                                'success': False, 
                                'message': 'Missing prompt or API key'
                            })
                            return

                        # Ensure file session
                        try:
                            if _file_manager:
                                _file_manager.start_new_session()
                        except Exception:
                            pass

                        # API request starting
                        _app.log('CADAgent: Starting API request to backend...')

                        # Use API client (preferred) to avoid Fusion API usage in thread
                        result = None
                        try:
                            if _api_client is not None:
                                _app.log('CADAgent: Using API client for generation')
                                try:
                                    _api_client.set_api_key(api_key)
                                    _app.log('CADAgent: API key set on client successfully')
                                    try:
                                        key_length = len(_api_client.api_key or '')
                                    except Exception:
                                        key_length = 0
                                    _app.log(f'CADAgent: API client key length after set: {key_length}')
                                except Exception as set_key_e:
                                    _app.log(f'CADAgent: Failed to set API key on client: {set_key_e}')
                                
                                _app.log(f'CADAgent: About to call _api_client.generate with prompt: "{prompt}"')
                                try:
                                    result = _api_client.generate(prompt)
                                    _app.log('CADAgent: API client generate call completed successfully')
                                except Exception as gen_e:
                                    _app.log(f'CADAgent: EXCEPTION in _api_client.generate(): {gen_e}')
                                    _app.log(f'CADAgent: Exception type: {type(gen_e).__name__}')
                                    _app.log(f'CADAgent: Full traceback: {traceback.format_exc()}')
                                    raise gen_e  # Re-raise to be caught by outer exception handler
                            else:
                                _app.log('CADAgent: API client is None, using fallback')
                                # Fallback: use existing handler (urllib) in thread - no API key needed
                                result = self.handle_generate_model({'prompt': prompt})
                        except Exception as e:
                            _app.log(f'CADAgent: Exception in generation: {e}')
                            result = {'success': False, 'error': f'Generation failed: {str(e)}'}
                            
                        # API response received - log basic info
                        _app.log(f'CADAgent: API response received, type: {type(result).__name__}')

                        # Handle success path
                        if isinstance(result, dict) and result.get('success') and (result.get('step_file') or result.get('step_file_data')):
                            step_b64 = result.get('step_file') or result.get('step_file_data')
                            model_id = result.get('model_id')
                            planning_summary = result.get('planning_summary')
                            parameters = result.get('parameters')
                            
                            # Prepare for STEP import on main thread
                            _app.log(f'CADAgent: Preparing STEP import - model_id: {model_id}, data length: {len(step_b64) if step_b64 else 0}')

                            def import_on_main_thread():
                                """Import STEP file on main thread - no self references allowed"""
                                try:
                                    _app.log(f'CADAgent: STEP import starting - model_id: {model_id}, is_iteration: False')
                                    _app.log(f'CADAgent: STEP data length: {len(step_b64) if step_b64 else 0} chars')
                                    
                                    if not step_b64:
                                        _app.log('CADAgent: ERROR - No STEP file data provided')
                                        if _palette:
                                            _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': 'No STEP file data provided'}))
                                        return
                                    
                                    _app.log('CADAgent: Starting STEP file import...')
                                    
                                    # Decode base64 STEP file data
                                    try:
                                        step_binary = base64.b64decode(step_b64)
                                        _app.log(f'CADAgent: Decoded STEP file successfully, size: {len(step_binary)} bytes')
                                    except Exception as e:
                                        error_msg = f'Failed to decode STEP data: {e}'
                                        _app.log(f'CADAgent: ERROR - {error_msg}')
                                        if _palette:
                                            _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                        return
                                    
                                    # Save to temporary file
                                    temp_path = None
                                    try:
                                        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as temp_file:
                                            temp_file.write(step_binary)
                                            temp_path = temp_file.name
                                        _app.log(f'CADAgent: Saved STEP file to temporary path: {temp_path}')
                                    except Exception as e:
                                        error_msg = f'Failed to save temporary STEP file: {e}'
                                        _app.log(f'CADAgent: ERROR - {error_msg}')
                                        if _palette:
                                            _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                        return
                                    
                                    try:
                                        # Import into Fusion 360 using Fusion API
                                        _app.log('CADAgent: Getting Fusion API objects...')
                                        app = adsk.core.Application.get()
                                        import_manager = app.importManager
                                        design = app.activeProduct
                                        
                                        if not design:
                                            error_msg = 'No active design found in Fusion 360'
                                            _app.log(f'CADAgent: ERROR - {error_msg}')
                                            if _palette:
                                                _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                            return
                                        
                                        root_component = design.rootComponent
                                        _app.log(f'CADAgent: Got Fusion API objects - design: {design.name if hasattr(design, "name") else "unnamed"}')
                                        
                                        # Create import options
                                        _app.log('CADAgent: Creating STEP import options...')
                                        step_options = import_manager.createSTEPImportOptions(temp_path)
                                        step_options.isViewFit = True  # Fit the imported geometry to the view
                                        _app.log('CADAgent: Import options created successfully')
                                        
                                        # Import the file - Fusion 360 handles transactions automatically
                                        _app.log('CADAgent: Starting STEP file import to Fusion 360...')
                                        import_manager.importToTarget(step_options, root_component)
                                        _app.log('CADAgent: STEP file imported successfully into Fusion 360!')
                                        
                                        # Save the imported model if file manager is available
                                        if _file_manager:
                                            try:
                                                _file_manager.start_new_session()
                                                saved_path = _file_manager.save_imported_step(temp_path, model_id)
                                                _app.log(f'CADAgent: Saved imported model to: {saved_path}')
                                            except Exception as save_e:
                                                _app.log(f'CADAgent: Warning - could not save imported model: {save_e}')
                                        
                                        final = {
                                            'success': True,
                                            'imported': True,
                                            'model_id': model_id,
                                            'planning_summary': planning_summary,
                                            'parameters': parameters,
                                            'model_replaced': False
                                        }
                                        _app.log(f'CADAgent: Import completed successfully - returning result: {final}')
                                        
                                        if _palette:
                                            _palette.sendInfoToHTML('generation_complete', json.dumps(final))
                                        
                                    finally:
                                        # Clean up temporary file
                                        if temp_path:
                                            try:
                                                _app.log(f'CADAgent: Cleaning up temporary file: {temp_path}')
                                                os.unlink(temp_path)
                                                _app.log('CADAgent: Temporary file cleaned up successfully')
                                            except Exception as cleanup_error:
                                                _app.log(f'CADAgent: Warning - could not delete temp file: {cleanup_error}')
                                        
                                except Exception as ie:
                                    error_msg = f'Import failed: {str(ie)}'
                                    _app.log(f'CADAgent: CRITICAL ERROR - {error_msg}')
                                    _app.log(f'CADAgent: Full traceback: {traceback.format_exc()}')
                                    if _palette:
                                        _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                        
                            # Signal HTML from background thread to trigger main thread import
                            _app.log('CADAgent: Signaling HTML to trigger main thread import...')
                            signal_success = signal_html_from_background('step_ready_for_import', {
                                'step_file_data': step_b64,
                                'model_id': model_id,
                                'is_iteration': False,
                                'planning_summary': planning_summary,
                                'parameters': parameters
                            })
                            if signal_success:
                                _app.log('CADAgent: HTML signaled successfully for STEP import')
                            else:
                                _app.log('CADAgent: ERROR - Failed to signal HTML')
                                # Fallback error signaling
                                signal_html_from_background('error', {'success': False, 'message': 'Failed to signal HTML for import'})

                        elif isinstance(result, dict) and result.get('large_file') and result.get('presigned_url'):
                            # Download from presigned URL, then import
                            def download_and_import():
                                try:
                                    import base64, urllib.request
                                    with urllib.request.urlopen(result['presigned_url'], timeout=120) as resp:
                                        data = resp.read()
                                    step_b64_2 = base64.b64encode(data).decode('ascii')
                                    model_id = result.get('model_id')
                                    planning_summary = result.get('planning_summary')
                                    parameters = result.get('parameters')
                                    
                                    def import_main():
                                        """Import STEP file from presigned URL - no self references allowed"""
                                        try:
                                            _app.log(f'CADAgent: STEP import starting from presigned URL - model_id: {model_id}')
                                            
                                            if not step_b64_2:
                                                _app.log('CADAgent: ERROR - No STEP file data from presigned URL')
                                                if _palette:
                                                    _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': 'No STEP file data from presigned URL'}))
                                                return
                                            
                                            # Decode base64 STEP file data
                                            try:
                                                step_binary = base64.b64decode(step_b64_2)
                                                _app.log(f'CADAgent: Decoded STEP file successfully, size: {len(step_binary)} bytes')
                                            except Exception as e:
                                                error_msg = f'Failed to decode STEP data: {e}'
                                                _app.log(f'CADAgent: ERROR - {error_msg}')
                                                if _palette:
                                                    _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                                return
                                            
                                            # Save to temporary file
                                            temp_path = None
                                            try:
                                                with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as temp_file:
                                                    temp_file.write(step_binary)
                                                    temp_path = temp_file.name
                                                _app.log(f'CADAgent: Saved STEP file to temporary path: {temp_path}')
                                            except Exception as e:
                                                error_msg = f'Failed to save temporary STEP file: {e}'
                                                _app.log(f'CADAgent: ERROR - {error_msg}')
                                                if _palette:
                                                    _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                                return
                                            
                                            try:
                                                # Import into Fusion 360 using Fusion API
                                                _app.log('CADAgent: Getting Fusion API objects...')
                                                app = adsk.core.Application.get()
                                                import_manager = app.importManager
                                                design = app.activeProduct
                                                
                                                if not design:
                                                    error_msg = 'No active design found in Fusion 360'
                                                    _app.log(f'CADAgent: ERROR - {error_msg}')
                                                    if _palette:
                                                        _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                                    return
                                                
                                                root_component = design.rootComponent
                                                _app.log(f'CADAgent: Got Fusion API objects - design: {design.name if hasattr(design, "name") else "unnamed"}')
                                                
                                                # Create import options
                                                _app.log('CADAgent: Creating STEP import options...')
                                                step_options = import_manager.createSTEPImportOptions(temp_path)
                                                step_options.isViewFit = True
                                                _app.log('CADAgent: Import options created successfully')
                                                
                                                # Import the file - Fusion 360 handles transactions automatically
                                                _app.log('CADAgent: Starting STEP file import to Fusion 360...')
                                                import_manager.importToTarget(step_options, root_component)
                                                _app.log('CADAgent: STEP file imported successfully into Fusion 360!')
                                                
                                                final = {
                                                    'success': True,
                                                    'imported': True,
                                                    'model_id': model_id,
                                                    'planning_summary': planning_summary,
                                                    'parameters': parameters,
                                                    'model_replaced': False
                                                }
                                                _app.log(f'CADAgent: Import completed successfully - returning result: {final}')
                                                
                                                if _palette:
                                                    _palette.sendInfoToHTML('generation_complete', json.dumps(final))
                                                
                                            finally:
                                                # Clean up temporary file
                                                if temp_path:
                                                    try:
                                                        _app.log(f'CADAgent: Cleaning up temporary file: {temp_path}')
                                                        os.unlink(temp_path)
                                                        _app.log('CADAgent: Temporary file cleaned up successfully')
                                                    except Exception as cleanup_error:
                                                        _app.log(f'CADAgent: Warning - could not delete temp file: {cleanup_error}')
                                            
                                        except Exception as ie2:
                                            error_msg = f'Import failed: {str(ie2)}'
                                            _app.log(f'CADAgent: CRITICAL ERROR - {error_msg}')
                                            _app.log(f'CADAgent: Full traceback: {traceback.format_exc()}')
                                            if _palette:
                                                _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                    # Signal HTML from background thread for large file import
                                    signal_success = signal_html_from_background('step_ready_for_import', {
                                        'step_file_data': step_b64_2,
                                        'model_id': model_id,
                                        'is_iteration': False,
                                        'planning_summary': planning_summary,
                                        'parameters': parameters
                                    })
                                    if not signal_success:
                                        _app.log('CADAgent: ERROR - Failed to signal HTML for large file import')
                                        signal_html_from_background('error', {'success': False, 'message': 'Failed to signal HTML for import'})
                                except Exception as de:
                                    if _palette:
                                        _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': f'Download failed: {str(de)}'}))
                            # Run the downloader in another short-lived thread to avoid blocking
                            t2 = threading.Thread(target=download_and_import, daemon=True)
                            t2.start()
                        else:
                            # Failure path - signal error directly to HTML
                            _app.log(f'CADAgent: Generation failed - result: {str(result)[:300] if result else "None"}')
                            raw_msg = (result or {}).get('error') or (result or {}).get('message') or 'Generation failed'
                            # Convert to user-friendly message
                            user_msg = make_error_user_friendly(raw_msg)
                            signal_html_from_background('error', {'success': False, 'message': user_msg})
                    except Exception as e:
                        _app.log(f'CADAgent: Background thread exception: {str(e)}')
                        # Signal error directly to HTML with user-friendly message
                        error_msg = make_error_user_friendly(f'Generation error: {str(e)}')
                        signal_html_from_background('error', {'success': False, 'message': error_msg})

                t = threading.Thread(target=generate_in_background, daemon=True)
                t.start()
                
            elif action == 'iterate_model':
                # Kick off iteration in background, ACK immediately
                try:
                    html_args.returnData = json.dumps({'success': True, 'processing': True, 'message': 'iteration started'})
                except Exception:
                    html_args.returnData = '{"success": true, "processing": true}'

                def iterate_in_background():
                    try:
                        model_id = request_data.get('model_id', '')
                        prompt = request_data.get('prompt', '')
                        api_key = request_data.get('anthropic_api_key', '').strip()

                        # If no API key provided from UI, retrieve from all cache sources
                        if not api_key:
                            _app.log('CADAgent: No API key from UI for iteration, checking all cache sources...')
                            try:
                                cache_result = self.handle_get_cached_api_key({})
                                if cache_result.get('success') and cache_result.get('api_key'):
                                    api_key = cache_result['api_key']
                                    _app.log(f'CADAgent: Using cached API key from {cache_result.get("source", "unknown")} for iteration')
                                else:
                                    _app.log('CADAgent: No cached API key found in any source for iteration')
                            except Exception as cache_e:
                                _app.log(f'CADAgent: Error retrieving cached API key for iteration: {cache_e}')

                        # Final fallback to environment and settings
                        if not api_key:
                            api_key = os.environ.get('ANTHROPIC_API_KEY', '')
                            if api_key:
                                _app.log('CADAgent: Using API key from environment variable for iteration')

                        if not api_key:
                            try:
                                config_path = os.path.join(os.path.dirname(__file__), 'config')
                                if config_path not in sys.path:
                                    sys.path.append(config_path)
                                from settings import DEFAULT_ANTHROPIC_API_KEY
                                api_key = DEFAULT_ANTHROPIC_API_KEY
                                if api_key:
                                    _app.log('CADAgent: Using default API key from settings for iteration')
                            except ImportError:
                                pass
                        
                        if not model_id or not prompt or not api_key:
                            # Signal error directly to HTML
                            signal_html_from_background('error', {'success': False, 'message': 'Missing model_id, prompt, or API key'})
                            return

                        # Use API client when available
                        result = None
                        try:
                            if _api_client is not None:
                                try:
                                    _api_client.set_api_key(api_key)
                                except Exception:
                                    pass
                                result = _api_client.iterate(model_id, prompt)
                            else:
                                result = self.handle_iterate_model({'model_id': model_id, 'prompt': prompt, 'anthropic_api_key': api_key})
                        except Exception as e:
                            result = {'success': False, 'error': f'Iteration failed: {str(e)}'}

                        if isinstance(result, dict) and result.get('success') and (result.get('step_file') or result.get('step_file_data')):
                            step_b64 = result.get('step_file') or result.get('step_file_data')
                            model_id = result.get('model_id')
                            planning_summary = result.get('planning_summary')
                            parameters = result.get('parameters')
                            
                            def import_on_main_thread():
                                """Import STEP file on main thread for iteration - no self references allowed"""
                                try:
                                    _app.log(f'CADAgent: STEP import starting - model_id: {model_id}, is_iteration: True')
                                    _app.log(f'CADAgent: STEP data length: {len(step_b64) if step_b64 else 0} chars')
                                    
                                    if not step_b64:
                                        _app.log('CADAgent: ERROR - No STEP file data provided')
                                        if _palette:
                                            _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': 'No STEP file data provided'}))
                                        return
                                    
                                    _app.log('CADAgent: Starting STEP file import...')
                                    
                                    # Decode base64 STEP file data
                                    try:
                                        step_binary = base64.b64decode(step_b64)
                                        _app.log(f'CADAgent: Decoded STEP file successfully, size: {len(step_binary)} bytes')
                                    except Exception as e:
                                        error_msg = f'Failed to decode STEP data: {e}'
                                        _app.log(f'CADAgent: ERROR - {error_msg}')
                                        if _palette:
                                            _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                        return
                                    
                                    # Save to temporary file
                                    temp_path = None
                                    try:
                                        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as temp_file:
                                            temp_file.write(step_binary)
                                            temp_path = temp_file.name
                                        _app.log(f'CADAgent: Saved STEP file to temporary path: {temp_path}')
                                    except Exception as e:
                                        error_msg = f'Failed to save temporary STEP file: {e}'
                                        _app.log(f'CADAgent: ERROR - {error_msg}')
                                        if _palette:
                                            _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                        return
                                    
                                    try:
                                        # Import into Fusion 360 using Fusion API
                                        _app.log('CADAgent: Getting Fusion API objects...')
                                        app = adsk.core.Application.get()
                                        import_manager = app.importManager
                                        design = app.activeProduct
                                        
                                        if not design:
                                            error_msg = 'No active design found in Fusion 360'
                                            _app.log(f'CADAgent: ERROR - {error_msg}')
                                            if _palette:
                                                _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                            return
                                        
                                        root_component = design.rootComponent
                                        _app.log(f'CADAgent: Got Fusion API objects - design: {design.name if hasattr(design, "name") else "unnamed"}')
                                        
                                        # For iterations, clear the existing model to prevent overlap/intersection
                                        _app.log('CADAgent: Iteration detected - clearing existing model before import')
                                        try:
                                            # Clear all bodies from the root component
                                            bodies_to_remove = []
                                            for body in root_component.bRepBodies:
                                                bodies_to_remove.append(body)
                                            
                                            for body in bodies_to_remove:
                                                body.deleteMe()
                                                
                                            # Clear all occurrences (child components)
                                            occurrences_to_remove = []
                                            for occurrence in root_component.occurrences:
                                                occurrences_to_remove.append(occurrence)
                                                
                                            for occurrence in occurrences_to_remove:
                                                occurrence.deleteMe()
                                                
                                            _app.log(f'CADAgent: Cleared {len(bodies_to_remove)} bodies and {len(occurrences_to_remove)} occurrences')
                                            
                                        except Exception as clear_error:
                                            _app.log(f'CADAgent: Warning - could not fully clear previous model: {clear_error}')
                                            # Continue with import even if clearing fails
                                        
                                        # Create import options
                                        _app.log('CADAgent: Creating STEP import options...')
                                        step_options = import_manager.createSTEPImportOptions(temp_path)
                                        step_options.isViewFit = True  # Fit the imported geometry to the view
                                        _app.log('CADAgent: Import options created successfully')
                                        
                                        # Import the file - Fusion 360 handles transactions automatically
                                        _app.log('CADAgent: Starting STEP file import to Fusion 360...')
                                        import_manager.importToTarget(step_options, root_component)
                                        _app.log('CADAgent: STEP file imported successfully into Fusion 360!')
                                        
                                        # Save the imported model if file manager is available
                                        if _file_manager:
                                            try:
                                                saved_path = _file_manager.save_imported_step(temp_path, model_id)
                                                _app.log(f'CADAgent: Saved imported model to: {saved_path}')
                                            except Exception as save_e:
                                                _app.log(f'CADAgent: Warning - could not save imported model: {save_e}')
                                        
                                        final = {
                                            'success': True,
                                            'imported': True,
                                            'model_id': model_id,
                                            'planning_summary': planning_summary,
                                            'parameters': parameters,
                                            'model_replaced': True
                                        }
                                        _app.log(f'CADAgent: Import completed successfully - returning result: {final}')
                                        
                                        if _palette:
                                            _palette.sendInfoToHTML('iteration_complete', json.dumps(final))
                                        
                                    finally:
                                        # Clean up temporary file
                                        if temp_path:
                                            try:
                                                _app.log(f'CADAgent: Cleaning up temporary file: {temp_path}')
                                                os.unlink(temp_path)
                                                _app.log('CADAgent: Temporary file cleaned up successfully')
                                            except Exception as cleanup_error:
                                                _app.log(f'CADAgent: Warning - could not delete temp file: {cleanup_error}')
                                        
                                except Exception as ie:
                                    error_msg = f'Import failed: {str(ie)}'
                                    _app.log(f'CADAgent: CRITICAL ERROR - {error_msg}')
                                    _app.log(f'CADAgent: Full traceback: {traceback.format_exc()}')
                                    if _palette:
                                        _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                            
                            # Signal HTML from background thread to trigger main thread iteration import
                            _app.log('CADAgent: Signaling HTML to trigger main thread iteration import...')
                            signal_success = signal_html_from_background('step_ready_for_import', {
                                'step_file_data': step_b64,
                                'model_id': model_id,
                                'is_iteration': True,
                                'planning_summary': planning_summary,
                                'parameters': parameters
                            })
                            if signal_success:
                                _app.log('CADAgent: HTML signaled successfully for iteration STEP import')
                            else:
                                _app.log('CADAgent: ERROR - Failed to signal HTML for iteration')
                                signal_html_from_background('error', {'success': False, 'message': 'Failed to signal HTML for iteration import'})
                        elif isinstance(result, dict) and result.get('large_file') and result.get('presigned_url'):
                            def download_and_import():
                                try:
                                    import base64, urllib.request
                                    with urllib.request.urlopen(result['presigned_url'], timeout=120) as resp:
                                        data = resp.read()
                                    step_b64_2 = base64.b64encode(data).decode('ascii')
                                    model_id = result.get('model_id')
                                    planning_summary = result.get('planning_summary')
                                    parameters = result.get('parameters')
                                    
                                    def import_main():
                                        """Import STEP file from presigned URL for iteration - no self references allowed"""
                                        try:
                                            _app.log(f'CADAgent: STEP iteration import starting from presigned URL - model_id: {model_id}')
                                            
                                            if not step_b64_2:
                                                _app.log('CADAgent: ERROR - No STEP file data from presigned URL')
                                                if _palette:
                                                    _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': 'No STEP file data from presigned URL'}))
                                                return
                                            
                                            # Decode base64 STEP file data
                                            try:
                                                step_binary = base64.b64decode(step_b64_2)
                                                _app.log(f'CADAgent: Decoded STEP file successfully, size: {len(step_binary)} bytes')
                                            except Exception as e:
                                                error_msg = f'Failed to decode STEP data: {e}'
                                                _app.log(f'CADAgent: ERROR - {error_msg}')
                                                if _palette:
                                                    _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                                return
                                            
                                            # Save to temporary file
                                            temp_path = None
                                            try:
                                                with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as temp_file:
                                                    temp_file.write(step_binary)
                                                    temp_path = temp_file.name
                                                _app.log(f'CADAgent: Saved STEP file to temporary path: {temp_path}')
                                            except Exception as e:
                                                error_msg = f'Failed to save temporary STEP file: {e}'
                                                _app.log(f'CADAgent: ERROR - {error_msg}')
                                                if _palette:
                                                    _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                                return
                                            
                                            try:
                                                # Import into Fusion 360 using Fusion API
                                                _app.log('CADAgent: Getting Fusion API objects...')
                                                app = adsk.core.Application.get()
                                                import_manager = app.importManager
                                                design = app.activeProduct
                                                
                                                if not design:
                                                    error_msg = 'No active design found in Fusion 360'
                                                    _app.log(f'CADAgent: ERROR - {error_msg}')
                                                    if _palette:
                                                        _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                                    return
                                                
                                                root_component = design.rootComponent
                                                _app.log(f'CADAgent: Got Fusion API objects - design: {design.name if hasattr(design, "name") else "unnamed"}')
                                                
                                                # For iterations, clear the existing model to prevent overlap/intersection
                                                _app.log('CADAgent: Iteration detected - clearing existing model before import')
                                                try:
                                                    # Clear all bodies from the root component
                                                    bodies_to_remove = []
                                                    for body in root_component.bRepBodies:
                                                        bodies_to_remove.append(body)
                                                    
                                                    for body in bodies_to_remove:
                                                        body.deleteMe()
                                                        
                                                    # Clear all occurrences (child components)
                                                    occurrences_to_remove = []
                                                    for occurrence in root_component.occurrences:
                                                        occurrences_to_remove.append(occurrence)
                                                        
                                                    for occurrence in occurrences_to_remove:
                                                        occurrence.deleteMe()
                                                        
                                                    _app.log(f'CADAgent: Cleared {len(bodies_to_remove)} bodies and {len(occurrences_to_remove)} occurrences')
                                                    
                                                except Exception as clear_error:
                                                    _app.log(f'CADAgent: Warning - could not fully clear previous model: {clear_error}')
                                                    # Continue with import even if clearing fails
                                                
                                                # Create import options
                                                _app.log('CADAgent: Creating STEP import options...')
                                                step_options = import_manager.createSTEPImportOptions(temp_path)
                                                step_options.isViewFit = True
                                                _app.log('CADAgent: Import options created successfully')
                                                
                                                # Import the file - Fusion 360 handles transactions automatically
                                                _app.log('CADAgent: Starting STEP file import to Fusion 360...')
                                                import_manager.importToTarget(step_options, root_component)
                                                _app.log('CADAgent: STEP file imported successfully into Fusion 360!')
                                                
                                                final = {
                                                    'success': True,
                                                    'imported': True,
                                                    'model_id': model_id,
                                                    'planning_summary': planning_summary,
                                                    'parameters': parameters,
                                                    'model_replaced': True
                                                }
                                                _app.log(f'CADAgent: Import completed successfully - returning result: {final}')
                                                
                                                if _palette:
                                                    _palette.sendInfoToHTML('iteration_complete', json.dumps(final))
                                                
                                            finally:
                                                # Clean up temporary file
                                                if temp_path:
                                                    try:
                                                        _app.log(f'CADAgent: Cleaning up temporary file: {temp_path}')
                                                        os.unlink(temp_path)
                                                        _app.log('CADAgent: Temporary file cleaned up successfully')
                                                    except Exception as cleanup_error:
                                                        _app.log(f'CADAgent: Warning - could not delete temp file: {cleanup_error}')
                                            
                                        except Exception as ie2:
                                            error_msg = f'Import failed: {str(ie2)}'
                                            _app.log(f'CADAgent: CRITICAL ERROR - {error_msg}')
                                            _app.log(f'CADAgent: Full traceback: {traceback.format_exc()}')
                                            if _palette:
                                                _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': error_msg}))
                                    
                                    # Signal HTML from background thread for large file iteration import
                                    signal_success = signal_html_from_background('step_ready_for_import', {
                                        'step_file_data': step_b64_2,
                                        'model_id': model_id,
                                        'is_iteration': True,
                                        'planning_summary': planning_summary,
                                        'parameters': parameters
                                    })
                                    if not signal_success:
                                        _app.log('CADAgent: ERROR - Failed to signal HTML for large file iteration import')
                                        signal_html_from_background('error', {'success': False, 'message': 'Failed to signal HTML for iteration import'})
                                except Exception as de:
                                    if _palette:
                                        _palette.sendInfoToHTML('error', json.dumps({'success': False, 'message': f'Download failed: {str(de)}'}))
                            t2 = threading.Thread(target=download_and_import, daemon=True)
                            t2.start()
                        else:
                            # Iteration failure path - signal error directly to HTML
                            raw_msg = (result or {}).get('error') or (result or {}).get('message') or 'Iteration failed'
                            user_msg = make_error_user_friendly(raw_msg)
                            signal_html_from_background('error', {'success': False, 'message': user_msg})
                    except Exception as e:
                        # Signal iteration error directly to HTML
                        error_msg = make_error_user_friendly(f'Iteration error: {str(e)}')
                        signal_html_from_background('error', {'success': False, 'message': error_msg})

                t = threading.Thread(target=iterate_in_background, daemon=True)
                t.start()
            
            elif action == 'store_api_key':
                # Store API key in memory cache
                result = self.handle_store_api_key(request_data)
                html_args.returnData = json.dumps(result)
            
            elif action == 'get_cached_api_key':
                # Retrieve cached API key
                result = self.handle_get_cached_api_key(request_data)
                html_args.returnData = json.dumps(result)

            elif action == 'clear_cached_api_key':
                # Clear all cached API keys
                result = self.handle_clear_cached_api_key(request_data)
                html_args.returnData = json.dumps(result)

            elif action == 'show_notification':
                # Show native Fusion 360 notification
                result = self.handle_show_notification(request_data)
                html_args.returnData = json.dumps(result)
            
            elif action == 'get_parameters':
                # Get parameters for a model using deployment backend API
                result = self.handle_get_parameters(request_data)
                html_args.returnData = json.dumps(result)
            
            elif action == 'update_parameters':
                # Update parameters for a model using deployment backend API
                result = self.handle_update_parameters(request_data)
                html_args.returnData = json.dumps(result)
                
            else:
                _app.log(f'CADAgent: Unknown action: {action}')
                html_args.returnData = json.dumps({
                    'success': False, 
                    'error': f'Unknown action: {action}',
                    'note': 'This Python backend only handles Fusion API operations'
                })
            
        except Exception as e:
            _app.log(f'CADAgent: Error in HTML event handler: {e}')
            _app.log(traceback.format_exc())
            try:
                html_args.returnData = json.dumps({
                    'success': False, 
                    'error': f'Handler error: {str(e)}'
                })
            except:
                pass
    
    def handle_step_import(self, request_data):
        """
        Import STEP file data into Fusion 360.
        This is a core Fusion API operation - exactly what Python should handle.
        """
        try:
            step_file_data = request_data.get('step_file_data', '')
            model_id = request_data.get('model_id', 'imported_model')
            is_iteration = request_data.get('is_iteration', False)
            
            _app.log(f'CADAgent: STEP import starting - model_id: {model_id}, is_iteration: {is_iteration}')
            _app.log(f'CADAgent: STEP data length: {len(step_file_data) if step_file_data else 0} chars')
            
            if not step_file_data:
                _app.log('CADAgent: ERROR - No STEP file data provided')
                return {'success': False, 'error': 'No STEP file data provided'}
            
            _app.log('CADAgent: Starting STEP file import...')
            
            # Decode base64 STEP file data
            try:
                step_binary = base64.b64decode(step_file_data)
                _app.log(f'CADAgent: Decoded STEP file successfully, size: {len(step_binary)} bytes')
            except Exception as e:
                error_msg = f'Failed to decode STEP data: {e}'
                _app.log(f'CADAgent: ERROR - {error_msg}')
                return {'success': False, 'error': error_msg}
            
            # Save to temporary file
            try:
                with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as temp_file:
                    temp_file.write(step_binary)
                    temp_path = temp_file.name
                
                _app.log(f'CADAgent: Saved STEP file to temporary path: {temp_path}')
            except Exception as e:
                error_msg = f'Failed to save temporary STEP file: {e}'
                _app.log(f'CADAgent: ERROR - {error_msg}')
                return {'success': False, 'error': error_msg}
            
            try:
                # Import into Fusion 360 using Fusion API
                _app.log('CADAgent: Getting Fusion API objects...')
                app = adsk.core.Application.get()
                import_manager = app.importManager
                design = app.activeProduct
                
                if not design:
                    error_msg = 'No active design found in Fusion 360'
                    _app.log(f'CADAgent: ERROR - {error_msg}')
                    return {'success': False, 'error': error_msg}
                
                root_component = design.rootComponent
                _app.log(f'CADAgent: Got Fusion API objects - design: {design.name if hasattr(design, "name") else "unnamed"}')
                
                # For iterations, clear the existing model to prevent overlap/intersection
                if is_iteration:
                    _app.log('CADAgent: Iteration detected - clearing existing model before import')
                    try:
                        # Clear all bodies from the root component
                        bodies_to_remove = []
                        for body in root_component.bRepBodies:
                            bodies_to_remove.append(body)
                        
                        for body in bodies_to_remove:
                            body.deleteMe()
                            
                        # Clear all occurrences (child components)
                        occurrences_to_remove = []
                        for occurrence in root_component.occurrences:
                            occurrences_to_remove.append(occurrence)
                            
                        for occurrence in occurrences_to_remove:
                            occurrence.deleteMe()
                            
                        _app.log(f'CADAgent: Cleared {len(bodies_to_remove)} bodies and {len(occurrences_to_remove)} occurrences')
                        
                    except Exception as clear_error:
                        _app.log(f'CADAgent: Warning - could not fully clear previous model: {clear_error}')
                        # Continue with import even if clearing fails
                
                # Create import options
                _app.log('CADAgent: Creating STEP import options...')
                step_options = import_manager.createSTEPImportOptions(temp_path)
                step_options.isViewFit = True  # Fit the imported geometry to the view
                _app.log('CADAgent: Import options created successfully')
                
                # Debug: Validate STEP file and design state before import
                _app.log('CADAgent: Starting STEP file import to Fusion 360...')
                _app.log(f'CADAgent: Debug - STEP file path: {temp_path}')
                _app.log(f'CADAgent: Debug - STEP file exists: {os.path.exists(temp_path)}')
                if os.path.exists(temp_path):
                    file_size = os.path.getsize(temp_path)
                    _app.log(f'CADAgent: Debug - STEP file size: {file_size} bytes')
                    
                    # Inspect STEP file content for debugging
                    try:
                        with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                            first_lines = []
                            for i in range(10):  # Read first 10 lines
                                line = f.readline()
                                if not line:
                                    break
                                first_lines.append(line.strip())
                            _app.log(f'CADAgent: Debug - STEP file starts with: {first_lines[:3]}')
                            
                            # Check for STEP file markers
                            has_iso_step = any('ISO-10303' in line for line in first_lines)
                            has_header = any('HEADER' in line for line in first_lines)
                            has_data = any('DATA' in line for line in first_lines)
                            _app.log(f'CADAgent: Debug - STEP markers - ISO: {has_iso_step}, HEADER: {has_header}, DATA: {has_data}')
                            
                    except Exception as read_error:
                        _app.log(f'CADAgent: Debug - Could not read STEP file: {read_error}')
                
                _app.log(f'CADAgent: Debug - Design name: {design.name if hasattr(design, "name") else "no_name_attr"}')
                _app.log(f'CADAgent: Debug - Design type: {type(design).__name__}')
                _app.log(f'CADAgent: Debug - Root component: {root_component}')
                _app.log(f'CADAgent: Debug - Import manager: {import_manager}')
                _app.log(f'CADAgent: Debug - Import options: {step_options}')
                
                # Try the import with detailed error handling and fallbacks
                try:
                    import_manager.importToTarget(step_options, root_component)
                    _app.log('CADAgent: STEP file imported successfully into Fusion 360!')
                except Exception as import_error:
                    _app.log(f'CADAgent: Import error type: {type(import_error).__name__}')
                    _app.log(f'CADAgent: Import error args: {import_error.args}')
                    _app.log(f'CADAgent: Import error str: {str(import_error)}')
                    
                    # Try alternative approaches for InternalValidationError
                    if 'InternalValidationError' in str(import_error):
                        _app.log('CADAgent: Attempting alternative import approaches...')
                        
                        # Approach 1: Try with different import options
                        try:
                            _app.log('CADAgent: Trying with modified import options...')
                            alt_options = import_manager.createSTEPImportOptions(temp_path)
                            alt_options.isViewFit = False  # Try without view fit
                            alt_options.isAutoUnfold = True if hasattr(alt_options, 'isAutoUnfold') else alt_options.isAutoUnfold
                            import_manager.importToTarget(alt_options, root_component)
                            _app.log('CADAgent: SUCCESS with alternative import options!')
                        except Exception as alt_error:
                            _app.log(f'CADAgent: Alternative approach 1 failed: {alt_error}')
                            
                            # Approach 2: Try creating a new document first
                            try:
                                _app.log('CADAgent: Trying to create new document context...')
                                docs = app.documents
                                new_doc = docs.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
                                if new_doc:
                                    new_design = new_doc.products.itemByProductType("DesignProductType")
                                    if new_design:
                                        new_root = new_design.rootComponent
                                        new_options = import_manager.createSTEPImportOptions(temp_path)
                                        import_manager.importToTarget(new_options, new_root)
                                        _app.log('CADAgent: SUCCESS with new document context!')
                                    else:
                                        raise Exception("Failed to get design from new document")
                                else:
                                    raise Exception("Failed to create new document")
                            except Exception as alt2_error:
                                _app.log(f'CADAgent: Alternative approach 2 failed: {alt2_error}')
                                raise import_error  # Re-raise original error
                    else:
                        raise import_error
                
                # Save the imported model if file manager is available
                if _file_manager:
                    try:
                        if not is_iteration:
                            _file_manager.start_new_session()
                        saved_path = _file_manager.save_imported_step(temp_path, model_id)
                        _app.log(f'CADAgent: Saved imported model to: {saved_path}')
                    except Exception as save_e:
                        _app.log(f'CADAgent: Warning - could not save imported model: {save_e}')
                
                success_result = {
                    'success': True,
                    'message': 'STEP file imported successfully' + (' (previous model replaced)' if is_iteration else ''),
                    'model_id': model_id,
                    'is_iteration': is_iteration,
                    'model_replaced': is_iteration
                }
                _app.log(f'CADAgent: Import completed successfully - returning result: {success_result}')
                return success_result
                
            finally:
                # Clean up temporary file
                try:
                    _app.log(f'CADAgent: Cleaning up temporary file: {temp_path}')
                    os.unlink(temp_path)
                    _app.log('CADAgent: Temporary file cleaned up successfully')
                except Exception as cleanup_error:
                    _app.log(f'CADAgent: Warning - could not delete temp file: {cleanup_error}')
            
        except Exception as e:
            error_msg = f'Import failed: {str(e)}'
            _app.log(f'CADAgent: CRITICAL ERROR - {error_msg}')
            _app.log(f'CADAgent: Full traceback: {traceback.format_exc()}')
            return {'success': False, 'error': error_msg}
    
    def handle_step_export(self, request_data):
        """
        Export current design as STEP file.
        Another core Fusion API operation.
        """
        try:
            _app.log('CADAgent: Starting STEP file export...')
            
            app = adsk.core.Application.get()
            design = app.activeProduct
            
            if not design:
                return {'success': False, 'error': 'No active design to export'}
            
            # Create temporary file for export
            with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Export as STEP using Fusion API
                export_manager = design.exportManager
                step_options = export_manager.createSTEPExportOptions(temp_path)
                export_manager.execute(step_options)
                
                # Read the exported file and encode as base64
                with open(temp_path, 'rb') as f:
                    step_data = f.read()
                
                step_b64 = base64.b64encode(step_data).decode('ascii')
                
                _app.log(f'CADAgent: STEP file exported successfully, size: {len(step_data)} bytes')
                
                return {
                    'success': True,
                    'message': 'STEP file exported successfully',
                    'step_file_data': step_b64,
                    'file_size': len(step_data)
                }
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
        except Exception as e:
            _app.log(f'CADAgent: Error exporting STEP file: {e}')
            return {'success': False, 'error': f'Export failed: {str(e)}'}
    
    def handle_get_design_info(self, request_data):
        """
        Get information about the current design.
        Uses Fusion API to extract design metadata.
        """
        try:
            app = adsk.core.Application.get()
            design = app.activeProduct
            
            if not design:
                return {'success': False, 'error': 'No active design'}
            
            # Get design information using Fusion API
            info = {
                'design_name': design.parentDocument.name if design.parentDocument else 'Untitled',
                'has_bodies': len(design.rootComponent.bRepBodies) > 0,
                'body_count': len(design.rootComponent.bRepBodies),
                'component_count': len(design.rootComponent.allOccurrences),
                'features_count': len(design.rootComponent.features.count) if hasattr(design.rootComponent.features, 'count') else 0
            }
            
            return {
                'success': True,
                'design_info': info
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Failed to get design info: {str(e)}'}
    
    def handle_validate_api_key(self, request_data):
        """
        DISABLED - API key validation is now local-only.
        Just stores the API key locally without any server validation.
        """
        try:
            api_key = request_data.get('api_key', '')
            if not api_key:
                return {'success': False, 'error': 'No API key provided'}
            
            # Store the API key locally without any server validation
            store_result = self.handle_store_api_key({'api_key': api_key})
            if store_result.get('success'):
                return {'success': True, 'message': 'API key saved locally (no server validation)'}
            else:
                return store_result
                
        except Exception as e:
            _app.log(f'CADAgent: Error storing API key: {e}')
            return {'success': False, 'error': f'Failed to store API key: {str(e)}'}
    
    def handle_generate_model(self, request_data):
        """
        Generate model using backend API.
        Following Fusion 360 best practices: Python handles all network requests.
        """
        try:
            prompt = (request_data.get('prompt', '') or '').strip()
            
            if not prompt:
                return {'success': False, 'error': 'No prompt provided'}
            
            # Backend configuration
            config_path = os.path.join(os.path.dirname(__file__), 'config')
            if config_path not in sys.path:
                sys.path.append(config_path)
            from settings import BACKEND_BASE_URL  # type: ignore[import-not-found]
            backend_url = BACKEND_BASE_URL
            generate_endpoint = f"{backend_url}/api/v1/direct/generate"
            
            # Prepare request data
            request_payload = {
                'prompt': prompt
            }
            
            # Create HTTP request
            data = json.dumps(request_payload).encode('utf-8')
            req = urllib.request.Request(
                generate_endpoint,
                data=data,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'User-Agent': 'Space-Fusion-Addon/1.0'
                },
                method='POST'
            )
            
            _app.log('CADAgent: Making generation request to backend...')
            
            # Make request with extended timeout (generation takes time)
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    if response.status == 200:
                        result_data = json.loads(response.read().decode('utf-8'))
                        _app.log('CADAgent: Model generation successful')
                        return {
                            'success': True,
                            'message': 'Model generated successfully',
                            **result_data
                        }
                    else:
                        error_data = response.read().decode('utf-8')
                        _app.log(f'CADAgent: Generation failed with HTTP {response.status}')
                        return {'success': False, 'error': f'Generation failed: HTTP {response.status}'}
                        
            except urllib.error.URLError as e:
                _app.log(f'CADAgent: Network error during generation: {e}')
                return {'success': False, 'error': f'Network error: {str(e)}'}
            except urllib.error.HTTPError as e:
                _app.log(f'CADAgent: HTTP error during generation: {e}')
                return {'success': False, 'error': f'HTTP error: {e.code} {e.reason}'}
                
        except Exception as e:
            _app.log(f'CADAgent: Error generating model: {e}')
            return {'success': False, 'error': f'Generation failed: {str(e)}'}
    
    def handle_iterate_model(self, request_data):
        """
        Iterate existing model using backend API.
        Following Fusion 360 best practices: Python handles all network requests.
        """
        try:
            model_id = request_data.get('model_id', '')
            prompt = request_data.get('prompt', '')
            
            if not model_id:
                return {'success': False, 'error': 'No model ID provided'}
            if not prompt:
                return {'success': False, 'error': 'No prompt provided'}
            
            # Backend configuration
            config_path = os.path.join(os.path.dirname(__file__), 'config')
            if config_path not in sys.path:
                sys.path.append(config_path)
            from settings import BACKEND_BASE_URL
            backend_url = BACKEND_BASE_URL
            iterate_endpoint = f"{backend_url}/api/v1/direct/iterate/{model_id}"
            
            # Prepare request data
            request_payload = {
                'prompt': prompt
            }
            
            # Create HTTP request
            data = json.dumps(request_payload).encode('utf-8')
            req = urllib.request.Request(
                iterate_endpoint,
                data=data,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'User-Agent': 'Space-Fusion-Addon/1.0'
                },
                method='POST'
            )
            
            _app.log(f'CADAgent: Making iteration request for model {model_id}...')
            
            # Make request with extended timeout
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    if response.status == 200:
                        result_data = json.loads(response.read().decode('utf-8'))
                        _app.log('CADAgent: Model iteration successful')
                        return {
                            'success': True,
                            'message': 'Model iterated successfully',
                            **result_data
                        }
                    else:
                        error_data = response.read().decode('utf-8')
                        _app.log(f'CADAgent: Iteration failed with HTTP {response.status}')
                        return {'success': False, 'error': f'Iteration failed: HTTP {response.status}'}
                        
            except urllib.error.URLError as e:
                _app.log(f'CADAgent: Network error during iteration: {e}')
                return {'success': False, 'error': f'Network error: {str(e)}'}
            except urllib.error.HTTPError as e:
                _app.log(f'CADAgent: HTTP error during iteration: {e}')
                return {'success': False, 'error': f'HTTP error: {e.code} {e.reason}'}
                
        except Exception as e:
            _app.log(f'CADAgent: Error iterating model: {e}')
            return {'success': False, 'error': f'Iteration failed: {str(e)}'}
    
    def handle_store_api_key(self, request_data):
        """
        Store API key both in memory cache and Fusion design attributes for persistence.
        This provides both session-level and cross-session persistence.
        """
        try:
            global _cached_api_key

            api_key = request_data.get('api_key', '')
            if not api_key:
                return {'success': False, 'error': 'No API key provided'}

            # Store in global cache (for current session)
            _cached_api_key = api_key
            _app.log('CADAgent: API key stored in memory cache')

            # Also store in Fusion design attributes for persistence across sessions
            persistent_stored = False
            if _fusion_utils:
                try:
                    persistent_stored = _fusion_utils.store_api_key(api_key)
                    if persistent_stored:
                        _app.log('CADAgent: API key stored persistently in Fusion design attributes')
                    else:
                        _app.log('CADAgent: Warning - failed to store API key persistently')
                except Exception as fusion_e:
                    _app.log(f'CADAgent: Warning - could not store API key in Fusion attributes: {fusion_e}')

            # Also update API client if available
            if _api_client:
                try:
                    _api_client.set_api_key(api_key)
                    _app.log('CADAgent: API key updated in API client')
                except Exception as client_e:
                    _app.log(f'CADAgent: Warning - could not update API client: {client_e}')

            success_msg = 'API key cached successfully'
            if persistent_stored:
                success_msg += ' (persistent storage enabled)'

            _app.log(f'CADAgent: {success_msg}')

            return {
                'success': True,
                'message': success_msg,
                'persistent_storage': persistent_stored
            }

        except Exception as e:
            _app.log(f'CADAgent: Error caching API key: {e}')
            return {'success': False, 'error': f'Failed to cache API key: {str(e)}'}
    
    def handle_get_cached_api_key(self, request_data):
        """
        Retrieve cached API key from all available sources:
        1. Memory cache (_cached_api_key)
        2. Fusion design attributes (persistent)
        3. API client cache
        4. Environment variable (.env file)
        """
        try:
            global _cached_api_key

            # Check memory cache first (fastest)
            if _cached_api_key:
                _app.log('CADAgent: Retrieved API key from memory cache')
                return {
                    'success': True,
                    'api_key': _cached_api_key,
                    'has_cached_key': True,
                    'source': 'memory'
                }

            # Check Fusion design attributes (persistent across sessions)
            if _fusion_utils:
                try:
                    fusion_key = _fusion_utils.retrieve_api_key()
                    if fusion_key:
                        _app.log('CADAgent: Retrieved API key from Fusion design attributes')
                        # Cache it in memory for this session
                        _cached_api_key = fusion_key
                        return {
                            'success': True,
                            'api_key': fusion_key,
                            'has_cached_key': True,
                            'source': 'fusion_attributes'
                        }
                except Exception as fusion_e:
                    _app.log(f'CADAgent: Could not retrieve from Fusion attributes: {fusion_e}')

            # Check API client cache
            if _api_client:
                try:
                    if hasattr(_api_client, 'api_key') and _api_client.api_key:
                        client_key = _api_client.api_key
                        _app.log('CADAgent: Retrieved API key from API client')
                        # Cache it in memory for this session
                        _cached_api_key = client_key
                        return {
                            'success': True,
                            'api_key': client_key,
                            'has_cached_key': True,
                            'source': 'api_client'
                        }
                except Exception as client_e:
                    _app.log(f'CADAgent: Could not retrieve from API client: {client_e}')

            # Check os.environ (populated from .env file at module load time)
            env_api_key = os.environ.get('ANTHROPIC_API_KEY', '')
            if env_api_key:
                _app.log('CADAgent: Retrieved API key from os.environ (loaded from .env at module init)')
                # Cache it in memory for future use
                _cached_api_key = env_api_key
                return {
                    'success': True,
                    'api_key': env_api_key,
                    'has_cached_key': True,
                    'source': 'environment'
                }

            # No key found anywhere
            _app.log('CADAgent: No API key found in any cache source')
            return {
                'success': True,
                'api_key': None,
                'has_cached_key': False,
                'source': 'none'
            }

        except Exception as e:
            _app.log(f'CADAgent: Error retrieving cached API key: {e}')
            return {'success': False, 'error': f'Failed to retrieve cached API key: {str(e)}'}

    def handle_clear_cached_api_key(self, request_data):
        """
        Clear all cached API keys from all storage locations:
        1. Memory cache (_cached_api_key)
        2. Fusion design attributes (persistent)
        3. API client cache
        """
        try:
            global _cached_api_key
            cleared_sources = []

            _app.log('CADAgent: Starting comprehensive API key clearing process...')

            # Clear memory cache
            if _cached_api_key:
                _cached_api_key = None
                cleared_sources.append('memory')
                _app.log('CADAgent: Cleared API key from memory cache')
            else:
                _app.log('CADAgent: Memory cache was already empty')

            # Clear Fusion design attributes
            if _fusion_utils:
                try:
                    _app.log('CADAgent: Attempting to clear API key from Fusion design attributes...')
                    result = _fusion_utils.clear_api_key()
                    if result:
                        cleared_sources.append('fusion_attributes')
                        _app.log('CADAgent: Successfully cleared API key from Fusion design attributes')
                    else:
                        _app.log('CADAgent: Fusion clear_api_key returned False - may not have existed')
                    
                    # Verify it's actually gone by trying to retrieve it
                    verification = _fusion_utils.retrieve_api_key()
                    if verification:
                        _app.log(f'CADAgent: WARNING - API key still exists after clear attempt! Length: {len(verification)}')
                    else:
                        _app.log('CADAgent: Verification confirmed - no API key found in Fusion attributes')
                        
                except Exception as fusion_e:
                    _app.log(f'CADAgent: Error clearing Fusion attributes: {fusion_e}')
                    _app.log(f'CADAgent: Full traceback: {traceback.format_exc()}')
            else:
                _app.log('CADAgent: No fusion_utils available for clearing')

            # Clear API client cache
            if _api_client:
                try:
                    _api_client.set_api_key(None)
                    cleared_sources.append('api_client')
                    _app.log('CADAgent: Cleared API key from API client')
                except Exception as client_e:
                    _app.log(f'CADAgent: Warning - could not clear API client: {client_e}')
            else:
                _app.log('CADAgent: No API client available for clearing')

            # Check if API key still exists from environment - this is expected and OK
            env_key_present = bool(os.environ.get('ANTHROPIC_API_KEY', ''))
            if env_key_present:
                _app.log('CADAgent: Note - API key still available from .env file (system-level, not cleared)')
            
            # Note: We don't clear environment variables as they're system-level configuration

            if cleared_sources:
                success_msg = f'Cached API key cleared from {len(cleared_sources)} sources ({", ".join(cleared_sources)})'
                if env_key_present:
                    success_msg += '. System environment key remains active.'
            else:
                success_msg = 'No cached keys found to clear'
                if env_key_present:
                    success_msg += ', but system environment key remains active'

            _app.log(f'CADAgent: {success_msg}')

            return {
                'success': True,
                'message': success_msg,
                'cleared_sources': cleared_sources,
                'env_key_present': env_key_present
            }

        except Exception as e:
            _app.log(f'CADAgent: Error clearing cached API key: {e}')
            _app.log(f'CADAgent: Full traceback: {traceback.format_exc()}')
            return {'success': False, 'error': f'Failed to clear cached API key: {str(e)}'}

    def handle_show_notification(self, request_data):
        """
        Show native Fusion 360 notification dialog.
        Following best practices: Python handles UI dialogs.
        """
        try:
            message = request_data.get('message', 'Notification')
            notification_type = request_data.get('type', 'info')  # 'info', 'warning', 'error'
            
            # Convert technical error messages to user-friendly ones
            if notification_type == 'error':
                message = make_error_user_friendly(message)
            
            # Use Fusion 360's native notification system
            app = adsk.core.Application.get()
            ui = app.userInterface
            
            # Map notification types to Fusion 360 message box types
            if notification_type == 'error':
                ui.messageBox(message, 'CADAgent AI CAD - Error', adsk.core.MessageBoxButtonTypes.OKButtonType, 
                             adsk.core.MessageBoxIconTypes.CriticalIconType)
            elif notification_type == 'warning':
                ui.messageBox(message, 'CADAgent AI CAD - Warning', adsk.core.MessageBoxButtonTypes.OKButtonType, 
                             adsk.core.MessageBoxIconTypes.WarningIconType)
            else:  # info or default
                ui.messageBox(message, 'CADAgent AI CAD', adsk.core.MessageBoxButtonTypes.OKButtonType, 
                             adsk.core.MessageBoxIconTypes.InformationIconType)
            
            _app.log(f'CADAgent: Showed native notification: {notification_type} - {message}')
            
            return {
                'success': True,
                'message': 'Notification displayed'
            }
            
        except Exception as e:
            _app.log(f'CADAgent: Error showing notification: {e}')
            return {'success': False, 'error': f'Failed to show notification: {str(e)}'}
    
    def handle_get_parameters(self, request_data):
        """
        Get parameters for a model using deployment backend API.
        Following best practices: Python handles network requests.
        """
        try:
            model_id = request_data.get('model_id')
            if not model_id:
                return {'success': False, 'error': 'model_id is required'}
            
            _app.log(f'CADAgent: Getting parameters for model: {model_id}')
            
            # Use API client to call deployment backend
            if _api_client is not None:
                result = _api_client.get_parameters(model_id)
                _app.log(f'CADAgent: Parameter retrieval result: {result.get("success", False)}')
                return result
            else:
                return {'success': False, 'error': 'API client not available'}
                
        except Exception as e:
            _app.log(f'CADAgent: Error getting parameters: {e}')
            return {'success': False, 'error': f'Failed to get parameters: {str(e)}'}
    
    def handle_update_parameters(self, request_data):
        """
        Update parameters for a model using deployment backend API.
        Following best practices: Python handles network requests.
        """
        try:
            model_id = request_data.get('model_id')
            updates = request_data.get('updates', [])
            
            if not model_id:
                return {'success': False, 'error': 'model_id is required'}
            if not updates:
                return {'success': False, 'error': 'updates array is required'}
            
            _app.log(f'CADAgent: Updating parameters for model: {model_id}, updates: {len(updates)}')
            
            # Use API client to call deployment backend
            if _api_client is not None:
                result = _api_client.update_parameters(model_id, updates)
                _app.log(f'CADAgent: Parameter update result: {result.get("success", False)}')
                return result
            else:
                return {'success': False, 'error': 'API client not available'}
                
        except Exception as e:
            _app.log(f'CADAgent: Error updating parameters: {e}')
            return {'success': False, 'error': f'Failed to update parameters: {str(e)}'}