"""
CADAgent Add-on File Manager
Handles STEP file import and session management in Fusion 360.
Following best practices for file operations.
"""

import os
import tempfile
import base64
from datetime import datetime
import threading
import adsk.core
import adsk.fusion

# Thread-safe logging function (follows Fusion 360 best practices)
def safe_log(message, app_instance=None):
    """
    Thread-safe logging that follows Fusion 360 best practices.
    Uses print() from background threads, app.log() only from main thread.
    """
    try:
        if threading.current_thread() is threading.main_thread():
            # Main thread: use Fusion API logging
            if app_instance:
                app_instance.log(message)
            else:
                adsk.core.Application.get().log(message)
        else:
            # Background thread: use print() (thread-safe)
            print(f"[BACKGROUND-FileManager] {message}")
    except Exception:
        # Fallback to print if anything fails
        print(f"[FALLBACK-FileManager] {message}")


class SpaceFileManager:
    """
    Manages file operations for the CADAgent add-on.
    Creates session folders and handles STEP file import.
    """
    
    def __init__(self):
        self.app = adsk.core.Application.get()
        self.current_session_folder = None
        self.current_iteration = 0
        
        # Create base sessions folder in user's Documents
        self._create_sessions_folder()
    
    def _create_sessions_folder(self):
        """Create the base sessions folder in user's documents."""
        try:
            # Get user's documents folder
            user_folder = os.path.expanduser('~')
            documents_folder = os.path.join(user_folder, 'Documents')
            
            # Create Space sessions folder
            self.sessions_base = os.path.join(documents_folder, 'Space_Sessions')
            if not os.path.exists(self.sessions_base):
                os.makedirs(self.sessions_base)
                
        except Exception as e:
            safe_log(f'Space: Failed to create sessions folder: {str(e)}', self.app)
            # Fallback to temp directory
            self.sessions_base = os.path.join(tempfile.gettempdir(), 'Space_Sessions')
            if not os.path.exists(self.sessions_base):
                os.makedirs(self.sessions_base)
    
    def start_new_session(self, session_name=None):
        """
        Start a new session with its own folder.
        
        Args:
            session_name (str, optional): Custom session name, otherwise uses timestamp
            
        Returns:
            str: Path to the session folder
        """
        if not session_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_name = f'session_{timestamp}'
        
        # Create session folder
        self.current_session_folder = os.path.join(self.sessions_base, session_name)
        
        try:
            if not os.path.exists(self.current_session_folder):
                os.makedirs(self.current_session_folder)
            
            # Reset iteration counter
            self.current_iteration = 0
            
            safe_log(f'Space: Started new session at {self.current_session_folder}', self.app)
            return self.current_session_folder
            
        except Exception as e:
            safe_log(f'Space: Failed to create session folder: {str(e)}', self.app)
            return None
    
    def get_next_iteration_name(self, base_name='model'):
        """
        Get the next iteration filename.
        
        Args:
            base_name (str): Base name for the files
            
        Returns:
            tuple: (step_filename, yaml_filename)
        """
        self.current_iteration += 1
        step_name = f'{base_name}{self.current_iteration}.step'
        yaml_name = f'{base_name}{self.current_iteration}.yaml'
        
        return step_name, yaml_name
    
    def save_step_file(self, step_data_b64, filename):
        """
        Save base64 encoded STEP file to current session folder.
        
        Args:
            step_data_b64 (str): Base64 encoded STEP file content
            filename (str): Filename to save as
            
        Returns:
            str: Full path to saved file, or None if failed
        """
        if not self.current_session_folder:
            safe_log('Space: No active session folder', self.app)
            return None
        
        try:
            # Decode base64 data
            step_data = base64.b64decode(step_data_b64)
            
            # Save to session folder
            file_path = os.path.join(self.current_session_folder, filename)
            
            with open(file_path, 'wb') as f:
                f.write(step_data)
            
            safe_log(f'Space: Saved STEP file to {file_path}', self.app)
            return file_path
            
        except Exception as e:
            safe_log(f'Space: Failed to save STEP file: {str(e)}', self.app)
            return None
    
    
    # Import into Fusion is handled centrally by SpaceFusionUtils on the main thread via custom events.
    
    def save_imported_step(self, temp_step_path, model_id):
        """
        Save an imported STEP file to the current session folder.
        
        Args:
            temp_step_path (str): Path to the temporary STEP file
            model_id (str): Model identifier for naming
            
        Returns:
            str: Path where the file was saved, or None if failed
        """
        try:
            if not self.current_session_folder:
                safe_log('Space: No active session for saving imported STEP file', self.app)
                return None
            
            # Create filename based on model_id and iteration
            filename = f"{model_id}_iteration_{self.current_iteration:03d}.step"
            file_path = os.path.join(self.current_session_folder, filename)
            
            # Copy the temporary file to session folder
            import shutil
            shutil.copy2(temp_step_path, file_path)
            
            safe_log(f'Space: Saved imported STEP file to: {file_path}', self.app)
            
            # Increment iteration counter
            self.current_iteration += 1
            
            return file_path
            
        except Exception as e:
            safe_log(f'Space: Failed to save imported STEP file: {str(e)}', self.app)
            return None
    
    def get_session_info(self):
        """
        Get information about the current session.
        
        Returns:
            dict: Session information
        """
        return {
            'session_folder': self.current_session_folder,
            'current_iteration': self.current_iteration,
            'sessions_base': self.sessions_base
        }
