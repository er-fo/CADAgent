"""
CADAgent Add-on Fusion 360 Utilities
Helper functions for working with Fusion 360 API.
"""

import adsk.core
import adsk.fusion


class SpaceFusionUtils:
    """
    Utility functions for Fusion 360 operations.
    Following best practices for API usage.
    """
    
    def __init__(self):
        self.app = adsk.core.Application.get()
        self.ui = self.app.userInterface
    
    def get_active_design(self):
        """
        Get the currently active design.
        
        Returns:
            adsk.fusion.Design: Active design or None
        """
        try:
            product = self.app.activeProduct
            if product and hasattr(product, 'designType'):
                return product
            return None
        except:
            return None
    
    def get_root_component(self):
        """
        Get the root component of the active design.
        
        Returns:
            adsk.fusion.Component: Root component or None
        """
        design = self.get_active_design()
        if design:
            return design.rootComponent
        return None
    
    def clear_existing_bodies(self):
        """
        Clear existing bodies from the design.
        Used before importing new geometry.
        
        Returns:
            bool: True if successful
        """
        try:
            root_comp = self.get_root_component()
            if not root_comp:
                return False
            
            # Remove all bodies
            bodies_to_remove = []
            for i in range(root_comp.bRepBodies.count):
                body = root_comp.bRepBodies.item(i)
                if not body.isTransient:
                    bodies_to_remove.append(body)
            
            for body in bodies_to_remove:
                try:
                    body.deleteMe()
                except:
                    # Continue if deletion fails
                    pass
            
            return True
            
        except Exception as e:
            self.app.log(f'CADAgent: Failed to clear bodies: {str(e)}')
            return False
    
    def import_step_file(self, step_file_path):
        """
        Import a STEP file into the current design.
        
        Args:
            step_file_path (str): Path to STEP file
            
        Returns:
            bool: True if successful
        """
        try:
            root_comp = self.get_root_component()
            if not root_comp:
                return False
            
            # Create import manager and options
            import_manager = self.app.importManager
            step_options = import_manager.createSTEPImportOptions(step_file_path)
            
            # Configure import options
            step_options.isViewFit = True
            
            # Execute import - Fusion 360 handles transactions automatically
            import_manager.importToTarget(step_options, root_comp)
            
            # Refresh viewport
            self.app.activeViewport.refresh()
            
            return True
            
        except Exception as e:
            self.app.log(f'CADAgent: Failed to import STEP file: {str(e)}')
            return False
    
    def fit_view(self):
        """Fit the view to show all geometry."""
        try:
            self.app.activeViewport.fit()
        except:
            pass
    
    def show_message(self, title, message):
        """
        Show a message box to the user.
        
        Args:
            title (str): Message title
            message (str): Message content
        """
        try:
            self.ui.messageBox(message, title)
        except:
            # Fallback to logging
            self.app.log(f'{title}: {message}')
