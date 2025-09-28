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

    def store_api_key(self, api_key):
        """
        Store API key as design attribute for persistence across sessions.

        Args:
            api_key (str): The Anthropic API key to store

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            design = self.get_active_design()
            if not design:
                self.app.log('CADAgent: No active design to store API key')
                return False

            # Remove existing attribute if present
            try:
                existing = design.attributes.itemByName("CADAgent", "anthropic_api_key")
                if existing:
                    existing.deleteMe()
            except:
                # Attribute doesn't exist yet, which is fine
                pass

            # Store the API key as a design attribute
            design.attributes.add("CADAgent", "anthropic_api_key", api_key)
            self.app.log('CADAgent: API key stored successfully')
            return True

        except Exception as e:
            self.app.log(f'CADAgent: Failed to store API key: {str(e)}')
            return False

    def retrieve_api_key(self):
        """
        Retrieve cached API key from design attributes.

        Returns:
            str: The stored API key or None if not found
        """
        try:
            self.app.log('CADAgent: Starting API key retrieval...')
            design = self.get_active_design()
            if not design:
                self.app.log('CADAgent: No active design found - cannot retrieve cached API key')
                return None

            self.app.log('CADAgent: Active design found, checking for cached API key attribute...')

            # Try to get the stored attribute
            try:
                attr = design.attributes.itemByName("CADAgent", "anthropic_api_key")
                if attr:
                    self.app.log(f'CADAgent: Retrieved cached API key, length={len(attr.value) if attr.value else 0}')
                    return attr.value
                else:
                    self.app.log('CADAgent: No cached API key attribute found')
                    return None
            except Exception as attr_e:
                self.app.log(f'CADAgent: Error accessing attribute: {str(attr_e)}')
                return None

        except Exception as e:
            self.app.log(f'CADAgent: Failed to retrieve API key: {type(e).__name__}: {str(e)}')
            import traceback
            self.app.log(f'CADAgent: Traceback: {traceback.format_exc()}')
            return None

    def clear_api_key(self):
        """
        Clear stored API key from design attributes for security.

        Returns:
            bool: True if successful or key didn't exist, False on error
        """
        try:
            design = self.get_active_design()
            if not design:
                return True  # No design means no key to clear

            # Try to remove the attribute
            try:
                existing = design.attributes.itemByName("CADAgent", "anthropic_api_key")
                if existing:
                    existing.deleteMe()
                    self.app.log('CADAgent: API key cleared successfully')
                else:
                    self.app.log('CADAgent: No API key found to clear')
                return True
            except:
                # Attribute doesn't exist, which means it's already "cleared"
                return True

        except Exception as e:
            self.app.log(f'CADAgent: Failed to clear API key: {str(e)}')
            return False
