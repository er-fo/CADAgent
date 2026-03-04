# CADAgent for macOS

## Installation

1. Download and extract `CADAgent.bundle`

2. Copy the entire `CADAgent.bundle` folder to Fusion 360's ApplicationPlugins directory:
   ```
   ~/Library/Application Support/Autodesk/ApplicationPlugins/CADAgent.bundle
   ```

   **Mac App Store version of Fusion?** Use this path instead:
   ```
   ~/Library/Containers/com.autodesk.mas.fusion360/Data/Library/Application Support/Autodesk/ApplicationPlugins/CADAgent.bundle
   ```

   **Alternative:** Copy just the `CADAgent` folder from inside the bundle to the add-ins directory:
   ```
   ~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/AddIns/CADAgent/
   ```

3. Restart Fusion 360

Fusion will detect the add-in automatically on launch.

## Verify Installation

1. Open Fusion 360
2. Go to **Tools** > **Add-Ins**
3. CADAgent should appear in the list
