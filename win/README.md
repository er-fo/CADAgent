# CADAgent for Windows

## Installation

1. Download and extract `CADAgent.bundle`

2. Copy the `Contents` folder from inside `CADAgent.bundle` to the Fusion 360 add-ins directory, renaming it to `CADAgent`:
   ```
   %appdata%\Autodesk\Autodesk Fusion 360\API\AddIns\CADAgent\
   ```

   Full path example:
   ```
   C:\Users\YourName\AppData\Roaming\Autodesk\Autodesk Fusion 360\API\AddIns\CADAgent\
   ```

   The folder should contain `CADAgent.py`, `CADAgent.manifest`, and the other plugin files directly (not nested inside another folder).

3. Restart Fusion 360

Fusion will detect the add-in automatically on launch.

## Verify Installation

1. Open Fusion 360
2. Go to **Tools** > **Add-Ins**
3. CADAgent should appear in the list
