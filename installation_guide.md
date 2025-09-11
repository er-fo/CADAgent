# CADAgent Installation Guide

This guide covers downloading CADAgent from GitHub and installing it as a Fusion 360 add-in.

## Prerequisites

- Fusion 360 installed and running
- Administrative access to your computer
- Internet connection

## Download from GitHub

1. Go to https://github.com/er-fo/CADAgent
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to a temporary location
5. Navigate to the extracted folder and locate the `CADAgent` directory

## Installation

### Windows

1. **Locate the Fusion 360 Add-ins folder:**
   ```
   %APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\
   ```
   
2. **Open the folder:**
   - Press `Windows + R`
   - Type `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\`
   - Press Enter

3. **Copy the add-in:**
   - Copy the entire `CADAgent` folder to this location
   - **Important:** The folder name must be exactly `CADAgent` (no extra characters)
   - The final path should be: `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\CADAgent\`

### macOS

1. **Locate the Fusion 360 Add-ins folder:**
   ```
   ~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/AddIns/
   ```

2. **Open the folder:**
   - Open Finder
   - Press `Cmd + Shift + G`
   - Type `~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/AddIns/`
   - Press Enter

3. **Copy the add-in:**
   - Copy the entire `CADAgent` folder to this location
   - **Important:** The folder name must be exactly `CADAgent` (no extra characters)
   - The final path should be: `~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/AddIns/CADAgent/`

## Enable the Add-in

1. **Open Fusion 360**
2. **Access Add-ins panel:**
   - Go to `Tools` > `Add-Ins`
   - Or press `Shift + S`

3. **Find CADAgent:**
   - Look for "CADAgent" in the list
   - If not visible, click "Refresh" or restart Fusion 360

4. **Enable the add-in:**
   - Check the box next to "CADAgent"
   - Click "Run" to start the add-in

## Configuration

1. **API Key Setup:**
   - Click on the dropdown menu in the CADAgent chat UI "API Configuration"
   - Add your Anthropic API key (get one from https://console.anthropic.com/account/keys)
   - Press "Save API Key"

2. **Test the installation:**
   - Launch CADAgent from the Add-Ins panel
   - The chat interface should appear
   - Try a simple command like "create a cube"

## Troubleshooting

**Add-in doesn't appear:**
- Verify the folder structure is correct
- Restart Fusion 360
- Check that all files were copied properly

**Connection errors:**
- Verify your API key is correct
- Check internet connection
- Ensure firewall allows Fusion 360 to access external services

**Permission issues (Windows):**
- Run Fusion 360 as administrator
- Check folder permissions in the AddIns directory

**Permission issues (macOS):**
- Grant Fusion 360 full disk access in System Preferences > Security & Privacy
- Check folder permissions using `ls -la` in Terminal

## File Structure

After installation, your add-in folder should contain:
```
CADAgent/
├── CADAgent.py
├── CADAgent.manifest
├── README.md
├── installation_guide.md
├── config/
│   └── settings.py
├── lib/
│   ├── __init__.py
│   ├── api_client.py
│   ├── file_manager.py
│   └── fusion_utils.py
└── ui/
    ├── chat.css
    ├── chat.html
    ├── chat.js
    ├── minimal.html
    └── test.html
```

The add-in is now ready to use. Launch it from the Fusion 360 Add-Ins panel and start creating models with natural language descriptions.

Questions? Contact – Answering everyone
- Email: eriknollett@gmail.com
- Email: hi@cadagentpro.com
- Website: cadagentpro.com