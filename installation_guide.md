# CADAgent Installation Guide (v0.1.34)

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

## One-Click Installation (Recommended)

After downloading and extracting CADAgent:

### Windows
1. **Navigate to the CADAgent folder**
2. **Double-click `Install-CADAgent-Windows.bat`**
3. **The installer will automatically:**
   - Locate your Fusion 360 add-ins directory
   - Copy all CADAgent files to the correct location
   - Handle any existing installations
4. **Done!** Proceed to "Enable the Add-in" section below

### macOS
1. **Navigate to the CADAgent folder**
2. **Double-click `Install-CADAgent-macOS.sh`**
   - If prompted about security, right-click the file and select "Open"
3. **The installer will automatically:**
   - Locate your Fusion 360 add-ins directory
   - Copy all CADAgent files to the correct location
   - Handle any existing installations
4. **Done!** Proceed to "Enable the Add-in" section below

## Manual Installation (Alternative Method)

If you prefer to install manually or the one-click installer doesn't work:

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
   - Make sure name of the top level folder is exactly "CADAgent" 

4. **Enable the add-in:**
   - Check the box next to "CADAgent"
   - Click "Run" to start the add-in

## Configuration

### Getting Your Anthropic API Key (Required)

CADAgent requires an **Anthropic API key** to function. This powers the AI that converts your natural language into CAD models.

1. **Create an Anthropic Account:**
   - Go to https://console.anthropic.com/
   - Sign up for a new account or log in if you already have one
   - **Note:** You'll need to add a payment method, but usage costs are typically very low (around $0.01-0.10 per model)

2. **Generate an API Key:**
   - Once logged in, navigate to https://console.anthropic.com/account/keys
   - Click "Create Key"
   - Give it a name like "CADAgent" 
   - Copy the generated key (starts with `sk-ant-api03-`)
   - **Important:** Save this key securely - you won't be able to see it again!

3. **Add Credits (if needed):**
   - New accounts may need to add credits to use the API
   - Go to https://console.anthropic.com/account/billing
   - Add a small amount ($5-10 is usually plenty to start)

### Configure CADAgent with Your API Key

1. **API Key Setup in Fusion 360:**
   - Launch CADAgent from the Fusion 360 Add-Ins panel
   - Click on the dropdown menu in the CADAgent chat UI "API Configuration"
   - Paste your Anthropic API key in the field
   - Press "Save API Key"
   - **Security Note:** The key is stored locally on your computer only

2. **Test the installation:**
   - Launch CADAgent from the Add-Ins panel
   - The chat interface should appear
   - Try a simple command like "create a cube"

## Troubleshooting

**Add-in doesn't appear:**
- Verify the folder structure is correct
- Restart Fusion 360
- Check that all files were copied properly

**API key caching & clearing:**
- The add-in now caches your key in-memory and in Fusion design attributes so you only enter it once.
- Click "Reset" in the green status pill (when a cached key is active) to wipe all stored copies.
- Optional: create a `.env` file alongside `CADAgent.py` with `ANTHROPIC_API_KEY=sk-...` for a fallback source.

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
