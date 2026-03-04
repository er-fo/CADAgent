# CADAgent - Fusion 360 Add-in

AI-powered CAD modeling assistant for Autodesk Fusion 360. Submit natural language requests via WebSocket backend to generate 3D models automatically.

## Quick Start

### Installation

1. **Locate Fusion 360 Add-ins Directory**:
   - **Windows**: `%APPDATA%\Autodesk\ApplicationPlugins`
   - **Mac**: `~/Library/Application Support/Autodesk/ApplicationPlugins`

2. **Copy CADAgent.bundle Folder**:
   ```bash
   # Mac/Linux
   cp -r CADAgent.bundle ~/Library/Application\ Support/Autodesk/ApplicationPlugins/

   # Windows (PowerShell)
   Copy-Item -Recurse CADAgent.bundle "$env:APPDATA\Autodesk\ApplicationPlugins\"
   ```

- Load production defaults automatically from `wss://ws.cadagentpro.com/ws/{session_id}`
- Override for local/dev before launching Fusion 360:
   ```bash
   export BACKEND_HOST=localhost
   export BACKEND_PORT=8000
   export BACKEND_USE_SSL=false   # keeps ws:// for local
   # Or provide a full template:
   # export BACKEND_URL=\"wss://ws.cadagentpro.com/ws/{session_id}\"
   export CADAGENT_DEBUG=false
   ```

### Loading the Add-in

1. Open Fusion 360
2. Go to **Tools** → **Add-Ins** → **Scripts and Add-Ins**
   - Do not use **Create Script or Add-In** (it creates sample projects like "Command Dialog Sample")
3. Select **CADAgent** from the list
4. Click **Run**
5. The **CADAgent palette** appears as a dockable panel

## Usage

The CADAgent palette provides a persistent, always-visible interface with:
- **Connection status indicator** (green = connected, red = disconnected)
- **Text input area** for CAD requests
- **Planning mode toggle** (review plans before execution)
- **Real-time activity log** showing agent progress
- **Execute and Clear buttons**

### Creating a Model

1. Ensure the backend is connected (green status indicator)
2. Type your CAD request in the text area
   - Example: "Create a cylinder 5cm tall with 2cm diameter"
3. Enable **Planning Mode** (recommended) to review the plan first
4. Click **Execute Request**
5. Watch the activity log for real-time progress
6. The AI will build your model step-by-step in Fusion 360

## Features

- **Natural Language Interface**: Describe what you want to model in plain English
- **Planning Mode**: Review AI-generated execution plans before proceeding
- **Real-time Feedback**: Connection status and progress updates
- **Thread-safe Operations**: Asynchronous WebSocket communication without UI freezing
- **Multi-line Input**: Detailed requests with examples and tooltips
- **Professional Error Handling**: Clear error messages with appropriate icons

## Architecture

```
CADAgent/
├── CADAgent.manifest      # Add-in metadata
├── CADAgent.py           # Main entry point (run/stop functions)
├── config.py             # Configuration settings
├── code_executor.py      # Executes Python code in Fusion context
├── websocket_client.py   # WebSocket connection management
├── commands/             # Individual command modules
│   ├── execute_command/  # Main CAD request submission UI
│   ├── settings_command/ # Backend configuration
│   └── status_command/   # Connection status display
├── lib/                  # Utility functions
│   ├── event_utils.py    # Event handler management
│   └── general_utils.py  # Common helpers
└── resources/            # Icons and assets

```

## Requirements

- Autodesk Fusion 360 (latest version)
- Python 3.x (included with Fusion 360)
- Running backend server (WebSocket endpoint)

**Note**: The `websockets` library is bundled with this add-in - no external dependencies required!

## Troubleshooting

**Connection Failed**:
- Verify backend server is running (`wss://ws.cadagentpro.com/health` for prod)
- Check backend env overrides (BACKEND_HOST / BACKEND_PORT / BACKEND_USE_SSL / BACKEND_URL)
- Review logs in Fusion's Text Commands window

**Add-in Won't Load**:
- Ensure all files are in the correct directory
- Check that the `lib/websockets/` folder exists with bundled library
- Enable debug mode: `export CADAGENT_DEBUG=true`

**"ModuleNotFoundError: No module named 'websockets'"**:
- The bundled websockets library may be missing from `lib/websockets/`
- Re-copy the entire CADAgent folder to ensure all files are present

## Development

Enable debug mode for verbose logging:
```bash
export CADAGENT_DEBUG=true
```

Logs appear in:
- Fusion 360 Text Commands console
- System console (if launched from terminal)

## License

See project root for license information.
