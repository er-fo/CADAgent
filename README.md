# CADAgent

Transform 3D modeling from complex menus to simple conversations. Create and modify CAD models using plain English descriptions.

## What it does

Turn plain English into 3D models:

- "Create a mounting bracket with two holes"
- "Make a perforated tray, 30cm long with 20 holes"  
- "Build a sphere inside a cubic frame"

Models generate directly in Fusion 360.

## How it works

CADAgent connects natural language processing with Fusion 360's modeling engine. Describe what you want to build, and the AI interprets your requirements into precise CAD operations.

The add-in provides a chat interface within Fusion 360 where you can:
- Describe parts in plain English
- Iterate on designs through conversation
- Generate parametric models that update automatically
- Export standard CAD formats

## Requirements

- Fusion 360 (version 2.0.16000 or newer)
- Windows or macOS
- Internet connection for AI processing
- **Anthropic API key** (required for AI functionality)
  - Sign up at https://console.anthropic.com/
  - Generate API key at https://console.anthropic.com/account/keys
  - Typical usage costs: $0.01-0.10 per model generation

## Installation

### One-Click Installation (Recommended)
1. Download CADAgent from GitHub
2. Extract the ZIP file
3. **Double-click the installer for your platform:**
   - **Windows:** `Install-CADAgent-Windows.bat`
   - **macOS:** `Install-CADAgent-macOS.sh`
4. **Get Anthropic API key:** Visit https://console.anthropic.com/account/keys
5. Configure the API key in CADAgent's settings within Fusion 360
6. Start creating models with natural language!

### Manual Installation
See `installation_guide.md` for detailed manual setup instructions.

## Usage

1. Open Fusion 360
2. Launch CADAgent from the Add-Ins panel
3. Configure your Anthropic API key (first time only)
4. Type your part description in the chat interface
5. Review and refine the generated model
6. Save or export your design

## Examples

**Basic shapes:**
"Create a 50mm cube with rounded corners"

**Functional parts:**
"Design a phone stand with 15-degree angle and cable slot"

**Assemblies:**
"Build a simple hinge with two parts and mounting holes"


## Good to know

**Cost?**
- The software is currently free to use, i (the author) make no money off this beta, completely non-profit. – It is BYO API key. 

**Limitations**
- Currently limited to fillets, chamfers, boolean operations and rotation + movement. (parameters quickly changeable via UI) - Organic models will (although funny to look at) produce results below expectations.

## Support

For installation issues, check the installation guide. For modeling questions, describe your requirements as specifically as possible - the AI works best with clear, detailed descriptions. (or email me at eriknollett@gmail.com)

Questions? Contact – Answering everyone
- Email: eriknollett@gmail.com
- Email: hi@cadagentpro.com
- Website: https://cadagentpro.com
