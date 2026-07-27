# CADAgent

AI-powered CAD modeling assistant for Autodesk Fusion 360. Describe what you want in plain English, and CADAgent builds it.

This repository holds the whole product:

| Part | Where | What it does |
|---|---|---|
| Fusion 360 add-in | `mac/CADAgent/`, `win/CADAgent/` | Runs inside Fusion. Sends your request to the backend over a websocket and applies the returned CAD operations. |
| Backend | `backend/` | FastAPI service that turns natural-language requests into tool calls the add-in executes. Self-hosted: run it locally or on your own server. |
| Infra templates | `infra/`, `appspec.yml`, `scripts/` | Reference AWS deploy setup (GitHub Actions → S3 → CodeDeploy → EC2) from when the backend ran as a hosted service. |

## Install the add-in

Download the latest release:

- **Windows**: [CADAgent-Windows.zip](https://github.com/er-fo/CADAgent/releases/download/v1.0.4-win/CADAgent-Windows.zip)
- **macOS**: [CADAgent-macOS.zip](https://github.com/er-fo/CADAgent/releases/download/v1.0.4-mac/CADAgent-macOS.zip)

Copy the `CADAgent` folder to Fusion's add-ins directory:

- Windows: `%AppData%\Autodesk\Autodesk Fusion 360\API\AddIns`
- macOS: `~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/AddIns`

Restart Fusion 360, then run it from **Tools > Add-Ins > CADAgent > Run**. Type a request such as "Create a 5cm cube" and click **Execute**.

## Run the backend locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in the keys you need
python start_backend.py
```

The service listens on `ws://localhost:8000/ws/{session_id}` with a health check at `/health`. `.env.example` documents every variable; point the add-in at your backend via `BACKEND_HOST` in its `.env.cadagent`. The hosted backend at `ws.cadagentpro.com` is shut down — running your own backend is the supported path.

## Documentation

- `docs/backend/README.md` — production runbook (systemd, nginx, CodeDeploy).
- `docs/backend/deployment.md` — the CI/CD pipeline end to end.
- `mac/CADAgent/README.md` / `win/CADAgent/README.md` — add-in internals and the websocket protocol.

## Links

- [Website](https://cadagentpro.com)
- [Issues](https://github.com/er-fo/CADAgent/issues)
