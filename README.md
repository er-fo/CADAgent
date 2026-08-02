# CADAgent

AI-powered CAD modeling assistant for Autodesk Fusion 360. Describe what you want in plain English, and CADAgent builds it.

This repository holds the whole product:

| Part | Where | What it does |
|---|---|---|
| Fusion 360 add-in | `apps/addin/mac/`, `apps/addin/win/` | Runs inside Fusion. Sends your request to the backend over a websocket and applies the returned CAD operations. |
| Backend | `apps/backend/` | FastAPI service that turns natural-language requests into tool calls the add-in executes. Self-hosted: run it locally or on your own server. |
| Infra templates | `infra/` | Reference AWS deploy setup (GitHub Actions → S3 → CodeDeploy → EC2) from when the backend ran as a hosted service. |
| Supabase | `supabase/` | Auth and quota migrations plus edge functions. |

```
apps/
  addin/     mac/CADAgent/, win/CADAgent/   Fusion add-in (vendored deps per platform)
  backend/   backend/  tests/  ops/         FastAPI service, its tests, and ops scripts
infra/       appspec.yml, hooks/, systemd/, nginx/, codedeploy/, iam/, monitoring/
supabase/    migrations/, functions/
docs/
```

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
cd apps/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in the keys you need
python start_backend.py
```

Run the tests from the same directory:

```bash
pip install -r requirements-dev.txt
pytest -q
```

The service listens on `ws://localhost:8000/ws/{session_id}` with a health check at `/health`. `.env.example` documents every variable; point the add-in at your backend via `BACKEND_HOST` in its `.env.cadagent`. The hosted backend at `ws.cadagentpro.com` is shut down — running your own backend is the supported path.

## Documentation

- `docs/backend/README.md` — production runbook (systemd, nginx, CodeDeploy).
- `docs/backend/deployment.md` — the CI/CD pipeline end to end.
- `apps/addin/mac/CADAgent/README.md` / `apps/addin/win/CADAgent/README.md` — add-in internals and the websocket protocol.

## Links

- [Website](https://cadagentpro.com)
- [Issues](https://github.com/er-fo/CADAgent/issues)
