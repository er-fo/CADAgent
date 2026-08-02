# CADAgent backend

FastAPI websocket service that turns natural-language CAD requests into tool calls
the Fusion add-in executes.

## Layout

| Path | What |
|---|---|
| `backend/` | The importable Python package (`from backend.…`). Contains `ir/` and `backends/`. |
| `tests/` | Pytest suite. |
| `ops/` | Operational scripts: Brevo user sync, production error monitor. |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in the keys you need
python start_backend.py
```

Listens on `ws://localhost:8000/ws/{session_id}`, health check at `/health`.

## Test it

```bash
pip install -r requirements-dev.txt
pytest -q
```

Run pytest from this directory, not the repo root. `pytest.ini` sets
`testpaths = tests` and `pythonpath = .`, which is what makes `from backend.…`
and `from ops.…` resolve.

## Deployment

The hosted backend is retired. `infra/` at the repo root keeps the AWS
CodeDeploy/EC2 setup as reference — see `docs/backend/deployment.md`.
