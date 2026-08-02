# CADAgent Legacy Backend

> **Status: the hosted backend is retired.** `ws.cadagentpro.com` is shut down and the
> `deploy.yml` workflow was removed on 2026-07-27. Everything below describing the EC2
> pipeline is kept as reference for anyone self-hosting or reviving it — it does not
> describe a running system. The supported path is running the backend yourself
> (see the root `README.md`).

This document covers the legacy CADAgent backend that served `ws.cadagentpro.com`
through `cadagent-backend-legacy.service` on the EC2 host.

## Runtime

- FastAPI + uvicorn
- Deployed to `/opt/backend-legacy`
- Managed by `systemd` service `cadagent-backend-legacy.service`
- Public traffic routed by nginx to `127.0.0.1:8001`
- Canonical app SSOT: `apps/backend/`
- Versioned reference templates (never auto-enforced): `infra/systemd/`, `infra/nginx/`, `apps/backend/.env.example`

## Local development

```bash
cd apps/backend
pyenv install 3.11.0  # optional
pyenv local 3.11.0
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
python start_backend.py
```

`pytest.ini` sets `testpaths = tests` and `pythonpath = .`, so run pytest from
`apps/backend/` — not from the repo root.

## Runtime configuration

- Example env contract: `apps/backend/.env.example`
- Runtime env file path used by the template unit: `/opt/backend-legacy/apps/backend/.env`
- Systemd unit template (manual apply): `infra/systemd/cadagent-backend-legacy.service`
- Nginx site template (manual apply): `infra/nginx/cadagent-backend-legacy.conf`
- Prompt routing model: `ROUTER_MODEL`
  - default: `gpt-4.1-nano`
  - MiniMax M2.5: `minimax.minimax-m2.5`
  - MiniMax uses `AWS_BEARER_TOKEN_BEDROCK` plus the Bedrock OpenAI-compatible endpoint in `ROUTER_BEDROCK_BASE_URL`

## WebSocket BYOK behavior

- `update_api_keys` is accepted pre-auth to support early BYOK updates.
- Payload compatibility: both `llm_api_keys` and `api_keys`.
- Server ack on success: `{"type":"api_keys_updated"}`.
- Other privileged paths (for example `execute_request`) are still rejected pre-auth.

## Deployment flow (historical)

```text
git push main
  -> GitHub Actions
  -> test + bundle
  -> upload artifact to S3
  -> create AWS CodeDeploy deployment
  -> EC2 installs dependencies and restarts cadagent-backend-legacy.service
```

This flow ran through `.github/workflows/deploy.yml`, which no longer exists.
Reviving it requires recreating that workflow and placing `appspec.yml` at the
root of the deployment bundle — it now lives at `infra/appspec.yml` in the repo.

Health endpoints, when a host is running:

- public: `https://ws.cadagentpro.com/health`
- host-local: `http://127.0.0.1:8001/health`

Bundle layout the hooks expect, relative to `/opt/backend-legacy`:

- `apps/backend/backend/`
- `apps/backend/requirements.txt`
- `apps/backend/start_backend.py`
- `infra/hooks/`

## Deployment files

- `infra/appspec.yml`
- `infra/hooks/before_install.sh`
- `infra/hooks/after_install.sh`
- `infra/hooks/start_server.sh`
- `infra/hooks/validate_service.sh`
