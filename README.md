# CADAgent Legacy Backend

This repository tracks the production legacy CADAgent backend that currently serves
`ws.cadagentpro.com` through `cadagent-backend-legacy.service` on the EC2 host.

## Runtime

- FastAPI + uvicorn
- Deployed to `/opt/backend-legacy`
- Managed by `systemd` service `cadagent-backend-legacy.service`
- Public traffic routed by nginx to `127.0.0.1:8001`
- Canonical app SSOT: `backend/`, `requirements*.txt`, deploy hooks/workflow in this repo
- Versioned reference templates (not auto-enforced by deploy.yml): `infra/systemd/`, `infra/nginx/`, `.env.example`

## Local development

```bash
pyenv install 3.11.0  # optional
pyenv local 3.11.0
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
python start_backend.py
```

## Runtime configuration

- Example env contract: `.env.example`
- Runtime env file path used by the template unit: `/opt/backend-legacy/.env`
- Systemd unit template (manual apply): `infra/systemd/cadagent-backend-legacy.service`
- Nginx site template (manual apply): `infra/nginx/cadagent-backend-legacy.conf`

## WebSocket BYOK behavior

- `update_api_keys` is accepted pre-auth to support early BYOK updates.
- Payload compatibility: both `llm_api_keys` and `api_keys`.
- Server ack on success: `{"type":"api_keys_updated"}`.
- Other privileged paths (for example `execute_request`) are still rejected pre-auth.

## Deployment flow

```text
git push main
  -> GitHub Actions
  -> test + bundle
  -> upload artifact to S3
  -> create AWS CodeDeploy deployment
  -> EC2 installs dependencies and restarts cadagent-backend-legacy.service
```

## Deployment files

- `appspec.yml`
- `scripts/before_install.sh`
- `scripts/after_install.sh`
- `scripts/start_server.sh`
- `scripts/validate_service.sh`
- `.github/workflows/deploy.yml`

Deploy bundle currently does **not** include `infra/` templates or `.env.example`.
