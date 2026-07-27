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
- Prompt routing model: `ROUTER_MODEL`
  - default: `gpt-4.1-nano`
  - MiniMax M2.5: `minimax.minimax-m2.5`
  - MiniMax uses `AWS_BEARER_TOKEN_BEDROCK` plus the Bedrock OpenAI-compatible endpoint in `ROUTER_BEDROCK_BASE_URL`

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

## Correct CI/CD usage

Use this flow when you want production to match the latest backend code:

1. Work in a branch and run the full test suite locally with Python `3.11`.
2. Fast-forward or merge the approved changes onto `main`.
3. Push `main` to GitHub.
4. Let `.github/workflows/deploy.yml` run the tests, build the bundle, upload to S3, and create the CodeDeploy deployment.
5. Verify both:
   - the GitHub Actions run completed successfully
   - the AWS CodeDeploy deployment completed successfully
6. Verify service health after deploy:
   - public: `https://ws.cadagentpro.com/health`
   - host-local: `http://127.0.0.1:8001/health`

Recommended local verification:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

Important limits:

- The production deployment source of truth is **GitHub `main`**, not a local working tree and not the AWS filesystem.
- Do **not** patch production code directly on the EC2 host.
- The deploy bundle only ships runtime files:
  - `appspec.yml`
  - `backend/`
  - `requirements.txt`
  - `start_backend.py`
  - `scripts/`
- Repo-only reference artifacts such as `infra/`, `.env.example`, `README.md`, `docs/`, and `RECONCILIATION.md` are versioned in GitHub but are **not** copied to `/opt/backend-legacy` by the current pipeline.

## Deployment files

- `appspec.yml`
- `scripts/before_install.sh`
- `scripts/after_install.sh`
- `scripts/start_server.sh`
- `scripts/validate_service.sh`
- `.github/workflows/deploy.yml`

Deploy bundle currently does **not** include `infra/` templates or `.env.example`.
