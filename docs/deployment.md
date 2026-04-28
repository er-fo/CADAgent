# Deployment Notes

## Production target

- Instance: `i-007470a13ac95c876`
- Service: `cadagent-backend-legacy.service`
- Working directory: `/opt/backend-legacy`
- Health check: `http://127.0.0.1:8001/health`

## Canonical backend source

- This repository (`cadagent-backend-legacy`) is the canonical source for backend app code and deploy scripts.
- The AWS filesystem snapshot is forensic only and should not be deployed or merged wholesale.
- Production code changes should flow through `main` -> GitHub Actions -> S3 -> CodeDeploy -> EC2.
- Do not treat local unpushed changes or direct EC2 edits as deployable source of truth.

## Correct operator workflow

1. Make and validate changes locally in a branch.
2. Run the full suite in Python `3.11`.
3. Move the approved change onto `main`.
4. Push `main`.
5. Watch `.github/workflows/deploy.yml` to completion.
6. Confirm the matching CodeDeploy deployment succeeded.
7. Verify backend health after deployment.

Reference commands:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements-dev.txt
pytest -q

gh run list --repo er-fo/cadagent-backend-legacy --workflow deploy.yml --limit 5
gh run watch --repo er-fo/cadagent-backend-legacy <run-id> --exit-status

aws deploy list-deployments \
  --application-name cadagent-backend-legacy \
  --deployment-group-name cadagent-backend-legacy-ec2 \
  --profile er-fo-CLI \
  --region eu-north-1

curl https://ws.cadagentpro.com/health
```

## Versioned but not enforced templates

- `infra/systemd/cadagent-backend-legacy.service`, `infra/nginx/cadagent-backend-legacy.conf`, and `.env.example`
  are versioned reference templates.
- Current `deploy.yml` deployment bundle does **not** ship `infra/` or `.env.example`, so live host config can diverge
  unless manually synchronized.

## Why the deploy scripts use `/opt/backend/.venv`

The current systemd unit for the legacy backend starts uvicorn from
`/opt/backend/.venv/bin/uvicorn` while using `/opt/backend-legacy` as the working
directory. The deployment scripts preserve that runtime contract so CI/CD matches the
server as it exists today.

If you later want to simplify this, change the systemd unit to use a dedicated virtual
environment under `/opt/backend-legacy/.venv` and update the deploy hooks accordingly.

## Runtime contracts

- Systemd unit template: `infra/systemd/cadagent-backend-legacy.service` (includes `EnvironmentFile=-/opt/backend-legacy/.env`)
- Nginx site template: `infra/nginx/cadagent-backend-legacy.conf` (minimal HTTP proxy template; add TLS/443/cert config for production)
- Example env contract to seed `/opt/backend-legacy/.env`: `.env.example`

## Python/runtime version

- CI uses Python 3.11 (`.github/workflows/deploy.yml`).
- Local/dev should align with Python 3.11 (`.python-version`).
- Production currently runs the venv at `/opt/backend/.venv`; confirm its Python
  version matches 3.11 during the next infra refresh.
