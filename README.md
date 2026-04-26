# CADAgent Legacy Backend

This repository tracks the production legacy CADAgent backend that currently serves
`ws.cadagentpro.com` through `cadagent-backend-legacy.service` on the EC2 host.

## Runtime

- FastAPI + uvicorn
- Deployed to `/opt/backend-legacy`
- Managed by `systemd` service `cadagent-backend-legacy.service`
- Public traffic routed by nginx to `127.0.0.1:8001`

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
python start_backend.py
```

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
