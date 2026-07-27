# CADAgent Legacy Backend CI/CD Runbook

Last updated: 2026-05-10

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

## Primary deployment workflow

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

## Deploy workflow details

Workflow: `.github/workflows/deploy.yml`

Triggers:

- Push to `main`
- Manual `workflow_dispatch`

GitHub permissions:

- `id-token: write`
- `contents: read`

AWS identity:

- Role: `arn:aws:iam::<AWS_ACCOUNT_ID>:role/github-actions-cadagent-backend-legacy-deploy-role`
- Region: `eu-north-1`

Deploy artifact:

- Bucket: `cadagent-backend-legacy-deploy-<AWS_ACCOUNT_ID>`
- Prefix: `backend-legacy`
- Object pattern: `backend-legacy/deployment-${GITHUB_SHA}.zip`

Deploy target:

- CodeDeploy application: `cadagent-backend-legacy`
- CodeDeploy deployment group: `cadagent-backend-legacy-ec2`

Workflow steps:

1. Checkout repository.
2. Install Python `3.11`.
3. Assume the GitHub deploy IAM role through OIDC.
4. Install `requirements-dev.txt`.
5. Run `pytest` with `PYTHONPATH=.`.
6. Build `deployment.zip`.
7. Upload the bundle to S3.
8. Create a CodeDeploy deployment with `file-exists-behavior OVERWRITE`.
9. Wait for `deployment-successful`.
10. Print CodeDeploy diagnostics if the deployment fails.

## Deploy bundle and CodeDeploy hooks

The deploy bundle contains only runtime files:

- `appspec.yml`
- `backend/`
- `requirements.txt`
- `start_backend.py`
- `scripts/`

CodeDeploy copies the bundle root to `/opt/backend-legacy` and runs these hooks:

- `BeforeInstall`: `scripts/before_install.sh`
  - Ensures `/opt/backend-legacy`, `/opt/backend-legacy/.deploy-backups`, and `/opt/deploy` exist.
  - Backs up the previous `backend/` and `requirements.txt` when present.
- `AfterInstall`: `scripts/after_install.sh`
  - Uses Python 3 to create or reuse `/opt/backend/.venv`.
  - Installs `requirements.txt` into that venv.
  - Removes Python cache files.
  - Sets `/opt/backend-legacy` ownership to `www-data:www-data`.
- `ApplicationStart`: `scripts/start_server.sh`
  - Runs `systemctl daemon-reload`.
  - Restarts `cadagent-backend-legacy.service`.
- `ValidateService`: `scripts/validate_service.sh`
  - Polls `http://127.0.0.1:8001/health` up to 30 times.
  - Dumps systemd status and recent journal logs when validation fails.

## Production error monitor

Workflow: `.github/workflows/prod-error-monitor.yml`

Purpose:

- Detect unusual production errors, tracebacks, service failures, and warning spikes.
- Open or update deduped GitHub issues for production anomalies.
- Include production updates from the 36 hours before the anomaly first appeared.
- Use conservative causal language such as `deployment-correlated regression candidate`.

Triggers:

- Scheduled every 15 minutes: `*/15 * * * *`
- Manual `workflow_dispatch`

Manual inputs:

- `dry_run`: run detection without creating or updating issues.
- `window_minutes`: trailing log window, default `30`.
- `lookback_hours`: production update lookback, default `36`.

GitHub permissions:

- `id-token: write`
- `contents: read`
- `actions: read`
- `issues: write`

AWS identity:

- Role variable: `AWS_MONITOR_ROLE_ARN`
- Current role: `arn:aws:iam::<AWS_ACCOUNT_ID>:role/github-actions-cadagent-backend-legacy-monitor-role`
- Inline policy: `cadagent-backend-legacy-monitor-read-policy`
- Policy source: `infra/iam/github-monitor-policy.json`

Configured log sources:

- Variable: `CLOUDWATCH_LOG_GROUPS`
- Current value: `/cadagent/ec2/syslog,/cadagent/nginx/error`

Monitor implementation:

- Script: `scripts/prod_error_monitor.py`
- Config: `infra/monitoring/prod-error-monitor.json`
- Dependencies: `requirements-monitor.txt`

Detection defaults:

- New error, traceback, critical log, or service failure: issue at 1 occurrence.
- New warning fingerprint: issue at 3 occurrences.
- Existing fingerprint spike: issue at `3x` baseline and at least `+3` events.
- Baseline: prior 24 hours plus 7-day same-hour comparison.
- Duplicate suppression: one open issue per normalized fingerprint.
- Repeat comments: at most once per hour unless severity materially increases.
- Evidence: redacted samples plus log source and deployment context, not raw log spam.

Dry-run command:

```bash
gh workflow run prod-error-monitor.yml \
  --repo er-fo/cadagent-backend-legacy \
  --ref main \
  -f dry_run=true \
  -f window_minutes=30 \
  -f lookback_hours=36
```

Recent validation on 2026-05-10:

- Dry-run workflow: `https://github.com/er-fo/cadagent-backend-legacy/actions/runs/25635507289`
- Result: `{"anomalies": 0, "dry_run": true}`

## CI/CD verification checklist

Before merging or pushing to `main`:

- Run targeted tests for changed behavior.
- Run `pytest -q` for backend-wide validation.
- Confirm the worktree is based on current `origin/main`.

After pushing to `main`:

- Confirm `.github/workflows/deploy.yml` succeeds.
- Confirm the matching CodeDeploy deployment succeeds.
- Confirm public health:

```bash
curl https://ws.cadagentpro.com/health
```

- For monitor changes, run a manual dry-run and verify the log includes:

```json
{
  "anomalies": 0,
  "dry_run": true
}
```

Use nonzero anomalies as an investigation signal, not as proof that the monitor is broken.

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

## Known limits

- `infra/`, `.env.example`, docs, and reference templates are versioned in GitHub but are not copied to `/opt/backend-legacy`.
- Live nginx, systemd, and environment configuration can diverge from the templates unless manually synchronized.
- The monitor currently reads `/cadagent/ec2/syslog` and `/cadagent/nginx/error`; nginx access logs are not part of anomaly detection.
- Scheduled monitoring depends on GitHub Actions availability, GitHub issue permissions, CloudWatch log ingestion, and the dedicated monitor IAM role.
- Node.js 20 deprecation warnings currently appear for GitHub-hosted actions; monitor and deploy runs still succeed as of 2026-05-10.
