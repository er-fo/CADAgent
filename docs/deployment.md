# Deployment Notes

## Production target

- Instance: `i-007470a13ac95c876`
- Service: `cadagent-backend-legacy.service`
- Working directory: `/opt/backend-legacy`
- Health check: `http://127.0.0.1:8001/health`

## Why the deploy scripts use `/opt/backend/.venv`

The current systemd unit for the legacy backend starts uvicorn from
`/opt/backend/.venv/bin/uvicorn` while using `/opt/backend-legacy` as the working
directory. The deployment scripts preserve that runtime contract so CI/CD matches the
server as it exists today.

If you later want to simplify this, change the systemd unit to use a dedicated virtual
environment under `/opt/backend-legacy/.venv` and update the deploy hooks accordingly.
