#!/bin/bash
set -euo pipefail

SERVICE_NAME="cadagent-backend-legacy.service"
HEALTH_URL="http://127.0.0.1:8001/health"
MAX_ATTEMPTS=30
SLEEP_SECONDS=2

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if curl --fail --silent --show-error "$HEALTH_URL" >/dev/null; then
    echo "Health check passed on attempt ${attempt}/${MAX_ATTEMPTS}."
    exit 0
  fi

  if ! systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "Service '${SERVICE_NAME}' is not active during health validation (attempt ${attempt}/${MAX_ATTEMPTS})." >&2
    systemctl --no-pager -n 60 status "$SERVICE_NAME" || true
  fi

  sleep "$SLEEP_SECONDS"
done

echo "Health check failed after ${MAX_ATTEMPTS} attempts: ${HEALTH_URL}" >&2
systemctl --no-pager -n 80 status "$SERVICE_NAME" || true
journalctl --no-pager -u "$SERVICE_NAME" -n 120 || true
exit 1
