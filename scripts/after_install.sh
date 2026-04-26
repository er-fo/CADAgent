#!/bin/bash
set -euo pipefail

APP_DIR="/opt/backend-legacy"
RUNTIME_VENV="/opt/backend/.venv"

cd "$APP_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required on the target instance" >&2
  exit 1
fi

if [ ! -d "$RUNTIME_VENV" ]; then
  python3 -m venv "$RUNTIME_VENV"
fi

"$RUNTIME_VENV/bin/pip" install --upgrade pip
"$RUNTIME_VENV/bin/pip" install -r "$APP_DIR/requirements.txt"

find "$APP_DIR" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$APP_DIR" -name '*.pyc' -delete

chown -R www-data:www-data "$APP_DIR"
