#!/bin/bash
set -euo pipefail

install -d -m 755 /opt/backend-legacy
install -d -m 755 /opt/backend-legacy/.deploy-backups
install -d -m 755 /opt/deploy

if [ -d /opt/backend-legacy/backend ]; then
  backup_path="/opt/backend-legacy/.deploy-backups/backup-$(date +%Y%m%d%H%M%S)"
  mkdir -p "$backup_path"
  cp -a /opt/backend-legacy/backend "$backup_path/"
  if [ -f /opt/backend-legacy/requirements.txt ]; then
    cp -a /opt/backend-legacy/requirements.txt "$backup_path/"
  fi
fi
