#!/bin/bash
set -euo pipefail

curl --fail --silent --show-error http://127.0.0.1:8001/health >/dev/null
