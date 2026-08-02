# Backend Reconciliation Report

Date: 2026-04-28

## Scope

Reconciled `cadagent-backend-legacy` against forensic evidence from `backend-from-aws`, while keeping this repository as the canonical source for application behavior.

## Decisions implemented

### WebSocket BYOK path (`update_api_keys`)

- Added behavior-level WebSocket tests on `/ws/{session_id}` in `apps/backend/backend/test_byok_session_keys.py`.
- Covered both payload keys:
  - `llm_api_keys`
  - `api_keys` (alias compatibility)
- Verified sanitization and storage via `ConnectionManager.get_llm_api_keys(...)`.
- Verified pre-auth policy:
  - `update_api_keys` is allowed pre-auth.
  - `execute_request` is still rejected pre-auth with `authentication_error` and close code `1008`.
- Added ack emission in `apps/backend/backend/main.py`:
  - `{"type":"api_keys_updated"}`

### BYOK test naming/coverage cleanup

- Replaced misleading unit-only naming with explicit behavior:
  - `test_connection_manager_set_llm_api_keys_sanitizes_and_cleans_up`
  - `test_websocket_update_api_keys_llm_payload_pre_auth_and_execute_rejected`
  - `test_websocket_update_api_keys_api_keys_alias_pre_auth_stores_sanitized_keys`
- Kept one focused unit-level sanitization test and added real WebSocket-path tests.

### Trio support model resolution

- Chosen model: **asyncio-only test execution is intentional** for this backend.
- Why: backend runtime and manager internals are asyncio-native (`asyncio.Queue`, FastAPI/uvicorn flow).
- Change: constrained AnyIO backend in `apps/backend/backend/test_reasoning_context.py`:
  - `anyio_backend` fixture now returns `"asyncio"`.
- `requirements-dev.txt` intentionally remains without `trio`.

### SSOT wording and infra template truthfulness

- Clarified in `README.md` and `docs/deployment.md`:
  - Canonical SSOT = backend app code + deploy workflow/scripts in this repo.
  - `infra/systemd`, `infra/nginx`, `.env.example` are versioned templates/reference artifacts.
  - Current `deploy.yml` bundle does **not** ship `infra/` or `.env.example`.

### Runtime env contract

- Connected `.env.example` to runtime contract:
  - Added `EnvironmentFile=-/opt/backend-legacy/.env` to `infra/systemd/cadagent-backend-legacy.service`.
  - Documented that `.env.example` seeds `/opt/backend-legacy/.env`.

### Nginx template status

- Added header in `infra/nginx/cadagent-backend-legacy.conf` clarifying:
  - Minimal/reference-only template.
  - Not auto-applied by deploy workflow.
  - Production requires explicit TLS (`443` + cert/key config).

## Test reporting (honest and reproducible)

### Verified locally on this machine

#### Default shell environment

Commands:

```bash
python3 --version
python3 -m pytest -q backend/test_byok_session_keys.py
python3 -m pytest -q
```

Observed:

- Python `3.9.6`
- `apps/backend/backend/test_byok_session_keys.py`: `3 passed`
- Full suite: `125 passed, 2 xfailed`

#### Managed project environment used for verification

Commands:

```bash
../.venv312/bin/python --version
../.venv312/bin/pytest -q backend/test_byok_session_keys.py
../.venv312/bin/pytest -q
```

Observed:

- Python `3.12.13`
- `apps/backend/backend/test_byok_session_keys.py`: `3 passed`
- Full suite: `125 passed, 2 xfailed`
- Warnings: `11` third-party deprecation warnings

Difference explained:

- Both verified environments now pass, including the dedicated WebSocket BYOK tests.
- The default shell uses system Python `3.9.6`.
- The managed project environment available during this reconciliation was `../.venv312` on Python `3.12.13`.

#### Managed Python 3.11 verification environment

Commands:

```bash
./.venv311/bin/python --version
./.venv311/bin/pytest -q backend/test_byok_session_keys.py
./.venv311/bin/pytest -q
```

Observed:

- Python `3.11.15`
- `apps/backend/backend/test_byok_session_keys.py`: `3 passed`
- Full suite: `125 passed, 2 xfailed`
- Warnings: `11` third-party deprecation warnings

### Python 3.11 alignment status

- `.python-version` pins `3.11` and CI uses Python `3.11`.
- Local verification now includes a full Python `3.11` run in `./.venv311`.
- Python `3.11` is therefore both the declared compatibility contract and a locally verified test environment for this reconciliation.

## What remains non-SSOT

- Live nginx/TLS configuration is not enforced from this repo today.
- Live systemd unit on host may diverge from `infra/systemd/cadagent-backend-legacy.service`.
- Live environment-variable source of truth (host `.env` / secret distribution path) is not proven versioned/enforced by current deploy flow.
