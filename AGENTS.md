# Agent context — CADAgent monorepo

Layout:

- `apps/addin/mac/CADAgent/`, `apps/addin/win/CADAgent/` — the Fusion 360 add-in (Python, vendored deps per platform).
- `apps/backend/` — the FastAPI websocket backend the add-in talks to.
  - `backend/` the importable package, `tests/` the pytest suite, `ops/` operational scripts.
  - Backend-scoped config lives here too: `requirements*.txt`, `pytest.ini`, `.python-version`, `.env.example`.
- `infra/` — reference AWS deploy setup: `appspec.yml`, `hooks/`, `systemd/`, `nginx/`, `codedeploy/`, `iam/`, `monitoring/`.
- `supabase/` — auth and quota migrations plus edge functions.
- `docs/backend/` — backend runbook and design notes. `docs/backend/README.md` is the ops runbook.

Rules:

- Never commit secrets. `apps/backend/.env.example` documents the env contract; real values live outside git.
  The add-in's tracked `.env.cadagent` holds only the public Supabase URL and publishable key.
- Run backend tests from `apps/backend/` (`pytest -q`), not from the repo root — `pytest.ini` sets
  `testpaths = tests` and `pythonpath = .` so `from backend.…` and `from ops.…` resolve.
- Keep the two add-in trees (`apps/addin/mac/`, `apps/addin/win/`) in sync when you change shared code.
  They are currently byte-identical.
- Releases: `release-addin.yml` (manual) packages the add-in for both platforms.
  `backend-ci.yml` runs the backend suite, path-filtered to `apps/backend/**`.
  The hosted backend is retired, so there is no deploy workflow — `infra/` is reference material.
  Reviving CodeDeploy means placing `appspec.yml` at the deployment bundle root.
- Add a CODEOWNERS entry for any new top-level directory. Last matching pattern wins.
- Document non-trivial changes in `docs/`.
