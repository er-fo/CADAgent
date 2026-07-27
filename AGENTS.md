# Agent context — CADAgent monorepo

This repository holds both halves of CADAgent:

- `mac/CADAgent/`, `win/CADAgent/` — the Fusion 360 add-in (Python, vendored deps per platform).
- `backend/` — the FastAPI websocket backend the add-in talks to.
- `infra/`, `appspec.yml`, `scripts/` — deploy pipeline (GitHub Actions, S3, CodeDeploy, EC2).
- `docs/backend/` — backend runbook and design notes. `docs/backend/README.md` is the ops runbook.

Rules:

- Never commit secrets. `.env.example` documents the env contract; real values live outside git.
- The add-in and backend are released separately: `release-addin.yml` (manual) packages the add-in; `deploy.yml` deploys the backend on pushes that touch backend paths.
- Keep the two add-in trees (`mac/`, `win/`) in sync when you change shared code.
- Document non-trivial changes in `docs/`.
