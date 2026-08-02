# Monorepo merge — backend into CADAgent (2026-07-27)

## What was implemented

Merged `er-fo/cadagent-backend-legacy` (branch `main`, 61 commits, full history
preserved) into this repository with `git merge --allow-unrelated-histories`.
The backend now lives beside the add-in:

- Backend code, deploy pipeline, and infra land at repo root (`backend/`,
  `infra/`, `scripts/`, `appspec.yml`, `start_backend.py`, `.env.example`,
  requirements files, `supabase/`). No path in the CodeDeploy pipeline changed.
- The backend's ops README and all its docs moved to `docs/backend/` so the
  root `README.md` and existing `docs/` survived the merge.
- `.gitignore` is the union of both files, with `!.env.example` added so the
  env contract stays tracked.

## Why

One public repository ("CADAgent") should hold the whole product. The add-in
was already public; the backend was private. Merging into the add-in repo keeps
its stars, releases, and URL.

## Key decisions

- **Backend at root, add-in stays in `mac/`/`win/`**: both release pipelines
  (`release-addin.yml`, `deploy.yml`) keep their hardcoded paths untouched.
- **AWS account ID scrubbed**: `deploy.yml` now reads `DEPLOY_S3_BUCKET` and
  `AWS_DEPLOY_ROLE_ARN` from repository variables (same pattern as
  `prod-error-monitor.yml`). IAM templates and `docs/backend/deployment.md`
  use `<AWS_ACCOUNT_ID>`.
- **`deploy.yml` got a paths filter** so add-in-only pushes to `main` no
  longer trigger a backend deploy to EC2.
- **`AGENTS.md`/`CLAUDE.md` symlinks replaced with committed files**, per the
  SSOT policy's own rule for shared repositories.

## Setup notes / breaking changes

Before the first backend deploy from this repo, set two repository variables
in GitHub settings: `DEPLOY_S3_BUCKET` and `AWS_DEPLOY_ROLE_ARN`. The CodeDeploy
application still points at the old repo name; it deploys from S3, so no change
is needed there.
