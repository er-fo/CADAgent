# Brevo Users List Reconciliation

## What Changed

- Added `apps/backend/ops/sync_brevo_users.py` to reconcile confirmed Supabase Auth users into the Brevo operational users list.
- Added `.github/workflows/sync-brevo-users.yml` for scheduled reconciliation and explicit manual dry-run/apply runs.
- Added unit tests for email normalization, reconciliation planning, and destructive-change guardrails.

## Why

CADAgent has two different contact populations:

- Waitlisters with marketing consent, protected as Brevo list `2`.
- Product users from Supabase Auth, mirrored for operational/product messaging in Brevo list `5`.

The sync must keep the users list current without treating users as marketing subscribers.

## Safety Decisions

- The users list ID is required explicitly.
- The job refuses to target the protected waitlist/marketing list ID.
- Local runs default to dry-run.
- Scheduled GitHub Action runs apply automatically against list `5`, bounded by the removal cap.
- Apply runs remove contacts only from the users list, never globally from Brevo.
- Removal count is capped by `--max-removals`.
- Audit summaries use counts and hashed email samples, not raw email addresses.

## Setup Notes

GitHub Actions requires these secrets:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `BREVO_API_KEY`

Manual apply example:

```bash
python apps/backend/ops/sync_brevo_users.py --list-id 5 --apply
```

Dry-run example:

```bash
python apps/backend/ops/sync_brevo_users.py --list-id 5 --dry-run
```
