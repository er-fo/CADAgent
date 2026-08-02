# Auth and Free-Tier Enforcement

## Purpose

Document the current authentication and quota path in `cadagent-backend-legacy`,
and define the simplest production-safe approach for enforcing per-user free-tier
limits with Supabase.

## Current auth flow

### Fusion add-in

- The add-in uses **Supabase Auth** for sign-in.
- Session state is stored locally in `~/.cadagent/session.json`.
- The client refreshes access tokens when needed before sending requests.
- The websocket client sends an `authenticate` message containing:
  - the Supabase access token
  - the user's BYOK provider keys

Relevant files:

- `apps/addin/mac/CADAgent/supabase_auth.py`
- `apps/addin/mac/CADAgent/websocket_client.py`

### Legacy backend

- The websocket backend requires authentication before privileged message types such
  as `execute_request`.
- JWT validation uses either:
  - `SUPABASE_JWT_SECRET` for HS tokens, or
  - Supabase JWKS from `SUPABASE_URL/auth/v1/.well-known/jwks.json`
- After validation, the backend extracts the JWT `sub` claim and stores it as the
  session's `user_id`.
- The validated user token is then passed into the workflow LLM call path.

Relevant files:

- `apps/backend/backend/main.py`
- `apps/backend/backend/websocket_manager.py`
- `apps/backend/backend/agent_workflow.py`

## Current quota/enforcement state

- `apps/backend/backend/rate_limiter.py` provides per-user request throttling in process memory.
- That module also defines a daily token quota abstraction, but it is **not** the
  durable production free-tier entitlement system.
- In-memory quota state resets on process restart and cannot be treated as billing
  or subscription truth.
- The durable quota path is the **Supabase Edge Function gateway** used by
  `apps/backend/backend/supabase_client.py`.
- `apps/backend/backend/llm_client.py` already routes authenticated LLM calls through
  `functions/v1/api-generate`, which is intended to perform:
  - authentication
  - quota enforcement
  - usage tracking
  - cost attribution

## Current gap

The main weakness is not login. The main weakness is that authenticated production
traffic is only correctly limited if the Supabase gateway is the mandatory path.

The following flags can bypass quota tracking and therefore must remain disabled in
production:

- `CADAGENT_AUTH_BYPASS`
- `AUTH_BYPASS`
- `CADAGENT_DEV_MODE`
- `BYPASS_SUPABASE_GATEWAY`

## Recommended production model

Use **Supabase Auth user identity** as the single source of truth and enforce the
free tier only in Supabase-managed infrastructure.

### Entitlement model

- Track a per-user token budget, not literal iteration count.
- Keep the user-facing "iterations" number only as an approximate display derived
  from the token budget.
- Initial launch target from `free-tier-economics-2026-05.md`:
  - **150 iteration-equivalents**
  - approximately **600k total tokens per user**

### Minimal backend architecture

1. User signs in through Supabase Auth.
2. Fusion add-in sends the Supabase access token to the backend websocket.
3. Backend validates the token and forwards it to `api-generate`.
4. `api-generate` verifies the user, checks remaining entitlement, records usage,
   and rejects over-budget requests.
5. Backend surfaces quota exhaustion clearly to the UI.

### Minimal data model

- `user_entitlements`
  - `user_id`
  - `plan`
  - `token_budget`
  - `period_start`
  - `period_end`
- `usage_ledger`
  - `id`
  - `user_id`
  - `request_id`
  - `input_tokens`
  - `output_tokens`
  - `cost_cents`
  - `created_at`

## Decision

For production free-tier enforcement, **Supabase Auth + Supabase Edge Function quota
enforcement** is the correct simple model. Keep the local rate limiter as abuse
protection only, not as entitlement truth.
