create table if not exists public.user_entitlements (
  user_id uuid primary key references auth.users(id) on delete cascade,
  plan text not null default 'free',
  token_budget integer not null default 600000,
  period_start timestamptz not null default now(),
  period_end timestamptz not null default (now() + interval '30 days'),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.usage_ledger (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  request_id text not null unique,
  provider text not null,
  model text not null,
  input_tokens integer not null default 0,
  output_tokens integer not null default 0,
  total_tokens integer generated always as (input_tokens + output_tokens) stored,
  cost_cents numeric(12, 6) not null default 0,
  status text not null default 'succeeded',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists usage_ledger_user_created_at_idx
  on public.usage_ledger (user_id, created_at desc);

create index if not exists usage_ledger_user_model_created_at_idx
  on public.usage_ledger (user_id, model, created_at desc);

alter table public.user_entitlements enable row level security;
alter table public.usage_ledger enable row level security;

drop policy if exists "Users can read their own entitlement" on public.user_entitlements;
create policy "Users can read their own entitlement"
  on public.user_entitlements
  for select
  to authenticated
  using (auth.uid() = user_id);

drop policy if exists "Users can read their own usage" on public.usage_ledger;
create policy "Users can read their own usage"
  on public.usage_ledger
  for select
  to authenticated
  using (auth.uid() = user_id);
