create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;
create table if not exists public.marketing_preferences (
  email text primary key,
  user_id uuid references auth.users(id) on delete set null,
  marketing_opt_in boolean not null default false,
  consent_source text not null,
  consent_version text not null,
  consent_at timestamptz not null default now(),
  latest_feedback_rating smallint check (latest_feedback_rating between 1 and 5),
  latest_feedback_text text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create trigger trg_marketing_preferences_updated_at
before update on public.marketing_preferences
for each row
execute function public.set_updated_at();
create table if not exists public.export_feedback (
  id bigserial primary key,
  session_id text,
  format text check (format in ('step', 'stl', 'glb')),
  email text,
  user_id uuid references auth.users(id) on delete set null,
  marketing_opt_in boolean,
  rating smallint check (rating between 1 and 5),
  feedback_text text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);
create index if not exists idx_export_feedback_created_at on public.export_feedback(created_at desc);
create index if not exists idx_export_feedback_session_id on public.export_feedback(session_id);
create index if not exists idx_export_feedback_email on public.export_feedback(email);
alter table public.marketing_preferences enable row level security;
alter table public.export_feedback enable row level security;
revoke all on table public.marketing_preferences from anon, authenticated;
revoke all on table public.export_feedback from anon, authenticated;
