create table if not exists public.marketing_email_consents (
    user_id uuid primary key references auth.users(id) on delete cascade,
    email text not null,
    consent_status boolean not null,
    consent_timestamp timestamptz not null,
    consent_source text not null,
    consent_text_version text not null,
    consent_text text not null,
    withdrawn_at timestamptz null,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now())
);

create index if not exists idx_marketing_email_consents_email
    on public.marketing_email_consents (email);

create or replace function public.set_marketing_email_consents_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = timezone('utc', now());
    return new;
end;
$$;

drop trigger if exists trg_marketing_email_consents_updated_at on public.marketing_email_consents;
create trigger trg_marketing_email_consents_updated_at
before update on public.marketing_email_consents
for each row
execute procedure public.set_marketing_email_consents_updated_at();

alter table public.marketing_email_consents enable row level security;

grant select, insert, update on table public.marketing_email_consents to authenticated;

drop policy if exists "marketing_email_consents_select_own" on public.marketing_email_consents;
create policy "marketing_email_consents_select_own"
on public.marketing_email_consents
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "marketing_email_consents_insert_own" on public.marketing_email_consents;
create policy "marketing_email_consents_insert_own"
on public.marketing_email_consents
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists "marketing_email_consents_update_own" on public.marketing_email_consents;
create policy "marketing_email_consents_update_own"
on public.marketing_email_consents
for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);
