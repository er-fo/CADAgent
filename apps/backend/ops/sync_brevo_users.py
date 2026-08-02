#!/usr/bin/env python3
"""Reconcile confirmed Supabase Auth users with the Brevo operational users list."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx


BREVO_API_BASE = "https://api.brevo.com/v3"
DEFAULT_BREVO_WAITLIST_LIST_ID = 2
DEFAULT_MAX_REMOVALS = 10
REQUEST_TIMEOUT_SECONDS = 30.0


class SyncError(RuntimeError):
    """Raised for configuration, API, or guardrail failures."""


@dataclass(frozen=True)
class ReconciliationPlan:
    supabase_count: int
    brevo_count: int
    add: tuple[str, ...]
    remove: tuple[str, ...]

    @property
    def add_count(self) -> int:
        return len(self.add)

    @property
    def remove_count(self) -> int:
        return len(self.remove)


def normalize_email(email: str | None) -> str | None:
    if email is None:
        return None
    normalized = email.strip().lower()
    if not normalized or "@" not in normalized:
        return None
    return normalized


def normalize_email_set(emails: Iterable[str | None]) -> set[str]:
    return {email for raw in emails if (email := normalize_email(raw))}


def build_reconciliation_plan(supabase_emails: Iterable[str], brevo_emails: Iterable[str]) -> ReconciliationPlan:
    supabase_set = normalize_email_set(supabase_emails)
    brevo_set = normalize_email_set(brevo_emails)
    return ReconciliationPlan(
        supabase_count=len(supabase_set),
        brevo_count=len(brevo_set),
        add=tuple(sorted(supabase_set - brevo_set)),
        remove=tuple(sorted(brevo_set - supabase_set)),
    )


def hash_email(email: str) -> str:
    return hashlib.sha256(email.encode("utf-8")).hexdigest()[:12]


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SyncError(f"{name} is required")
    return value


def validate_list_ids(users_list_id: int, waitlist_list_id: int) -> None:
    if users_list_id <= 0:
        raise SyncError("--list-id must be a positive Brevo list ID")
    if users_list_id == waitlist_list_id:
        raise SyncError(
            f"Refusing to sync users into waitlist/marketing list {waitlist_list_id}. "
            "Use the operational users list ID instead."
        )


def assert_removal_guardrail(plan: ReconciliationPlan, max_removals: int) -> None:
    if max_removals < 0:
        raise SyncError("--max-removals must be zero or greater")
    if plan.remove_count > max_removals:
        raise SyncError(
            f"Refusing to remove {plan.remove_count} contacts from the users list because "
            f"--max-removals is {max_removals}. Inspect the dry-run summary before raising it."
        )


class SupabaseAuthClient:
    def __init__(self, supabase_url: str, service_role_key: str) -> None:
        self.supabase_url = supabase_url.rstrip("/")
        self.headers = {
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
        }

    async def fetch_confirmed_user_emails(self) -> set[str]:
        emails: set[str] = set()
        page = 1
        per_page = 1000
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            while True:
                response = await client.get(
                    f"{self.supabase_url}/auth/v1/admin/users",
                    params={"page": page, "per_page": per_page},
                    headers=self.headers,
                )
                response.raise_for_status()
                payload = response.json()
                users = payload.get("users")
                if not isinstance(users, list):
                    raise SyncError("Unexpected Supabase Auth Admin response: missing users list")

                for user in users:
                    if not isinstance(user, dict):
                        continue
                    if user.get("email_confirmed_at") is None:
                        continue
                    if email := normalize_email(user.get("email")):
                        emails.add(email)

                if len(users) < per_page:
                    break
                page += 1

        return emails


class BrevoClient:
    def __init__(self, api_key: str) -> None:
        self.headers = {
            "api-key": api_key,
            "accept": "application/json",
            "content-type": "application/json",
        }

    async def fetch_list_emails(self, list_id: int) -> set[str]:
        emails: set[str] = set()
        limit = 500
        offset = 0
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            while True:
                response = await client.get(
                    f"{BREVO_API_BASE}/contacts/lists/{list_id}/contacts",
                    params={"limit": limit, "offset": offset, "sort": "asc"},
                    headers=self.headers,
                )
                response.raise_for_status()
                payload = response.json()
                contacts = payload.get("contacts")
                if not isinstance(contacts, list):
                    raise SyncError("Unexpected Brevo list response: missing contacts list")

                for contact in contacts:
                    if isinstance(contact, dict) and (email := normalize_email(contact.get("email"))):
                        emails.add(email)

                count = payload.get("count")
                offset += len(contacts)
                if not contacts or (isinstance(count, int) and offset >= count) or len(contacts) < limit:
                    break

        return emails

    async def add_contacts_to_list(self, list_id: int, emails: Iterable[str]) -> dict[str, Any] | None:
        contacts = [{"EMAIL": email} for email in sorted(emails)]
        if not contacts:
            return None
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{BREVO_API_BASE}/contacts/import",
                headers=self.headers,
                json={
                    "jsonBody": contacts,
                    "listIds": [list_id],
                    "updateExistingContacts": True,
                    "emptyContactsAttributes": False,
                    "disableNotification": True,
                },
            )
            response.raise_for_status()
            return response.json()

    async def remove_contacts_from_list(self, list_id: int, emails: Iterable[str]) -> dict[str, Any] | None:
        sorted_emails = sorted(emails)
        if not sorted_emails:
            return None
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{BREVO_API_BASE}/contacts/lists/{list_id}/contacts/remove",
                headers=self.headers,
                json={"emails": sorted_emails},
            )
            response.raise_for_status()
            return response.json()


def build_summary(
    *,
    list_id: int,
    waitlist_list_id: int,
    dry_run: bool,
    max_removals: int,
    plan: ReconciliationPlan,
    add_result: dict[str, Any] | None = None,
    remove_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "brevo_users_list_id": list_id,
        "protected_waitlist_list_id": waitlist_list_id,
        "max_removals": max_removals,
        "supabase_confirmed_users": plan.supabase_count,
        "brevo_users_contacts": plan.brevo_count,
        "to_add": plan.add_count,
        "to_remove": plan.remove_count,
        "add_sample_hashes": [hash_email(email) for email in plan.add[:10]],
        "remove_sample_hashes": [hash_email(email) for email in plan.remove[:10]],
        "brevo_add_result": add_result,
        "brevo_remove_result": remove_result,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    waitlist_list_id = args.waitlist_list_id
    validate_list_ids(args.list_id, waitlist_list_id)

    supabase = SupabaseAuthClient(
        supabase_url=require_env("SUPABASE_URL"),
        service_role_key=require_env("SUPABASE_SERVICE_ROLE_KEY"),
    )
    brevo = BrevoClient(api_key=require_env("BREVO_API_KEY"))

    supabase_emails = await supabase.fetch_confirmed_user_emails()
    brevo_emails = await brevo.fetch_list_emails(args.list_id)
    plan = build_reconciliation_plan(supabase_emails, brevo_emails)
    assert_removal_guardrail(plan, args.max_removals)

    add_result = None
    remove_result = None
    if not args.dry_run:
        add_result = await brevo.add_contacts_to_list(args.list_id, plan.add)
        remove_result = await brevo.remove_contacts_from_list(args.list_id, plan.remove)

    summary = build_summary(
        list_id=args.list_id,
        waitlist_list_id=waitlist_list_id,
        dry_run=args.dry_run,
        max_removals=args.max_removals,
        plan=plan,
        add_result=add_result,
        remove_result=remove_result,
    )
    if args.summary_file:
        Path(args.summary_file).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync confirmed Supabase Auth users into the Brevo users list.")
    parser.add_argument("--list-id", type=int, required=True, help="Brevo operational users list ID. Must not be the waitlist list.")
    parser.add_argument(
        "--waitlist-list-id",
        type=int,
        default=int(os.environ.get("BREVO_WAITLIST_LIST_ID", DEFAULT_BREVO_WAITLIST_LIST_ID)),
        help="Protected Brevo marketing/waitlist list ID. The sync refuses to target this list.",
    )
    parser.add_argument(
        "--max-removals",
        type=int,
        default=int(os.environ.get("BREVO_USERS_SYNC_MAX_REMOVALS", DEFAULT_MAX_REMOVALS)),
        help="Maximum contacts allowed to be removed from the users list in one run.",
    )
    parser.add_argument("--summary-file", help="Optional path for the JSON audit summary.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", dest="dry_run", help="Compute and print the plan without writing to Brevo.")
    mode.add_argument("--apply", action="store_false", dest="dry_run", help="Apply additions and list-membership removals in Brevo.")
    parser.set_defaults(dry_run=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import asyncio

    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        summary = asyncio.run(run(args))
    except httpx.HTTPStatusError as exc:
        message = exc.response.text[:500]
        print(f"Brevo/Supabase HTTP error {exc.response.status_code}: {message}", file=sys.stderr)
        return 1
    except SyncError as exc:
        print(f"Sync error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
