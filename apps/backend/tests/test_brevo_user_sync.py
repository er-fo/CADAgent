import pytest

from ops.sync_brevo_users import (
    ReconciliationPlan,
    SyncError,
    assert_removal_guardrail,
    build_reconciliation_plan,
    normalize_email,
    normalize_email_set,
    validate_list_ids,
)


def test_normalize_email_handles_case_space_and_invalid_values():
    assert normalize_email(" User@Example.COM ") == "user@example.com"
    assert normalize_email("") is None
    assert normalize_email("not-an-email") is None
    assert normalize_email(None) is None


def test_normalize_email_set_deduplicates_valid_addresses():
    assert normalize_email_set(["A@EXAMPLE.com", "a@example.com", "bad"]) == {"a@example.com"}


def test_build_reconciliation_plan_adds_and_removes_against_supabase_source_of_truth():
    plan = build_reconciliation_plan(
        supabase_emails=["a@example.com", "b@example.com", "C@example.com"],
        brevo_emails=["b@example.com", "old@example.com"],
    )

    assert plan.supabase_count == 3
    assert plan.brevo_count == 2
    assert plan.add == ("a@example.com", "c@example.com")
    assert plan.remove == ("old@example.com",)


def test_refuses_to_target_marketing_waitlist_list():
    with pytest.raises(SyncError, match="waitlist/marketing"):
        validate_list_ids(users_list_id=2, waitlist_list_id=2)


def test_removal_guardrail_blocks_large_unexpected_removals():
    plan = ReconciliationPlan(
        supabase_count=1,
        brevo_count=4,
        add=(),
        remove=("a@example.com", "b@example.com", "c@example.com"),
    )

    with pytest.raises(SyncError, match="Refusing to remove 3 contacts"):
        assert_removal_guardrail(plan, max_removals=2)


def test_removal_guardrail_allows_threshold_match():
    plan = ReconciliationPlan(
        supabase_count=1,
        brevo_count=3,
        add=(),
        remove=("a@example.com", "b@example.com"),
    )

    assert_removal_guardrail(plan, max_removals=2)
